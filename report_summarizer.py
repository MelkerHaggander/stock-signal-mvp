"""
Feature 2 – Quarterly report summarizer.

LLM step that reads a raw quarterly report text and produces an interpreted
summary for retail investors. Separate from the 8-step signal pipeline by
design: this feature operates directly on the full raw report text rather
than on normalized news items.

Follows the same retry / backoff / temperature conventions as synthesizer.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import anthropic

from models import ReportMetric, ReportSummary
from prompts import REPORT_SUMMARY_SYSTEM_PROMPT, REPORT_SUMMARY_USER_TEMPLATE

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2048
_TEMPERATURE = 0.2
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0  # seconds – doubles each retry (2 s, 4 s)
# Raw quarterly reports vary wildly in size. Press releases (NVDA) are ~30k
# chars; interim quarterly reports (INVE-B) are ~90k; full annual reports
# (NOVO B) can exceed 400k. We truncate to keep prompt size predictable and
# cost bounded. 200k chars ≈ 50k input tokens – comfortable inside Claude
# Sonnet 4's 200k context window, and enough to reach the financial
# statements section in the longest file observed today.
_MAX_REPORT_CHARS = 200_000

# Allowed verdict values – mirror the verdict_rubric in the system prompt.
_VALID_VERDICTS = {"strong", "mixed", "weak"}


async def summarize_report(
    report_text: str,
    symbol: str,
    company_name: str,
    language: str,
    client: anthropic.AsyncAnthropic,
) -> ReportSummary:
    """
    Send the raw quarterly report text to Claude and return a structured
    interpreted summary ready for the frontend.
    """
    if len(report_text) > _MAX_REPORT_CHARS:
        logger.info(
            "Report for %s truncated from %d to %d chars",
            symbol, len(report_text), _MAX_REPORT_CHARS,
        )
        report_text = report_text[:_MAX_REPORT_CHARS]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    system = REPORT_SUMMARY_SYSTEM_PROMPT.format(output_language=language)
    user_msg = REPORT_SUMMARY_USER_TEMPLATE.format(
        symbol=symbol,
        company_name=company_name,
        report_text=report_text,
    )

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            raw = (
                raw.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            parsed = json.loads(raw)

            verdict = str(parsed.get("verdict", "mixed")).lower().strip()
            if verdict not in _VALID_VERDICTS:
                logger.warning(
                    "Unexpected verdict '%s' from model; coercing to 'mixed'",
                    verdict,
                )
                verdict = "mixed"

            metrics_raw = parsed.get("key_metrics") or []
            metrics: list[ReportMetric] = []
            for m in metrics_raw:
                if not isinstance(m, dict):
                    continue
                label = str(m.get("label", "")).strip()
                if not label:
                    continue
                metrics.append(ReportMetric(
                    label=label,
                    value=str(m.get("value", "")).strip(),
                    interpretation=str(m.get("interpretation", "")).strip(),
                ))

            positives = [
                str(s).strip()
                for s in (parsed.get("positives") or [])
                if str(s).strip()
            ]
            concerns = [
                str(s).strip()
                for s in (parsed.get("concerns") or [])
                if str(s).strip()
            ]

            return ReportSummary(
                ticker=symbol,
                company_name=company_name,
                generated_at=now,
                verdict=verdict,
                headline=str(parsed.get("headline", "")).strip(),
                overview=str(parsed.get("overview", "")).strip(),
                key_metrics=metrics,
                positives=positives,
                concerns=concerns,
                bottom_line=str(parsed.get("bottom_line", "")).strip(),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Report summary parse error attempt %d: %s", attempt + 1, exc,
            )
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Report summarization failed after {_MAX_RETRIES + 1} attempts"
                ) from exc
        except anthropic.APIError as exc:
            logger.warning(
                "Anthropic API error attempt %d: %s", attempt + 1, exc,
            )
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info(
                    "Backing off %.1fs before report summary retry %d",
                    delay, attempt + 2,
                )
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(
                    f"Report summary API failed after {_MAX_RETRIES + 1} attempts"
                ) from exc

    raise RuntimeError("Report summarization failed unexpectedly")
