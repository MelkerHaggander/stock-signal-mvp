"""
Step 5 – Classify filtered data points into predefined signals using Claude LLM.
All data points are sent in a single batched request for efficiency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import anthropic

from models import ClassifiedSignal, NormalizedData
from prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 4096
_TEMPERATURE = 0.0
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0  # seconds — doubles each retry (2s, 4s)


async def classify_signals(
    data: NormalizedData,
    client: anthropic.AsyncAnthropic,
) -> list[ClassifiedSignal]:
    """
    Send all news items to Claude in a single batched request for signal classification.
    Returns a flat list of all detected signals.
    """
    if not data.news:
        return []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build source metadata lookup
    source_meta: dict[str, dict[str, str]] = {}
    for item in data.news:
        source_meta[item.source_id] = {
            "source_name": item.source,
            "source_headline": item.headline,
            "source_url": item.url,
        }

    # Build XML block for all data points
    xml_parts: list[str] = []
    for item in data.news:
        content = item.headline
        if item.body:
            body_excerpt = item.body[:500]
            content = f"{item.headline}\n\n{body_excerpt}"

        xml_parts.append(
            f"<data_point>\n"
            f"Source ID: {item.source_id}\n"
            f"Published: {item.published_at}\n"
            f"Content: {content}\n"
            f"</data_point>"
        )

    data_points_xml = "\n".join(xml_parts)

    user_msg = CLASSIFICATION_USER_TEMPLATE.format(
        symbol=data.asset.symbol,
        company_name=data.asset.name,
        data_points_xml=data_points_xml,
        detected_at=now,
    )

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)

            signals: list[ClassifiedSignal] = []
            for result in parsed.get("results", []):
                source_id = result.get("source_id", "")
                meta = source_meta.get(source_id, {})
                for sig in result.get("signals", []):
                    signals.append(ClassifiedSignal(
                        signal_id=sig["signal_id"],
                        type=sig["type"],
                        subtype=sig["subtype"],
                        source_id=sig["source_id"],
                        published_at=sig["published_at"],
                        detected_at=sig.get("detected_at", now),
                        source_name=meta.get("source_name", ""),
                        source_headline=meta.get("source_headline", ""),
                        source_url=meta.get("source_url", ""),
                    ))

            logger.info("Batch classification: %d signals from %d data points in 1 API call",
                        len(signals), len(data.news))
            return signals

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning(
                "Classification parse error on attempt %d: %s",
                attempt + 1, exc,
            )
            if attempt == _MAX_RETRIES:
                logger.error("Classification failed after %d retries", _MAX_RETRIES)
                return []
        except anthropic.APIError as exc:
            logger.warning("Anthropic API error on attempt %d: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("Backing off %.1fs before classification retry %d", delay, attempt + 2)
                await asyncio.sleep(delay)
            else:
                logger.error("Classification failed after API errors")
                return []

    return []
