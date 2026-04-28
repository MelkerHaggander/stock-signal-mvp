"""
Step 7 – Synthesize top-ranked signals into a structured briefing via Claude LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import anthropic

from models import NormalizedData, ScoredSignal, SynthesizedOutput, Sections, DriverItem
from prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2048
_TEMPERATURE = 0.2
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0  # seconds – doubles each retry (2 s, 4 s)
_TOP_N = 10  # max signals sent to synthesis


def _parse_drivers(raw) -> list[DriverItem]:
    """Coerce the model's drivers output into a list[DriverItem].

    The current contract is a list of {heading, description} objects. We also
    tolerate a string fallback (legacy shape or model deviation) by wrapping
    it in a single item, so the pipeline never breaks on a borderline response.
    """
    if isinstance(raw, list):
        items: list[DriverItem] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            heading = str(entry.get("heading", "")).strip()
            description = str(entry.get("description", "")).strip()
            if not heading and not description:
                continue
            items.append(DriverItem(
                heading=heading or "Driver",
                description=description,
            ))
        return items
    if isinstance(raw, str) and raw.strip():
        return [DriverItem(heading="Drivers", description=raw.strip())]
    return []


async def synthesize(
    signals: list[ScoredSignal],
    data: NormalizedData,
    language: str,
    client: anthropic.AsyncAnthropic,
) -> SynthesizedOutput:
    """
    Send top signals to Claude for synthesis.
    Returns a structured briefing.
    """
    top = signals[:_TOP_N]
    now = datetime.utcnow().isoformat() + "Z"

    ranked_json = json.dumps([s.model_dump() for s in top], indent=2)

    system = SYNTHESIS_SYSTEM_PROMPT.format(output_language=language)
    user_msg = SYNTHESIS_USER_TEMPLATE.format(
        symbol=data.asset.symbol,
        company_name=data.asset.name,
        ranked_signals_json=ranked_json,
        generated_at=now,
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
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)

            sections_raw = parsed["sections"]
            sections = Sections(
                what_matters_now=str(sections_raw.get("what_matters_now", "")).strip(),
                drivers=_parse_drivers(sections_raw.get("drivers")),
                monitoring=str(sections_raw.get("monitoring", "")).strip(),
                conclusion=str(sections_raw.get("conclusion", "")).strip(),
            )

            return SynthesizedOutput(
                symbol=parsed["symbol"],
                generated_at=parsed["generated_at"],
                sections=sections,
                signal_ids_used=parsed.get("signal_ids_used", []),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Synthesis parse error attempt %d: %s", attempt + 1, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError(f"Synthesis failed after {_MAX_RETRIES + 1} attempts") from exc
        except anthropic.APIError as exc:
            logger.warning("Anthropic API error attempt %d: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("Backing off %.1fs before synthesis retry %d", delay, attempt + 2)
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(f"Synthesis API failed after {_MAX_RETRIES + 1} attempts") from exc

    raise RuntimeError("Synthesis failed unexpectedly")
