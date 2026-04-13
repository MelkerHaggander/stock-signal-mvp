"""
Step 5 – Classify filtered data points into predefined signals using Claude LLM.
Each news item is sent individually to the LLM for classification.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import anthropic

from models import ClassifiedSignal, NormalizedData
from prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1024
_TEMPERATURE = 0.0
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0   # seconds – doubles each retry (2 s, 4 s)
_INTER_REQUEST_DELAY = 0.5  # seconds – pause between data points


async def classify_signals(
    data: NormalizedData,
    client: anthropic.AsyncAnthropic,
) -> list[ClassifiedSignal]:
    """
    Send each news item to Claude for signal classification.
    Returns a flat list of all detected signals.
    """
    signals: list[ClassifiedSignal] = []
    now = datetime.utcnow().isoformat() + "Z"

    # Build source metadata lookup
    source_meta: dict[str, dict[str, str]] = {}
    for item in data.news:
        source_meta[item.source_id] = {
            "source_name": item.source,
            "source_headline": item.headline,
            "source_url": item.url,
        }

    # Build data points from news items
    data_points = []
    for item in data.news:
        # Include body excerpt for richer classification
        content = item.headline
        if item.body:
            body_excerpt = item.body[:500]
            content = f"{item.headline}\n\n{body_excerpt}"

        data_points.append({
            "source_id": item.source_id,
            "published_at": item.published_at,
            "content": content,
        })

    for i, dp in enumerate(data_points):
        # Pause between requests to avoid bursts
        if i > 0:
            await asyncio.sleep(_INTER_REQUEST_DELAY)

        user_msg = CLASSIFICATION_USER_TEMPLATE.format(
            symbol=data.asset.symbol,
            company_name=data.asset.name,
            source_id=dp["source_id"],
            published_at=dp["published_at"],
            content=dp["content"],
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

                for sig in parsed.get("signals", []):
                    meta = source_meta.get(dp["source_id"], {})
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
                break  # success
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.warning(
                    "Classification parse error on attempt %d for %s: %s",
                    attempt + 1, dp["source_id"], exc,
                )
                if attempt == _MAX_RETRIES:
                    logger.error("Skipping data point %s after %d retries", dp["source_id"], _MAX_RETRIES)
            except anthropic.APIError as exc:
                logger.warning("Anthropic API error on attempt %d: %s", attempt + 1, exc)
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.info("Backing off %.1fs before retry %d", delay, attempt + 2)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Skipping data point %s after API errors", dp["source_id"])

    return signals
