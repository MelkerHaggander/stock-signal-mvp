"""
Pipeline orchestrator – runs all 8 steps sequentially.
"""

from __future__ import annotations

import asyncio
import logging
import time

import anthropic

from models import PipelineResponse, SourceReference
from identifier import identify
from fetcher import fetch_raw_data, fetch_financial_data
from normalizer import normalize
from filter import filter_noise
from classifier import classify_signals
from scorer import score_signals
from synthesizer import synthesize
from validator import validate

logger = logging.getLogger(__name__)

# ── Shared Anthropic client ─────────────────────────────────────────────────
# Module-level singleton. Reused across all pipeline runs so the httpx
# connection pool and TLS session to api.anthropic.com stay warm, and the
# Anthropic SDK is only initialised once per process.
# The app pre-warms this client at startup (see main.py lifespan) so the
# first real analysis does not pay the cold-start cost.
# max_retries=0 keeps SDK retries disabled – we handle retries explicitly in
# classifier.py and synthesizer.py to avoid multiplying request volume.
_client = anthropic.AsyncAnthropic(max_retries=0)


def _build_sources(scored_signals) -> list[SourceReference]:
    """Build deduplicated source references from scored signals."""
    seen: set[str] = set()
    sources: list[SourceReference] = []
    for s in scored_signals:
        if s.signal_id in seen:
            continue
        seen.add(s.signal_id)
        sources.append(SourceReference(
            signal_id=s.signal_id,
            type=s.type,
            subtype=s.subtype,
            headline=s.source_headline,
            source=s.source_name,
            url=s.source_url,
            published_at=s.published_at,
            why_it_matters=s.why_it_matters,
        ))
    return sources


async def run_pipeline_full(query: str, language: str = "english") -> tuple:
    """
    Run the full pipeline and return
    (PipelineResponse, raw_text, list[ScoredSignal], financial_data | None).
    Used by the frontend-facing endpoints that need intermediate data.

    `financial_data` is fetched in parallel from GitHub (<github_key>.json)
    and contains structured ratios + EPS history for the frontend.
    It is None if the file is missing or unreadable.
    """
    from models import NormalizedData, ScoredSignal  # avoid circular at module level
    t0 = time.monotonic()
    client = _client

    asset = identify(query)
    # News text (required) and financial data (optional) are fetched in parallel
    raw_text, financial_data = await asyncio.gather(
        fetch_raw_data(asset.github_key),
        fetch_financial_data(asset.github_key),
    )
    data = normalize(raw_text)
    data = filter_noise(data)
    signals = await classify_signals(data, client, language)
    scored = score_signals(signals)

    if not scored:
        resp = PipelineResponse(
            symbol=asset.symbol,
            company_name=asset.name,
            generated_at=data.collected_at,
            sections={
                "what_matters_now": "No actionable signals detected for this asset in the current data window.",
                "drivers": [
                    {"heading": "No drivers detected", "description": "Insufficient signal data to identify current drivers."}
                ],
                "monitoring": "Continue monitoring for new earnings reports, guidance changes, or material events.",
                "conclusion": "Signal landscape is quiet. No clear directional bias from current data.",
            },
            validation_status="blocked",
            signals_used=0,
            top_signal_type="none",
            sources=[],
        )
        return resp, raw_text, scored, financial_data

    output = await synthesize(scored, data, language, client)
    validation = validate(output, scored)
    sources = _build_sources(scored)

    elapsed = time.monotonic() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)

    resp = PipelineResponse(
        symbol=asset.symbol,
        company_name=asset.name,
        generated_at=output.generated_at,
        sections=output.sections,
        validation_status=validation.status.value,
        signals_used=len(scored),
        top_signal_type=scored[0].type if scored else "none",
        sources=sources,
    )
    return resp, raw_text, scored, financial_data


async def run_pipeline(query: str, language: str = "english") -> PipelineResponse:
    """
    Execute the full 8-step pipeline for a given user query.
    Returns a PipelineResponse ready for the frontend.
    """
    t0 = time.monotonic()
    client = _client

    # Step 1 – Identify
    logger.info("Step 1: Identifying asset from '%s'", query)
    asset = identify(query)
    logger.info("  -> %s (%s)", asset.symbol, asset.name)

    # Step 2 – Fetch
    logger.info("Step 2: Fetching data from GitHub for %s", asset.github_key)
    raw_text = await fetch_raw_data(asset.github_key)
    logger.info("  -> %d chars fetched", len(raw_text))

    # Step 3 – Normalize
    logger.info("Step 3: Normalizing data")
    data = normalize(raw_text)
    logger.info("  -> %d news items parsed", len(data.news))

    # Step 4 – Filter
    logger.info("Step 4: Filtering noise")
    data = filter_noise(data)
    logger.info("  -> %d news items after filtering", len(data.news))

    # Step 5 – Classify (LLM)
    logger.info("Step 5: Classifying signals via LLM")
    signals = await classify_signals(data, client, language)
    logger.info("  -> %d signals classified", len(signals))

    # Step 6 – Score
    logger.info("Step 6: Scoring and ranking signals")
    scored = score_signals(signals)
    logger.info("  -> %d signals scored (after expiry filter)", len(scored))

    if not scored:
        return PipelineResponse(
            symbol=asset.symbol,
            company_name=asset.name,
            generated_at=data.collected_at,
            sections={
                "what_matters_now": "No actionable signals detected for this asset in the current data window.",
                "drivers": [
                    {"heading": "No drivers detected", "description": "Insufficient signal data to identify current drivers."}
                ],
                "monitoring": "Continue monitoring for new earnings reports, guidance changes, or material events.",
                "conclusion": "Signal landscape is quiet. No clear directional bias from current data.",
            },
            validation_status="blocked",
            signals_used=0,
            top_signal_type="none",
            sources=[],
        )

    # Step 7 – Synthesize (LLM)
    logger.info("Step 7: Synthesizing output via LLM")
    output = await synthesize(scored, data, language, client)
    logger.info("  -> Output generated for %s", output.symbol)

    # Step 8 – Validate
    logger.info("Step 8: Validating output")
    validation = validate(output, scored)
    logger.info("  -> Status: %s, Flags: %s", validation.status, validation.flags)

    elapsed = time.monotonic() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)

    # Build sources from scored signals
    sources = _build_sources(scored)

    return PipelineResponse(
        symbol=asset.symbol,
        company_name=asset.name,
        generated_at=output.generated_at,
        sections=output.sections,
        validation_status=validation.status.value,
        signals_used=len(scored),
        top_signal_type=scored[0].type if scored else "none",
        sources=sources,
    )
