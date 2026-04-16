"""
Step 2 – Fetch raw data from GitHub.
Pure data transport – no transformation or interpretation.
"""

from __future__ import annotations

import json
import logging

import httpx

_BASE_URL = "https://raw.githubusercontent.com/MelkerHaggander/Stock-Data/main"
_TIMEOUT = 15.0

logger = logging.getLogger(__name__)


async def fetch_raw_data(github_key: str) -> str:
    """
    Fetch the raw news text file for a given asset from GitHub.
    Returns the raw text content exactly as stored.
    Raises RuntimeError on network or HTTP errors.
    """
    url = f"{_BASE_URL}/{github_key}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GitHub returned {exc.response.status_code} for {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Network error fetching {url}: {exc}"
        ) from exc


async def fetch_financial_data(github_key: str) -> dict | None:
    """
    Fetch structured financial data (price, ratios, EPS history) for an asset.
    Expected file: <github_key>.json on the same GitHub repo.

    Returns the parsed JSON as a dict, or None if the file does not exist,
    cannot be reached, or is malformed. A missing file is non-fatal –
    the frontend will gracefully render placeholders.
    """
    url = f"{_BASE_URL}/{github_key}.json"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                logger.info("No financial data file at %s (404)", url)
                return None
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to fetch financial data from %s: %s", url, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in financial data at %s: %s", url, exc)
        return None
