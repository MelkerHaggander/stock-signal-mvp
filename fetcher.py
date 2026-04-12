"""
Step 2 – Fetch raw data from GitHub.
Pure data transport – no transformation or interpretation.
"""

from __future__ import annotations

import httpx

_BASE_URL = "https://raw.githubusercontent.com/MelkerHaggander/Stock-Data/main"
_TIMEOUT = 15.0


async def fetch_raw_data(github_key: str) -> str:
    """
    Fetch the raw text file for a given asset from GitHub.
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
