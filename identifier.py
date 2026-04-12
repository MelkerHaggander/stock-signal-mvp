"""
Step 1 – Identify the asset from user input.
Deterministic entity matching against a fixed allow-list of three stocks.
"""

from __future__ import annotations

import re

from models import IdentifiedAsset

# ── Allow-list with aliases ──────────────────────────────────────────────────

_ALIASES: dict[str, IdentifiedAsset] = {}

_DEFINITIONS = [
    {
        "symbol": "INVE-B.ST",
        "name": "Investor AB (publ)",
        "github_key": "INVE-B",
        "aliases": [
            # Official names and tickers
            "investor", "investor ab", "investor ab publ", "investor b",
            "inve-b", "inve b", "inve-b.st", "inveb", "investorab",
            "inve-a", "inve a", "inve-a.st",
            # Swedish variations
            "investor aktie", "investor aktien", "investor bolaget",
            "investor ab b", "investor b aktie",
            # Common misspellings
            "investror", "investoe", "invester",
            # Informal / colloquial
            "wallenberg", "wallenbergs", "wallenbergbolag",
            # ISIN
            "se0015811955",
        ],
    },
    {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "github_key": "NVDA",
        "aliases": [
            # Official names and tickers
            "nvidia", "nvda", "nvidia corp", "nvidia corporation",
            "nvidia inc", "nvidia co",
            # Product-based references
            "geforce", "cuda", "tegra", "jetson", "blackwell",
            # Common misspellings
            "nvida", "nvidea", "nviida", "nvidai", "nividia", "nvdia",
            # Informal
            "jensen huang", "jensen", "nvidia gpu", "nvidia ai",
            "nvidia chip", "nvidia chips",
            # ISIN
            "us67066g1040",
        ],
    },
    {
        "symbol": "NOVO-B.CO",
        "name": "Novo Nordisk A/S",
        "github_key": "NOVO B",
        "aliases": [
            # Official names and tickers
            "novo nordisk", "novo", "novo b", "novo-b", "novo-b.co",
            "novob", "novonordisk", "novo nordisk b", "novo nordisk a/s",
            "novo nordisk as",
            # Product-based references
            "ozempic", "wegovy", "semaglutide", "saxenda", "victoza",
            "rybelsus", "tresiba", "levemir", "norditropin",
            # Common misspellings
            "novo nrodisk", "novo nordsik", "novonordsik",
            "novordisk", "novo nordik",
            # Informal / colloquial
            "glp-1", "glp1", "novo aktie", "novo aktien",
            # Danish references
            "novo b aktie",
            # ISIN
            "dk0062498333",
        ],
    },
]

for defn in _DEFINITIONS:
    asset = IdentifiedAsset(
        symbol=defn["symbol"],
        name=defn["name"],
        github_key=defn["github_key"],
    )
    for alias in defn["aliases"]:
        _ALIASES[alias.lower().strip()] = asset


def identify(user_input: str) -> IdentifiedAsset:
    """
    Match free-text user input to one of the three allowed assets.
    Raises ValueError if no match is found.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9åäöÅÄÖæøÆØ\s\-./]", "", user_input).strip().lower()

    # Exact match first
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]

    # Substring match – pick longest matching alias
    best: IdentifiedAsset | None = None
    best_len = 0
    for alias, asset in _ALIASES.items():
        if alias in cleaned and len(alias) > best_len:
            best = asset
            best_len = len(alias)

    if best is not None:
        return best

    raise ValueError(
        f"Could not identify asset from input: '{user_input}'. "
        f"Supported: Investor AB (INVE-B), NVIDIA (NVDA), Novo Nordisk (NOVO-B)."
    )