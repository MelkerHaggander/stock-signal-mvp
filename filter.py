"""
Step 4 – Filter noise from normalized data.
Rule-based only – no semantic interpretation.
Supports both legacy (source-whitelisted) and article-based data.
"""

from __future__ import annotations

from models import NormalizedData, NewsItem

# ── Whitelist of approved sources ────────────────────────────────────────────
# Includes both legacy source names and URL-derived source names.

_APPROVED_SOURCES = {
    "reuters", "bloomberg", "financial times", "wall street journal",
    "sec", "sec filing", "company report", "press release",
    "dagens industri", "di", "ft",
    # URL-derived source names from article-based data
    "tradingview", "yahoo finance", "cnbc", "barron's", "marketwatch",
    "seeking alpha", "investorplace", "motley fool", "zacks",
    "tipranks", "morningstar",
}

# ── Business-relevant keywords (at least one must appear in headline) ────────

_BUSINESS_KEYWORDS = {
    "earnings", "revenue", "profit", "loss", "guidance", "forecast",
    "acquisition", "acquires", "merger", "divest", "divestiture",
    "dividend", "buyback", "repurchase",
    "ceo", "cfo", "management", "appoint", "resign", "board",
    "fda", "approval", "regulatory", "investigation", "fine", "lawsuit",
    "restatement", "restate",
    "patent", "r&d", "research", "development", "partnership",
    "credit rating", "downgrade", "upgrade",
    "insider", "purchase", "stake",
    "tariff", "export", "restriction", "sanction",
    "results", "beat", "miss", "surprise", "outlook",
    "chip", "drug", "product", "launch", "unveil",
    "raises", "cuts", "increase", "decrease", "growth",
    "stake", "investment", "strategy", "infrastructure",
    "q1", "q2", "q3", "q4", "annual", "full-year",
    # Extended keywords for article-based data
    "valuation", "p/e", "price target", "breakout", "rally", "shares",
    "eps", "income", "cash flow", "debt", "margin", "roe",
    "gpu", "ai", "data center", "semiconductor",
    "clinical", "trial", "obesity", "glp-1", "fda",
    "buys", "sells", "executive", "offering",
    "billion", "million", "record",
}


def _is_approved_source(source: str) -> bool:
    return source.strip().lower() in _APPROVED_SOURCES


def _has_business_keyword(headline: str, body: str = "") -> bool:
    lower = (headline + " " + body).lower()
    return any(kw in lower for kw in _BUSINESS_KEYWORDS)


def _mentions_company(headline: str, body: str, symbol: str, name: str) -> bool:
    """Return True if the company is referenced in the headline or the body."""
    lower = (headline + " " + body).lower()
    base_symbol = symbol.split(".")[0].lower()
    if base_symbol and base_symbol in lower:
        return True
    # Keep name parts that are substantive (>2 chars) — filters out "AB", "AS", "Co"
    # while keeping "Nvidia", "Novo", "Investor".
    name_parts = [p.lower() for p in name.split() if len(p) > 2]
    return any(part in lower for part in name_parts)


def _has_required_fields(item: NewsItem) -> bool:
    return bool(item.headline and item.published_at)


def filter_noise(data: NormalizedData) -> NormalizedData:
    """
    Remove noise from news items. Returns a new NormalizedData with
    only high-quality, business-relevant items.
    """
    seen_hashes: set[str] = set()
    filtered_news: list[NewsItem] = []

    for item in data.news:
        # Required fields
        if not _has_required_fields(item):
            continue

        # Deduplication
        h = item.content_hash()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Source check: approved source OR has a URL (article-based data always has URLs)
        if not _is_approved_source(item.source) and not item.url:
            continue

        # Company relevance (skip check if asset has no symbol/name yet –
        # identifier will have filled those in via pipeline)
        if data.asset.symbol or data.asset.name:
            if not _mentions_company(item.headline, item.body, data.asset.symbol, data.asset.name):
                continue

        # Business keyword gate (check headline + body for article-based data)
        if not _has_business_keyword(item.headline, item.body):
            continue

        filtered_news.append(item)

    return data.model_copy(update={"news": filtered_news})
