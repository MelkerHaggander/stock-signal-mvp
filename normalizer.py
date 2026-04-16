"""
Step 3 – Normalize raw text data into structured NewsItems.
Format-agnostic: extracts articles from any text that contains
headlines, dates and URLs – regardless of exact formatting.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from models import Asset, NewsItem, NormalizedData

# ── Date patterns ────────────────────────────────────────────────────────────
# Order matters: most specific first.

_DATE_PATTERNS = [
    # Full ISO-8601 with time and optional Z/offset: "2026-03-27T10:30:00Z"
    re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)"
    ),
    # English month name: "Apr 9, 2026, 17:07 GMT+2" or "Apr 9, 2026"
    re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?"
        r"\s+\d{1,2},\s+\d{4})"
        r"(?:,?\s+(\d{2}:\d{2})(?:\s+GMT([+\-]?\d+))?)?"
    ),
    # Swedish/Danish month name: "9 april 2026", "27 mars 2026"
    re.compile(
        r"(\d{1,2}\s+(?:januari|februari|mars|april|maj|juni|juli|augusti|"
        r"september|oktober|november|december|"
        r"jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)\s+\d{4})",
        re.IGNORECASE,
    ),
    # Plain ISO date: "2026-03-27"
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    # European numeric: "27/03/2026" or "27.03.2026"
    re.compile(r"(\d{1,2}[./]\d{1,2}[./]\d{4})"),
]

_URL_RE = re.compile(r"https?://\S+")

# Swedish/Danish month name -> month number
_SV_MONTHS: dict[str, int] = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "mars": 3, "mar": 3,
    "april": 4, "apr": 4,
    "maj": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "augusti": 8, "aug": 8,
    "september": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# ── Source extraction from URL domain ────────────────────────────────────────

_DOMAIN_TO_SOURCE: dict[str, str] = {
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "ft.com": "Financial Times",
    "wsj.com": "Wall Street Journal",
    "sec.gov": "SEC Filing",
    "tradingview.com": "TradingView",
    "finance.yahoo.com": "Yahoo Finance",
    "cnbc.com": "CNBC",
    "barrons.com": "Barron's",
    "marketwatch.com": "MarketWatch",
    "seekingalpha.com": "Seeking Alpha",
    "investorplace.com": "InvestorPlace",
    "fool.com": "Motley Fool",
    "zacks.com": "Zacks",
    "tipranks.com": "TipRanks",
    "morningstar.com": "Morningstar",
    "di.se": "Dagens Industri",
    "nvidianews.nvidia.com": "Company Report",
    "novonordisk.com": "Company Report",
    "investorab.com": "Company Report",
}


def _source_from_url(url: str) -> str:
    """Extract a human-readable source name from a URL."""
    url_lower = url.lower()
    for domain, name in _DOMAIN_TO_SOURCE.items():
        if domain in url_lower:
            return name
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else "Unknown"


def _try_parse_date(text: str) -> str:
    """
    Try to parse a date string into ISO-8601 UTC format.
    Returns '' on failure. Output always ends with 'Z'.
    """
    # 1. Full ISO-8601 with time — parse directly via fromisoformat.
    m = _DATE_PATTERNS[0].search(text)
    if m:
        try:
            raw = m.group(1).replace("Z", "+00:00")
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # 2. English month name + day + year (+ optional time/GMT offset)
    m = _DATE_PATTERNS[1].search(text)
    if m:
        date_part = m.group(1)
        # Normalize full month name to 3-letter for strptime %b
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                dt = datetime.strptime(date_part, fmt)
                # Attach time if captured
                if m.lastindex and m.lastindex >= 2 and m.group(2):
                    try:
                        h, mi = m.group(2).split(":")
                        dt = dt.replace(hour=int(h), minute=int(mi))
                    except (ValueError, TypeError):
                        pass
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue

    # 3. Swedish/Danish month name
    m = _DATE_PATTERNS[2].search(text)
    if m:
        parts = m.group(1).strip().split()
        if len(parts) == 3:
            try:
                day = int(parts[0])
                month = _SV_MONTHS.get(parts[1].lower())
                year = int(parts[2])
                if month:
                    dt = datetime(year, month, day, tzinfo=timezone.utc)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                pass

    # 4. Plain ISO date
    m = _DATE_PATTERNS[3].search(text)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # 5. European numeric date
    m = _DATE_PATTERNS[4].search(text)
    if m:
        for fmt in ("%d/%m/%Y", "%d.%m.%Y"):
            try:
                dt = datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue

    return ""


def _extract_profile(text: str) -> Asset:
    """Try to extract company profile info from embedded profile data."""
    sector = ""
    industry = ""
    name = ""

    m = re.search(r"Sector:\s*\n?\s*(\S[^\n]*)", text)
    if m:
        sector = m.group(1).strip()
    m = re.search(r"Industry:\s*(\S[^\n]*)", text)
    if m:
        industry = m.group(1).strip()

    desc_match = re.search(
        r"\nDescription\n(.+?)(?:\n\n|\nCorporate Governance)", text, re.DOTALL
    )
    if desc_match:
        first_sentence = desc_match.group(1).strip().split(".")[0]
        name_match = re.match(
            r"^(.+?)\s+(?:operates|is|provides|designs|develops)", first_sentence
        )
        if name_match:
            name = name_match.group(1).strip()

    return Asset(symbol="", name=name, sector=sector, industry=industry)


def normalize(raw_text: str) -> NormalizedData:
    """
    Extract articles from unstructured text.
    Splits on blank-line gaps, then looks for headline + date + URL
    in each block. Robust to varying formatting.
    """
    # Split on 2+ consecutive blank lines to find article blocks
    blocks = re.split(r"\n{3,}", raw_text.strip())

    news_items: list[NewsItem] = []
    seen_headlines: set[str] = set()

    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines or len(lines[0]) < 10:
            continue

        # Find URL (scan from end)
        url = ""
        for line in reversed(lines):
            m = _URL_RE.search(line)
            if m:
                url = m.group(0)
                break

        # Find date (scan first 5 lines)
        date_str = ""
        date_line_idx = -1
        for i, line in enumerate(lines[:5]):
            date_str = _try_parse_date(line)
            if date_str:
                date_line_idx = i
                break

        # Must have at least a date to be considered an article
        if not date_str:
            continue

        # Headline is the first line
        headline = lines[0]

        # Deduplicate
        h_key = headline.lower().strip()
        if h_key in seen_headlines:
            continue
        seen_headlines.add(h_key)

        # Body: everything between headline/date and URL
        body_start = max(1, date_line_idx + 1) if date_line_idx >= 0 else 1
        body_lines = []
        for line in lines[body_start:]:
            if _URL_RE.match(line):
                break
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        source = _source_from_url(url) if url else "Unknown"
        source_id = hashlib.sha256(
            f"{headline}|{date_str}".encode()
        ).hexdigest()[:12]

        news_items.append(
            NewsItem(
                source_id=source_id,
                headline=headline,
                source=source,
                url=url,
                published_at=date_str,
                body=body,
            )
        )

    # Try to extract profile info from the raw text
    asset = _extract_profile(raw_text)

    return NormalizedData(asset=asset, news=news_items)
