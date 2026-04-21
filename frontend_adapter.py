"""
Transforms pipeline output into the JSON shape expected by the StockView frontend.

Financial ratios, price, and EPS history are provided by the caller as a
dict (typically fetched from <github_key>.json on GitHub). This adapter
does not contain any hardcoded per-asset data – everything is driven by
the JSON payload stored in the Stock-Data repo.

Expected financial_data shape (all fields optional):
    {
        "currency": "USD",
        "price": 188.20,
        "change_val": -0.43,
        "change_pct": -0.23,
        "ts_label": "Apr 15, 2026",
        "metrics": {
            "pe": "38.4", "fpe": "17.0", "pb": "29.1", "eveb": "33.9",
            "margin": "60.4%", "roe": "101.5%", "divy": "0.02%",
            "cap": "$4.57T", "fcf": "$58.1B", "fcfClass": "up"
        },
        "eps_labels": ["Q1 FY25", "Q2 FY25", ...],
        "eps_data": [0.60, 0.67, ...],
        "eps_unit": "$"
    }
Missing fields render as placeholders (\u2013) on the frontend.
"""

from __future__ import annotations

from models import PipelineResponse, ScoredSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_currency(symbol: str) -> str:
    if symbol.endswith(".ST"):
        return "SEK"
    if symbol.endswith(".CO"):
        return "DKK"
    return "USD"


def _fmt_price(value: float, currency: str) -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{currency} {value:,.2f}"


def _extract_title_and_body(text: str) -> tuple[str, str]:
    """
    Split synthesized text into (title, body) at the first sentence break.
    Falls back gracefully when the text has no clean break — ensures the
    frontend always has something meaningful to render in both slots.
    """
    text = text.strip()
    if not text:
        return "", ""

    # Try common sentence terminators in order of preference.
    for sep in (". ", ".\n", "! ", "!\n", "? ", "?\n", ".\t"):
        idx = text.find(sep)
        if idx > 0:
            return text[:idx].strip(), text[idx + len(sep):].strip()

    # Single-sentence text: use the whole thing as title and put it in body too
    # so the UI has content to show. This is better than leaving body empty.
    if len(text) <= 120:
        return text, ""
    # Long unbroken text: split at midpoint on a word boundary.
    mid = len(text) // 2
    space = text.rfind(" ", 0, mid)
    if space > 0:
        return text[:space].strip(), text[space + 1:].strip()
    return text, ""


def _split_monitoring(text: str) -> tuple[str, str]:
    """
    Split the monitoring section into short-term and long-term parts.
    Looks for language-specific markers; falls back to a midpoint split.
    """
    lower = text.lower()
    markers = (
        # Swedish
        "lång sikt", "lang sikt", "långsikt", "langsikt", "på lång sikt",
        "90+ dagar", "över 90 dagar",
        # English
        "long term", "long-term", "longer term", "longer-term",
        "over the long", "in the long run",
        # Danish
        "lang sigt", "på lang sigt", "langsigt",
    )
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            split = text.rfind(".", 0, idx)
            if split != -1:
                return text[:split + 1].strip(), text[split + 1:].strip()
    sentences = text.split(". ")
    mid = max(len(sentences) // 2, 1)
    return (". ".join(sentences[:mid]) + ".").strip(), ". ".join(sentences[mid:]).strip()


def _source_short(name: str) -> str:
    n = name.lower()
    if "reuters" in n:
        return "Rtr"
    if "yahoo" in n:
        return "YF"
    if "investor relation" in n or n == "ir":
        return "IR"
    if "nasdaq" in n:
        return "Nas"
    if "press" in n:
        return "PR"
    if "bloomberg" in n:
        return "BB"
    if "financial times" in n:
        return "FT"
    if "tradingview" in n:
        return "TR"
    return name[:2].upper() if name else "??"


# ── Main builder ──────────────────────────────────────────────────────────────

def build_frontend_payload(
    raw_text: str,
    response: PipelineResponse,
    scored_signals: list[ScoredSignal],
    financial_data: dict | None = None,
) -> dict:
    """Build the JSON shape that the StockView frontend renderStock() expects.

    Args:
        raw_text: Raw news text from GitHub (used for signal pipeline, not display).
        response: Structured pipeline output with synthesized sections.
        scored_signals: Scored signals (kept for interface compatibility; unused here).
        financial_data: Optional dict with price, metrics, and EPS history,
            typically parsed from <github_key>.json on GitHub. When None,
            placeholder values are shown.
    """
    sec = response.sections
    fin = financial_data or {}

    currency = fin.get("currency") or _default_currency(response.symbol)

    display_ticker = response.symbol
    for sfx in [".ST", ".CO", ".OL", ".HE"]:
        display_ticker = display_ticker.replace(sfx, "")

    # Price and change
    price = fin.get("price", 0.0)
    change_val = fin.get("change_val", 0.0)
    change_pct = fin.get("change_pct", 0.0)
    arrow = "\u25b2" if change_val >= 0 else "\u25bc"
    sign = "+" if change_val >= 0 else ""
    change_str = f"{arrow} {sign}{change_val:.2f} ({sign}{change_pct:.1f}%)"
    change_class = "negative" if change_val < 0 else ""

    # Timestamp label
    if fin.get("ts_label"):
        ts_text = f"{fin['ts_label']} \u00b7 Source: Yahoo Finance, SEC filings"
    else:
        ts_text = f"Generated {response.generated_at[:10]}"

    # Metrics and EPS
    metrics = fin.get("metrics") or {}
    eps_labels = fin.get("eps_labels") or ["\u2014"]
    eps_data = fin.get("eps_data") or [0]
    eps_unit = fin.get("eps_unit") or ("$" if currency == "USD" else currency)

    # Analysis sections
    wm_title, wm_text = _extract_title_and_body(sec.what_matters_now)
    mon_short, mon_long = _split_monitoring(sec.monitoring)
    conc_title, conc_text = _extract_title_and_body(sec.conclusion)

    # Sources from pipeline signals
    sources = []
    for src in response.sources:
        sources.append({
            "short": _source_short(src.source),
            "title": src.headline or src.type,
            "subtitle": src.source,
            "href": src.url or "#",
            "whyItMatters": src.why_it_matters,
        })
    if not sources:
        sources.append({
            "short": "??",
            "title": "Data source",
            "subtitle": "Pipeline",
            "href": "#",
            "whyItMatters": "",
        })

    return {
        "name": response.company_name,
        "ticker": display_ticker,
        "price": _fmt_price(price, currency),
        "change": change_str,
        "changeClass": change_class,
        "ts": ts_text,
        "epsLabels": eps_labels,
        "epsData": eps_data,
        "epsUnits": eps_unit,
        "metrics": {
            "pe": metrics.get("pe", "\u2013"),
            "fpe": metrics.get("fpe", "\u2013"),
            "pb": metrics.get("pb", "\u2013"),
            "eveb": metrics.get("eveb", "\u2013"),
            "margin": metrics.get("margin", "\u2013"),
            "roe": metrics.get("roe", "\u2013"),
            "divy": metrics.get("divy", "\u2013"),
            "cap": metrics.get("cap", "\u2013"),
            "fcf": metrics.get("fcf", "\u2013"),
            "fcfClass": metrics.get("fcfClass", ""),
        },
        "wm": {
            "title": wm_title,
            "text": wm_text,
        },
        "drivers": sec.drivers,
        "monShort": mon_short,
        "monLong": mon_long,
        "conclusion": {
            "title": conc_title,
            "text": conc_text,
        },
        "sources": sources,
    }
