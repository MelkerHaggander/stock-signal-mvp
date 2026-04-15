"""
Transforms pipeline output into the JSON shape expected by the StockView frontend.

Financial ratios and EPS history are sourced from verified data (Yahoo Finance,
SEC filings, Nasdaq Nordic) and stored as a lookup table. The raw GitHub text
contains only news articles and does not carry structured financial fields.
"""

from __future__ import annotations

from models import PipelineResponse, ScoredSignal


# ── Verified financial data per symbol ────────────────────────────────────────
# Sources: Yahoo Finance key-statistics, SEC 10-K/20-F filings, Nasdaq Nordic.
# Last updated: 2026-04-15.

_FINANCIAL_DATA: dict[str, dict] = {
    "NVDA": {
        "currency": "USD",
        "price": 188.20,
        "change_val": -0.43,
        "change_pct": -0.23,
        "ts_label": "Apr 15, 2026",
        "metrics": {
            "pe": "38.4",
            "fpe": "17.0",
            "pb": "29.1",
            "eveb": "33.9",
            "margin": "60.4%",
            "roe": "101.5%",
            "divy": "0.02%",
            "cap": "$4.57T",
            "fcf": "$58.1B",
            "fcfClass": "up",
        },
        "eps_labels": [
            "Q1 FY25", "Q2 FY25", "Q3 FY25", "Q4 FY25",
            "Q1 FY26", "Q2 FY26", "Q3 FY26", "Q4 FY26",
        ],
        "eps_data": [0.60, 0.67, 0.78, 0.89, 0.76, 1.08, 1.30, 1.76],
        "eps_unit": "$",
    },
    "INVE-B.ST": {
        "currency": "SEK",
        "price": 364.10,
        "change_val": 9.80,
        "change_pct": 2.77,
        "ts_label": "Apr 15, 2026",
        "metrics": {
            "pe": "6.0",
            "fpe": "\u2013",
            "pb": "1.06",
            "eveb": "\u2013",
            "margin": "\u2013",
            "roe": "16.5%",
            "divy": "1.69%",
            "cap": "SEK 1,012B",
            "fcf": "SEK 19.0B",
            "fcfClass": "up",
        },
        "eps_labels": [
            "Q4 2024", "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025",
        ],
        "eps_data": [-10.35, -0.99, 14.78, 19.53, 18.07],
        "eps_unit": "SEK",
    },
    "NOVO-B.CO": {
        "currency": "DKK",
        "price": 240.50,
        "change_val": -2.29,
        "change_pct": -0.94,
        "ts_label": "Apr 15, 2026",
        "metrics": {
            "pe": "10.5",
            "fpe": "11.1",
            "pb": "5.50",
            "eveb": "7.5",
            "margin": "44.5%",
            "roe": "60.7%",
            "divy": "6.61%",
            "cap": "DKK 1.07T",
            "fcf": "DKK 29.0B",
            "fcfClass": "up",
        },
        "eps_labels": [
            "Q4 2024", "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025",
        ],
        "eps_data": [6.34, 6.53, 5.96, 4.50, 6.04],
        "eps_unit": "DKK",
    },
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_price(value: float, currency: str) -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{currency} {value:,.2f}"


# ── Section splitters ─────────────────────────────────────────────────────────

def _extract_title_and_body(text: str) -> tuple[str, str]:
    """Split synthesized text into (title, body).

    Uses the first sentence as the title. Never truncates mid-sentence
    so the frontend always shows a complete, readable heading.
    """
    text = text.strip()
    for sep in [". ", ".\n"]:
        idx = text.find(sep)
        if idx > 0:
            return text[:idx], text[idx + 2:].strip()
    return text, ""


def _split_monitoring(text: str) -> tuple[str, str]:
    lower = text.lower()
    for marker in ["lang sikt", "long term", "90+ dagar", "long-term", "langsikt"]:
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
) -> dict:
    """Build the JSON shape that the StockView frontend renderStock() expects."""

    sec = response.sections

    # Look up verified financial data for this symbol
    fin = _FINANCIAL_DATA.get(response.symbol, {})
    currency = fin.get("currency", "USD")

    display_ticker = response.symbol
    for sfx in [".ST", ".CO", ".OL", ".HE"]:
        display_ticker = display_ticker.replace(sfx, "")

    # Price and change from verified data
    price = fin.get("price", 0.0)
    change_val = fin.get("change_val", 0.0)
    change_pct = fin.get("change_pct", 0.0)
    arrow = "\u25b2" if change_val >= 0 else "\u25bc"
    sign = "+" if change_val >= 0 else ""
    change_str = f"{arrow} {sign}{change_val:.2f} ({sign}{change_pct:.1f}%)"
    change_class = "negative" if change_val < 0 else ""

    ts_label = fin.get("ts_label", response.generated_at[:10])

    # Metrics from verified lookup
    metrics = fin.get("metrics", {})

    # EPS trend from verified lookup
    eps_labels = fin.get("eps_labels", ["Latest"])
    eps_data = fin.get("eps_data", [0])
    eps_unit = fin.get("eps_unit", "$" if currency == "USD" else currency)

    # Section splitting
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
        })
    if not sources:
        sources.append({
            "short": "??",
            "title": "Data source",
            "subtitle": "Pipeline",
            "href": "#",
        })

    return {
        "name": response.company_name,
        "ticker": display_ticker,
        "price": _fmt_price(price, currency),
        "change": change_str,
        "changeClass": change_class,
        "ts": f"{ts_label} \u00b7 Source: Yahoo Finance, SEC filings",
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
