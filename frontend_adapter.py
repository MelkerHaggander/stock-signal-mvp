"""
Transforms pipeline output into the JSON shape expected by the StockView frontend.
Parses market data directly from the raw GitHub text to avoid dependency on
model fields that may vary between environments.
"""

from __future__ import annotations

import re
from models import PipelineResponse, ScoredSignal


# -- Raw-text parser --

def _parse_raw_kv(raw_text: str) -> dict[str, str]:
    result = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("http"):
            key, _, val = line.partition(":")
            if key.strip().upper() == "URL":
                continue
            result[key.strip()] = val.strip()
    return result


def _num(raw: str) -> float:
    if not raw:
        return 0.0
    raw = re.sub(r"\s*(USD|SEK|DKK)\s*$", "", raw, flags=re.IGNORECASE)
    if "(" in raw:
        raw = raw.split("(")[0].strip()
    raw = raw.replace(",", "").replace("+", "").replace("%", "")
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if raw.upper().endswith(suffix):
            return float(raw[:-1].strip()) * mult
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _parse_change_pct(raw: str) -> float:
    m = re.search(r"\(([\+\-]?[\d.]+)%\)", raw)
    return float(m.group(1)) if m else 0.0


# -- Currency helpers --

_SYMBOL_CURRENCY = {"NVDA": "USD", "INVE-B.ST": "SEK", "NOVO-B.CO": "DKK"}


def _guess_currency(symbol: str, kv: dict[str, str]) -> str:
    c = kv.get("Currency", "")
    if c:
        return c
    if symbol in _SYMBOL_CURRENCY:
        return _SYMBOL_CURRENCY[symbol]
    if symbol.endswith(".ST"):
        return "SEK"
    if symbol.endswith(".CO"):
        return "DKK"
    return "USD"


def _fmt_price(value: float, currency: str) -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{currency} {value:,.2f}"


def _fmt_large(value: float, currency: str) -> str:
    prefix = "$" if currency == "USD" else f"{currency} "
    if abs(value) >= 1e12:
        return f"{prefix}{value / 1e12:.1f}T"
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.0f}B"
    if abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.0f}M"
    return f"{prefix}{value:,.0f}"


def _pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _dir_class(value: float) -> str:
    return "up" if value > 0 else ("down" if value < 0 else "")


# -- Section splitters --

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
    # No sentence break found — the whole text is one sentence
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
    if "reuters" in n: return "Rtr"
    if "yahoo" in n: return "YF"
    if "investor relation" in n or n == "ir": return "IR"
    if "nasdaq" in n: return "Nas"
    if "press" in n: return "PR"
    if "bloomberg" in n: return "BB"
    if "financial times" in n: return "FT"
    return name[:2].upper() if name else "??"


# -- Main builder --

def build_frontend_payload(
    raw_text: str,
    response: PipelineResponse,
    scored_signals: list[ScoredSignal],
) -> dict:
    """Build the JSON shape that the StockView frontend renderStock() expects."""

    kv = _parse_raw_kv(raw_text)
    currency = _guess_currency(response.symbol, kv)
    unit = "$" if currency == "USD" else currency
    sec = response.sections

    display_ticker = response.symbol
    for sfx in [".ST", ".CO", ".OL", ".HE"]:
        display_ticker = display_ticker.replace(sfx, "")

    price = _num(kv.get("Price", "0"))
    change_raw = kv.get("Change", "+0 (0%)")
    change_val = _num(change_raw)
    change_pct = _parse_change_pct(change_raw)
    arrow = "\u25b2" if change_val >= 0 else "\u25bc"
    sign = "+" if change_val >= 0 else ""
    change_str = f"{arrow} {sign}{change_val:.2f} ({sign}{change_pct:.1f}%)"
    change_class = "negative" if change_val < 0 else ""

    pe = kv.get("P/E", "0")
    fpe = kv.get("Forward P/E", "0")
    market_cap = _num(kv.get("Market Cap", "0"))

    revenue = _num(kv.get("Revenue", "0"))
    op_income = _num(kv.get("Operating Income", "0"))
    fcf = _num(kv.get("Free Cash Flow", "0"))
    op_margin = (op_income / revenue * 100) if revenue else 0

    eps_actual = float(kv.get("EPS Actual", "0") or "0")
    eps_estimate = float(kv.get("EPS Estimate", "0") or "0")
    surprise_raw = kv.get("Surprise", "0%").replace("%", "").replace("+", "")
    surprise = float(surprise_raw) if surprise_raw else 0
    eps_diff = eps_actual - eps_estimate
    eps_arrow = "\u25b2" if eps_diff >= 0 else "\u25bc"
    eps_class = "up" if eps_diff >= 0 else "down"
    earnings_date = kv.get("Earnings Date", "Latest")

    wm_title, wm_text = _extract_title_and_body(sec.what_matters_now)

    mon_short, mon_long = _split_monitoring(sec.monitoring)
    conc_title, conc_text = _extract_title_and_body(sec.conclusion)

    sources = []
    for src in response.sources:
        sources.append({
            "short": _source_short(src.source),
            "title": src.headline or src.type,
            "subtitle": src.source,
            "href": src.url or "#",
        })
    if not sources:
        sources.append({"short": "??", "title": "Data source", "subtitle": "Pipeline", "href": "#"})

    return {
        "name": response.company_name,
        "ticker": display_ticker,
        "price": _fmt_price(price, currency),
        "change": change_str,
        "changeClass": change_class,
        "ts": f"Generated {response.generated_at[:10]}",
        "epsLabels": [earnings_date],
        "epsData": [eps_actual],
        "epsUnits": unit,
        "metrics": {
            "pe": pe,
            "fpe": fpe,
            "rev": _pct(surprise),
            "revClass": _dir_class(surprise),
            "margin": f"{op_margin:.0f}%",
            "eps": f"{unit} {eps_actual:.2f} {eps_arrow}",
            "epsClass": eps_class,
            "cap": _fmt_large(market_cap, currency),
            "fcf": _fmt_large(fcf, currency),
            "fcfClass": _dir_class(fcf),
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