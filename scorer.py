"""
Step 6 – Score and rank classified signals.
Deterministic: final_weight = base_score × age_factor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models import ClassifiedSignal, ScoredSignal

# ── Base scores ──────────────────────────────────────────────────────────────

_HIGH = 1.0
_MEDIUM = 0.6

_BASE_SCORES: dict[str, float] = {
    "earnings": _HIGH,
    "guidance": _HIGH,
    "mna": _HIGH,
    "dividend_change": _HIGH,
    "management_change": _HIGH,
    "regulatory_legal": _HIGH,
    "restatement": _HIGH,
    "product_approval": _HIGH,
    "macro_company_specific": _MEDIUM,
    "insider_transaction": _MEDIUM,
    "credit_rating": _MEDIUM,
    "r_and_d_investment": _MEDIUM,
    "patent": _MEDIUM,
}

# ── Expiry in days ───────────────────────────────────────────────────────────

_EXPIRY_DAYS: dict[str, int] = {
    "earnings": 90,
    "guidance": 90,
    "mna": 180,  # default; acquirer=180, target=30, divestiture=365
    "dividend_change": 365,
    "management_change": 365,
    "regulatory_legal": 730,
    "restatement": 365,
    "product_approval": 180,
    "macro_company_specific": 90,
    "insider_transaction": 180,
    "credit_rating": 365,
    "r_and_d_investment": 365,
    "patent": 365,
}

# Override expiry for specific subtypes
_SUBTYPE_EXPIRY: dict[str, int] = {
    "target": 30,
    "divestiture": 365,
}

# ── Specificity tiebreaker (lower = more specific) ──────────────────────────

_SPECIFICITY: dict[str, int] = {
    "earnings": 1,
    "guidance": 2,
    "mna": 3,
    "dividend_change": 4,
    "management_change": 5,
    "regulatory_legal": 6,
    "restatement": 7,
    "product_approval": 8,
    "macro_company_specific": 20,
    "insider_transaction": 15,
    "credit_rating": 14,
    "r_and_d_investment": 16,
    "patent": 17,
}


def _get_expiry(signal_type: str, subtype: str) -> int:
    if subtype in _SUBTYPE_EXPIRY:
        return _SUBTYPE_EXPIRY[subtype]
    return _EXPIRY_DAYS.get(signal_type, 180)


def _compute_age_factor(published_at: str, signal_type: str, subtype: str, now: datetime) -> float:
    """
    Compute age factor. `now` must be timezone-aware (UTC).
    Returns 0.0 for unparseable dates or expired signals.
    """
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        # If parsing yielded a naive datetime, assume UTC (published_at is ISO from our normalizer).
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return 0.0

    age_days = (now - pub).days
    expiry = _get_expiry(signal_type, subtype)

    if age_days < 0:
        age_days = 0
    if age_days > expiry:
        return 0.0  # expired
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.7
    return 0.4  # 91 days to expiry


def score_signals(
    signals: list[ClassifiedSignal],
    reference_date: datetime | None = None,
) -> list[ScoredSignal]:
    """
    Score and rank signals. Returns sorted list (highest weight first).
    Expired or invalid signals are excluded.
    """
    now = reference_date or datetime.now(timezone.utc)
    # Ensure `now` is tz-aware so arithmetic with pub (tz-aware) works.
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    scored: list[ScoredSignal] = []

    for sig in signals:
        base = _BASE_SCORES.get(sig.type, 0.0)
        if base == 0.0:
            continue

        age_factor = _compute_age_factor(sig.published_at, sig.type, sig.subtype, now)
        if age_factor == 0.0:
            continue  # expired

        scored.append(ScoredSignal(
            signal_id=sig.signal_id,
            type=sig.type,
            subtype=sig.subtype,
            source_id=sig.source_id,
            published_at=sig.published_at,
            score=base,
            age_factor=age_factor,
            final_weight=round(base * age_factor, 4),
            source_name=sig.source_name,
            source_headline=sig.source_headline,
            source_url=sig.source_url,
            why_it_matters=sig.why_it_matters,
        ))

    # Sort order (highest priority first):
    #   1. final_weight desc  — best signals first
    #   2. published_at desc  — newest wins on ties
    #   3. base score desc    — higher-impact signal types win next
    #   4. specificity asc    — more specific types win as final tiebreaker
    #
    # Python's sort is stable, so we do this in two passes: first the
    # tiebreakers, then the primary key last. This avoids fighting with
    # mixed ascending/descending keys on string fields in a single tuple.
    scored.sort(key=lambda s: (
        _SPECIFICITY.get(s.type, 99),      # ascending
        -s.score,                           # descending (via negation)
    ))
    scored.sort(key=lambda s: s.published_at, reverse=True)  # newest first
    scored.sort(key=lambda s: s.final_weight, reverse=True)  # primary key

    return scored
