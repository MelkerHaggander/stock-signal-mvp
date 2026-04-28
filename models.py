"""
Pydantic data models for the stock signal MVP pipeline.
Covers all 8 steps: identification -> fetching -> normalization -> filtering ->
classification -> scoring -> synthesis -> validation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# -- Enums --

class SignalType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MNA = "mna"
    DIVIDEND_CHANGE = "dividend_change"
    MANAGEMENT_CHANGE = "management_change"
    REGULATORY_LEGAL = "regulatory_legal"
    RESTATEMENT = "restatement"
    PRODUCT_APPROVAL = "product_approval"
    MACRO_COMPANY_SPECIFIC = "macro_company_specific"
    INSIDER_TRANSACTION = "insider_transaction"
    CREDIT_RATING = "credit_rating"
    R_AND_D_INVESTMENT = "r_and_d_investment"
    PATENT = "patent"


class ValidationStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    BLOCKED = "blocked"


# -- Step 1 --

class IdentifiedAsset(BaseModel):
    symbol: str
    name: str
    github_key: str


# -- Step 3 --

class Asset(BaseModel):
    symbol: str
    name: str
    sector: str = ""
    industry: str = ""


class NewsItem(BaseModel):
    source_id: str
    headline: str
    source: str
    url: str
    published_at: str
    body: str = ""

    def content_hash(self) -> str:
        raw = f"{self.headline}|{self.source}|{self.published_at}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class NormalizedData(BaseModel):
    asset: Asset
    news: list[NewsItem]
    collected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# -- Step 5 --

class ClassifiedSignal(BaseModel):
    signal_id: str
    type: str
    subtype: str
    source_id: str
    published_at: str
    detected_at: str
    source_name: str = ""
    source_headline: str = ""
    source_url: str = ""
    # Mechanism-level explanation of how this event affects the company's
    # fundamentals. Generated in the classification step (LLM) alongside
    # the structural classification. Language follows the pipeline's
    # output_language. Empty string when unavailable.
    why_it_matters: str = ""


# -- Step 6 --

class ScoredSignal(BaseModel):
    signal_id: str
    type: str
    subtype: str
    source_id: str
    published_at: str
    score: float
    age_factor: float
    final_weight: float
    source_name: str = ""
    source_headline: str = ""
    source_url: str = ""
    why_it_matters: str = ""


# -- Step 7 --

class DriverItem(BaseModel):
    """A single value driver – heading + one short sentence. Designed for
    skim-readability: the frontend renders these as a structured list.
    """
    heading: str
    description: str


class Sections(BaseModel):
    what_matters_now: str
    drivers: list[DriverItem]
    monitoring: str
    conclusion: str


class SynthesizedOutput(BaseModel):
    symbol: str
    generated_at: str
    sections: Sections
    signal_ids_used: list[str]


# -- Step 8 --

class ValidationResult(BaseModel):
    status: ValidationStatus
    flags: list[str] = []


# -- Source reference for API response --

class SourceReference(BaseModel):
    signal_id: str
    type: str
    subtype: str
    headline: str
    source: str
    url: str
    published_at: str
    why_it_matters: str = ""


# -- Final API response --

class PipelineResponse(BaseModel):
    symbol: str
    company_name: str
    generated_at: str
    sections: Sections
    validation_status: str
    signals_used: int
    top_signal_type: str
    sources: list[SourceReference] = []


class PipelineRequest(BaseModel):
    query: str
    language: str = "english"


# -- Feature 2: Quarterly report summary --

class ReportMetric(BaseModel):
    label: str
    value: str
    # Prior-period figure with delta, e.g. "$39.3B in Q4 FY25 (+73% YoY)".
    # Empty string when the report does not contain a comparable prior figure.
    comparison: str = ""
    interpretation: str


class ReportSummary(BaseModel):
    ticker: str
    company_name: str
    generated_at: str
    # Overall verdict – one of "strong", "mixed", "weak".
    # Used by the frontend to render a colored verdict chip.
    verdict: str
    headline: str
    overview: str
    key_metrics: list[ReportMetric] = []
    positives: list[str] = []
    concerns: list[str] = []
    bottom_line: str = ""
