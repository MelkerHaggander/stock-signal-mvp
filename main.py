"""
FastAPI application – exposes the stock signal pipeline as an HTTP endpoint.
Run with: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env – try project root first, then parent (legacy folder structure)
_project_dir = Path(__file__).resolve().parent
load_dotenv(_project_dir / ".env")
load_dotenv(_project_dir.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import PipelineRequest, PipelineResponse
from pipeline import run_pipeline, run_pipeline_full
from frontend_adapter import build_frontend_payload

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("stock-signal-mvp")

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Signal MVP",
    description="Signal-based stock briefing pipeline for retail investors.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP – tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/analyze", response_model=PipelineResponse)
async def analyze(request: PipelineRequest):
    """
    Run the full pipeline for a given stock query.

    **Request body:**
    - `query` (str): Stock name or ticker, e.g. "NVDA", "Nvidia", "Investor AB"
    - `language` (str, optional): Output language. Default: "english"

    **Response:** Structured briefing with four sections + metadata.
    """
    try:
        result = await run_pipeline(request.query, request.language)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Pipeline runtime error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in pipeline")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Frontend-facing endpoints ────────────────────────────────────────────────
# These return the JSON shape expected by index.html's renderStock() function.


@app.get("/api/stock/{ticker}")
async def get_stock(ticker: str):
    """Run pipeline for a ticker and return frontend-compatible JSON."""
    try:
        response, raw_text, scored = await run_pipeline_full(ticker)
        return build_frontend_payload(raw_text, response, scored)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /api/stock/%s", ticker)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/search")
async def search_stock(q: str = ""):
    """Search by company name or ticker, return frontend-compatible JSON."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")
    try:
        response, raw_text, scored = await run_pipeline_full(q.strip())
        return build_frontend_payload(raw_text, response, scored)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /api/search?q=%s", q)
        raise HTTPException(status_code=500, detail="Internal server error")

# ── Frontend ────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    return FileResponse("index.html")
