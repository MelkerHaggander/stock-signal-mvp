# Stock Signal MVP

A signal-based stock briefing pipeline for retail investors. Cuts through noise and delivers only what matters — in four sections, under 60 seconds of reading.

## Architecture

```
User input → [1. Identify] → [2. Fetch] → [3. Normalize] → [4. Filter] →
             [5. Classify (LLM)] → [6. Score] → [7. Synthesize (LLM)] → [8. Validate] → Frontend
```

**Algorithmic steps** (1, 2, 3, 4, 6, 8): Deterministic — same input always gives same output.  
**LLM steps** (5, 7): Claude Sonnet 4 with low temperature for near-deterministic behavior.

## Supported Stocks

| Ticker | Company | Exchange |
|--------|---------|----------|
| INVE-B.ST | Investor AB | Nasdaq Stockholm |
| NVDA | NVIDIA Corporation | NASDAQ |
| NOVO-B.CO | Novo Nordisk A/S | OMX Copenhagen |

## Setup

### Prerequisites
- Python 3.11+
- Anthropic API key

### Install

```bash
cd stock-signal-mvp
pip install -r requirements.txt
```

### Configure

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Run Frontend

Open `frontend/index.html` in your browser, or use VS Code Live Server.

## API

### `POST /analyze`

**Request:**
```json
{
  "query": "NVDA",
  "language": "english"
}
```

**Response:**
```json
{
  "symbol": "NVDA",
  "company_name": "NVIDIA Corporation",
  "generated_at": "2026-03-27T10:30:00Z",
  "sections": {
    "what_matters_now": "...",
    "drivers": "...",
    "monitoring": "...",
    "conclusion": "..."
  },
  "confidence_score": 0.92,
  "validation_status": "approved",
  "signals_used": 5,
  "top_signal_type": "earnings"
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Project Structure

```
stock-signal-mvp/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── pipeline.py        # Orchestrator (8 steps)
│   ├── models.py          # Pydantic schemas
│   ├── identifier.py      # Step 1: Entity matching
│   ├── fetcher.py         # Step 2: GitHub data fetch
│   ├── normalizer.py      # Step 3: Text → JSON
│   ├── filter.py          # Step 4: Noise removal
│   ├── classifier.py      # Step 5: LLM signal classification
│   ├── scorer.py          # Step 6: Scoring & ranking
│   ├── synthesizer.py     # Step 7: LLM synthesis
│   ├── validator.py       # Step 8: Output validation
│   └── prompts.py         # LLM prompt templates
├── frontend/
│   └── index.html         # Single-page UI
├── requirements.txt
└── README.md
```

## Signal Types

**High impact (score 1.0):** earnings, guidance, M&A, dividend changes, management changes, regulatory/legal, restatements, product approvals.

**Medium impact (score 0.6):** macro with company link, insider transactions, credit ratings, R&D investments, patents.

## Scoring Formula

```
final_weight = base_score × age_factor

age_factor:
  0–30 days  → 1.0
  31–90 days → 0.7
  91 days–expiry → 0.4
  past expiry → excluded
```
