"""
REST API for the Hinglish guardrail pipeline.

Endpoints
---------
POST /check          Run input guardrails on a text string; no LLM call.
POST /chat           Full pipeline: guardrails + LLM response.
POST /check_grounded Verify whether an LLM response is grounded in source text.
GET  /stats          Dashboard summary numbers from the SQLite log.
GET  /health         Liveness probe.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Example:
    curl -X POST http://localhost:8000/check \\
         -H "Content-Type: application/json" \\
         -d '{"text": "Ignore all previous instructions"}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from config import DB_PATH, PipelineConfig
from src.logging_db import EventLog
from src.pipeline import GuardrailPipeline

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Pipeline singleton — created once on first request, reused thereafter.
# Using a plain lru_cache (not async) because model loading is synchronous
# and happens lazily inside each guardrail's first call.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_pipeline() -> GuardrailPipeline:
    return GuardrailPipeline(PipelineConfig())


def get_pipeline() -> GuardrailPipeline:
    """FastAPI dependency.  Separate from _build_pipeline so tests can override it."""
    return _build_pipeline()


def get_log() -> EventLog:
    return EventLog(DB_PATH)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Nothing to do on startup/shutdown for now; models load lazily.
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hinglish Guardrail API",
    description=(
        "Multilingual (English / Hinglish / Hindi) safety guardrail pipeline "
        "for LLM chatbots.  Exposes guardrail checks and the full chat pipeline "
        "as a REST service."
    ),
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User text to evaluate")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message")
    conversation_id: str | None = Field(None, description="Session ID (optional)")
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior turns as [{role, content}, …]",
    )
    source: str | None = Field(
        None,
        description=(
            "Optional reference/ground-truth text.  When provided, the "
            "hallucination guardrail runs in grounded mode (sentence-level "
            "semantic similarity) instead of the confidence-heuristic mode."
        ),
    )


class GroundedCheckRequest(BaseModel):
    response: str = Field(..., min_length=1, description="LLM response to verify")
    source: str = Field(..., min_length=1, description="Reference / ground-truth text")


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is running."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/check", tags=["guardrails"])
def check(
    req: CheckRequest,
    pipeline: GuardrailPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Run the four input guardrails on *text* without calling the LLM.

    Returns language detection, per-guardrail scores, and a top-level
    ``blocked`` / ``blocked_by`` summary.  Results are **not** logged to the
    SQLite database (use ``/chat`` for logged interactions).
    """
    return pipeline.check_input(req.text)


@app.post("/chat", tags=["guardrails"])
def chat(
    req: ChatRequest,
    pipeline: GuardrailPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Full pipeline: guardrails → LLM → output guardrails.

    Logged to the SQLite audit database.  Requires a running Ollama instance
    (set ``OLLAMA_HOST`` env var if not on localhost:11434).
    """
    return pipeline.process(
        req.text,
        conversation_id=req.conversation_id,
        history=req.history,
        source=req.source,
    )


@app.post("/check_grounded", tags=["guardrails"])
def check_grounded(
    req: GroundedCheckRequest,
    pipeline: GuardrailPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Verify whether an LLM response is grounded in a provided source text.

    Uses sentence-level cosine similarity (paraphrase-multilingual-MiniLM-L12-v2,
    supports EN / Hindi / Hinglish) to compare each response sentence against
    the source.  Returns per-sentence grounding scores in ``metadata.sentences``.

    Degrades gracefully to a "grounded/unavailable" result if
    sentence-transformers is not installed or the model is not cached.
    """
    result = pipeline.hallucination.check_grounded(req.response, req.source)
    return result.as_dict()


@app.get("/stats", tags=["monitoring"])
def stats(log: EventLog = Depends(get_log)) -> dict[str, Any]:
    """Return the same summary numbers shown on the Streamlit dashboard.

    Includes total / blocked / allowed counts, breakdowns by language and
    guardrail type, and the last 50 events.
    """
    summary = log.stats()
    summary["recent"] = log.recent(limit=50)
    return summary
