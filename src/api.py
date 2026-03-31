"""
FastAPI REST wrapper around BedrockRAG.
"""
from __future__ import annotations

import uuid
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import BedrockRAG, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bedrock RAG API",
    description="AWS Bedrock RetrieveAndGenerate pipeline",
    version="1.0.0",
)

_rag = BedrockRAG()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversations",
    )


class CitationOut(BaseModel):
    text: str
    location: str
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    guardrail_triggered: bool
    session_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class RetrieveResponse(BaseModel):
    results: list[CitationOut]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Full RAG: retrieve + generate with guardrails."""
    session_id = req.session_id or str(uuid.uuid4())
    try:
        result: RAGResponse = _rag.query(req.query, session_id=session_id)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(text=c.text, location=c.location, score=c.score)
            for c in result.citations
        ],
        guardrail_triggered=result.guardrail_triggered,
        session_id=session_id,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant chunks only (no generation)."""
    try:
        citations = _rag.retrieve(req.query)
    except Exception as exc:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RetrieveResponse(
        results=[
            CitationOut(text=c.text, location=c.location, score=c.score)
            for c in citations
        ]
    )
