"""
Integration tests for the FastAPI endpoints.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("KNOWLEDGE_BASE_ID", "test-kb-id")
os.environ.setdefault("GUARDRAIL_ID", "test-gr-id")
os.environ.setdefault("S3_BUCKET", "test-bucket")

from src.rag import Citation, RAGResponse

MOCK_RESULT = RAGResponse(
    answer="The answer is 42.",
    citations=[Citation(text="Source text.", location="s3://bucket/doc.pdf", score=0.9)],
    guardrail_triggered=False,
)

MOCK_CITATIONS = [
    Citation(text="Source text.", location="s3://bucket/doc.pdf", score=0.9)
]


@pytest.fixture
def client():
    with patch("src.api._rag") as mock_rag:
        mock_rag.query.return_value = MOCK_RESULT
        mock_rag.retrieve.return_value = MOCK_CITATIONS
        # Import app AFTER patching so it picks up the mock
        from src.api import app
        yield TestClient(app), mock_rag


class TestAPI:
    def test_health(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_query_success(self, client):
        tc, _ = client
        resp = tc.post("/query", json={"query": "What is the answer?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer is 42."
        assert len(data["citations"]) == 1
        assert data["guardrail_triggered"] is False
        assert "session_id" in data

    def test_query_with_session_id(self, client):
        tc, mock_rag = client
        resp = tc.post(
            "/query", json={"query": "Hello", "session_id": "test-session"}
        )
        assert resp.status_code == 200
        mock_rag.query.assert_called_once_with("Hello", session_id="test-session")

    def test_retrieve_success(self, client):
        tc, _ = client
        resp = tc.post("/retrieve", json={"query": "refund policy"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.9

    def test_query_empty_string_rejected(self, client):
        tc, _ = client
        resp = tc.post("/query", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation error
