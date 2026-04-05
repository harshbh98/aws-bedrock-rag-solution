"""
Unit tests for BedrockRAG — uses mocked boto3 responses.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("KNOWLEDGE_BASE_ID", "test-kb-id")
os.environ.setdefault("GUARDRAIL_ID", "test-gr-id")
os.environ.setdefault("S3_BUCKET", "test-bucket")

from src.rag import BedrockRAG, RAGResponse


MOCK_RAG_RESPONSE = {
    "output": {"text": "The refund policy is 30 days."},
    "citations": [
        {
            "retrievedReferences": [
                {
                    "content": {"text": "Refunds are processed within 30 days."},
                    "location": {"s3Location": {"uri": "s3://bucket/policy.pdf"}},
                }
            ]
        }
    ],
    "guardrailAction": "NONE",
    "sessionId": "session-123",
}

MOCK_RETRIEVE_RESPONSE = {
    "retrievalResults": [
        {
            "content": {"text": "Refunds are processed within 30 days."},
            "location": {"s3Location": {"uri": "s3://bucket/policy.pdf"}},
            "score": 0.92,
        }
    ]
}


@pytest.fixture
def rag_with_mock():
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.retrieve_and_generate.return_value = MOCK_RAG_RESPONSE
        mock_client.retrieve.return_value = MOCK_RETRIEVE_RESPONSE
        yield BedrockRAG(), mock_client


class TestBedrockRAG:
    def test_query_returns_answer(self, rag_with_mock):
        rag, _ = rag_with_mock
        result = rag.query("What is the refund policy?")
        assert isinstance(result, RAGResponse)
        assert "30 days" in result.answer

    def test_query_parses_citations(self, rag_with_mock):
        rag, _ = rag_with_mock
        result = rag.query("What is the refund policy?")
        assert len(result.citations) == 1
        assert result.citations[0].location == "s3://bucket/policy.pdf"

    def test_guardrail_not_triggered(self, rag_with_mock):
        rag, _ = rag_with_mock
        result = rag.query("What is the refund policy?")
        assert result.guardrail_triggered is False

    def test_guardrail_triggered(self, rag_with_mock):
        rag, mock_client = rag_with_mock
        mock_client.retrieve_and_generate.return_value = {
            **MOCK_RAG_RESPONSE,
            "guardrailAction": "BLOCKED",
            "output": {"text": "I cannot provide that information."},
        }
        result = rag.query("How do I do something harmful?")
        assert result.guardrail_triggered is True

    def test_retrieve_returns_citations(self, rag_with_mock):
        rag, _ = rag_with_mock
        citations = rag.retrieve("refund policy")
        assert len(citations) == 1
        assert citations[0].score == 0.92

    def test_session_id_passed(self, rag_with_mock):
        rag, mock_client = rag_with_mock
        rag.query("question", session_id="my-session")
        call_kwargs = mock_client.retrieve_and_generate.call_args[1]
        assert call_kwargs.get("sessionId") == "my-session"
