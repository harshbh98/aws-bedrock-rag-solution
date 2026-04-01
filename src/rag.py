"""
Core RAG logic — wraps Bedrock RetrieveAndGenerate and plain Retrieve APIs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    text: str
    location: str
    score: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    guardrail_triggered: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class BedrockRAG:
    """High-level interface for AWS Bedrock RetrieveAndGenerate."""

    def __init__(self) -> None:
        self._runtime = boto3.client(
            "bedrock-agent-runtime", region_name=settings.aws_region
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str, session_id: str | None = None) -> RAGResponse:
        """Send a query through the full RAG pipeline."""
        params = self._build_params(user_query, session_id)
        try:
            raw = self._runtime.retrieve_and_generate(**params)
        except ClientError as exc:
            logger.error("Bedrock API error: %s", exc)
            raise

        return self._parse_response(raw)

    def retrieve(self, user_query: str) -> list[Citation]:
        """Retrieve relevant chunks without generation (useful for debugging)."""
        raw = self._runtime.retrieve(
            retrievalQuery={"text": user_query},
            knowledgeBaseId=settings.knowledge_base_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": settings.number_of_results,
                    "overrideSearchType": settings.search_type,
                }
            },
        )
        return [
            Citation(
                text=r["content"]["text"],
                location=r.get("location", {}).get("s3Location", {}).get("uri", ""),
                score=r.get("score", 0.0),
            )
            for r in raw.get("retrievalResults", [])
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_params(
        self, user_query: str, session_id: str | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "input": {"text": user_query},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": settings.knowledge_base_id,
                    "modelArn": settings.model_arn,
                    "generationConfiguration": {
                        "guardrailConfiguration": {
                            "guardrailId": settings.guardrail_id,
                            "guardrailVersion": settings.guardrail_version,
                        }
                    },
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": settings.number_of_results,
                            "overrideSearchType": settings.search_type,
                        }
                    },
                },
            },
        }
        if session_id:
            params["sessionId"] = session_id
        return params

    @staticmethod
    def _parse_response(raw: dict[str, Any]) -> RAGResponse:
        answer = raw.get("output", {}).get("text", "")
        guardrail_triggered = (
            raw.get("guardrailAction", "NONE") != "NONE"
        )

        citations: list[Citation] = []
        for citation_block in raw.get("citations", []):
            for ref in citation_block.get("retrievedReferences", []):
                citations.append(
                    Citation(
                        text=ref["content"]["text"],
                        location=(
                            ref.get("location", {})
                            .get("s3Location", {})
                            .get("uri", "")
                        ),
                    )
                )

        return RAGResponse(
            answer=answer,
            citations=citations,
            guardrail_triggered=guardrail_triggered,
            raw=raw,
        )
