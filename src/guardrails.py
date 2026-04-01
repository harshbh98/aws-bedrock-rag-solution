"""
Create and manage Bedrock Guardrails for content safety.
"""
from __future__ import annotations

import logging

import boto3

from .config import settings

logger = logging.getLogger(__name__)


class GuardrailManager:
    def __init__(self) -> None:
        self._bedrock = boto3.client("bedrock", region_name=settings.aws_region)

    def create_guardrail(self, name: str) -> tuple[str, str]:
        """
        Create a guardrail with sensible defaults.
        Returns (guardrail_id, version).
        """
        resp = self._bedrock.create_guardrail(
            name=name,
            description="RAG pipeline content safety guardrail",
            # --- Blocked topics ---
            topicPolicyConfig={
                "topicsConfig": [
                    {
                        "name": "Violence",
                        "definition": "Content that promotes or depicts violence",
                        "examples": ["How do I hurt someone?"],
                        "type": "DENY",
                    },
                    {
                        "name": "IllegalActivities",
                        "definition": "Instructions for illegal activities",
                        "examples": ["How do I make drugs?"],
                        "type": "DENY",
                    },
                ]
            },
            # --- Content filters ---
            contentPolicyConfig={
                "filtersConfig": [
                    {
                        "type": "HATE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "VIOLENCE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "SEXUAL",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "INSULTS",
                        "inputStrength": "MEDIUM",
                        "outputStrength": "MEDIUM",
                    },
                ]
            },
            # --- PII redaction ---
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {"type": "EMAIL", "action": "ANONYMIZE"},
                    {"type": "PHONE", "action": "ANONYMIZE"},
                    {"type": "SSN", "action": "BLOCK"},
                    {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                ]
            },
            blockedInputMessaging=(
                "I'm sorry, your request contains content that cannot be processed."
            ),
            blockedOutputsMessaging=(
                "I'm sorry, I cannot provide that information."
            ),
        )

        guardrail_id: str = resp["guardrailId"]
        version: str = resp["version"]
        logger.info("Created Guardrail %s (version %s)", guardrail_id, version)
        return guardrail_id, version

    def get_guardrail(self, guardrail_id: str) -> dict:
        return self._bedrock.get_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=settings.guardrail_version,
        )

    def list_guardrails(self) -> list[dict]:
        return self._bedrock.list_guardrails()["guardrails"]
