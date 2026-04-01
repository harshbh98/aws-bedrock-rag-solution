"""
Create, update, and manage Bedrock Knowledge Bases and Data Sources.
"""
from __future__ import annotations

import logging
import time

import boto3

from .config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    def __init__(self) -> None:
        self._agent = boto3.client(
            "bedrock-agent", region_name=settings.aws_region
        )

    # ------------------------------------------------------------------
    # Knowledge Base
    # ------------------------------------------------------------------

    def create_knowledge_base(self, name: str) -> str:
        """Create a KB backed by OpenSearch Serverless. Returns KB ID."""
        resp = self._agent.create_knowledge_base(
            name=name,
            roleArn=settings.iam_role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": (
                        "arn:aws:bedrock:us-east-1::foundation-model/"
                        "amazon.titan-embed-text-v2:0"
                    )
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": settings.opensearch_collection_arn,
                    "vectorIndexName": f"{name}-index",
                    "fieldMapping": {
                        "vectorField": "embedding",
                        "textField": "text",
                        "metadataField": "metadata",
                    },
                },
            },
        )
        kb_id: str = resp["knowledgeBase"]["knowledgeBaseId"]
        logger.info("Created Knowledge Base: %s", kb_id)
        return kb_id

    def get_knowledge_base(self, kb_id: str) -> dict:
        return self._agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]

    # ------------------------------------------------------------------
    # Data Source
    # ------------------------------------------------------------------

    def create_data_source(self, kb_id: str, name: str) -> str:
        """Attach an S3 data source to the KB. Returns data source ID."""
        resp = self._agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=name,
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{settings.s3_bucket}"
                },
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {
                    "chunkingStrategy": "FIXED_SIZE",
                    "fixedSizeChunkingConfiguration": {
                        "maxTokens": 512,
                        "overlapPercentage": 20,
                    },
                }
            },
        )
        ds_id: str = resp["dataSource"]["dataSourceId"]
        logger.info("Created Data Source: %s", ds_id)
        return ds_id

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def sync(self, kb_id: str, ds_id: str, wait: bool = True) -> str:
        """Start an ingestion job and optionally wait for completion."""
        resp = self._agent.start_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id
        )
        job_id: str = resp["ingestionJob"]["ingestionJobId"]
        logger.info("Started ingestion job: %s", job_id)

        if wait:
            self._wait_for_ingestion(kb_id, ds_id, job_id)

        return job_id

    def _wait_for_ingestion(
        self, kb_id: str, ds_id: str, job_id: str, poll_secs: int = 10
    ) -> None:
        while True:
            job = self._agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"]
            status = job["status"]
            logger.info("Ingestion status: %s", status)
            if status in ("COMPLETE", "FAILED", "STOPPED"):
                if status != "COMPLETE":
                    raise RuntimeError(f"Ingestion job ended with status: {status}")
                break
            time.sleep(poll_secs)
