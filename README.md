# AWS Bedrock RAG Solution

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline using AWS Bedrock's `RetrieveAndGenerate` API, Knowledge Bases, and Guardrails.

## Architecture

```
User → RetrieveAndGenerate API → Knowledge Base (Vector Search) → InvokeModel API → Response
                                        ↕
                                   Guardrails (Safety)
```

## Features

- ✅ Fully managed RAG with AWS Bedrock Knowledge Bases
- ✅ Hybrid (semantic + keyword) search
- ✅ Guardrails for content safety
- ✅ Infrastructure as Code (CloudFormation)
- ✅ REST API wrapper (FastAPI)
- ✅ Unit & integration tests

## Prerequisites

- Python 3.11+
- AWS CLI configured (`aws configure`)
- AWS account with Bedrock model access enabled
- S3 bucket with your documents

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_ORG/bedrock-rag.git
cd bedrock-rag
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your values

# 3. Deploy infrastructure
./scripts/deploy.sh

# 4. Ingest documents
python scripts/ingest.py --bucket your-s3-bucket

# 5. Run the API
uvicorn src.api:app --reload
```

## Project Structure

```
bedrock-rag/
├── src/
│   ├── rag.py           # Core RAG logic
│   ├── api.py           # FastAPI wrapper
│   ├── guardrails.py    # Guardrail setup
│   └── knowledge_base.py # KB management
├── infra/
│   └── cloudformation.yaml
├── scripts/
│   ├── deploy.sh
│   └── ingest.py
├── tests/
│   ├── test_rag.py
│   └── test_api.py
└── .github/workflows/
    └── deploy.yml
```

## Environment Variables

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID |
| `GUARDRAIL_ID` | Bedrock Guardrail ID |
| `GUARDRAIL_VERSION` | Guardrail version (default: `1`) |
| `MODEL_ARN` | Foundation model ARN |
| `S3_BUCKET` | S3 bucket with source documents |

## License

MIT
