from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    knowledge_base_id: str
    guardrail_id: str
    guardrail_version: str = "1"
    model_arn: str = (
        "arn:aws:bedrock:us-east-1::foundation-model/"
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    s3_bucket: str
    iam_role_arn: str = ""
    opensearch_collection_arn: str = ""
    number_of_results: int = 5
    search_type: str = "HYBRID"  # HYBRID | SEMANTIC

    class Config:
        env_file = ".env"


settings = Settings()
