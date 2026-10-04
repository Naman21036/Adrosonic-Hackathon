from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
    hf_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "HF_MODEL_NAME"),
    )
    hf_model_revision: str = "main"
    huggingface_api_key: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_account_id: str | None = None
    cloudflare_model: str = "@cf/openai/gpt-oss-20b"
    langfuse_public_key: str | None = Field(default=None, validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str | None = Field(default=None, validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_BASEURL"),
    )
    langfuse_enabled: bool = Field(default=False, validation_alias=AliasChoices("LANGFUSE_ENABLED"))
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_file: str = Field(default="logs/app.jsonl", validation_alias=AliasChoices("LOG_FILE"))
    azure_openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_API_KEY")
    )
    azure_openai_endpoint: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_deployment: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT")
    )
    azure_openai_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    azure_openai_max_tokens: int = Field(default=1024, ge=1, le=8192)
    qdrant_path: str = "storage/qdrant"
    qdrant_collection: str = "passages"
    bm25_index_path: str = "storage/sparse.sqlite3"
    journal_path: str = "storage/update_journal.jsonl"
    dataset_name: str = "microsoft/ms_marco"
    dataset_config: str = "v1.1"
    evaluation_queries: int = Field(default=20, ge=1, validation_alias=AliasChoices("EVALUATION_QUERIES"))
    data_limit: int = 100_000
    validation_expected_passages: int = Field(
        default=100_000, ge=1, validation_alias=AliasChoices("VALIDATION_EXPECTED_PASSAGES")
    )
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto", validation_alias=AliasChoices("EMBEDDING_DEVICE", "DEVICE")
    )
    embedding_precision: Literal["auto", "float16", "bfloat16", "float32"] = "float16"
    embedding_max_seq_length: int = Field(default=256, ge=8, le=512)
    qdrant_upsert_batch_size: int = Field(default=128, ge=1, le=2048)
    top_k: int = 5
    rrf_k: int = 60
    dense_candidate_k: int = 50
    sparse_candidate_k: int = 50
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 64
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    mistral_timeout_seconds: float = 15.0
    mistral_free_tier_confirmed: bool = False


settings = Settings()
