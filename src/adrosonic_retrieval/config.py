"""Environment-backed application configuration."""
from functools import lru_cache
from pathlib import Path
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_name: str = "Adrosonic Dense Retrieval"
    hf_dataset_id: str = "sentence-transformers/msmarco-corpus"
    hf_dataset_config: str = "passage"
    hf_dataset_split: str = "train"
    eval_dataset_id: str = "microsoft/ms_marco"
    eval_dataset_config: str = "v1.1"
    eval_dataset_split: str = "validation"
    # Qwen is the optimization candidate. The prior MiniLM configuration stays
    # available as BASELINE_EMBEDDING_MODEL for controlled comparisons.
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    baseline_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(0, ge=0, le=1024)  # 0 means model-native maximum.
    embedding_precision: str = "fp32"
    model_revision: str | None = None
    query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    hf_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices(
        "HUGGINGFACE_API_KEY", "HF_TOKEN", "HF_API_KEY", "HUGGINGFACEHUB_API_TOKEN"))
    device: str = "cpu"
    batch_size: int = Field(32, ge=1, le=4096)
    data_limit: int = Field(100_000, ge=1)
    collection_name: str = "msmarco_qwen3_embedding_0_6b"
    baseline_collection_name: str = "msmarco_passages"
    qdrant_path: Path = Path("./storage/qdrant")
    top_k: int = Field(5, ge=1, le=100)
    ef_search: int = Field(64, ge=1)
    hnsw_m: int = Field(16, ge=4, le=128)
    hnsw_ef_construct: int = Field(100, ge=8)
    log_level: str = "INFO"
    output_dir: Path = Path("./artifacts")

    @property
    def hf_token_value(self) -> str | None:
        """Return the configured token only for internal authenticated Hub calls."""
        if self.hf_api_key is None:
            return None
        token = self.hf_api_key.get_secret_value().strip()
        return token or None

    @field_validator("embedding_precision")
    @classmethod
    def supported_precision(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"fp32", "fp16", "bf16"}:
            raise ValueError("EMBEDDING_PRECISION must be fp32, fp16, or bf16")
        return value

    @field_validator("model_revision", mode="before")
    @classmethod
    def empty_revision_is_unpinned(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    @field_validator("collection_name")
    @classmethod
    def valid_collection(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("COLLECTION_NAME cannot be empty")
        return value.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
