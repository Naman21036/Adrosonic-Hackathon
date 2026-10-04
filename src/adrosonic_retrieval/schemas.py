"""Public retrieval API types."""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class SearchResult(BaseModel):
    passage_id: str
    text: str
    score: float
    source: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    retrieval_mode: str = "dense"


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float
    retrieval_mode: str = "dense"


class HealthResponse(BaseModel):
    status: str
    collection: str
    indexed_passages: int | None
    model_loaded: bool


class IndexStatusResponse(BaseModel):
    collection: str
    exists: bool
    indexed_passages: int
    vector_size: int | None
    distance: str | None
