from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from .config import settings
from .embedding import QwenEmbedder
from .azure_openai import AzureOpenAIError, grounded_answer
from .observability import new_request_id, request_id_context
from .observability.langfuse import flush_langfuse, observation, trace
from .observability.logging import log_event
from .retrieval import search
from .storage import HybridIndex, PassageNotFoundError, source_id_from_passage_id

index: HybridIndex | None = None
model_ready=False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index,model_ready
    try:
        index=HybridIndex(QwenEmbedder(settings.hf_model_name,settings.hf_model_revision,settings.device,settings.huggingface_api_key))
        model_ready=True
    except Exception as exc:
        log_event("APPLICATION_STARTUP_FAILED",level="ERROR",exception_type=type(exc).__name__)
    try:
        yield
    finally:
        if index: index.close()
        flush_langfuse()

app=FastAPI(title="Adrosonic Precision Retrieval API",version="1.0.0",lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1,max_length=4096)
    mode: Literal["dense","hybrid"] = "hybrid"
    top_k: int = Field(default=5,ge=1,le=20)
    filter: dict[str,Any] = Field(default_factory=dict)
    filters: dict[str,Any] | None = None
    generate_answer: bool = False

    @field_validator("filter", "filters", mode="before")
    @classmethod
    def validate_filter_shape(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("filter must be an object")
        if "source_id" in value and (type(value["source_id"]) is not int or value["source_id"] < 0):
            raise ValueError("source_id must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def validate_filter_aliases(self):
        if self.filter and self.filters and self.filter != self.filters:
            raise ValueError("provide either filter or filters, not conflicting values")
        if self.filter and set(self.filter) != {"source_id"}:
            raise ValueError("filter currently supports source_id only")
        return self

class PassageRequest(BaseModel):
    passage_id: str = Field(min_length=1,max_length=256,validation_alias=AliasChoices("passage_id", "id"))
    text: str = Field(min_length=1,max_length=20000)
    metadata: dict[str,Any] = Field(default_factory=dict)
    source_id: int | str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_ids(cls, values):
        if isinstance(values, dict) and "passage_id" in values and "id" in values and values["passage_id"] != values["id"]:
            raise ValueError("id and passage_id must identify the same passage")
        return values

    @field_validator("passage_id")
    @classmethod
    def validate_passage_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("passage_id must not be empty")
        if value != value.strip():
            raise ValueError("passage_id must not include leading or trailing whitespace")
        return value

    @field_validator("text")
    @classmethod
    def validate_passage_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: int | str | None) -> int | str | None:
        if value is None:
            return None
        if type(value) is int and value >= 0:
            return value
        if type(value) is str and 0 < len(value.strip()) <= 256:
            return value.strip()
        raise ValueError("source_id must be a non-negative integer or a non-empty string up to 256 characters")

    @model_validator(mode="after")
    def combine_source_metadata(self):
        existing = self.metadata.get("source_id")
        if self.source_id is not None and existing is not None and existing != self.source_id:
            raise ValueError("source_id must match metadata.source_id when both are provided")
        derived = source_id_from_passage_id(self.passage_id)
        if derived is None and self.source_id is None and existing is None:
            raise ValueError("source_id is required for non-canonical passage IDs")
        if derived is not None:
            supplied = self.source_id if self.source_id is not None else existing
            if supplied is not None:
                if isinstance(supplied, str) and supplied.isdecimal():
                    supplied = int(supplied)
                if supplied != derived:
                    raise ValueError("source_id must match the numeric source ID encoded in the canonical passage ID")
            self.source_id = derived
            if existing is not None:
                self.metadata["source_id"] = derived
        return self

    def storage_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.source_id is not None:
            metadata["source_id"] = self.source_id
        return metadata


class LiveMutationResponse(BaseModel):
    success: bool
    operation: Literal["upsert", "delete"]
    status: Literal["created", "updated", "deleted"]
    passage_id: str
    source_id: int | str | None = None
    created: bool | None = None
    updated: bool | None = None
    embedding_dimension: int | None = None
    embedding_count_delta: int
    indexed_passages_before: int
    indexed_passages_after: int
    dense_count_delta: int
    sparse_count_before: int
    sparse_count_after: int
    qdrant_updated: bool | None = None
    bm25_updated: bool | None = None
    deleted_from_qdrant: bool | None = None
    deleted_from_bm25: bool | None = None
    latency_ms: float
    stage_latency_ms: dict[str, float] | None = None
    pending_mutations: int

def require_index() -> HybridIndex:
    if index is None or not model_ready: raise HTTPException(503,"Retrieval service is not ready; check model and index configuration")
    return index

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid=new_request_id(request.headers.get("x-request-id"))
    request.state.request_id=rid
    started=time.perf_counter()
    with request_id_context(rid):
        log_event("API_REQUEST_STARTED",method=request.method,path=request.url.path)
        try:
            response=await call_next(request)
        except Exception as exc:
            log_event(
                "API_REQUEST_FAILED",
                level="EXCEPTION",
                method=request.method,
                path=request.url.path,
                latency_ms=(time.perf_counter()-started)*1000,
                exception_type=type(exc).__name__,
            )
            raise
        response.headers["x-request-id"]=rid
        log_event(
            "API_REQUEST_COMPLETED",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=(time.perf_counter()-started)*1000,
        )
        return response

@app.get("/health")
def health():
    ready=index is not None and model_ready
    azure_ready=bool(settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_deployment)
    return {"status":"ready" if ready else "degraded","embedding_model_ready":model_ready,"qdrant_ready":index is not None,"sparse_ready":bool(index),"azure_ready":azure_ready}

@app.get("/status")
def status():
    if index is None: return {"indexed_passages":0,"collection":settings.qdrant_collection,"dimension":None,"sparse_ready":False,"model":settings.hf_model_name,"sync_status":"unknown"}
    dense_count=index.count(); sparse_count=index.sparse.count()
    return {"indexed_passages":dense_count,"collection":index.collection,"dimension":index.embedder.dimension,"sparse_passages":sparse_count,"pending_mutations":index.pending_count(),"sparse_ready":True,"model":settings.hf_model_name,"sync_status":"operational" if dense_count==sparse_count and index.pending_count()==0 else "needs_reconciliation"}

@app.post("/search")
async def post_search(body: SearchRequest):
    try:
        metadata_filter=body.filter or body.filters or {}
        filter_dimension="source_id" if list(metadata_filter)==["source_id"] else sorted(metadata_filter)
        with trace(
            "rag_request",
            input={"query":body.query,"retrieval_mode":body.mode,"top_k":body.top_k,"filter":metadata_filter},
            metadata={"filter_enabled":bool(metadata_filter),"filter_dimension":filter_dimension,"filter_value":metadata_filter.get("source_id") if len(metadata_filter)==1 else metadata_filter or None},
        ) as rag_trace:
            log_event(
                "RAG_REQUEST_STARTED",
                retrieval_mode=body.mode,
                top_k=body.top_k,
                filter_enabled=bool(metadata_filter),
                filter_dimension=filter_dimension,
                filter_value=metadata_filter.get("source_id") if len(metadata_filter)==1 else metadata_filter or None,
            )
            result=search(require_index(),body.query,body.mode,body.top_k,metadata_filter)
            with observation(
                "context_assembly",
                input={"retrieval_mode":body.mode,"top_k":body.top_k},
            ) as context_observation:
                contexts=result["results"]
                context_ids=[str(item["id"]) for item in contexts]
                context_chars=[len(item.get("text", "")) for item in contexts]
                context_observation.update(
                    output={"selected_document_ids":context_ids,"retrieved_count":len(contexts)},
                    metadata={"context_chars":context_chars},
                )
                log_event(
                    "CONTEXT_ASSEMBLY_COMPLETED",
                    retrieval_mode=body.mode,
                    retrieved_count=len(contexts),
                    context_ids=context_ids,
                    context_chars=context_chars,
                )
            answer_text=None
            result["retrieval_latency_ms"] = result.get("latency_ms")
            result["total_latency_ms"] = result.get("latency_ms")
            if body.generate_answer:
                start=time.perf_counter()
                try:
                    answer=await grounded_answer(body.query,result["results"])
                    answer_text=answer.get("answer")
                    generation_latency=(time.perf_counter()-start)*1000
                    result["answer"]={**answer,"generation_latency_ms":generation_latency}
                    result.setdefault("stage_latency_ms", {})["llm_latency_ms"] = round(generation_latency, 3)
                    result["total_latency_ms"] = round((result.get("latency_ms") or 0) + generation_latency, 3)
                except AzureOpenAIError as exc:
                    raise HTTPException(503,str(exc)) from exc
            rag_trace.update(
                output={"retrieved_ids":context_ids,"answer":answer_text},
                metadata={"retrieval_latency_ms":result.get("latency_ms")},
            )
            log_event(
                "RAG_REQUEST_COMPLETED",
                retrieval_mode=body.mode,
                top_k=body.top_k,
                retrieved_count=len(context_ids),
                latency_ms=result.get("latency_ms"),
                trace_id=rag_trace.trace_id,
            )
            return result
    except HTTPException: raise
    except ValueError as e: raise HTTPException(400,str(e)) from e
    except Exception as e:
        log_event("RAG_REQUEST_FAILED",level="ERROR",exception_type=type(e).__name__)
        raise HTTPException(500,"Search failed; inspect server logs with the request ID") from e

@app.post("/upsert",status_code=200, response_model=LiveMutationResponse)
def upsert(body: PassageRequest):
    started = time.perf_counter()
    metadata = body.storage_metadata()
    source_id = metadata.get("source_id", source_id_from_passage_id(body.passage_id))
    log_event("LIVE_UPSERT_STARTED", operation="upsert", passage_id=body.passage_id, source_id=source_id, status="started")
    try:
        active_index = require_index()
        with trace(
            "live_index_upsert",
            input={"passage_id": body.passage_id},
            metadata={
                "operation": "upsert",
                "passage_id": body.passage_id,
                "source_id": source_id,
                "embedding_model": settings.hf_model_name,
            },
        ) as live_trace:
            result = active_index.upsert(body.passage_id, body.text, metadata)
            result["success"] = True
            live_trace.update(
                output={
                    "status": result["status"],
                    "embedding_dimension": result["embedding_dimension"],
                    "embedding_count_delta": result["embedding_count_delta"],
                    "indexed_passages_before": result["indexed_passages_before"],
                    "indexed_passages_after": result["indexed_passages_after"],
                },
                metadata={"latency_ms": result["latency_ms"], "status": "completed"},
            )
        return result
    except HTTPException: raise
    except ValueError as e:
        log_event("LIVE_UPSERT_FAILED",level="ERROR",operation="upsert",passage_id=body.passage_id,source_id=source_id,latency_ms=(time.perf_counter()-started)*1000,status="validation_failed",exception_type=type(e).__name__)
        raise HTTPException(400,str(e)) from e
    except Exception as e:
        log_event("LIVE_UPSERT_FAILED",level="ERROR",operation="upsert",passage_id=body.passage_id,source_id=source_id,latency_ms=(time.perf_counter()-started)*1000,status="failed_or_pending",exception_type=type(e).__name__)
        raise HTTPException(500,"Upsert failed; check synchronization status") from e

@app.delete("/passages/{passage_id}", response_model=LiveMutationResponse)
def delete(passage_id: str):
    started = time.perf_counter()
    log_event(
        "LIVE_DELETE_STARTED",
        operation="delete",
        passage_id=passage_id,
        source_id=source_id_from_passage_id(passage_id),
        status="started",
    )
    try:
        with trace(
            "live_index_delete",
            input={"passage_id": passage_id},
            metadata={"operation": "delete", "passage_id": passage_id},
        ) as live_trace:
            result = require_index().delete(passage_id)
            result["success"] = True
            live_trace.update(
                output={
                    "status": result["status"],
                    "deleted_from_qdrant": result["deleted_from_qdrant"],
                    "deleted_from_bm25": result["deleted_from_bm25"],
                    "embedding_count_delta": 0,
                    "indexed_passages_before": result["indexed_passages_before"],
                    "indexed_passages_after": result["indexed_passages_after"],
                },
                metadata={"latency_ms": result["latency_ms"], "status": "completed"},
            )
        return result
    except HTTPException: raise
    except PassageNotFoundError as e:
        log_event("LIVE_DELETE_NOT_FOUND",level="WARNING",operation="delete",passage_id=passage_id,latency_ms=(time.perf_counter()-started)*1000,status="not_found")
        raise HTTPException(404,"Passage ID was not found in Qdrant or BM25") from e
    except ValueError as e:
        log_event("LIVE_DELETE_FAILED",level="ERROR",operation="delete",passage_id=passage_id,latency_ms=(time.perf_counter()-started)*1000,status="validation_failed",exception_type=type(e).__name__)
        raise HTTPException(400,str(e)) from e
    except Exception as e:
        log_event("LIVE_DELETE_FAILED",level="ERROR",operation="delete",passage_id=passage_id,latency_ms=(time.perf_counter()-started)*1000,status="failed_or_pending",exception_type=type(e).__name__)
        raise HTTPException(500,"Delete failed; check synchronization status") from e
