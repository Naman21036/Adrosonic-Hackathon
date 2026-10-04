from __future__ import annotations

import time
from typing import Any

from .config import settings
from .observability.langfuse import observation
from .observability.logging import log_event
from .storage import HybridIndex


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Fuse ranked lists with one-based RRF ranks and stable passage-ID tie breaks."""
    fused: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen=set()
        for rank, hit in enumerate(ranking, start=1):
            pid=hit["id"]
            if pid in seen: continue
            seen.add(pid)
            entry=fused.setdefault(pid,{**hit,"score":0.0,"retrieval_scores":{}})
            entry["score"] += 1.0/(k+rank)
            entry["retrieval_scores"]["dense" if ranking is rankings[0] else "sparse"] = hit.get("score")
            entry["text"] = hit.get("text",entry.get("text",""))
            entry["metadata"] = hit.get("metadata",entry.get("metadata",{}))
    return sorted(fused.values(),key=lambda x:(-x["score"],x["id"]))


def search(index: HybridIndex, query: str, mode: str = "hybrid", top_k: int = 5, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if not query.strip(): raise ValueError("query must not be empty")
    if mode not in {"dense","hybrid"}: raise ValueError("mode must be dense or hybrid")
    started=time.perf_counter()
    filters=filters or {}
    dense_limit=max(top_k,settings.dense_candidate_k)
    filter_dimension="source_id" if list(filters)==["source_id"] else sorted(filters)
    log_event(
        "RETRIEVAL_FILTER_CONFIGURED",
        filter_enabled=bool(filters),
        filter_dimension=filter_dimension,
        filter_value=filters.get("source_id") if len(filters)==1 else filters or None,
        retrieval_mode=mode,
        top_k=top_k,
    )
    log_event(
        "DENSE_RETRIEVAL_STARTED",
        retrieval_mode=mode,
        dense_candidate_k=dense_limit,
        sparse_candidate_k=max(top_k,settings.sparse_candidate_k) if mode == "hybrid" else 0,
        top_k=top_k,
        qdrant_collection=index.collection,
        filter_enabled=bool(filters),
        filter_dimension=filter_dimension,
        filter_value=filters.get("source_id") if len(filters)==1 else filters or None,
    )
    dense_started=time.perf_counter()
    dense_stage_timings: dict[str, float] = {}
    with observation(
        "dense_retrieval",
        input={"query": query, "candidate_k": dense_limit, "top_k": top_k, "collection": index.collection, "filter": filters},
        metadata={"retrieval_mode": mode, "filter_enabled": bool(filters), "filter": filters},
    ) as dense_observation:
        try:
            timed_dense_search = getattr(index, "dense_search_with_timings", None)
            if callable(timed_dense_search):
                dense, dense_stage_timings = timed_dense_search(query,dense_limit,filters)
            else:
                dense=index.dense_search(query,dense_limit,filters)
        except Exception as exc:
            latency=(time.perf_counter()-dense_started)*1000
            log_event("DENSE_RETRIEVAL_FAILED",level="ERROR",retrieval_mode=mode,dense_latency_ms=latency,exception_type=type(exc).__name__)
            dense_observation.update(level="ERROR",status_message=type(exc).__name__,metadata={"latency_ms":latency})
            raise
    dense_latency=(time.perf_counter()-dense_started)*1000
    dense_top=dense[:5]
    log_event(
        "DENSE_RETRIEVAL_COMPLETED",
        retrieval_mode=mode,
        candidate_count=len(dense),
        context_ids=[hit["id"] for hit in dense_top],
        top_scores=[hit["score"] for hit in dense_top],
        dense_latency_ms=dense_latency,
    )
    dense_observation.update(
        output={"candidate_count":len(dense),"document_ids":[hit["id"] for hit in dense_top],"scores":[hit["score"] for hit in dense_top]},
        metadata={"latency_ms":dense_latency,"qdrant_collection":index.collection},
    )
    sparse=[]
    sparse_latency = 0.0
    if mode == "hybrid":
        sparse_limit=max(top_k,settings.sparse_candidate_k)
        sparse_started=time.perf_counter()
        try:
            sparse=index.sparse_search(query,sparse_limit,filters)
        except Exception as exc:
            log_event("BM25_RETRIEVAL_FAILED",level="ERROR",retrieval_mode=mode,exception_type=type(exc).__name__)
            raise
        sparse_latency=(time.perf_counter()-sparse_started)*1000
    rrf_latency = 0.0
    if mode == "dense":
        hits=[{**hit,"retrieval_scores":{"dense":hit["score"]}} for hit in dense[:top_k]]
    else:
        rrf_started=time.perf_counter()
        log_event("RRF_STARTED",retrieval_mode=mode,dense_candidates=len(dense),sparse_candidates=len(sparse),rrf_k=settings.rrf_k,top_k=top_k)
        with observation(
            "rrf",
            input={"dense_candidates":len(dense),"sparse_candidates":len(sparse),"rrf_k":settings.rrf_k,"top_k":top_k},
        ) as rrf_observation:
            try:
                hits=reciprocal_rank_fusion([dense,sparse],settings.rrf_k)[:top_k]
            except Exception as exc:
                latency=(time.perf_counter()-rrf_started)*1000
                log_event("RRF_FAILED",level="ERROR",retrieval_mode=mode,rrf_latency_ms=latency,exception_type=type(exc).__name__)
                rrf_observation.update(level="ERROR",status_message=type(exc).__name__,metadata={"latency_ms":latency})
                raise
        rrf_latency=(time.perf_counter()-rrf_started)*1000
        log_event(
            "RRF_COMPLETED",
            retrieval_mode=mode,
            context_ids=[hit["id"] for hit in hits],
            fused_scores=[hit["score"] for hit in hits],
            top_k=top_k,
            rrf_latency_ms=rrf_latency,
        )
        rrf_observation.update(
            output={"document_ids":[hit["id"] for hit in hits],"fused_scores":[hit["score"] for hit in hits],"top_k":top_k},
            metadata={"latency_ms":rrf_latency},
        )
    requested_source_id=filters.get("source_id")
    if requested_source_id is not None and any(hit.get("source_id") != requested_source_id for hit in hits):
        log_event(
            "METADATA_FILTER_VIOLATION",
            level="ERROR",
            filter_dimension="source_id",
            filter_value=requested_source_id,
            returned_ids=[hit.get("id") for hit in hits],
        )
        raise RuntimeError("A retrieval backend returned a passage outside the source_id filter")
    elapsed=(time.perf_counter()-started)*1000
    stage_latency_ms = {
        **dense_stage_timings,
        "dense_latency_ms": round(dense_latency, 3),
    }
    if mode == "hybrid":
        stage_latency_ms.update({
            "sparse_latency_ms": round(sparse_latency, 3),
            "rrf_latency_ms": round(rrf_latency, 3),
        })
    return {
        "query":query,
        "mode":mode,
        "top_k":top_k,
        "filter":filters,
        "filters":filters,
        "latency_ms":elapsed,
        "total_latency_ms":round(elapsed, 3),
        "stage_latency_ms":stage_latency_ms,
        "results":[{**hit,"retrieval_mode":mode,"score_type":"ranking_score"} for hit in hits],
    }
