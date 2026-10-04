"""Benchmark dense, sparse BM25, and hybrid RRF search on MS MARCO validation queries."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .datasets import load_split
from .embedding import QwenEmbedder
from .retrieval import search
from .storage import HybridIndex

MODES = ("dense", "sparse", "hybrid")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _empty_mode() -> dict:
    return {
        "count": 0,
        "failures": 0,
        "mean_ms": None,
        "min_ms": None,
        "max_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
    }


def _save_summary(output: Path, summary: dict) -> None:
    summary_path = output.parent / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", default="reports/phase2/benchmarks/benchmark.jsonl")
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("--queries must be positive")

    output = Path(args.output)
    started = datetime.now(timezone.utc).isoformat()
    config = {
        "top_k": settings.top_k,
        "rrf_k": settings.rrf_k,
        "dense_candidate_k": settings.dense_candidate_k,
        "sparse_candidate_k": settings.sparse_candidate_k,
    }
    query_source = f"{settings.dataset_name}/{settings.dataset_config}:validation"
    try:
        queries: list[str] = []
        seen: set[str] = set()
        local_eval = Path("reports/phase2/evaluation/evaluation_dataset.jsonl")
        manifest_path = local_eval.with_name("evaluation_dataset_manifest.json")
        if local_eval.is_file():
            for line in local_eval.open(encoding="utf-8"):
                row = json.loads(line)
                query = row.get("query", "")
                if query and query not in seen:
                    queries.append(query)
                    seen.add(query)
                if len(queries) >= args.queries:
                    break
            if len(queries) >= args.queries:
                source = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                dataset_source = source.get("source", {})
                query_source = f"local prepared evaluation data: {dataset_source.get('dataset', 'unknown')}/{dataset_source.get('config', 'unknown')}:{dataset_source.get('split', 'validation')}"
            else:
                queries.clear()
                seen.clear()
        if not queries:
            dataset = load_split("validation")
            for row in dataset:
                query = row.get("query", "")
                if query and query not in seen:
                    queries.append(query)
                    seen.add(query)
                if len(queries) >= args.queries:
                    break
    except Exception as exc:
        _save_summary(output, {
            "status": "blocked",
            "timestamp_utc": started,
            "query_source": query_source,
            "requested_queries": args.queries,
            "query_count": 0,
            "retrieval_config": config,
            "latency_ms": {mode: _empty_mode() for mode in MODES},
            "error": f"{type(exc).__name__}: {exc}",
            "raw_requests_file": None,
        })
        return
    if not queries:
        _save_summary(output, {
            "status": "blocked",
            "timestamp_utc": started,
            "query_source": query_source,
            "requested_queries": args.queries,
            "query_count": 0,
            "retrieval_config": config,
            "latency_ms": {mode: _empty_mode() for mode in MODES},
            "error": "Validation split contained no usable queries",
            "raw_requests_file": None,
        })
        return

    try:
        index = HybridIndex(QwenEmbedder(
            settings.hf_model_name,
            settings.hf_model_revision,
            settings.device,
            settings.huggingface_api_key,
        ))
    except Exception as exc:
        _save_summary(output, {
            "status": "blocked",
            "timestamp_utc": started,
            "query_source": query_source,
            "requested_queries": args.queries,
            "query_count": len(queries),
            "retrieval_config": config,
            "latency_ms": {mode: _empty_mode() for mode in MODES},
            "error": f"Could not initialize the configured retrieval index/model: {type(exc).__name__}: {exc}",
            "raw_requests_file": None,
        })
        return

    indexed_count = index.count()
    logs: list[dict] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for mode in MODES:
                for number, query in enumerate(queries, 1):
                    started_query = time.perf_counter()
                    error = None
                    try:
                        if mode == "sparse":
                            index.sparse_search(query, max(settings.top_k, settings.sparse_candidate_k))
                        else:
                            search(index, query, mode, settings.top_k)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "mode": mode,
                        "query_number": number,
                        "query": query,
                        "top_k": settings.top_k,
                        "latency_ms": (time.perf_counter() - started_query) * 1000,
                        "error": error,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    logs.append(record)
    finally:
        index.close()

    latency = {}
    for mode in MODES:
        records = [row for row in logs if row["mode"] == mode]
        values = [row["latency_ms"] for row in records if row["error"] is None]
        latency[mode] = {
            "count": len(records),
            "successful": len(values),
            "failures": sum(row["error"] is not None for row in records),
            "mean_ms": statistics.mean(values) if values else None,
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "p99_ms": _percentile(values, 0.99),
        }
    try:
        import psutil
        memory_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        memory_gb = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu = None
    _save_summary(output, {
        "status": "complete" if all(latency[mode]["failures"] == 0 for mode in MODES) else "partial",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_source": query_source,
        "requested_queries": args.queries,
        "query_count": len(queries),
        "retrieval_config": config,
        "latency_ms": latency,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "memory_total_gb": memory_gb,
            "gpu": gpu,
            "embedding_model": settings.hf_model_name,
            "embedding_dimension": index.embedder.dimension,
            "indexed_passages": indexed_count,
        },
        "raw_requests_file": str(output),
    })


if __name__ == "__main__":
    main()
