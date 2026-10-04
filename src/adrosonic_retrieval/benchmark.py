"""100+ query latency benchmark with per-query high-resolution timings."""
import argparse, csv, json, statistics, time
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from .config import get_settings
from .dataset import load_eval_queries
from .embeddings import Embedder
from .store import PassageStore
from .retrieval import RetrievalService

def run_benchmark(count: int = 100, cfg=None, output_name: str = "benchmark") -> dict:
    cfg = cfg or get_settings()
    queries = load_eval_queries(cfg.eval_dataset_id, cfg.eval_dataset_config, cfg.eval_dataset_split, count, cfg.hf_token_value)
    if len(queries) < count:
        raise RuntimeError(f"Need {count} labeled queries; dataset yielded {len(queries)}")
    embedder = Embedder(cfg.embedding_model, cfg.device, dimension=cfg.embedding_dimension,
        precision=cfg.embedding_precision, query_instruction=cfg.query_instruction,
        token=cfg.hf_token_value, revision=cfg.model_revision)
    store = PassageStore(cfg.qdrant_path, cfg.collection_name)
    service = RetrievalService(embedder, store, cfg.top_k, cfg.ef_search)
    rows = []
    try:
        for i, item in enumerate(queries):
            start = time.perf_counter_ns()
            result = service.search(item["query"])
            end = time.perf_counter_ns()
            rows.append({"index": i, "query_id": item["query_id"], "latency_ms": (end-start)/1e6,
                         "embedding_plus_search_ms": result["latency_ms"],
                         "embedding_ms": result["timings_ms"]["embedding"],
                         "qdrant_search_ms": result["timings_ms"]["qdrant_search"]})
    finally:
        store.close()
    vals = [x["latency_ms"] for x in rows]
    try:
        st_version = version("sentence-transformers")
    except PackageNotFoundError:
        st_version = None
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "query_count": len(rows), "warm": True,
        "configuration": {"embedding_model": cfg.embedding_model, "baseline_embedding_model": cfg.baseline_embedding_model,
            "model_revision": embedder.model_revision, "embedding_dimension": embedder.dimension,
            "precision": embedder.precision, "batch_size": cfg.batch_size, "top_k": cfg.top_k,
            "hnsw_m": cfg.hnsw_m, "hnsw_ef_construct": cfg.hnsw_ef_construct, "ef_search": cfg.ef_search},
        "packages": {"sentence-transformers": st_version},
        "hardware": {"platform": platform.platform(), "device_requested": cfg.device, "device_actual": embedder.device},
        "end_to_end_definition": "in-process service call, includes query encoding, Qdrant search, Python result conversion; excludes HTTP and JSON serialization; model initialized before timing",
        "p50_ms": statistics.median(vals), "p95_ms": sorted(vals)[int(.95*(len(vals)-1))],
        "p99_ms": sorted(vals)[int(.99*(len(vals)-1))], "mean_ms": statistics.mean(vals),
        "min_ms": min(vals), "max_ms": max(vals), "target_p95_under_300ms": sorted(vals)[int(.95*(len(vals)-1))] < 300}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if output_name == "benchmark" else f"{output_name}_"
    with (cfg.output_dir / f"{prefix}latency_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (cfg.output_dir / f"{prefix}benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--queries", type=int, default=100)
    print(json.dumps(run_benchmark(parser.parse_args().queries), indent=2))
