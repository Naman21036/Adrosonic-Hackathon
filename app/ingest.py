"""Stream MS MARCO into Qdrant and SQLite with deterministic resume support."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .datasets import load_split
from .embedding import QwenEmbedder
from .storage import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
REPORT_PATH = Path("reports/phase2/indexing_run.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=settings.data_limit, help="Target total records in the indexes")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    report_path = REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    index: HybridIndex | None = None
    encoder: QwenEmbedder | None = None
    new_count = 0
    source_count = 0
    malformed_count = 0
    duplicate_source_count = 0
    resumed_count = 0
    seen: set[str] = set()
    batch: list[tuple[str, str, dict]] = []
    stage_timings = {"embedding_seconds": 0.0, "sparse_seconds": 0.0, "qdrant_seconds": 0.0}
    terminal_error = None
    final_status = "failed"

    def snapshot(status: str, error: str | None = None) -> dict:
        elapsed = time.perf_counter() - started
        try:
            qdrant_total = index.count() if index else None
            sparse_total = index.sparse.count() if index else None
        except Exception:
            qdrant_total = sparse_total = None
        try:
            import psutil
            memory = {
                "total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                "available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
                "process_rss_gb": round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2),
            }
        except Exception:
            memory = None
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
            if cuda_available:
                gpu_memory = {
                    "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
                    "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
                }
            else:
                gpu_memory = None
        except Exception:
            cuda_available = False
            gpu_name = None
            gpu_memory = None
        qpath = Path(settings.qdrant_path)
        sparse_path = Path(settings.bm25_index_path)
        local_qdrant_storage_bytes = sum(p.stat().st_size for p in qpath.rglob("*") if p.is_file()) if qpath.exists() else 0
        sparse_storage_bytes = 0
        if sparse_path.exists():
            sparse_storage_bytes = sparse_path.stat().st_size
        return {
            "status": status,
            "dataset": f"{settings.dataset_name}/{settings.dataset_config}:train",
            "target_total": args.limit,
            "passages_indexed_this_run": new_count,
            "passages_seen_in_source": source_count,
            "malformed_or_empty_passages": malformed_count,
            "duplicate_source_ids": duplicate_source_count,
            "existing_passages_skipped_on_resume": resumed_count,
            "qdrant_total": qdrant_total,
            "qdrant_backend": "local",
            "sparse_total": sparse_total,
            "dimension": encoder.dimension if encoder else None,
            "model": settings.hf_model_name,
            "revision": settings.hf_model_revision,
            "device": encoder.device if encoder else None,
            "gpu_name": encoder.gpu_name if encoder else gpu_name,
            "cuda_available": cuda_available,
            "precision": encoder.effective_precision if encoder else None,
            "configured_embedding_batch_size": settings.embedding_batch_size,
            "effective_embedding_batch_size": encoder.batch_size if encoder else None,
            "max_sequence_length": settings.embedding_max_seq_length,
            "truncation": "right truncation at max_sequence_length",
            "qdrant_upsert_batch_size": settings.qdrant_upsert_batch_size,
            "stage_timings_seconds": stage_timings,
            "hnsw": {"m": settings.hnsw_m, "ef_construct": settings.hnsw_ef_construct, "ef_search": settings.hnsw_ef_search},
            "cpu_count": os.cpu_count(),
            "memory": memory,
            "gpu_memory": gpu_memory,
            "index_storage_bytes": local_qdrant_storage_bytes + sparse_storage_bytes,
            "sparse_storage_bytes": sparse_storage_bytes,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "duration_seconds": round(elapsed, 3),
            "throughput_passages_per_second": round(new_count / max(elapsed, 1e-9), 3),
            "started_utc": started_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }

    def save(status: str, error: str | None = None) -> None:
        report_path.write_text(json.dumps(snapshot(status, error), indent=2), encoding="utf-8")

    def flush_batch() -> None:
        nonlocal new_count, batch
        if not batch:
            return
        index.upsert_many(batch)
        new_count += len(batch)
        for key, value in index.last_upsert_timings.items():
            stage_timings[key] += value
        batch = []
        total = index.count()
        elapsed = time.perf_counter() - started
        throughput = new_count / max(elapsed, 1e-9)
        eta_seconds = max(args.limit - total, 0) / max(throughput, 1e-9)
        logger.info(
            "indexed_total=%d/%d new_this_run=%d elapsed_s=%.1f passages_per_s=%.2f eta_min=%.1f embedding_batch=%d",
            total, args.limit, new_count, elapsed, throughput, eta_seconds / 60,
            encoder.batch_size,
        )
        save("running")

    try:
        logger.info("Streaming dataset split before loading the embedding model")
        dataset = load_split("train")
        encoder = QwenEmbedder(settings.hf_model_name, settings.hf_model_revision, settings.device, settings.huggingface_api_key)
        index = HybridIndex(encoder)
        existing_dense = index.count()
        existing_sparse = index.sparse.count()
        if existing_dense != existing_sparse:
            raise RuntimeError(
                f"Cannot resume with inconsistent stores (Qdrant={existing_dense}, BM25={existing_sparse}); "
                "repair/reconcile the indexes before ingesting."
            )
        if existing_dense >= args.limit:
            logger.info("Target already satisfied: qdrant=%d bm25=%d target=%d", existing_dense, existing_sparse, args.limit)
        else:
            for row_index, row in enumerate(dataset):
                passages = row.get("passages", {})
                texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
                selected = passages.get("is_selected", []) if isinstance(passages, dict) else []
                if not isinstance(texts, (list, tuple)):
                    texts = []
                if not isinstance(selected, (list, tuple)):
                    selected = []
                query_id = row.get("query_id", row_index)
                for passage_index, text in enumerate(texts):
                    source_count += 1
                    if not isinstance(text, str) or not text.strip():
                        malformed_count += 1
                        continue
                    passage_id = f"msmarco-{query_id}-{passage_index}"
                    if passage_id in seen:
                        duplicate_source_count += 1
                        continue
                    seen.add(passage_id)
                    if index.sparse.contains(passage_id):
                        resumed_count += 1
                        continue
                    batch.append((passage_id, text, {
                        "source": "ms_marco",
                        "split": "train",
                        "relevance": bool(selected[passage_index]) if passage_index < len(selected) else False,
                    }))
                    remaining = args.limit - existing_dense - new_count
                    if len(batch) >= settings.qdrant_upsert_batch_size or len(batch) >= remaining:
                        flush_batch()
                    if existing_dense + new_count >= args.limit:
                        break
                if existing_dense + new_count >= args.limit:
                    break
            flush_batch()

        total = index.count()
        if total >= args.limit and index.sparse.count() == total:
            final_status = "complete"
            save("complete")
        else:
            final_status = "incomplete"
            save("incomplete", f"Source ended before the target: qdrant={total}, target={args.limit}")
    except Exception as exc:
        terminal_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Ingestion failed")
        save("failed", terminal_error)
        raise
    finally:
        if index:
            index.close()

    final = json.loads(report_path.read_text(encoding="utf-8"))
    logger.info(
        "run_finished status=%s qdrant=%s bm25=%s newly_indexed=%d elapsed_s=%.1f throughput=%.2f passages/s",
        final["status"], final["qdrant_total"], final["sparse_total"], new_count,
        final["duration_seconds"], final["throughput_passages_per_second"],
    )
    if final_status != "complete":
        sys.exit(2)


if __name__ == "__main__":
    main()
