"""Compare CPU/CUDA embedding and local dual-index ingestion throughput."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import settings
from .datasets import load_split
from .embedding import QwenEmbedder
from .storage import HybridIndex

logger = logging.getLogger(__name__)


def sync_if_cuda(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_device(device: str, texts: list[str], batch_sizes: list[int], torch) -> dict:
    embedder = QwenEmbedder(
        settings.hf_model_name,
        settings.hf_model_revision,
        device,
        settings.huggingface_api_key,
    )
    measurements = []
    for batch_size in batch_sizes:
        row = {"requested_batch_size": batch_size, "sample_count": len(texts)}
        try:
            # Warm the loaded model; do not include one-time model loading in timings.
            embedder.batch_size = batch_size
            embedder.encode_documents(texts[: min(2, len(texts))], batch_size=batch_size)
            sync_if_cuda(torch, device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            previous_setting = embedder.batch_size
            embedder.batch_size = batch_size
            start = time.perf_counter()
            vectors = embedder.encode_documents(texts, batch_size=batch_size)
            sync_if_cuda(torch, device)
            embedding_seconds = time.perf_counter() - start
            row.update({
                "status": "ok",
                "embedding_seconds": embedding_seconds,
                "embedding_passages_per_second": len(texts) / max(embedding_seconds, 1e-12),
                "effective_batch_size_after_oom_adaptation": embedder.batch_size,
                "embedding_dimension": vectors.shape[1],
                "all_vectors_finite": bool(np.isfinite(vectors).all()),
            })
            if device == "cuda":
                row["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                row["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
                row["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                row["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3

            # Benchmark the actual existing Qdrant + SQLite write path in an isolated,
            # temporary store so benchmark documents never enter the user's corpus.
            embedder.batch_size = batch_size
            old_journal = settings.journal_path
            try:
                with tempfile.TemporaryDirectory(prefix="adrosonic-ingest-bench-") as directory:
                    root = Path(directory)
                    settings.journal_path = str(root / "journal.jsonl")
                    index = HybridIndex(
                        embedder,
                        qdrant_path=str(root / "qdrant"),
                        collection="ingestion_benchmark",
                        sparse_path=str(root / "sparse.sqlite3"),
                    )
                    index_started = time.perf_counter()
                    stage_totals = {"embedding_seconds": 0.0, "sparse_seconds": 0.0, "qdrant_seconds": 0.0}
                    qdrant_batch_size = settings.qdrant_upsert_batch_size
                    for offset in range(0, len(texts), qdrant_batch_size):
                        rows = [
                            (f"benchmark-{device}-{batch_size}-{offset + i}", text, {"source": "benchmark"})
                            for i, text in enumerate(texts[offset:offset + qdrant_batch_size])
                        ]
                        index.upsert_many(rows)
                        for key, value in index.last_upsert_timings.items():
                            stage_totals[key] += value
                    sync_if_cuda(torch, device)
                    total_seconds = time.perf_counter() - index_started
                    indexed = index.count()
                    index.close()
                    row.update({
                        "ingestion_seconds": total_seconds,
                        "ingestion_passages_per_second": indexed / max(total_seconds, 1e-12),
                        "qdrant_upsert_batch_size": qdrant_batch_size,
                        "indexed_in_temporary_qdrant_and_bm25": indexed,
                        "ingestion_stage_seconds": stage_totals,
                    })
            finally:
                settings.journal_path = old_journal
            embedder.batch_size = previous_setting
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            logger.exception("Benchmark failed for device=%s batch=%d", device, batch_size)
        measurements.append(row)
        logger.info("benchmark device=%s batch=%d result=%s", device, batch_size, row.get("status"))
    return {
        "device": device,
        "gpu_name": embedder.gpu_name,
        "precision": embedder.effective_precision,
        "embedding_model": settings.hf_model_name,
        "model_revision": settings.hf_model_revision,
        "max_sequence_length": settings.embedding_max_seq_length,
        "results": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output", default="reports/phase2/benchmarks/embedding_benchmark.json")
    args = parser.parse_args()
    if args.samples < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("--samples and --batch-sizes must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    logger.info(
        "torch=%s cuda_available=%s cuda_build=%s gpu=%s",
        torch.__version__, torch.cuda.is_available(), torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
    )
    ds = load_split("train")
    texts = []
    for record in ds:
        passages = record.get("passages", {})
        values = passages.get("passage_text", []) if isinstance(passages, dict) else []
        texts.extend(text.strip() for text in values if isinstance(text, str) and text.strip())
        if len(texts) >= args.samples:
            texts = texts[:args.samples]
            break
    if len(texts) < args.samples:
        raise RuntimeError(f"Dataset yielded only {len(texts)} usable passages; requested {args.samples}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"{settings.dataset_name}/{settings.dataset_config}:train",
        "sample_count": len(texts),
        "batch_sizes": args.batch_sizes,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_benchmark_skipped_reason": None if torch.cuda.is_available() else "This PyTorch build cannot access CUDA",
        },
        "device_results": [],
    }
    for device in devices:
        results["device_results"].append(benchmark_device(device, texts, args.batch_sizes, torch))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
