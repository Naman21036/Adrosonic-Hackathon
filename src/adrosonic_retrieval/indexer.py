"""Bounded-memory, resumable passage ingestion."""
from itertools import islice
import json, logging, platform, socket, time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from .config import Settings
from .dataset import load_passages
from .embeddings import Embedder
from .store import PassageStore

log = logging.getLogger(__name__)

def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def batched(rows, size):
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        # Bound duplicate tracking to one batch; Qdrant upsert makes restart/dataset
        # duplicates safe because the stable passage ID is the point ID.
        unique = {row["passage_id"]: row for row in batch}
        yield list(unique.values())

def run_index(settings: Settings) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    embedder = Embedder(settings.embedding_model, settings.device, dimension=settings.embedding_dimension,
        precision=settings.embedding_precision, query_instruction=settings.query_instruction,
        token=settings.hf_token_value, revision=settings.model_revision)
    store = PassageStore(settings.qdrant_path, settings.collection_name, embedder.dimension,
                         settings.hnsw_m, settings.hnsw_ef_construct)
    seen_before = store.count()
    indexed = 0
    errors = 0
    dataset_stream_seconds = 0.0
    embedding_seconds = 0.0
    upsert_seconds = 0.0
    previous_batch_end = time.perf_counter()
    source = load_passages(settings.hf_dataset_id, settings.hf_dataset_config,
                           settings.hf_dataset_split, settings.data_limit, settings.hf_token_value)
    try:
        for batch_number, batch in enumerate(batched(source, settings.batch_size), start=1):
            batch_ready = time.perf_counter()
            dataset_stream_seconds += batch_ready - previous_batch_end
            log.info("Processing batch %d (%d passages)", batch_number, len(batch))
            try:
                embedding_start = time.perf_counter()
                vectors = embedder.encode_documents([row["text"] for row in batch], settings.batch_size)
                embedding_seconds += time.perf_counter() - embedding_start
                if vectors.shape != (len(batch), embedder.dimension):
                    raise ValueError(f"Batch vector dimensions mismatch: {vectors.shape}")
                upsert_start = time.perf_counter()
                store.upsert(batch, vectors)
                upsert_seconds += time.perf_counter() - upsert_start
                indexed += len(batch)
                log.info("Indexed %d passages", indexed)
                previous_batch_end = time.perf_counter()
            except Exception as batch_error:
                errors += len(batch)
                log.error("Failed to index a batch starting at local row %d (%s)", indexed, type(batch_error).__name__)
                raise
    except Exception as exc:
        failure_detail = str(exc)
        if settings.hf_token_value:
            failure_detail = failure_detail.replace(settings.hf_token_value, "[REDACTED]")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "started_at_utc": started_at, "dataset": settings.hf_dataset_id,
            "config": settings.hf_dataset_config, "split": settings.hf_dataset_split,
            "embedding_model": settings.embedding_model, "model_revision": embedder.model_revision,
            "vector_dimension": embedder.dimension, "precision": embedder.precision,
            "requested_precision": settings.embedding_precision,
            "batch_size": settings.batch_size, "requested_limit": settings.data_limit,
            "indexed_this_run": indexed, "count_before": seen_before, "count_after": store.count(),
            "errors": errors, "duration_seconds": time.perf_counter() - started,
            "stage_seconds": {"dataset_stream": dataset_stream_seconds, "embedding_and_vector_conversion": embedding_seconds,
                              "qdrant_upsert": upsert_seconds},
            "hnsw": {"m": settings.hnsw_m, "ef_construct": settings.hnsw_ef_construct},
            "failure": f"{type(exc).__name__}: {failure_detail}",
            "packages": {"datasets": _version("datasets"), "sentence-transformers": _version("sentence-transformers"),
                         "transformers": _version("transformers"), "torch": _version("torch"), "qdrant-client": _version("qdrant-client")},
            "hardware": {"machine": platform.platform(), "cpu": platform.processor() or socket.gethostname(),
                         "python": platform.python_version(), "device_requested": settings.device,
                         "device_actual": embedder.device}}
        (settings.output_dir / "indexing_report.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        store.close()
        raise
    try:
        duration = time.perf_counter() - started
        report = {"started_at_utc": started_at, "dataset": settings.hf_dataset_id,
            "config": settings.hf_dataset_config, "split": settings.hf_dataset_split,
            "embedding_model": settings.embedding_model, "baseline_embedding_model": settings.baseline_embedding_model,
            "model_revision": embedder.model_revision, "vector_dimension": embedder.dimension,
            "requested_dimension": settings.embedding_dimension or None, "precision": embedder.precision,
            "requested_precision": settings.embedding_precision,
            "batch_size": settings.batch_size, "requested_limit": settings.data_limit,
            "hnsw": {"m": settings.hnsw_m, "ef_construct": settings.hnsw_ef_construct},
            "indexed_this_run": indexed, "count_before": seen_before, "count_after": store.count(),
            "errors": errors, "duration_seconds": duration,
            "passages_per_second": indexed / duration if duration > 0 else None,
            "stage_seconds": {"dataset_stream": dataset_stream_seconds, "embedding_and_vector_conversion": embedding_seconds,
                              "qdrant_upsert": upsert_seconds},
            "packages": {"datasets": _version("datasets"), "sentence-transformers": _version("sentence-transformers"),
                         "transformers": _version("transformers"), "torch": _version("torch"), "qdrant-client": _version("qdrant-client")},
            "hardware": {"machine": platform.platform(),
                "cpu": platform.processor() or socket.gethostname(), "python": platform.python_version(),
                "device_requested": settings.device, "device_actual": embedder.device}}
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_dir / "indexing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        store.close()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from .config import get_settings
    print(json.dumps(run_index(get_settings()), indent=2))
