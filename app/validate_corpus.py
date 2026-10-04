"""Validate dense Qdrant and sparse SQLite corpus parity.

Run after ingestion with ``python -m app.validate_corpus``. The command always
writes a machine-readable report, including when a prerequisite is unavailable.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

_ID_NAMESPACE = uuid.UUID("db8dcbb1-8011-4a09-83ec-6644a2ad64e8")
REPORT = Path("reports/phase2/corpus_validation.json")


def validate() -> dict:
    result = {
        "status": "blocked",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": f"{settings.dataset_name}/{settings.dataset_config}:train",
        "expected_passages": settings.validation_expected_passages,
        "qdrant_count": None,
        "sparse_count": None,
        "qdrant_unique_ids": None,
        "sparse_unique_ids": None,
        "ids_match": None,
        "nonempty_texts": None,
        "qdrant_nonempty_texts": None,
        "embedding_dimension": None,
        "qdrant_distance": None,
        "qdrant_search_succeeded": None,
        "sample_ids_retrievable": None,
        "errors": [],
    }
    sparse_path = Path(settings.bm25_index_path)
    qdrant_path = Path(settings.qdrant_path)
    if not sparse_path.exists():
        result["errors"].append(f"Sparse index is absent: {sparse_path}")
    if not qdrant_path.exists():
        result["errors"].append(f"Qdrant storage is absent: {qdrant_path}")
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        result["errors"].append("qdrant-client is not installed; Qdrant cannot be inspected")
        return result
    if not sparse_path.exists() or not qdrant_path.exists():
        return result

    try:
        client = QdrantClient(path=str(qdrant_path))
        if not client.collection_exists(settings.qdrant_collection):
            client.close()
            result["errors"].append(f"Qdrant collection is absent: {settings.qdrant_collection}")
            return result
        db = sqlite3.connect(f"file:{sparse_path.resolve().as_posix()}?mode=ro", uri=True)
    except Exception as exc:
        result["errors"].append(f"Could not open configured indexes: {type(exc).__name__}: {exc}")
        try:
            client.close()
        except Exception:
            pass
        return result
    try:
        result["qdrant_count"] = client.count(settings.qdrant_collection, exact=True).count
        sparse_rows = db.execute("SELECT id,text FROM passage ORDER BY id").fetchall()
        result["sparse_count"] = len(sparse_rows)
        sparse_ids = [row[0] for row in sparse_rows]
        sparse_text_ids = {row[0] for row in sparse_rows if row[1] and row[1].strip()}
        result["sparse_unique_ids"] = len(set(sparse_ids))
        result["nonempty_texts"] = len(sparse_text_ids) == len(sparse_rows)
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            with_payload=["passage_id", "text"],
            with_vectors=False,
        )
        dense_ids = {point.payload.get("passage_id") for point in points if point.payload}
        dense_texts_nonempty = all(
            point.payload.get("text", "").strip()
            for point in points if point.payload
        )
        result["qdrant_unique_ids"] = len(dense_ids)
        # Scroll all pages: page size is bounded to avoid a large transient allocation.
        offset = points[-1].id if points else None
        while offset is not None:
            page, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=1000,
                offset=offset,
                with_payload=["passage_id", "text"],
                with_vectors=False,
            )
            dense_ids.update(point.payload.get("passage_id") for point in page if point.payload)
            dense_texts_nonempty = dense_texts_nonempty and all(
                point.payload.get("text", "").strip()
                for point in page if point.payload
            )
        result["qdrant_unique_ids"] = len(dense_ids)
        result["qdrant_nonempty_texts"] = dense_texts_nonempty and len(dense_ids) == result["qdrant_count"]
        result["ids_match"] = set(sparse_ids) == dense_ids and len(sparse_ids) == len(set(sparse_ids))
        result["qdrant_count"] = client.count(settings.qdrant_collection, exact=True).count
        result["qdrant_nonempty_texts"] = dense_texts_nonempty and len(dense_ids) == result["qdrant_count"]
        expected = settings.validation_expected_passages
        count_ok = result["qdrant_count"] == expected and result["sparse_count"] == expected
        if count_ok and result["ids_match"] and result["nonempty_texts"] and result["qdrant_nonempty_texts"]:
            result["status"] = "passed"
        else:
            result["errors"].append("Dense/sparse counts, IDs, or passage text failed corpus acceptance checks")
            result["status"] = "failed"
        if sparse_ids:
            samples = [sparse_ids[0], sparse_ids[len(sparse_ids) // 2], sparse_ids[-1]]
            retrieved = client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[str(uuid.uuid5(_ID_NAMESPACE, pid)) for pid in samples],
                with_payload=["passage_id", "text"],
            )
            result["sample_ids_retrievable"] = len(retrieved) == len(set(samples))
        else:
            result["sample_ids_retrievable"] = False
        # Search execution and vector configuration require the configured live embedder.
        try:
            from .embedding import QwenEmbedder
            embedder = QwenEmbedder(
                settings.hf_model_name, settings.hf_model_revision,
                settings.device, settings.huggingface_api_key,
            )
            result["embedding_dimension"] = embedder.dimension
            vector_info = client.get_collection(settings.qdrant_collection).config.params.vectors
            result["qdrant_distance"] = str(vector_info.distance)
            if vector_info.size != embedder.dimension:
                result["errors"].append("Qdrant vector size differs from the loaded embedding dimension")
                result["status"] = "failed"
            if str(vector_info.distance).lower() != "cosine":
                result["errors"].append("Qdrant collection distance is not cosine")
                result["status"] = "failed"
            result["qdrant_search_succeeded"] = bool(client.query_points(
                collection_name=settings.qdrant_collection,
                query=embedder.encode_query("validation query").tolist(),
                limit=1,
            ).points)
            del embedder
        except Exception as exc:
            result["errors"].append(f"Embedding/search verification failed: {type(exc).__name__}: {exc}")
            result["qdrant_search_succeeded"] = False
            result["status"] = "failed"
    finally:
        db.close()
        client.close()
    return result


def main() -> None:
    report = validate()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
