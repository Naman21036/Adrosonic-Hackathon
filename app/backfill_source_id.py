"""Safely derive and index source_id in the existing Local Qdrant payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from .config import settings
from .retrieval import search as retrieval_search
from .storage import HybridIndex, SparseStore

_PASSAGE_ID = re.compile(r"^msmarco-(\d+)-(\d+)$")
_REPORT_PATH = Path("reports/phase2/metadata_filter_migration.json")
_SCAN_PAGE_SIZE = 2048


def _parse_source_id(passage_id: str) -> int | None:
    match = _PASSAGE_ID.fullmatch(passage_id)
    return int(match.group(1)) if match else None


def _xor_hash(accumulator: bytearray, value: bytes) -> None:
    digest = hashlib.sha256(value).digest()
    for position, byte in enumerate(digest):
        accumulator[position] ^= byte


def _read_sparse_ids() -> tuple[set[str], Counter[int], dict[int, str]]:
    path = Path(settings.bm25_index_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Sparse database is missing: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        ids: set[str] = set()
        counts: Counter[int] = Counter()
        first_ids: dict[int, str] = {}
        for (passage_id,) in connection.execute("SELECT id FROM passage"):
            if not isinstance(passage_id, str) or (source_id := _parse_source_id(passage_id)) is None:
                raise RuntimeError(f"Sparse ID is not canonical MS MARCO form: {passage_id!r}")
            if passage_id in ids:
                raise RuntimeError(f"Sparse index contains a duplicate passage ID: {passage_id}")
            ids.add(passage_id)
            counts[source_id] += 1
            first_ids.setdefault(source_id, passage_id)
        return ids, counts, first_ids
    finally:
        connection.close()


def _vector_values(point: Any) -> np.ndarray:
    vector = point.vector
    if isinstance(vector, dict):
        if len(vector) != 1:
            raise RuntimeError(f"Expected one dense vector per point, found {len(vector)}")
        vector = next(iter(vector.values()))
    values = np.asarray(vector, dtype="<f8")
    if values.ndim != 1 or values.size != 1024 or not np.isfinite(values).all():
        raise RuntimeError(f"Point {point.id} has an invalid vector shape or values: {values.shape}")
    return values


def _scan_collection(
    client: QdrantClient,
    expected_count: int,
    sparse_ids: set[str],
    first_ids: dict[int, str],
    candidate_sources: set[int],
) -> dict[str, Any]:
    info = client.get_collection(settings.qdrant_collection)
    vector_config = info.config.params.vectors
    if isinstance(vector_config, dict):
        raise RuntimeError("The collection uses named vectors; expected the validated single-vector configuration")
    if vector_config.size != 1024 or vector_config.distance != models.Distance.COSINE:
        raise RuntimeError(
            f"Unexpected Qdrant vector configuration: size={vector_config.size}, distance={vector_config.distance}"
        )
    if info.points_count != expected_count:
        raise RuntimeError(f"Expected {expected_count} Qdrant points, found {info.points_count}")

    point_ids: set[str] = set()
    passage_ids: set[str] = set()
    source_counts: Counter[int] = Counter()
    missing_payload_ids: dict[int, list[str]] = defaultdict(list)
    sample_vectors: dict[int, tuple[str, list[float]]] = {}
    vector_digest = bytearray(32)
    text_digest = bytearray(32)
    malformed = 0
    offset = None

    while True:
        page, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=_SCAN_PAGE_SIZE,
            offset=offset,
            with_payload=["passage_id", "text", "metadata", "source_id"],
            with_vectors=True,
        )
        for point in page:
            payload = point.payload or {}
            passage_id = payload.get("passage_id")
            text = payload.get("text")
            source_id = _parse_source_id(passage_id) if isinstance(passage_id, str) else None
            if source_id is None or not isinstance(text, str) or not text.strip():
                malformed += 1
                continue
            if str(point.id) in point_ids or passage_id in passage_ids:
                raise RuntimeError(f"Qdrant contains a duplicate point or passage ID: {passage_id}")
            point_ids.add(str(point.id))
            passage_ids.add(passage_id)
            source_counts[source_id] += 1

            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise RuntimeError(f"Qdrant metadata is not an object for {passage_id}")
            nested_source = metadata.get("source_id")
            if nested_source is not None and (type(nested_source) is not int or nested_source != source_id):
                raise RuntimeError(f"Conflicting nested source_id in Qdrant metadata for {passage_id}")
            if "source_id" in payload:
                if type(payload["source_id"]) is not int or payload["source_id"] != source_id:
                    raise RuntimeError(f"Conflicting top-level source_id in Qdrant payload for {passage_id}")
            else:
                missing_payload_ids[source_id].append(str(point.id))

            vector = _vector_values(point)
            _xor_hash(vector_digest, str(point.id).encode("utf-8") + b"\0" + vector.tobytes())
            _xor_hash(text_digest, passage_id.encode("utf-8") + b"\0" + text.encode("utf-8"))
            if source_id in candidate_sources and source_id not in sample_vectors and passage_id == first_ids[source_id]:
                sample_vectors[source_id] = (passage_id, vector.tolist())

        if next_offset is None:
            break
        if next_offset == offset:
            raise RuntimeError("Qdrant scroll did not advance")
        offset = next_offset

    if malformed:
        raise RuntimeError(f"Qdrant contained {malformed} malformed IDs or empty payload texts")
    if len(point_ids) != expected_count or len(passage_ids) != expected_count:
        raise RuntimeError(
            f"Qdrant unique counts differ from expected: points={len(point_ids)}, passages={len(passage_ids)}"
        )
    if passage_ids != sparse_ids:
        dense_only = len(passage_ids - sparse_ids)
        sparse_only = len(sparse_ids - passage_ids)
        raise RuntimeError(f"Dense/sparse IDs differ: dense_only={dense_only}, sparse_only={sparse_only}")

    return {
        "point_ids": point_ids,
        "passage_ids": passage_ids,
        "source_counts": source_counts,
        "missing_payload_ids": dict(missing_payload_ids),
        "sample_vectors": sample_vectors,
        "vector_digest": vector_digest.hex(),
        "text_digest": text_digest.hex(),
        "points_count": info.points_count,
        "payload_schema": getattr(info, "payload_schema", {}) or {},
    }


def _is_local_qdrant(client: QdrantClient) -> bool:
    return getattr(getattr(client, "_client", None), "__class__", type(None)).__name__ == "QdrantLocal"


def _index_source_id(client: QdrantClient, schema: dict[str, Any]) -> bool:
    current = schema.get("source_id")
    if current is not None:
        if getattr(current, "data_type", None) != models.PayloadSchemaType.INTEGER:
            raise RuntimeError("An existing source_id payload index has a non-integer schema; aborting")
        return True
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source_id",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True,
    )
    if _is_local_qdrant(client):
        return False
    schema_after = getattr(client.get_collection(settings.qdrant_collection), "payload_schema", {}) or {}
    source_index = schema_after.get("source_id")
    if getattr(source_index, "data_type", None) != models.PayloadSchemaType.INTEGER:
        raise RuntimeError("Qdrant did not create the INTEGER source_id payload index")
    return True


def _query_point(client: QdrantClient, vector: list[float], limit: int, source_id: int | None = None):
    query_filter = None
    if source_id is not None:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
        )
    started = time.perf_counter()
    points = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=["passage_id", "source_id"],
        with_vectors=False,
    ).points
    return points, (time.perf_counter() - started) * 1000, query_filter


class _StoredVectorEmbedder:
    """Return one existing corpus vector for retrieval-path verification, without loading Qwen."""

    dimension = 1024
    model_name = "stored-corpus-vector-fr4-smoke"

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.vector


class _QueryFilterCapture:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self.filtered_calls = 0

    def query_points(self, **kwargs):
        if kwargs.get("query_filter") is not None:
            self.filtered_calls += 1
        return self.client.query_points(**kwargs)


def _save_report(report: dict[str, Any]) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, default=89_896)
    parser.add_argument("--source-id", type=int, default=19_699)
    parser.add_argument("--apply", action="store_true", help="write only derived source_id payloads and its index")
    args = parser.parse_args()
    if args.expected_count < 1 or args.source_id < 0:
        parser.error("expected-count must be positive and source-id non-negative")

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight",
        "qdrant_path": str(Path(settings.qdrant_path).resolve()),
        "collection": settings.qdrant_collection,
        "expected_count": args.expected_count,
        "source_id_format": "msmarco-{numeric source_id}-{numeric passage_position}",
        "write_mode": "payload_only" if args.apply else "dry_run",
    }
    client = None
    try:
        sparse_ids, sparse_counts, first_ids = _read_sparse_ids()
        if len(sparse_ids) != args.expected_count:
            raise RuntimeError(f"Expected {args.expected_count} sparse IDs, found {len(sparse_ids)}")
        if sparse_counts[args.source_id] < 2:
            raise RuntimeError(f"Requested demo source_id={args.source_id} has fewer than two sparse passages")
        candidate_order = [args.source_id]
        candidate_order.extend(
            source_id for source_id, _ in sparse_counts.most_common(48) if source_id != args.source_id
        )
        candidate_sources = set(candidate_order)
        client = QdrantClient(path=settings.qdrant_path)
        before = _scan_collection(client, args.expected_count, sparse_ids, first_ids, candidate_sources)
        if before["source_counts"][args.source_id] < 2:
            raise RuntimeError(f"Requested demo source_id={args.source_id} has fewer than two Qdrant passages")

        missing = before["missing_payload_ids"]
        to_update = sum(len(ids) for ids in missing.values())
        report.update(
            status="preflight_passed",
            dense_count=before["points_count"],
            sparse_count=len(sparse_ids),
            dense_unique_ids=len(before["passage_ids"]),
            sparse_unique_ids=len(sparse_ids),
            ids_match=True,
            embedding_dimension=1024,
            distance="Cosine",
            existing_payload_index_fields=sorted(before["payload_schema"]),
            qdrant_source_id_missing_before=to_update,
            qdrant_source_id_existing_and_valid=before["points_count"] - to_update,
            selected_source_id=args.source_id,
            selected_source_passages=before["source_counts"][args.source_id],
            vectors_sha256_multiset_before=before["vector_digest"],
            passage_text_sha256_multiset_before=before["text_digest"],
        )
        print(
            f"preflight_passed=true qdrant_count={before['points_count']} sparse_count={len(sparse_ids)} "
            f"ids_match=true source_id={args.source_id} source_passages={before['source_counts'][args.source_id]} "
            f"payloads_to_update={to_update}"
        )
        if not args.apply:
            report["status"] = "dry_run_complete"
            _save_report(report)
            print(f"dry_run_only=true report={_REPORT_PATH}")
            return

        payload_index_created = _index_source_id(client, before["payload_schema"])
        operations: list[Any] = []
        for source_id, point_ids in missing.items():
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload={"source_id": source_id}, points=point_ids)
                )
            )
        updated = 0
        operation_groups = 128
        for start in range(0, len(operations), operation_groups):
            batch = operations[start : start + operation_groups]
            results = client.batch_update_points(
                collection_name=settings.qdrant_collection,
                update_operations=batch,
                wait=True,
            )
            if len(results) != len(batch) or any(
                str(getattr(result.status, "value", result.status)).lower() not in {"completed", "acknowledged"}
                for result in results
            ):
                raise RuntimeError(f"Qdrant did not acknowledge source_id payload batch {start // operation_groups + 1}")
            updated += sum(len(operation.set_payload.points or []) for operation in batch)
            print(f"payload_backfill_progress={updated}/{to_update}")

        after = _scan_collection(client, args.expected_count, sparse_ids, first_ids, candidate_sources)
        if after["missing_payload_ids"]:
            left = sum(len(ids) for ids in after["missing_payload_ids"].values())
            raise RuntimeError(f"{left} Qdrant points remain without source_id")
        if after["point_ids"] != before["point_ids"] or after["passage_ids"] != before["passage_ids"]:
            raise RuntimeError("Qdrant point or passage IDs changed during metadata backfill")
        if after["text_digest"] != before["text_digest"]:
            raise RuntimeError("Qdrant passage text changed during metadata backfill")
        if after["vector_digest"] != before["vector_digest"]:
            raise RuntimeError("Qdrant vectors changed during metadata backfill")

        selected = None
        demo = None
        fallback_demo = None
        for source_id in candidate_order:
            sample = after["sample_vectors"].get(source_id)
            if sample is None or after["source_counts"][source_id] < 2:
                continue
            passage_id, vector = sample
            unfiltered, unfiltered_ms, _ = _query_point(client, vector, limit=5)
            filtered, filtered_ms, query_filter = _query_point(client, vector, limit=5, source_id=source_id)
            unfiltered_sources = sorted({point.payload.get("source_id") for point in unfiltered if point.payload})
            filtered_ids = [point.payload.get("passage_id") for point in filtered if point.payload]
            filtered_sources = sorted({point.payload.get("source_id") for point in filtered if point.payload})
            if filtered_sources == [source_id] and all(
                point.payload.get("source_id") == source_id for point in filtered if point.payload
            ):
                candidate_demo = {
                    "source_id": source_id,
                    "seed_passage_id": passage_id,
                    "unfiltered_dense_count": len(unfiltered),
                    "unfiltered_dense_source_ids": unfiltered_sources,
                    "filtered_dense_count": len(filtered),
                    "filtered_dense_result_source_ids": filtered_sources,
                    "filtered_dense_passage_ids": filtered_ids,
                    "unfiltered_latency_ms": unfiltered_ms,
                    "filtered_latency_ms": filtered_ms,
                    "qdrant_filter_applied_before_similarity_search": query_filter is not None,
                    "qdrant_filter_dimension": "source_id",
                }
                if fallback_demo is None:
                    fallback_demo = candidate_demo
                if len(unfiltered_sources) > 1:
                    selected = source_id
                    demo = candidate_demo
                    break
        if demo is None:
            selected = fallback_demo["source_id"] if fallback_demo else None
            demo = fallback_demo
        if demo is None:
            raise RuntimeError("No stored query vector successfully exercised the source_id Qdrant filter")
        demo["unfiltered_dense_has_multiple_source_ids"] = len(demo["unfiltered_dense_source_ids"]) > 1

        demo_passage_id = demo["seed_passage_id"]
        sparse_store = SparseStore(settings.bm25_index_path)
        try:
            text_row = sparse_store.db.execute("SELECT text FROM passage WHERE id=?", (demo_passage_id,)).fetchone()
            if not text_row:
                raise RuntimeError(f"Demo passage is missing from BM25: {demo_passage_id}")
            demo_query = " ".join(re.findall(r"[\w]+", text_row[0], re.UNICODE)[:12])
            if not demo_query:
                raise RuntimeError("Could not derive a non-empty demo query from the real indexed passage")
            vector = after["sample_vectors"][selected][1]
            index = object.__new__(HybridIndex)
            index.embedder = _StoredVectorEmbedder(vector)
            index.collection = settings.qdrant_collection
            qdrant_capture = _QueryFilterCapture(client)
            index.qdrant = qdrant_capture
            index.sparse = sparse_store

            dense_unfiltered = retrieval_search(index, demo_query, "dense", 5, {})
            dense_filtered = retrieval_search(index, demo_query, "dense", 5, {"source_id": selected})
            sparse_started = time.perf_counter()
            sparse_unfiltered = index.sparse_search(demo_query, settings.sparse_candidate_k, {})[:5]
            sparse_unfiltered_ms = (time.perf_counter() - sparse_started) * 1000
            sparse_started = time.perf_counter()
            sparse_filtered = index.sparse_search(demo_query, settings.sparse_candidate_k, {"source_id": selected})[:5]
            sparse_filtered_ms = (time.perf_counter() - sparse_started) * 1000
            hybrid_unfiltered = retrieval_search(index, demo_query, "hybrid", 5, {})
            hybrid_filtered = retrieval_search(index, demo_query, "hybrid", 5, {"source_id": selected})

            for mode, hits in (
                ("dense", dense_filtered["results"]),
                ("sparse", sparse_filtered),
                ("hybrid", hybrid_filtered["results"]),
            ):
                if any(hit.get("source_id") != selected for hit in hits):
                    raise RuntimeError(f"Filtered {mode} retrieval returned a foreign source_id")
            missing_source_id = 987654321
            missing_dense = retrieval_search(index, demo_query, "dense", 5, {"source_id": missing_source_id})
            missing_sparse = index.sparse_search(demo_query, 5, {"source_id": missing_source_id})
            missing_hybrid = retrieval_search(index, demo_query, "hybrid", 5, {"source_id": missing_source_id})
            if missing_dense["results"] or missing_sparse or missing_hybrid["results"]:
                raise RuntimeError("A nonexistent source_id unexpectedly returned retrieval results")
            if qdrant_capture.filtered_calls < 4:
                raise RuntimeError("The Qdrant query_filter was not supplied for all filtered dense/hybrid searches")

            source_ids = lambda hits: sorted({hit.get("source_id") for hit in hits if hit.get("source_id") is not None})
            demo.update(
                query_source="first 12 terms from the selected indexed passage; not represented as an official query",
                query_preview=demo_query,
                dense_unfiltered_top5_ids=[hit["id"] for hit in dense_unfiltered["results"]],
                dense_unfiltered_top5_source_ids=source_ids(dense_unfiltered["results"]),
                dense_filtered_top5_ids=[hit["id"] for hit in dense_filtered["results"]],
                dense_filtered_top5_source_ids=source_ids(dense_filtered["results"]),
                dense_unfiltered_latency_ms=dense_unfiltered["latency_ms"],
                dense_filtered_latency_ms=dense_filtered["latency_ms"],
                sparse_unfiltered_top5_ids=[hit["id"] for hit in sparse_unfiltered],
                sparse_unfiltered_top5_source_ids=source_ids(sparse_unfiltered),
                sparse_filtered_top5_ids=[hit["id"] for hit in sparse_filtered],
                sparse_filtered_top5_source_ids=source_ids(sparse_filtered),
                sparse_unfiltered_latency_ms=sparse_unfiltered_ms,
                sparse_filtered_latency_ms=sparse_filtered_ms,
                hybrid_unfiltered_top5_ids=[hit["id"] for hit in hybrid_unfiltered["results"]],
                hybrid_unfiltered_top5_source_ids=source_ids(hybrid_unfiltered["results"]),
                hybrid_filtered_top5_ids=[hit["id"] for hit in hybrid_filtered["results"]],
                hybrid_filtered_top5_source_ids=source_ids(hybrid_filtered["results"]),
                hybrid_unfiltered_latency_ms=hybrid_unfiltered["latency_ms"],
                hybrid_filtered_latency_ms=hybrid_filtered["latency_ms"],
                filtered_modes_all_match_source_id=True,
                qdrant_query_filter_calls_captured=qdrant_capture.filtered_calls,
                nonexistent_source_id=missing_source_id,
                nonexistent_source_id_returns_empty=True,
            )
        finally:
            sparse_store.close()

        if payload_index_created:
            schema_after = getattr(client.get_collection(settings.qdrant_collection), "payload_schema", {}) or {}
            source_index = schema_after.get("source_id")
            if getattr(source_index, "data_type", None) != models.PayloadSchemaType.INTEGER:
                raise RuntimeError("Qdrant source_id index was not created with INTEGER schema")
        report.update(
            status="payload_backfill_complete" if payload_index_created else "payload_backfill_complete_local_index_unavailable",
            qdrant_source_id_missing_after=0,
            qdrant_source_id_payloads_updated=updated,
            qdrant_payload_index_created=payload_index_created,
            qdrant_source_id_payload_index_type="integer" if payload_index_created else None,
            qdrant_payload_index_limitation=(
                None
                if payload_index_created
                else "Qdrant Local accepted the create request as a no-op; its documented backend implementation does not support payload indexes."
            ),
            qdrant_count_after=after["points_count"],
            sparse_count_after=len(sparse_ids),
            ids_unchanged=after["point_ids"] == before["point_ids"] and after["passage_ids"] == before["passage_ids"],
            texts_unchanged=after["text_digest"] == before["text_digest"],
            vectors_unchanged=after["vector_digest"] == before["vector_digest"],
            vectors_sha256_multiset_after=after["vector_digest"],
            passage_text_sha256_multiset_after=after["text_digest"],
            demo_source_id=selected,
            retrieval_demo=demo,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _save_report(report)
        print(
            f"payload_backfill_complete=true updated={updated} qdrant_count={after['points_count']} "
            f"sparse_count={len(sparse_ids)} ids_unchanged=true texts_unchanged=true vectors_unchanged=true"
        )
        index_text = "integer" if payload_index_created else "unavailable_in_qdrant_local"
        print(f"source_id_payload_index={index_text} demo_source_id={selected} report={_REPORT_PATH}")
    except Exception as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__, error=str(exc))
        _save_report(report)
        raise
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
