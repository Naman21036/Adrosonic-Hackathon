from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    from qdrant_client import QdrantClient, models
except ImportError:  # Keep unit tests and documentation tools usable before optional installation.
    QdrantClient = None
    models = None

from .config import settings
from .embedding import Embedder
from .observability.langfuse import observation
from .observability.logging import log_event

_ID_NAMESPACE = uuid.UUID("db8dcbb1-8011-4a09-83ec-6644a2ad64e8")
_SOURCE_ID_PATTERN = re.compile(r"^msmarco-(\d+)-(\d+)$")
_SOURCE_ID_SQL = "CAST(substr(id, 9, instr(substr(id, 9), '-') - 1) AS INTEGER)"

def _point_id(pid: str) -> str: return str(uuid.uuid5(_ID_NAMESPACE, pid))


def source_id_from_passage_id(passage_id: str) -> int | None:
    """Return the numeric MS MARCO query ID encoded in a canonical passage ID."""
    match = _SOURCE_ID_PATTERN.fullmatch(passage_id)
    return int(match.group(1)) if match else None


def _point_payload(passage_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {"passage_id": passage_id, "text": text, "metadata": metadata}
    source_id = source_id_from_passage_id(passage_id)
    if source_id is None:
        source_id = metadata.get("source_id")
    if source_id is not None:
        payload["source_id"] = source_id
    return payload


def validate_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    for key in meta:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
            raise ValueError(f"Invalid metadata field: {key}")
        if meta[key] is not None and type(meta[key]) not in (str,int,float,bool):
            raise ValueError(f"Metadata values must be scalar strings, numbers, booleans, or null: {key}")
        if key == "source_id" and not (
            (type(meta[key]) is int and meta[key] >= 0)
            or (type(meta[key]) is str and 0 < len(meta[key].strip()) <= 256)
        ):
            raise ValueError("source_id must be a non-negative integer or a non-empty string up to 256 characters")
    return meta


class SparseStore:
    """Persistent SQLite FTS5 BM25 index; SQL metadata predicates run before ranking."""
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS passage (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)")
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite FTS5 is required for the sparse index") from exc
        # The canonical passage ID encodes source_id. This expression index lets
        # SQLite restrict the candidate set before BM25 ranking without rewriting
        # any existing sparse records or rebuilding the FTS index.
        self.db.execute(f"CREATE INDEX IF NOT EXISTS idx_passage_source_id ON passage ({_SOURCE_ID_SQL})")
        self.db.commit()

    def upsert(self, pid: str, text: str, metadata: dict[str, Any]) -> None:
        self.upsert_many([(pid,text,metadata)])

    def upsert_many(self, rows: list[tuple[str,str,dict[str,Any]]]) -> None:
        with self.db:
            for pid,text,metadata in rows:
                self.db.execute("DELETE FROM passage_fts WHERE id=?", (pid,))
                self.db.execute("INSERT INTO passage_fts(id,text) VALUES (?,?)", (pid, text))
                self.db.execute("INSERT INTO passage(id,text,metadata) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text,metadata=excluded.metadata", (pid,text,json.dumps(metadata,ensure_ascii=False)))

    def delete(self, pid: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM passage_fts WHERE id=?", (pid,))
            self.db.execute("DELETE FROM passage WHERE id=?", (pid,))

    def search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        terms = re.findall(r"[\w]+", query, re.UNICODE)
        if not terms: return []
        match = " OR ".join('"' + t.replace('"','""') + '"' for t in terms)
        sql = "SELECT p.id,p.text,p.metadata,bm25(passage_fts) AS score FROM passage_fts JOIN passage p ON p.id=passage_fts.id WHERE passage_fts MATCH ?"
        args: list[Any] = [match]
        for key, value in (filters or {}).items():
            validate_metadata({key:value})
            if key == "source_id":
                if type(value) is int:
                    # Match the canonical MS MARCO ID prefix and explicitly stored
                    # source IDs without loading or rebuilding any FTS records.
                    sql += (
                        f" AND ((p.id LIKE ? AND CAST(substr(p.id, 9, instr(substr(p.id, 9), '-') - 1) AS INTEGER) = ?)"
                        " OR json_extract(p.metadata, '$.source_id') = ?)"
                    )
                    args.extend((f"msmarco-{value}-%", value, value))
                else:
                    sql += " AND json_extract(p.metadata, '$.source_id') = ?"
                    args.append(value)
            else:
                sql += " AND json_extract(p.metadata, ?) = ?"
                args.extend((f'$."{key}"', value))
        sql += " ORDER BY score ASC, p.id ASC LIMIT ?"
        args.append(limit)
        return [
            {
                "id":r[0],
                "text":r[1],
                "metadata":json.loads(r[2]),
                "source_id":source_id_from_passage_id(r[0]) if source_id_from_passage_id(r[0]) is not None else json.loads(r[2]).get("source_id"),
                "score":-float(r[3]),
            }
            for r in self.db.execute(sql,args).fetchall()
        ]

    def count(self) -> int: return int(self.db.execute("SELECT count(*) FROM passage").fetchone()[0])
    def contains(self, pid: str) -> bool: return self.db.execute("SELECT 1 FROM passage WHERE id=?",(pid,)).fetchone() is not None
    def source_id(self, pid: str) -> int | str | None:
        row = self.db.execute("SELECT metadata FROM passage WHERE id=?", (pid,)).fetchone()
        return json.loads(row[0]).get("source_id") if row else None
    def close(self) -> None: self.db.close()


class PassageNotFoundError(LookupError):
    """Raised when neither index contains the requested passage ID."""


class HybridIndex:
    def __init__(self, embedder: Embedder, qdrant_path: str | None = None, collection: str | None = None, sparse_path: str | None = None):
        self.embedder=embedder
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed; install requirements.txt")
        self.collection=collection or settings.qdrant_collection
        path=qdrant_path or settings.qdrant_path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.qdrant=QdrantClient(path=path)
        self.sparse=SparseStore(sparse_path or settings.bm25_index_path)
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(self.collection,vectors_config=models.VectorParams(size=embedder.dimension,distance=models.Distance.COSINE), hnsw_config=models.HnswConfigDiff(m=settings.hnsw_m,ef_construct=settings.hnsw_ef_construct))
        info=self.qdrant.get_collection(self.collection)
        if info.config.params.vectors.size != embedder.dimension:
            raise ValueError(f"Qdrant dimension {info.config.params.vectors.size} does not match embedder dimension {embedder.dimension}")
        for key, schema in (("source",models.PayloadSchemaType.KEYWORD),("split",models.PayloadSchemaType.KEYWORD),("category",models.PayloadSchemaType.KEYWORD),("relevance",models.PayloadSchemaType.BOOL)):
            try: self.qdrant.create_payload_index(self.collection,key=f"metadata.{key}",field_schema=schema)
            except Exception: pass
        try:
            self.qdrant.create_payload_index(
                self.collection,
                field_name="source_id",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception:
            payload_schema = getattr(self.qdrant.get_collection(self.collection), "payload_schema", {}) or {}
            source_index = payload_schema.get("source_id")
            source_type = getattr(source_index, "data_type", None)
            if source_type != models.PayloadSchemaType.INTEGER:
                raise RuntimeError("Qdrant source_id integer payload index is unavailable")
        qdrant_backend = getattr(self.qdrant, "_client", None)
        if getattr(qdrant_backend, "__class__", type(None)).__name__ == "QdrantLocal":
            log_event(
                "QDRANT_PAYLOAD_INDEX_UNSUPPORTED_LOCAL",
                level="WARNING",
                filter_dimension="source_id",
                requested_index_type="integer",
            )
        self._lock=threading.RLock()
        self._pending=set()
        self.last_upsert_timings={"embedding_seconds":0.0,"sparse_seconds":0.0,"qdrant_seconds":0.0}
        self.recover_pending()

    def _qfilter(self, filters: dict[str, Any] | None):
        if not filters: return None
        validate_metadata(filters)
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id" if key == "source_id" else f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
                for key,value in filters.items()
            ]
        )

    def dense_search(self, query: str, limit: int, filters: dict[str, Any] | None = None):
        results, _ = self.dense_search_with_timings(query, limit, filters)
        return results

    def dense_search_with_timings(self, query: str, limit: int, filters: dict[str, Any] | None = None):
        embedding_started = time.perf_counter()
        log_event("EMBEDDING_STARTED", model=self.embedder.__class__.__name__)
        with observation(
            "embedding",
            input={"query": query},
            metadata={"model": getattr(self.embedder, "model_name", self.embedder.__class__.__name__)},
        ) as embedding_observation:
            try:
                vector = self.embedder.encode_query(query)
            except Exception as exc:
                elapsed = (time.perf_counter() - embedding_started) * 1000
                log_event("EMBEDDING_FAILED", level="ERROR", embedding_latency_ms=elapsed, exception_type=type(exc).__name__)
                embedding_observation.update(level="ERROR", status_message=type(exc).__name__, metadata={"latency_ms": elapsed})
                raise
            embedding_latency = (time.perf_counter() - embedding_started) * 1000
            log_event(
                "EMBEDDING_COMPLETED",
                embedding_latency_ms=embedding_latency,
                embedding_dimension=int(len(vector)),
                model=getattr(self.embedder, "model_name", self.embedder.__class__.__name__),
            )
            embedding_observation.update(
                output={"dimension": int(len(vector)), "model": getattr(self.embedder, "model_name", self.embedder.__class__.__name__)},
                metadata={"latency_ms": embedding_latency},
            )

        qdrant_started = time.perf_counter()
        qdrant_filter = self._qfilter(filters)
        if filters:
            log_event(
                "QDRANT_FILTER_APPLIED",
                qdrant_filter_applied=qdrant_filter is not None,
                filter_enabled=True,
                filter_dimension=sorted(filters),
                filter_value=filters.get("source_id") if len(filters) == 1 else filters,
            )
        with observation(
            "qdrant_query",
            input={"collection": self.collection, "limit": limit, "filter": filters or {}},
        ) as qdrant_observation:
            try:
                hits = self.qdrant.query_points(
                    collection_name=self.collection,
                    query=vector.tolist(),
                    query_filter=qdrant_filter,
                    limit=limit,
                    search_params=models.SearchParams(hnsw_ef=settings.hnsw_ef_search),
                ).points
            except Exception as exc:
                latency = (time.perf_counter() - qdrant_started) * 1000
                log_event("QDRANT_QUERY_FAILED", level="ERROR", qdrant_collection=self.collection, qdrant_latency_ms=latency, exception_type=type(exc).__name__)
                qdrant_observation.update(level="ERROR", status_message=type(exc).__name__, metadata={"qdrant_latency_ms": latency})
                raise
        qdrant_latency = (time.perf_counter() - qdrant_started) * 1000
        results = [
            {
                "id": p.payload["passage_id"],
                "text": p.payload["text"],
                "metadata": p.payload.get("metadata", {}),
                "source_id": p.payload.get("source_id", source_id_from_passage_id(p.payload["passage_id"])),
                "score": float(p.score),
            }
            for p in hits
        ]
        top = results[:5]
        log_event(
            "QDRANT_QUERY_COMPLETED",
            qdrant_collection=self.collection,
            candidate_count=len(results),
            context_ids=[hit["id"] for hit in top],
            top_scores=[hit["score"] for hit in top],
            qdrant_latency_ms=qdrant_latency,
        )
        qdrant_observation.update(
            output={"candidate_count": len(results), "document_ids": [hit["id"] for hit in top], "scores": [hit["score"] for hit in top]},
            metadata={"qdrant_latency_ms": qdrant_latency},
        )
        return results, {
            "embedding_latency_ms": round(embedding_latency, 3),
            "qdrant_latency_ms": round(qdrant_latency, 3),
        }

    def sparse_search(self, query: str, limit: int, filters: dict[str, Any] | None = None):
        started = time.perf_counter()
        log_event("BM25_RETRIEVAL_STARTED", candidate_k=limit, filter_fields=sorted((filters or {}).keys()))
        with observation(
            "bm25_retrieval",
            input={"query":query,"candidate_k":limit,"filter_fields":sorted((filters or {}).keys())},
        ) as bm25_observation:
            try:
                results = self.sparse.search(query, limit, filters)
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                log_event("BM25_RETRIEVAL_FAILED", level="ERROR", bm25_latency_ms=latency, exception_type=type(exc).__name__)
                bm25_observation.update(level="ERROR", status_message=type(exc).__name__, metadata={"latency_ms": latency})
                raise
        latency = (time.perf_counter() - started) * 1000
        top = results[:5]
        log_event(
            "BM25_RETRIEVAL_COMPLETED",
            candidate_count=len(results),
            context_ids=[hit["id"] for hit in top],
            top_scores=[hit["score"] for hit in top],
            bm25_latency_ms=latency,
        )
        bm25_observation.update(
            output={"candidate_count": len(results), "document_ids": [hit["id"] for hit in top], "scores": [hit["score"] for hit in top]},
            metadata={"latency_ms": latency},
        )
        return results

    def upsert(self, pid: str, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        operation_started = time.perf_counter()
        if not isinstance(pid, str) or not pid.strip() or len(pid) > 256:
            raise ValueError("passage_id must be a non-empty string up to 256 characters")
        if not isinstance(text, str) or not text.strip() or len(text) > 20_000:
            raise ValueError("text must be a non-empty string up to 20000 characters")
        normalized_metadata = validate_metadata(dict(metadata or {}))
        derived_source_id = source_id_from_passage_id(pid)
        requested_source_id = normalized_metadata.get("source_id")
        if derived_source_id is not None and requested_source_id is not None and requested_source_id != derived_source_id:
            raise ValueError("source_id must match the numeric source ID encoded in the canonical passage ID")

        with self._lock:
            # Resolve earlier journaled work before appending another mutation for
            # the same index. This keeps event replay in write order after a failure.
            self.recover_pending()
            dense_before = self.count()
            sparse_before = self.sparse.count()
            dense_existing = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[_point_id(pid)],
                with_payload=False,
                with_vectors=False,
            )
            sparse_existing = self.sparse.contains(pid)
            updated = bool(dense_existing or sparse_existing)
            self.upsert_many([(pid, text, normalized_metadata)], live_mutation=True)
            dense_after = self.count()
            sparse_after = self.sparse.count()
            result = {
                "operation": "upsert",
                "status": "updated" if updated else "created",
                "created": not updated,
                "updated": updated,
                "passage_id": pid,
                "source_id": derived_source_id if derived_source_id is not None else normalized_metadata.get("source_id"),
                "embedding_dimension": int(self.embedder.dimension),
                "embedding_count_delta": 1,
                "indexed_passages_before": dense_before,
                "indexed_passages_after": dense_after,
                "dense_count_delta": dense_after - dense_before,
                "sparse_count_before": sparse_before,
                "sparse_count_after": sparse_after,
                "qdrant_updated": True,
                "bm25_updated": True,
                "stage_latency_ms": {
                    key.removesuffix("_seconds"): round(value * 1000, 3)
                    for key, value in self.last_upsert_timings.items()
                },
                "latency_ms": round((time.perf_counter() - operation_started) * 1000, 3),
                "pending_mutations": self.pending_count(),
            }
        log_event(
            "LIVE_UPSERT_COMPLETED",
            operation="upsert",
            passage_id=pid,
            source_id=result["source_id"],
            embedding_model=getattr(self.embedder, "model_name", self.embedder.__class__.__name__),
            embedding_dimension=result["embedding_dimension"],
            embedding_count_delta=1,
            latency_ms=result["latency_ms"],
            status=result["status"],
        )
        return result

    def upsert_many(self, rows: list[tuple[str,str,dict[str,Any]]], *, live_mutation: bool = False) -> None:
        if not rows: return
        normalized=[]
        for pid,text,metadata in rows:
            if not pid.strip() or not text.strip(): raise ValueError("passage_id and text must be non-empty")
            normalized.append((pid,text,validate_metadata(metadata)))
        embedding_started=time.perf_counter()
        vector_batches=[]
        texts=[text for _,text,_ in normalized]
        encode_batch_size=getattr(self.embedder,"batch_size",settings.embedding_batch_size)
        for start in range(0,len(texts),encode_batch_size):
            vector_batches.append(self.embedder.encode_documents(texts[start:start+encode_batch_size]))
        vectors=np.concatenate(vector_batches,axis=0)
        expected_shape=(len(normalized),self.embedder.dimension)
        if vectors.shape != expected_shape:
            raise ValueError(f"Embedding batch has shape {vectors.shape}; expected {expected_shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding batch contains NaN or infinite values")
        embedding_seconds=time.perf_counter()-embedding_started
        if live_mutation:
            pid, _, metadata = normalized[0]
            log_event(
                "LIVE_UPSERT_EMBEDDED",
                operation="upsert",
                passage_id=pid,
                source_id=metadata.get("source_id", source_id_from_passage_id(pid)),
                embedding_model=getattr(self.embedder, "model_name", self.embedder.__class__.__name__),
                embedding_dimension=int(vectors.shape[1]),
                embedding_count_delta=int(vectors.shape[0]),
                latency_ms=round(embedding_seconds * 1000, 3),
                status="completed",
            )
        with self._lock:
            events=[]
            for pid,text,metadata in normalized:
                event=uuid.uuid4().hex; events.append((event,pid))
                self._pending.add(event)
            self._journal_many([{"op":"upsert","event":event,"id":pid,"text":text,"metadata":metadata,"state":"pending"} for (event,pid),(pid,text,metadata) in zip(events,normalized)])
            # Keep the vector array float32 through point construction. Qdrant Local
            # may still widen persisted Python lists to float64 when reopening; that
            # backend limitation is addressed by using the local server for this corpus.
            vectors=np.asarray(vectors,dtype=np.float32,order="C")
            points=[models.PointStruct(id=_point_id(pid),vector=vector.tolist(),payload=_point_payload(pid,text,metadata)) for (pid,text,metadata),vector in zip(normalized,vectors)]
            qdrant_started=time.perf_counter()
            response=self.qdrant.upsert(self.collection,points,wait=True)
            status=getattr(response,"status",None)
            if status is not None and str(getattr(status,"value",status)).lower() not in {"completed","acknowledged"}:
                raise RuntimeError(f"Qdrant did not confirm the upsert (status={status})")
            qdrant_seconds=time.perf_counter()-qdrant_started
            if live_mutation:
                pid, _, metadata = normalized[0]
                log_event(
                    "LIVE_UPSERT_QDRANT_SUCCESS",
                    operation="upsert",
                    passage_id=pid,
                    source_id=metadata.get("source_id", source_id_from_passage_id(pid)),
                    latency_ms=round(qdrant_seconds * 1000, 3),
                    status="completed",
                )
            # Qdrant must be durable before BM25 acknowledges the same IDs. If this
            # SQLite transaction fails, the pending journal rows replay it on startup.
            sparse_started=time.perf_counter()
            self.sparse.upsert_many(normalized)
            sparse_seconds=time.perf_counter()-sparse_started
            if live_mutation:
                pid, _, metadata = normalized[0]
                log_event(
                    "LIVE_UPSERT_BM25_SUCCESS",
                    operation="upsert",
                    passage_id=pid,
                    source_id=metadata.get("source_id", source_id_from_passage_id(pid)),
                    latency_ms=round(sparse_seconds * 1000, 3),
                    status="completed",
                )
            self._journal_many([{"op":"commit","event":event,"id":pid} for event,pid in events])
            for event,_ in events: self._pending.discard(event)
            self._compact_journal()
            self.last_upsert_timings={"embedding_seconds":embedding_seconds,"sparse_seconds":sparse_seconds,"qdrant_seconds":qdrant_seconds}

    def delete(self, pid: str) -> dict[str, Any]:
        operation_started = time.perf_counter()
        if not isinstance(pid, str) or not pid.strip() or len(pid) > 256:
            raise ValueError("passage_id must be a non-empty string up to 256 characters")
        with self._lock:
            self.recover_pending()
            dense_before = self.count()
            sparse_before = self.sparse.count()
            dense_existing = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[_point_id(pid)],
                with_payload=True,
                with_vectors=False,
            )
            sparse_existing = self.sparse.contains(pid)
            if not dense_existing and not sparse_existing:
                raise PassageNotFoundError(pid)
            source_id = source_id_from_passage_id(pid)
            if source_id is None:
                source_id = (dense_existing[0].payload or {}).get("source_id") if dense_existing else None
            if source_id is None:
                source_id = self.sparse.source_id(pid)
            event=uuid.uuid4().hex
            self._journal({"op":"delete","event":event,"id":pid,"state":"pending"})
            self._pending.add(event)
            response = self.qdrant.delete(self.collection,points_selector=models.PointIdsList(points=[_point_id(pid)]),wait=True)
            status=getattr(response,"status",None)
            if status is not None and str(getattr(status,"value",status)).lower() not in {"completed","acknowledged"}:
                raise RuntimeError(f"Qdrant did not confirm the delete (status={status})")
            log_event(
                "LIVE_DELETE_QDRANT_SUCCESS",
                operation="delete",
                passage_id=pid,
                source_id=source_id,
                deleted_from_qdrant=bool(dense_existing),
                status="completed",
            )
            self.sparse.delete(pid)
            log_event(
                "LIVE_DELETE_BM25_SUCCESS",
                operation="delete",
                passage_id=pid,
                source_id=source_id,
                deleted_from_bm25=bool(sparse_existing),
                status="completed",
            )
            self._journal({"op":"commit","event":event,"id":pid}); self._pending.discard(event); self._compact_journal()
            dense_after = self.count()
            sparse_after = self.sparse.count()
            result = {
                "operation": "delete",
                "status": "deleted",
                "passage_id": pid,
                "source_id": source_id,
                "deleted_from_qdrant": bool(dense_existing),
                "deleted_from_bm25": bool(sparse_existing),
                "embedding_count_delta": 0,
                "indexed_passages_before": dense_before,
                "indexed_passages_after": dense_after,
                "dense_count_delta": dense_after - dense_before,
                "sparse_count_before": sparse_before,
                "sparse_count_after": sparse_after,
                "latency_ms": round((time.perf_counter() - operation_started) * 1000, 3),
                "pending_mutations": self.pending_count(),
            }
        log_event(
            "LIVE_DELETE_COMPLETED",
            operation="delete",
            passage_id=pid,
            source_id=result["source_id"],
            embedding_count_delta=0,
            latency_ms=result["latency_ms"],
            status="completed",
        )
        return result

    @staticmethod
    def _journal(row: dict[str, Any]) -> None:
        HybridIndex._journal_many([row])

    @staticmethod
    def _journal_many(rows: list[dict[str, Any]]) -> None:
        if not rows: return
        Path(settings.journal_path).parent.mkdir(parents=True,exist_ok=True)
        with open(settings.journal_path,"a",encoding="utf-8") as log:
            log.writelines(json.dumps(row,ensure_ascii=False)+"\n" for row in rows)
            log.flush(); os.fsync(log.fileno())

    def _compact_journal(self) -> None:
        if self._pending: return
        path=Path(settings.journal_path)
        if not path.exists(): return
        temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_text("",encoding="utf-8")
        os.replace(temporary,path)

    def recover_pending(self) -> None:
        path=Path(settings.journal_path)
        if not path.exists(): return
        pending={}
        for line in path.read_text(encoding="utf-8").splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if row.get("op") in {"upsert","delete"}: pending[row.get("event",row.get("id"))]=row
            elif row.get("op")=="commit": pending.pop(row.get("event",row.get("id")),None)
        for event,row in pending.items():
            self._pending.add(event)
            pid=row["id"]
            if row["op"]=="upsert":
                meta=row.get("metadata",{})
                existing=self.qdrant.retrieve(
                    collection_name=self.collection,
                    ids=[_point_id(pid)],
                    with_payload=True,
                    with_vectors=False,
                )
                payload=existing[0].payload if existing else None
                if not payload or payload.get("passage_id") != pid or payload.get("text") != row["text"]:
                    vector=np.asarray(self.embedder.encode_documents([row["text"]])[0],dtype=np.float32)
                    response=self.qdrant.upsert(self.collection,[models.PointStruct(id=_point_id(pid),vector=vector,payload=_point_payload(pid,row["text"],meta))],wait=True)
                    status=getattr(response,"status",None)
                    if status is not None and str(getattr(status,"value",status)).lower() not in {"completed","acknowledged"}:
                        raise RuntimeError(f"Qdrant did not confirm journal recovery for {pid} (status={status})")
                    existing=self.qdrant.retrieve(collection_name=self.collection,ids=[_point_id(pid)],with_payload=True,with_vectors=False)
                    payload=existing[0].payload if existing else None
                    if not payload or payload.get("passage_id") != pid or payload.get("text") != row["text"]:
                        raise RuntimeError(f"Qdrant does not contain the recovered passage {pid}")
                self.sparse.upsert(pid,row["text"],meta)
            else:
                self.qdrant.delete(self.collection,points_selector=models.PointIdsList(points=[_point_id(pid)]),wait=True)
                self.sparse.delete(pid)
            self._journal({"op":"commit","event":row.get("event",event),"id":pid,"recovered":True})
            self._pending.discard(event)
        self._compact_journal()

    def count(self) -> int: return self.qdrant.count(self.collection,exact=True).count
    def pending_count(self) -> int: return len(self._pending)
    def close(self) -> None: self.sparse.close(); self.qdrant.close()
