"""Persistent Qdrant local collection management."""
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient, models


class PassageStore:
    def __init__(self, path: Path, collection: str, vector_size: int | None = None,
                 hnsw_m: int = 16, ef_construct: int = 100) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection
        self._m, self._ef_construct = hnsw_m, ef_construct
        if vector_size is not None:
            self.ensure_collection(vector_size)

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            params = info.config.params.vectors
            if isinstance(params, dict):
                raise ValueError("Named-vector collections are not supported by this baseline")
            if params.size != vector_size or params.distance != models.Distance.COSINE:
                raise ValueError(f"Collection schema mismatch: expected {vector_size}/COSINE, found {params}")
            hnsw = info.config.hnsw_config
            if hnsw.m != self._m or hnsw.ef_construct != self._ef_construct:
                raise ValueError(f"Collection HNSW schema mismatch: expected m={self._m}, "
                                 f"ef_construct={self._ef_construct}; found m={hnsw.m}, "
                                 f"ef_construct={hnsw.ef_construct}")
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=self._m, ef_construct=self._ef_construct),
        )

    def upsert(self, records: list[dict[str, Any]], vectors: Any) -> None:
        points = []
        for row, vector in zip(records, vectors):
            # Numeric MS MARCO passage IDs map directly to Qdrant's compact int64 point IDs.
            raw_id = str(row["passage_id"])
            point_id: int | str = int(raw_id) if raw_id.isdigit() else raw_id
            points.append(models.PointStruct(id=point_id, vector=vector.tolist(), payload={
                "passage_id": raw_id, "text": row["text"], "source": row["source"],
                "dataset_config": row["dataset_config"], "dataset_split": row["dataset_split"]}))
        self.client.upload_points(collection_name=self.collection, points=points, batch_size=len(points), wait=True)

    def search(self, vector: Any, limit: int, ef_search: int = 64) -> list[dict[str, Any]]:
        hits = self.client.query_points(collection_name=self.collection, query=vector.tolist(), limit=limit,
            search_params=models.SearchParams(hnsw_ef=ef_search, exact=False), with_payload=True).points
        result = []
        for hit in hits:
            payload = hit.payload or {}
            result.append({"passage_id": str(payload.get("passage_id", hit.id)), "text": str(payload.get("text", "")),
                "score": float(hit.score), "source": str(payload.get("source", "")),
                "metadata": {k: payload.get(k) for k in ("dataset_config", "dataset_split")},
                "retrieval_mode": "dense"})
        return result

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def status(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection):
            return {"exists": False, "count": 0, "vector_size": None, "distance": None}
        info = self.client.get_collection(self.collection)
        params = info.config.params.vectors
        return {"exists": True, "count": int(info.points_count or 0),
                "vector_size": params.size if not isinstance(params, dict) else None,
                "distance": params.distance.value if not isinstance(params, dict) else None}

    def close(self) -> None:
        self.client.close()
