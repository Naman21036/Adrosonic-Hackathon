from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import SearchRequest
from app.config import settings
from app.retrieval import search
from app.storage import HybridIndex, SparseStore, source_id_from_passage_id, validate_metadata


def _hit(passage_id: str, source_id: int, score: float = 1.0) -> dict:
    return {
        "id": passage_id,
        "text": passage_id,
        "metadata": {},
        "source_id": source_id,
        "score": score,
    }


def test_source_id_is_parsed_only_from_canonical_ids():
    assert source_id_from_passage_id("msmarco-23013-4") == 23013
    assert source_id_from_passage_id("msmarco-0-0") == 0
    assert source_id_from_passage_id("msmarco-23x-4") is None
    assert source_id_from_passage_id("ordinary-id") is None


def test_source_id_filter_requires_nonnegative_json_integer():
    assert SearchRequest(query="example", filter={"source_id": 23013}).filter == {"source_id": 23013}
    assert SearchRequest(query="example", filter={}).filter == {}
    with pytest.raises(ValueError, match="source_id"):
        SearchRequest(query="example", filter={"source_id": "23013"})
    with pytest.raises(ValueError, match="source_id"):
        validate_metadata({"source_id": True})
    with pytest.raises(ValueError, match="source_id"):
        validate_metadata({"source_id": -1})
    with pytest.raises(ValueError, match="source_id only"):
        SearchRequest(query="example", filter={"source": "ms_marco"})


def test_sparse_source_filter_applies_before_limit_and_uses_expression_index():
    store = SparseStore(":memory:")
    try:
        rows = [
            (f"msmarco-{source}-0", "blue whale ocean mammal", {"source": "test"})
            for source in range(1000, 1010)
        ]
        rows.append(("msmarco-23013-0", "blue whale ocean mammal", {"source": "test"}))
        rows.append(("msmarco-23013-1", "blue whale ocean mammal", {"source": "test"}))
        rows.append(("msmarco-77777-0", "blue whale ocean mammal", {"source": "test"}))
        rows.append(("ordinary-id", "blue whale ocean mammal", {"source": "test"}))
        rows.append(("catalog-item-1", "blue whale ocean mammal", {"source_id": "demo-catalog"}))
        store.upsert_many(rows)

        unfiltered = store.search("blue whale", 1)
        filtered = store.search("blue whale", 1, {"source_id": 23013})
        assert unfiltered[0]["source_id"] != 23013
        assert len(filtered) == 1
        assert filtered[0]["id"] == "msmarco-23013-0"
        assert filtered[0]["source_id"] == 23013
        single_passage = store.search("blue whale", 5, {"source_id": 77777})
        assert len(single_passage) == 1
        assert single_passage[0]["source_id"] == 77777
        assert store.search("blue whale", 10, {"source_id": 987654321}) == []
        assert store.search("blue whale", 10, {}) == store.search("blue whale", 10)
        custom_source = store.search("blue whale", 5, {"source_id": "demo-catalog"})
        assert [row["id"] for row in custom_source] == ["catalog-item-1"]

        plan = store.db.execute(
            "EXPLAIN QUERY PLAN SELECT p.id FROM passage p "
            "WHERE p.id LIKE ? AND CAST(substr(p.id, 9, instr(substr(p.id, 9), '-') - 1) AS INTEGER)=?",
            ("msmarco-23013-%", 23013),
        ).fetchall()
        assert any("idx_passage_source_id" in row[3] for row in plan)
    finally:
        store.close()


def test_qdrant_filter_is_passed_into_similarity_query():
    class Embedder:
        dimension = 2

        def encode_query(self, query):
            return np.asarray([0.25, 0.75], dtype=np.float32)

    class FakeQdrant:
        query_filter = None

        def query_points(self, **kwargs):
            self.query_filter = kwargs["query_filter"]
            return SimpleNamespace(points=[
                SimpleNamespace(
                    payload={
                        "passage_id": "msmarco-23013-0",
                        "text": "matched passage",
                        "metadata": {},
                        "source_id": 23013,
                    },
                    score=0.9,
                )
            ])

    index = object.__new__(HybridIndex)
    index.embedder = Embedder()
    index.collection = "test"
    index.qdrant = FakeQdrant()
    hits = index.dense_search("sample query", 5, {"source_id": 23013})
    condition = index.qdrant.query_filter.must[0]
    assert condition.key == "source_id"
    assert condition.match.value == 23013
    assert hits[0]["source_id"] == 23013


def test_hybrid_passes_filter_to_both_candidate_retrievers_and_never_falls_back():
    class FakeIndex:
        collection = "test"

        def __init__(self):
            self.filters = []

        def dense_search(self, query, limit, filters):
            self.filters.append(("dense", filters))
            return [_hit("msmarco-23013-0", 23013), _hit("msmarco-23013-1", 23013, 0.8)]

        def sparse_search(self, query, limit, filters):
            self.filters.append(("sparse", filters))
            return [_hit("msmarco-23013-1", 23013), _hit("msmarco-23013-2", 23013, 0.8)]

    index = FakeIndex()
    result = search(index, "sample query", "hybrid", 5, {"source_id": 23013})
    assert [kind for kind, _ in index.filters] == ["dense", "sparse"]
    assert all(filters == {"source_id": 23013} for _, filters in index.filters)
    assert result["filter"] == {"source_id": 23013}
    assert result["top_k"] == 5
    assert result["results"]
    assert all(hit["source_id"] == 23013 for hit in result["results"])

    class EmptyIndex(FakeIndex):
        def dense_search(self, query, limit, filters):
            self.filters.append(("dense", filters))
            return []

        def sparse_search(self, query, limit, filters):
            self.filters.append(("sparse", filters))
            return []

    missing = search(EmptyIndex(), "sample query", "hybrid", 5, {"source_id": 987654321})
    assert missing["results"] == []


def test_api_filter_aliases_response_and_observability(monkeypatch):
    import app.api as api

    monkeypatch.setattr(api.settings, "langfuse_enabled", False)
    monkeypatch.setattr(api, "index", SimpleNamespace())
    monkeypatch.setattr(api, "model_ready", True)
    observed = {"search": None, "trace": None, "events": []}

    @contextmanager
    def capture_trace(name, **kwargs):
        observed["trace"] = (name, kwargs)
        yield SimpleNamespace(trace_id="test-trace", update=lambda **values: None)

    def capture_search(index, query, mode, top_k, filters):
        observed["search"] = filters
        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "filter": filters,
            "filters": filters,
            "latency_ms": 1.0,
            "results": [_hit("msmarco-23013-0", 23013)],
        }

    monkeypatch.setattr(api, "trace", capture_trace)
    monkeypatch.setattr(api, "search", capture_search)
    monkeypatch.setattr(api, "log_event", lambda event, **fields: observed["events"].append((event, fields)))
    client = TestClient(api.app, raise_server_exceptions=False)

    response = client.post("/search", json={"query": "sample", "mode": "dense", "top_k": 5, "filter": {"source_id": 23013}})
    assert response.status_code == 200
    assert observed["search"] == {"source_id": 23013}
    assert response.json()["filter"] == {"source_id": 23013}
    assert response.json()["results"][0]["source_id"] == 23013
    trace_name, trace_kwargs = observed["trace"]
    assert trace_name == "rag_request"
    assert trace_kwargs["input"]["filter"] == {"source_id": 23013}
    assert trace_kwargs["metadata"]["filter_value"] == 23013
    started = next(fields for event, fields in observed["events"] if event == "RAG_REQUEST_STARTED")
    assert started["filter_enabled"] is True
    assert started["filter_dimension"] == "source_id"
    assert started["filter_value"] == 23013

    response = client.post("/search", json={"query": "sample", "filters": {"source": "ms_marco"}})
    assert response.status_code == 200
    assert observed["search"] == {"source": "ms_marco"}
    response = client.post("/search", json={"query": "sample"})
    assert response.status_code == 200
    assert observed["search"] == {}
    assert client.post("/search", json={"query": "sample", "filter": {"source_id": "23013"}}).status_code == 422
