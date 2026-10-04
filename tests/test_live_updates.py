from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.embedding import HashEmbedder
from app.retrieval import search
from app.storage import HybridIndex, PassageNotFoundError, _point_id


@pytest.fixture
def live_index(monkeypatch):
    pytest.importorskip("qdrant_client")
    from app.config import settings

    root = Path.cwd() / ".test-tmp"
    root.mkdir(exist_ok=True)
    path = root / f"fr5-live-{uuid4().hex}"
    path.mkdir()
    monkeypatch.setattr(settings, "journal_path", str(path / "journal.jsonl"))
    index = HybridIndex(HashEmbedder(), str(path / "qdrant"), "live-updates-test", str(path / "sparse.sqlite3"))
    try:
        yield index, path
    finally:
        index.close()


def test_live_upsert_update_retrieval_and_delete_api(live_index, monkeypatch):
    from fastapi.testclient import TestClient
    import app.api as api

    index, _ = live_index
    monkeypatch.setattr(api, "index", index)
    monkeypatch.setattr(api, "model_ready", True)
    monkeypatch.setattr(api.settings, "langfuse_enabled", True)
    monkeypatch.setattr(api.settings, "langfuse_public_key", None)
    monkeypatch.setattr(api.settings, "langfuse_secret_key", None)
    import app.observability.langfuse as tracing
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_warning_emitted", False)
    client = TestClient(api.app, raise_server_exceptions=False)

    invalid = client.post("/upsert", json={"id": "missing-source", "text": "custom IDs require a source"})
    assert invalid.status_code == 422
    created = client.post(
        "/upsert",
        json={
            "id": "demo-live-001",
            "text": "Adrosonic demonstrates incremental live index updates without full corpus reindexing.",
            "source_id": "demo",
        },
    )
    assert created.status_code == 200, created.text
    first = created.json()
    assert first["success"] is True
    assert first["status"] == "created"
    assert first["embedding_count_delta"] == 1
    assert first["embedding_dimension"] == index.embedder.dimension
    assert first["indexed_passages_before"] == 0
    assert first["indexed_passages_after"] == 1
    assert first["source_id"] == "demo"

    # Validate each retrieval path against the disposable test collection.
    query_v1 = "How does Adrosonic demonstrate live index updates?"
    assert index.dense_search(query_v1, 5)[0]["id"] == "demo-live-001"
    assert index.sparse_search(query_v1, 5)[0]["id"] == "demo-live-001"
    assert "demo-live-001" in [row["id"] for row in search(index, query_v1, "hybrid", 5)["results"]]

    updated = client.post(
        "/upsert",
        json={
            "passage_id": "demo-live-001",
            "text": "Adrosonic supports incremental vector database updates without rebuilding the corpus.",
            "source_id": "demo",
        },
    )
    assert updated.status_code == 200, updated.text
    second = updated.json()
    assert second["status"] == "updated"
    assert second["embedding_count_delta"] == 1
    assert second["indexed_passages_before"] == second["indexed_passages_after"] == 1
    assert index.count() == index.sparse.count() == 1
    assert index.sparse_search("reindexing", 5) == []
    query_v2 = "How does Adrosonic support vector database updates?"
    expected_v2 = "Adrosonic supports incremental vector database updates without rebuilding the corpus."
    assert index.dense_search(query_v2, 5)[0]["text"] == expected_v2
    assert index.sparse_search(query_v2, 5)[0]["text"] == expected_v2
    assert "demo-live-001" in [row["id"] for row in client.post("/search", json={"query": query_v2, "mode": "hybrid"}).json()["results"]]

    point = index.qdrant.retrieve(collection_name=index.collection, ids=[_point_id("demo-live-001")], with_payload=True)[0]
    assert point.payload["text"] == expected_v2

    deleted = client.delete("/passages/demo-live-001")
    assert deleted.status_code == 200, deleted.text
    result = deleted.json()
    assert result["success"] is True
    assert result["deleted_from_qdrant"] is True
    assert result["deleted_from_bm25"] is True
    assert result["embedding_count_delta"] == 0
    assert result["indexed_passages_before"] == 1
    assert result["indexed_passages_after"] == 0
    assert index.count() == index.sparse.count() == 0
    assert index.qdrant.retrieve(collection_name=index.collection, ids=[_point_id("demo-live-001")], with_payload=False) == []
    assert index.sparse.contains("demo-live-001") is False
    for mode in ("dense", "hybrid"):
        response = client.post("/search", json={"query": query_v2, "mode": mode})
        assert response.status_code == 200
        assert all(row["id"] != "demo-live-001" for row in response.json()["results"])

    missing = client.delete("/passages/not-present")
    assert missing.status_code == 404


def test_live_upsert_request_validation():
    from pydantic import ValidationError
    from app.api import PassageRequest

    assert PassageRequest(id="custom", text="valid", source_id="catalog").passage_id == "custom"
    assert PassageRequest(id="msmarco-23013-4", text="valid").passage_id == "msmarco-23013-4"
    assert PassageRequest(id="msmarco-23013-4", text="valid", source_id="23013").storage_metadata()["source_id"] == 23013
    with pytest.raises(ValidationError, match="source_id is required"):
        PassageRequest(id="custom", text="valid")
    with pytest.raises(ValidationError):
        PassageRequest(id="custom", text="   ", source_id="catalog")
    with pytest.raises(ValidationError, match="same passage"):
        PassageRequest(id="one", passage_id="two", text="valid", source_id="catalog")
    with pytest.raises(ValidationError, match="numeric source ID"):
        PassageRequest(id="msmarco-23013-4", text="valid", source_id="unrelated")


def test_qdrant_failure_does_not_update_bm25_and_journal_recovers(live_index, monkeypatch):
    index, path = live_index
    original_upsert = index.qdrant.upsert

    def fail_qdrant(*args, **kwargs):
        raise OSError("simulated Qdrant write failure")

    monkeypatch.setattr(index.qdrant, "upsert", fail_qdrant)
    with pytest.raises(OSError, match="simulated"):
        index.upsert("pending-qdrant", "journal replay passage", {"source_id": "test"})
    assert index.sparse.contains("pending-qdrant") is False
    assert index.pending_count() == 1
    pending = [json.loads(line) for line in (path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["op"] for row in pending] == ["upsert"]
    assert pending[0]["state"] == "pending"

    monkeypatch.setattr(index.qdrant, "upsert", original_upsert)
    index.recover_pending()
    assert index.count() == index.sparse.count() == 1
    assert index.pending_count() == 0


def test_bm25_failure_leaves_recoverable_qdrant_first_write(live_index, monkeypatch):
    index, _ = live_index
    original_sparse_upsert = index.sparse.upsert_many

    def fail_sparse(*args, **kwargs):
        raise OSError("simulated BM25 write failure")

    monkeypatch.setattr(index.sparse, "upsert_many", fail_sparse)
    with pytest.raises(OSError, match="simulated"):
        index.upsert("pending-bm25", "durable qdrant passage", {"source_id": "test"})
    assert index.qdrant.retrieve(collection_name=index.collection, ids=[_point_id("pending-bm25")], with_payload=True)[0].payload["text"] == "durable qdrant passage"
    assert index.sparse.contains("pending-bm25") is False
    assert index.pending_count() == 1

    monkeypatch.setattr(index.sparse, "upsert_many", original_sparse_upsert)
    index.recover_pending()
    assert index.count() == index.sparse.count() == 1
    assert index.sparse.search("durable qdrant passage", 1)[0]["id"] == "pending-bm25"
    assert index.pending_count() == 0


def test_delete_failure_is_replayed_to_both_indexes(live_index, monkeypatch):
    index, _ = live_index
    index.upsert("pending-delete", "passage to remove safely", {"source_id": "test"})
    original_delete = index.sparse.delete

    def fail_sparse_delete(*args, **kwargs):
        raise OSError("simulated BM25 delete failure")

    monkeypatch.setattr(index.sparse, "delete", fail_sparse_delete)
    with pytest.raises(OSError, match="simulated"):
        index.delete("pending-delete")
    assert index.qdrant.retrieve(collection_name=index.collection, ids=[_point_id("pending-delete")], with_payload=False) == []
    assert index.sparse.contains("pending-delete") is True
    assert index.pending_count() == 1

    monkeypatch.setattr(index.sparse, "delete", original_delete)
    index.recover_pending()
    assert index.count() == index.sparse.count() == 0
    assert index.pending_count() == 0


def test_delete_returns_not_found_without_appending_journal(live_index):
    index, path = live_index
    with pytest.raises(PassageNotFoundError):
        index.delete("missing")
    assert index.pending_count() == 0
    assert not (path / "journal.jsonl").exists()
