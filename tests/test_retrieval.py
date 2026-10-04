"""Small, download-free unit and local-Qdrant integration tests."""
import numpy as np
import pytest
import sys
from types import SimpleNamespace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from adrosonic_retrieval.embeddings import Embedder
from adrosonic_retrieval.retrieval import RetrievalService, CollectionUnavailable
from adrosonic_retrieval.store import PassageStore
from adrosonic_retrieval.schemas import SearchRequest
from adrosonic_retrieval.indexer import batched
from adrosonic_retrieval import indexer
from adrosonic_retrieval import evaluate as evaluation
from adrosonic_retrieval.config import Settings

class FakeEmbedder:
    dimension = 3
    def encode_query(self, text):
        return np.array([1, 0, 0], dtype=np.float32)

class FakeModel:
    def get_embedding_dimension(self): return 3
    def encode_query(self, texts, **kwargs): return np.ones((len(texts), 2), dtype=np.float32)

def make_store(tmp_path, count=8):
    store = PassageStore(tmp_path, "test_collection", vector_size=3)
    records = [{"passage_id": str(i), "text": f"passage {i}", "source": "fixture",
        "dataset_config": "test", "dataset_split": "train"} for i in range(count)]
    vectors = np.array([[1, i / 100, 0] for i in range(count)], dtype=np.float32)
    store.upsert(records, vectors)
    return store

def test_embedding_dimension_validation():
    embedder = object.__new__(Embedder)
    embedder.dimension, embedder.model = 3, FakeModel()
    with pytest.raises(ValueError, match="Embedding shape"):
        embedder.encode_query("hello")

def test_result_format_top_five_and_empty_query(tmp_path):
    store = make_store(tmp_path)
    service = RetrievalService(FakeEmbedder(), store, top_k=5)
    result = service.search("question")
    assert len(result["results"]) == 5
    assert {"passage_id", "text", "score", "source", "metadata", "retrieval_mode"} <= result["results"][0].keys()
    assert result["results"][0]["retrieval_mode"] == "dense"
    with pytest.raises(ValueError, match="empty"):
        service.search("  ")
    store.close()

def test_missing_collection_is_explicit(tmp_path):
    store = PassageStore(tmp_path, "missing")
    with pytest.raises(CollectionUnavailable):
        RetrievalService(FakeEmbedder(), store).search("hello")
    store.close()

def test_qdrant_persistence_and_schema(tmp_path):
    store = make_store(tmp_path, count=2)
    assert store.count() == 2
    store.close()
    reopened = PassageStore(tmp_path, "test_collection")
    assert reopened.count() == 2
    with pytest.raises(ValueError, match="schema mismatch"):
        reopened.ensure_collection(5)
    reopened.close()

def test_empty_and_malformed_requests():
    with pytest.raises(ValueError):
        SearchRequest(query="   ")
    with pytest.raises(ValueError):
        SearchRequest(query="hello", top_k=0)
    with pytest.raises(ValueError):
        SearchRequest(query="hello", extra_field=True)

def test_ingestion_batches_are_bounded_and_deduplicate_stable_ids():
    rows = [{"passage_id": str(i), "text": str(i)} for i in range(4)]
    rows.insert(1, {"passage_id": "0", "text": "duplicate"})
    batches = list(batched(rows, 2))
    assert all(len(batch) <= 2 for batch in batches)
    assert len([row for batch in batches for row in batch]) == 4

def test_indexer_logs_progress_for_batches_after_the_first(monkeypatch, tmp_path, caplog):
    class FakeEmbedder:
        dimension = 3
        model_revision = None
        precision = "fp32"
        device = "cpu"

        def __init__(self, *args, **kwargs): pass
        def encode_documents(self, texts, batch_size):
            return np.ones((len(texts), self.dimension), dtype=np.float32)

    class FakeStore:
        def __init__(self, *args, **kwargs): self.indexed = 0
        def count(self): return self.indexed
        def upsert(self, rows, vectors): self.indexed += len(rows)
        def close(self): pass

    rows = [{"passage_id": str(i), "text": f"passage {i}"} for i in range(300)]
    monkeypatch.setattr(indexer, "Embedder", FakeEmbedder)
    monkeypatch.setattr(indexer, "PassageStore", FakeStore)
    monkeypatch.setattr(indexer, "load_passages", lambda *args: iter(rows))
    settings = Settings(batch_size=128, data_limit=300, output_dir=tmp_path)

    with caplog.at_level("INFO", logger=indexer.__name__):
        report = indexer.run_index(settings)

    assert report["indexed_this_run"] == 300
    assert [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("Indexed ")] == [
        "Indexed 128 passages", "Indexed 256 passages", "Indexed 300 passages"]

def test_huggingface_api_key_loads_from_dotenv_without_plaintext_repr(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("HUGGINGFACE_API_KEY=fixture-token-value\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.hf_token_value == "fixture-token-value"
    assert "fixture-token-value" not in repr(settings.hf_api_key)

def test_qwen_embedding_configuration_accepts_matryoshka_sizes():
    for dimension in (256, 512, 768, 1024):
        assert Settings(embedding_dimension=dimension).embedding_dimension == dimension
    with pytest.raises(ValueError):
        Settings(embedding_precision="int8")
    assert Settings(model_revision="").model_revision is None

def test_qwen_embedder_uses_instruction_token_and_requested_dimension(monkeypatch):
    instances = []
    class FakeSentenceTransformer:
        def __init__(self, model_name, device, **kwargs):
            self.kwargs = kwargs
            self.prompts = {"query": "old prompt"}
            self.dimension = kwargs.get("truncate_dim", 1024)
            instances.append(self)
        def get_embedding_dimension(self): return self.dimension
        def modules(self): return []
        def eval(self): return self
        def encode_query(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            row = np.zeros((1, self.dimension), dtype=np.float32); row[0, 0] = 1.0
            return np.tile(row, (len(texts), 1))
        def encode_document(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            row = np.zeros((1, self.dimension), dtype=np.float32); row[0, 0] = 1.0
            return np.tile(row, (len(texts), 1))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    from adrosonic_retrieval.embeddings import Embedder
    model = Embedder("Qwen/Qwen3-Embedding-0.6B", dimension=512, precision="fp16",
        query_instruction="retrieve answer passages", token="fixture-token", revision="abc123")
    assert model.dimension == 512
    assert model.precision == "fp32"  # CPU inference safely falls back from FP16.
    assert model.model.kwargs["truncate_dim"] == 512
    assert model.model.kwargs["token"] == "fixture-token"
    assert model.model.kwargs["revision"] == "abc123"
    assert model.model.prompts["query"] == "Instruct: retrieve answer passages\nQuery:"
    assert np.allclose(model.encode_query("sample")[:2], [1.0, 0.0])
    assert model.encode_documents(["passage"]).shape == (1, 512)

def test_ragas_non_llm_context_precision_imports_when_installed():
    pytest.importorskip("ragas")
    from ragas.metrics import NonLLMContextPrecisionWithReference

    assert NonLLMContextPrecisionWithReference is not None

def test_evaluation_filters_references_to_existing_index():
    rows = [
        {"query": "first", "reference_contexts": ["Not indexed", "  Indexed\n passage "]},
        {"query": "second", "reference_contexts": ["Also absent"]},
        {"query": "third", "reference_contexts": ["indexed passage"]},
    ]

    selected = evaluation.select_evaluable_rows(rows, {"indexed passage"}, count=10)

    assert [row["query"] for row in selected] == ["first", "third"]
    assert selected[0]["reference_contexts"] == ["  Indexed\n passage "]
    assert selected[1]["reference_contexts"] == ["indexed passage"]

def test_evaluation_reference_scan_reads_qdrant_pages_and_stops_when_complete():
    class FakeClient:
        def __init__(self):
            self.calls = 0
        def scroll(self, **kwargs):
            self.calls += 1
            return ([SimpleNamespace(payload={"text": "reference"})], None)

    class FakeStore:
        collection = "test"
        client = FakeClient()

    rows = [{"reference_contexts": ["Reference"]}]
    store = FakeStore()

    assert evaluation.indexed_reference_contexts(store, rows) == {"reference"}
    assert store.client.calls == 1
