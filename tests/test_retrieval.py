import pytest
from pathlib import Path
from uuid import uuid4

from app.embedding import HashEmbedder
from app.retrieval import reciprocal_rank_fusion
from app.storage import HybridIndex, PassageNotFoundError


def fresh_local_store_path() -> Path:
    root = Path.cwd() / ".test-tmp"
    root.mkdir(exist_ok=True)
    path = root / f"retrieval-{uuid4().hex}"
    path.mkdir()
    return path

def hit(pid,score=1): return {"id":pid,"text":pid,"metadata":{},"score":score}

def test_rrf_one_based_ranks_and_tie_break():
    result=reciprocal_rank_fusion([[hit("b"),hit("a")],[hit("a"),hit("b")]],k=0)
    assert [x["id"] for x in result]==["a","b"]
    assert result[0]["score"]==pytest.approx(1+1/2)

def test_sparse_metadata_precedes_ranking_and_mutations():
    pytest.importorskip("qdrant_client")
    from app.config import settings
    tmp_path=fresh_local_store_path()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings,"journal_path",str(tmp_path/"journal.jsonl"))
        idx=HybridIndex(HashEmbedder(),str(tmp_path/"qdrant"),"test",str(tmp_path/"sparse.sqlite3"))
        try:
            idx.upsert("one","blue whale swims in the ocean",{"source":"a"})
            idx.upsert("two","blue whale found in the sea",{"source":"b"})
            assert {h["id"] for h in idx.sparse_search("blue whale",10,{"source":"a"})}=={"one"}
            assert idx.dense_search("blue whale",10,{"source":"b"})[0]["id"]=="two"
            idx.delete("one")
            with pytest.raises(PassageNotFoundError):
                idx.delete("one")
            assert all(h["id"]!="one" for h in idx.sparse_search("blue whale",10))
        finally:
            idx.close()

def test_dense_sparse_persistence_across_reopen():
    pytest.importorskip("qdrant_client")
    from app.config import settings
    tmp_path=fresh_local_store_path()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings,"journal_path",str(tmp_path/"journal.jsonl"))
        args=(HashEmbedder(),str(tmp_path/"qdrant"),"persist",str(tmp_path/"sparse.sqlite3"))
        idx=HybridIndex(*args)
        idx.upsert("persisted","water boils at one hundred degrees",{"source":"test"})
        idx.close()
        idx=HybridIndex(*args)
        try:
            assert idx.count()==1
            assert idx.sparse.count()==1
            assert idx.dense_search("water boils",1)[0]["id"]=="persisted"
        finally:
            idx.close()

def test_invalid_metadata_field_rejected():
    from app.storage import validate_metadata
    with pytest.raises(ValueError): validate_metadata({"source.foo":"x"})

def test_sparse_store_exact_metadata_filter_and_upsert():
    from app.storage import SparseStore
    store=SparseStore(":memory:")
    store.upsert("one","blue whale swims",{"source":"first"})
    store.upsert("two","blue whale swims",{"source":"second"})
    assert [x["id"] for x in store.search("whale",10,{"source":"second"})]==["two"]
    store.upsert("two","orcas swim in cold water",{"source":"second"})
    assert store.search("whale",10,{"source":"second"})==[]
    store.delete("one")
    assert store.count()==1
    store.close()

def test_optional_mistral_is_disabled_without_free_tier_confirmation(monkeypatch):
    import asyncio
    from app.config import settings
    from app.mistral import MistralUnavailable,grounded_answer
    monkeypatch.setitem(settings.__dict__,"mistral_api_key","example-key")
    monkeypatch.setattr(settings,"mistral_free_tier_confirmed",False)
    with pytest.raises(MistralUnavailable): asyncio.run(grounded_answer("question",[]))

def test_api_search_request_bounds_and_defaults():
    from pydantic import ValidationError
    from app.api import SearchRequest
    assert SearchRequest(query="test").top_k==5
    with pytest.raises(ValidationError): SearchRequest(query="test",top_k=21)

def test_mistral_mock_requires_valid_passage_citation(monkeypatch):
    import asyncio
    from app.config import settings
    import app.mistral as mistral
    monkeypatch.setitem(settings.__dict__,"mistral_api_key","test-secret")
    monkeypatch.setitem(settings.__dict__,"mistral_model","mistral-small-latest")
    monkeypatch.setattr(settings,"mistral_free_tier_confirmed",True)
    class Response:
        status_code=200
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":"Grounded claim [p1]."}}]}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return None
        async def post(self,*args,**kwargs): return Response()
    monkeypatch.setattr(mistral.httpx,"AsyncClient",lambda **kwargs:Client())
    result=asyncio.run(mistral.grounded_answer("q",[{"id":"p1","text":"evidence"}]))
    assert result["supporting_passage_ids"]==["p1"]

def test_ms_marco_loader_uses_streaming_parquet_for_current_hub_format(monkeypatch):
    import app.datasets as loader
    from app.config import settings
    calls=[]
    monkeypatch.setattr(loader,"load_dataset",lambda *args,**kwargs:(calls.append((args,kwargs)) or "stream"))
    monkeypatch.setattr(settings,"dataset_name","microsoft/ms_marco")
    monkeypatch.setattr(settings,"dataset_config","v1.1")
    assert loader.load_split("validation")=="stream"
    assert calls[0][0]==("parquet",)
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["data_files"]["validation"].endswith("v1.1/validation-00000-of-00001.parquet")

def test_api_returns_unavailable_for_mutations_without_an_index(monkeypatch):
    from fastapi.testclient import TestClient
    import app.api as api
    monkeypatch.setattr(api,"index",None)
    monkeypatch.setattr(api,"model_ready",False)
    def unavailable_embedder(*args,**kwargs):
        raise RuntimeError("test startup intentionally unavailable")
    monkeypatch.setattr(api,"QwenEmbedder",unavailable_embedder)
    with TestClient(api.app,raise_server_exceptions=False) as client:
        assert client.get("/health").json()["status"]=="degraded"
        assert client.post("/search",json={"query":"hello"}).status_code==503
        assert client.post("/upsert",json={"passage_id":"x","text":"hello","source_id":"test"}).status_code==503
        assert client.delete("/passages/x").status_code==503

def test_embedding_device_auto_and_cpu_fallback():
    from types import SimpleNamespace
    from app.embedding import resolve_device
    cpu_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    cuda_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))
    assert resolve_device(cpu_torch,"auto")=="cpu"
    assert resolve_device(cuda_torch,"auto")=="cuda"
    with pytest.raises(RuntimeError,match="cannot access CUDA"):
        resolve_device(cpu_torch,"cuda")

def test_embedding_reduces_batch_and_retries_after_cuda_oom(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from app.embedding import QwenEmbedder

    if not torch.cuda.is_available():
        pytest.skip("the CUDA OOM simulation requires a CUDA-enabled torch runtime")

    class FakeBatch(dict):
        def to(self,device):
            return FakeBatch({key:value.to(device) for key,value in self.items()})
    class FakeTokenizer:
        def __call__(self,texts,**kwargs):
            if len(texts)>2: raise torch.cuda.OutOfMemoryError("simulated OOM")
            return FakeBatch({"attention_mask":torch.ones((len(texts),2),dtype=torch.long)})
    class FakeModel:
        def __call__(self,**batch):
            return SimpleNamespace(last_hidden_state=torch.ones((batch["attention_mask"].shape[0],2,2),device=batch["attention_mask"].device))

    embedder=object.__new__(QwenEmbedder)
    embedder.torch=torch
    embedder.tokenizer=FakeTokenizer()
    embedder.model=FakeModel()
    embedder.device="cuda"
    embedder.dimension=2
    embedder.max_seq_length=8
    embedder.batch_size=4
    vectors=embedder.encode_documents(["one","two","three","four"])
    assert vectors.shape==(4,2)
    assert embedder.batch_size==2
    assert np.isfinite(vectors).all()

def test_gpu_embedding_configuration_aliases_and_bounds(monkeypatch):
    from pydantic import ValidationError
    from app.config import Settings
    for key,value in {
        "EMBEDDING_MODEL":"Qwen/Qwen3-Embedding-0.6B",
        "EMBEDDING_DEVICE":"auto",
        "EMBEDDING_BATCH_SIZE":"16",
        "EMBEDDING_MAX_SEQ_LENGTH":"256",
        "EMBEDDING_PRECISION":"float16",
        "QDRANT_UPSERT_BATCH_SIZE":"128",
    }.items():
        monkeypatch.setenv(key,value)
    config=Settings(_env_file=None)
    assert config.hf_model_name=="Qwen/Qwen3-Embedding-0.6B"
    assert config.device=="auto"
    assert config.embedding_batch_size==16
    assert config.embedding_max_seq_length==256
    assert config.embedding_precision=="float16"
    assert config.qdrant_upsert_batch_size==128
    with pytest.raises(ValidationError):
        Settings(_env_file=None,embedding_batch_size=0)

def test_ingestion_resume_targets_total_count_and_skips_existing(monkeypatch):
    import sys
    import json
    from pathlib import Path
    import app.ingest as ingest

    class FakeEncoder:
        device="cpu"; gpu_name=None; effective_precision="float32"; batch_size=2; dimension=1024

    class FakeSparse:
        def __init__(self,ids): self.ids=set(ids)
        def count(self): return len(self.ids)
        def contains(self,pid): return pid in self.ids

    class FakeIndex:
        def __init__(self):
            self.ids={"msmarco-existing-0"}
            self.sparse=FakeSparse(self.ids)
            self.last_upsert_timings={"embedding_seconds":0.01,"sparse_seconds":0.01,"qdrant_seconds":0.01}
        def count(self): return len(self.ids)
        def upsert_many(self,rows):
            self.ids.update(row[0] for row in rows)
            self.sparse.ids.update(row[0] for row in rows)
        def close(self): pass

    rows=[
        {"query_id":"existing","passages":{"passage_text":["already indexed","new one"],"is_selected":[False,True]}},
        {"query_id":"second","passages":{"passage_text":["new two"],"is_selected":[False]}},
        {"query_id":"third","passages":{"passage_text":["must not exceed target"],"is_selected":[False]}},
    ]
    fake_index=FakeIndex()
    temp_root=Path.cwd()/".test-tmp"
    temp_root.mkdir(exist_ok=True)
    report_path=temp_root/"ingest_resume_test.json"
    monkeypatch.setattr(ingest,"REPORT_PATH",report_path)
    monkeypatch.setattr(sys,"argv",["app.ingest","--limit","3"])
    monkeypatch.setattr(ingest,"load_split",lambda split:rows)
    monkeypatch.setattr(ingest,"QwenEmbedder",lambda *args,**kwargs:FakeEncoder())
    monkeypatch.setattr(ingest,"HybridIndex",lambda *args,**kwargs:fake_index)
    ingest.main()
    assert fake_index.count()==3
    assert fake_index.ids=={"msmarco-existing-0","msmarco-existing-1","msmarco-second-0"}
    report=json.loads(report_path.read_text())
    assert report["status"]=="complete"
    assert report["existing_passages_skipped_on_resume"]==1
    assert report["passages_indexed_this_run"]==2
