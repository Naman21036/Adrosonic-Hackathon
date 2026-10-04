import json
import logging as std_logging
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from app.config import settings
from app.observability.context import get_request_id, new_request_id, request_id_context
from app.observability.logging import configure_logging, log_event


@pytest.fixture
def structured_log_file():
    log_dir=Path.cwd()/".test-tmp"/("obs-log-"+uuid.uuid4().hex)
    log_dir.mkdir(parents=True,exist_ok=False)
    path=log_dir/"observability.jsonl"
    configure_logging(force=True,log_file=str(path),console=False)
    yield path
    logger.complete()
    configure_logging(force=True,log_file=settings.log_file,console=False)


def _records(path):
    logger.complete()
    for _ in range(10):
        if path.exists():
            try:
                return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            except OSError:
                pass
        time.sleep(0.01)
    return []


def test_request_id_generation_and_context(structured_log_file):
    request_id=new_request_id("client-request_123")
    assert request_id=="client-request_123"
    generated=new_request_id("bad\nrequest")
    assert str(uuid.UUID(generated))==generated
    with request_id_context(request_id):
        assert get_request_id()==request_id
        log_event("REQUEST_CONTEXT_TEST",retrieval_mode="hybrid")
    record=_records(structured_log_file)[-1]
    assert record["record"]["extra"]["request_id"]==request_id
    assert record["record"]["extra"]["fields"]["retrieval_mode"]=="hybrid"
    assert get_request_id() is None


def test_secrets_are_removed_from_structured_logs(structured_log_file,monkeypatch):
    secret="azure-test-secret-value-8765"
    monkeypatch.setattr(settings,"azure_openai_api_key",secret)
    log_event("SECRET_REDACTION_TEST",api_key=secret,detail=f"request used {secret}")
    content=structured_log_file.read_text(encoding="utf-8")
    assert secret not in content
    record=_records(structured_log_file)[-1]
    fields=record["record"]["extra"]["fields"]
    assert "api_key" not in fields
    assert "[REDACTED]" in fields["detail"]


def test_standard_logging_bridge_uses_the_structured_loguru_sink(structured_log_file):
    std_logging.getLogger("observability.fixture").warning("bridge message")
    record=_records(structured_log_file)[-1]["record"]
    assert record["message"]=="bridge message"
    assert record["extra"]["event"]=="STDLIB_LOG"
    assert record["extra"]["fields"]["source_logger"]=="observability.fixture"


def test_langfuse_disabled_and_missing_credentials_are_noop(structured_log_file,monkeypatch):
    import app.observability.langfuse as tracing

    monkeypatch.setattr(settings,"langfuse_enabled",False)
    with tracing.trace("disabled-test",input={"query":"private"}) as root:
        with tracing.observation("child-test") as child:
            child.update(output={"ok":True})
        root.update(output={"ok":True})

    monkeypatch.setattr(settings,"langfuse_enabled",True)
    monkeypatch.setattr(settings,"langfuse_public_key",None)
    monkeypatch.setattr(settings,"langfuse_secret_key",None)
    assert tracing.get_langfuse_client() is None
    with tracing.trace("missing-credentials-test") as root:
        root.update(output={"ok":True})
    records=_records(structured_log_file)
    assert any(r["record"]["extra"].get("event")=="LANGFUSE_DISABLED_MISSING_CREDENTIALS" for r in records)


def test_langfuse_initialization_failure_does_not_raise(structured_log_file,monkeypatch):
    import langfuse
    import app.observability.langfuse as tracing

    monkeypatch.setattr(settings,"langfuse_enabled",True)
    monkeypatch.setattr(settings,"langfuse_public_key","test-public-key")
    monkeypatch.setattr(settings,"langfuse_secret_key","test-secret-key")
    monkeypatch.setattr(tracing,"_client",None)
    monkeypatch.setattr(tracing,"_client_warning_emitted",False)
    def unavailable(**kwargs):
        raise OSError("test endpoint unavailable")
    monkeypatch.setattr(langfuse,"Langfuse",unavailable)
    with tracing.trace("unavailable-test") as root:
        root.update(output={"application_continued":True})
    records=_records(structured_log_file)
    assert any(r["record"]["extra"].get("event")=="LANGFUSE_INITIALIZATION_FAILED" for r in records)
    content=structured_log_file.read_text(encoding="utf-8")
    assert "test-secret-key" not in content


def test_langfuse_child_spans_and_generation_are_nested(structured_log_file,monkeypatch):
    import app.observability.langfuse as tracing

    class FakeObservation:
        def __init__(self,name,kind,parent=None):
            self.name=name
            self.kind=kind
            self.parent=parent
            self.trace_id="test-trace"
            self.id="test-"+name
            self.children=[]
            self.fields={}
        def span(self,**fields):
            child=FakeObservation(fields["name"],"span",self)
            self.children.append((child,fields))
            return child
        def generation(self,**fields):
            child=FakeObservation(fields["name"],"generation",self)
            self.children.append((child,fields))
            return child
        def update(self,**fields):
            self.fields.update(fields)
        def end(self,**fields):
            self.fields.update(fields)
    class FakeClient:
        def trace(self,**fields):
            self.root=FakeObservation(fields["name"],"trace")
            self.root.fields.update(fields)
            return self.root
    fake=FakeClient()
    monkeypatch.setattr(settings,"langfuse_enabled",True)
    monkeypatch.setattr(tracing,"_client",fake)
    with tracing.trace("rag_request",input={"query":"sample"}) as root:
        with tracing.observation("dense_retrieval") as dense:
            dense.update(output={"document_ids":["p1"]})
        with tracing.observation("azure_llm_generation",kind="generation",model="gpt-4.1") as generation:
            generation.update(output="answer",usage_details={"input":3,"output":1})
    assert root.trace_id=="test-trace"
    assert [child.name for child,_ in fake.root.children]==["dense_retrieval","azure_llm_generation"]
    assert fake.root.children[0][0].fields["output"]=={"document_ids":["p1"]}
    assert fake.root.children[1][0].kind=="generation"
    assert fake.root.children[1][0].fields["usage_details"]=={"input":3,"output":1}


def test_api_request_id_propagates_and_disabled_langfuse_keeps_response(structured_log_file,monkeypatch):
    import app.api as api

    class FakeIndex:
        collection="test-collection"

    result={
        "query":"known query",
        "mode":"hybrid",
        "filters":{},
        "latency_ms":1.25,
        "results":[{"id":"p-1","text":"evidence","metadata":{},"score":0.5,"retrieval_mode":"hybrid","score_type":"ranking_score"}],
    }
    observed=[]
    monkeypatch.setattr(api,"index",FakeIndex())
    monkeypatch.setattr(api,"model_ready",True)
    monkeypatch.setattr(settings,"langfuse_enabled",False)
    monkeypatch.setattr(api,"search",lambda *args:(observed.append(get_request_id()) or result))
    client=TestClient(api.app,raise_server_exceptions=False)
    response=client.post("/search",headers={"x-request-id":"regression-request-1"},json={"query":"known query","mode":"hybrid","top_k":5})
    assert response.status_code==200
    assert response.headers["x-request-id"]=="regression-request-1"
    assert observed==["regression-request-1"]
    assert response.json()==result
    records=_records(structured_log_file)
    ids={r["record"]["extra"].get("request_id") for r in records}
    assert "regression-request-1" in ids


def test_azure_failure_is_logged_without_changing_safe_error(structured_log_file,monkeypatch):
    import asyncio
    import app.azure_openai as azure

    monkeypatch.setattr(settings,"langfuse_enabled",False)
    def fail_client():
        raise azure.AzureOpenAIError("Azure OpenAI configuration unavailable")
    monkeypatch.setattr(azure,"_client",fail_client)
    with pytest.raises(azure.AzureOpenAIError,match="configuration unavailable"):
        asyncio.run(azure.grounded_answer("query",[{"id":"p1","text":"context"}]))
    events=[r["record"]["extra"].get("event") for r in _records(structured_log_file)]
    assert "LLM_REQUEST_STARTED" in events
    assert "LLM_REQUEST_FAILED" in events


def test_retrieval_failure_is_logged(structured_log_file,monkeypatch):
    from app.retrieval import search

    class BrokenIndex:
        collection="test-collection"
        def dense_search(self,*args,**kwargs):
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(settings,"dense_candidate_k",5)
    with request_id_context("retrieval-failure-request"):
        with pytest.raises(RuntimeError,match="fixture failure"):
            search(BrokenIndex(),"query","hybrid",5)
    records=_records(structured_log_file)
    events=[r["record"]["extra"].get("event") for r in records]
    assert "DENSE_RETRIEVAL_STARTED" in events
    assert "DENSE_RETRIEVAL_FAILED" in events
    assert all(r["record"]["extra"].get("request_id")=="retrieval-failure-request" for r in records)


def test_instrumented_search_preserves_dense_sparse_rrf_order(structured_log_file,monkeypatch):
    from app.retrieval import reciprocal_rank_fusion, search

    def hit(pid,score):
        return {"id":pid,"text":pid,"metadata":{},"score":score}

    dense=[hit("d1",.9),hit("both",.8),hit("d2",.7),hit("d3",.6)]
    sparse=[hit("s1",1.3),hit("both",1.1),hit("s2",1.0),hit("d2",.8)]
    class FakeIndex:
        collection="test-collection"
        def dense_search(self,query,limit,filters): return dense[:limit]
        def sparse_search(self,query,limit,filters): return sparse[:limit]

    monkeypatch.setattr(settings,"dense_candidate_k",4)
    monkeypatch.setattr(settings,"sparse_candidate_k",4)
    monkeypatch.setattr(settings,"rrf_k",60)
    scores={}
    for rows in (dense,sparse):
        seen=set()
        for rank,item in enumerate(rows,1):
            if item["id"] not in seen:
                scores[item["id"]]=scores.get(item["id"],0)+1/(60+rank)
                seen.add(item["id"])
    expected_ids=[pid for pid,_ in sorted(scores.items(),key=lambda item:(-item[1],item[0]))[:5]]
    result=search(FakeIndex(),"query","hybrid",5)
    assert result["mode"]=="hybrid"
    assert len(result["results"])==5
    assert [item["id"] for item in result["results"]]==expected_ids
    assert [item["id"] for item in result["results"]]==[item["id"] for item in reciprocal_rank_fusion([dense,sparse],60)[:5]]
