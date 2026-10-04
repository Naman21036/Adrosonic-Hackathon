import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from adrosonic_retrieval.api import app
from adrosonic_retrieval.retrieval import RetrievalService
from adrosonic_retrieval.store import PassageStore

class FakeEmbedder:
    def encode_query(self, text): return np.array([1, 0, 0], dtype=np.float32)

@pytest.mark.asyncio
async def test_search_health_and_status_api(tmp_path):
    store=PassageStore(tmp_path,"api_test",vector_size=3)
    records=[{"passage_id":str(i),"text":f"text {i}","source":"fixture","dataset_config":"x","dataset_split":"test"} for i in range(6)]
    store.upsert(records,np.array([[1,i/10,0] for i in range(6)],dtype=np.float32))
    app.state.store=store; app.state.embedder=FakeEmbedder()
    app.state.service=RetrievalService(FakeEmbedder(),store,top_k=5)
    try:
        async with AsyncClient(transport=ASGITransport(app=app),base_url="http://test") as client:
            response=await client.post("/search",json={"query":"sample"})
            assert response.status_code==200
            data=response.json(); assert len(data["results"])==5 and data["retrieval_mode"]=="dense"
            assert (await client.post("/search",json={"query":"  "})).status_code==422
            assert (await client.get("/health")).json()["indexed_passages"]==6
            assert (await client.get("/index/status")).json()["vector_size"]==3
    finally:
        store.close()
