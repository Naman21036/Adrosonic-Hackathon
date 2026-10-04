"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .config import get_settings
from .embeddings import Embedder
from .store import PassageStore
from .retrieval import RetrievalService, CollectionUnavailable
from .schemas import SearchRequest, SearchResponse, HealthResponse, IndexStatusResponse

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder(settings.embedding_model, settings.device, dimension=settings.embedding_dimension,
        precision=settings.embedding_precision, query_instruction=settings.query_instruction,
        token=settings.hf_token_value, revision=settings.model_revision)
    store = PassageStore(settings.qdrant_path, settings.collection_name)
    app.state.service = RetrievalService(embedder, store, settings.top_k, settings.ef_search)
    app.state.embedder, app.state.store = embedder, store
    yield
    store.close()

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not initialized")
    try:
        data = service.search(request.query, request.top_k)
        return SearchResponse(query=request.query, results=data["results"], latency_ms=data["latency_ms"])
    except CollectionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Search failed")
        raise HTTPException(status_code=500, detail="Retrieval failed due to a database or model error") from exc

@app.get("/health", response_model=HealthResponse)
def health():
    store = getattr(app.state, "store", None)
    embedder = getattr(app.state, "embedder", None)
    status = store.status() if store else {"exists": False, "count": 0}
    return HealthResponse(status="ok" if status["exists"] and status["count"] else "index_missing", collection=settings.collection_name,
        indexed_passages=status["count"], model_loaded=embedder is not None)

@app.get("/index/status", response_model=IndexStatusResponse)
def index_status():
    store = getattr(app.state, "store", None)
    status = store.status() if store else {"exists": False, "count": 0, "vector_size": None, "distance": None}
    return IndexStatusResponse(collection=settings.collection_name, exists=status["exists"], indexed_passages=status["count"],
                               vector_size=status["vector_size"], distance=status["distance"])
