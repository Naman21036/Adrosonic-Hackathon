"""Framework-independent dense retrieval service."""
import time
from .embeddings import Embedder
from .store import PassageStore

class CollectionUnavailable(RuntimeError):
    pass

class RetrievalService:
    def __init__(self, embedder: Embedder, store: PassageStore, top_k: int = 5, ef_search: int = 64) -> None:
        self.embedder, self.store, self.top_k, self.ef_search = embedder, store, top_k, ef_search
        # Check once during service construction; keep the query hot path to one DB search call.
        self.collection_status = store.status()

    def search(self, query: str, top_k: int | None = None) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not self.collection_status["exists"] or not self.collection_status["count"]:
            raise CollectionUnavailable("The vector collection is missing or empty; run the index command first")
        limit = self.top_k if top_k is None else top_k
        if not 1 <= limit <= 100:
            raise ValueError("top_k must be between 1 and 100")
        start = time.perf_counter()
        vector = self.embedder.encode_query(query)
        embedding_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        results = self.store.search(vector, limit, self.ef_search)
        search_ms = (time.perf_counter() - start) * 1000
        return {"results": results, "latency_ms": embedding_ms + search_ms,
                "timings_ms": {"embedding": embedding_ms, "qdrant_search": search_ms}}
