# Frontend integration and backend API gaps

The Next.js dashboard consumes the existing FastAPI routes through a same-origin server proxy:

- `GET /health`
- `GET /status`
- `POST /search` (`dense` and `hybrid`)
- `POST /upsert`
- `DELETE /passages/{passage_id}`

The UI reads the existing evaluation, benchmark, and Loguru artifacts from the local project directory on the Next.js server. These read-only report handlers are not presented as FastAPI endpoints.

The following minimum FastAPI additions would enable the currently unavailable UI actions and data:

1. Add `mode: "sparse"` to `POST /search` and route it to the existing `HybridIndex.sparse_search` implementation.
2. Add a safe `GET /metrics` or `GET /evaluation` endpoint that returns an allow-listed, paginated summary from the stored reports. Do not return credentials or full contexts by default.
3. Add a background-job `POST /benchmark` plus read-only job status/result route; the current benchmark command initializes the Qwen model and is not exposed over HTTP.
4. Add a read-only `GET /observability` route for recent sanitized Loguru events and, if needed, recent Langfuse trace metadata.
5. Extend `GET /status` with source-ID cardinality and exact parity metadata only if those values can be computed without rescanning large indexes on every request.
6. Extend `GET /health` with Langfuse configuration and connectivity state separately; the current Azure field reports configuration presence, not provider connectivity.
7. Return backend stage timings from `POST /search` if the UI should show embedding, dense, BM25, and RRF durations. It currently returns total retrieval latency only.
8. Return mutation latency and a conflict/not-found outcome from `/upsert` and `/passages/{id}` if server-measured mutation duration is required.

The frontend leaves these missing capabilities visibly unavailable and never substitutes sample values for live retrieval, RAGAS, latency, traces, or index state.
