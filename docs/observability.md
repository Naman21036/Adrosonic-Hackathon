# Application observability

This project uses Loguru for local structured logs and Langfuse for RAG and LLM traces. The instrumentation is additive: it does not alter Qwen encoding, Qdrant or BM25 queries, RRF ranking, Azure request parameters, or RAGAS metric calculations. No Prometheus, Grafana, standalone OpenTelemetry instrumentation, exporters, or metrics service is configured.

## Configuration

Add these settings to `.env` (the Langfuse credentials and base URL are already configured in this workspace):

```dotenv
LOG_LEVEL=INFO
LOG_FILE=logs/app.jsonl
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

The application accepts `LANGFUSE_HOST`, `LANGFUSE_BASE_URL`, and the legacy `LANGFUSE_BASEURL` setting. If `LANGFUSE_ENABLED` is false or either credential is missing, tracing is disabled and requests continue normally. The default level is `INFO`; set `LOG_LEVEL=DEBUG` to increase application detail.

Never commit `.env`. Keep credentials out of command output, logs, traces, screenshots, and reports. Local logging drops credential-shaped fields and redacts configured keys, bearer values, and common `sk-`/`pk-` tokens. Query and selected context text are intentionally present in Langfuse observations for the RAG trace, so configure the Langfuse project and access controls accordingly.

## Loguru

Logging is configured once by `app.observability.logging`. Existing standard-library loggers are bridged into Loguru so the application has one output path. The console sink is human-readable; the rotating JSON Lines file defaults to `logs/app.jsonl` (20 MB rotation, 14-day retention). The `logs/` directory is ignored by Git.

Each record includes timestamp, level, source module/function, event name, request ID when available, and structured event fields. Request-scoped correlation values such as `evaluation_id`, `query_id`, and `retrieval_mode` are attached automatically while those contexts are active.

Events include:

- API: `API_REQUEST_STARTED`, `API_REQUEST_COMPLETED`, `API_REQUEST_FAILED`, `RAG_REQUEST_STARTED`, `RAG_REQUEST_COMPLETED`, `RAG_REQUEST_FAILED`
- Retrieval: `EMBEDDING_STARTED`, `EMBEDDING_COMPLETED`, `EMBEDDING_FAILED`, `DENSE_RETRIEVAL_STARTED`, `DENSE_RETRIEVAL_COMPLETED`, `DENSE_RETRIEVAL_FAILED`, `QDRANT_QUERY_COMPLETED`, `BM25_RETRIEVAL_STARTED`, `BM25_RETRIEVAL_COMPLETED`, `BM25_RETRIEVAL_FAILED`, `RRF_STARTED`, `RRF_COMPLETED`
- Generation: `LLM_REQUEST_STARTED`, `LLM_REQUEST_COMPLETED`, `LLM_REQUEST_FAILED`
- Evaluation: `RAGAS_EVALUATION_STARTED`, `RAGAS_EVALUATION_COMPLETED`, `RAGAS_EVALUATION_FAILED`, plus mode/query events

Retrieval log fields include candidate counts, top document IDs/scores, selected context IDs and character counts, collection, and stage latency. Full passage text and embedding vectors are not written to local logs.

## Request IDs

Every FastAPI request gets an `x-request-id`. A safe caller-provided ID is reused for correlation; otherwise the middleware generates a UUID. The ID is included in the response header and propagated through request logs and Langfuse observation metadata. Use it to find all stages belonging to one request.

## Langfuse traces

When enabled, `/search` creates one `rag_request` trace. Nested observations include:

```text
rag_request
└── dense_retrieval
    ├── embedding
    └── qdrant_query
└── bm25_retrieval                 (hybrid mode)
└── rrf                            (hybrid mode)
└── context_assembly
└── azure_llm_generation           (only when generate_answer=true)
```

The root contains the query, retrieval mode, and `top_k`. Embedding captures query input, model, dimension, and latency, never the vector. Retrieval captures top IDs/scores and latency. Context assembly records selected IDs, count, and character lengths. Azure generation records deployment, query plus selected context IDs/text, answer, latency, and token usage when returned by Azure. Errors are recorded by safe error type; credentials and authorization headers are never included.

RAGAS evaluation creates `ragas_evaluation` traces with evaluation ID, mode, query IDs, RAGAS version, judge provider/model, per-mode latency, success counts, and Context Precision/Recall. It does not add reference answers or retrieved passage contents to Langfuse evaluation traces.

## SDK compatibility note

The project explicitly excludes OpenTelemetry instrumentation. It currently pins Langfuse Python SDK `2.60.9` and uses that installed SDK's `trace()`, nested `span()`/`generation()`, `end()`, and `flush()` APIs. This avoids adding a separate OpenTelemetry layer. Langfuse's current Python SDK is v4 and uses OpenTelemetry internally; Langfuse documents the legacy Cloud ingestion endpoint used by SDK v2 as deprecated and scheduled to stop accepting trace events on **November 16, 2026**. At that point, complying with the no-OpenTelemetry constraint will require a different Langfuse transport or an explicit change to allow the newer SDK. See [Langfuse's SDK overview](https://langfuse.com/docs/observability/sdk/overview) and [compatibility matrix](https://langfuse.com/docs/compatibility).

## Verify setup

Run local checks:

```powershell
.\venv\Scripts\python.exe -m compileall -q app tests
.\venv\Scripts\python.exe -m pytest
```

Start the API with Langfuse enabled and issue a request:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -Headers @{ 'x-request-id' = 'observability-smoke-001' } -ContentType 'application/json' -Body '{"query":"your question","mode":"hybrid","top_k":5,"generate_answer":true}'
```

The response's `x-request-id` should match the log records in `logs/app.jsonl`. Inspect Langfuse for a `rag_request` trace and its nested observations. The FastAPI lifespan flushes pending traces at shutdown. To verify the disabled path, set `LANGFUSE_ENABLED=false` and issue the same request; retrieval and generation continue without Langfuse observations.

## Troubleshooting

- **No Langfuse trace:** Check that `LANGFUSE_ENABLED=true`, both credentials are non-empty, and the base URL points to the correct Langfuse region. Do not print the credentials when checking.
- **Trace arrives after a delay:** The SDK batches events; shut down the API cleanly to flush them. Check DNS and outbound HTTPS connectivity to the configured host.
- **Langfuse is unavailable:** A warning is written locally and the request continues. Langfuse is deliberately isolated from Qwen, Qdrant, BM25, RRF, Azure generation, and evaluation.
- **No local file:** Confirm the process can create `logs/` in the project directory and check `LOG_FILE` and `LOG_LEVEL`.
- **Azure failure:** Search local logs by request ID and inspect the safe error type. Azure response bodies and authorization details are not copied into logs.
