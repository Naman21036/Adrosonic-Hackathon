# RAG Precision Engine frontend

Next.js App Router dashboard for the Adrosonic FastAPI retrieval project. It provides system diagnostics, dense/hybrid search, source filtering, evaluation reporting, an explicitly simulated benchmark demo, index updates, and observability status.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Start FastAPI separately from the repository root:

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

The browser uses the same-origin Next.js proxy (`NEXT_PUBLIC_API_URL=/api/backend`); the server-side proxy calls `BACKEND_API_URL`, defaulting to `http://127.0.0.1:8000`. Copy `.env.local.example` to `.env.local` only when those non-secret local URLs need an override. Backend API keys are never exposed to browser code.

## Routes

- `/` Overview and current index/evaluation summary
- `/playground` Live dense/hybrid retrieval and source-ID filter demo
- `/evaluation` Stored RAGAS report and matched-corpus results
- `/benchmark` Separate Actual Results and 100-Query Demo tabs
- `/architecture` Retrieval architecture
- `/index` Live index/corpus status
- `/updates` Single-passage upsert/delete controls
- `/observability` Sanitized local logs and Langfuse availability
- `/settings` Safe runtime/report configuration

`/benchmark` Actual Results read only the stored matched-corpus evaluation report. The 100-Query Demo reads `reports/demo/benchmark_100_simulated.json`; every row is marked `status=SIMULATED` and `executed=false`. It does not call the backend or use the production benchmark data path.

## Checks

```powershell
npm run lint
npm run typecheck
npm run build
```
