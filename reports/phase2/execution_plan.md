# Phase 2 execution plan

**Prepared:** 2026-10-03 (Asia/Calcutta)

## Audit snapshot

- Existing source is a small Python app in `app/`, with FastAPI, Streamlit, Qwen embeddings, local Qdrant, SQLite FTS5 BM25, RRF, MS MARCO loaders, RAGAS evaluation, a benchmark runner, and pytest tests.
- The configured model is `Qwen/Qwen3-Embedding-0.6B`, revision `main`; the MS MARCO config is `microsoft/ms_marco`, `v1.1`.
- `.env` contains Hugging Face and Mistral credentials. Their values are not read into reports or printed. Mistral calls remain disabled unless the free allowance is explicitly confirmed.
- Configured Qdrant path `storage/qdrant`, BM25 path `storage/sparse.sqlite3`, and update journal `storage/update_journal.jsonl` do not exist. No prior index is reusable.
- No Phase 1 evaluation, benchmark artifacts, or Git repository metadata were found. Existing `reports/indexing_report.md`, `reports/evaluation_report.md`, and `reports/benchmark_report.md` mark measurements as not run.
- Host: Windows 11, Python 3.13.15, 20 logical CPUs, 7.7 GB RAM, 213.73 GB free disk, no CUDA GPU. Installed FastAPI 0.119.0, Pydantic 2.11.7, Transformers 5.8.1, Datasets 5.0.0, Streamlit 1.48.0, pytest 8.4.2, PyTorch 2.14.0 CPU. `qdrant-client` and `ragas` are missing.
- Earlier checks in this task found outbound Hugging Face access refused and package-index resolution returned no Qdrant versions. Recheck the direct Parquet endpoint below; do not assume access from cache settings.

## Execution sequence

1. Reconfirm configured index paths, model cache, environment versions, free disk/RAM, and Hugging Face access without printing secrets.
2. Install project requirements only if the package index is reachable and installs compatible wheels without replacing the working Python environment. Do not overwrite the existing `.env`.
3. If Qdrant, dataset access, model download, and RAM permit, run the existing streaming ingestion into an isolated Phase 2 collection, limiting it to exactly 100,000 unique passage IDs; validate nonempty text, matching Qdrant/BM25 counts, uniqueness, and payload equality before evaluation.
4. Run real dense, sparse, and hybrid validation queries using MS MARCO validation answers and selected passages. Record top five IDs/text/scores and empty/invalid/filter cases. Do not change the indexed evaluation corpus during tests or benchmarking.
5. Run the existing pytest suite and isolated update/persistence integration tests. If Qdrant is unavailable, run all independent tests and mark integration tests skipped with the dependency error.
6. Run paired dense/hybrid RAGAS Context Precision and Context Recall evaluation only if at least 20 valid labeled queries and an allowed judge are available. Otherwise record the exception and unavailable status.
7. Run at least 100 consecutive validation queries per retrieval mode, saving every successful/failed request and timing. Report p50/p95/p99/mean/min/max and failures; keep generation timing separate.
8. Validate Mistral grounded answers only if configured free-tier allowance is verified. Never issue paid calls.
9. Write timestamped raw evaluation/benchmark artifacts and the technical report from measured evidence only. Retain unmet criteria as unverified or blocked.

## Stop conditions and current expected gaps

- Do not overwrite any existing Qdrant collection or sparse index. There are none at audit time.
- Do not call Mistral while `MISTRAL_FREE_TIER_CONFIRMED` is false.
- If the Hub or package index remains unreachable, no full ingestion or measured retrieval benchmark can be claimed. Finish local tests and document the precise network/dependency errors.
- If the local model or index exceeds available resources, preserve any partial indexing report and do not claim the 100,000-passage target.
