# Phase 2 indexing report

**Run date:** 2026-10-03 (Asia/Calcutta)  
**Status:** Blocked before ingestion; no index data was written.

## Audit and attempted execution

- Dataset configured: `microsoft/ms_marco`, config `v1.1`; ingestion split: `train`.
- Dataset access route: the existing compatibility loader requested the public MS MARCO v1.1 Parquet file from Hugging Face, using the configured HF token without printing it.
- Observed failure while opening the validation split: `httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it`.
- Configured Qdrant local path `storage/qdrant`, sparse path `storage/sparse.sqlite3`, and journal path `storage/update_journal.jsonl` were all absent during audit.
- `qdrant-client` is not installed. A previous `pip install 'qdrant-client>=1.12,<2'` attempt returned `ERROR: Could not find a version that satisfies the requirement` and `No matching distribution found`, consistent with package-index access being unavailable.
- The local Qwen model snapshot is not cached. No Qwen inference was started.

## Counts

| Measure | Actual value |
|---|---:|
| Source rows read | 0 |
| Source passages read | 0 |
| Unique IDs processed | 0 |
| Nonempty passages indexed | 0 |
| Failed records | 0 (ingestion never began) |
| Duplicate records observed | 0 (ingestion never began) |
| Qdrant collection | Absent; client dependency missing |
| Qdrant points | No local collection exists at the configured path; count API unavailable |
| BM25 documents | Sparse database absent; no records at the configured path |
| Matching IDs across both indexes | Not applicable |

## Configuration and host

- Embedding model: `Qwen/Qwen3-Embedding-0.6B`, revision `main` (not commit-pinned).
- Embedding dimension: not measured; model was not loaded.
- Effective precision: not measured. `EMBEDDING_PRECISION` is present in settings, but the current embedding loader does not apply it to the model dtype.
- Configured batch size: 4; HNSW: `m=16`, `ef_construct=100`, `ef_search=64`.
- Host: Windows 11, Python 3.13.15, 20 logical CPUs, CPU-only, 7.7 GB RAM, 213.73 GB free disk at audit.
- Indexing duration, throughput, index disk use, and process memory during indexing: not measured.

No 100,000-passage acceptance claim is made. Retry with package-index and Hugging Face connectivity, then run `python -m app.ingest --limit 100000`; verify both index counts and passage ID equality before proceeding.
