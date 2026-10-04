# Phase 2 ingestion diagnostics

**Audit date:** 2026-10-03 (Asia/Calcutta)  
**Outcome:** ingestion was invoked with a 100,000-passage limit and stopped before reading any source rows. No passages were indexed.

## Evidence and actual cause

| Check | Observed evidence | Finding |
|---|---|---|
| Dataset and configuration | `microsoft/ms_marco`, config `v1.1`; ingestion requests `train`; the evaluation loader requests `validation` | Intended dataset/config and splits are explicit in configuration and source. |
| Configured limit | `DATA_LIMIT=100000`; invoked as `python -m app.ingest --limit 100000` | Limit was not zero or too small. |
| Loader invocation | Ingestion calls `load_split("train")` before constructing the embedder/index | Loader executes before indexing begins. |
| Dataset access | Opening the official HF Parquet route failed with `httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it` | The host/network refused the Hugging Face connection. Dataset iteration never started. |
| Dataset implementation | Installed Datasets is 5.0.0; the old dataset-script route is incompatible with the modern library path | Replaced the dataset-script approach with the repository's official v1.1 Parquet file URL. This is a valid route, but it cannot be verified/downloaded while network access is refused. |
| Qdrant | `qdrant-client` is absent; attempted install returned “No matching distribution found”; configured Qdrant directory is absent | No Qdrant collection could be created, counted, or persisted. Package index access is unavailable or filtered. |
| Embeddings | Qwen snapshot is not present in the local Hugging Face cache; no CUDA device | Model loading/inference was never reached because source loading fails first; no embedding values/dimension were measured. |
| Sparse index | Configured SQLite path is absent | No BM25/FTS records were written. |
| Skips/exceptions | Loader exception propagated out of the run; it did not reach passage processing | No silent record skipping occurred; 0 rows were read and 0 were indexed. |
| Persistence/restart | No Qdrant or sparse store exists | Persistence and journal recovery could not be exercised on a real corpus. |

## Root cause and attempted resolution

The immediate, observed failure is outbound access to Hugging Face being actively refused. A parquet-based MS MARCO v1.1 loader was implemented to bypass the legacy dataset-script loading path on Datasets 5.x. The source was then actually attempted via `python -m app.ingest --limit 100000`; it still failed at the network connection step before the first row. Installing `qdrant-client` was also attempted but the package index yielded no matching distribution. No credentials or key values were included in logs or artifacts.

The limit, split selection, and control flow do not explain this run's zero count. `app.ingest` constructs the dataset first and does not hide its download exception. The loader path update resolves the code-path compatibility issue, but external source connectivity and package availability remain unresolved in this environment. The Qwen model is also uncached and would need an accessible model download (or an already populated cache) before embeddings can be produced.

## Code change for verifiable follow-up

Added `python -m app.validate_corpus`, an independent gate that compares all logical passage IDs from Qdrant payloads against IDs in the SQLite BM25 table, checks exact counts and nonempty text, checks first/middle/last ID retrieval, verifies the live Qwen vector dimension, and makes a real Qdrant query. It writes `reports/phase2/corpus_validation.json`; it exits nonzero if the required stores/dependencies are absent or any check fails. On this host it records blockers instead of inventing a corpus result.

## Required conditions to resume

Run from a network-enabled host where `huggingface.co` and the configured Python package index are reachable, with `qdrant-client` and project requirements installed. Ensure Qwen can be downloaded or is cached. Then run, in order:

```powershell
python -m app.ingest --limit 100000
python -m app.validate_corpus
```

Only proceed to retrieval, Mistral/RAGAS evaluation, and 100-query benchmarks if corpus validation reports `passed`. Keep `MISTRAL_FREE_TIER_CONFIRMED=false` unless the account/model allowance is independently verified.
