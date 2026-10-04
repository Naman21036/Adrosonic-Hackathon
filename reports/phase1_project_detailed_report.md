# Adrosonic Phase 1 Project Report

**Report date:** 2026-10-03  
**Scope:** Dense-only retrieval project for the Adrosonic vector database challenge  
**Status:** Implementation and fixture checks complete; full model/data performance evaluation not run

## Executive summary

The workspace did not contain an existing project implementation, benchmark measurements, or a production vector collection when work began. A runnable Phase 1 dense retrieval application was assembled and then updated to make Qwen3-Embedding-0.6B the primary candidate. The implementation includes Hugging Face dataset ingestion, normalized Sentence Transformers embeddings, persistent local Qdrant cosine/HNSW search, a FastAPI service, Streamlit UI, evaluation and benchmark entry points, and a controlled experiment runner.

The owner-configured Hugging Face key was found in `.env`, loaded as a masked secret, passed to Hugging Face dataset/model calls, and validated with an authenticated Hub identity request. The actual `.env` was not modified and the key value was never printed or recorded.

The automated fixture suite passes **10 tests**. However, the installed environment does not have the required Qwen/Sentence Transformers and Datasets runtime stack or RAGAS. No corpus was indexed and no model was loaded. Therefore, there are **no measured baseline or Qwen scores**, no measured indexing speed or memory, and no latency percentiles. These values are unavailable, not zero. In particular, the requested p95 target of less than 300 ms has not been demonstrated.

## Starting state and audit findings

- The project directory was empty at task start; no working application or Git history was available to inspect or preserve.
- There were no pre-existing indexed passages, baseline benchmark numbers, or baseline evaluation scores. The baseline configuration below is the prior project default subsequently encoded in the project, not a measured baseline run.
- Windows hardware inventory was unavailable, and Python was not available as a system `python` command. The project-local virtual environment was used for fixture tests.
- The full machine-learning dependency setup did not complete: installation stalled while unpacking PyTorch and was stopped. The Qwen-compatible Sentence Transformers/Transformers stack, Hugging Face Datasets runtime, and RAGAS are not available for a full run.
- Existing status/report snapshots were retained and updated to document unavailable results rather than replacing them with estimates.

## What was implemented

### Retrieval application

- Dense-only passage retrieval over the MS MARCO passage corpus, preserving original passage text and stable passage IDs.
- Batched streaming ingestion with bounded working memory and resumable/upsert behavior based on stable Qdrant point IDs.
- Persistent local Qdrant storage, cosine distance, configurable HNSW, schema validation, and top-k search.
- FastAPI search/health/index-status routes and the Streamlit interface.
- A shared retrieval service used by the API, evaluation code, and benchmark code.
- A Sentence Transformers wrapper that loads a model once per application instance, uses inference mode, encodes queries/documents through their role-specific methods when available, normalizes vectors, checks dimensions and finite values, and converts output to FP32 arrays.

### Qwen and secret handling

- Primary candidate: `Qwen/Qwen3-Embedding-0.6B`.
- Preserved baseline candidate: `sentence-transformers/all-MiniLM-L6-v2`.
- Qwen query instruction is configurable; the Qwen query prompt is formatted as `Instruct: <instruction>\nQuery:` while document passages remain unprefixed.
- Output dimension is detected from the loaded model. Optional truncation is configurable; Qwen validation allows dimensions from 32 through 1024.
- Requested device and precision are configurable. CPU inference uses FP32; unavailable CUDA and unsupported CUDA BF16 requests fall back to CPU/FP32 or FP32 with warnings.
- `HUGGINGFACE_API_KEY` (also common HF aliases) is read into a Pydantic `SecretStr`. Token use is wired into Datasets and Sentence Transformers downloads. The actual `.env` remains unchanged and ignored by Git; `.env.example` contains only a blank placeholder.
- Authenticated Hugging Face `whoami` validation succeeded. Only the success status was retained.

### Experiment and measurement support

- `experiments/phase1_screening.json` defines 15 sequential screening configurations with distinct Qdrant collection names and a 10,000-passage screening limit.
- The experiment runner can list experiments, select runs, index each isolated collection, run a benchmark, optionally evaluate 20 examples, and save timestamped aggregate JSON plus per-run outputs.
- Indexing reports record run configuration, model/revision, requested/effective precision and device, throughput/duration, stage timings for streaming, embedding/vector conversion, and upsert, package versions, and available host details.
- The benchmark runs 100 queries by default, retains per-query latency rows, reports p50/p95/p99/mean/min/max, and records embedding and Qdrant search timing. The measured scope is an in-process service call; HTTP and JSON serialization are excluded.
- The evaluator is configured for 20 MS MARCO v1.1 validation examples. It uses RAGAS `NonLLMContextPrecisionWithReference` for Context Precision and reports exact-normalized-text Recall@5 and MRR@5 as separate diagnostics. Context Recall remains unavailable because no LLM judge or aligned passage-ID qrels are configured.

## Evaluation metrics and their interpretation

| Metric | How it is defined/used here | Result |
|---|---|---|
| RAGAS Context Precision | Reference-based precision comparing retrieved passages with the selected reference passage contexts; intended run size is 20 queries. Higher is better. | **Unavailable.** RAGAS is not installed, the model is not loaded, and no retrieval run occurred. The preserved evaluation attempt reports `ModuleNotFoundError: No module named 'ragas'`. |
| RAGAS Context Recall | Coverage of relevant reference information by retrieved contexts. The current offline evaluator does not have a semantic judge or passage-ID qrels to calculate it validly. | **Unavailable by design in this configuration.** No numeric score was generated. |
| Recall@5 diagnostic | For each query, fraction of exact normalized reference-context texts retrieved in the first five results; averaged across queries. This is separate from RAGAS. | **Unavailable.** No retrieval/evaluation run occurred. |
| MRR@5 diagnostic | Mean reciprocal rank of the first exact normalized reference-context match among the first five results; zero contribution when none is found. | **Unavailable.** No retrieval/evaluation run occurred. |
| Query latency p50/p95/p99 | Percentiles over each recorded in-process query call, including query embedding, Qdrant search, and result conversion. HTTP/JSON serialization is excluded. | **Unavailable.** No model/index benchmark ran; 100 consecutive latency samples do not exist. |
| Mean/min/max latency | Summary statistics over the same per-query in-process latency values. | **Unavailable.** No latency samples exist. |
| Indexing passages/sec and duration | Indexed passages divided by elapsed run time; duration and stage durations are saved in indexing report. | **Unavailable.** No passages were indexed. |
| Peak system/GPU memory | Required by the challenge for actual hardware runs. | **Unavailable.** No full workload ran and hardware inventory was not available. |
| Quantization quality/latency | Challenge asks for comparison of full-precision vectors and Qdrant INT8 scalar quantization/rescoring when supported. | **Not implemented or measured.** No quantization comparison is in the current screening plan. |

No unavailable metric has been replaced with an estimate or zero. No quality or latency improvement is claimed.

## Configuration comparison

The following “baseline” values reproduce the prior source defaults specified in the experiment plan; they are not measured baseline outcomes.

| Parameter | Baseline reference | Current Qwen candidate default | Measured/selected? |
|---|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `Qwen/Qwen3-Embedding-0.6B` | Candidate configured, not loaded/evaluated |
| Embedding dimension | Model native (MiniLM model dimension) | 1024 native; config supports smaller output dimensions | Runtime dimension detection implemented; not measured |
| Precision | FP32 | FP32 | Configured, not benchmarked |
| Device | CPU default | CPU default, CUDA configurable | Configured, no GPU run |
| Ingestion/embedding batch size | 128 | 128 | Configured, no throughput comparison |
| HNSW `m` | 16 | 16 | Configured, no tuning results |
| HNSW `ef_construct` | 100 | 100 | Configured, no tuning results |
| Query `ef_search` | 64 | 64 | Configured, no tuning results |
| Distance | Cosine | Cosine | Implemented |
| Public top-k | 5 | 5 | Implemented; no measured quality results |
| Screening corpus size | None measured | 10,000 per screening experiment | Plan only; not run |
| Acceptance corpus | Challenge target: at least 100,000 passages; full corpus target is approximately 9.85M | Default `DATA_LIMIT=100000` | Not indexed |

The Qwen defaults are an initial candidate configuration, **not a final tuned winner**. Final parameter selection must be based on measured quality, latency, throughput, and memory.

## Experiment matrix defined in the project

All listed screening experiments use FP32, a 10,000-passage data limit, and separate collections. Variables not named in a row remain at `m=16`, `ef_construct=100`, `ef_search=64`, and batch size 64 for the HNSW/dimension screening rows.

| Experiment group | Values present in plan | Count | Status |
|---|---|---:|---|
| MiniLM reference | MiniLM, native dimension, batch 128, `m=16`, `ef_construct=100`, `ef_search=64` | 1 | Not run |
| Qwen batch screening | 1024 dimensions; batches 16, 32, 64, 128 | 4 | Not run |
| Qwen dimension screening | 1024, 768, 512, 256 dimensions; batch 64 | 3 reduced-dimension entries (1024 is covered in batch group) | Not run |
| HNSW `ef_construct` | 200, 400; plus plan reference 100 | 2 variants | Not run |
| HNSW `m` | 8, 32; plus plan reference 16 | 2 variants | Not run |
| Query `ef_search` | 32, 128, 256; plus plan reference 64 | 3 variants | Not run |
| **Total plan entries** | Baseline plus Qwen screening | **15** | Runner `--list` verified; experiments not run |

The original problem also requests CPU and GPU comparisons, supported FP16/BF16, separate data-loader and embedding batch controls, embedding-cache safety, INT8 scalar quantization/rescoring at 1x/2x/4x, storage-mode comparison where meaningful, and top-k sensitivity tests. Those are **not currently represented in the 15-entry screening plan** and remain future experiment work. The current batch setting controls the ingestion/embedding batch together; independent data-loading and model batch sizes have not been implemented.

## Results against acceptance targets

| Problem-statement target | Current result |
|---|---|
| Qwen evaluated as primary model | **Not met yet:** integration/configuration exists; no Qwen model run. |
| At least 100 consecutive query timings; report percentiles | **Not met yet:** benchmark capability exists; no run/log. |
| Warm p95 below 300 ms | **Unverified:** no latency results. |
| RAGAS Context Precision and Context Recall from 20 valid examples | **Unavailable:** Context Precision failed because RAGAS is missing; Context Recall has no configured valid judge/qrels. |
| Recall@5 and MRR@5 diagnostics | **Implemented in evaluator, not measured.** |
| At least 100,000 passages indexed; target duration below 2 hours | **Not met yet:** no full corpus index or duration measurement. |
| Stage-level indexing duration and throughput | **Instrumentation implemented; no actual report from a successful ingestion.** |
| Peak RAM/GPU memory | **Not measured.** |
| Batch and HNSW parameter comparison | **Experiment plan/runner implemented; no experiments run.** |
| FP32 versus Qdrant INT8 quantization comparison | **Not run and not in current experiment matrix.** |
| Preserve dense-only Phase 1 scope | **Met:** no BM25, hybrid fusion, reranking, or other Phase 2 retrieval feature was added. |
| Automated fixture tests | **Met:** 10 passed in 5.33 seconds using project `.venv`. |
| Hugging Face secret safety/authentication | **Met:** key present, authenticated request succeeded, token value not exposed; `.env` unchanged. |

## Reproduction commands

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The conditional copy preserves an existing `.env`. Ensure its `HUGGINGFACE_API_KEY` stays local and secret. Install the optional evaluation set when evaluating:

```powershell
python -m pip install -r requirements-eval.txt
```

Then use:

```powershell
python -m pytest -q
python -m adrosonic_retrieval.indexer
uvicorn adrosonic_retrieval.api:app --host 127.0.0.1 --port 8000
streamlit run src/adrosonic_retrieval/ui.py
python -m adrosonic_retrieval.benchmark --queries 100
python -m adrosonic_retrieval.evaluate
python -m adrosonic_retrieval.experiment_runner --list
python -m adrosonic_retrieval.experiment_runner --experiment qwen-1024-batch64 --run --queries 100
python -m adrosonic_retrieval.experiment_runner --experiment qwen-1024-batch64 --run --queries 100 --evaluate
```

Run indexer and API against different times; the local Qdrant path should not be opened for ingestion and serving concurrently. Generated result files are written beneath `OUTPUT_DIR` (default `artifacts/`). Existing markdown summaries are in `reports/`.

## Files changed or added

- `src/adrosonic_retrieval/config.py` — model, dimension, precision, HNSW and token configuration.
- `src/adrosonic_retrieval/embeddings.py` — model loading, Qwen prompt, role-aware encoding, normalization and device/precision fallback.
- `src/adrosonic_retrieval/dataset.py` — streaming corpus/evaluation dataset adapters and authenticated Hub calls.
- `src/adrosonic_retrieval/indexer.py` — bounded ingestion, resumable stable-ID upsert, stage timings and indexing report.
- `src/adrosonic_retrieval/store.py` — Qdrant collection schema validation, HNSW config and dense search.
- `src/adrosonic_retrieval/benchmark.py` — 100-query benchmark and per-query latency output.
- `src/adrosonic_retrieval/evaluate.py` — RAGAS precision and separate exact-match retrieval diagnostics.
- `src/adrosonic_retrieval/experiment_runner.py` — sequential experiment execution and isolated collections.
- `experiments/phase1_screening.json` — 15-entry experiment plan.
- `requirements.txt`, `requirements-eval.txt`, `.env.example`, `.gitignore`, `README.md` — dependencies, setup and secret-safe docs.
- `reports/indexing_report.md`, `reports/benchmark_report.md`, `reports/evaluation_report.md`, `reports/optimization_report.md` — status and results documentation.
- This report: `reports/phase1_project_detailed_report.md`.

## Remaining work to reach the problem-statement acceptance criteria

1. Complete a compatible dependency installation and confirm package versions on the target machine.
2. Run the 10k baseline and Qwen screening plan; record failures, package/model revisions, hardware and raw outputs.
3. Evaluate the same valid 20+ query set for baseline and Qwen candidates. Resolve a valid Context Recall method (permitted local/free judge or aligned passage-ID qrels) before reporting that metric.
4. Add and run the missing GPU/precision, independent loader-batch, INT8/rescoring, memory/storage, and top-k experiments where the installed hardware and Qdrant support them.
5. Select a configuration based on retrieval quality and latency together; then index at least 100,000 passages and record peak memory and full ingestion duration.
6. Run the final configuration for at least 100 consecutive representative queries. Record raw per-query data, p50/p95/p99/mean/min/max, and clearly report whether p95 is below 300 ms. Add HTTP/JSON timing if acceptance requires actual API end-to-end rather than in-process retrieval.
7. Update the reports with actual results and make no performance claim until those runs have completed.
