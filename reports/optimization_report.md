# Phase 1 optimization report

## Scope and changes

This remains a dense-only Phase 1 system. No BM25, hybrid fusion, reranker, query expansion, or other Phase 2 feature was added.

- Qwen3-Embedding-0.6B is now the configured primary candidate; the earlier MiniLM model remains available through `BASELINE_EMBEDDING_MODEL` and a separate collection name. The previous status reports and evaluation failure output were preserved.
- Query instruction, output dimension, precision, model revision, device, batch size, and HNSW values are configurable. The embedding dimension is read from the loaded model, with optional Sentence Transformers truncation. Qwen's model card documents the 1024 native dimension, Matryoshka dimensions down to 32, and the query instruction flow.
- The existing `HUGGINGFACE_API_KEY` in `.env` is now loaded as a `SecretStr`, passed to Hugging Face model/dataset calls, and kept out of logs/reports. Authenticated identity validation succeeded. `.env` itself was not modified; `.env.example` only has an empty variable placeholder.
- The experiment plan screens a preserved MiniLM reference against Qwen batch sizes, dimensions (1024, 768, 512, and 256), and HNSW values using isolated collection names. A runner writes per-experiment reports and query logs without overwriting existing reports.
- Embedding inference is wrapped in `torch.inference_mode()`. Qwen query prompts use an instruction while document text is left unprefixed. CPU half-precision requests fall back to FP32.

## Baseline and results

The initial source default was `sentence-transformers/all-MiniLM-L6-v2`, batch size 128, HNSW `m=16`, `ef_construct=100`, `ef_search=64`, top-k 5. There were no indexed passages or actual baseline performance metrics in the workspace, so no measured comparison can be made.

The optimized Qwen configuration has **not been loaded or evaluated**. The environment lacked the Qwen-compatible Sentence Transformers/Transformers stack and model weights; the full dependency installation was interrupted after it stalled during PyTorch unpacking. The owner-provided Hub token was present and validated, but a valid token does not replace the missing model, dataset, and vector index.

The `.env` key check found the nonempty `HUGGINGFACE_API_KEY` entry and the authenticated Hub `whoami` request succeeded. No token value was printed or saved in reports.

| Measurement | Baseline | Qwen candidate | Status |
|---|---:|---:|---|
| Context Precision / Recall | unavailable | unavailable | RAGAS package and model/index not available |
| Recall@5 / MRR@5 | unavailable | unavailable | no measured relevance run |
| Indexing throughput / duration | unavailable | unavailable | no corpus indexed |
| Peak RAM / GPU memory | unavailable | unavailable | host inventory unavailable; no model run |
| p50 / p95 / p99 query latency | unavailable | unavailable | no index/model; 300 ms p95 target unverified |
| INT8 quantization quality/latency | not run | not run | requires controlled indexed comparisons |

## Reproduction

Use `experiments/phase1_screening.json` and `python -m adrosonic_retrieval.experiment_runner --list`. Select experiments explicitly with `--run`; screening entries use 10,000 passages. After selecting a quality/latency tradeoff from actual results, rerun that setting with `DATA_LIMIT=100000` for acceptance. Each model/dimension and HNSW configuration has its own Qdrant collection. The runner persists per-query logs and a timestamped raw manifest. Do not treat the plan as completed results.

## Remaining work

The complete model/data dependencies must be installed, then the experiment plan, full 100,000-passage ingestion, at least 100-query benchmark, and 20-query RAGAS evaluation need to run on the target hardware. Add FP16/BF16 and quantization experiments only where hardware and quality judgments permit, and select a configuration from measured retrieval quality, memory, indexing throughput, and latency.
