# Phase 2 validation run

**Run date:** 2026-10-03 (Asia/Calcutta)

Command: `python -m compileall -q app streamlit_app.py tests`  
Result: passed with no syntax errors.

Command: `python -m pytest tests -p no:cacheprovider -q`  
Result: **8 passed, 2 skipped**, exit code 0.

The skipped tests are local-Qdrant integration tests for dense/sparse filter behavior, live upsert/delete synchronization, and persistence after reopening. They skip because `qdrant-client` is absent. The passing tests cover RRF ordering, scalar metadata validation, SQLite FTS5 BM25 filtering/update/delete, API request bounds, degraded-index API responses, Mistral gating/citation checks, the MS MARCO Parquet loader configuration, device selection/configuration, simulated CUDA OOM batch reduction, and deterministic ingestion resume-to-total behavior.

The API degraded-mode probe found that upsert/delete had wrapped readiness `503` errors into `500`; exception handling was fixed. The post-fix test verifies `/search`, `/upsert`, and `/passages/{id}` all return `503` when the service has no usable index. No model, dataset, judge, or Mistral request was made by the tests.

Additional check: `python -m app.embedding_benchmark --help` passed. No performance measurements were collected by that help check.

Hardware/runtime check: the NVIDIA driver sees an RTX 3050 with 6 GiB VRAM and reports CUDA 13.0, but the active Python has `torch 2.14.0+cpu`, `torch.version.cuda is None`, and `torch.cuda.is_available()` is false. Therefore the simulated CUDA tests do not establish physical GPU inference; real CUDA inference, VRAM use, full ingestion, and throughput remain unverified until a CUDA-enabled PyTorch build is installed.
