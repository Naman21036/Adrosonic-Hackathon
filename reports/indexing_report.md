# Indexing report

**Status: not run in this environment.** The project directory was empty at task start; Python 3.11 was not available on PATH. The bundled workspace runtime provides Python 3.12.14. Core fixture-test dependencies were installed, but the full requirements installation stalled while unpacking the ML stack and was stopped; `sentence-transformers` and `datasets` are absent from that environment. No model weights, MS MARCO passages, or production Qdrant collection were created. No ingestion duration or indexed-count claim is made.

Hardware note: the temporary Qdrant integration tests ran on the Windows desktop host. Windows hardware inventory calls were denied, so CPU model, RAM, and accelerator details are unavailable. No full-scale indexing was performed on this hardware.

Fixture verification: 10 tests pass, including Qdrant 1.13.3 temporary-collection persistence/schema checks and indexing batching/deduplication. Default acceptance run uses 100,000 unique MS MARCO corpus passages. On a completed run, `artifacts/indexing_report.json` records the UTC run start, dataset/config/split, model and revision, vector dimension, requested/effective precision and device, batch size, counts before/after, elapsed duration, per-stage timing, throughput, package versions, hardware/platform details, and errors. Full-corpus target (under two hours) remains unmeasured.
