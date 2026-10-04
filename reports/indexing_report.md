# Indexing report

**Status: NOT RUN.** No dataset passages are indexed in this workspace.

The current host reports Python 3.13.15, 20 logical CPUs, 7.7 GB RAM, no CUDA GPU, and approximately 214 GB free on C:. The requested Qdrant dependency is not installed. Installing `qdrant-client` failed because the package index returned no available versions in this network-restricted environment. Therefore, there is no honest corpus count, duration, throughput, or storage measurement to report yet.

Run `python -m app.ingest --limit 100000` after dependencies and dataset access are available. The ingestion process writes its measured hardware, passage counts, duration, throughput, and errors to `artifacts/indexing_report.json`.
