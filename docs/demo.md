# Ten-minute dashboard demo

## Before the demo

1. Generate or validate the deterministic UI fixture: `python scripts/generate_demo_benchmark.py`.
2. Start the Next.js dashboard from `frontend/` with `npm run dev`.
3. Start FastAPI only if you will show live search, service readiness, or live index operations. The benchmark page's two report tabs can read their saved files without making a retrieval request.
4. Never run ingestion or a new benchmark as part of the simulated walkthrough.

## Benchmark page flow

1. Open **Benchmark & Demo** at `/benchmark`.
2. Start on **ACTUAL RESULTS**. Point to the label **Measured - 10-query matched-corpus evaluation**, explain that Hit@K/MRR are retrieval metrics and the RAGAS columns are Context Precision/Recall, then call out that this is a matched-corpus pilot rather than official MS MARCO.
3. Select **100-QUERY DEMO**. The persistent **DEMO / SIMULATED** badge and notice explain that the 100-query workload has not been executed.
4. Show the 100 query IDs and three mode datasets, then walk through latency distribution, latency by query, and mode comparison. Change the metric-focus selector to compare Dense, BM25, and Hybrid/RRF summary statistics.
5. Close by distinguishing this visualization from a real performance run. Its generated numbers must not be cited as measured throughput, latency, or production performance.

## Data labels

- **Measured**: sourced from the completed 10-query RAGAS/matched-corpus report.
- **Simulated**: generated from the fixed seed in `scripts/generate_demo_benchmark.py`; each dataset row says `SIMULATED` and `executed=false`.
- **Pending**: no real result is available. The interface uses an explicit empty state instead of filling missing actual metrics with demo values.

The simulated file is `reports/demo/benchmark_100_simulated.json`. It does not replace or update `reports/phase2/benchmarks/benchmark_summary.json`, raw benchmark rows, or RAGAS result files.
