# Retrieval latency report

**Status: NOT RUN.** No indexed corpus or 100-query benchmark log exists yet.

| Mode | Queries | Failures | Mean | p50 | p95 | p99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0 | — | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Hybrid | 0 | — | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

Run `python -m app.benchmark --queries 100`. The command saves every result, timeout, and failure to `artifacts/benchmark.jsonl`, then writes actual statistics to `artifacts/benchmark_summary.json`. The p95 target of 300 ms is unverified.
