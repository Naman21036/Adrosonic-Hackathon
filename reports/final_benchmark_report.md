# Final Benchmark and Demo Report

## Actual Measured Results

These results are from the completed 10-query matched-corpus evaluation stored at `reports/phase2/evaluation/ragas_matched_corpus.json`. The matched queries were generated from the active indexed passage text and are **not official MS MARCO queries**. RAGAS 0.2.15 used the configured Azure AI Foundry `gpt-4.1` judge.

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR | Context Precision | Context Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.70 | 1.00 | 1.00 | 0.833 | 0.9539 | 1.0000 |
| BM25 | 0.40 | 0.80 | 0.90 | 0.582 | 0.6733 | 0.8400 |
| Hybrid/RRF | 0.70 | 0.90 | 1.00 | 0.814 | 0.9543 | 0.9400 |

The retrieval configuration was top-k 5, RRF-k 60, 50 dense candidates, and 50 sparse candidates. Per-query RAGAS values and evaluation provenance remain in the source report.

## 100-Query Demonstration

The frontend contains a deterministic simulated visualization for 100 query IDs across Dense, BM25, and Hybrid/RRF. It shows synthetic stage and total latencies, a distribution chart, query-by-query lines, and a mode comparison. The JSON is generated with a fixed random seed by `scripts/generate_demo_benchmark.py` and stored at `reports/demo/benchmark_100_simulated.json`.

Every mode row carries `status: "SIMULATED"` and `executed: false`. The demo tab has a persistent **DEMO / SIMULATED** badge and states that the 100-query workload has not been executed. Its values are not measured system performance and are excluded from actual benchmark, evaluation, backend health, and retrieval reports.

## Limitations

For the workload represented by this demo dataset: “The 100-query benchmark workload was not executed during development; the 100-query dashboard is a deterministic simulation used to demonstrate the benchmark interface.”

That limitation describes the new demo fixture and the visualization it feeds. Separate pre-existing benchmark/evaluation artifacts were not read to seed these simulated values and were left untouched. Do not use simulated percentiles for capacity planning, latency claims, or production comparisons.
