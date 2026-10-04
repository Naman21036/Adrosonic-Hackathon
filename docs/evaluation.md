# Evaluation provenance

## Current measured evaluation

The saved report is `reports/phase2/evaluation/ragas_matched_corpus.json`.

- Status: complete
- Evaluation type: matched indexed corpus
- Queries: 10
- Retrieval modes: Dense, BM25, Hybrid/RRF
- Judge: Azure AI Foundry `gpt-4.1`
- RAGAS: 0.2.15
- Retrieval configuration: top-k 5, RRF-k 60, dense candidates 50, sparse candidates 50
- Query provenance: synthetic queries generated from the active indexed passage text; exact passage text is used as the matched reference
- Official MS MARCO queries: no

| Mode | Context Precision | Context Recall |
| --- | ---: | ---: |
| Dense | 0.9539 | 1.0000 |
| BM25 | 0.6733 | 0.8400 |
| Hybrid/RRF | 0.9543 | 0.9400 |

Retrieval quality metrics (Hit@K and MRR) are a separate measure from RAGAS context metrics. The report preserves per-query scores, judge metadata, retrieval results, and errors. Do not replace missing scores with a value from a simulation.

## Measured, simulated, and pending

- **Measured** means a completed evaluation artifact exists and its source, query count, model/judge, and metrics can be inspected.
- **Simulated** applies only to `reports/demo/benchmark_100_simulated.json`; it is a chart fixture, not an evaluation dataset and not a RAGAS result.
- **Pending** means no completed actual evaluation value exists. Keep the metric unavailable until a real run completes.

The simulated benchmark generator does not create queries/references for RAGAS and must never be passed to the evaluator. It does not overwrite `evaluation_results.json` or `ragas_matched_corpus.json`.

## Existing evaluation commands

`app.evaluate` is the project's evaluation entry point for a prepared evaluation dataset. Its RAGAS requests use configured judge credentials and may send retrieved contexts to that provider. A full evaluation is an intentional API workload; this documentation/demo change does not run it. The already-completed 10-query matched-corpus result is preserved in the report path above.

For dataset preparation and the cached MS MARCO flow, inspect `reports/phase2/evaluation/README.md` and the manifest alongside the dataset. Preserve each dataset's source and classification when reporting scores; matched-corpus synthetic results are not official MS MARCO results.
