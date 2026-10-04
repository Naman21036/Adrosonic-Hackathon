# RAGAS evaluation

**Status: unavailable.** The evaluator was invoked and wrote `artifacts/evaluation_report.json`. It failed before loading data with the exact error `ModuleNotFoundError: No module named 'ragas'`. The optional evaluation dependencies, model, validation data, and indexed collection are not locally available. Both metric values are `null`; no score is reported.

The evaluator requires 20 examples with query, answer, and selected reference passage contexts. It records the RAGAS version, configuration, model, timestamp, and Context Precision output in `artifacts/evaluation_report.json`. The offline path uses RAGAS `NonLLMContextPrecisionWithReference`; Context Recall is marked unavailable because this configuration does not use an LLM judge and has no matching passage-ID qrels for the selected MS MARCO v1.1 context set. No paid API or LLM is required for dense search.
