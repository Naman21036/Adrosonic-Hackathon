# Benchmark Report

## Summary
- Date (UTC): 2026-10-03T09:52:44.711718Z
- Queries: 100
- Run type: warm
- Target: p95 < 300 ms
- Result: not met

## Latency (ms)
| Metric | Value |
| --- | ---: |
| p50 | 219.99 |
| p95 | 305.64 |
| p99 | 516.80 |
| Mean | 235.31 |
| Min | 199.43 |
| Max | 572.08 |

## Configuration
- Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Precision: fp32
- Batch size: 32
- Top-k: 5
- Hardware: Windows-11-10.0.26200-SP0, CPU

## Notes
- Benchmark measures in-process service latency: query encoding + Qdrant search + Python result conversion.
- HTTP transport and JSON serialization are excluded.
- Model initialization occurs before the timed section.
