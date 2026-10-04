"""Generate the deterministic, non-executed benchmark dashboard fixture."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demo" / "benchmark_100_simulated.json"
SEED = 20261004
GENERATION_VERSION = "demo-benchmark-v1"
MODES = ("dense", "bm25", "hybrid")
BUCKETS = (
    (0, 50, "0–49 ms"),
    (50, 100, "50–99 ms"),
    (100, 150, "100–149 ms"),
    (150, 200, "150–199 ms"),
    (200, 300, "200–299 ms"),
    (300, 450, "300–449 ms"),
    (450, 650, "450–649 ms"),
    (650, None, "650+ ms"),
)


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for the sorted observations."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stage_times(mode: str, query_number: int) -> dict[str, float]:
    mode_index = MODES.index(mode)
    rng = random.Random(SEED + mode_index * 10_000 + query_number)

    if mode == "dense":
        values = {
            "embedding_latency_ms": max(35.0, rng.gauss(116.0, 28.0)),
            "dense_latency_ms": max(18.0, rng.gauss(73.0, 19.0)),
            "bm25_latency_ms": 0.0,
            "rrf_latency_ms": 0.0,
        }
    elif mode == "bm25":
        values = {
            "embedding_latency_ms": 0.0,
            "dense_latency_ms": 0.0,
            "bm25_latency_ms": max(5.0, rng.lognormvariate(3.25, 0.42)),
            "rrf_latency_ms": 0.0,
        }
    else:
        values = {
            "embedding_latency_ms": max(35.0, rng.gauss(116.0, 28.0)),
            "dense_latency_ms": max(18.0, rng.gauss(73.0, 19.0)),
            "bm25_latency_ms": max(5.0, rng.lognormvariate(3.25, 0.42)),
            "rrf_latency_ms": max(0.2, rng.gauss(2.8, 1.0)),
        }

    # A fixed set of synthetic tail-latency events keeps the charts varied.
    if mode != "bm25" and query_number % 23 == 0:
        values["embedding_latency_ms"] *= 2.2
    if mode == "bm25" and query_number % 19 == 0:
        values["bm25_latency_ms"] *= 2.8
    if mode == "hybrid" and query_number % 31 == 0:
        values["bm25_latency_ms"] *= 2.4

    return {key: round(value, 3) for key, value in values.items()}


def build_records(mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query_number in range(1, 101):
        stages = stage_times(mode, query_number)
        total = round(sum(stages.values()), 3)
        records.append(
            {
                "query_id": f"demo_{query_number:03d}",
                "mode": mode,
                "latency_ms": total,
                **stages,
                "total_latency_ms": total,
                "status": "SIMULATED",
                "executed": False,
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [record["latency_ms"] for record in records]
    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
    }


def distribution(records: list[dict[str, Any]]) -> list[dict[str, int | float | str | None]]:
    values = [record["latency_ms"] for record in records]
    return [
        {
            "label": label,
            "min_ms": low,
            "max_ms": high,
            "count": sum(value >= low and (high is None or value < high) for value in values),
        }
        for low, high, label in BUCKETS
    ]


def main() -> None:
    datasets = {mode: build_records(mode) for mode in MODES}
    report = {
        "source": "DEMO / SIMULATED",
        "status": "SIMULATED",
        "executed": False,
        "generation_version": GENERATION_VERSION,
        "random_seed": SEED,
        "query_count": 100,
        "mode_count": len(MODES),
        "unique_query_ids": 100,
        "records_per_mode": 100,
        "mode_record_count": sum(len(records) for records in datasets.values()),
        "modes": list(MODES),
        "datasets": datasets,
        "summary": {mode: summarize(datasets[mode]) for mode in MODES},
        "latency_distribution": {mode: distribution(datasets[mode]) for mode in MODES},
        "notice": "Synthetic visualization only. No benchmark queries were executed and these values are not measured system performance.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote deterministic demo fixture: {OUTPUT.relative_to(ROOT)}")
    print("query_count=100 unique_query_ids=100 records_per_mode=100 mode_record_count=300")


if __name__ == "__main__":
    main()
