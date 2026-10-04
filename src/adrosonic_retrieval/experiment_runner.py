"""Run a controlled, resumable screening plan into isolated collections."""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from .benchmark import run_benchmark
from .config import Settings, get_settings
from .indexer import run_index
from .evaluate import evaluate

log = logging.getLogger(__name__)


def run_experiments(plan_path: Path, selected: set[str], queries: int, evaluate_quality: bool = False) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = get_settings()
    experiments = plan["experiments"]
    chosen = [item for item in experiments if not selected or item["id"] in selected]
    unknown = selected - {item["id"] for item in experiments}
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {', '.join(sorted(unknown))}")
    records = []
    for item in chosen:
        experiment_id = item["id"]
        values = base.model_dump()
        values.update({key: value for key, value in item.items() if key in {
            "embedding_model", "embedding_dimension", "embedding_precision", "batch_size",
            "data_limit", "hnsw_m", "hnsw_ef_construct", "ef_search"}})
        values["collection_name"] = item["collection_name"]
        values["output_dir"] = base.output_dir / "experiments" / experiment_id
        cfg = Settings(**values)
        record = {"id": experiment_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "configuration": {key: getattr(cfg, key) for key in (
                      "embedding_model", "embedding_dimension", "embedding_precision", "batch_size",
                      "data_limit", "hnsw_m", "hnsw_ef_construct", "ef_search", "collection_name")},
                  "status": "running"}
        log.info("Starting experiment %s", experiment_id)
        try:
            record["indexing"] = run_index(cfg)
            record["benchmark"] = run_benchmark(queries, cfg=cfg, output_name=experiment_id)
            if evaluate_quality:
                record["evaluation"] = asyncio.run(evaluate(20, cfg=cfg))
            record["status"] = "complete"
        except Exception as exc:
            # Keep secrets and potentially sensitive Hub error strings out of experiment logs.
            record.update({"status": "failed", "error_type": type(exc).__name__})
            log.error("Experiment %s failed (%s)", experiment_id, type(exc).__name__)
        records.append(record)
    base.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = base.output_dir / f"experiment_results_{stamp}.json"
    result_path.write_text(json.dumps({"plan": str(plan_path), "results": records}, indent=2, default=str), encoding="utf-8")
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/phase1_screening.json"))
    parser.add_argument("--experiment", action="append", default=[], help="Experiment ID to run; repeat to select multiple")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--evaluate", action="store_true", help="Also run RAGAS on 20 labeled queries")
    parser.add_argument("--list", action="store_true", help="List plan entries without running them")
    parser.add_argument("--run", action="store_true", help="Execute indexing and benchmark; downloads data/model")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.list:
        for item in plan["experiments"]:
            print(f"{item['id']}: {item.get('description', '')}")
    elif not args.run:
        parser.error("Pass --list to inspect the plan or --run to execute experiments")
    else:
        logging.basicConfig(level=get_settings().log_level.upper())
        print(run_experiments(args.plan, set(args.experiment), args.queries, args.evaluate))
