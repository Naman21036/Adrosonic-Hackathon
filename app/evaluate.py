"""Evaluate dense, sparse BM25, and hybrid RRF retrieval with RAGAS."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .config import settings
from .embedding import QwenEmbedder
from .observability import correlation_context
from .observability.langfuse import flush_langfuse, observation, trace
from .observability.logging import log_event
from .retrieval import search
from .storage import HybridIndex

MODES = ("dense", "sparse", "hybrid")
EVALUATION_DIR = Path("reports/phase2/evaluation")
OFFICIAL_DATASET = EVALUATION_DIR / "evaluation_dataset.jsonl"
FALLBACK_DATASET = EVALUATION_DIR / "fallback_evaluation_dataset.jsonl"


class RetryCounter(logging.Handler):
    """Count retry attempts made by the OpenAI SDK used by ChatOpenAI."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.retry_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Retrying request to "):
            self.retry_count += 1


def _config() -> dict:
    return {
        "top_k": settings.top_k,
        "rrf_k": settings.rrf_k,
        "dense_candidate_k": settings.dense_candidate_k,
        "sparse_candidate_k": settings.sparse_candidate_k,
    }


def _manifest_for(dataset_path: Path) -> Path:
    if dataset_path.name == FALLBACK_DATASET.name:
        return dataset_path.with_name("fallback_evaluation_manifest.json")
    return dataset_path.with_name("evaluation_dataset_manifest.json")


def _read_rows(dataset_path: Path) -> tuple[list[dict], str | None]:
    try:
        rows = [json.loads(line) for line in dataset_path.open(encoding="utf-8") if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if len(rows) != 100:
        return rows, f"Expected exactly 100 evaluation rows; found {len(rows)}"
    seen_ids: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            return rows, f"Evaluation row {position} is not a JSON object"
        for field in ("query_id", "query", "reference"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                return rows, f"Evaluation row {position} has missing or empty {field}"
        if row["query_id"] in seen_ids:
            return rows, f"Duplicate query_id in evaluation row {position}: {row['query_id']}"
        seen_ids.add(row["query_id"])
    return rows, None


def _select_dataset(explicit: str | None) -> tuple[Path, list[dict], dict, str | None]:
    candidates = [Path(explicit)] if explicit else [OFFICIAL_DATASET, FALLBACK_DATASET]
    errors = []
    for path in candidates:
        if not path.is_file():
            errors.append(f"not found: {path}")
            continue
        rows, error = _read_rows(path)
        if error:
            errors.append(f"{path}: {error}")
            continue
        manifest_path = _manifest_for(path)
        try:
            provenance = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid manifest {manifest_path}: {type(exc).__name__}: {exc}")
            continue
        is_fallback = path.name == FALLBACK_DATASET.name or provenance.get("source") == "synthetic fallback evaluation set"
        return path, rows, provenance, None
    return candidates[0], [], {}, "; ".join(errors) or "No usable evaluation dataset found"


def _prior_benchmark_validation(rows: list[dict], provenance: dict) -> dict | None:
    """Reuse a completed benchmark only when its 100 logged queries match this dataset."""
    summary_path = Path("reports/phase2/benchmarks/benchmark_summary.json")
    raw_path = Path("reports/phase2/benchmarks/benchmark.jsonl")
    source = provenance.get("source", {})
    if not rows or not isinstance(source, dict) or source.get("dataset") != "microsoft/ms_marco":
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict]] = {mode: [] for mode in MODES}
        for line in raw_path.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("mode") in grouped:
                grouped[record["mode"]].append(record)
        for records in grouped.values():
            records.sort(key=lambda record: record.get("query_number", 0))
        expected_queries = [row["query"] for row in rows]
        if summary.get("status") != "complete" or summary.get("query_count") != 100:
            return None
        if any(
            len(grouped[mode]) != 100
            or [record.get("query") for record in grouped[mode]] != expected_queries
            or any(record.get("error") for record in grouped[mode])
            for mode in MODES
        ):
            return None
        return {
            "source_report": str(summary_path),
            "timestamp_utc": summary.get("timestamp_utc"),
            "query_count": 100,
            "same_queries_verified": True,
            "per_mode": {
                mode: {"successful_queries": 100, "failed_queries": 0}
                for mode in MODES
            },
            "note": "Prior completed benchmark for the identical 100 queries; this preflight-only invocation did not rerun retrieval.",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _report_status_path(output_path: Path) -> Path:
    if output_path.name == "evaluation_results.json":
        return output_path.with_name("evaluation_status.json")
    return output_path.with_name(f"{output_path.stem}_status.json")


def _save(report: dict, output_path: Path) -> None:
    report["errors"] = [report["evaluation_error"]] if report.get("evaluation_error") else []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    status_path = _report_status_path(output_path)
    previous = {}
    if status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    update_delete_checks = previous.get("update_delete_checks")
    if update_delete_checks is None:
        checks_path = output_path.with_name("update_delete_checks.json")
        if checks_path.is_file():
            try:
                update_delete_checks = json.loads(checks_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                update_delete_checks = None
    status = {
        "status": report["status"],
        "created_at_utc": report["run_timestamp_utc"],
        "evaluation_command_run_utc": report["run_timestamp_utc"],
        "dataset": report["dataset"],
        "retrieval_config": report["retrieval_config"],
        "retrieval_modes": report["retrieval_modes"],
        "valid_queries": report["valid_query_count"],
        "retrieval_queries_executed": sum(v["retrieval_successes"] for v in report["per_mode_results"].values()),
        "ragas_installed": report["ragas_version"] is not None,
        "ragas_version": report["ragas_version"],
        "dataset_library_version": report["dataset_library_version"],
        "judge": {**report["judge"], "blocked_reason": report.get("evaluation_error")},
        "judge_provider": report["judge_provider"],
        "judge_model": report["judge_model"],
        "judge_enabled": report["judge_enabled"],
        "judge_preflight_attempted": report["judge_preflight_attempted"],
        "judge_preflight_passed": report["judge_preflight_passed"],
        "judge_base_url": report["judge_base_url"],
        "azure_calls": int(bool(report["judge_preflight_attempted"])),
        "retry_count": report["judge"].get("retry_count", 0),
        "context_precision": None,
        "context_recall": None,
        "per_mode_metrics": report["aggregate"],
        "validated_corpus": previous.get("validated_corpus"),
        "errors": [report["evaluation_error"]] if report.get("evaluation_error") else [],
    }
    if update_delete_checks is not None:
        status["update_delete_checks"] = update_delete_checks
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _judge_config() -> tuple[str | None, str | None, str | None, float, str | None, str | None]:
    deployment = settings.azure_openai_deployment
    api_key = settings.azure_openai_api_key
    endpoint = settings.azure_openai_endpoint
    timeout = settings.azure_openai_timeout_seconds
    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        )
        if not value or not value.strip()
    ]
    if missing:
        return deployment, api_key, endpoint, timeout, None, "Missing Azure OpenAI configuration: " + ", ".join(missing)

    parsed = urlsplit(endpoint.strip().rstrip("/"))
    if parsed.scheme != "https" or not parsed.netloc:
        return deployment, api_key, endpoint, timeout, None, "AZURE_OPENAI_ENDPOINT must be a valid HTTPS endpoint URL"
    path = parsed.path.rstrip("/")
    if path.lower().endswith("/openai/v1/responses"):
        path = path[: -len("/responses")]
    elif not path.lower().endswith("/openai/v1"):
        path = f"{path}/openai/v1"
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path + "/", "", ""))
    return deployment, api_key, endpoint, timeout, base_url, None


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _safe_error(exc: Exception, api_token: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if api_token:
        message = message.replace(api_token, "[REDACTED]")
    status = _http_status(exc)
    if status is not None:
        message = f"HTTP {status}: {message}"
    return message


def _evaluation_result_rows(result, samples: list[dict]) -> list[dict]:
    """Normalize RAGAS result rows across EvaluationResult API versions."""
    scores = getattr(result, "scores", None)
    if isinstance(scores, list) and all(isinstance(row, dict) for row in scores):
        score_rows = scores
    else:
        to_pandas = getattr(result, "to_pandas", None)
        if callable(to_pandas):
            score_rows = to_pandas().to_dict(orient="records")
        elif isinstance(result, Mapping):
            score_rows = result.get("scores")
            if not isinstance(score_rows, list):
                score_columns = {
                    name: result.get(name)
                    for name in ("context_precision", "context_recall")
                }
                available_columns = {
                    name: values
                    for name, values in score_columns.items()
                    if isinstance(values, (list, tuple))
                }
                if available_columns and len({len(values) for values in available_columns.values()}) == 1:
                    score_rows = [
                        {
                            name: values[index] if isinstance(values, (list, tuple)) else None
                            for name, values in score_columns.items()
                        }
                        for index in range(len(next(iter(available_columns.values()))))
                    ]
        else:
            score_rows = None

    if not isinstance(score_rows, list) or len(score_rows) != len(samples):
        raise TypeError(
            "Unsupported RAGAS EvaluationResult structure: expected one score row per sample"
        )

    rows = []
    for sample, scores_for_sample in zip(samples, score_rows):
        if not isinstance(scores_for_sample, dict):
            raise TypeError("RAGAS EvaluationResult contains a non-object score row")
        rows.append({**sample, **scores_for_sample})
    return rows


def _aggregate_metric_rows(metric_rows: list[dict]) -> dict[str, float]:
    """Return means only for finite scores actually returned by RAGAS."""
    aggregates = {}
    for metric in ("context_precision", "context_recall"):
        values = [
            row[metric]
            for row in metric_rows
            if isinstance(row.get(metric), (int, float))
            and not isinstance(row[metric], bool)
            and math.isfinite(row[metric])
        ]
        if values:
            aggregates[metric] = math.fsum(values) / len(values)
    return aggregates


def _reuse_cached_retrieval(rows: list[dict]) -> tuple[dict[str, list[dict]], list[dict]] | None:
    """Reuse prior retrieval IDs and load their stored passage text without querying embeddings."""
    cache_path = EVALUATION_DIR / "evaluation_results.json"
    if not cache_path.is_file():
        return None
    try:
        cached_report = json.loads(cache_path.read_text(encoding="utf-8"))
        lookup = {
            (record["query_id"], record["mode"]): record
            for record in cached_report.get("results", [])
            if isinstance(record, dict)
            and record.get("mode") in MODES
            and isinstance(record.get("query_id"), str)
        }
        selected = [
            lookup[(row["query_id"], mode)]
            for mode in MODES
            for row in rows
        ]
        if any(record.get("errors") or len(record.get("passage_ids", [])) != settings.top_k for record in selected):
            return None

        ordered_ids = list(dict.fromkeys(
            passage_id
            for record in selected
            for passage_id in record["passage_ids"]
        ))
        sparse_path = Path(settings.bm25_index_path).resolve()
        connection = sqlite3.connect(sparse_path.as_uri() + "?mode=ro", uri=True)
        try:
            passages = {}
            for start in range(0, len(ordered_ids), 900):
                batch = ordered_ids[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                for passage_id, text in connection.execute(
                    f"SELECT id, text FROM passage WHERE id IN ({placeholders})", batch
                ):
                    passages[passage_id] = text
        finally:
            connection.close()
        if any(passage_id not in passages for passage_id in ordered_ids):
            return None

        queries_for_metrics: dict[str, list[dict]] = {mode: [] for mode in MODES}
        selected_records = []
        for mode in MODES:
            for row in rows:
                record = lookup[(row["query_id"], mode)]
                selected_records.append(record)
                queries_for_metrics[mode].append({
                    "question": row["query"],
                    "contexts": [passages[passage_id] for passage_id in record["passage_ids"]],
                    "ground_truth": row["reference"],
                    "answer": row["reference"],
                    "query_id": row["query_id"],
                })
        return queries_for_metrics, selected_records
    except Exception:
        return None


def main(
    dataset_path: str | None = None,
    output_path: str = "storage/evaluation_results.json",
    preflight_only: bool = False,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset_path, help="Optional explicit evaluation JSONL dataset")
    parser.add_argument("--output", default=output_path)
    parser.add_argument("--preflight-only", action="store_true", help="Run only the Azure judge connectivity preflight")
    parser.add_argument("--limit-queries", type=int, help="Evaluate only the first N rows (for a small validation run)")
    args = parser.parse_args()
    preflight_only = preflight_only or args.preflight_only
    if args.limit_queries is not None and args.limit_queries < 1:
        parser.error("--limit-queries must be at least 1")
    output = Path(args.output)
    selected_path, rows, provenance, dataset_error = _select_dataset(args.dataset)
    query_limit = args.limit_queries or settings.evaluation_queries
    if not dataset_error:
        rows = rows[:query_limit]
    deployment, api_token, endpoint, timeout, base_url, config_error = _judge_config()
    judge_enabled = bool(api_token and endpoint and deployment and not config_error)
    source = provenance.get("source", {}) if isinstance(provenance, dict) else {}
    fallback = (
        selected_path.name == FALLBACK_DATASET.name
        or (isinstance(source, dict) and source.get("source") == "synthetic fallback evaluation set")
    )
    report = {
        "status": "blocked",
        "dataset": {
            "path": str(selected_path),
            "source": source.get("source", source.get("dataset", "unknown")) if isinstance(source, dict) else "unknown",
            "classification": "fallback_synthetic" if fallback else provenance.get("classification", "official_or_cached"),
            "official_ms_marco": bool(not fallback and source.get("dataset") == "microsoft/ms_marco"),
            "provenance": provenance,
        },
        "query_count": len(rows),
        "valid_query_count": len(rows) if not dataset_error else 0,
        "prior_retrieval_validation": _prior_benchmark_validation(rows, provenance),
        "retrieval_config": _config(),
        "retrieval_modes": list(MODES),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.hf_model_name,
        "retrieval_mode": ",".join(MODES),
        "top_k": settings.top_k,
        "evaluation_queries": len(rows),
        "llm_provider": "azure_ai_foundry",
        "azure_deployment": deployment,
        "metrics": {},
        "ragas_version": None,
        "dataset_library_version": None,
        "judge_provider": "azure_ai_foundry",
        "judge_model": deployment,
        "judge_enabled": judge_enabled,
        "judge_preflight_attempted": False,
        "judge_preflight_passed": False,
        "judge_base_url": base_url,
        "judge": {
            "provider": "azure_ai_foundry",
            "model": deployment,
            "deployment": deployment,
            "enabled": judge_enabled,
            "base_url": base_url,
            "request_made": False,
            "blocked_reason": config_error,
            "retry_count": 0,
            "preflight": {
                "attempted": False,
                "passed": False,
                "http_status": None,
                "http_status_note": None,
                "error": None,
            },
        },
        "per_mode_results": {
            mode: {"retrieval_successes": 0, "retrieval_failures": 0, "context_precision": None, "context_recall": None}
            for mode in MODES
        },
        "results": [],
        "retrieval_results": [],
        "per_query_metrics": {},
        "aggregate": {},
        "evaluation_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(selected_path),
            "dataset_provenance": provenance,
            "configured_query_count": settings.evaluation_queries,
            "effective_query_count": len(rows),
        },
        "errors": [],
        "evaluation_error": dataset_error,
    }
    for package, field in (("ragas", "ragas_version"), ("datasets", "dataset_library_version")):
        try:
            from importlib.metadata import version
            report[field] = version(package)
        except PackageNotFoundError:
            report[field] = None

    if dataset_error:
        _save(report, output)
        return

    if config_error:
        report["judge"]["blocked_reason"] = config_error
        report["evaluation_error"] = config_error
        print(f"Azure preflight not attempted: {config_error}")
        _save(report, output)
        return

    retry_counter = RetryCounter()
    openai_logger = logging.getLogger("openai._base_client")
    openai_logger.addHandler(retry_counter)
    try:
        from langchain_openai import ChatOpenAI

        judge_chat = ChatOpenAI(
            model=deployment,
            api_key=api_token,
            base_url=base_url,
            timeout=timeout,
            temperature=0,
            max_tokens=settings.azure_openai_max_tokens,
            max_retries=3,
        )
        report["judge_preflight_attempted"] = True
        report["judge"]["preflight"]["attempted"] = True
        report["judge"]["request_made"] = True
        try:
            test_response = judge_chat.invoke("Reply only: OK")
            test_text = getattr(test_response, "content", "")
            if not isinstance(test_text, str) or not test_text.strip():
                raise RuntimeError("Azure preflight returned an empty response")
            metadata = getattr(test_response, "response_metadata", {}) or {}
            status = metadata.get("status_code")
            report["judge"]["preflight"]["http_status"] = int(status) if status else 200
            report["judge"]["preflight"]["passed"] = True
            report["judge_preflight_passed"] = True
        except Exception as exc:
            error = _safe_error(exc, api_token)
            status = _http_status(exc)
            report["judge"]["preflight"]["http_status"] = status
            if status is None:
                report["judge"]["preflight"]["http_status_note"] = "No HTTP response was received; connection failed before a status was available."
            report["judge"]["preflight"]["error"] = error
            report["judge"]["blocked_reason"] = f"Azure preflight failed: {error}"
            report["evaluation_error"] = report["judge"]["blocked_reason"]
            report["judge"]["retry_count"] = retry_counter.retry_count
            print(f"Azure preflight failed: {error}")
            _save(report, output)
            openai_logger.removeHandler(retry_counter)
            return

        if preflight_only:
            report["status"] = "preflight_passed"
            report["evaluation_error"] = None
            report["judge"]["blocked_reason"] = None
            _save(report, output)
            openai_logger.removeHandler(retry_counter)
            return

        import ragas
        import datasets as datasets_package
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import context_precision, context_recall
        from ragas.run_config import RunConfig

        report["ragas_version"] = getattr(ragas, "__version__", report["ragas_version"])
        report["dataset_library_version"] = getattr(datasets_package, "__version__", report["dataset_library_version"])
        judge = LangchainLLMWrapper(judge_chat)
        run_config = RunConfig(timeout=timeout, max_retries=1, max_wait=60, max_workers=1, log_tenacity=True)
    except Exception as exc:
        report["judge"]["blocked_reason"] = f"Could not initialize Azure RAGAS judge: {_safe_error(exc, api_token)}"
        report["evaluation_error"] = report["judge"]["blocked_reason"]
        report["judge"]["retry_count"] = retry_counter.retry_count
        _save(report, output)
        return

    queries_for_metrics: dict[str, list[dict]] = {mode: [] for mode in MODES}
    cached = _reuse_cached_retrieval(rows)
    if cached is not None:
        queries_for_metrics, report["results"] = cached
        report["retrieval_cache_reused"] = True
        for mode in MODES:
            report["per_mode_results"][mode]["retrieval_successes"] = len(queries_for_metrics[mode])
    else:
        report["retrieval_cache_reused"] = False
        index = None
        try:
            index = HybridIndex(QwenEmbedder(
                settings.hf_model_name,
                settings.hf_model_revision,
                settings.device,
                settings.huggingface_api_key,
            ))
            for mode in MODES:
                for row in rows:
                    record = {"query_id": row["query_id"], "mode": mode, "passage_ids": [], "errors": []}
                    try:
                        if mode == "sparse":
                            hits = index.sparse_search(row["query"], settings.sparse_candidate_k)[:settings.top_k]
                        else:
                            hits = search(index, row["query"], mode, settings.top_k)["results"]
                        if not hits:
                            raise RuntimeError("Retrieval returned no passages")
                        record["passage_ids"] = [hit["id"] for hit in hits]
                        report["per_mode_results"][mode]["retrieval_successes"] += 1
                        queries_for_metrics[mode].append({
                            "question": row["query"],
                            "contexts": [hit["text"] for hit in hits],
                            "ground_truth": row["reference"],
                            "answer": row["reference"],
                            "query_id": row["query_id"],
                        })
                    except Exception as exc:
                        record["errors"].append(f"{type(exc).__name__}: {exc}")
                        report["per_mode_results"][mode]["retrieval_failures"] += 1
                    report["results"].append(record)
        except Exception as exc:
            report["evaluation_error"] = f"Retrieval initialization failed: {type(exc).__name__}: {exc}"
        finally:
            if index is not None:
                index.close()
    report["retrieval_results"] = report["results"]

    if report["evaluation_error"]:
        report["status"] = "partial" if report["results"] else "blocked"
        report["judge"]["retry_count"] = retry_counter.retry_count
        _save(report, output)
        openai_logger.removeHandler(retry_counter)
        return

    evaluation_id = uuid.uuid4().hex
    evaluation_started = time.perf_counter()
    log_event(
        "RAGAS_EVALUATION_STARTED",
        evaluation_id=evaluation_id,
        query_count=len(rows),
        modes=list(MODES),
        ragas_version=report.get("ragas_version"),
        judge_provider=report.get("judge_provider"),
        judge_model=report.get("judge_model"),
    )
    try:
        with trace(
            "ragas_evaluation",
            input={"evaluation_id":evaluation_id,"query_count":len(rows),"modes":list(MODES)},
            metadata={
                "evaluation_id":evaluation_id,
                "ragas_version":report.get("ragas_version"),
                "judge_provider":report.get("judge_provider"),
                "judge_model":report.get("judge_model"),
                "dataset_source":report["dataset"].get("source"),
            },
        ) as evaluation_observation:
            for mode in MODES:
                samples = queries_for_metrics[mode]
                if not samples:
                    report["per_mode_results"][mode]["metric_error"] = "No successful retrievals for this mode"
                    continue
                mode_started=time.perf_counter()
                log_event("RAGAS_MODE_EVALUATION_STARTED",evaluation_id=evaluation_id,mode=mode,query_count=len(samples))
                with correlation_context(evaluation_id=evaluation_id,retrieval_mode=mode):
                    with observation(
                        "ragas_mode_evaluation",
                        input={"evaluation_id":evaluation_id,"mode":mode,"query_count":len(samples)},
                        metadata={"ragas_version":report.get("ragas_version"),"judge_provider":report.get("judge_provider"),"judge_model":report.get("judge_model")},
                    ) as mode_observation:
                        result = evaluate(
                            Dataset.from_list(samples),
                            metrics=[context_precision, context_recall],
                            llm=judge,
                            run_config=run_config,
                        )
                        metric_rows = _evaluation_result_rows(result, samples)
                        metric_successes = sum(
                            isinstance(item.get("context_precision"), (int, float))
                            and not isinstance(item.get("context_precision"), bool)
                            and math.isfinite(item["context_precision"])
                            and isinstance(item.get("context_recall"), (int, float))
                            and not isinstance(item.get("context_recall"), bool)
                            and math.isfinite(item["context_recall"])
                            for item in metric_rows
                        )
                        report["per_mode_results"][mode]["evaluation_successes"] = metric_successes
                        report["per_mode_results"][mode]["evaluation_failures"] = len(samples) - metric_successes
                        values = _aggregate_metric_rows(metric_rows)
                        report["aggregate"][mode] = values
                        report["per_mode_results"][mode].update(values)
                        report["per_query_metrics"][mode] = metric_rows
                        if mode == "hybrid":
                            report["metrics"] = values.copy()
                        mode_latency=(time.perf_counter()-mode_started)*1000
                        mode_observation.update(
                            output={**values,"successful_queries":metric_successes,"failed_queries":len(samples)-metric_successes},
                            metadata={"latency_ms":mode_latency},
                        )
                        for metric_row in metric_rows:
                            query_id=str(metric_row.get("query_id", "unknown"))
                            query_success=all(
                                isinstance(metric_row.get(name),(int,float))
                                and not isinstance(metric_row.get(name),bool)
                                and math.isfinite(metric_row[name])
                                for name in ("context_precision","context_recall")
                            )
                            with correlation_context(query_id=query_id):
                                with observation(
                                    "ragas_query_evaluation",
                                    input={"query_id":query_id,"mode":mode},
                                    metadata={"evaluation_id":evaluation_id,"status":"complete" if query_success else "failed"},
                                ) as query_observation:
                                    query_observation.update(
                                        output={name:metric_row.get(name) for name in ("context_precision","context_recall")},
                                        metadata={"status":"complete" if query_success else "failed"},
                                    )
                        log_event(
                            "RAGAS_MODE_EVALUATION_COMPLETED",
                            evaluation_id=evaluation_id,
                            mode=mode,
                            query_count=len(samples),
                            successful_queries=metric_successes,
                            failed_queries=len(samples)-metric_successes,
                            evaluation_latency_ms=mode_latency,
                            context_precision=values.get("context_precision"),
                            context_recall=values.get("context_recall"),
                        )
                        report["judge"]["retry_count"] = retry_counter.retry_count
                        _save(report, output)
            evaluation_latency=(time.perf_counter()-evaluation_started)*1000
            evaluation_observation.update(
                output={"aggregate":report.get("aggregate",{}),"evaluation_id":evaluation_id},
                metadata={"evaluation_latency_ms":evaluation_latency},
            )
    except Exception as exc:
        report["evaluation_error"] = f"Azure RAGAS request failed: {_safe_error(exc, api_token)}"
        report["judge"]["blocked_reason"] = report["evaluation_error"]
        log_event(
            "RAGAS_EVALUATION_FAILED",
            level="ERROR",
            evaluation_id=evaluation_id,
            evaluation_latency_ms=(time.perf_counter()-evaluation_started)*1000,
            error_type=type(exc).__name__,
        )
        failed_mode = next((mode for mode in MODES if mode not in report["aggregate"]), None)
        if failed_mode is not None:
            report["per_mode_results"][failed_mode]["evaluation_successes"] = 0
            report["per_mode_results"][failed_mode]["evaluation_failures"] = len(rows)

    report["judge"]["retry_count"] = retry_counter.retry_count
    openai_logger.removeHandler(retry_counter)
    if not report["evaluation_error"]:
        log_event(
            "RAGAS_EVALUATION_COMPLETED",
            evaluation_id=evaluation_id,
            query_count=len(rows),
            evaluation_latency_ms=(time.perf_counter()-evaluation_started)*1000,
            modes=list(MODES),
        )
    all_modes_have_metrics = all(
        isinstance(report["aggregate"].get(mode), dict)
        and "context_precision" in report["aggregate"][mode]
        and "context_recall" in report["aggregate"][mode]
        for mode in MODES
    )
    report["status"] = "complete" if all_modes_have_metrics and not report["evaluation_error"] else "partial"
    _save(report, output)
    flush_langfuse()


if __name__ == "__main__":
    main()
