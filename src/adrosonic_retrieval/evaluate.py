"""RAGAS evaluation against labeled MS MARCO v1.1 validation contexts."""
import asyncio
import json
import random
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from .config import get_settings
from .dataset import load_eval_queries
from .embeddings import Embedder
from .store import PassageStore
from .retrieval import RetrievalService


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalize_context(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def indexed_reference_contexts(store: PassageStore, rows: list[dict]) -> set[str]:
    """Find exact labeled contexts present in the current Qdrant collection."""
    remaining = {
        normalize_context(text)
        for row in rows
        for text in row["reference_contexts"]
    }
    indexed: set[str] = set()
    offset = None
    while remaining:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            limit=2048,
            offset=offset,
            with_payload=["text"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            if isinstance(text, str):
                normalized = normalize_context(text)
                if normalized in remaining:
                    indexed.add(normalized)
                    remaining.remove(normalized)
        if offset is None:
            break
    return indexed


def select_evaluable_rows(rows: list[dict], indexed_references: set[str], count: int) -> list[dict]:
    """Keep queries with at least one reference in the index, trimming unavailable refs."""
    eligible = []
    for row in rows:
        references = [
            text for text in row["reference_contexts"]
            if normalize_context(text) in indexed_references
        ]
        if references:
            eligible.append({**row, "reference_contexts": references})
    if len(eligible) > count:
        eligible = random.Random(42).sample(eligible, count)
    return eligible


async def evaluate(count: int = 20, cfg=None) -> dict:
    cfg = cfg or get_settings()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset": cfg.eval_dataset_id,
        "config": cfg.eval_dataset_config, "split": cfg.eval_dataset_split, "requested_query_count": count,
        "embedding_model": cfg.embedding_model, "top_k": cfg.top_k,
        "packages": {"ragas": package_version("ragas"), "datasets": package_version("datasets"),
                     "sentence-transformers": package_version("sentence-transformers")},
        "metrics": {"context_precision": None, "context_recall": None}}
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate as ragas_evaluate
        from ragas.metrics import NonLLMContextPrecisionWithReference

        validation_rows = load_eval_queries(
            cfg.eval_dataset_id, cfg.eval_dataset_config, cfg.eval_dataset_split,
            None, cfg.hf_token_value)
        if not validation_rows:
            raise RuntimeError("No labeled validation queries with answers and selected contexts were found")
        store = PassageStore(cfg.qdrant_path, cfg.collection_name)
        try:
            point_count = store.count()
            indexed_references = indexed_reference_contexts(store, validation_rows)
        finally:
            store.close()
        rows = select_evaluable_rows(validation_rows, indexed_references, count)
        if not rows:
            raise RuntimeError(
                "None of the selected validation passages are present in the indexed collection; "
                "cannot compute reference-based retrieval metrics without reindexing or different labels"
            )
        total_references = {
            normalize_context(text)
            for row in validation_rows
            for text in row["reference_contexts"]
        }
        report["evaluation_scope"] = {
            "collection": cfg.collection_name,
            "indexed_passage_count": point_count,
            "validation_queries_checked": len(validation_rows),
            "queries_with_indexed_references": len([
                row for row in validation_rows
                if any(normalize_context(text) in indexed_references for text in row["reference_contexts"])
            ]),
            "selected_reference_contexts_in_index": len(indexed_references),
            "selected_reference_contexts_checked": len(total_references),
            "interpretation": (
                "Metrics use only queries with at least one exact normalized selected reference passage "
                "present in this collection; this is a partial-index, filtered evaluation, not a full-corpus score."
            ),
        }
        embedder = Embedder(cfg.embedding_model, cfg.device, dimension=cfg.embedding_dimension,
            precision=cfg.embedding_precision, query_instruction=cfg.query_instruction,
            token=cfg.hf_token_value, revision=cfg.model_revision)
        report.update({"model_revision": embedder.model_revision, "embedding_dimension": embedder.dimension,
                       "precision": embedder.precision, "device_requested": cfg.device,
                       "device_actual": embedder.device})
        store = PassageStore(cfg.qdrant_path, cfg.collection_name)
        service = RetrievalService(embedder, store, cfg.top_k, cfg.ef_search)
        samples = []
        retrieved_by_query: list[list[str]] = []
        try:
            for row in rows:
                found = service.search(row["query"])["results"]
                retrieved_by_query.append([item["text"] for item in found])
                samples.append(SingleTurnSample(user_input=row["query"], response=row["reference_answer"],
                    reference=row["reference_answer"], reference_contexts=row["reference_contexts"],
                    retrieved_contexts=[item["text"] for item in found]))
        finally:
            store.close()

        # RAGAS computes reference-context precision without an LLM judge.
        # Recall needs semantic judging or aligned passage-ID qrels, neither configured here.
        metric = NonLLMContextPrecisionWithReference()
        result = ragas_evaluate(EvaluationDataset(samples=samples), metrics=[metric])
        frame = result.to_pandas()
        metric_column = getattr(metric, "name", "non_llm_context_precision_with_reference")
        if metric_column not in frame.columns:
            candidates = [column for column in frame.columns if "context_precision" in column]
            if not candidates:
                raise RuntimeError(f"RAGAS result has no context precision column: {list(frame.columns)}")
            metric_column = candidates[0]
        scores = frame[metric_column].dropna().tolist()
        recall_values, reciprocal_ranks = [], []
        for row, retrieved in zip(rows, retrieved_by_query):
            relevant = {normalize_context(text) for text in row["reference_contexts"]}
            ranked = [normalize_context(text) for text in retrieved]
            hits = [rank for rank, text in enumerate(ranked, 1) if text in relevant]
            recall_values.append(len(set(ranked) & relevant) / len(relevant) if relevant else 0.0)
            reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
        report.update({"status": "complete" if len(rows) >= count else "partial", "query_count": len(rows),
            "metrics": {"context_precision": sum(scores) / len(scores) if scores else None,
                        "context_recall": None},
            "context_precision_per_query": scores,
            "retrieval_diagnostics": {"recall_at_5": sum(recall_values) / len(recall_values),
                "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "judgment": "exact normalized text match against MS MARCO is_selected reference passages; reported separately from RAGAS"},
            "precision_method": "RAGAS NonLLMContextPrecisionWithReference compares retrieved passages to the selected MS MARCO reference passage texts.",
            "recall_unavailable_reason": "No LLM judge or aligned passage-ID qrels are configured for semantic reference coverage."})
    except Exception as exc:
        detail = str(exc)
        if cfg.hf_token_value:
            detail = detail.replace(cfg.hf_token_value, "[REDACTED]")
        report.update({"status": "unavailable", "failure": f"{type(exc).__name__}: {detail}"})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate(count=29)), indent=2))
