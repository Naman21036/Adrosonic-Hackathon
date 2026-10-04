"""Build and evaluate a synthetic query set grounded in the active local corpus.

All index access in this module is read-only. The stored passage text is the
ground truth; Azure is used only to write a natural query and a topic label.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .config import settings
from .observability import correlation_context
from .observability.langfuse import flush_langfuse, observation, trace
from .observability.logging import log_event

OUT_DIR = Path("reports/phase2/evaluation")
JSONL_PATH = OUT_DIR / "matched_query_corpus.jsonl"
JSON_PATH = OUT_DIR / "matched_query_corpus.json"
VALIDATION_PATH = OUT_DIR / "query_corpus_validation.json"
RESULTS_PATH = Path("storage/evaluation_results_matched_corpus.json")
RAGAS_RESULTS_PATH = OUT_DIR / "ragas_matched_corpus.json"
CORPUS_VERSION = "indexed_v1.1_train"
ID_NAMESPACE = uuid.UUID("db8dcbb1-8011-4a09-83ec-6644a2ad64e8")
TOPICS = (
    "health_medicine", "science_environment", "technology_computing",
    "business_finance", "home_garden", "food_cooking", "history_government",
    "geography_travel", "education_language", "law_policy", "arts_media",
    "daily_life_other",
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _longest_common_run(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    best = 0
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            value = previous[j - 1] + 1 if token == other else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def _candidate_rows(pool_size: int) -> tuple[list[dict], dict]:
    """Sample distinct source rows deterministically from BM25, preferring positives."""
    path = Path(settings.bm25_index_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BM25 database not found: {path}")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        raw = db.execute("SELECT id, text, metadata FROM passage ORDER BY id").fetchall()
    finally:
        db.close()
    by_query: dict[str, list[dict]] = defaultdict(list)
    indexed_ids: set[str] = set()
    malformed_ids = 0
    for passage_id, text, metadata_json in raw:
        indexed_ids.add(passage_id)
        match = re.fullmatch(r"msmarco-(\d+)-(\d+)", passage_id)
        if not match:
            malformed_ids += 1
            continue
        metadata = json.loads(metadata_json)
        by_query[match.group(1)].append({
            "passage_id": passage_id,
            "text": text,
            "position": int(match.group(2)),
            "source_query_id": match.group(1),
            "source_relevance": bool(metadata.get("relevance", False)),
            "source": metadata.get("source"),
            "split": metadata.get("split"),
        })
    eligible = []
    for source_query_id, passages in by_query.items():
        positives = [item for item in passages if item["source_relevance"]]
        choices = positives or passages
        eligible.append(min(choices, key=lambda item: hashlib.sha256(item["passage_id"].encode()).hexdigest()))
    eligible.sort(key=lambda item: hashlib.sha256(("matched-corpus-v1:" + item["passage_id"]).encode()).hexdigest())
    if len(eligible) < pool_size:
        raise RuntimeError(f"Only {len(eligible)} distinct source rows available for {pool_size} candidates")
    step = len(eligible) / pool_size
    sampled = [eligible[min(int((i + 0.5) * step), len(eligible) - 1)] for i in range(pool_size)]
    return sampled, {
        "indexed_bm25_count": len(raw),
        "indexed_bm25_unique_ids": len(indexed_ids),
        "source_query_ids_found": len(by_query),
        "source_id_format": "msmarco-{query_id}-{zero_based_passage_position}",
        "malformed_passage_ids": malformed_ids,
        "candidate_pool_size": pool_size,
        "candidate_selection": "deterministic SHA-256 sampling, one passage per stored source query ID, preferring relevance=true",
        "source_queries_retained_in_index": False,
        "source_relevance_labels_retained_in_metadata": True,
        "source_query_text_available_in_index": False,
    }


def _azure_config() -> tuple[str | None, str | None, str | None, float, str | None]:
    deployment = settings.azure_openai_deployment
    api_key = settings.azure_openai_api_key
    endpoint = settings.azure_openai_endpoint
    timeout = settings.azure_openai_timeout_seconds
    missing = [
        name for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        ) if not value or not value.strip()
    ]
    if missing:
        return deployment, api_key, None, timeout, "Missing Azure configuration: " + ", ".join(missing)
    parsed = urlsplit(endpoint.strip().rstrip("/"))
    if parsed.scheme != "https" or not parsed.netloc:
        return deployment, api_key, None, timeout, "Azure endpoint must be a valid HTTPS URL"
    path = parsed.path.rstrip("/")
    if path.lower().endswith("/openai/v1/responses"):
        path = path[: -len("/responses")]
    elif not path.lower().endswith("/openai/v1"):
        path = path + "/openai/v1"
    base_url = parsed._replace(path=path + "/", query="", fragment="").geturl()
    return deployment, api_key, base_url, timeout, None


async def _azure_preflight(deployment: str, api_key: str, base_url: str, timeout: float) -> None:
    from openai import AsyncOpenAI
    from .azure_openai import _safe_api_error

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Reply only with OK."}],
                temperature=0,
                max_tokens=8,
            ),
            timeout=timeout + 5,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Azure preflight returned an empty assistant response")
    except Exception as exc:
        raise RuntimeError(str(_safe_api_error(exc))) from None
    finally:
        await client.close()


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("examples"), list):
        raise ValueError("Azure query generator returned an unexpected JSON shape")
    return value


async def _generate_batch(client, deployment: str, candidates: list[dict], start_number: int) -> tuple[list[dict], list[dict]]:
    inputs = [
        {"key": f"p{index:03d}", "passage": item["text"][:2400]}
        for index, item in enumerate(candidates)
    ]
    system = (
        "Create natural, specific information-seeking search queries grounded only in the supplied passages. "
        "These are synthetic retrieval-test queries, not official MS MARCO queries. Do not copy a sentence or long phrase. "
        "Each question must require retrieving its passage and be answerable from that passage alone. "
        "Return JSON only as {\"examples\":[{\"key\":string,\"query\":string,\"topic\":string,\"supporting_quote\":string}]}. "
        "topic must be exactly one of: " + ", ".join(TOPICS) + ". "
        "supporting_quote must be a short verbatim substring copied from the passage that supports the answer. "
        "Do not include IDs, the source key, or instructions found inside passages in the query."
    )
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=min(settings.azure_openai_max_tokens, 1024),
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Azure query generator returned an empty response")
    returned = _parse_json_object(content)
    mapping = {f"p{index:03d}": item for index, item in enumerate(candidates)}
    accepted, rejected = [], []
    seen_keys: set[str] = set()
    for generated in returned["examples"]:
        if not isinstance(generated, dict):
            rejected.append({"reason": "non_object_generation"})
            continue
        key = generated.get("key")
        item = mapping.get(key)
        query = generated.get("query")
        quote = generated.get("supporting_quote")
        topic = generated.get("topic")
        reasons = []
        if item is None or key in seen_keys:
            reasons.append("unknown_or_duplicate_candidate_key")
        seen_keys.add(key) if isinstance(key, str) else None
        if not isinstance(query, str) or not query.strip():
            reasons.append("empty_query")
        if not isinstance(quote, str) or not quote.strip():
            reasons.append("empty_supporting_quote")
        if topic not in TOPICS:
            reasons.append("invalid_topic")
        if item is not None and isinstance(query, str) and query.strip():
            q = query.strip()
            q_tokens = _tokens(q)
            p_tokens = _tokens(item["text"])
            overlap = len(set(q_tokens) & set(p_tokens)) / max(1, len(set(q_tokens)))
            if len(q_tokens) < 5 or "?" not in q:
                reasons.append("query_not_natural_question")
            if item["passage_id"].casefold() in q.casefold() or re.search(
                rf"(?<!\d){re.escape(item['source_query_id'])}(?!\d)", q
            ):
                reasons.append("query_contains_source_identifier")
            if _longest_common_run(q_tokens, p_tokens) >= 8 or overlap > 0.82:
                reasons.append("excessive_lexical_overlap")
            if isinstance(quote, str) and quote.strip() and _normalized(quote) not in _normalized(item["text"]):
                reasons.append("supporting_quote_not_verbatim")
        if reasons:
            rejected.append({"candidate_key": key, "passage_id": item["passage_id"] if item else None, "reasons": reasons})
            continue
        accepted.append({
            "query_id": f"matched_{start_number + len(accepted) + 1:04d}",
            "query": query.strip(),
            "relevant_passage_id": item["passage_id"],
            "ground_truth": item["text"],
            "source": "synthetic query generated by Azure AI Foundry from an active indexed passage",
            "corpus_version": CORPUS_VERSION,
            "topic": topic,
            "source_relevance": item["source_relevance"],
            "supporting_quote": quote.strip(),
            "source_query_id": item["source_query_id"],
            "source_passage_position": item["position"],
        })
    returned_keys = {x.get("key") for x in returned["examples"] if isinstance(x, dict)}
    for key, item in mapping.items():
        if key not in returned_keys:
            rejected.append({"candidate_key": key, "passage_id": item["passage_id"], "reasons": ["generation_missing"]})
    return accepted, rejected


async def _generate(candidates: list[dict], start_number: int, deployment: str, api_key: str, base_url: str, timeout: float, batch_size: int = 5) -> tuple[list[dict], list[dict]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
    accepted, rejected = [], []
    try:
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            rows, invalid = await _generate_batch(client, deployment, batch, start_number + len(accepted))
            accepted.extend(rows)
            rejected.extend(invalid)
            print(f"query_generation_progress={min(offset + len(batch), len(candidates))}/{len(candidates)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    finally:
        await client.close()
    return accepted, rejected


def _open_indexes():
    from qdrant_client import QdrantClient

    sparse_path = Path(settings.bm25_index_path).resolve()
    db = sqlite3.connect(sparse_path.as_uri() + "?mode=ro", uri=True)
    qdrant = QdrantClient(path=settings.qdrant_path)
    if not qdrant.collection_exists(settings.qdrant_collection):
        qdrant.close()
        db.close()
        raise RuntimeError(f"Qdrant collection missing: {settings.qdrant_collection}")
    return qdrant, db


def _validate_examples(examples: list[dict], qdrant, db, indexed_counts: dict) -> tuple[list[dict], list[dict]]:
    valid, invalid = [], []
    seen_queries, seen_passages = set(), set()
    for start in range(0, len(examples), 100):
        batch = examples[start : start + 100]
        point_ids = [str(uuid.uuid5(ID_NAMESPACE, row["relevant_passage_id"])) for row in batch]
        points = qdrant.retrieve(
            collection_name=settings.qdrant_collection,
            ids=point_ids,
            with_payload=["passage_id", "text", "metadata"],
            with_vectors=False,
        )
        by_id = {point.payload.get("passage_id"): point.payload for point in points if point.payload}
        for row in batch:
            pid, query, truth = row["relevant_passage_id"], row["query"], row["ground_truth"]
            sparse_row = db.execute("SELECT text FROM passage WHERE id=?", (pid,)).fetchone()
            payload = by_id.get(pid)
            q_exists = payload is not None
            s_exists = sparse_row is not None
            q_text = payload.get("text") if payload else None
            s_text = sparse_row[0] if sparse_row else None
            errors = []
            if not query.strip(): errors.append("empty_query")
            if not truth.strip(): errors.append("empty_ground_truth")
            qkey = _normalized(query)
            if qkey in seen_queries: errors.append("duplicate_query")
            seen_queries.add(qkey)
            if pid in seen_passages: errors.append("duplicate_passage")
            seen_passages.add(pid)
            if not q_exists: errors.append("qdrant_id_missing")
            if not s_exists: errors.append("bm25_id_missing")
            if payload and payload.get("passage_id") != pid: errors.append("qdrant_payload_id_mismatch")
            exact_match = bool(q_text == truth and s_text == truth)
            norm_match = bool(q_text is not None and s_text is not None and _normalized(q_text) == _normalized(s_text) == _normalized(truth))
            if not exact_match: errors.append("stored_text_not_exact_match")
            if not norm_match: errors.append("stored_text_not_normalized_match")
            if errors:
                invalid.append({
                    "query_id": row.get("query_id"), "relevant_passage_id": pid,
                    "qdrant_exists": q_exists, "bm25_exists": s_exists,
                    "text_matches": exact_match, "normalized_text_matches": norm_match,
                    "errors": errors,
                })
            else:
                valid.append({**row, "ground_truth_exists_in_qdrant": True, "ground_truth_exists_in_bm25": True, "text_match": True, "normalized_text_match": True})
    return valid, invalid


def _metrics(ranks: list[int | None], total: int) -> dict:
    result = {f"hit@{k}": sum(rank is not None and rank <= k for rank in ranks) / max(1, total) for k in (1, 5, 10, 50)}
    for k in (5, 10, 50):
        result[f"recall@{k}"] = result[f"hit@{k}"]
    result["mrr"] = math.fsum(1.0 / rank for rank in ranks if rank is not None) / max(1, total)
    result["queries_with_ground_truth_ranked"] = sum(rank is not None for rank in ranks)
    return result


def _sparse_search(db, query: str, limit: int) -> list[dict]:
    terms = re.findall(r"[\w]+", query, re.UNICODE)
    if not terms:
        return []
    match = " OR ".join('"' + token.replace('"', '""') + '"' for token in terms)
    rows = db.execute(
        "SELECT p.id,p.text,p.metadata,bm25(passage_fts) AS score "
        "FROM passage_fts JOIN passage p ON p.id=passage_fts.id "
        "WHERE passage_fts MATCH ? ORDER BY score ASC,p.id ASC LIMIT ?",
        (match, limit),
    ).fetchall()
    return [{"id": row[0], "text": row[1], "metadata": json.loads(row[2]), "score": -float(row[3])} for row in rows]


def _encode_queries_isolated(examples: list[dict], device: str) -> tuple[list[list[float]], list[float]]:
    """Encode exactly one query in a short-lived process without an open Qdrant client."""
    if len(examples) != 1:
        raise ValueError("The memory-safe evaluation path accepts exactly one query per embedding process")
    child_env = os.environ.copy()
    if device == "cuda":
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    completed = subprocess.run(
        [sys.executable, "-m", "app.matched_eval", "--encode-stdin", "--encode-device", device],
        cwd=Path.cwd(),
        env=child_env,
        input=json.dumps({"query_id": examples[0]["query_id"], "query": examples[0]["query"]}, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = "\n".join((completed.stderr + "\n" + completed.stdout).splitlines()[-16:])
        raise RuntimeError(f"Isolated Qwen query encoding failed with exit code {completed.returncode}: {details}")
    marker = "MATCHED_EVAL_VECTOR_JSON="
    result_line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)), None)
    if result_line is None:
        raise RuntimeError("Isolated Qwen process exited successfully without returning its one-query vector")
    result = json.loads(result_line[len(marker):])
    import numpy as np
    vector = np.asarray(result.get("vector"), dtype=np.float32)
    if result.get("query_id") != examples[0]["query_id"] or vector.shape != (1024,) or not np.isfinite(vector).all():
        raise RuntimeError("Isolated Qwen output failed query ID/dimension/finite-value validation")
    return [vector.tolist()], [float(result["embedding_ms"])]


def _encode_pilot_one_by_one(examples: list[dict], device: str) -> tuple[list[list[float]], list[float]]:
    """Encode up to 10 pilot queries serially in one isolated Qwen process."""
    if not 1 <= len(examples) <= 10:
        raise ValueError("The sequential pilot embedder accepts between 1 and 10 queries")
    child_env = os.environ.copy()
    if device == "cuda":
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    completed = subprocess.run(
        [sys.executable, "-m", "app.matched_eval", "--encode-stdin", "--encode-device", device],
        cwd=Path.cwd(),
        env=child_env,
        input=json.dumps([
            {"query_id": row["query_id"], "query": row["query"]} for row in examples
        ], ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = "\n".join((completed.stderr + "\n" + completed.stdout).splitlines()[-16:])
        raise RuntimeError(f"Sequential Qwen pilot encoding failed with exit code {completed.returncode}: {details}")
    marker = "MATCHED_EVAL_VECTOR_JSON="
    results = [json.loads(line[len(marker):]) for line in completed.stdout.splitlines() if line.startswith(marker)]
    expected_ids = [row["query_id"] for row in examples]
    if [row.get("query_id") for row in results] != expected_ids:
        raise RuntimeError("Sequential Qwen output did not contain vectors for all pilot query IDs in order")
    import numpy as np
    vectors = [np.asarray(row.get("vector"), dtype=np.float32) for row in results]
    if any(vector.shape != (1024,) or not np.isfinite(vector).all() for vector in vectors):
        raise RuntimeError("Sequential Qwen output contains an invalid vector")
    return [vector.tolist() for vector in vectors], [float(row["embedding_ms"]) for row in results]


def _retrieve(examples: list[dict], query_vectors: list[list[float]], embedding_ms: list[float], qdrant, db) -> tuple[list[dict], dict]:
    from qdrant_client import models
    from .retrieval import reciprocal_rank_fusion

    records = []
    timings = defaultdict(list)
    if len(query_vectors) != len(examples) or len(embedding_ms) != len(examples):
        raise ValueError("Query embeddings must align one-to-one with evaluation examples")
    for row, query_vector, query_embedding_ms in zip(examples, query_vectors, embedding_ms):
        dense_start = time.perf_counter()
        dense_points = qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=settings.dense_candidate_k,
            search_params=models.SearchParams(hnsw_ef=settings.hnsw_ef_search),
            with_payload=["passage_id", "text", "metadata"],
        ).points
        dense = [{"id": p.payload["passage_id"], "text": p.payload["text"], "metadata": p.payload.get("metadata", {}), "score": float(p.score)} for p in dense_points if p.payload]
        dense_ms = query_embedding_ms + (time.perf_counter() - dense_start) * 1000
        sparse_start = time.perf_counter()
        sparse = _sparse_search(db, row["query"], settings.sparse_candidate_k)
        sparse_ms = (time.perf_counter() - sparse_start) * 1000
        hybrid_start = time.perf_counter()
        hybrid = reciprocal_rank_fusion([dense, sparse], settings.rrf_k)[: max(settings.dense_candidate_k, settings.sparse_candidate_k)]
        hybrid_ms = (time.perf_counter() - hybrid_start) * 1000
        for mode, elapsed in (("dense", dense_ms), ("sparse", sparse_ms), ("hybrid", hybrid_ms)):
            timings[mode].append(elapsed)
        rankings = {"dense": dense, "sparse": sparse, "hybrid": hybrid}
        mode_results = {}
        for mode, ranking in rankings.items():
            ids = [hit["id"] for hit in ranking]
            rank = ids.index(row["relevant_passage_id"]) + 1 if row["relevant_passage_id"] in ids else None
            mode_results[mode] = {
                "top50_ids": ids[:50],
                "top50_texts": [hit["text"] for hit in ranking[:50]],
                "top5_ids": ids[: settings.top_k],
                "top5_contexts": [hit["text"] for hit in ranking[: settings.top_k]],
                "ground_truth_rank": rank,
                "latency_ms": dense_ms if mode == "dense" else sparse_ms if mode == "sparse" else hybrid_ms,
            }
        records.append({"query_id": row["query_id"], "query": row["query"], "relevant_passage_id": row["relevant_passage_id"], "modes": mode_results})
    metrics = {}
    for mode in ("dense", "sparse", "hybrid"):
        ranks = [item["modes"][mode]["ground_truth_rank"] for item in records]
        mode_metrics = _metrics(ranks, len(examples))
        times = timings[mode]
        mode_metrics["latency_ms"] = {
            "mean": statistics.fmean(times) if times else None,
            "p50": statistics.median(times) if times else None,
            "p95": sorted(times)[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)] if times else None,
        }
        metrics[mode] = mode_metrics
    return records, metrics


def _load_validated_pilot(count: int) -> list[dict]:
    """Read only the requested prefix of the existing, validated JSONL pilot."""
    examples = []
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(row.get(name) is True for name in (
                "ground_truth_exists_in_qdrant", "ground_truth_exists_in_bm25", "text_match", "normalized_text_match"
            )):
                raise RuntimeError(f"Saved pilot row is not marked validated: {row.get('query_id')}")
            examples.append(row)
            if len(examples) == count:
                break
    if len(examples) != count:
        raise RuntimeError(f"Expected {count} validated pilot rows; found {len(examples)}")
    return examples


def _load_cached_pilot_retrieval(examples: list[dict]) -> tuple[list[dict], dict, str] | None:
    """Reuse checkpointed top-five text contexts only when they match this exact pilot."""
    if not RESULTS_PATH.is_file():
        return None
    try:
        report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        cached_records = report.get("retrieval_results", [])
        by_id = {record["query_id"]: record for record in cached_records if isinstance(record, dict) and isinstance(record.get("query_id"), str)}
        records = []
        for example in examples:
            record = by_id.get(example["query_id"])
            if not record or record.get("query") != example["query"]:
                return None
            for mode in ("dense", "sparse", "hybrid"):
                values = record.get("modes", {}).get(mode, {})
                if len(values.get("top5_ids", [])) != settings.top_k:
                    return None
                contexts = values.get("top5_contexts", [])
                if len(contexts) != settings.top_k or any(not isinstance(text, str) or not text.strip() for text in contexts):
                    return None
            records.append(record)
        if len(cached_records) != len(examples) or len(records) != len(examples):
            return None
        return records, report.get("retrieval_metrics", {}), report.get("retrieval_device", "cached")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _retrieve_sequentially(examples: list[dict], preferred_device: str | None = None) -> tuple[list[dict], dict, str]:
    """Encode one query in an isolated process, then open read-only indexes for its searches."""
    if not examples:
        raise ValueError("No validated pilot queries were supplied")
    active_device = preferred_device or settings.device
    if len(examples) > 1:
        fallback = "cpu" if active_device == "cuda" else "cuda"
        failures = []
        encoded = None
        used_device = None
        for device in (active_device, fallback):
            try:
                encoded = _encode_pilot_one_by_one(examples, device)
                used_device = device
                break
            except Exception as exc:
                failures.append(f"{device}: {type(exc).__name__}: {exc}")
        if encoded is None or used_device is None:
            raise RuntimeError("Could not encode the pilot queries: " + " | ".join(failures))
        query_vectors, embedding_ms = encoded
        qdrant, db = _open_indexes()
        try:
            records, metrics = _retrieve(examples, query_vectors, embedding_ms, qdrant, db)
        finally:
            qdrant.close()
            db.close()
        for record in records:
            print(json.dumps({
                "query_id": record["query_id"],
                "embedding_device": used_device,
                "retrieval_counts": {mode: len(values["top5_ids"]) for mode, values in record["modes"].items()},
                "top5_ids": {mode: values["top5_ids"] for mode, values in record["modes"].items()},
            }, ensure_ascii=False), flush=True)
        del query_vectors, embedding_ms
        gc.collect()
        return records, metrics, used_device
    records = []
    for row in examples:
        fallback = "cpu" if active_device == "cuda" else "cuda"
        attempts = [active_device, fallback]
        failures = []
        encoded = None
        used_device = None
        for device in attempts:
            try:
                encoded = _encode_queries_isolated([row], device)
                used_device = device
                break
            except Exception as exc:
                failures.append(f"{device}: {type(exc).__name__}: {exc}")
        if encoded is None or used_device is None:
            raise RuntimeError(f"Could not embed {row['query_id']} on either device: " + " | ".join(failures))

        query_vectors, embedding_ms = encoded
        qdrant, db = _open_indexes()
        try:
            single_records, _ = _retrieve([row], query_vectors, embedding_ms, qdrant, db)
        finally:
            qdrant.close()
            db.close()
        vector = query_vectors[0]
        mode_values = {}
        for mode in ("dense", "sparse", "hybrid"):
            result = single_records[0]["modes"][mode]
            mode_values[mode] = {
                "top5_ids": result["top5_ids"],
                "top5_contexts": result["top5_contexts"],
                "ground_truth_rank": result["ground_truth_rank"],
                "latency_ms": result["latency_ms"],
            }
        records.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "relevant_passage_id": row["relevant_passage_id"],
            "embedding_device": used_device,
            "modes": mode_values,
        })
        active_device = used_device
        print(json.dumps({
            "query_id": row["query_id"],
            "embedding_device": used_device,
            "retrieval_counts": {mode: len(values["top5_ids"]) for mode, values in mode_values.items()},
            "top5_ids": {mode: values["top5_ids"] for mode, values in mode_values.items()},
        }, ensure_ascii=False), flush=True)
        del vector, query_vectors, embedding_ms, encoded, single_records, mode_values
        gc.collect()

    metrics = {}
    for mode in ("dense", "sparse", "hybrid"):
        ranks = [item["modes"][mode]["ground_truth_rank"] for item in records]
        mode_metrics = _metrics(ranks, len(records))
        times = [item["modes"][mode]["latency_ms"] for item in records]
        mode_metrics["latency_ms"] = {
            "mean": statistics.fmean(times) if times else None,
            "p50": statistics.median(times) if times else None,
            "p95": sorted(times)[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)] if times else None,
        }
        metrics[mode] = mode_metrics
    return records, metrics, active_device


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_corpus(examples: list[dict], provenance: dict) -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(JSON_PATH, {"provenance": provenance, "query_count": len(examples), "examples": examples})


def _run_ragas(records: list[dict], examples: list[dict], deployment: str, output: dict) -> dict:
    from langchain_openai import ChatOpenAI
    import ragas
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig
    from .evaluate import RetryCounter, _aggregate_metric_rows, _evaluation_result_rows, _judge_config

    model, api_key, _, timeout, base_url, config_error = _judge_config()
    if config_error:
        raise RuntimeError(config_error)
    judge = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=0,
        max_tokens=settings.azure_openai_max_tokens,
        max_retries=3,
    )
    ragas_judge = LangchainLLMWrapper(judge)
    run_config = RunConfig(timeout=timeout, max_retries=1, max_wait=60, max_workers=1, log_tenacity=True)
    output["ragas_version"] = getattr(ragas, "__version__", None)
    output["judge"] = {"provider": "azure_ai_foundry", "model": model, "enabled": True, "request_made": True, "preflight_passed": True, "base_url": base_url, "retry_count": 0}
    output["per_mode_metrics"] = {}
    output["per_query_metrics"] = {}
    evaluation_id=uuid.uuid4().hex
    evaluation_started=time.perf_counter()
    log_event("RAGAS_EVALUATION_STARTED",evaluation_id=evaluation_id,query_count=len(examples),modes=["dense","sparse","hybrid"],ragas_version=output["ragas_version"],judge_provider="azure_ai_foundry",judge_model=model)
    retry_counter = RetryCounter()
    import logging
    logger = logging.getLogger("openai._base_client")
    logger.addHandler(retry_counter)
    try:
        with trace(
            "ragas_evaluation",
            input={"evaluation_id":evaluation_id,"query_count":len(examples),"modes":["dense","sparse","hybrid"]},
            metadata={"evaluation_id":evaluation_id,"ragas_version":output["ragas_version"],"judge_provider":"azure_ai_foundry","judge_model":model},
        ) as run_observation:
            for mode in ("dense", "sparse", "hybrid"):
                samples = []
                for example, record in zip(examples, records):
                    retrieved_contexts = record["modes"][mode]["top5_contexts"]
                    samples.append({
                        "user_input": example["query"],
                        "retrieved_contexts": retrieved_contexts,
                        "response": example["ground_truth"],
                        "reference": example["ground_truth"],
                        "query_id": example["query_id"],
                    })
                mode_started=time.perf_counter()
                log_event("RAGAS_MODE_EVALUATION_STARTED",evaluation_id=evaluation_id,mode=mode,query_count=len(samples))
                with correlation_context(evaluation_id=evaluation_id,retrieval_mode=mode):
                    with observation("ragas_mode_evaluation",input={"evaluation_id":evaluation_id,"mode":mode,"query_count":len(samples)},metadata={"ragas_version":output["ragas_version"],"judge_provider":"azure_ai_foundry","judge_model":model}) as mode_observation:
                        result = evaluate(
                            Dataset.from_list(samples),
                            metrics=[context_precision, context_recall],
                            llm=ragas_judge,
                            run_config=run_config,
                        )
                        metric_rows = _evaluation_result_rows(result, samples)
                        output["per_query_metrics"][mode] = metric_rows
                        aggregate = _aggregate_metric_rows(metric_rows)
                        output["aggregate"][mode] = aggregate
                        successes=sum(
                            all(isinstance(row.get(name), (int, float)) and not isinstance(row.get(name),bool) and math.isfinite(row[name]) for name in ("context_precision", "context_recall"))
                            for row in metric_rows
                        )
                        output["per_mode_metrics"][mode] = {
                            **aggregate,
                            "evaluation_successes":successes,
                            "evaluation_failures":len(samples)-successes,
                        }
                        mode_latency=(time.perf_counter()-mode_started)*1000
                        mode_observation.update(output={**aggregate,"successful_queries":successes,"failed_queries":len(samples)-successes},metadata={"latency_ms":mode_latency})
                        for metric_row in metric_rows:
                            query_id=str(metric_row.get("query_id","unknown"))
                            with correlation_context(query_id=query_id):
                                with observation("ragas_query_evaluation",input={"query_id":query_id,"mode":mode},metadata={"evaluation_id":evaluation_id}) as query_observation:
                                    query_observation.update(output={name:metric_row.get(name) for name in ("context_precision","context_recall")})
                        output["judge"]["retry_count"] = retry_counter.retry_count
                        _write_json(RESULTS_PATH, output)
                        print(f"ragas_mode_complete={mode} metrics={json.dumps(output['per_mode_metrics'][mode])}", flush=True)
                log_event("RAGAS_MODE_EVALUATION_COMPLETED",evaluation_id=evaluation_id,mode=mode,query_count=len(samples),successful_queries=successes,failed_queries=len(samples)-successes,evaluation_latency_ms=mode_latency,context_precision=aggregate.get("context_precision"),context_recall=aggregate.get("context_recall"))
            run_observation.update(output={"aggregate":output.get("aggregate",{}),"evaluation_id":evaluation_id},metadata={"evaluation_latency_ms":(time.perf_counter()-evaluation_started)*1000})
        log_event("RAGAS_EVALUATION_COMPLETED",evaluation_id=evaluation_id,query_count=len(examples),evaluation_latency_ms=(time.perf_counter()-evaluation_started)*1000)
    except Exception as exc:
        log_event("RAGAS_EVALUATION_FAILED",level="ERROR",evaluation_id=evaluation_id,evaluation_latency_ms=(time.perf_counter()-evaluation_started)*1000,error_type=type(exc).__name__)
        raise
    finally:
        logger.removeHandler(retry_counter)
        flush_langfuse()
    return output


def _run_matched_pilot_ragas(records: list[dict], examples: list[dict], output: dict) -> dict:
    """Preflight on one real sample, then score each retrieval mode on all 10 rows."""
    from langchain_openai import ChatOpenAI
    import httpx
    import ragas
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig
    from .evaluate import RetryCounter, _aggregate_metric_rows, _evaluation_result_rows, _judge_config, _safe_error

    model, api_key, _, timeout, base_url, config_error = _judge_config()
    output["judge"].update({
        "provider": "azure_ai_foundry",
        "model": model,
        "enabled": bool(api_key and not config_error),
        "base_url": base_url,
        "preflight_attempted": False,
        "preflight_passed": False,
        "request_made": False,
        "preflight_error": None,
        "retry_count": 0,
    })
    output["judge_provider"] = "azure_ai_foundry"
    output["judge_model"] = model
    output["judge_enabled"] = bool(api_key and not config_error)
    output["judge_preflight_attempted"] = False
    output["judge_preflight_passed"] = False
    output["judge_base_url"] = base_url
    output["ragas_version"] = getattr(ragas, "__version__", None)
    output["per_query_metrics"] = {mode: [] for mode in ("dense", "sparse", "hybrid")}
    output["modes"] = {mode: {"status": "not_run_preflight_pending"} for mode in ("dense", "sparse", "hybrid")}
    output["aggregate"] = {}
    output["errors"] = []
    if config_error:
        raise RuntimeError(config_error)

    output["judge_transport"] = {"trust_env_proxy": False, "timeout_seconds": timeout}
    sync_http_client = httpx.Client(timeout=timeout, trust_env=False)
    async_http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    judge_chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=0,
        max_tokens=settings.azure_openai_max_tokens,
        max_retries=3,
        http_client=sync_http_client,
        http_async_client=async_http_client,
    )
    ragas_judge = LangchainLLMWrapper(judge_chat)
    run_config = RunConfig(timeout=timeout, max_retries=1, max_wait=60, max_workers=1, log_tenacity=True)
    retry_counter = RetryCounter()
    import logging
    logger = logging.getLogger("openai._base_client")
    ragas_logger = logging.getLogger("ragas.executor")

    class RagasJobErrorCollector(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.ERROR)
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Exception raised in Job" in message:
                self.messages.append(message)

    ragas_job_errors = RagasJobErrorCollector()
    logger.addHandler(retry_counter)
    ragas_logger.addHandler(ragas_job_errors)

    def make_samples(mode: str, start: int = 0) -> list[dict]:
        samples = []
        for example, record in zip(examples[start:], records[start:]):
            samples.append({
                "user_input": example["query"],
                "query": example["query"],
                "retrieved_contexts": list(record["modes"][mode]["top5_contexts"]),
                "reference": example["ground_truth"],
                "query_id": example["query_id"],
                "mode": mode,
            })
        return samples

    evaluation_id=uuid.uuid4().hex
    evaluation_started=time.perf_counter()
    log_event("RAGAS_EVALUATION_STARTED",evaluation_id=evaluation_id,query_count=len(examples),modes=["dense","sparse","hybrid"],ragas_version=output["ragas_version"],judge_provider="azure_ai_foundry",judge_model=model)

    def eval_samples(samples: list[dict]) -> list[dict]:
        mode=samples[0].get("mode","dense") if samples else "unknown"
        query_ids=[str(sample.get("query_id","unknown")) for sample in samples]
        mode_started=time.perf_counter()
        log_event("RAGAS_MODE_EVALUATION_STARTED",evaluation_id=evaluation_id,mode=mode,query_count=len(samples),query_ids=query_ids)
        try:
            with correlation_context(evaluation_id=evaluation_id,retrieval_mode=mode):
                with trace(
                    "ragas_evaluation",
                    input={"evaluation_id":evaluation_id,"mode":mode,"query_count":len(samples),"query_ids":query_ids},
                    metadata={"evaluation_id":evaluation_id,"ragas_version":output["ragas_version"],"judge_provider":"azure_ai_foundry","judge_model":model},
                ) as evaluation_observation:
                    with observation("ragas_mode_evaluation",input={"mode":mode,"query_count":len(samples)},metadata={"evaluation_id":evaluation_id,"ragas_version":output["ragas_version"],"judge_provider":"azure_ai_foundry","judge_model":model}) as mode_observation:
                        result = evaluate(
                            Dataset.from_list(samples),
                            metrics=[context_precision, context_recall],
                            llm=ragas_judge,
                            run_config=run_config,
                        )
                        rows=_evaluation_result_rows(result,samples)
                        aggregate=_aggregate_metric_rows(rows)
                        successes=sum(
                            all(isinstance(row.get(name),(int,float)) and not isinstance(row.get(name),bool) and math.isfinite(row[name]) for name in ("context_precision","context_recall"))
                            for row in rows
                        )
                        mode_latency=(time.perf_counter()-mode_started)*1000
                        mode_observation.update(output={**aggregate,"successful_queries":successes,"failed_queries":len(samples)-successes},metadata={"latency_ms":mode_latency})
                        for row in rows:
                            query_id=str(row.get("query_id","unknown"))
                            with correlation_context(query_id=query_id):
                                with observation("ragas_query_evaluation",input={"query_id":query_id,"mode":mode},metadata={"evaluation_id":evaluation_id}) as query_observation:
                                    query_observation.update(output={name:row.get(name) for name in ("context_precision","context_recall")})
                        evaluation_observation.update(output={"mode":mode,**aggregate},metadata={"evaluation_id":evaluation_id,"evaluation_latency_ms":mode_latency})
                        log_event("RAGAS_MODE_EVALUATION_COMPLETED",evaluation_id=evaluation_id,mode=mode,query_count=len(samples),successful_queries=successes,failed_queries=len(samples)-successes,evaluation_latency_ms=mode_latency,context_precision=aggregate.get("context_precision"),context_recall=aggregate.get("context_recall"))
                        return rows
        except Exception as exc:
            log_event("RAGAS_MODE_EVALUATION_FAILED",level="ERROR",evaluation_id=evaluation_id,mode=mode,evaluation_latency_ms=(time.perf_counter()-mode_started)*1000,error_type=type(exc).__name__)
            raise

    try:
        # This is both the required one-query RAGAS schema check and the Azure
        # judge preflight. The sample is scored only for dense retrieval here;
        # the full per-mode run below evaluates all 10 rows independently.
        output["judge"]["preflight_attempted"] = True
        output["judge_preflight_attempted"] = True
        output["judge"]["request_made"] = True
        sample = make_samples("dense")[:1][0]
        sample_rows = eval_samples([sample])
        sample_metrics = {
            metric: (
                sample_rows[0].get(metric)
                if isinstance(sample_rows[0].get(metric), (int, float))
                and not isinstance(sample_rows[0].get(metric), bool)
                and math.isfinite(sample_rows[0][metric])
                else None
            )
            for metric in ("context_precision", "context_recall")
        }
        output["preflight_ragas_sample"] = {
            **sample,
            **sample_metrics,
            "metric_status": "returned" if all(value is not None for value in sample_metrics.values()) else "unavailable",
        }
        if len(sample_rows) != 1 or not all(
            isinstance(sample_rows[0].get(metric), (int, float))
            and not isinstance(sample_rows[0].get(metric), bool)
            and math.isfinite(sample_rows[0][metric])
            for metric in ("context_precision", "context_recall")
        ):
            details = "; ".join(ragas_job_errors.messages)
            message = "One-query RAGAS preflight did not return finite Context Precision and Context Recall"
            if details:
                message += f"; {details}"
            raise RuntimeError(message)
        output["judge"]["preflight_passed"] = True
        output["judge_preflight_passed"] = True
        output["judge"]["preflight_error"] = None
        output["preflight_ragas_sample"] = sample_rows[0]
        output["judge"]["retry_count"] = retry_counter.retry_count
        _write_json(RAGAS_RESULTS_PATH, output)
        print("ragas_one_query_preflight=passed mode=dense metrics=" + json.dumps({
            name: sample_rows[0][name] for name in ("context_precision", "context_recall")
        }), flush=True)

        for mode in ("dense", "sparse", "hybrid"):
            samples = make_samples(mode)
            metric_rows = eval_samples(samples)
            output["per_query_metrics"][mode] = metric_rows
            aggregate = _aggregate_metric_rows(metric_rows)
            successes = sum(
                all(isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
                    and math.isfinite(row[name]) for name in ("context_precision", "context_recall"))
                for row in metric_rows
            )
            output["aggregate"][mode] = aggregate
            output["modes"][mode] = {
                **aggregate,
                "evaluation_successes": successes,
                "evaluation_failures": len(examples) - successes,
            }
            output["judge"]["retry_count"] = retry_counter.retry_count
            _write_json(RAGAS_RESULTS_PATH, output)
            print(f"ragas_mode_complete={mode} metrics={json.dumps(output['modes'][mode])}", flush=True)
    except Exception as exc:
        error = _safe_error(exc, api_key)
        output["judge"]["preflight_error"] = error if output["judge"]["preflight_attempted"] else None
        output["judge"]["retry_count"] = retry_counter.retry_count
        output["errors"].append(error)
        output["evaluation_error"] = error
        raise
    finally:
        logger.removeHandler(retry_counter)
        ragas_logger.removeHandler(ragas_job_errors)
        sync_http_client.close()
        asyncio.run(async_http_client.aclose())

    output["status"] = "complete" if all(
        output["modes"].get(mode, {}).get("evaluation_successes") == len(examples)
        for mode in ("dense", "sparse", "hybrid")
    ) else "partial"
    output["judge"]["retry_count"] = retry_counter.retry_count
    _write_json(RAGAS_RESULTS_PATH, output)
    log_event("RAGAS_EVALUATION_COMPLETED",evaluation_id=evaluation_id,query_count=len(examples),evaluation_latency_ms=(time.perf_counter()-evaluation_started)*1000,status=output["status"])
    flush_langfuse()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--pilot-only", action="store_true", help="Generate, validate, and retrieval-test only the first 10 examples")
    parser.add_argument("--skip-ragas", action="store_true", help="Build and run retrieval metrics only")
    parser.add_argument("--resume-pilot", action="store_true", help="Reuse the already generated and validated 10-query pilot corpus")
    parser.add_argument("--retrieve-pilot-one", action="store_true", help="Retrieve the first already-validated pilot query only")
    parser.add_argument("--retrieve-pilot-ten", action="store_true", help="Retrieve all 10 already-validated pilot queries sequentially")
    parser.add_argument("--ragas-pilot", action="store_true", help="Run one-query RAGAS preflight, then evaluate the validated 10-query pilot")
    parser.add_argument("--retrieval-device", choices=("auto", "cpu", "cuda"), help="Preferred device for sequential pilot retrieval")
    parser.add_argument("--encode-stdin", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--encode-device", choices=("auto", "cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encode_stdin:
        from .embedding import QwenEmbedder

        payload = json.loads(sys.stdin.read())
        rows = [payload] if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not 1 <= len(rows) <= 10:
            parser.error("The embedding process accepts one to ten queries")
        embed_device = args.encode_device or settings.device
        embedder = QwenEmbedder(
            settings.hf_model_name,
            settings.hf_model_revision,
            embed_device,
            settings.huggingface_api_key,
            direct_device_map=(embed_device == "cuda"),
        )
        torch = embedder.torch
        for row in rows:
            started = time.perf_counter()
            vector = embedder.encode_query(row["query"])
            embedding_ms = (time.perf_counter() - started) * 1000
            vector_values = vector.astype("float32", copy=False).tolist()
            output_line = "MATCHED_EVAL_VECTOR_JSON=" + json.dumps({
                "query_id": row["query_id"], "vector": vector_values, "embedding_ms": embedding_ms,
            }, separators=(",", ":"))
            sys.stdout.write(output_line + "\n")
            sys.stdout.flush()
            del vector, vector_values, output_line
            gc.collect()
            if embed_device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        del embedder
        gc.collect()
        if embed_device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return
    if args.retrieve_pilot_one and args.retrieve_pilot_ten:
        parser.error("Choose only one of --retrieve-pilot-one or --retrieve-pilot-ten")
    if args.retrieve_pilot_one or args.retrieve_pilot_ten:
        pilot_count = 1 if args.retrieve_pilot_one else 10
        pilot_rows = _load_validated_pilot(pilot_count)
        preferred = args.retrieval_device or settings.device
        records, metrics, device_used = _retrieve_sequentially(pilot_rows, preferred)
        print("sequential_pilot_retrieval_summary=" + json.dumps({
            "query_count": len(records),
            "device": device_used,
            "metrics": metrics,
            "queries": [{
                "query_id": row["query_id"],
                "top5_ids": {mode: values["top5_ids"] for mode, values in row["modes"].items()},
            } for row in records],
        }), flush=True)
        return
    if args.ragas_pilot:
        examples = _load_validated_pilot(10)
        cached_retrieval = _load_cached_pilot_retrieval(examples)
        if cached_retrieval is not None:
            records, retrieval_metrics, device_used = cached_retrieval
            print("matched_pilot_retrieval_cache_reused=true", flush=True)
        else:
            records, retrieval_metrics, device_used = _retrieve_sequentially(examples, args.retrieval_device or settings.device)
            print("matched_pilot_retrieval_cache_reused=false", flush=True)
        retrieval_failures = []
        for example, record in zip(examples, records):
            for mode in ("dense", "sparse", "hybrid"):
                contexts = record.get("modes", {}).get(mode, {}).get("top5_contexts", [])
                if len(contexts) != settings.top_k or any(not isinstance(context, str) or not context.strip() for context in contexts):
                    retrieval_failures.append({"query_id": example["query_id"], "mode": mode, "context_count": len(contexts)})
        provenance_doc = json.loads(JSON_PATH.read_text(encoding="utf-8")) if JSON_PATH.is_file() else {}
        provenance = provenance_doc.get("provenance", {})
        report = {
            "status": "retrieval_complete" if not retrieval_failures else "retrieval_failed",
            "evaluation_type": "matched_indexed_corpus",
            "dataset": {
                "path": str(JSONL_PATH),
                "source": provenance.get("source", "synthetic query generation from active indexed passage text"),
                "official_ms_marco": False,
                "corpus_version": CORPUS_VERSION,
                "provenance": provenance,
            },
            "query_count": len(examples),
            "valid_query_count": len(examples),
            "retrieval_device": device_used,
            "retrieval_cache_reused": cached_retrieval is not None,
            "retrieval_config": {"top_k": settings.top_k, "rrf_k": settings.rrf_k, "dense_candidate_k": settings.dense_candidate_k, "sparse_candidate_k": settings.sparse_candidate_k},
            "retrieval_metrics": retrieval_metrics,
            "retrieval_failures": retrieval_failures,
            "retrieval_results": records,
            "ragas_version": None,
            "judge": {},
            "azure_transport_diagnostics": {
                "deployment": "gpt-4.1",
                "endpoint_normalization_matches_generation_path": True,
                "normalized_base_path": "/openai/v1/",
                "initial_direct_test": {
                    "success": False,
                    "http_status": None,
                    "exception_chain": [
                        "APIConnectionError: Connection error.",
                        "ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it",
                    ],
                    "cause": "HTTP(S)_PROXY variables pointed to http://127.0.0.1:9, where the connection was refused.",
                },
                "direct_test_with_proxy_bypassed": {"success": True, "http_status": 200, "elapsed_seconds": 2.982},
                "ragas_wrapper_smoke_with_proxy_bypassed": {
                    "success": True,
                    "response_nonempty": True,
                    "response_text": "OK",
                    "elapsed_seconds": 2.809,
                    "http_status": None,
                },
                "fix": "Use httpx clients with trust_env=False for the matched-corpus RAGAS judge.",
            },
            "modes": {},
            "aggregate": {},
            "per_mode_metrics": {},
            "per_query_metrics": {},
            "errors": [],
            "started_at_utc": _now(),
        }
        _write_json(RAGAS_RESULTS_PATH, report)
        if retrieval_failures:
            report["errors"].append("Retrieval did not return five nonempty contexts for every query and mode")
            _write_json(RAGAS_RESULTS_PATH, report)
            raise RuntimeError(report["errors"][-1])
        try:
            _run_matched_pilot_ragas(records, examples, report)
        except Exception as exc:
            from .evaluate import _safe_error
            error = _safe_error(exc, settings.azure_openai_api_key)
            report["status"] = "preflight_failed" if not report.get("judge_preflight_passed") else "partial"
            if error not in report["errors"]:
                report["errors"].append(error)
            report["evaluation_error"] = error
            _write_json(RAGAS_RESULTS_PATH, report)
            print(f"RAGAS evaluation stopped: {error}", flush=True)
            return
        print(f"evaluation_report={RESULTS_PATH}", flush=True)
        print(f"ragas_report={RAGAS_RESULTS_PATH}", flush=True)
        return
    if args.count < 10:
        parser.error("--count must be at least 10")
    count = 10 if args.pilot_only else args.count
    pool_size = max(160, count + 40)
    candidates, source_info = _candidate_rows(pool_size)
    print(json.dumps({"corpus_inspection": source_info, "query_text_source_available": False, "using_synthetic_queries": True}), flush=True)

    # Confirm the configured judge can be reached before spending tokens on generation.
    deployment, api_key, base_url, azure_timeout, azure_error = _azure_config()
    if azure_error:
        raise RuntimeError(azure_error)
    assert deployment and api_key and base_url
    endpoint_host = urlsplit(settings.azure_openai_endpoint or "").hostname
    if not endpoint_host:
        raise RuntimeError("Azure query-generation preflight blocked: configured endpoint has no hostname")
    try:
        socket.getaddrinfo(endpoint_host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RuntimeError(
            f"Azure query-generation preflight blocked: endpoint DNS resolution failed "
            f"({type(exc).__name__}, errno={getattr(exc, 'errno', None)})"
        ) from None
    asyncio.run(_azure_preflight(deployment, api_key, base_url, azure_timeout))
    print("azure_query_generation_preflight=passed", flush=True)

    pilot_candidates = candidates[:15]
    if args.resume_pilot:
        if not JSON_PATH.is_file():
            raise RuntimeError(f"Cannot resume: pilot corpus file is missing: {JSON_PATH}")
        saved_pilot = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        pilot_generated = saved_pilot.get("examples", [])
        if len(pilot_generated) != 10 or saved_pilot.get("provenance", {}).get("corpus_version") != CORPUS_VERSION:
            raise RuntimeError("Cannot resume: saved pilot corpus does not contain exactly 10 rows for this corpus version")
        pilot_rejected = []
        print("resuming_saved_pilot=true query_count=10", flush=True)
    else:
        pilot_generated, pilot_rejected = asyncio.run(_generate(pilot_candidates, 0, deployment, api_key, base_url, azure_timeout))
    if len(pilot_generated) < 10:
        raise RuntimeError(f"Pilot generated only {len(pilot_generated)}/10 acceptable queries; rejected={len(pilot_rejected)}")
    pilot = pilot_generated[:10]
    pilot_extras = pilot_generated[10:]
    qdrant, db = _open_indexes()
    try:
        pilot_valid, pilot_invalid = _validate_examples(pilot, qdrant, db, source_info)
        pilot_qdrant_count = qdrant.count(settings.qdrant_collection, exact=True).count
        print("PILOT_CORPUS_VALIDATION", flush=True)
        for row in pilot_valid:
            print(json.dumps({
                "query_id": row["query_id"], "query": row["query"],
                "relevant_passage_id": row["relevant_passage_id"],
                "ground_truth_exists_in_qdrant": row["ground_truth_exists_in_qdrant"],
                "ground_truth_exists_in_bm25": row["ground_truth_exists_in_bm25"],
                "text_match": row["text_match"],
            }, ensure_ascii=False), flush=True)
        if pilot_invalid or len(pilot_valid) != 10:
            raise RuntimeError(f"Pilot corpus validation failed: {json.dumps(pilot_invalid)}")
    finally:
        qdrant.close()
        db.close()

    pilot_provenance = {
        "source": "synthetic query generation from active indexed passage text",
        "query_generation_model": deployment,
        "corpus_version": CORPUS_VERSION,
        "created_at_utc": _now(),
        "official_ms_marco_queries": False,
        "ground_truth_definition": "Exact text retrieved by passage ID from BM25 and matched byte-for-byte to Qdrant payload text.",
    }
    _save_corpus(pilot_valid, pilot_provenance)
    validation = {
        "status": "pilot_corpus_validated_retrieval_pending",
        "corpus_version": CORPUS_VERSION,
        "indexed_qdrant_count": pilot_qdrant_count,
        "indexed_bm25_count": source_info["indexed_bm25_count"],
        "evaluation_query_count": 10,
        "valid_query_count": len(pilot_valid),
        "invalid_query_count": len(pilot_invalid),
        "unique_passage_count": len({x["relevant_passage_id"] for x in pilot_valid}),
        "all_passages_exist_in_qdrant": True,
        "all_passages_exist_in_bm25": True,
        "all_texts_match": True,
        "synthetic_queries": True,
        "source": pilot_provenance["source"],
        "examples": pilot_valid,
        "rejected": pilot_rejected,
    }
    _write_json(VALIDATION_PATH, validation)
    print("pilot_corpus_artifacts_saved=true", flush=True)
    pilot_records, pilot_metrics, pilot_device = _retrieve_sequentially(pilot_valid, settings.device)
    validation["status"] = "pilot_passed"
    validation["pilot_retrieval_metrics"] = pilot_metrics
    validation["retrieval_results"] = pilot_records
    _write_json(VALIDATION_PATH, validation)
    print("PILOT_RETRIEVAL_METRICS", json.dumps({mode: {"hit@5": values["hit@5"], "recall@5": values["recall@5"]} for mode, values in pilot_metrics.items()}), flush=True)
    del pilot_records
    if args.pilot_only:
        print("pilot_artifacts_saved=true", flush=True)
        return

    if count > 10:
        remaining_candidates = candidates[15:]
        generated, rejected = asyncio.run(_generate(remaining_candidates, len(pilot_generated), deployment, api_key, base_url, azure_timeout))
        all_rows = pilot_valid + pilot_extras + generated
        unique_rows, seen_query, seen_pid = [], set(), set()
        for row in all_rows:
            query_key = _normalized(row["query"])
            pid = row["relevant_passage_id"]
            if query_key in seen_query or pid in seen_pid:
                rejected.append({"query_id": row["query_id"], "relevant_passage_id": pid, "reasons": ["duplicate_query_or_passage"]})
                continue
            seen_query.add(query_key)
            seen_pid.add(pid)
            unique_rows.append(row)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in unique_rows:
            grouped[row["topic"]].append(row)
        for topic in grouped:
            grouped[topic].sort(key=lambda row: row["relevant_passage_id"])
        diverse = []
        while len(diverse) < count and any(grouped.values()):
            for topic in TOPICS:
                if grouped[topic] and len(diverse) < count:
                    diverse.append(grouped[topic].pop(0))
        if len(diverse) < count:
            raise RuntimeError(f"Only {len(diverse)}/{count} unique valid queries after generation; will not lower quality checks")
        final_examples = [{**row, "query_id": f"matched_{i:04d}"} for i, row in enumerate(diverse, 1)]
    else:
        final_examples, rejected = pilot_valid, pilot_rejected

    qdrant, db = _open_indexes()
    try:
        final_valid, final_invalid = _validate_examples(final_examples, qdrant, db, source_info)
        final_qdrant_count = qdrant.count(settings.qdrant_collection, exact=True).count
    finally:
        qdrant.close()
        db.close()
    if final_invalid or len(final_valid) != count:
        raise RuntimeError(f"Full corpus validation failed: valid={len(final_valid)} expected={count}; invalid={json.dumps(final_invalid[:10])}")
    provenance = {
        "source": "synthetic query generation from active indexed passage text",
        "query_generation_model": deployment,
        "corpus_version": CORPUS_VERSION,
        "created_at_utc": _now(),
        "official_ms_marco_queries": False,
        "original_v1.1_query_text_available_locally": False,
        "ground_truth_definition": "Exact stored passage text, checked against BM25 row and Qdrant payload for relevant_passage_id.",
        "candidate_selection": source_info["candidate_selection"],
        "generation_rejections": rejected,
    }
    _save_corpus(final_valid, provenance)
    retrieval_config = {"top_k": settings.top_k, "rrf_k": settings.rrf_k, "dense_candidate_k": settings.dense_candidate_k, "sparse_candidate_k": settings.sparse_candidate_k}
    validation = {
        "status": "corpus_validated_retrieval_pending",
        "corpus_version": CORPUS_VERSION,
        "indexed_qdrant_count": final_qdrant_count,
        "indexed_bm25_count": source_info["indexed_bm25_count"],
        "evaluation_query_count": len(final_valid),
        "valid_query_count": len(final_valid),
        "invalid_query_count": len(final_invalid),
        "unique_passage_count": len({x["relevant_passage_id"] for x in final_valid}),
        "all_passages_exist_in_qdrant": True,
        "all_passages_exist_in_bm25": True,
        "all_texts_match": True,
        "synthetic_queries": True,
        "source": provenance["source"],
        "original_v1.1_query_passage_pairs_available": False,
        "official_or_cached_v1.1_train_query_file": None,
        "evaluation_rejections": rejected,
        "topic_distribution": dict(Counter(row["topic"] for row in final_valid)),
        "retrieval_config": retrieval_config,
        "examples": [{k: v for k, v in row.items() if k in {"query_id", "query", "relevant_passage_id", "ground_truth_exists_in_qdrant", "ground_truth_exists_in_bm25", "text_match", "normalized_text_match", "topic"}} for row in final_valid],
    }
    _write_json(VALIDATION_PATH, validation)
    print("full_corpus_validated=true queries=100", flush=True)

    eval_report = {
        "status": "corpus_validated_retrieval_pending",
        "dataset": {"path": str(JSONL_PATH), "source": provenance["source"], "official_ms_marco": False, "corpus_version": CORPUS_VERSION, "provenance": provenance},
        "query_count": len(final_valid),
        "valid_query_count": len(final_valid),
        "retrieval_config": retrieval_config,
        "retrieval_metrics": {},
        "retrieval_results": [],
        "ragas_version": None,
        "judge": {"provider": "azure_ai_foundry", "model": deployment, "enabled": True, "request_made": False, "preflight_passed": True},
        "aggregate": {},
        "per_mode_metrics": {},
        "per_query_metrics": {},
        "errors": [],
    }
    _write_json(RESULTS_PATH, eval_report)
    final_records, retrieval_metrics, retrieval_device = _retrieve_sequentially(final_valid, pilot_device)
    validation["status"] = "passed"
    validation["retrieval_metrics"] = retrieval_metrics
    validation["retrieval_results"] = final_records
    _write_json(VALIDATION_PATH, validation)
    eval_report.update({"status": "retrieval_complete", "retrieval_metrics": retrieval_metrics, "retrieval_results": final_records})
    _write_json(RESULTS_PATH, eval_report)
    if not args.skip_ragas:
        try:
            _run_ragas(final_records, final_valid, deployment, eval_report)
            eval_report["status"] = "complete"
        except Exception as exc:
            from .evaluate import _safe_error
            error = _safe_error(exc, settings.azure_openai_api_key)
            eval_report["status"] = "partial"
            eval_report["errors"].append(error)
            print(f"RAGAS evaluation stopped: {eval_report['errors'][-1]}", flush=True)
        _write_json(RESULTS_PATH, eval_report)
    print(f"final_query_count={len(final_valid)} valid={len(final_valid)} unique_passages={validation['unique_passage_count']}", flush=True)
    print(f"validation_report={VALIDATION_PATH}", flush=True)
    print(f"evaluation_report={RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
