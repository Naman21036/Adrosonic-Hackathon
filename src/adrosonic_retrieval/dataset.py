"""Hugging Face dataset adapters; iteration is streaming and bounded-memory."""
from collections.abc import Iterator, Mapping
from typing import Any


def load_passages(dataset_id: str, config: str, split: str, limit: int,
                  token: str | None = None, revision: str | None = None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset
    rows = load_dataset(dataset_id, config, split=split, streaming=True, token=token, revision=revision)
    emitted = 0
    for row in rows:
        pid = row.get("pid", row.get("passage_id", row.get("id")))
        text = row.get("text", row.get("passage"))
        if pid is None or text is None:
            raise ValueError(f"Dataset row lacks pid/text fields; found keys {list(row.keys())}")
        pid = str(pid)
        if not str(text).strip():
            continue
        yield {"passage_id": pid, "text": str(text), "source": dataset_id,
               "dataset_config": config, "dataset_split": split}
        emitted += 1
        if emitted >= limit:
            break


def load_eval_queries(dataset_id: str, config: str, split: str, limit: int | None,
                      token: str | None = None) -> list[dict[str, Any]]:
    """Read labeled MS MARCO QA rows and retain selected reference passages."""
    from datasets import load_dataset
    dataset = load_dataset(dataset_id, config, split=split, token=token)
    records: list[dict[str, Any]] = []
    for row in dataset:
        passages = row.get("passages", {})
        texts = passages.get("passage_text", []) if isinstance(passages, Mapping) else []
        selected = passages.get("is_selected", []) if isinstance(passages, Mapping) else []
        references = [str(text) for text, flag in zip(texts, selected) if flag and str(text).strip()]
        answers = row.get("answers", [])
        answer = next((str(a) for a in answers if str(a).strip()), "") if isinstance(answers, list) else str(answers or "")
        query = str(row.get("query", "")).strip()
        if query and references and answer:
            records.append({"query": query, "reference_answer": answer, "reference_contexts": references,
                            "query_id": str(row.get("query_id", ""))})
            if limit is not None and len(records) >= limit:
                break
    return records
