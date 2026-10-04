"""Read-only audit of the on-disk Qdrant Local IDs, sparse rows, and journal."""
from __future__ import annotations

import base64
import json
import pickle
import sqlite3
import uuid
from pathlib import Path

from .config import settings

_ID_NAMESPACE = uuid.UUID("db8dcbb1-8011-4a09-83ec-6644a2ad64e8")
REPORT = Path("reports/phase2/reconciliation_audit.json")


def audit() -> dict:
    source = Path(settings.qdrant_path) / "collection" / settings.qdrant_collection / "storage.sqlite"
    source_uri = f"file:{source.resolve().as_posix()}?mode=ro"
    sparse_uri = f"file:{Path(settings.bm25_index_path).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as db:
        qdrant_point_ids = {str(pickle.loads(base64.b64decode(row[0]))) for row in db.execute("SELECT id FROM points")}
    with sqlite3.connect(sparse_uri, uri=True) as db:
        rows = db.execute("SELECT id,text FROM passage").fetchall()
    sparse_texts = {pid: text for pid, text in rows}
    sparse_ids = set(sparse_texts)
    qdrant_passage_ids = {
        pid for pid in sparse_ids
        if str(uuid.uuid5(_ID_NAMESPACE, pid)) in qdrant_point_ids
    }

    pending: dict[str, dict] = {}
    journal = Path(settings.journal_path)
    malformed_journal_lines = 0
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed_journal_lines += 1
                continue
            key = event.get("event", event.get("id"))
            if event.get("op") in {"upsert", "delete"}:
                pending[key] = event
            elif event.get("op") == "commit":
                pending.pop(key, None)
    pending_upserts = [row for row in pending.values() if row.get("op") == "upsert"]
    sparse_only = sorted(sparse_ids - qdrant_passage_ids)
    journal_pending_ids = sorted(row["id"] for row in pending_upserts)
    journal_pending_match = all(
        row["id"] in sparse_texts and sparse_texts[row["id"]] == row.get("text")
        for row in pending_upserts
    )
    pending_ids_match_sparse_only = set(journal_pending_ids) == set(sparse_only)
    return {
        "status": "passed" if pending_ids_match_sparse_only and journal_pending_match and not malformed_journal_lines else "failed",
        "qdrant_local_point_count": len(qdrant_point_ids),
        "sparse_count": len(sparse_ids),
        "qdrant_passage_ids_mapped_from_sparse": len(qdrant_passage_ids),
        "sparse_only_count": len(sparse_only),
        "sparse_only_ids": sparse_only,
        "pending_upsert_count": len(pending_upserts),
        "pending_upsert_ids": journal_pending_ids,
        "pending_journal_text_matches_sparse": journal_pending_match,
        "pending_ids_equal_sparse_only_ids": pending_ids_match_sparse_only,
        "malformed_journal_lines": malformed_journal_lines,
        "qdrant_point_ids_without_sparse_mapping_count": len(qdrant_point_ids - {
            str(uuid.uuid5(_ID_NAMESPACE, pid)) for pid in sparse_ids
        }),
        "source_database_open_mode": "read-only SQLite; Qdrant Local client not opened",
    }


def main() -> None:
    result = audit()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if not key.endswith("_ids")}
    print(json.dumps(summary, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
