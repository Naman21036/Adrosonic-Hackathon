# FR-5: Live index updates

The live update routes change one passage in the configured Qdrant collection and SQLite FTS5/BM25 database. They do not scan or rebuild the corpus. Each upsert embeds one passage; each delete embeds none. Qdrant's existing deterministic passage-ID mapping is reused.

## Upsert

`POST /upsert`

```json
{
  "id": "demo-live-001",
  "text": "Adrosonic demonstrates incremental live index updates.",
  "source_id": "demo",
  "metadata": { "category": "demo" }
}
```

`passage_id` is accepted as an alias for `id`. Custom passage IDs require a non-empty `source_id` either at the top level or in metadata. Canonical IDs such as `msmarco-23013-4` retain their numeric source ID convention; a supplied source ID must match the ID prefix. Text is limited to 20,000 characters, IDs and source IDs to 256 characters.

The response includes `status` (`created` or `updated`), the embedding dimension, `embedding_count_delta: 1`, dense and sparse counts before and after, stage/total latency, and the pending mutation count. Updating an existing ID replaces its Qdrant point and BM25/FTS row; it does not create a duplicate.

## Delete

`DELETE /passages/{passage_id}`

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/passages/demo-live-001' -Method Delete
```

The route removes that exact ID from Qdrant and BM25, reports per-index deletion flags and count changes, and returns 404 when neither index contains the ID. The frontend requires a confirmation step before sending this request. There is no bulk-delete route.

## Consistency and recovery

The update journal records and fsyncs a pending intent before changing Qdrant. Writes use the existing Qdrant-first order. After a confirmed Qdrant write, the SQLite transaction replaces the passage row and matching FTS row; only after both stores succeed is the journal event committed and compacted. If Qdrant fails, BM25 is left unchanged. If the BM25 transaction fails after Qdrant succeeds, the API reports a failure and leaves the event pending. Startup and the next live mutation replay pending events before proceeding. A failed operation is never returned as successful.

Structured Loguru events include `LIVE_UPSERT_STARTED`, `LIVE_UPSERT_EMBEDDED`, `LIVE_UPSERT_QDRANT_SUCCESS`, `LIVE_UPSERT_BM25_SUCCESS`, `LIVE_UPSERT_COMPLETED`, and the corresponding delete events. They carry request ID context, passage ID, source ID where available, elapsed time, status, and embedding count delta; passage text and credentials are not logged. Langfuse traces record operation metadata and counts. Tracing is optional and failures do not prevent indexing.

## UI walkthrough

Open **Live Index Updates** (`/updates`) while the API is online:

1. Select **Prepare demo passage**, then **Upsert passage**.
2. Verify the inserted text with Dense or Hybrid/RRF retrieval.
3. Load version 2 and upsert the same ID; the result should say `updated`, with unchanged index counts.
4. Verify the replacement text, then choose **Delete demo passage** and confirm.
5. Verify that the demo ID no longer appears. Hybrid retrieval exercises BM25 and RRF; the HTTP search API does not expose a standalone BM25 mode.

The page targets the configured backend. Use the dedicated API tests for isolated verification; the tests use a temporary Qdrant collection, temporary sparse database, and deterministic hash embedder, never the production corpus.
