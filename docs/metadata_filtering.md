# FR-4: Metadata Filtering

## Metadata dimension

The corpus stores Qdrant point payloads with `passage_id`, `text`, and a nested
`metadata` object (`source`, `split`, and `relevance`). The corpus inspection
found no existing `source_id` field. Its passage IDs use the verified canonical
shape `msmarco-{numeric query_id}-{numeric passage_position}`. FR-4 therefore
uses the integer query ID as `source_id` and adds it as a top-level Qdrant
payload field. Vectors, texts, IDs, and existing payload metadata are preserved.

The application requests an integer payload index on top-level `source_id`.
The installed Qdrant Local backend accepts the API call but implements payload
index creation as a no-op, so this backend cannot provide a physical payload
index or its filtering speedup. The backfill is idempotent and has an explicit preflight that
requires the dense and sparse IDs to match the expected corpus before it writes
anything. It verifies point IDs, passage IDs, text digests, vector digests,
count, dimension, and distance after the payload-only update.

Run the one-time payload backfill with:

```powershell
.\venv\Scripts\python.exe -m app.backfill_source_id --expected-count 89896 --source-id 19699 --apply
```

This command does not instantiate Qwen, re-embed passages, recreate the
collection, or write vectors. It writes only derived integer `source_id`
payload values and requests the Qdrant payload index. The companion BM25 filter uses an
SQLite expression index over the already-encoded passage ID; it leaves the
passage rows and FTS5 index unchanged.

## Query API

Use the singular `filter` field for the FR-4 source filter:

```json
{
  "query": "why do nails get rusty",
  "mode": "hybrid",
  "top_k": 5,
  "filter": {"source_id": 19699}
}
```

`source_id` must be a non-negative JSON integer. An unknown source ID returns
an empty `results` list; it never falls back to unfiltered retrieval. An empty
or omitted filter searches the full eligible index. The older generic `filters`
field remains accepted for callers already using metadata predicates.

The response includes `filter` (and retains `filters` as a compatibility field)
and each result includes its numeric `source_id` alongside the existing
passage ID, text, score, and metadata.

## Pre-retrieval enforcement

Dense retrieval constructs a Qdrant `Filter` on the indexed `source_id` field
and passes it as `query_filter` to `query_points`. Qdrant applies the metadata
constraint during similarity search, before selecting dense candidates.
BM25 adds a source-ID predicate to its SQL candidate query before its BM25
ordering and `LIMIT`; an expression index avoids a corpus scan. Hybrid retrieval
passes the same filter to both candidate retrievers before RRF. A defensive
check raises on a backend result that violates the requested source ID; it
does not post-filter results.

**Live demo explanation:** “The metadata constraint is applied to Qdrant before
similarity search, so documents outside the selected source are never
considered by the dense retrieval stage.”

## Demo

Start the API and Streamlit UI using the project’s normal commands. In
`Metadata filters`, enable **Filter before retrieval by source_id**, enter an
existing numeric source ID (the input defaults to `19699`), and run Dense,
Hybrid, or Compare. The response displays the selected filter and each result’s
`source_id`. With the filter disabled, the same query searches the full
eligible corpus.

The deterministic source ID and selected demo query are recorded in
`reports/phase2/metadata_filter_migration.json`. The migration probe selects a
real source with multiple passages, uses one of its stored vectors as the dense
smoke-test query vector, and verifies the Qdrant filter object was supplied to
the similarity query. It also records filtered/unfiltered latency and result
source IDs.

## Logging and tracing

Structured logs record `filter_enabled`, `filter_dimension`, `filter_value`,
`retrieval_mode`, `top_k`, and `qdrant_filter_applied`. Langfuse receives the
filter in the request trace input and metadata. Existing secret-redaction rules
remain in effect.

## Tests

Run focused filtering tests:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_metadata_filter.py -q -p no:cacheprovider
```

These verify strict input validation, Qdrant `query_filter` forwarding, the
indexed BM25 pre-limit predicate, filtered dense/BM25/hybrid behavior, the
empty result for nonexistent source IDs, legacy/no-filter API behavior, and
filter logging/trace metadata.
