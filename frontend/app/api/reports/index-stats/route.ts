import { NextResponse } from "next/server";
import { asBoolean, asNumber, asRecord, asString, readReport } from "@/lib/server-data";
import type { IndexStats } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  const [snapshot, validation, benchmark] = await Promise.all([
    readReport<Record<string, unknown>>("reports/phase2/frontend_index_snapshot.json"),
    readReport<Record<string, unknown>>("reports/phase2/corpus_validation.json"),
    readReport<Record<string, unknown>>("reports/phase2/benchmarks/benchmark_summary.json"),
  ]);
  const environment = asRecord(benchmark?.environment);
  const result: IndexStats = {
    qdrant_count: asNumber(snapshot?.qdrant_count) ?? asNumber(validation?.qdrant_count),
    sparse_count: asNumber(snapshot?.sparse_count) ?? asNumber(validation?.sparse_count),
    qdrant_unique_ids: asNumber(snapshot?.qdrant_unique_ids) ?? asNumber(validation?.qdrant_unique_ids),
    sparse_unique_ids: asNumber(snapshot?.sparse_unique_ids) ?? asNumber(validation?.sparse_unique_ids),
    source_query_count: asNumber(snapshot?.source_query_count),
    dimension: asNumber(snapshot?.dimension) ?? asNumber(environment.embedding_dimension) ?? asNumber(validation?.embedding_dimension),
    model: asString(snapshot?.model) ?? asString(environment.embedding_model),
    collection: asString(snapshot?.collection),
    ids_match: asBoolean(snapshot?.ids_match) ?? asBoolean(validation?.ids_match),
    snapshot_as_of: asString(snapshot?.observed_at_utc) ?? asString(validation?.validated_at_utc),
    validation_status: asString(snapshot?.count_and_id_validation) ?? asString(validation?.status),
    validation_error: asString(snapshot?.validation_error),
  };
  return NextResponse.json(result, { headers: { "cache-control": "no-store" } });
}
