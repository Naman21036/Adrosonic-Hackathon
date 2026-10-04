import { NextRequest, NextResponse } from "next/server";
import { asNumber, asRecord, asString, readJsonLines, readReport } from "@/lib/server-data";
import type { BenchmarkReport, BenchmarkSample, LatencySummary } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const summary = await readReport<Record<string, unknown>>("reports/phase2/benchmarks/benchmark_summary.json");
  const rawRows = await readJsonLines("reports/phase2/benchmarks/benchmark.jsonl");
  const requestedMode = request.nextUrl.searchParams.get("mode");
  const samples: BenchmarkSample[] = rawRows.map(asRecord).flatMap((row) => {
    const mode = asString(row.mode);
    const queryNumber = asNumber(row.query_number);
    const latency = asNumber(row.latency_ms);
    if (!mode || queryNumber === null || latency === null) return [];
    return [{ mode, query_number: queryNumber, latency_ms: latency, error: asString(row.error) }];
  }).filter((row) => !requestedMode || requestedMode === "all" || row.mode === requestedMode);

  const latencyRaw = asRecord(summary?.latency_ms);
  const latency = Object.fromEntries(Object.entries(latencyRaw).map(([mode, value]) => [mode, asRecord(value) as LatencySummary]));
  const output: BenchmarkReport = {
    status: asString(summary?.status) ?? "unavailable",
    queryCount: asNumber(summary?.query_count),
    requestedQueries: asNumber(summary?.requested_queries),
    querySource: asString(summary?.query_source),
    retrievalConfig: Object.fromEntries(Object.entries(asRecord(summary?.retrieval_config)).flatMap(([key, value]) => {
      const number = asNumber(value);
      return number === null ? [] : [[key, number]];
    })),
    latency,
    environment: Object.fromEntries(Object.entries(asRecord(summary?.environment)).flatMap(([key, value]) => {
      if (typeof value === "string" || typeof value === "number" || value === null) return [[key, value]];
      return [];
    })),
    samples,
  };
  return NextResponse.json(output, { headers: { "cache-control": "no-store" } });
}
