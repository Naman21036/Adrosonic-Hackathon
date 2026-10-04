import { NextResponse } from "next/server";
import { asNumber, asString, readReport } from "@/lib/server-data";
import type { FilterDemoReport } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  const report = await readReport<Record<string, unknown>>("reports/phase2/metadata_filter_migration.json");
  const output: FilterDemoReport = {
    status: asString(report?.status) ?? "not_run",
    sourceId: asNumber(report?.demo_source_id) ?? asNumber(report?.selected_source_id),
    sourcePassages: asNumber(report?.selected_source_passages),
    payloadIndexCreated: report?.qdrant_payload_index_created === true,
    payloadIndexLimitation: asString(report?.qdrant_payload_index_limitation),
  };
  return NextResponse.json(output, { headers: { "cache-control": "no-store" } });
}
