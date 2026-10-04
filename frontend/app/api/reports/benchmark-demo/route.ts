import { NextResponse } from "next/server";
import { readReport } from "@/lib/server-data";
import type { DemoBenchmarkReport } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  const report = await readReport<DemoBenchmarkReport>("reports/demo/benchmark_100_simulated.json");
  if (!report || report.status !== "SIMULATED" || report.executed !== false) {
    return NextResponse.json({ detail: "The simulated benchmark fixture is unavailable or invalid." }, { status: 404 });
  }
  return NextResponse.json(report, { headers: { "cache-control": "no-store" } });
}
