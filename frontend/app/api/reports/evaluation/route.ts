import { NextRequest, NextResponse } from "next/server";
import { asNumber, asRecord, asString, readReport } from "@/lib/server-data";
import type { EvaluationModeResult, EvaluationReport, EvaluationRow, RetrievalModeMetrics } from "@/lib/types";

export const dynamic = "force-dynamic";

const modes = ["dense", "sparse", "hybrid"] as const;
type EvaluationMode = typeof modes[number];

function metricValue(value: unknown): number | null {
  return asNumber(value);
}

function textArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

export async function GET(request: NextRequest) {
  const report = await readReport<Record<string, unknown>>("reports/phase2/evaluation/ragas_matched_corpus.json");
  const retrievalRows = Array.isArray(report?.retrieval_results) ? report.retrieval_results.map(asRecord) : [];
  const retrievalById = new Map<string, Record<string, unknown>>();
  for (const row of retrievalRows) {
    const queryId = asString(row.query_id);
    if (queryId) retrievalById.set(queryId, row);
  }

  const rawPerQuery = asRecord(report?.per_query_metrics);
  const metricRowsByMode = new Map<EvaluationMode, Map<string, Record<string, unknown>>>();
  for (const mode of modes) {
    const entries = rawPerQuery[mode];
    const byId = new Map<string, Record<string, unknown>>();
    if (Array.isArray(entries)) {
      for (const entry of entries.map(asRecord)) {
        const queryId = asString(entry.query_id);
        if (queryId) byId.set(queryId, entry);
      }
    }
    metricRowsByMode.set(mode, byId);
  }

  const rows: EvaluationRow[] = [];
  for (const mode of modes) {
    for (const [queryId, metrics] of metricRowsByMode.get(mode) ?? []) {
      const retrieved = retrievalById.get(queryId) ?? {};
      const retrievalModes = asRecord(retrieved.modes);
      const modeResult = asRecord(retrievalModes[mode]);
      rows.push({
        queryId,
        mode,
        query: asString(metrics.query) ?? asString(metrics.user_input) ?? asString(retrieved.query) ?? "Query text unavailable in the stored report",
        reference: asString(metrics.reference) ?? asString(metrics.ground_truth),
        passageIds: textArray(modeResult.top5_ids),
        errors: textArray(modeResult.errors),
        contextPrecision: metricValue(metrics.context_precision),
        contextRecall: metricValue(metrics.context_recall),
      });
    }
  }

  const requestedMode = request.nextUrl.searchParams.get("mode");
  const filteredRows = requestedMode && requestedMode !== "all" ? rows.filter((row) => row.mode === requestedMode) : rows;
  const limit = Math.min(25, Math.max(1, Number(request.nextUrl.searchParams.get("limit")) || 10));
  const offset = Math.max(0, Number(request.nextUrl.searchParams.get("offset")) || 0);
  const rawModes = asRecord(report?.modes);
  const aggregate: Record<string, EvaluationModeResult> = {};
  const perModeResults: Record<string, EvaluationModeResult> = {};
  const retrievalMetrics: Record<string, RetrievalModeMetrics> = {};

  for (const mode of modes) {
    const result = asRecord(rawModes[mode]);
    const metrics = asRecord(asRecord(report?.retrieval_metrics)[mode]);
    const evaluationResult: EvaluationModeResult = {
      retrieval_successes: asNumber(result.retrieval_successes),
      retrieval_failures: asNumber(result.retrieval_failures),
      context_precision: metricValue(result.context_precision),
      context_recall: metricValue(result.context_recall),
      evaluation_successes: asNumber(result.evaluation_successes),
      evaluation_failures: asNumber(result.evaluation_failures),
      status: asString(result.status) ?? undefined,
    };
    aggregate[mode] = evaluationResult;
    perModeResults[mode] = evaluationResult;
    retrievalMetrics[mode] = {
      hit_at_1: metricValue(metrics["hit@1"]),
      hit_at_5: metricValue(metrics["hit@5"]),
      hit_at_10: metricValue(metrics["hit@10"]),
      mrr: metricValue(metrics.mrr),
      queries_with_ground_truth_ranked: metricValue(metrics.queries_with_ground_truth_ranked),
    };
  }

  const dataset = asRecord(report?.dataset);
  const datasetClassification = typeof dataset.official_ms_marco === "boolean"
    ? dataset.official_ms_marco
      ? "Official/cached MS MARCO evaluation dataset"
      : "Matched-corpus pilot; synthetic queries grounded in indexed passages (not official MS MARCO)"
    : asString(dataset.classification) ?? asString(report?.evaluation_type);
  const output: EvaluationReport = {
    status: asString(report?.status) ?? "unavailable",
    evaluationType: asString(report?.evaluation_type),
    completedAt: asString(report?.completed_at_utc),
    datasetSource: asString(dataset.source),
    datasetClassification,
    officialMsMarco: typeof dataset.official_ms_marco === "boolean" ? dataset.official_ms_marco : null,
    queryCount: asNumber(report?.query_count),
    validQueryCount: asNumber(report?.valid_query_count),
    judgeProvider: asString(report?.judge_provider),
    judgeModel: asString(report?.judge_model),
    ragasVersion: asString(report?.ragas_version),
    judgeEnabled: typeof report?.judge_enabled === "boolean" ? report.judge_enabled : null,
    judgePreflightPassed: typeof report?.judge_preflight_passed === "boolean" ? report.judge_preflight_passed : null,
    retrievalConfig: Object.fromEntries(
      Object.entries(asRecord(report?.retrieval_config)).flatMap(([key, value]) => {
        const number = asNumber(value);
        return number === null ? [] : [[key, number]];
      }),
    ),
    perModeResults,
    retrievalMetrics,
    aggregate,
    evaluationError: asString(report?.evaluation_error),
    rows: filteredRows.slice(offset, offset + limit),
    totalRows: filteredRows.length,
    offset,
    limit,
  };
  return NextResponse.json(output, { headers: { "cache-control": "no-store" } });
}
