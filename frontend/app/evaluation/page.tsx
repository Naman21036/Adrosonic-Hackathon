"use client";

import { Fragment, useEffect, useState } from "react";
import { AlertTriangle, BookOpenCheck, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, Gauge, Sparkles } from "lucide-react";
import { PageHeading, SectionTitle, StatusPill, Surface, formatInteger, formatMetric } from "@/components/ui";
import { PilotRetrievalChart } from "@/components/charts";
import { useResource } from "@/lib/use-resource";
import { completeDemoStep } from "@/lib/demo-progress";
import type { EvaluationReport } from "@/lib/types";

type ModeTab = "all" | "dense" | "sparse" | "hybrid";

export default function EvaluationPage() {
  const [mode, setMode] = useState<ModeTab>("all");
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);
  const { data, loading, error } = useResource<EvaluationReport>(`/api/reports/evaluation?mode=${mode}&limit=10&offset=${page * 10}`);
  const precision = data?.aggregate.hybrid?.context_precision ?? null;
  const recall = data?.aggregate.hybrid?.context_recall ?? null;
  const dense = data?.aggregate.dense;

  useEffect(() => {
    if (data?.status === "complete" && precision !== null && recall !== null) completeDemoStep("ragas");
  }, [data?.status, precision, recall]);
  const totalPages = Math.max(1, Math.ceil((data?.totalRows ?? 0) / 10));
  const statusTone = data?.status === "complete" ? "good" : data?.status === "running" ? "blue" : data?.status === "partial" ? "warn" : "neutral";

  return (
    <div className="stack">
      <PageHeading eyebrow="QUALITY & GROUNDING" title="Retrieval Evaluation" description="Latest completed matched-corpus evaluation with stored RAGAS scores and retrieval-quality results." action={<StatusPill tone={statusTone}>{data?.status?.toUpperCase() ?? (loading ? "LOADING" : "UNAVAILABLE")}</StatusPill>} />

      {data?.evaluationError ? <div className="notice notice-warn"><AlertTriangle size={15} /><span>Latest evaluation stopped before scoring: {data.evaluationError}</span></div> : null}

      <div className="grid grid-3">
        <EvaluationKpi label="Hybrid Context Precision" value={precision === null ? "—" : precision.toFixed(4)} description={precision === null ? "No completed judge score" : "Latest completed matched-corpus result"} icon={<Gauge size={16} />} />
        <EvaluationKpi label="Hybrid Context Recall" value={recall === null ? "—" : recall.toFixed(4)} description={recall === null ? "No completed judge score" : "Latest completed matched-corpus result"} icon={<BookOpenCheck size={16} />} tone="gold" />
        <EvaluationKpi label="Valid queries" value={`${formatInteger(data?.validQueryCount)} / ${formatInteger(data?.queryCount)}`} description={`${formatInteger(data?.totalRows)} query-mode scores`} icon={<CheckCircle2 size={16} />} tone="mint" />
      </div>

      <Surface>
        <SectionTitle title="Latest completed RAGAS scores" detail={`Matched-corpus evaluation · ${data?.validQueryCount ?? "—"} queries · Context Precision and Context Recall`} action={<div className="badge-row"><StatusPill tone="good">LATEST COMPLETED EVALUATION</StatusPill><StatusPill tone="blue">MATCHED CORPUS</StatusPill></div>} />
        <div className="table-wrap"><table className="data-table"><thead><tr><th>Mode</th><th>Context Precision</th><th>Context Recall</th><th>Successful evaluations</th></tr></thead><tbody>
          {(["dense", "sparse", "hybrid"] as const).map((mode) => {
            const result = data?.aggregate[mode];
            const label = mode === "sparse" ? "BM25 / Sparse" : mode === "hybrid" ? "Hybrid / RRF" : "Dense";
            return <tr key={mode}><td>{label}</td><td className="metric-value">{result?.context_precision == null ? "—" : result.context_precision.toFixed(4)}</td><td className="metric-value">{result?.context_recall == null ? "—" : result.context_recall.toFixed(4)}</td><td>{formatInteger(result?.evaluation_successes)}</td></tr>;
          })}
        </tbody></table></div>
        <div className="ragas-highlight-row">
          <div className="ragas-highlight dense"><span>Dense</span><strong>{dense?.context_precision == null ? "—" : `${(dense.context_precision * 100).toFixed(2)}%`} <small>precision</small></strong><strong>{dense?.context_recall == null ? "—" : `${(dense.context_recall * 100).toFixed(2)}%`} <small>recall</small></strong></div>
          <div className="ragas-highlight hybrid"><span>Hybrid / RRF</span><strong>{precision === null ? "—" : `${(precision * 100).toFixed(2)}%`} <small>precision</small></strong><strong>{recall === null ? "—" : `${(recall * 100).toFixed(2)}%`} <small>recall</small></strong></div>
        </div>
      </Surface>

      <div className="grid grid-2">
        <Surface>
          <SectionTitle title="RAGAS judge" detail="Provider and model are read from the completed evaluation report." action={<StatusPill tone={data?.judgeEnabled && data.status === "complete" ? "good" : "warn"}>{data?.judgeEnabled && data.status === "complete" ? "EVALUATION COMPLETE" : "CHECK REPORT"}</StatusPill>} />
          <div className="config-row"><span>Judge provider</span><strong>{data?.judgeProvider ?? "—"}</strong></div>
          <div className="config-row"><span>Judge model</span><strong>{data?.judgeModel ?? "—"}</strong></div>
          <div className="config-row"><span>Judge enabled</span><strong>{data?.judgeEnabled === null || data?.judgeEnabled === undefined ? "Unknown" : data.judgeEnabled ? "Yes" : "No"}</strong></div>
          <div className="config-row"><span>Evaluation state</span><strong>{data?.status ?? "Report unavailable"}</strong></div>
          <div className="config-row"><span>Completed at</span><strong>{data?.completedAt ? new Date(data.completedAt).toLocaleString() : "Not reported"}</strong></div>
          <div className="notice notice-info mt-16"><Sparkles size={15} /><span>Hybrid achieves {precision === null ? "—" : precision.toFixed(4)} precision and {recall === null ? "—" : recall.toFixed(4)} recall. Scores are reported per mode; Dense and Hybrid are not ranked as universally better.</span></div>
        </Surface>
        <Surface>
          <SectionTitle title="Evaluation dataset" detail="Provenance is preserved from the local evaluation manifest." />
          <div className="config-row"><span>Dataset</span><strong>{data?.datasetSource ?? "—"}</strong></div>
          <div className="config-row"><span>Classification</span><strong>{data?.datasetClassification ?? "—"}</strong></div>
          <div className="config-row"><span>Official MS MARCO</span><strong>{data?.officialMsMarco === null || data?.officialMsMarco === undefined ? "Unknown" : data.officialMsMarco ? "Yes · cached validation split" : "No · synthetic matched-corpus queries"}</strong></div>
          <div className="config-row"><span>Query count</span><strong>{formatInteger(data?.validQueryCount)}</strong></div>
          <div className="config-row"><span>Retrieval setup</span><strong>Top K {data?.retrievalConfig.top_k ?? "—"} · RRF K {data?.retrievalConfig.rrf_k ?? "—"}</strong></div>
        </Surface>
      </div>

      <Surface>
        <SectionTitle title="10-query matched-corpus retrieval pilot" detail="Retrieval hit rates and MRR from the same pilot query set; separate from the 100-query latency benchmark." action={<StatusPill tone="blue">PILOT</StatusPill>} />
        <PilotRetrievalChart metrics={data?.retrievalMetrics ?? {}} />
        <div className="pilot-legend"><span><i className="legend-swatch swatch-gold" />Hit@1</span><span><i className="legend-swatch swatch-blue" />Hit@5</span><span><i className="legend-swatch swatch-mint" />Hit@10</span><span><i className="legend-line" />MRR</span></div>
      </Surface>

      <div className="notice notice-warn"><AlertTriangle size={15} /><span>There is no completed Phase 1 judge report in the stored artifacts, so this page makes no Phase 1 versus Phase 2 RAGAS comparison.</span></div>

      <Surface>
        <SectionTitle title="Per-query evaluation" detail="Stored retrieval IDs and query metadata. Context metrics are only shown when present in the report." action={<div className="compact-select-wrap"><label htmlFor="eval-mode" className="sr-only">Filter evaluation mode</label><select id="eval-mode" className="select-field compact-select" value={mode} onChange={(event) => { setMode(event.target.value as ModeTab); setPage(0); }}><option value="all">All modes</option><option value="dense">Dense</option><option value="sparse">BM25</option><option value="hybrid">Hybrid</option></select></div>} />
        {error ? <div className="notice notice-error">{error}</div> : null}
        {loading && !data ? <div className="skeleton-table"><span /><span /><span /><span /></div> : null}
        {data && data.rows.length > 0 ? <>
          <div className="table-wrap"><table className="data-table"><thead><tr><th>Query ID</th><th>Mode</th><th>Retrieved</th><th>Context Precision</th><th>Context Recall</th><th>Status</th><th /></tr></thead><tbody>{data.rows.map((row) => {
            const key = `${row.mode}:${row.queryId}`;
            const isOpen = expanded === key;
            const modeLabel = row.mode === "sparse" ? "BM25" : row.mode;
            const rowStatus = row.errors.length ? "Retrieval issue" : row.contextPrecision !== null || row.contextRecall !== null ? "Scored" : "Judge pending";
            return <Fragment key={key}><tr className="click-row" onClick={() => setExpanded(isOpen ? null : key)}><td className="mono">{row.queryId}</td><td><span className={`mode-tag tag-${row.mode}`}>{modeLabel}</span></td><td>{row.passageIds.length} passages</td><td className="metric-value">{formatMetric(row.contextPrecision)}</td><td className="metric-value">{formatMetric(row.contextRecall)}</td><td><span className={`row-status ${row.errors.length ? "row-bad" : row.contextPrecision !== null ? "row-good" : "row-warn"}`}>{rowStatus}</span></td><td><button className="icon-button table-expand" aria-label={isOpen ? "Collapse row" : "Expand row"}><ChevronDown size={14} className={isOpen ? "rotate-180" : ""} /></button></td></tr>{isOpen ? <tr><td colSpan={7}><div className="query-detail"><div><span>Query</span><p>{row.query}</p></div>{row.reference ? <div><span>Reference answer</span><p>{row.reference}</p></div> : null}<div><span>Retrieved passage IDs</span><p className="mono-wrap">{row.passageIds.length ? row.passageIds.join(" · ") : "None stored"}</p></div>{row.errors.length ? <div className="error-text">{row.errors.join(" · ")}</div> : null}</div></td></tr> : null}</Fragment>;
          })}</tbody></table></div>
          <div className="pagination"><span>Showing {data.offset + 1}–{Math.min(data.offset + data.rows.length, data.totalRows)} of {data.totalRows}</span><div><button className="button button-quiet" disabled={page === 0 || loading} onClick={() => setPage((value) => Math.max(0, value - 1))}><ChevronLeft size={14} />Previous</button><span className="page-count">{page + 1} / {totalPages}</span><button className="button button-quiet" disabled={page + 1 >= totalPages || loading} onClick={() => setPage((value) => value + 1)}>Next<ChevronRight size={14} /></button></div></div>
        </> : !loading ? <div className="pending-box"><div><strong>Evaluation rows unavailable</strong><span>The API does not expose per-query evaluation data and the local report has no scored rows.</span></div></div> : null}
      </Surface>
    </div>
  );
}

function EvaluationKpi({ label, value, description, icon, tone = "blue" }: { label: string; value: string; description: string; icon: React.ReactNode; tone?: "blue" | "gold" | "mint" }) {
  return <article className={`stat-card stat-${tone}`}><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><div className={`stat-value ${value === "Pending" ? "pending-metric" : ""}`}>{value}</div><div className="stat-detail">{description}</div></article>;
}
