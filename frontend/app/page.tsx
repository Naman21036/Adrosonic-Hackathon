"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BookOpenCheck,
  BrainCircuit,
  Check,
  Clock3,
  Cpu,
  Database,
  Gauge,
  Layers3,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { PilotRetrievalChart, RetrievalMetricsChart } from "@/components/charts";
import { SectionTitle, StatusPill, Surface, formatInteger, formatMs } from "@/components/ui";
import { useResource } from "@/lib/use-resource";
import { useSystemStatus } from "@/components/system-provider";
import type { BenchmarkReport, EvaluationReport, RetrievalModeMetrics } from "@/lib/types";

const modes = [
  { key: "dense", label: "Dense", className: "tag-dense" },
  { key: "sparse", label: "BM25 / Sparse", className: "tag-sparse" },
  { key: "hybrid", label: "Hybrid / RRF", className: "tag-hybrid" },
] as const;

const pipeline = [
  { title: "QUERY", detail: "Natural language", icon: Search },
  { title: "QWEN EMBEDDING", detail: "1024 dimensions", icon: BrainCircuit },
  { title: "QDRANT + BM25", detail: "Dense + sparse", icon: Database },
  { title: "RRF FUSION", detail: "k = 60", icon: Layers3 },
  { title: "TOP-K CONTEXT", detail: "5 passages", icon: ShieldCheck },
  { title: "AZURE LLM", detail: "Grounded answer", icon: Sparkles },
];

function metricText(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percentText(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function connectionLabel(connectivity: string): { label: string; good: boolean; tone: "good" | "warn" | "bad" | "neutral" } {
  if (connectivity === "ONLINE") return { label: "API reachable", good: true, tone: "good" };
  if (connectivity === "CHECKING") return { label: "Checking", good: false, tone: "neutral" };
  if (connectivity === "TIMEOUT") return { label: "API timeout", good: false, tone: "warn" };
  if (connectivity === "ERROR") return { label: "API error", good: false, tone: "warn" };
  return { label: "API Offline", good: false, tone: "bad" };
}

function serviceState(value: boolean | undefined, online: boolean): { label: string; good: boolean; neutral: boolean } {
  if (value === true) return { label: "Ready", good: true, neutral: false };
  if (value === false && online) return { label: "Not ready", good: false, neutral: false };
  return { label: "Not exposed while API is offline", good: false, neutral: true };
}

function modeName(key: string): string {
  return modes.find((mode) => mode.key === key)?.label ?? key;
}

export default function OverviewPage() {
  const { health, status, indexStats, apiOnline } = useSystemStatus();
  const { data: evaluation } = useResource<EvaluationReport>("/api/reports/evaluation?limit=1");
  const { data: benchmark } = useResource<BenchmarkReport>("/api/reports/benchmark?mode=all");
  const apiState = connectionLabel(health.connectivity);
  const model = status?.model ?? indexStats?.model ?? "—";
  const dimension = status?.dimension ?? indexStats?.dimension ?? null;
  const sourceCount = status?.source_query_count ?? indexStats?.source_query_count ?? null;
  const passageCount = status?.indexed_passages ?? indexStats?.qdrant_count ?? null;
  const dataSource = status?.data_source === "live" ? "Live FastAPI status" : "Last recorded index snapshot";
  const sourceDate = status?.snapshot_as_of ? ` · ${new Date(status.snapshot_as_of).toLocaleDateString()}` : "";
  const topK = evaluation?.retrievalConfig.top_k ?? benchmark?.retrievalConfig.top_k ?? null;
  const retrievalModes = Object.keys(evaluation?.retrievalMetrics ?? {}).length;
  const serviceRows: { name: string; label: string; good: boolean; neutral: boolean; icon: LucideIcon }[] = [
    { name: "FastAPI", ...apiState, neutral: false, icon: Activity },
    { name: "Qdrant", ...serviceState(health.qdrant_ready, apiOnline), icon: Database },
    { name: "BM25", ...serviceState(health.sparse_ready, apiOnline), icon: Layers3 },
    { name: "Embeddings", ...serviceState(health.embedding_model_ready, apiOnline), icon: Cpu },
    {
      name: "Azure AI",
      label: health.azure_ready === true ? "Configured · connectivity unverified" : health.azure_ready === false ? "Not configured" : "Not exposed while API is offline",
      good: false,
      neutral: health.azure_ready === undefined,
      icon: Sparkles,
    },
    { name: "Langfuse", label: "Not exposed by the backend", good: false, neutral: true, icon: Activity },
  ];

  return (
    <div className="stack">
      <section className="hero-panel">
        <div className="hero-content">
          <div className="hero-kicker"><span className="eyebrow-dot" />ADROSONIC · RETRIEVAL INTELLIGENCE</div>
          <h1>Precision Retrieval<br />for Production RAG.</h1>
          <p>Dense + BM25 + RRF + metadata filtering. Explore the live retrieval pipeline and inspect the evidence behind every result.</p>
          <div className="hero-actions">
            <Link className="button button-primary" href="/playground"><Search size={15} />Open Playground<ArrowRight size={14} /></Link>
            <Link className="button button-quiet" href="/architecture">View architecture<ArrowUpRight size={14} /></Link>
          </div>
        </div>
        <div className="hero-diagram" aria-hidden="true"><span className="hero-node"><BrainCircuit size={19} /></span><span className="hero-connector" /><span className="hero-node"><Database size={19} /></span><span className="hero-connector" /><span className="hero-node gold"><Layers3 size={19} /></span></div>
      </section>

      <div className="grid grid-6">
        <MetricCard label="Indexed Passages" value={formatInteger(passageCount)} detail={`${dataSource}${sourceDate}`} icon={<Database size={15} />} delay={0.02} />
        <MetricCard label="Source Query IDs" value={formatInteger(sourceCount)} detail="Distinct IDs in the read-only sparse index snapshot" icon={<Network size={15} />} tone="gold" delay={0.06} />
        <MetricCard label="Embedding Model" value={model} detail="Full model identifier" icon={<BrainCircuit size={15} />} tone="violet" delay={0.1} />
        <MetricCard label="Vector Dimension" value={dimension ?? "—"} detail="Qdrant vector dimension" icon={<Layers3 size={15} />} delay={0.14} />
        <MetricCard label="Top K" value={topK ?? "—"} detail="From the completed retrieval configuration" icon={<Gauge size={15} />} tone="mint" delay={0.18} />
        <MetricCard label="Retrieval Modes" value={retrievalModes || "—"} detail="Dense · BM25 · Hybrid / RRF" icon={<Zap size={15} />} tone="gold" delay={0.22} />
      </div>

      <Surface>
        <SectionTitle title="Live system status" detail="Each service is shown from its own reported field. A cached index snapshot is clearly marked when FastAPI is offline." action={<StatusPill tone={apiState.tone}>{apiState.label}</StatusPill>} />
        <div className="grid grid-3">
          {serviceRows.map((item, index) => <motion.div key={item.name} className="system-status-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .035, duration: .22 }}><span className="system-status-icon"><item.icon size={16} /></span><span className="system-status-copy"><strong>{item.name}</strong><small>{item.label}</small></span><span className={`status-light-lg ${item.good ? "on" : item.neutral ? "neutral" : "off"}`} /></motion.div>)}
        </div>
        <details className="connection-details">
          <summary>Connection details</summary>
          <dl>
            <div><dt>FastAPI health endpoint</dt><dd>{health.endpoint ?? "Waiting for first check"}</dd></div>
            <div><dt>Connection state</dt><dd>{health.connectivity}{health.http_status ? ` · HTTP ${health.http_status}` : ""}</dd></div>
            <div><dt>Last checked</dt><dd>{health.checked_at ? new Date(health.checked_at).toLocaleString() : "Not checked"}</dd></div>
            {health.error ? <div><dt>Error</dt><dd>{health.error}</dd></div> : null}
            {status?.validation_error ? <div><dt>Last full validation</dt><dd>{status.validation_error}</dd></div> : null}
          </dl>
        </details>
        {!apiOnline ? <div className="notice notice-warn mt-16"><AlertTriangle size={15} /><span>The health check reports {health.connectivity.toLowerCase()}. Index counts below come from the saved count/ID snapshot; they are not a live connection claim.</span></div> : null}
      </Surface>

      <Surface>
        <SectionTitle title="Retrieval pipeline" detail="From question to grounded context. Stages illuminate as live requests complete." />
        <div className="pipeline">
          {pipeline.map(({ title, detail, icon: Icon }, index) => <motion.div key={title} className="pipeline-node" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .045, duration: .22 }}><Icon size={15} /><strong>{title}</strong><small>{detail}</small></motion.div>)}
        </div>
      </Surface>

      <Surface>
        <SectionTitle title="Latest completed RAGAS evaluation" detail={`${evaluation?.validQueryCount ?? "—"} matched-corpus queries · RAGAS ${evaluation?.ragasVersion ?? "—"}`} action={<div className="badge-row"><StatusPill tone={evaluation?.status === "complete" ? "good" : "neutral"}>{evaluation?.status === "complete" ? "LATEST COMPLETED EVALUATION" : "NO COMPLETED REPORT"}</StatusPill><StatusPill tone="blue">MATCHED CORPUS</StatusPill></div>} />
        {evaluation?.officialMsMarco === false ? <p className="fine-print">Synthetic questions grounded in active indexed passages; these are not official MS MARCO queries.</p> : null}
        <div className="table-wrap">
          <table className="data-table"><thead><tr><th>Mode</th><th>Context Precision</th><th>Context Recall</th></tr></thead><tbody>
            {modes.map((mode) => {
              const metrics = evaluation?.aggregate[mode.key];
              return <tr key={mode.key}><td><span className={`mode-tag ${mode.className}`}>{mode.label}</span></td><td className="metric-value">{metricText(metrics?.context_precision)}</td><td className="metric-value">{metricText(metrics?.context_recall)}</td></tr>;
            })}
          </tbody></table>
        </div>
        <div className="ragas-highlight-row">
          {(["dense", "hybrid"] as const).map((mode) => {
            const metrics = evaluation?.aggregate[mode];
            return <div className={`ragas-highlight ${mode}`} key={mode}><span>{modeName(mode)} {mode === "hybrid" ? "· RRF" : ""}</span><strong>{percentText(metrics?.context_precision)} <small>precision</small></strong><strong>{percentText(metrics?.context_recall)} <small>recall</small></strong></div>;
          })}
        </div>
        <p className="fine-print">Hybrid achieves {metricText(evaluation?.aggregate.hybrid?.context_precision)} precision and {metricText(evaluation?.aggregate.hybrid?.context_recall)} recall. Dense and hybrid results are shown for comparison; neither is described as universally better.</p>
      </Surface>

      <div className="grid grid-2">
        <Surface>
          <SectionTitle title="10-query matched-corpus pilot" detail="Retrieval-quality results from the completed matched-corpus pilot; not the 100-query benchmark." action={<StatusPill tone="blue">PILOT</StatusPill>} />
          <PilotRetrievalChart metrics={evaluation?.retrievalMetrics ?? {}} />
          <div className="pilot-legend"><span><i className="legend-swatch swatch-gold" />Hit@1</span><span><i className="legend-swatch swatch-blue" />Hit@5</span><span><i className="legend-swatch swatch-mint" />Hit@10</span><span><i className="legend-line" />MRR</span></div>
        </Surface>
        <Surface>
          <SectionTitle title="Pilot retrieval metrics" detail="Stored hit rates and MRR from the same 10-query pilot." />
          <RetrievalMetricsChart metrics={evaluation?.retrievalMetrics ?? {}} />
          <div className="table-wrap compact-metrics-table"><table className="data-table"><thead><tr><th>Mode</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th></tr></thead><tbody>
            {modes.map((mode) => <RetrievalMetricRow key={mode.key} mode={mode.label} metrics={evaluation?.retrievalMetrics[mode.key]} className={mode.className} />)}
          </tbody></table></div>
        </Surface>
      </div>

      <Surface>
        <SectionTitle title="Measured retrieval latency" detail={`${formatInteger(benchmark?.queryCount)} actual requests · ${benchmark?.querySource ?? "Stored benchmark report"}`} action={<Link href="/benchmark" className="inline-link">Open latency charts<ArrowRight size={13} /></Link>} />
        <div className="table-wrap"><table className="data-table"><thead><tr><th>Mode</th><th>Mean</th><th>P50</th><th>P95</th><th>P99</th><th>Min</th><th>Max</th></tr></thead><tbody>
          {modes.map((mode) => {
            const stats = benchmark?.latency[mode.key];
            return <tr key={mode.key}><td><span className={`mode-tag ${mode.className}`}>{mode.label}</span></td><td>{formatMs(stats?.mean_ms)}</td><td>{formatMs(stats?.p50_ms)}</td><td>{formatMs(stats?.p95_ms)}</td><td>{formatMs(stats?.p99_ms)}</td><td>{formatMs(stats?.min_ms)}</td><td>{formatMs(stats?.max_ms)}</td></tr>;
          })}
        </tbody></table></div>
        <p className="fine-print">Source: completed 100-query retrieval benchmark. These are end-to-end mode measurements; no per-stage timing is inferred from them.</p>
      </Surface>

      <div className="grid grid-3">
        <Link href="/playground" className="launch-card"><span className="launch-icon"><Search size={17} /></span><span><strong>Run a live query</strong><small>Compare dense and hybrid retrieval.</small></span><ArrowRight size={15} /></Link>
        <Link href="/evaluation" className="launch-card"><span className="launch-icon gold-icon"><BookOpenCheck size={17} /></span><span><strong>Review evaluation</strong><small>Inspect the completed RAGAS report and provenance.</small></span><ArrowRight size={15} /></Link>
        <Link href="/benchmark" className="launch-card"><span className="launch-icon mint-icon"><Clock3 size={17} /></span><span><strong>Inspect benchmark</strong><small>100-query latency chart, per retrieval mode.</small></span><ArrowRight size={15} /></Link>
      </div>

      <div className="pilot-footnote"><Check size={13} />RAGAS and retrieval quality are from the 10-query matched-corpus pilot; latency is from the separate completed 100-query benchmark.</div>
    </div>
  );
}

function RetrievalMetricRow({ mode, metrics, className }: { mode: string; metrics?: RetrievalModeMetrics; className: string }) {
  return <tr><td><span className={`mode-tag ${className}`}>{mode}</span></td><td>{percentText(metrics?.hit_at_1, 0)}</td><td>{percentText(metrics?.hit_at_5, 0)}</td><td>{percentText(metrics?.hit_at_10, 0)}</td><td>{metricText(metrics?.mrr, 3)}</td></tr>;
}

function MetricCard({ label, value, detail, icon, tone = "blue", delay }: { label: string; value: string | number; detail: string; icon: React.ReactNode; tone?: "blue" | "gold" | "mint" | "violet"; delay: number }) {
  return <motion.article className={`stat-card stat-${tone}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .25, delay }}><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><div className={`stat-value${label === "Embedding Model" ? " small-stat-value" : ""}`}>{value}</div><div className="stat-detail">{detail}</div></motion.article>;
}
