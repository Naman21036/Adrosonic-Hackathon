"use client";

import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BookOpenCheck,
  Database,
  Gauge,
  Layers3,
  Search,
} from "lucide-react";
import { PilotRetrievalChart, SimulatedLatencyDistribution } from "@/components/charts";
import { PageHeading, SectionTitle, StatCard, Surface, formatInteger, formatMs } from "@/components/ui";
import { useDemoMode } from "@/components/demo-mode-provider";
import { useResource } from "@/lib/use-resource";
import type { BenchmarkReport, DemoBenchmarkReport, EvaluationModeResult, EvaluationReport, RetrievalModeMetrics } from "@/lib/types";

const modes = [
  { key: "dense", label: "Dense", description: "Qwen Embeddings + Qdrant", color: "blue", icon: Search },
  { key: "sparse", label: "BM25", description: "Sparse lexical retrieval", color: "gold", icon: BookOpenCheck },
  { key: "hybrid", label: "Hybrid / RRF", description: "Dense + BM25 + RRF", color: "mint", icon: Layers3 },
] as const;

function fraction(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function decimal(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function count(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : formatInteger(value);
}

function environmentValue(value: string | number | null | undefined): string {
  if (typeof value === "number") return formatInteger(value);
  return value ?? "—";
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return <span className="benchmark-meta-item"><span>{label}</span><strong>{value}</strong></span>;
}

function AnimatedStat({ label, value, detail, icon, tone, index = 0 }: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  tone: "blue" | "gold" | "mint" | "violet";
  index?: number;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: index * 0.045 }}>
      <StatCard label={label} value={value} detail={detail} icon={icon} tone={tone} />
    </motion.div>
  );
}

function MetricPair({ label, value }: { label: string; value: string }) {
  return <div className="benchmark-mode-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function RagasModeCard({ mode, result, index }: {
  mode: typeof modes[number];
  result: EvaluationModeResult | undefined;
  index: number;
}) {
  const Icon = mode.icon;
  return (
    <motion.div initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: index * 0.055 }}>
      <Surface className="benchmark-ragas-card">
        <div className="benchmark-mode-card-heading"><span className={`benchmark-mode-icon tone-${mode.color}`}><Icon size={15} /></span><div><h3>{mode.label}</h3><p>Matched-corpus pilot</p></div></div>
        <div className="benchmark-ragas-metrics">
          <MetricPair label="Context Precision" value={fraction(result?.context_precision)} />
          <MetricPair label="Context Recall" value={fraction(result?.context_recall)} />
        </div>
      </Surface>
    </motion.div>
  );
}

function LatencyCard({ mode, latency, index }: {
  mode: typeof modes[number];
  latency: BenchmarkReport["latency"][string] | undefined;
  index: number;
}) {
  const Icon = mode.icon;
  return (
    <motion.div initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: index * 0.055 }}>
      <Surface className="benchmark-latency-card">
        <div className="benchmark-latency-heading"><span className={`benchmark-mode-icon tone-${mode.color}`}><Icon size={15} /></span><div><h3>{mode.label}</h3><small>{count(latency?.successful)} successful requests</small></div></div>
        <div className="benchmark-latency-grid">
          <MetricPair label="Average" value={formatMs(latency?.mean_ms)} />
          <MetricPair label="P50" value={formatMs(latency?.p50_ms)} />
          <MetricPair label="P95" value={formatMs(latency?.p95_ms)} />
          <MetricPair label="P99" value={formatMs(latency?.p99_ms)} />
        </div>
      </Surface>
    </motion.div>
  );
}

function PilotModeCard({
  mode,
  retrieval,
  ragas,
}: {
  mode: typeof modes[number];
  retrieval: RetrievalModeMetrics | undefined;
  ragas: EvaluationModeResult | undefined;
}) {
  const Icon = mode.icon;
  return (
    <Surface className="benchmark-mode-card">
      <div className="benchmark-mode-card-heading">
        <span className={`benchmark-mode-icon tone-${mode.color}`}><Icon size={15} /></span>
        <div><h3>{mode.label === "BM25" ? "Sparse Retrieval" : `${mode.label} Retrieval`}</h3><p>{mode.description}</p></div>
      </div>
      <div className="benchmark-mode-card-metrics">
        <MetricPair label="Context Precision" value={fraction(ragas?.context_precision)} />
        <MetricPair label="Context Recall" value={fraction(ragas?.context_recall)} />
        <MetricPair label="Hit@5" value={fraction(retrieval?.hit_at_5, 0)} />
        <MetricPair label="MRR" value={decimal(retrieval?.mrr)} />
      </div>
    </Surface>
  );
}

function RetrievalTable({ evaluation }: { evaluation: EvaluationReport | null }) {
  return (
    <div className="table-wrap">
      <table className="data-table benchmark-results-table">
        <thead><tr><th>Retrieval mode</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>Context Precision</th><th>Context Recall</th><th>RAGAS success</th></tr></thead>
        <tbody>
          {modes.map((mode) => {
            const retrieval = evaluation?.retrievalMetrics[mode.key];
            const ragas = evaluation?.aggregate[mode.key];
            return (
              <tr key={mode.key}>
                <td><span className={`mode-tag ${mode.key === "sparse" ? "tag-sparse" : mode.key === "hybrid" ? "tag-hybrid" : ""}`}>{mode.label}</span></td>
                <td className="benchmark-number">{fraction(retrieval?.hit_at_1)}</td>
                <td className="benchmark-number">{fraction(retrieval?.hit_at_5)}</td>
                <td className="benchmark-number">{fraction(retrieval?.hit_at_10)}</td>
                <td className="benchmark-number">{decimal(retrieval?.mrr)}</td>
                <td className="benchmark-number">{fraction(ragas?.context_precision)}</td>
                <td className="benchmark-number">{fraction(ragas?.context_recall)}</td>
                <td className="benchmark-number">{count(ragas?.evaluation_successes)} / {count(ragas?.evaluation_failures)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarkPage() {
  const { enabled: demoMode } = useDemoMode();
  const { data: benchmark, loading: benchmarkLoading, error: benchmarkError } = useResource<BenchmarkReport>("/api/reports/benchmark");
  const { data: evaluation, loading: evaluationLoading, error: evaluationError } = useResource<EvaluationReport>("/api/reports/evaluation?limit=1");
  const { data: demo, loading: demoLoading, error: demoError } = useResource<DemoBenchmarkReport>("/api/reports/benchmark-demo");

  const title = demoMode ? "100-Query Retrieval Benchmark" : "Measured Evaluation Results";
  const description = demoMode
    ? "Large-scale retrieval quality comparison across Dense, BM25, and Hybrid/RRF."
    : "Stored retrieval quality and latency metrics from completed benchmark runs.";

  return (
    <div className="stack benchmark-page">
      <PageHeading
        eyebrow="RETRIEVAL PERFORMANCE"
        title={title}
        description={description}
        action={<span className={`benchmark-view-badge ${demoMode ? "is-demo" : ""}`}>{demoMode ? "Demo View" : "Measured View"}</span>}
      />

      {demoMode ? (
        <DemoDashboard benchmark={benchmark} evaluation={evaluation} demo={demo} loading={benchmarkLoading || evaluationLoading || demoLoading} errors={[benchmarkError, evaluationError, demoError]} />
      ) : (
        <MeasuredResults benchmark={benchmark} evaluation={evaluation} loading={benchmarkLoading || evaluationLoading} errors={[benchmarkError, evaluationError]} />
      )}
    </div>
  );
}

function DemoDashboard({
  benchmark,
  evaluation,
  demo,
  loading,
  errors,
}: {
  benchmark: BenchmarkReport | null;
  evaluation: EvaluationReport | null;
  demo: DemoBenchmarkReport | null;
  loading: boolean;
  errors: (string | null)[];
}) {
  return (
    <div className="stack">
      <div className="benchmark-meta-row">
        <MetaItem label="Evaluation size" value={`${count(benchmark?.queryCount)} queries`} />
        <MetaItem label="Top-K" value={count(benchmark?.retrievalConfig.top_k)} />
        <MetaItem label="Corpus" value={`${environmentValue(benchmark?.environment.indexed_passages)} passages`} />
        <MetaItem label="Embedding" value={environmentValue(benchmark?.environment.embedding_model)} />
      </div>

      {errors.map((error, index) => error ? <div className="notice notice-error" key={`${index}-${error}`}><AlertTriangle size={15} /><span>{error}</span></div> : null)}

      <section className="grid grid-4 benchmark-summary-grid" aria-label="Benchmark summary">
        <AnimatedStat index={0} label="Queries" value={count(benchmark?.queryCount)} detail="100-query latency run" icon={<Search size={15} />} tone="blue" />
        <AnimatedStat index={1} label="Retrieval Modes" value="3" detail="Dense · BM25 · Hybrid/RRF" icon={<Layers3 size={15} />} tone="violet" />
        <AnimatedStat index={2} label="Top-K" value={count(benchmark?.retrievalConfig.top_k)} detail="Passages returned per query" icon={<Gauge size={15} />} tone="gold" />
        <AnimatedStat index={3} label="Indexed Passages" value={environmentValue(benchmark?.environment.indexed_passages)} detail="Validated local corpus" icon={<Database size={15} />} tone="mint" />
      </section>

      <section className="stack benchmark-hit5-section">
        <div className="benchmark-section-heading"><div><h2>Top-5 Retrieval</h2><p>Measured Hit@5 from the 10-query matched-corpus pilot.</p></div><span className="benchmark-presentation-label">Presentation benchmark view</span></div>
        <div className="grid grid-3">
          {modes.map((mode, index) => <AnimatedStat key={mode.key} index={index} label={mode.label} value={fraction(evaluation?.retrievalMetrics[mode.key]?.hit_at_5, 0)} detail="Hit@5" icon={<Search size={15} />} tone={mode.color} />)}
        </div>
      </section>

      <Surface className="benchmark-ragas-surface">
        <SectionTitle
          title="RAGAS Evaluation"
          detail="Measured matched-corpus pilot scores across the three retrieval modes."
          action={<span className="benchmark-presentation-label">Presentation benchmark view</span>}
        />
        <div className="grid grid-3">
          {modes.map((mode, index) => <RagasModeCard key={mode.key} mode={mode} result={evaluation?.aggregate[mode.key]} index={index} />)}
        </div>
      </Surface>

      <div className="grid grid-2 benchmark-chart-row">
        <Surface>
          <SectionTitle title="Retrieval Quality Comparison" detail="Measured 10-query matched-corpus pilot · Hit@1, Hit@5, Hit@10, and MRR." />
          {evaluation ? <PilotRetrievalChart metrics={evaluation.retrievalMetrics} /> : <div className="loading-state">{loading ? "Loading stored retrieval metrics." : "No measured retrieval metrics are available."}</div>}
        </Surface>
        <Surface>
          <SectionTitle title="Evaluation Coverage" detail="Queries represented in the benchmark dashboard." action={<span className="benchmark-coverage-count">{count(demo?.unique_query_ids)} / {count(demo?.query_count)}</span>} />
          <div className="benchmark-coverage-track" role="progressbar" aria-label="Evaluation dashboard coverage" aria-valuemin={0} aria-valuemax={demo?.query_count ?? 100} aria-valuenow={demo?.unique_query_ids ?? 0}>
            <motion.div className="benchmark-coverage-fill" initial={{ width: 0 }} animate={{ width: `${demo && demo.query_count > 0 ? Math.min(100, demo.unique_query_ids / demo.query_count * 100) : 0}%` }} transition={{ duration: 0.7, ease: "easeOut" }} />
          </div>
          <div className="benchmark-coverage-caption"><strong>{count(demo?.unique_query_ids)} of {count(demo?.query_count)}</strong><span>queries represented</span></div>
          <p className="benchmark-subtle-note">Execution counts are reported separately in the measured benchmark results.</p>
        </Surface>
      </div>

      <Surface>
        <SectionTitle title="100-Query Latency Distribution" detail="Illustrative distribution layout based on the presentation fixture." action={<span className="benchmark-presentation-label">Presentation visualization</span>} />
        {demo ? <SimulatedLatencyDistribution report={demo} /> : <div className="loading-state">{loading ? "Loading presentation visualization." : "Presentation visualization is unavailable."}</div>}
      </Surface>

      <section className="stack">
        <div className="benchmark-section-heading"><div><h2>Mode Comparison</h2><p>Actual matched-corpus pilot metrics for each retrieval path.</p></div><span className="benchmark-presentation-label">Presentation benchmark view</span></div>
        <div className="grid grid-3">
          {modes.map((mode) => <PilotModeCard key={mode.key} mode={mode} retrieval={evaluation?.retrievalMetrics[mode.key]} ragas={evaluation?.aggregate[mode.key]} />)}
        </div>
      </section>

      <LatencyDashboard benchmark={benchmark} loading={loading} />
    </div>
  );
}

function LatencyDashboard({ benchmark, loading }: { benchmark: BenchmarkReport | null; loading: boolean }) {
  return (
    <section className="stack">
      <div className="benchmark-section-heading"><div><h2>Measured Latency</h2><p>Completed 100-query run · latency records are separate from the matched-corpus quality pilot.</p></div><span className="benchmark-measured-label"><Activity size={12} /> MEASURED</span></div>
      {benchmark ? <div className="grid grid-3">{modes.map((mode, index) => <LatencyCard key={mode.key} mode={mode} latency={benchmark.latency[mode.key]} index={index} />)}</div> : <Surface><div className="loading-state">{loading ? "Loading the saved latency report." : "No completed latency report is available."}</div></Surface>}
      {benchmark ? <p className="benchmark-subtle-note">Source: {benchmark.querySource ?? "stored benchmark report"} · {count(benchmark.queryCount)} queries, {count(benchmark.retrievalConfig.top_k)} results per query.</p> : null}
    </section>
  );
}

function MeasuredResults({ benchmark, evaluation, loading, errors }: {
  benchmark: BenchmarkReport | null;
  evaluation: EvaluationReport | null;
  loading: boolean;
  errors: (string | null)[];
}) {
  return (
    <div className="stack">
      {errors.map((error, index) => error ? <div className="notice notice-error" key={`${index}-${error}`}><AlertTriangle size={15} /><span>{error}</span></div> : null)}
      <Surface>
        <SectionTitle
          title="Measured Retrieval and RAGAS Results"
          detail={evaluation?.datasetClassification ?? "Stored matched-corpus pilot evaluation"}
          action={<span className="benchmark-query-count">{count(evaluation?.queryCount)} evaluation queries · {count(evaluation?.totalRows)} mode rows</span>}
        />
        {evaluation ? <RetrievalTable evaluation={evaluation} /> : <div className="loading-state">{loading ? "Loading stored evaluation metrics." : "No completed evaluation report is available."}</div>}
        {!evaluation && !loading ? <div className="pending-box"><div><strong>Measured evaluation unavailable</strong><span>No presentation values are shown in this view.</span></div></div> : null}
      </Surface>

      {evaluation ? (
        <Surface>
          <SectionTitle title="Retrieval Quality" detail="Measured 10-query matched-corpus pilot metrics." />
          <PilotRetrievalChart metrics={evaluation.retrievalMetrics} />
        </Surface>
      ) : null}

      <LatencyDashboard benchmark={benchmark} loading={loading} />

      {evaluation ? (
        <div className="benchmark-provenance-grid">
          <Surface className="benchmark-provenance-card"><span className="eyebrow"><span className="eyebrow-dot" /> RAGAS</span><h3>Context Precision · Context Recall</h3><p>Stored judge results by retrieval mode, from {count(evaluation.queryCount)} matched-corpus queries.</p></Surface>
          <Surface className="benchmark-provenance-card"><span className="eyebrow"><span className="eyebrow-dot" /> DATASET</span><h3>{evaluation.officialMsMarco ? "MS MARCO evaluation" : "Matched-corpus pilot"}</h3><p>{evaluation.datasetClassification ?? "Evaluation provenance is recorded in the stored report."}</p></Surface>
          <Surface className="benchmark-provenance-card"><span className="eyebrow"><span className="eyebrow-dot" /> RETRIEVAL</span><h3>Dense · BM25 · Hybrid / RRF</h3><p>Hit@K and MRR are read from the completed matched-corpus retrieval results.</p></Surface>
        </div>
      ) : null}
    </div>
  );
}
