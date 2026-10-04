"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BenchmarkSample, DemoBenchmarkReport, RetrievalModeMetrics } from "@/lib/types";

const tooltipStyle = {
  background: "#101923",
  border: "1px solid rgba(191,211,232,.14)",
  borderRadius: 7,
  color: "#eaf1f8",
  fontSize: 10,
};

const retrievalModes = [
  { key: "dense", label: "Dense", color: "#72bdf0" },
  { key: "sparse", label: "BM25", color: "#d9b56b" },
  { key: "hybrid", label: "Hybrid", color: "#61d6b3" },
] as const;

function retrievalChartData(metrics: Record<string, RetrievalModeMetrics>) {
  return retrievalModes.flatMap(({ key, label }) => {
    const row = metrics[key];
    if (row?.hit_at_1 == null || row.hit_at_5 == null || row.hit_at_10 == null || row.mrr == null) return [];
    return [{ mode: label, hit1: row.hit_at_1, hit5: row.hit_at_5, hit10: row.hit_at_10, mrr: row.mrr }];
  });
}

export function PilotRetrievalChart({ metrics }: { metrics: Record<string, RetrievalModeMetrics> }) {
  const data = retrievalChartData(metrics);
  if (data.length === 0) return <div className="pending-box"><div><strong>Retrieval metrics unavailable</strong><span>No completed matched-corpus metrics were found in the stored evaluation report.</span></div></div>;
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 15, right: 6, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="mode" tick={{ fill: "#8292a1", fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} tick={{ fill: "#637383", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => typeof value === "number" ? `${(value * 100).toFixed(1)}%` : String(value)} />
          <Bar dataKey="hit1" name="Hit@1" fill="#d9b56b" radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="hit5" name="Hit@5" fill="#5ca8df" radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="hit10" name="Hit@10" fill="#62c9aa" radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Line type="monotone" dataKey="mrr" name="MRR" stroke="#bd9af5" strokeWidth={2} dot={{ r: 3, fill: "#bd9af5", strokeWidth: 0 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BenchmarkLatencyChart({ samples, mode }: { samples: BenchmarkSample[]; mode: string }) {
  const data = samples.filter((row) => row.mode === mode).sort((a, b) => a.query_number - b.query_number);
  if (data.length === 0) return <div className="pending-box"><div><strong>No benchmark series available</strong><span>The stored benchmark report has no rows for this retrieval mode.</span></div></div>;
  return (
    <div className="chart-box tall">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 14, right: 17, bottom: 5, left: -10 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="query_number" tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} tickMargin={8} />
          <YAxis tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(value: number) => `${Math.round(value)} ms`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => typeof value === "number" ? `${value.toFixed(1)} ms` : String(value)} labelFormatter={(label) => `Query ${label}`} />
          <ReferenceLine y={300} stroke="#dfb45c" strokeDasharray="4 4" label={{ value: "300 ms target", fill: "#d7b96f", fontSize: 9, position: "insideTopRight" }} />
          <Line type="monotone" dataKey="latency_ms" name="Latency" stroke="#70bff5" strokeWidth={1.7} dot={false} activeDot={{ r: 4, fill: "#b9e1ff", stroke: "#0c151e", strokeWidth: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const demoModes = [
  { key: "dense", label: "Dense", color: "#72bdf0" },
  { key: "bm25", label: "BM25", color: "#d9b56b" },
  { key: "hybrid", label: "Hybrid / RRF", color: "#61d6b3" },
] as const;

export function SimulatedLatencyDistribution({ report }: { report: DemoBenchmarkReport }) {
  const data = report.latency_distribution.dense.map((bucket, index) => ({
    bucket: bucket.label,
    dense: bucket.count,
    bm25: report.latency_distribution.bm25[index]?.count ?? 0,
    hybrid: report.latency_distribution.hybrid[index]?.count ?? 0,
  }));

  return (
    <div className="chart-box tall" aria-label="Presentation latency distribution by retrieval mode">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 12, right: 12, bottom: 5, left: -15 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="bucket" tick={{ fill: "#8292a1", fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis allowDecimals={false} tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${value} ${value === 1 ? "record" : "records"}`, "Count"]} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ color: "#9aabba", fontSize: 9 }} />
          {demoModes.map(({ key, label, color }) => <Bar key={key} dataKey={key} name={label} fill={color} radius={[3, 3, 0, 0]} maxBarSize={18} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SimulatedLatencyByQueryChart({ report }: { report: DemoBenchmarkReport }) {
  const data = report.datasets.dense.map((record, index) => ({
    query_number: index + 1,
    dense: record.latency_ms,
    bm25: report.datasets.bm25[index]?.latency_ms ?? 0,
    hybrid: report.datasets.hybrid[index]?.latency_ms ?? 0,
  }));

  return (
    <div className="chart-box tall" aria-label="Simulated latency by query for dense, BM25, and hybrid retrieval">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 14, bottom: 5, left: -10 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="query_number" tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} tickMargin={8} />
          <YAxis tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(value: number) => `${Math.round(value)} ms`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => typeof value === "number" ? [`${value.toFixed(1)} ms`, "Simulated latency"] : [String(value), "Simulated latency"]} labelFormatter={(label) => `Demo query ${label}`} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ color: "#9aabba", fontSize: 9 }} />
          {demoModes.map(({ key, label, color }) => <Line key={key} type="monotone" dataKey={key} name={label} stroke={color} strokeWidth={1.7} dot={false} activeDot={{ r: 3, fill: color, stroke: "#0c151e", strokeWidth: 2 }} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SimulatedModeComparisonChart({ report }: { report: DemoBenchmarkReport }) {
  const data = demoModes.map(({ key, label }) => ({
    mode: label,
    mean: report.summary[key].mean_ms ?? 0,
    p50: report.summary[key].p50_ms ?? 0,
    p95: report.summary[key].p95_ms ?? 0,
  }));

  return (
    <div className="chart-box" aria-label="Simulated latency comparison by retrieval mode">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 12, right: 12, bottom: 5, left: -10 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="mode" tick={{ fill: "#8292a1", fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#718292", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(value: number) => `${Math.round(value)} ms`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => typeof value === "number" ? `${value.toFixed(1)} ms` : String(value)} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ color: "#9aabba", fontSize: 9 }} />
          <Bar dataKey="mean" name="Average" fill="#5ca8df" radius={[3, 3, 0, 0]} maxBarSize={34} />
          <Line type="monotone" dataKey="p50" name="P50" stroke="#d9b56b" strokeWidth={2} dot={{ r: 3, fill: "#d9b56b", strokeWidth: 0 }} />
          <Line type="monotone" dataKey="p95" name="P95" stroke="#61d6b3" strokeWidth={2} dot={{ r: 3, fill: "#61d6b3", strokeWidth: 0 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RetrievalMetricsChart({ metrics }: { metrics: Record<string, RetrievalModeMetrics> }) {
  const data = retrievalChartData(metrics);
  if (data.length === 0) return <div className="pending-box"><div><strong>Retrieval metrics unavailable</strong><span>No completed matched-corpus metrics were found in the stored evaluation report.</span></div></div>;
  return (
    <div className="chart-box small">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="rgba(191,211,232,.07)" vertical={false} />
          <XAxis dataKey="mode" tick={{ fill: "#8292a1", fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} tick={{ fill: "#637383", fontSize: 8 }} axisLine={false} tickLine={false} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => typeof value === "number" ? `${(value * 100).toFixed(1)}%` : String(value)} />
          <Bar dataKey="hit5" name="Hit@5" radius={[4, 4, 0, 0]} maxBarSize={34}>
            {data.map((entry) => <Cell key={entry.mode} fill={retrievalModes.find((mode) => mode.label === entry.mode)?.color ?? "#72bdf0"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
