"use client";

import { Activity, ArrowUpRight, CircleAlert, ExternalLink, FileText, Radio, ShieldCheck } from "lucide-react";
import { PageHeading, SectionTitle, StatusPill, Surface, formatMs } from "@/components/ui";
import { useResource } from "@/lib/use-resource";
import type { ObservabilityReport } from "@/lib/types";

export default function ObservabilityPage() {
  const { data, loading, error, refresh } = useResource<ObservabilityReport>("/api/reports/observability");
  const langfuseConfigured = data?.langfuseEnabled === true && data.langfuseCredentialsPresent === true;
  const langfuseUrl = data?.langfuseHost ?? process.env.NEXT_PUBLIC_LANGFUSE_URL ?? null;

  return (
    <div className="stack">
      <PageHeading eyebrow="TELEMETRY & TRACEABILITY" title="Observability" description="Recent application events come from the sanitized Loguru JSONL sink. Langfuse trace listing is not exposed by the current FastAPI API." action={<button className="button button-quiet" onClick={() => void refresh()}><Activity size={14} />Refresh logs</button>} />
      <div className="grid grid-3">
        <ObservabilityCard label="Application events" value={data ? String(data.logCount) : loading ? "Loading" : "—"} detail="Recent retained records" icon={<FileText size={15} />} />
        <ObservabilityCard label="Langfuse configuration" value={data?.langfuseEnabled === null || data?.langfuseEnabled === undefined ? "Unknown" : data.langfuseEnabled ? langfuseConfigured ? "Configured" : "Missing credentials" : "Disabled"} detail="Configuration only; no connectivity probe" icon={<Radio size={15} />} tone="gold" />
        <ObservabilityCard label="Trace browser" value="Unavailable" detail="No trace-listing route in FastAPI" icon={<ShieldCheck size={15} />} tone="violet" />
      </div>

      <div className="grid grid-2">
        <Surface>
          <SectionTitle title="Application logs" detail="Secret-shaped fields and raw exception bodies are omitted from this table." action={<StatusPill tone={data?.logs.length ? "good" : loading ? "neutral" : "warn"}>{data?.logs.length ? "LIVE FILE DATA" : loading ? "LOADING" : "NO LOGS"}</StatusPill>} />
          {error ? <div className="notice notice-error"><CircleAlert size={14} />{error}</div> : null}
          {data?.logs.length ? <div className="logs-panel"><div className="log-row log-row-header"><span>Timestamp</span><span>Level</span><span>Event</span><span>Latency</span><span>Status</span></div>{data.logs.slice(0, 80).map((entry, index) => <div className="log-row" key={`${entry.timestamp}:${entry.event}:${index}`}><span className="log-time">{entry.timestamp.replace("T", " ").slice(0, 19) || "—"}</span><span className={`log-level level-${entry.level.toLowerCase()}`}>{entry.level}</span><span className="log-event" title={entry.event}>{entry.event}</span><span>{formatMs(entry.latencyMs)}</span><span className={entry.status === "ERROR" ? "error-text" : "success-text"}>{entry.status}</span></div>)}</div> : !loading ? <div className="pending-box"><div><strong>No sanitized log events available</strong><span>Start the FastAPI service and make a request to populate logs/app.jsonl.</span></div></div> : <div className="skeleton-table"><span /><span /><span /><span /></div>}
        </Surface>

        <Surface>
          <SectionTitle title="Langfuse" detail="Tracing is failure-isolated in the backend; the UI only displays trace data when an API provides it." action={<StatusPill tone={langfuseConfigured ? "warn" : "neutral"}>{langfuseConfigured ? "CONFIGURED · UNVERIFIED" : "STATUS UNKNOWN"}</StatusPill>} />
          <div className="trace-empty"><span className="trace-orbit"><Radio size={18} /></span><h3>Trace data is not available here</h3><p>{data?.langfuseMissingEndpoint ?? "The backend does not expose a trace-listing endpoint."}</p>{langfuseUrl ? <a className="button button-quiet" href={langfuseUrl} target="_blank" rel="noreferrer">Open Langfuse<ArrowUpRight size={14} /></a> : null}</div>
          <div className="divider" />
          <div className="config-row"><span>Trace exporter</span><strong>{data?.langfuseEnabled ? "Enabled in local config" : data?.langfuseEnabled === false ? "Disabled in local config" : "Not reported"}</strong></div>
          <div className="config-row"><span>Trace connectivity</span><strong>Not tested</strong></div>
          <div className="config-row"><span>Frontend trace API</span><strong>Missing</strong></div>
          <div className="notice notice-info mt-16"><ExternalLink size={14} /><span>Traces are never fabricated. Add a read-only backend route for recent trace metadata to display them in this panel.</span></div>
        </Surface>
      </div>
    </div>
  );
}

function ObservabilityCard({ label, value, detail, icon, tone = "blue" }: { label: string; value: string; detail: string; icon: React.ReactNode; tone?: "blue" | "gold" | "violet" }) {
  return <article className={`stat-card stat-${tone}`}><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><div className="stat-value small-stat-value">{value}</div><div className="stat-detail">{detail}</div></article>;
}
