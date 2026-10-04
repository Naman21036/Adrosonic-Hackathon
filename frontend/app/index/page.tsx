"use client";

import { Database, FileStack, Gauge, Layers3, Network, ShieldCheck } from "lucide-react";
import { PageHeading, SectionTitle, StatusPill, Surface, formatInteger } from "@/components/ui";
import { useSystemStatus } from "@/components/system-provider";
import { useResource } from "@/lib/use-resource";
import type { BenchmarkReport } from "@/lib/types";

const corpusTarget = 100_000;

export default function IndexPage() {
  const { health, status, apiOnline } = useSystemStatus();
  const { data: benchmark } = useResource<BenchmarkReport>("/api/reports/benchmark?mode=all");
  const currentCount = status?.indexed_passages ?? null;
  const sparseCount = status?.sparse_passages ?? null;
  const progress = currentCount === null ? 0 : Math.min(100, (currentCount / corpusTarget) * 100);
  const parity = currentCount !== null && sparseCount !== null && currentCount === sparseCount;
  const dimension = status?.dimension ?? benchmark?.environment.embedding_dimension ?? null;
  const model = status?.model ?? (typeof benchmark?.environment.embedding_model === "string" ? benchmark.environment.embedding_model : null);
  const healthy = health?.qdrant_ready === true && health?.sparse_ready === true && parity && status?.sync_status === "operational";
  return (
    <div className="stack">
      <PageHeading eyebrow="CORPUS & VECTOR STORE" title="Index & Corpus" description="Live counts come from FastAPI status. The progress track compares the current index with the 100,000-passage challenge goal; it is not a completion claim." action={<StatusPill tone={healthy ? "good" : apiOnline ? "warn" : "neutral"}>{healthy ? "HEALTHY" : apiOnline ? "CHECK STATUS" : "BACKEND OFFLINE"}</StatusPill>} />
      <div className="grid grid-4">
        <IndexCard label="Dense points" value={formatInteger(currentCount)} detail="Qdrant count from /status" icon={<Database size={16} />} />
        <IndexCard label="BM25 records" value={formatInteger(sparseCount)} detail="SQLite FTS index count" icon={<FileStack size={16} />} tone="gold" />
        <IndexCard label="Vector dimension" value={dimension ?? "—"} detail="Qdrant collection dimension" icon={<Layers3 size={16} />} tone="violet" />
        <IndexCard label="Source query IDs" value="≈ 10,938" detail="Approximate corpus audit count" icon={<Network size={16} />} tone="mint" />
      </div>
      <Surface>
        <SectionTitle title="Index coverage" detail="The current corpus remains at its live backend count; no data is inferred for unindexed passages." action={<StatusPill tone={currentCount === corpusTarget ? "good" : "warn"}>{currentCount === corpusTarget ? "TARGET REACHED" : "TARGET NOT REACHED"}</StatusPill>} />
        <div className="coverage-number-row"><div><strong>{formatInteger(currentCount)}</strong><span>currently indexed</span></div><div className="coverage-target"><strong>{formatInteger(corpusTarget)}</strong><span>challenge goal</span></div></div>
        <div className="progress-track progress-large"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
        <div className="coverage-foot"><span>{currentCount === null ? "Index count unavailable" : `${progress.toFixed(1)}% of challenge goal`}</span><span>{formatInteger(corpusTarget - (currentCount ?? 0))} remaining to goal</span></div>
      </Surface>
      <div className="grid grid-2">
        <Surface>
          <SectionTitle title="Store configuration" detail="Values exposed by the live API or established backend report." />
          <div className="config-row"><span>Corpus</span><strong>MS MARCO · cached validation subset</strong></div>
          <div className="config-row"><span>Dense store</span><strong>Local Qdrant · {status?.collection ?? "—"}</strong></div>
          <div className="config-row"><span>Qdrant path</span><strong>storage/qdrant</strong></div>
          <div className="config-row"><span>Distance metric</span><strong>Cosine</strong></div>
          <div className="config-row"><span>Embedding model</span><strong>{model ?? "—"}</strong></div>
          <div className="config-row"><span>Embedding dimension</span><strong>{dimension ?? "—"}</strong></div>
        </Surface>
        <Surface>
          <SectionTitle title="Integrity snapshot" detail="The API reports count parity and pending mutations; exact ID equality requires the corpus validation tool." />
          <IntegrityRow label="Dense / sparse count parity" good={parity} value={parity ? "Counts match" : "Unknown or mismatched"} icon={<Gauge size={15} />} />
          <IntegrityRow label="Index mutation journal" good={status?.pending_mutations === 0} value={status?.pending_mutations === 0 ? "No pending mutations" : status?.pending_mutations == null ? "Not reported" : `${status.pending_mutations} pending`} icon={<ShieldCheck size={15} />} />
          <IntegrityRow label="Qdrant API readiness" good={health?.qdrant_ready === true} value={health?.qdrant_ready ? "Ready" : "Not ready"} icon={<Database size={15} />} />
          <IntegrityRow label="BM25 readiness" good={health?.sparse_ready === true} value={health?.sparse_ready ? "Ready" : "Not ready"} icon={<FileStack size={15} />} />
          <div className="notice notice-info mt-16">The current FastAPI `/status` response does not expose unique source IDs or a point-by-point ID comparison.</div>
        </Surface>
      </div>
    </div>
  );
}

function IndexCard({ label, value, detail, icon, tone = "blue" }: { label: string; value: string | number; detail: string; icon: React.ReactNode; tone?: "blue" | "gold" | "mint" | "violet" }) {
  return <article className={`stat-card stat-${tone}`}><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></article>;
}

function IntegrityRow({ label, value, good, icon }: { label: string; value: string; good: boolean | undefined; icon: React.ReactNode }) {
  return <div className="integrity-row"><span className="integrity-icon">{icon}</span><div><strong>{label}</strong><small>{value}</small></div><StatusPill tone={good ? "good" : "warn"}>{good ? "OK" : "CHECK"}</StatusPill></div>;
}
