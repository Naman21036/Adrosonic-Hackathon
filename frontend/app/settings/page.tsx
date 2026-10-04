"use client";

import { useResource } from "@/lib/use-resource";
import { PageHeading, SectionTitle, StatusPill, Surface } from "@/components/ui";
import { useSystemStatus } from "@/components/system-provider";
import type { BenchmarkReport, EvaluationReport } from "@/lib/types";
import { Settings, ShieldCheck } from "lucide-react";

export default function SettingsPage() {
  const { health, status } = useSystemStatus();
  const { data: benchmark } = useResource<BenchmarkReport>("/api/reports/benchmark?mode=all");
  const { data: evaluation } = useResource<EvaluationReport>("/api/reports/evaluation?limit=1");
  const config = evaluation?.retrievalConfig ?? benchmark?.retrievalConfig ?? {};
  const environment = benchmark?.environment ?? {};
  const judge = evaluation?.judgeProvider && evaluation.judgeModel ? `${evaluation.judgeProvider} / ${evaluation.judgeModel}` : "Not reported";
  return (
    <div className="stack">
      <PageHeading eyebrow="RUNTIME & SECURITY" title="Configuration" description="Only non-secret settings are shown. Values without an API or report source are marked as unavailable instead of guessed." action={<StatusPill tone="good"><ShieldCheck size={12} />SECRETS HIDDEN</StatusPill>} />
      <div className="grid grid-2">
        <Surface>
          <SectionTitle title="Retrieval configuration" detail="Values from the latest stored retrieval/benchmark configuration." />
          <ConfigRow label="Embedding model" value={status?.model ?? (typeof environment.embedding_model === "string" ? environment.embedding_model : "Not exposed by the live API")} />
          <ConfigRow label="Embedding dimension" value={status?.dimension ?? environment.embedding_dimension ?? "Not reported"} />
          <ConfigRow label="Device" value={typeof environment.gpu === "string" && environment.gpu ? `Benchmark GPU: ${environment.gpu}` : "Runtime device not exposed"} />
          <ConfigRow label="Precision" value="Not exposed by the API/report" />
          <ConfigRow label="Top K" value={config.top_k ?? "Not reported"} />
          <ConfigRow label="Dense candidates" value={config.dense_candidate_k ?? "Not reported"} />
          <ConfigRow label="Sparse candidates" value={config.sparse_candidate_k ?? "Not reported"} />
          <ConfigRow label="RRF K" value={config.rrf_k ?? "Not reported"} />
        </Surface>
        <Surface>
          <SectionTitle title="Storage & providers" detail="Provider readiness is reported as configuration/readiness, not external connectivity." />
          <ConfigRow label="Qdrant backend" value="Local Qdrant" />
          <ConfigRow label="Qdrant collection" value={status?.collection ?? "Not available"} />
          <ConfigRow label="Qdrant path" value="storage/qdrant" />
          <ConfigRow label="BM25 store" value="SQLite FTS5 · storage/sparse.sqlite3" />
          <ConfigRow label="Azure AI Foundry" value={health?.azure_ready === true ? "Configured (connectivity unverified)" : health?.azure_ready === false ? "Not configured" : "Not exposed while API is offline"} />
          <ConfigRow label="RAGAS judge" value={judge} />
          <ConfigRow label="RAGAS version" value={evaluation?.ragasVersion ?? "Not reported"} />
          <ConfigRow label="Langfuse" value="Status is not exposed by the FastAPI health endpoint" />
          <div className="notice notice-info mt-16"><Settings size={14} /><span>The backend currently has no safe configuration endpoint for runtime device, precision, or package versions. API keys and tokens are never sent to this page.</span></div>
        </Surface>
      </div>
      <Surface className="security-card"><div className="security-icon"><ShieldCheck size={18} /></div><div><h2>Secret handling</h2><p>Azure, Hugging Face, Cloudflare, Mistral, and Langfuse credentials stay in the Python service environment. The browser talks to a same-origin Next.js proxy and never receives credential values.</p></div><StatusPill tone="good">SERVER-SIDE ONLY</StatusPill></Surface>
    </div>
  );
}

function ConfigRow({ label, value }: { label: string; value: string | number }) {
  return <div className="config-row"><span>{label}</span><strong>{value}</strong></div>;
}
