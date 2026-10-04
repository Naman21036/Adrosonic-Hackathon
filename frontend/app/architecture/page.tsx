"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowDown, Bot, Braces, ChevronDown, CircleHelp, Database, Filter, Layers3, MessageSquareText, Search, Sparkles, X } from "lucide-react";
import { PageHeading, SectionTitle, StatusPill, Surface } from "@/components/ui";
import { useSystemStatus } from "@/components/system-provider";

interface ArchitectureNode {
  id: string;
  title: string;
  subtitle: string;
  detail: string;
  icon: typeof Search;
  tone?: "gold";
}

const nodeDetails: ArchitectureNode[] = [
  { id: "query", title: "User query", subtitle: "POST /search", detail: "FastAPI validates the query, retrieval mode, top_k and optional source_id filter. Secrets remain server-side.", icon: MessageSquareText },
  { id: "processor", title: "Query processor", subtitle: "Request + filter validation", detail: "The API adds request IDs, logs the filter configuration and opens a Langfuse root trace when tracing is available.", icon: Braces },
  { id: "embedding", title: "Qwen3 Embedding", subtitle: "0.6B · 1,024 dimensions", detail: "The configured Qwen model creates one query vector. The indexed corpus vectors are reused; this request does not rebuild them.", icon: Sparkles },
  { id: "filter", title: "Metadata predicate", subtitle: "source_id", detail: "For dense search the predicate is passed in Qdrant query_points(query_filter=...). For BM25, SQL restricts candidates before BM25 sorting. Qdrant Local currently accepts the filter but does not support physical payload indexes.", icon: Filter, tone: "gold" },
  { id: "qdrant", title: "Qdrant dense", subtitle: "Local · Cosine", detail: "Dense candidate retrieval runs against the Local Qdrant `passages` collection with the optional source_id predicate inside the vector query.", icon: Database },
  { id: "bm25", title: "BM25 sparse", subtitle: "SQLite FTS5", detail: "The sparse search uses the same source_id condition in SQL before ranking. A SQLite expression index is created from the canonical passage ID prefix.", icon: Search, tone: "gold" },
  { id: "rrf", title: "Reciprocal Rank Fusion", subtitle: "k = 60", detail: "Hybrid mode combines ranked dense and sparse candidates using the configured Reciprocal Rank Fusion constant.", icon: Layers3 },
  { id: "context", title: "Top-K context", subtitle: "5 passages", detail: "The API returns passage IDs, passage text, ranking score, source_id, and retrieval score components where present.", icon: CircleHelp },
  { id: "azure", title: "Azure AI Foundry", subtitle: "Grounded generation · optional", detail: "Generation is requested only when the API request sets generate_answer=true. The browser never calls Azure directly.", icon: Bot, tone: "gold" },
];

export default function ArchitecturePage() {
  const [selected, setSelected] = useState("filter");
  const { health } = useSystemStatus();
  const detail = nodeDetails.find((node) => node.id === selected) ?? nodeDetails[0];
  return (
    <div className="stack">
      <PageHeading eyebrow="SYSTEM DESIGN" title="Retrieval Architecture" description="Explore the implemented request path from API validation through dense and sparse retrieval, fusion, and optional grounded generation." action={<StatusPill tone="blue">LOCAL QDRANT</StatusPill>} />
      <div className="architecture-layout">
        <Surface className="architecture-canvas">
          <SectionTitle title="Request path" detail="Select a node to inspect its current role and runtime constraints." />
          <div className="architecture-flow interactive-flow">
            <ArchitectureButton node={nodeDetails[0]} selected={selected === "query"} onSelect={setSelected} />
            <ArrowDown className="arch-arrow" size={16} />
            <ArchitectureButton node={nodeDetails[1]} selected={selected === "processor"} onSelect={setSelected} />
            <ArrowDown className="arch-arrow" size={16} />
            <ArchitectureButton node={nodeDetails[2]} selected={selected === "embedding"} onSelect={setSelected} />
            <div className="filter-branch"><button className={`architecture-box gold-box ${selected === "filter" ? "selected-arch" : ""}`} onClick={() => setSelected("filter")}><Filter size={16} /><strong>source_id predicate</strong><small>Applied before retrieval/ranking</small></button><span className="filter-branch-line" /><span className="filter-branch-copy">filter enters Qdrant query + BM25 SQL</span></div>
            <div className="arch-split arch-split-large"><ArchitectureButton node={nodeDetails[4]} selected={selected === "qdrant"} onSelect={setSelected} /><ArchitectureButton node={nodeDetails[5]} selected={selected === "bm25"} onSelect={setSelected} /></div>
            <div className="arch-merge"><span /><ArrowDown className="arch-arrow" size={16} /><span /></div>
            <ArchitectureButton node={nodeDetails[6]} selected={selected === "rrf"} onSelect={setSelected} />
            <ArrowDown className="arch-arrow" size={16} />
            <ArchitectureButton node={nodeDetails[7]} selected={selected === "context"} onSelect={setSelected} />
            {health?.azure_ready ? <><ArrowDown className="arch-arrow" size={16} /><ArchitectureButton node={nodeDetails[8]} selected={selected === "azure"} onSelect={setSelected} /></> : <div className="optional-node">Azure generation is optional · backend configuration not currently ready</div>}
          </div>
        </Surface>
        <div className="stack">
          <AnimatePresence mode="wait"><motion.div key={detail.id} initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -5 }} transition={{ duration: .16 }}><Surface className="architecture-details"><div className="detail-panel-icon"><detail.icon size={18} /></div><div className="eyebrow">COMPONENT DETAIL</div><h2>{detail.title}</h2><span className="detail-subtitle">{detail.subtitle}</span><p>{detail.detail}</p><StatusPill tone={detail.id === "filter" ? "warn" : health?.status === "ready" ? "good" : "neutral"}>{detail.id === "filter" ? "PRE-RETRIEVAL" : health?.status === "ready" ? "API READY" : "RUNTIME STATUS UNKNOWN"}</StatusPill></Surface></motion.div></AnimatePresence>
          <Surface><SectionTitle title="Filter execution points" detail="The filter stays attached to each retrieval branch." /><div className="filter-execution-row"><span className="execution-icon"><Database size={15} /></span><div><strong>Qdrant dense query</strong><small>query_filter = source_id before vector ranking</small></div><StatusPill tone="blue">NATIVE QUERY</StatusPill></div><div className="filter-execution-row"><span className="execution-icon gold-execution"><Search size={15} /></span><div><strong>BM25 candidate SQL</strong><small>source_id predicate before BM25 ORDER BY / LIMIT</small></div><StatusPill tone="blue">SQL PREDICATE</StatusPill></div><div className="notice notice-warn mt-16"><X size={14} /><span>Qdrant Local 1.19.1 does not create physical payload indexes. The filter is still supplied to the vector query; this backend cannot provide indexed filter performance.</span></div></Surface>
        </div>
      </div>
      <Surface className="architecture-security"><div><span className="execution-icon"><Sparkles size={15} /></span><strong>Browser → Next.js → FastAPI → Qdrant / BM25 / Azure</strong><small>Secrets remain in the Python backend. The frontend proxy forwards only the existing API routes.</small></div><StatusPill tone="good">NO BROWSER SECRETS</StatusPill></Surface>
    </div>
  );
}

function ArchitectureButton({ node, selected, onSelect }: { node: ArchitectureNode; selected: boolean; onSelect: (id: string) => void }) {
  const Icon = node.icon;
  return <button className={`architecture-box ${node.tone === "gold" ? "gold-box" : ""} ${selected ? "selected-arch" : ""}`} onClick={() => onSelect(node.id)} aria-pressed={selected}><Icon size={16} /><strong>{node.title}</strong><small>{node.subtitle}</small><span className="architecture-hover"><ChevronDown size={12} />details</span></button>;
}
