"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeftRight, ChevronDown, ChevronUp, CircleAlert, Clock3, Database, Filter, LockKeyhole, Play, Search, Sparkles, X } from "lucide-react";
import { PageHeading, SectionTitle, StatusPill, Surface, formatMs } from "@/components/ui";
import { apiRequest, safeErrorMessage } from "@/lib/api";
import { completeDemoStep } from "@/lib/demo-progress";
import { useResource } from "@/lib/use-resource";
import type { FilterDemoReport, QueryRequest, QueryResponse, RetrievedPassage } from "@/lib/types";

type SearchMode = "dense" | "hybrid";

export default function PlaygroundPage() {
  const { data: filterDemo } = useResource<FilterDemoReport>("/api/reports/filter-demo");
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [filterEnabled, setFilterEnabled] = useState(false);
  const [sourceIdOverride, setSourceIdOverride] = useState<string | null>(null);
  const sourceIdText = sourceIdOverride ?? String(filterDemo?.sourceId ?? 19699);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [comparison, setComparison] = useState<{ dense: QueryResponse; hybrid: QueryResponse } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sourceId = Number(sourceIdText);
  const sourceIdValid = Number.isInteger(sourceId) && sourceId >= 0;
  const activeFilter = filterEnabled && sourceIdValid ? { source_id: sourceId } : undefined;
  const resultRows = useMemo(() => result?.results.slice(0, topK) ?? [], [result, topK]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      const params = new URLSearchParams(window.location.search);
      const requestedMode = params.get("mode");
      if (requestedMode === "dense" || requestedMode === "hybrid") setMode(requestedMode);
      if (params.get("filter") === "source_id") setFilterEnabled(true);
    }, 0);
    return () => window.clearTimeout(initial);
  }, []);

  async function runSearch() {
    if (!query.trim()) { setError("Enter a query to run retrieval."); return; }
    if (filterEnabled && !sourceIdValid) { setError("Source ID must be a non-negative whole number."); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    setComparison(null);
    const requestFor = (retrievalMode: SearchMode): QueryRequest => ({
      query: query.trim(),
      mode: retrievalMode,
      top_k: topK,
      ...(activeFilter ? { filter: activeFilter } : {}),
    });
    try {
      const response = await apiRequest<QueryResponse>("/api/backend/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(requestFor(mode)),
      }, 120_000);
      setResult(response);
      if (response.mode === "dense" || response.mode === "hybrid") completeDemoStep(response.mode);
      if (activeFilter && response.filter?.source_id === activeFilter.source_id && response.results.every((passage) => passage.source_id === activeFilter.source_id)) completeDemoStep("filter");
    } catch (cause) {
      setError(safeErrorMessage(cause));
    } finally {
      setLoading(false);
    }
  }

  async function compareModes() {
    if (!query.trim()) { setError("Enter a query before comparing modes."); return; }
    if (filterEnabled && !sourceIdValid) { setError("Source ID must be a non-negative whole number."); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    setComparison(null);
    const requestFor = (retrievalMode: SearchMode): QueryRequest => ({
      query: query.trim(),
      mode: retrievalMode,
      top_k: topK,
      ...(activeFilter ? { filter: activeFilter } : {}),
    });
    try {
      const [dense, hybrid] = await Promise.all([
        apiRequest<QueryResponse>("/api/backend/search", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(requestFor("dense")) }, 120_000),
        apiRequest<QueryResponse>("/api/backend/search", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(requestFor("hybrid")) }, 120_000),
      ]);
      setComparison({ dense, hybrid });
      completeDemoStep("dense");
      completeDemoStep("hybrid");
    } catch (cause) {
      setError(safeErrorMessage(cause));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack">
      <PageHeading eyebrow="LIVE RETRIEVAL" title="Retrieval Playground" description="Send a real query to the FastAPI retrieval API and inspect the passages, scores, filters and backend-reported latency." action={<StatusPill tone="blue"><Database size={12} />LOCAL QDRANT</StatusPill>} />
      <div className="playground-layout">
        <Surface className="playground-controls">
          <SectionTitle title="Query configuration" detail="The API embeds once and applies retrieval filters before ranking." />
          <label className="label" htmlFor="query-input">Query</label>
          <textarea id="query-input" className="text-field" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask the retrieval engine anything." maxLength={4096} />
          <div className="control-block">
            <div className="label">Retrieval mode</div>
            <div className="segmented" role="group" aria-label="Retrieval mode">
              <button className={mode === "dense" ? "active" : ""} aria-pressed={mode === "dense"} onClick={() => setMode("dense")}>Dense</button>
              <button disabled title="The current FastAPI /search endpoint does not expose standalone BM25 mode.">BM25 unavailable</button>
              <button className={mode === "hybrid" ? "active" : ""} aria-pressed={mode === "hybrid"} onClick={() => setMode("hybrid")}>Hybrid / RRF</button>
            </div>
            <p className="field-help">Hybrid uses dense Qdrant + BM25 candidates fused with RRF.</p>
          </div>
          <div className="control-inline-row">
            <div className="control-block grow"><label className="label" htmlFor="top-k">Top K</label><select id="top-k" className="select-field" value={topK} onChange={(event) => setTopK(Number(event.target.value))}><option value={3}>3 passages</option><option value={5}>5 passages</option><option value={10}>10 passages</option><option value={20}>20 passages</option></select></div>
          </div>
          <div className="filter-control">
            <div className="switch-row"><div><div className="label no-margin">Metadata filter</div><p className="field-help no-margin">Restrict results by indexed source ID.</p></div><button type="button" role="switch" aria-checked={filterEnabled} aria-label="Enable source ID metadata filter" className="switch" onClick={() => setFilterEnabled(!filterEnabled)} /></div>
            <AnimatePresence initial={false}>{filterEnabled ? <motion.div className="filter-input-wrap" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: .18 }}><label className="label" htmlFor="source-id">Filter dimension <span className="muted">· Source ID</span></label><input id="source-id" className="field" type="number" min={0} step={1} value={sourceIdText} onChange={(event) => setSourceIdOverride(event.target.value)} aria-invalid={!sourceIdValid} /><div className="field-help">{filterDemo?.sourcePassages ? `Source ${sourceIdText} has ${filterDemo.sourcePassages} indexed passages in the migration audit.` : "Use a source ID from the indexed corpus."}</div></motion.div> : null}</AnimatePresence>
          </div>
          <div className="control-actions"><button className="button button-primary" disabled={loading || !query.trim() || (filterEnabled && !sourceIdValid)} onClick={() => void runSearch()}>{loading ? <span className="mini-spinner" /> : <Play size={14} fill="currentColor" />}{filterEnabled ? "Run Filtered Search" : "Run Retrieval"}</button><button className="button button-quiet" disabled={loading || !query.trim() || (filterEnabled && !sourceIdValid)} onClick={() => void compareModes()}><ArrowLeftRight size={14} />Compare</button></div>
          {filterEnabled ? <div className="filter-method-note"><span className="filter-lock"><LockKeyhole size={12} /></span><span>Source ID is passed to Qdrant query filtering and the BM25 SQL candidate predicate before ranking. Local Qdrant does not provide a physical payload index.</span></div> : null}
          {error ? <div className="notice notice-error" role="alert"><CircleAlert size={15} /><span>{error}</span><button className="icon-button notice-close" onClick={() => setError(null)} aria-label="Dismiss error"><X size={14} /></button></div> : null}
        </Surface>

        <div className="playground-results">
          {filterEnabled ? <div className="filter-banner"><span className="filter-lock"><Filter size={12} /></span><strong>PRE-RETRIEVAL FILTER</strong><span>source_id = {sourceIdValid ? sourceId : "invalid"}</span></div> : null}
          {loading ? <Surface className="results-loading"><div className="retrieval-progress"><span /><span /><span /><span /></div><strong>Retrieval request in progress</strong><p>Waiting for the backend response. Stage timings are displayed only if the API returns them.</p></Surface> : null}
          {!loading && result ? <SearchResults response={result} results={resultRows} /> : null}
          {!loading && comparison ? <ComparisonResults comparison={comparison} topK={topK} /> : null}
          {!loading && !result && !comparison && !error ? <Surface className="results-placeholder"><div className="placeholder-orbit"><Search size={22} /></div><h2>Your retrieval results appear here</h2><p>Choose a mode, optionally apply a source ID filter, and run a query. No sample results are fabricated.</p><div className="placeholder-tags"><span>QDRANT</span><span>BM25</span><span>RRF</span></div></Surface> : null}
        </div>
      </div>
    </div>
  );
}

function SearchResults({ response, results }: { response: QueryResponse; results: RetrievedPassage[] }) {
  const sourceIds = [...new Set(results.map((item) => item.source_id).filter((value): value is number => typeof value === "number"))];
  const stages: { label: string; value: number | undefined }[] = response.mode === "dense"
    ? [{ label: "Embedding", value: response.stage_latency_ms?.embedding_latency_ms }, { label: "Qdrant", value: response.stage_latency_ms?.qdrant_latency_ms }]
    : [{ label: "Embedding", value: response.stage_latency_ms?.embedding_latency_ms }, { label: "Qdrant", value: response.stage_latency_ms?.qdrant_latency_ms }, { label: "BM25", value: response.stage_latency_ms?.sparse_latency_ms }, { label: "RRF", value: response.stage_latency_ms?.rrf_latency_ms }];
  const shownStages = [
    ...stages,
    ...(response.answer?.generation_latency_ms !== undefined ? [{ label: "Generation", value: response.answer.generation_latency_ms }] : []),
  ].filter((entry): entry is { label: string; value: number } => typeof entry.value === "number" && Number.isFinite(entry.value));
  const totalLatency = response.total_latency_ms ?? response.latency_ms;
  return (
    <div className="stack">
      {response.filter && Object.keys(response.filter).length ? <div className="filter-banner"><span className="filter-lock"><LockKeyhole size={12} /></span><strong>PRE-RETRIEVAL FILTER</strong><span>source_id = {String(response.filter.source_id)}</span></div> : null}
      <Surface>
        <div className="result-summary-head"><div><div className="result-summary-label">{response.mode === "hybrid" ? "HYBRID · RRF" : "DENSE · QDRANT"}</div><h2>{results.length} passages retrieved</h2></div><div className="latency-badge"><Clock3 size={14} /><span>{formatMs(totalLatency)}</span><small>{response.answer ? "total incl. generation" : "total retrieval"}</small></div></div>
        <div className="result-summary-meta"><span>Returned top {response.top_k}</span><span>{sourceIds.length ? `Source IDs: ${sourceIds.join(", ")}` : "Source ID not returned"}</span></div>
        {shownStages.length ? <div className="stage-latency-grid">{shownStages.map(({ label, value }) => <StageLatency key={label} label={label} value={value} />)}</div> : <p className="field-help latency-footnote">The backend returned total latency but no stage-level timing fields for this request.</p>}
        <p className="field-help latency-footnote">Stage timings are measured by the backend during this request.</p>
      </Surface>
      <div className="result-list">{results.map((item, index) => <ResultCard key={item.id} passage={item} rank={index + 1} mode={response.mode} />)}</div>
      {response.answer ? <Surface><SectionTitle title="Grounded answer" detail="Generated from the retrieved passages by the configured Azure deployment." /><p className="answer-text">{response.answer.answer}</p></Surface> : null}
    </div>
  );
}

function StageLatency({ label, value }: { label: string; value: number }) {
  return <div className="stage-latency"><span>{label}</span><strong>{formatMs(value)}</strong></div>;
}

function ResultCard({ passage, rank, mode }: { passage: RetrievedPassage; rank: number; mode: string }) {
  const [expanded, setExpanded] = useState(false);
  const method = mode === "dense" ? "Dense" : passage.retrieval_scores ? Object.keys(passage.retrieval_scores).map((key) => key === "sparse" ? "BM25" : "Dense").join(" + ") : "Hybrid / RRF";
  return (
    <motion.article className="result-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .2, delay: Math.min(rank * .035, .18) }}>
      <div className="result-card-top"><span className="result-rank">RANK #{rank.toString().padStart(2, "0")}</span><span className="result-score">SCORE&nbsp; {passage.score.toFixed(4)}</span></div>
      <div className="result-body"><p className={`result-text ${expanded ? "expanded" : ""}`}>{passage.text}</p><div className="result-meta-row"><span><strong>Passage ID</strong><code className="mono">{passage.id}</code></span><span><strong>Source ID</strong><code className="mono">{typeof passage.source_id === "number" ? passage.source_id : "—"}</code></span><span><strong>Method</strong>{method}</span></div><button className="result-expand" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>{expanded ? <>Hide metadata <ChevronUp size={13} /></> : <>View metadata <ChevronDown size={13} /></>}</button>
        <AnimatePresence initial={false}>{expanded ? <motion.div className="result-details" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: .16 }}><div className="result-detail-grid"><DetailItem label="source_id" value={passage.source_id} /><DetailItem label="passage_id" value={passage.id} /><DetailItem label="score" value={passage.score} /><DetailItem label="rank" value={rank} />{passage.retrieval_scores ? Object.entries(passage.retrieval_scores).map(([key, value]) => <DetailItem key={key} label={`${key}_score`} value={value} />) : null}</div></motion.div> : null}</AnimatePresence>
      </div>
    </motion.article>
  );
}

function DetailItem({ label, value }: { label: string; value: unknown }) {
  const rendered = typeof value === "number" ? value.toFixed(4) : typeof value === "string" ? value : value === null || value === undefined ? "—" : String(value);
  return <div className="detail-item"><span>{label}</span><code>{rendered}</code></div>;
}

function ComparisonResults({ comparison, topK }: { comparison: { dense: QueryResponse; hybrid: QueryResponse }; topK: number }) {
  const denseHits = comparison.dense.results.slice(0, topK);
  const hybridHits = comparison.hybrid.results.slice(0, topK);
  const denseRanks = new Map(denseHits.map((hit, index) => [hit.id, { hit, rank: index + 1 }]));
  const hybridRanks = new Map(hybridHits.map((hit, index) => [hit.id, { hit, rank: index + 1 }]));
  const allIds = [...new Set([...denseHits, ...hybridHits].map((hit) => hit.id))];
  return (
    <Surface>
      <SectionTitle title="Dense vs hybrid comparison" detail="Same query and filter; scores retain their retrieval-specific meaning and should not be compared as calibrated probabilities." action={<StatusPill tone="blue">LIVE COMPARISON</StatusPill>} />
      <div className="comparison-head"><div><strong>Dense</strong><small>{formatMs(comparison.dense.latency_ms)}</small></div><div><strong>Hybrid / RRF</strong><small>{formatMs(comparison.hybrid.latency_ms)}</small></div><div><strong>Change</strong></div></div>
      <div className="comparison-list">{allIds.map((id) => {
        const dense = denseRanks.get(id);
        const hybrid = hybridRanks.get(id);
        const label = dense && hybrid ? dense.rank === hybrid.rank ? "Same result" : `Rank ${dense.rank} → ${hybrid.rank}` : hybrid ? "New in hybrid" : "Dense only";
        const source = hybrid?.hit.source_id ?? dense?.hit.source_id;
        return <div className="comparison-row" key={id}><div>{dense ? <><b>#{dense.rank}</b><code>{id}</code><small>score {dense.hit.score.toFixed(4)}</small></> : <span className="comparison-absent">—</span>}</div><div>{hybrid ? <><b>#{hybrid.rank}</b><code>{id}</code><small>score {hybrid.hit.score.toFixed(4)}</small></> : <span className="comparison-absent">—</span>}</div><div><span className={`comparison-change ${dense && hybrid && dense.rank === hybrid.rank ? "change-same" : "change-new"}`}>{label}</span><small>source_id {typeof source === "number" ? source : "—"}</small></div></div>;
      })}</div>
      <div className="notice notice-info mt-16"><Sparkles size={15} /><span>Standalone BM25 comparison is not available through the current API. The hybrid result combines the backend’s dense and BM25 candidate lists with RRF.</span></div>
    </Surface>
  );
}
