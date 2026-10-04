"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ArrowDownToLine, CheckCircle2, Clock3, FilePlus2, Play, Trash2, X } from "lucide-react";
import { PageHeading, StatusPill, Surface, formatMs } from "@/components/ui";
import { apiRequest, safeErrorMessage } from "@/lib/api";
import type { LiveMutationResponse, QueryResponse } from "@/lib/types";

type Tab = "upsert" | "delete";
type DemoStep = "idle" | "created" | "verified-v1" | "updated" | "verified-v2" | "deleted";

const DEMO_ID = "demo-live-001";
const DEMO_V1 = "Adrosonic demonstrates incremental live index updates without full corpus reindexing.";
const DEMO_V2 = "Adrosonic supports incremental vector database updates without rebuilding the corpus.";

export default function UpdatesPage() {
  const [tab, setTab] = useState<Tab>("upsert");
  const [passageId, setPassageId] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [text, setText] = useState("");
  const [deleteId, setDeleteId] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pending, setPending] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LiveMutationResponse | null>(null);
  const [demoStep, setDemoStep] = useState<DemoStep>("idle");
  const [demoMode, setDemoMode] = useState<"dense" | "hybrid">("hybrid");
  const [demoVerification, setDemoVerification] = useState<{ mode: string; retrieved: boolean; passage: string | null } | null>(null);

  const validUpsert = Boolean(passageId.trim().length > 0 && passageId.trim().length <= 256 && sourceId.trim().length > 0 && sourceId.trim().length <= 256 && text.trim().length > 0 && text.trim().length <= 20_000);

  async function submitUpsert(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null); setResult(null);
    if (!validUpsert) { setError("Passage ID, source ID, and passage text are required. IDs and source IDs must be at most 256 characters."); return; }
    setPending(true);
    try {
      const response = await apiRequest<LiveMutationResponse>("/api/backend/upsert", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id: passageId.trim(), text: text.trim(), source_id: sourceId.trim() }),
      });
      setResult(response);
      if (response.passage_id === DEMO_ID && text.trim() === DEMO_V1) { setDemoStep("created"); setDemoVerification(null); }
      if (response.passage_id === DEMO_ID && text.trim() === DEMO_V2) { setDemoStep("updated"); setDemoVerification(null); }
    } catch (cause) { setError(safeErrorMessage(cause)); }
    finally { setPending(false); }
  }

  async function submitDelete() {
    const id = deleteId.trim();
    if (!id || id.length > 256) { setError("Enter one exact passage ID up to 256 characters before deleting."); setConfirmDelete(false); return; }
    setError(null); setResult(null); setPending(true); setConfirmDelete(false);
    try {
      const response = await apiRequest<LiveMutationResponse>(`/api/backend/passages/${encodeURIComponent(id)}`, { method: "DELETE" });
      setResult(response);
      if (response.passage_id === DEMO_ID) { setDemoStep("deleted"); setDemoVerification(null); }
      setDeleteId("");
    } catch (cause) { setError(safeErrorMessage(cause)); }
    finally { setPending(false); }
  }

  function prepareDemo() {
    setTab("upsert");
    setPassageId(DEMO_ID);
    setSourceId("demo");
    setText(DEMO_V1);
    setDemoStep("idle");
    setDemoVerification(null);
    setError(null);
  }

  function prepareDemoUpdate() {
    setTab("upsert");
    setPassageId(DEMO_ID);
    setSourceId("demo");
    setText(DEMO_V2);
    setDemoVerification(null);
    setError(null);
  }

  async function verifyDemoRetrieval() {
    setError(null); setDemoVerification(null); setVerifying(true);
    const query = demoStep === "created" ? "How does Adrosonic demonstrate incremental live index updates?" : "How does Adrosonic support incremental vector database updates?";
    try {
      const response = await apiRequest<QueryResponse>("/api/backend/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, mode: demoMode, top_k: 5 }),
      });
      const passage = response.results.find((item) => item.id === DEMO_ID);
      const retrieved = Boolean(passage);
      setDemoVerification({ mode: demoMode, retrieved, passage: passage?.text ?? null });
      if (retrieved && demoStep === "created") setDemoStep("verified-v1");
      if (retrieved && demoStep === "updated") setDemoStep("verified-v2");
      if (!retrieved && demoStep === "deleted") setDemoVerification({ mode: demoMode, retrieved: false, passage: null });
    } catch (cause) { setError(safeErrorMessage(cause)); }
    finally { setVerifying(false); }
  }

  return (
    <div className="stack">
      <PageHeading eyebrow="INCREMENTAL INDEX MANAGEMENT" title="Live Index Updates" description="Use the existing passage mutation endpoints for a single upsert or delete. No corpus rebuild is triggered." action={<StatusPill tone="blue">SINGLE-PASSAGE OPS</StatusPill>} />
      <div className="notice notice-warn"><AlertTriangle size={15} /><span>These actions change the configured production index when the backend is online. Confirm passage IDs and content before submitting.</span></div>
      <Surface className="demo-workflow">
        <div className="demo-workflow-heading"><div><span className="eyebrow"><span className="eyebrow-dot" />ONE-MINUTE WALKTHROUGH</span><h2>CREATE → RETRIEVE → UPDATE → RETRIEVE → DELETE</h2><p>Uses one exact demo ID. Each action calls the same single-passage API as the manual forms.</p></div><StatusPill tone={demoStep === "deleted" ? "good" : "blue"}>{demoStep.replace("-", " ").toUpperCase()}</StatusPill></div>
        <div className="demo-workflow-controls"><button className="button button-quiet" onClick={prepareDemo}>Prepare demo passage</button><label className="demo-mode-select">Verification mode<select className="select-field compact-select" value={demoMode} onChange={(event) => setDemoMode(event.target.value as "dense" | "hybrid")}><option value="dense">Dense</option><option value="hybrid">Hybrid / RRF</option></select></label>
          {(demoStep === "created" || demoStep === "updated" || demoStep === "deleted") ? <button className="button button-primary" onClick={() => void verifyDemoRetrieval()} disabled={pending || verifying}>{verifying ? <span className="mini-spinner" /> : <Play size={13} />}Verify {demoMode === "hybrid" ? "Hybrid" : "Dense"} retrieval</button> : null}
          {demoStep === "verified-v1" ? <button className="button button-quiet" onClick={prepareDemoUpdate}>Load version 2 text</button> : null}
          {demoStep === "verified-v2" ? <button className="button button-danger" onClick={() => { setTab("delete"); setDeleteId(DEMO_ID); setConfirmDelete(true); }}>Delete demo passage</button> : null}
          {demoStep === "deleted" ? <StatusPill tone="good">Removed from both indexes</StatusPill> : null}
        </div>
        {demoVerification ? <div className={`demo-verification ${demoVerification.retrieved ? "demo-verification-good" : demoStep === "deleted" ? "demo-verification-good" : "demo-verification-pending"}`}><strong>{demoVerification.retrieved ? `${DEMO_ID} retrieved through ${demoVerification.mode === "hybrid" ? "Hybrid / RRF" : "Dense"}` : demoStep === "deleted" ? `No ${DEMO_ID} result after delete` : `${DEMO_ID} was not in the top 5`}</strong>{demoVerification.passage ? <p>{demoVerification.passage}</p> : null}</div> : null}
        <p className="field-help demo-workflow-note">BM25 is exercised by hybrid retrieval and the backend tests. The existing HTTP API does not expose standalone BM25 mode.</p>
      </Surface>
      <Surface className="updates-surface">
        <div className="updates-tabs" role="tablist" aria-label="Index mutation type"><button role="tab" aria-selected={tab === "upsert"} className={tab === "upsert" ? "active" : ""} onClick={() => { setTab("upsert"); setError(null); }}><FilePlus2 size={15} />Upsert passage</button><button role="tab" aria-selected={tab === "delete"} className={tab === "delete" ? "active tab-danger" : ""} onClick={() => { setTab("delete"); setError(null); }}><Trash2 size={15} />Delete passage</button></div>
        {tab === "upsert" ? <form className="mutation-form" onSubmit={(event) => void submitUpsert(event)}>
          <div className="form-intro"><span className="form-icon"><ArrowDownToLine size={17} /></span><div><h2>Upsert passage</h2><p>Embeds one passage and updates that exact Qdrant point, BM25 row, and recovery journal entry.</p></div></div>
          <div className="grid grid-2"><div><label className="label" htmlFor="passage-id">Passage ID</label><input id="passage-id" className="field" maxLength={256} value={passageId} onChange={(event) => setPassageId(event.target.value)} placeholder="demo-live-001 or msmarco-23013-4" autoComplete="off" /></div><div><label className="label" htmlFor="source-id-upsert">Source ID</label><input id="source-id-upsert" className="field" maxLength={256} value={sourceId} onChange={(event) => setSourceId(event.target.value)} placeholder="demo or 23013" /></div></div>
          <div><label className="label" htmlFor="passage-text">Passage text</label><textarea id="passage-text" className="text-field mutation-text" value={text} onChange={(event) => setText(event.target.value)} maxLength={20_000} placeholder="Paste the passage content to index." /></div>
          <div className="form-footer"><p>Canonical MS MARCO IDs derive their numeric source ID; custom IDs use the supplied source ID.</p><button className="button button-primary" type="submit" disabled={pending || !validUpsert}>{pending ? <span className="mini-spinner" /> : <FilePlus2 size={14} />}Upsert passage</button></div>
        </form> : <div className="mutation-form"><div className="form-intro"><span className="form-icon danger-form-icon"><Trash2 size={17} /></span><div><h2>Delete passage</h2><p>Remove one exact passage ID from the dense and sparse indexes.</p></div></div><div><label className="label" htmlFor="delete-id">Passage ID</label><input id="delete-id" className="field" value={deleteId} onChange={(event) => setDeleteId(event.target.value)} placeholder="msmarco-23013-4" autoComplete="off" /></div><div className="form-footer"><p>Deletion requires a second confirmation and cannot be undone from this page.</p><button className="button button-danger" disabled={pending || !deleteId.trim()} onClick={() => { setError(null); setConfirmDelete(true); }}><Trash2 size={14} />Review delete</button></div></div>}
        {error ? <div className="notice notice-error mt-20" role="alert"><AlertTriangle size={15} />{error}</div> : null}
        <AnimatePresence>{result ? <motion.div className="mutation-result" initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><span className="mutation-success"><CheckCircle2 size={16} /></span><div><strong>{result.operation === "upsert" ? `Passage ${result.status}` : "Passage deleted"}</strong><small><span className="mono">{result.passage_id}</span> · Qdrant {result.operation === "upsert" ? "updated" : result.deleted_from_qdrant ? "deleted" : "already absent"} · BM25 {result.operation === "upsert" ? "updated" : result.deleted_from_bm25 ? "deleted" : "already absent"}</small><small>Indexed passages {result.indexed_passages_before} → {result.indexed_passages_after} · embedding delta {result.embedding_count_delta}</small></div><div className="mutation-time"><Clock3 size={13} />{formatMs(result.latency_ms)} <small>backend latency</small></div></motion.div> : null}</AnimatePresence>
      </Surface>
      <div className="notice notice-info"><Clock3 size={14} /><span>The API returns backend operation latency, created/updated state, exact dense and sparse count deltas, and embedding count. Qdrant remains first; failed sparse synchronization stays journaled for recovery.</span></div>
      <AnimatePresence>{confirmDelete ? <motion.div className="modal-backdrop" role="presentation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(event) => { if (event.target === event.currentTarget) setConfirmDelete(false); }}><motion.section className="dialog-card" role="alertdialog" aria-modal="true" aria-labelledby="delete-title" initial={{ opacity: 0, scale: .97, y: 10 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: .97 }}><div className="modal-header"><div><span className="eyebrow error-eyebrow"><span className="eyebrow-dot" />DESTRUCTIVE ACTION</span><h2 id="delete-title">Delete this passage?</h2></div><button className="icon-button" aria-label="Close confirmation" onClick={() => setConfirmDelete(false)}><X size={17} /></button></div><p className="delete-confirm-copy">This will remove <code className="mono">{deleteId}</code> from Qdrant and BM25 through the configured API. This cannot be undone from the frontend.</p><div className="modal-actions"><button className="button button-quiet" onClick={() => setConfirmDelete(false)}>Cancel</button><button className="button button-danger" disabled={pending} onClick={() => void submitDelete()}>{pending ? <span className="mini-spinner" /> : <Trash2 size={14} />}Confirm delete</button></div></motion.section></motion.div> : null}</AnimatePresence>
    </div>
  );
}
