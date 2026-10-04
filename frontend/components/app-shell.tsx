"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BookOpenCheck,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  Gauge,
  LayoutDashboard,
  Menu,
  Network,
  RefreshCw,
  Search,
  Settings,
  X,
} from "lucide-react";
import { SystemProvider, useSystemStatus } from "@/components/system-provider";
import { DemoModeProvider, useDemoMode } from "@/components/demo-mode-provider";
import { formatInteger } from "@/components/ui";
import { readDemoProgress, type DemoStepId } from "@/lib/demo-progress";

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard, group: "Workspace" },
  { href: "/playground", label: "Playground", icon: Search, group: "Workspace" },
  { href: "/evaluation", label: "Evaluation", icon: BookOpenCheck, group: "Workspace" },
  { href: "/benchmark", label: "Benchmark", icon: Gauge, group: "Workspace" },
  { href: "/architecture", label: "Architecture", icon: Network, group: "Platform" },
  { href: "/index", label: "Index & Corpus", icon: Database, group: "Platform" },
  { href: "/updates", label: "Live Updates", icon: RefreshCw, group: "Platform" },
  { href: "/observability", label: "Observability", icon: Activity, group: "Platform" },
  { href: "/settings", label: "Settings", icon: Settings, group: "Platform" },
];

function ShellContent({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [demoOpen, setDemoOpen] = useState(false);
  const [completedDemoSteps, setCompletedDemoSteps] = useState<DemoStepId[]>([]);
  const { enabled: demoModeEnabled, toggle: toggleDemoMode } = useDemoMode();
  const { health, status, apiOnline, refreshing, lastChecked, refresh } = useSystemStatus();
  const current = useMemo(() => navItems.find((item) => item.href === pathname) ?? navItems[0], [pathname]);
  const qdrant = health?.qdrant_ready === true;
  const bm25 = health?.sparse_ready === true;
  const embedding = health?.embedding_model_ready === true;
  const notExposed = !apiOnline;
  const apiLabel = apiOnline
    ? health.status === "ready" ? "Online · ready" : "Reachable · degraded"
    : health.connectivity === "TIMEOUT" ? "Timeout"
      : health.connectivity === "ERROR" ? "Error"
        : health.connectivity === "CHECKING" ? "Checking"
          : "Offline";

  const closeMobile = () => setMobileOpen(false);
  const demoSteps: { id: DemoStepId; label: string; href: string }[] = [
    { id: "dense", label: "Dense retrieval", href: "/playground?mode=dense" },
    { id: "ragas", label: "RAGAS baseline", href: "/evaluation" },
    { id: "hybrid", label: "Hybrid search", href: "/playground?mode=hybrid" },
    { id: "comparison", label: "Phase comparison", href: "/evaluation" },
    { id: "benchmark", label: "Latency benchmark", href: "/benchmark" },
    { id: "filter", label: "Metadata filter", href: "/playground?filter=source_id" },
  ];

  useEffect(() => {
    const initial = window.setTimeout(() => setCompletedDemoSteps(readDemoProgress()), 0);
    const onProgress = (event: Event) => {
      const step = (event as CustomEvent<DemoStepId>).detail;
      setCompletedDemoSteps((currentSteps) => currentSteps.includes(step) ? currentSteps : [...currentSteps, step]);
    };
    window.addEventListener("adrosonic-demo-progress", onProgress);
    return () => {
      window.clearTimeout(initial);
      window.removeEventListener("adrosonic-demo-progress", onProgress);
    };
  }, []);

  return (
    <div className="app-frame">
      <AnimatePresence>{mobileOpen ? <motion.button className="mobile-scrim" aria-label="Close navigation" onClick={closeMobile} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} /> : null}</AnimatePresence>
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <Link href="/" className="brand-lockup" onClick={closeMobile}>
          <span className="brand-mark">A</span>
          <span><strong>ADROSONIC</strong><small>RAG PRECISION ENGINE</small></span>
        </Link>
        <div className="sidebar-env"><span className="env-light" />LOCAL / DEMO</div>
        <nav aria-label="Primary navigation" className="side-nav">
          {(["Workspace", "Platform"] as const).map((group) => (
            <div className="nav-group" key={group}>
              <div className="nav-group-label">{group}</div>
              {navItems.filter((item) => item.group === group).map(({ href, label, icon: Icon }) => {
                const active = pathname === href;
                return (
                  <Link href={href} key={href} onClick={closeMobile} className={`nav-item ${active ? "nav-active" : ""}`} aria-current={active ? "page" : undefined}>
                    <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{active ? <motion.i layoutId="nav-active" className="nav-active-mark" /> : null}
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="sidebar-divider" />
        <div className="system-mini-title"><span>System status</span><span className={`mini-pulse ${apiOnline ? "online" : "offline"}`} /></div>
        <div className="system-mini-list">
          <StatusLine label="API" value={apiLabel} good={apiOnline && health.status === "ready"} neutral={!apiOnline && health.connectivity === "CHECKING"} />
          <StatusLine label="Qdrant" value={qdrant ? "Ready" : notExposed ? "Not exposed" : "Not ready"} good={qdrant} neutral={notExposed} />
          <StatusLine label="BM25" value={bm25 ? "Ready" : notExposed ? "Not exposed" : "Not ready"} good={bm25} neutral={notExposed} />
          <StatusLine label="Embeddings" value={embedding ? "Ready" : notExposed ? "Not exposed" : "Not ready"} good={embedding} neutral={notExposed} />
          <StatusLine label="Azure AI" value={health.azure_ready === true ? "Configured · unverified" : health.azure_ready === false ? "Not configured" : "Not exposed"} good={false} neutral={health.azure_ready === undefined} />
          <StatusLine label="Langfuse" value="Not exposed" good={false} neutral />
        </div>
        <div className="sidebar-bottom"><span className="side-version">ADROSONIC · HACKATHON DEMO</span><span className="side-version">RAG / PHASE 2</span></div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
            <div className="crumb"><span>Workspace</span><ChevronRight size={14} /><strong>{current.label}</strong></div>
          </div>
          <div className="topbar-right">
            <div className="current-index"><Database size={14} /><span>{formatInteger(status?.indexed_passages)} <small>passages</small></span></div>
            <span className={`api-chip ${apiOnline && health.status === "ready" ? "api-up" : "api-down"}`}><i />{apiLabel}</span>
            <button className={`button button-quiet demo-button ${demoModeEnabled ? "demo-button-active" : ""}`} onClick={toggleDemoMode} aria-pressed={demoModeEnabled} title={demoModeEnabled ? "Turn Demo Mode off" : "Turn Demo Mode on"}><span className={`demo-spark ${demoModeEnabled ? "demo-spark-active" : ""}`} />Demo Mode</button>
            <button className="icon-button walkthrough-button" onClick={() => setDemoOpen(true)} aria-label="Open guided demo walkthrough" title="Guided walkthrough"><CircleHelp size={16} /></button>
            <button className={`icon-button ${refreshing ? "button-loading" : ""}`} onClick={() => void refresh()} aria-label="Refresh system status" title={lastChecked ? `Last checked ${lastChecked.toLocaleTimeString()}` : "Refresh status"}><RefreshCw size={16} /></button>
          </div>
        </header>
        <main id="main-content" className="main-content">
          <AnimatePresence mode="wait">
            <motion.div key={pathname} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.18 }}>
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <AnimatePresence>
        {demoOpen ? (
          <motion.div className="modal-backdrop" role="presentation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(event) => { if (event.target === event.currentTarget) setDemoOpen(false); }}>
            <motion.section className="demo-modal" role="dialog" aria-modal="true" aria-labelledby="demo-title" initial={{ opacity: 0, scale: 0.97, y: 12 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97, y: 12 }} transition={{ duration: 0.2 }}>
              <div className="modal-header"><div><span className="eyebrow"><span className="eyebrow-dot" />GUIDED WALKTHROUGH</span><h2 id="demo-title">10-minute demo flow</h2></div><button className="icon-button" onClick={() => setDemoOpen(false)} aria-label="Close demo flow"><X size={18} /></button></div>
              <p className="modal-intro">Each step opens the live page for that part of the retrieval story.</p>
              <div className="demo-step-list">{demoSteps.map((step, index) => <button key={step.id} className={`demo-step ${completedDemoSteps.includes(step.id) ? "demo-step-complete" : ""}`} onClick={() => { setDemoOpen(false); router.push(step.href); }}><span className="step-number">0{index + 1}</span><span>{step.label}</span>{completedDemoSteps.includes(step.id) ? <CheckCircle2 size={16} aria-label="Completed" /> : <ArrowRight size={16} />}</button>)}</div>
              <div className="notice notice-info"><CircleHelp size={15} /><span>The latest completed 10-query matched-corpus RAGAS evaluation and the separate 100-query latency benchmark are available in their report pages.</span></div>
            </motion.section>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function StatusLine({ label, value, good, neutral = false }: { label: string; value: string; good: boolean; neutral?: boolean }) {
  return <div className="system-mini-row"><span><i className={`state-light ${good ? "state-good" : neutral ? "state-neutral" : "state-muted"}`} />{label}</span><small>{value}</small></div>;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  return <SystemProvider><DemoModeProvider><ShellContent>{children}</ShellContent></DemoModeProvider></SystemProvider>;
}
