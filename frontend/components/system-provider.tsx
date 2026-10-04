"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { apiRequest, checkBackendHealth } from "@/lib/api";
import type { BackendStatus, IndexStats, SystemHealth } from "@/lib/types";

interface SystemState {
  health: SystemHealth;
  status: BackendStatus | null;
  indexStats: IndexStats | null;
  apiOnline: boolean;
  refreshing: boolean;
  lastChecked: Date | null;
  refresh: () => Promise<void>;
}

const SystemContext = createContext<SystemState | null>(null);

function statusFromSnapshot(snapshot: IndexStats): BackendStatus {
  return {
    indexed_passages: snapshot.qdrant_count ?? undefined,
    sparse_passages: snapshot.sparse_count ?? undefined,
    source_query_count: snapshot.source_query_count ?? undefined,
    collection: snapshot.collection ?? undefined,
    dimension: snapshot.dimension,
    sparse_ready: snapshot.sparse_count !== null,
    model: snapshot.model ?? undefined,
    sync_status: snapshot.ids_match === true ? "last_count_and_id_check_matched" : "last_validation_incomplete",
    data_source: "validated_snapshot",
    snapshot_as_of: snapshot.snapshot_as_of,
    validation_error: snapshot.validation_error,
  };
}

export function SystemProvider({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = useState<SystemHealth>({
    connectivity: "CHECKING",
    endpoint: null,
    error: null,
    http_status: null,
    checked_at: new Date().toISOString(),
  });
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [indexStats, setIndexStats] = useState<IndexStats | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setRefreshing(true);
    try {
      const [healthResult, liveStatusResult, snapshotResult] = await Promise.allSettled([
        checkBackendHealth(),
        apiRequest<BackendStatus>("/api/backend/status", undefined, 6_000),
        apiRequest<IndexStats>("/api/reports/index-stats", undefined, 6_000),
      ]);
      const nextHealth = healthResult.status === "fulfilled"
        ? healthResult.value
        : {
            connectivity: "ERROR" as const,
            endpoint: null,
            error: "Health check failed before a response was received.",
            http_status: null,
            checked_at: new Date().toISOString(),
          };
      setHealth(nextHealth);

      const snapshot = snapshotResult.status === "fulfilled" ? snapshotResult.value : null;
      setIndexStats(snapshot);
      if (liveStatusResult.status === "fulfilled" && (liveStatusResult.value.indexed_passages ?? 0) > 0) {
        setStatus({
          ...liveStatusResult.value,
          source_query_count: snapshot?.source_query_count ?? liveStatusResult.value.source_query_count,
          data_source: "live",
          snapshot_as_of: null,
        });
      } else if (snapshot) {
        setStatus(statusFromSnapshot(snapshot));
      } else if (liveStatusResult.status === "fulfilled") {
        setStatus({ ...liveStatusResult.value, data_source: "live" });
      } else {
        setStatus(null);
      }
      const checked = new Date(nextHealth.checked_at);
      setLastChecked(Number.isNaN(checked.getTime()) ? new Date() : checked);
    } finally {
      inFlight.current = false;
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, 20_000);
    const onFocus = () => void refresh();
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [refresh]);

  const value = useMemo(() => ({
    health,
    status,
    indexStats,
    apiOnline: health.connectivity === "ONLINE",
    refreshing,
    lastChecked,
    refresh,
  }), [health, status, indexStats, refreshing, lastChecked, refresh]);
  return <SystemContext.Provider value={value}>{children}</SystemContext.Provider>;
}

export function useSystemStatus(): SystemState {
  const value = useContext(SystemContext);
  if (!value) throw new Error("useSystemStatus must be used inside SystemProvider");
  return value;
}
