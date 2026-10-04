"use client";

import { createContext, useCallback, useContext, useMemo, useSyncExternalStore } from "react";

interface DemoModeState {
  enabled: boolean;
  toggle: () => void;
}

const DemoModeContext = createContext<DemoModeState | null>(null);
const storageKey = "adrosonic-demo-mode";
const changeEvent = "adrosonic-demo-mode-change";

function getSnapshot(): boolean {
  return window.localStorage.getItem(storageKey) === "true";
}

function getServerSnapshot(): boolean {
  return false;
}

function subscribe(callback: () => void): () => void {
  window.addEventListener("storage", callback);
  window.addEventListener(changeEvent, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(changeEvent, callback);
  };
}

export function DemoModeProvider({ children }: { children: React.ReactNode }) {
  const enabled = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const toggle = useCallback(() => {
    window.localStorage.setItem(storageKey, String(!getSnapshot()));
    window.dispatchEvent(new Event(changeEvent));
  }, []);
  const value = useMemo(() => ({ enabled, toggle }), [enabled, toggle]);

  return <DemoModeContext.Provider value={value}>{children}</DemoModeContext.Provider>;
}

export function useDemoMode(): DemoModeState {
  const value = useContext(DemoModeContext);
  if (!value) throw new Error("useDemoMode must be used inside DemoModeProvider");
  return value;
}
