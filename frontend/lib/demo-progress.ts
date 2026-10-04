export type DemoStepId = "dense" | "ragas" | "hybrid" | "comparison" | "benchmark" | "filter";

const STORAGE_KEY = "adrosonic-demo-completed-v1";

export function readDemoProgress(): DemoStepId[] {
  if (typeof window === "undefined") return [];
  try {
    const value: unknown = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(value)) return [];
    return value.filter((item): item is DemoStepId =>
      item === "dense" || item === "ragas" || item === "hybrid" || item === "comparison" || item === "benchmark" || item === "filter",
    );
  } catch {
    return [];
  }
}

export function completeDemoStep(step: DemoStepId): void {
  if (typeof window === "undefined") return;
  const completed = readDemoProgress();
  if (completed.includes(step)) return;
  window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...completed, step]));
  window.dispatchEvent(new CustomEvent<DemoStepId>("adrosonic-demo-progress", { detail: step }));
}
