"use client";

import { useCallback, useEffect, useState } from "react";
import { apiRequest, safeErrorMessage } from "@/lib/api";

export interface ResourceState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useResource<T>(url: string): ResourceState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setData(await apiRequest<T>(url));
      setError(null);
    } catch (cause) {
      setError(safeErrorMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(initial);
  }, [refresh]);
  return { data, loading, error, refresh };
}
