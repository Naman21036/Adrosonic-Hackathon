import type { BackendHealth, SystemHealth } from "@/lib/types";

const backendApiBase = (process.env.NEXT_PUBLIC_API_URL?.trim() || "/api/backend").replace(/\/$/, "");

export class ApiClientError extends Error {
  readonly status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
  }
}

function resolveApiUrl(url: string): string {
  if (url === "/api/backend") return backendApiBase;
  if (url.startsWith("/api/backend/")) return `${backendApiBase}${url.slice("/api/backend".length)}`;
  return url;
}

function responseError(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") {
    return body.detail;
  }
  return `Request failed with status ${status}`;
}

export async function apiRequest<T>(url: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> {
  let response: Response;
  try {
    response = await fetch(resolveApiUrl(url), {
      ...init,
      cache: "no-store",
      signal: init?.signal ?? AbortSignal.timeout(timeoutMs),
    });
  } catch (cause) {
    const timedOut = cause instanceof Error && (cause.name === "TimeoutError" || cause.name === "AbortError");
    throw new ApiClientError(timedOut ? `Request timed out after ${timeoutMs} ms.` : "The API request could not reach the Next.js backend proxy.");
  }

  const contentType = response.headers.get("content-type") ?? "";
  let body: unknown;
  try {
    body = contentType.includes("application/json") ? await response.json() : await response.text();
  } catch {
    throw new ApiClientError("The API returned an unreadable response.", response.status);
  }
  if (!response.ok) throw new ApiClientError(responseError(body, response.status), response.status);
  return body as T;
}

export async function checkBackendHealth(timeoutMs = 6_500): Promise<SystemHealth> {
  const checkedAt = new Date().toISOString();
  const requestUrl = `${backendApiBase}/health`;
  try {
    const response = await fetch(requestUrl, {
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    const endpoint = response.headers.get("x-backend-endpoint") || `${backendApiBase}/health`;
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    const payload = typeof body === "object" && body !== null ? body as BackendHealth & { detail?: unknown } : null;
    if (!response.ok) {
      const connectivity = response.status === 504 ? "TIMEOUT" : response.status === 502 || response.status === 503 ? "OFFLINE" : "ERROR";
      const error = typeof payload?.detail === "string" ? payload.detail : `Health endpoint returned HTTP ${response.status}.`;
      return { ...payload, connectivity, endpoint, error, http_status: response.status, checked_at: checkedAt };
    }
    if (!payload || typeof payload.status !== "string") {
      return { connectivity: "ERROR", endpoint, error: "FastAPI returned an invalid health response.", http_status: response.status, checked_at: checkedAt };
    }
    return { ...payload, connectivity: "ONLINE", endpoint, error: null, http_status: response.status, checked_at: checkedAt };
  } catch (cause) {
    const timedOut = cause instanceof Error && (cause.name === "TimeoutError" || cause.name === "AbortError");
    return {
      connectivity: timedOut ? "TIMEOUT" : "OFFLINE",
      endpoint: `${backendApiBase}/health`,
      error: timedOut ? `Health check timed out after ${timeoutMs} ms.` : "The dashboard could not connect to the FastAPI proxy.",
      http_status: null,
      checked_at: checkedAt,
    };
  }
}

export function safeErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) return error.message;
  return "The request could not be completed. Please try again.";
}
