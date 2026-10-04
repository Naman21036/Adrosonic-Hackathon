import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { asNumber, asRecord, asString, readJsonLines } from "@/lib/server-data";
import type { LogEntry, ObservabilityReport } from "@/lib/types";

export const dynamic = "force-dynamic";

const repoRoot = path.resolve(process.cwd(), "..");

async function readSafeEnvironment() {
  try {
    const contents = await readFile(path.join(repoRoot, ".env"), "utf8");
    const variables = new Map<string, string>();
    for (const line of contents.split(/\r?\n/)) {
      const match = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
      if (match) variables.set(match[1], match[2].replace(/^['"]|['"]$/g, ""));
    }
    const get = (key: string) => process.env[key] ?? variables.get(key) ?? "";
    const enabled = get("LANGFUSE_ENABLED").toLowerCase() === "true";
    const credentialsPresent = Boolean(get("LANGFUSE_PUBLIC_KEY") && get("LANGFUSE_SECRET_KEY"));
    const host = get("LANGFUSE_HOST") || get("LANGFUSE_BASE_URL") || get("LANGFUSE_BASEURL");
    return { enabled, credentialsPresent, host: host.startsWith("https://") ? host : null };
  } catch {
    return { enabled: null, credentialsPresent: null, host: null };
  }
}

export async function GET() {
  const [rawLogs, langfuse] = await Promise.all([
    readJsonLines("logs/app.jsonl"),
    readSafeEnvironment(),
  ]);
  const logs: LogEntry[] = rawLogs.map(asRecord).flatMap((row) => {
    const extra = asRecord(row.extra);
    const fields = asRecord(extra.fields);
    const level = asString(asRecord(row.level).name) ?? "INFO";
    const timestampValue = asRecord(row.time);
    return [{
      timestamp: asString(timestampValue.repr) ?? asString(timestampValue.timestamp) ?? "",
      level,
      requestId: asString(extra.request_id) ?? "—",
      event: asString(extra.event) ?? asString(row.message) ?? "application_event",
      latencyMs: asNumber(fields.latency_ms) ?? asNumber(fields.retrieval_latency_ms),
      status: level === "ERROR" || level === "EXCEPTION" ? "ERROR" : "SUCCESS",
    }];
  }).reverse();

  const output: ObservabilityReport = {
    logs: logs.slice(0, 250),
    logCount: logs.length,
    langfuseEnabled: langfuse.enabled,
    langfuseCredentialsPresent: langfuse.credentialsPresent,
    langfuseHost: langfuse.host,
    langfuseTracesAvailable: false,
    langfuseMissingEndpoint: "The FastAPI service has no trace-listing endpoint.",
  };
  return NextResponse.json(output, { headers: { "cache-control": "no-store" } });
}
