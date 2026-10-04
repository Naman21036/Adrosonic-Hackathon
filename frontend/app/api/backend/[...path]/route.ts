import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ path: string[] }> };

const backendBase = (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

function endpointFor(route: string): string {
  try {
    const endpoint = new URL(`${backendBase}/${route}`);
    endpoint.username = "";
    endpoint.password = "";
    endpoint.search = "";
    endpoint.hash = "";
    return endpoint.toString();
  } catch {
    return "FastAPI endpoint configured by BACKEND_API_URL";
  }
}

function errorCodes(value: unknown): string[] {
  const pending: unknown[] = [value];
  const found: string[] = [];
  while (pending.length > 0 && found.length < 8) {
    const current = pending.shift();
    if (typeof current !== "object" || current === null) continue;
    const item = current as { code?: unknown; cause?: unknown; errors?: unknown };
    if (typeof item.code === "string") found.push(item.code);
    if (item.cause) pending.push(item.cause);
    if (Array.isArray(item.errors)) pending.push(...item.errors);
  }
  return found;
}

function responseHeaders(upstream: Response, endpoint: string): HeadersInit {
  return {
    "content-type": upstream.headers.get("content-type") ?? "application/json",
    "x-backend-endpoint": endpoint,
  };
}

async function forward(request: NextRequest, path: string[], method: "GET" | "POST" | "DELETE") {
  const route = path.join("/");
  const isReadRoute = method === "GET" && (route === "health" || route === "status");
  const isWriteRoute = method === "POST" && (route === "search" || route === "upsert");
  const isDeleteRoute = method === "DELETE" && path.length === 2 && path[0] === "passages";
  if (!isReadRoute && !isWriteRoute && !isDeleteRoute) {
    return NextResponse.json({ detail: "Backend route is not available through this interface" }, { status: 404 });
  }

  const targetRoute = isDeleteRoute ? `passages/${encodeURIComponent(path[1])}` : route;
  const endpoint = endpointFor(targetRoute);
  const url = `${backendBase}/${targetRoute}`;
  const timeoutMs = route === "health" ? 5_000 : 45_000;
  try {
    const upstream = await fetch(url, {
      method,
      headers: method === "POST" ? { "content-type": "application/json" } : undefined,
      body: method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    return new NextResponse(await upstream.text(), {
      status: upstream.status,
      headers: responseHeaders(upstream, endpoint),
    });
  } catch (error) {
    const timedOut = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    const codes = errorCodes(error);
    const refused = codes.includes("ECONNREFUSED");
    const dnsFailure = codes.includes("ENOTFOUND") || codes.includes("EAI_AGAIN");
    const status = timedOut ? 504 : 502;
    const detail = timedOut
      ? `FastAPI did not answer at ${endpoint} within ${timeoutMs} ms.`
      : refused
        ? `Connection refused by FastAPI at ${endpoint}. Start the API service and retry.`
        : dnsFailure
          ? `The FastAPI hostname for ${endpoint} could not be resolved.`
        : `Could not connect to FastAPI at ${endpoint}. Verify the configured host and port.`;
    return NextResponse.json(
      { detail },
      { status, headers: { "x-backend-endpoint": endpoint, "cache-control": "no-store" } },
    );
  }
}

export async function GET(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).path, "GET");
}

export async function POST(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).path, "POST");
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).path, "DELETE");
}
