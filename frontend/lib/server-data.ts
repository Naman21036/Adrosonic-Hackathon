import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "..");

export async function readReport<T>(relativePath: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(path.join(repoRoot, relativePath), "utf8")) as T;
  } catch {
    return null;
  }
}

export async function readJsonLines(relativePath: string): Promise<unknown[]> {
  try {
    const target = path.join(repoRoot, relativePath);
    const fileInfo = await stat(target);
    const handle = await import("node:fs/promises").then((fs) => fs.open(target, "r"));
    try {
      const tailSize = Math.min(fileInfo.size, 3_000_000);
      const buffer = Buffer.alloc(tailSize);
      await handle.read(buffer, 0, tailSize, Math.max(0, fileInfo.size - tailSize));
      const contents = buffer.toString("utf8").split(/\r?\n/).slice(-1_000);
      return contents.flatMap((line) => {
        try {
          return line.trim() ? [JSON.parse(line) as unknown] : [];
        } catch {
          return [];
        }
      });
    } finally {
      await handle.close();
    }
  } catch {
    return [];
  }
}

export function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

export function asBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}
