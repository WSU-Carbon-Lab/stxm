import "server-only";

/**
 * Legacy Python bridge via `uv run stxm-bridge`.
 *
 * The Next.js web app uses TypeScript-native modules under `@/lib/stxm` for I/O,
 * catalog, reduction, regions, parquet, and store routes. This bridge remains
 * for LCF fitting (`/api/lcf`) and CXRO tabulated mass absorption (`mass-absorption` command)
 * until a TS port of periodictable optical constants exists.
 */

import { spawn } from "node:child_process";

import { getAllowedRoots } from "@/lib/allowed-paths.server";

export class StxmBridgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StxmBridgeError";
  }
}

function bridgeChildEnv(roots: string[]): NodeJS.ProcessEnv {
  const entries = Object.entries({
    ...process.env,
    STXM_ALLOWED_ROOTS: roots.join(":"),
  }).filter(([key]) => key !== "VIRTUAL_ENV" && key !== "CONDA_PREFIX");
  return Object.fromEntries(entries) as NodeJS.ProcessEnv;
}

function parseBridgeStdout<T>(stdout: string): T {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new StxmBridgeError("Empty response from stxm-bridge");
  }
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const lines = trimmed.split("\n").filter((line) => line.trim().length > 0);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const line = lines[index]?.trim() ?? "";
      if (!line.startsWith("{")) {
        continue;
      }
      try {
        return JSON.parse(line) as T;
      } catch {
        continue;
      }
    }
    throw new StxmBridgeError(`Invalid JSON from stxm-bridge: ${trimmed.slice(0, 200)}`);
  }
}

export async function runStxmBridge<T>(
  command: string,
  args: string[] = [],
): Promise<T> {
  const roots = getAllowedRoots();
  const bridgeArgs: string[] = [];
  for (const root of roots) {
    bridgeArgs.push("--allowed-root", root);
  }
  bridgeArgs.push(command, ...args);

  const repoRoot = process.cwd();
  const uvBin = process.env.UV_BIN ?? "uv";

  return await new Promise<T>((resolve, reject) => {
    const child = spawn(uvBin, ["run", "stxm-bridge", ...bridgeArgs], {
      cwd: repoRoot,
      env: bridgeChildEnv(roots),
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      reject(new StxmBridgeError(error.message));
    });
    child.on("close", (code) => {
      const tryResolveStdout = (): boolean => {
        try {
          const parsed = parseBridgeStdout<T & { ok?: boolean; error?: string }>(stdout);
          if (code !== 0 && parsed && typeof parsed === "object" && parsed.ok === false) {
            reject(new StxmBridgeError(String(parsed.error ?? "Bridge request failed")));
            return true;
          }
          if (code === 0) {
            resolve(parsed);
            return true;
          }
        } catch {
          return false;
        }
        return false;
      };

      if (tryResolveStdout()) {
        return;
      }

      if (code !== 0) {
        const stderrClean = stderr
          .split("\n")
          .filter(
            (line) =>
              line.trim().length > 0 &&
              !line.includes("VIRTUAL_ENV") &&
              !line.startsWith("warning:"),
          )
          .join("\n")
          .trim();
        reject(
          new StxmBridgeError(
            stderrClean || stdout.trim() || `stxm-bridge exited with code ${code}`,
          ),
        );
        return;
      }

      reject(new StxmBridgeError("Empty or invalid response from stxm-bridge"));
    });
  });
}
