import "server-only";

import fs from "node:fs";

import { validatePathUnderAllowedRoots } from "@/lib/allowed-paths.server";

export type ResolvedPath =
  | { ok: true; resolved: string }
  | { ok: false; error: string };

/**
 * Resolve and validate a filesystem path against STXM allowed roots.
 */
export function resolveAllowedPath(inputPath: string): ResolvedPath {
  return validatePathUnderAllowedRoots(inputPath);
}

/**
 * Resolve a path or return a structured error message for API payloads.
 */
export function requireAllowedPath(inputPath: string): string {
  const result = resolveAllowedPath(inputPath);
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.resolved;
}

/**
 * Resolve a path that must exist as a directory.
 */
export function requireAllowedDirectory(inputPath: string): string {
  const resolved = requireAllowedPath(inputPath);
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
    throw new Error(`Not a directory: ${resolved}`);
  }
  return resolved;
}

/**
 * Resolve a path that must exist as a regular file.
 */
export function requireAllowedFile(inputPath: string): string {
  const resolved = requireAllowedPath(inputPath);
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`Scan header not found: ${resolved}`);
  }
  return resolved;
}
