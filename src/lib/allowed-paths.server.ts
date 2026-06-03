import "server-only";

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { env } from "@/env.js";

function expandUser(inputPath: string): string {
  const trimmed = inputPath.trim();
  if (trimmed === "~") {
    return os.homedir();
  }
  if (trimmed.startsWith("~/")) {
    return path.join(os.homedir(), trimmed.slice(2));
  }
  return trimmed;
}

function resolveRootPath(root: string): string {
  try {
    return fs.realpathSync.native(expandUser(root));
  } catch {
    return path.resolve(expandUser(root));
  }
}

function dedupeRoots(roots: string[]): string[] {
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const root of roots) {
    const resolved = resolveRootPath(root);
    if (seen.has(resolved)) {
      continue;
    }
    seen.add(resolved);
    unique.push(root.trim());
  }
  return unique;
}

function detectMacCloudStorageRoots(home: string): string[] {
  const cloudStorage = path.join(home, "Library", "CloudStorage");
  if (!fs.existsSync(cloudStorage)) {
    return [];
  }
  const roots = [cloudStorage];
  try {
    for (const entry of fs.readdirSync(cloudStorage, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }
      const name = entry.name;
      if (name.startsWith("OneDrive-") || name.startsWith("OneDrive@")) {
        roots.push(path.join(cloudStorage, name));
      }
    }
  } catch {
    return roots;
  }
  return roots;
}

export function getDefaultAllowedRoots(): string[] {
  const home = os.homedir();
  const roots = [home];
  if (process.platform === "darwin") {
    roots.push(...detectMacCloudStorageRoots(home));
  }
  return dedupeRoots(roots);
}

function parseConfiguredAllowedRoots(): string[] {
  const configuredRaw = process.env.STXM_ALLOWED_ROOTS ?? env.STXM_ALLOWED_ROOTS;
  const configured =
    configuredRaw
      ?.split(":")
      .map((part) => part.trim())
      .filter(Boolean) ?? [];
  return configured;
}

export function getAllowedRoots(): string[] {
  const configured = parseConfiguredAllowedRoots();
  if (configured.length > 0) {
    return configured;
  }
  return getDefaultAllowedRoots();
}

function resolveFilesystemPath(inputPath: string): string {
  const expanded = expandUser(inputPath);
  try {
    return fs.realpathSync.native(expanded);
  } catch {
    return path.resolve(expanded);
  }
}

function isPathUnderRoot(resolved: string, rootResolved: string): boolean {
  if (resolved === rootResolved) {
    return true;
  }
  const relative = path.relative(rootResolved, resolved);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export type PathValidationResult =
  | { ok: true; resolved: string }
  | { ok: false; error: string };

export function validatePathUnderAllowedRoots(inputPath: string): PathValidationResult {
  const resolved = resolveFilesystemPath(inputPath);
  const roots = getAllowedRoots();
  for (const root of roots) {
    const rootResolved = resolveRootPath(root);
    if (isPathUnderRoot(resolved, rootResolved)) {
      return { ok: true, resolved };
    }
  }
  const rootsLabel = roots.join(", ");
  return {
    ok: false,
    error: `Path is outside allowed roots (${rootsLabel}): ${inputPath.trim()}`,
  };
}
