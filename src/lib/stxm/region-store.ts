import "server-only";

import fs from "node:fs";
import path from "node:path";

import type { StxmRegion } from "@/lib/stxm-types";

export const REGIONS_CONFIG_FILENAME = "regions.json";
export const REGIONS_SCHEMA_VERSION = 1;

export type SavedScanRegions = {
  izero_lo: number;
  izero_hi: number;
  regions: StxmRegion[];
};

type RegionsConfig = {
  version: number;
  scans: Record<string, SavedScanRegions>;
};

/**
 * Normalize a raw spot label value; defaults to "pure".
 */
export function normalizeSpotLabel(raw: unknown): string {
  if (typeof raw === "string") {
    const text = raw.trim();
    return text || "pure";
  }
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return String(raw);
  }
  return "pure";
}

function normalizeRegionEntry(entry: Record<string, unknown>): StxmRegion {
  if (!("sample_lo" in entry) || !("sample_hi" in entry)) {
    throw new Error("region entry requires sample_lo and sample_hi");
  }
  let sampleLo = Number(entry.sample_lo);
  let sampleHi = Number(entry.sample_hi);
  if (sampleLo > sampleHi) {
    [sampleLo, sampleHi] = [sampleHi, sampleLo];
  }
  return {
    sample_lo: sampleLo,
    sample_hi: sampleHi,
    spot_label: normalizeSpotLabel(entry.spot_label),
  };
}

function normalizeSavedScan(raw: Record<string, unknown> | SavedScanRegions): SavedScanRegions {
  if (!("izero_lo" in raw) || !("izero_hi" in raw)) {
    throw new Error("scan entry requires izero_lo and izero_hi");
  }
  let izeroLo = Number(raw.izero_lo);
  let izeroHi = Number(raw.izero_hi);
  if (izeroLo > izeroHi) {
    [izeroLo, izeroHi] = [izeroHi, izeroLo];
  }
  const regionsRaw = raw.regions;
  if (!Array.isArray(regionsRaw) || regionsRaw.length === 0) {
    throw new Error("scan entry requires a non-empty regions list");
  }
  const regions = regionsRaw
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map(normalizeRegionEntry);
  if (regions.length === 0) {
    throw new Error("scan entry has no valid regions");
  }
  return { izero_lo: izeroLo, izero_hi: izeroHi, regions };
}

function regionsConfigPath(experimentDir: string): string {
  return path.join(experimentDir, REGIONS_CONFIG_FILENAME);
}

function scanKeyFromPath(scanPath: string): string {
  return path.basename(scanPath);
}

/**
 * Load the full regions config for an experiment directory.
 */
export function loadRegionsConfig(experimentDir: string): RegionsConfig {
  const configPath = regionsConfigPath(experimentDir);
  if (!fs.existsSync(configPath) || !fs.statSync(configPath).isFile()) {
    return { version: REGIONS_SCHEMA_VERSION, scans: {} };
  }
  try {
    const raw = JSON.parse(fs.readFileSync(configPath, "utf8")) as unknown;
    if (typeof raw !== "object" || raw === null) {
      return { version: REGIONS_SCHEMA_VERSION, scans: {} };
    }
    const record = raw as Record<string, unknown>;
    const scans =
      typeof record.scans === "object" && record.scans !== null
        ? (record.scans as Record<string, SavedScanRegions>)
        : {};
    let version = REGIONS_SCHEMA_VERSION;
    if (typeof record.version === "number") {
      version = record.version;
    } else if (typeof record.version === "string") {
      const parsed = Number.parseInt(record.version, 10);
      if (!Number.isNaN(parsed)) {
        version = parsed;
      }
    }
    return { version, scans };
  } catch {
    return { version: REGIONS_SCHEMA_VERSION, scans: {} };
  }
}

/**
 * Write the full regions config atomically to regions.json.
 */
export function saveRegionsConfig(experimentDir: string, config: RegionsConfig): void {
  const configPath = regionsConfigPath(experimentDir);
  fs.mkdirSync(experimentDir, { recursive: true });
  const payload = { version: config.version, scans: config.scans };
  const tmp = `${configPath}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, configPath);
}

/**
 * Load saved ROI bounds for one scan, if present and valid.
 */
export function loadScanRegions(
  experimentDir: string,
  scanPath: string,
): SavedScanRegions | null {
  const key = scanKeyFromPath(scanPath);
  const config = loadRegionsConfig(experimentDir);
  const raw = config.scans[key];
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  try {
    return normalizeSavedScan(raw);
  } catch {
    return null;
  }
}

/**
 * Persist ROI bounds for one scan into regions.json.
 */
export function saveScanRegions(
  experimentDir: string,
  scanPath: string,
  payload: SavedScanRegions,
): void {
  if (!payload.regions.length) {
    throw new Error("regions must be non-empty");
  }
  const normalized = normalizeSavedScan({
    izero_lo: payload.izero_lo,
    izero_hi: payload.izero_hi,
    regions: payload.regions,
  });
  const key = scanKeyFromPath(scanPath);
  const config = loadRegionsConfig(experimentDir);
  const scans = { ...config.scans, [key]: normalized };
  saveRegionsConfig(experimentDir, {
    version: config.version,
    scans,
  });
}
