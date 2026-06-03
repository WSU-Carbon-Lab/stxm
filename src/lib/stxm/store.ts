import "server-only";

import fs from "node:fs";
import path from "node:path";

import { parquetReadObjects } from "hyparquet";

import { requireAllowedDirectory } from "@/lib/stxm/path-utils";

function iterSpectrumParquets(storeRoot: string): string[] {
  if (!fs.existsSync(storeRoot) || !fs.statSync(storeRoot).isDirectory()) {
    return [];
  }
  const results: string[] = [];
  const walk = (dir: string): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile() && entry.name.endsWith(".parquet")) {
        const rel = path.relative(storeRoot, full);
        const parts = rel.split(path.sep);
        if (
          parts.length >= 4 &&
          parts[0]?.startsWith("sample=") &&
          parts[1]?.startsWith("region=") &&
          parts[2]?.startsWith("edge=")
        ) {
          results.push(full);
        }
      }
    }
  };
  walk(storeRoot);
  return results.sort();
}

/**
 * List one row per stored spectrum file with partition keys and paths.
 */
export function listStoreManifest(storeRoot: string): {
  store_root: string;
  entries: Record<string, string>[];
} {
  const resolved = requireAllowedDirectory(storeRoot);
  const entries: Record<string, string>[] = [];
  for (const parquetPath of iterSpectrumParquets(resolved)) {
    const relParts = path.relative(resolved, parquetPath).split(path.sep);
    if (relParts.length < 4) {
      continue;
    }
    const jsonPath = parquetPath.replace(/\.parquet$/, ".json");
    const stem = path.basename(parquetPath, ".parquet");
    const scanId = stem.includes("__") ? (stem.split("__")[0] ?? stem) : stem;
    const created = stem.includes("__") ? (stem.split("__")[1] ?? "") : "";
    entries.push({
      sample: relParts[0]?.split("=", 2)[1] ?? "",
      region: relParts[1]?.split("=", 2)[1] ?? "",
      edge: relParts[2]?.split("=", 2)[1] ?? "",
      parquet_path: parquetPath,
      json_path: fs.existsSync(jsonPath) ? jsonPath : "",
      scan_id: scanId,
      created_utc: created,
    });
  }
  return { store_root: resolved, entries };
}

type StoreRow = Record<string, unknown>;

function cellString(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

/**
 * Query partitioned store spectra and return grouped overlay series.
 */
export async function queryStoreSpectra(
  storeRoot: string,
  filters: { sample?: string; region?: string; edge?: string },
): Promise<{
  series: Array<{
    label: string;
    energy_eV: number[];
    OD: number[];
    OD_err: number[];
  }>;
}> {
  const resolved = requireAllowedDirectory(storeRoot);
  const paths = iterSpectrumParquets(resolved);
  const frames: StoreRow[][] = [];
  for (const parquetPath of paths) {
    const relParts = path.relative(resolved, parquetPath).split(path.sep);
    if (relParts.length < 4) {
      continue;
    }
    const partSample = relParts[0]?.split("=", 2)[1] ?? "";
    const partRegion = relParts[1]?.split("=", 2)[1] ?? "";
    const partEdge = relParts[2]?.split("=", 2)[1] ?? "";
    if (filters.sample !== undefined && filters.sample !== "" && partSample !== filters.sample) {
      continue;
    }
    if (filters.region !== undefined && filters.region !== "" && partRegion !== filters.region) {
      continue;
    }
    if (filters.edge !== undefined && filters.edge !== "" && partEdge !== filters.edge) {
      continue;
    }
    try {
      const buffer = fs.readFileSync(parquetPath);
      const arrayBuffer = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength,
      );
      const rows = (await parquetReadObjects({
        file: arrayBuffer,
      })) as StoreRow[];
      frames.push(rows);
    } catch {
      continue;
    }
  }
  const combined = frames.flat();
  const groups = new Map<string, StoreRow[]>();
  for (const row of combined) {
    const sampleName = cellString(row.sample_name);
    const regionLabel = cellString(row.region_label);
    const key = `${sampleName}\0${regionLabel}`;
    const bucket = groups.get(key) ?? [];
    bucket.push(row);
    groups.set(key, bucket);
  }
  const series: Array<{
    label: string;
    energy_eV: number[];
    OD: number[];
    OD_err: number[];
  }> = [];
  for (const [key, group] of groups) {
    const [sampleName, regionLabel] = key.split("\0");
    const sorted = [...group].sort(
      (a, b) => Number(a.energy_eV ?? 0) - Number(b.energy_eV ?? 0),
    );
    series.push({
      label: `${sampleName} / ${regionLabel}`,
      energy_eV: sorted.map((row) => Number(row.energy_eV)),
      OD: sorted.map((row) => Number(row.OD)),
      OD_err: sorted.map((row) => Number(row.OD_err ?? 0)),
    });
  }
  return { series };
}
