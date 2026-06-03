import "server-only";

import fs from "node:fs";

import { parquetReadObjects } from "hyparquet";

import { requireAllowedFile } from "@/lib/stxm/path-utils";
import type { OverlaySeries } from "@/lib/stxm-types";

type ParquetRow = Record<string, unknown>;

function parquetCellString(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return "";
}

async function readParquetRows(parquetPath: string): Promise<ParquetRow[]> {
  const buffer = fs.readFileSync(parquetPath);
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const rows = await parquetReadObjects({
    file: arrayBuffer,
  }) as ParquetRow[];
  return rows;
}

/**
 * Load experiment parquet metadata for the preview panel.
 */
export async function parquetPreview(parquetPath: string): Promise<{
  parquet_path: string;
  row_count: number;
  columns: string[];
  sample_names: string[];
  spot_labels: string[];
  scan_paths: string[];
}> {
  const resolved = requireAllowedFile(parquetPath);
  const rows = await readParquetRows(resolved);
  const columns =
    rows.length > 0
      ? Object.keys(rows[0] ?? {})
      : [];
  const unique = (field: string): string[] => {
    if (!columns.includes(field)) {
      return [];
    }
    const values = new Set<string>();
    for (const row of rows) {
      const value = row[field];
      const text = parquetCellString(value);
      if (text !== "") {
        values.add(text);
      }
    }
    return [...values].filter(Boolean).sort();
  };
  return {
    parquet_path: resolved,
    row_count: rows.length,
    columns,
    sample_names: unique("sample_name"),
    spot_labels: unique("spot_label"),
    scan_paths: unique("scan_path"),
  };
}

/**
 * Load grouped overlay series from an experiment parquet file.
 */
export async function parquetSpectra(
  parquetPath: string,
  filters: {
    sampleName?: string;
    spotLabel?: string;
    scanPath?: string;
    useNormalized?: boolean;
  },
): Promise<{ series: OverlaySeries[] }> {
  const resolved = requireAllowedFile(parquetPath);
  let rows = await readParquetRows(resolved);
  if (filters.sampleName) {
    rows = rows.filter((row) => parquetCellString(row.sample_name) === filters.sampleName);
  }
  if (filters.spotLabel) {
    rows = rows.filter((row) => parquetCellString(row.spot_label) === filters.spotLabel);
  }
  if (filters.scanPath) {
    rows = rows.filter((row) => parquetCellString(row.scan_path) === filters.scanPath);
  }
  const yCol =
    filters.useNormalized && rows.some((row) => row.OD_normalized !== undefined)
      ? "OD_normalized"
      : "OD";
  const groupCols = ["sample_name", "spot_label", "scan_path", "film_region_name"].filter((col) =>
    rows.some((row) => row[col] !== undefined),
  );
  const groups = new Map<string, ParquetRow[]>();
  for (const row of rows) {
    const keyParts = groupCols.map((col) => `${col}=${parquetCellString(row[col])}`);
    const key = keyParts.join(", ");
    const bucket = groups.get(key) ?? [];
    bucket.push(row);
    groups.set(key, bucket);
  }
  const series: OverlaySeries[] = [];
  for (const [label, group] of groups) {
    const sorted = [...group].sort(
      (a, b) => Number(a.energy_eV ?? 0) - Number(b.energy_eV ?? 0),
    );
    series.push({
      label,
      energy_eV: sorted.map((row) => Number(row.energy_eV)),
      y: sorted.map((row) => Number(row[yCol])),
      y_err: sorted.map((row) => Number(row.OD_err ?? 0)),
    });
  }
  return { series };
}

export { readParquetRows };
