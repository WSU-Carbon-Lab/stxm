import "server-only";

import fs from "node:fs";
import path from "node:path";

import type { ScanCategory } from "@/lib/stxm-types";

export const NEXAFS_LINE_SCAN_TYPE = 'Type = "NEXAFS Line Scan"';
export const MIN_BYTES_PER_VALUE = 3;

const HDR_TYPE_PATTERN = /Type\s*=\s*"([^"]*)"/;

export type HdrMeta = {
  paxis_count: number;
  qaxis_count: number;
  raw: string;
  paxis_name?: string;
  qaxis_name?: string;
  paxis_points?: number[];
  qaxis_points?: number[];
  energy_eV?: number | null;
  energy_min_eV?: number | null;
  energy_max_eV?: number | null;
  num_energy_points?: number | null;
};

function parsePointsCount(text: string, axisName: string): number | null {
  const pattern = new RegExp(
    `${axisName}\\s*=\\s*\\{[^}]*Points\\s*=\\s*\\(\\s*(\\d+)`,
    "s",
  );
  const match = pattern.exec(text);
  return match ? Number.parseInt(match[1] ?? "", 10) : null;
}

function parsePointsArray(text: string, axisName: string): number[] | null {
  const pattern = new RegExp(
    `${axisName}\\s*=\\s*\\{[^}]*Points\\s*=\\s*\\(\\s*\\d+\\s*,\\s*([\\d\\s.,\\-eE+]+)\\)`,
    "s",
  );
  const match = pattern.exec(text);
  if (!match?.[1]) {
    return null;
  }
  return match[1]
    .replace(/,/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((value) => Number.parseFloat(value));
}

function parseHdrScalar(raw: string, field: string): number | null {
  const number = String.raw`[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?`;
  const pattern = new RegExp(String.raw`^\s*${field}\s*=\s*(${number})\s*(?:$|\#)`, "im");
  const match = pattern.exec(raw);
  return match?.[1] ? Number.parseFloat(match[1]) : null;
}

function parseHdrInt(raw: string, field: string): number | null {
  const pattern = new RegExp(String.raw`^\s*${field}\s*=\s*(\d+)\s*(?:$|\#)`, "im");
  const match = pattern.exec(raw);
  return match?.[1] ? Number.parseInt(match[1], 10) : null;
}

function axisIsEnergy(name: string): boolean {
  const lowered = name.toLowerCase();
  return lowered.includes("energy") || lowered.endsWith("ev") || lowered.includes("photon");
}

function energyAxisValues(meta: HdrMeta): number[] | null {
  const pName = meta.paxis_name ?? "";
  const qName = meta.qaxis_name ?? "";
  const pPoints = meta.paxis_points;
  const qPoints = meta.qaxis_points;
  if (axisIsEnergy(pName) && pPoints) {
    return pPoints;
  }
  if (axisIsEnergy(qName) && qPoints) {
    return qPoints;
  }
  return null;
}

/**
 * Derive catalog energy metadata from a parsed STXM header dict.
 */
export function extractScanEnergy(meta: Pick<HdrMeta, "raw"> & Partial<HdrMeta>): {
  energy_eV: number | null;
  energy_min_eV: number | null;
  energy_max_eV: number | null;
  num_energy_points: number | null;
} {
  const empty = {
    energy_eV: null,
    energy_min_eV: null,
    energy_max_eV: null,
    num_energy_points: null,
  };
  const axisValues = energyAxisValues(meta as HdrMeta);
  if (axisValues && axisValues.length > 0) {
    const lo = Math.min(...axisValues);
    const hi = Math.max(...axisValues);
    const count = axisValues.length;
    if (count === 1 || Math.abs(lo - hi) < 1e-6) {
      return {
        energy_eV: lo,
        energy_min_eV: lo,
        energy_max_eV: hi,
        num_energy_points: count,
      };
    }
    return {
      energy_eV: null,
      energy_min_eV: lo,
      energy_max_eV: hi,
      num_energy_points: count,
    };
  }
  const raw = meta.raw;
  let energy = parseHdrScalar(raw, "Energy");
  energy ??= parseHdrScalar(raw, "PhotonEnergy");
  const start = parseHdrScalar(raw, "StartEnergy");
  const end = parseHdrScalar(raw, "EndEnergy");
  const numEnergy = parseHdrInt(raw, "NumEnergy");
  if (energy !== null) {
    return {
      energy_eV: energy,
      energy_min_eV: energy,
      energy_max_eV: energy,
      num_energy_points: numEnergy ?? 1,
    };
  }
  if (start !== null && end !== null) {
    const lo = Math.min(start, end);
    const hi = Math.max(start, end);
    const count = numEnergy;
    if (Math.abs(lo - hi) < 1e-6) {
      return {
        energy_eV: lo,
        energy_min_eV: lo,
        energy_max_eV: hi,
        num_energy_points: count ?? 1,
      };
    }
    return {
      energy_eV: null,
      energy_min_eV: lo,
      energy_max_eV: hi,
      num_energy_points: count,
    };
  }
  if (start !== null) {
    return {
      energy_eV: start,
      energy_min_eV: start,
      energy_max_eV: start,
      num_energy_points: numEnergy ?? 1,
    };
  }
  if (numEnergy !== null && numEnergy > 1) {
    return {
      energy_eV: null,
      energy_min_eV: null,
      energy_max_eV: null,
      num_energy_points: numEnergy,
    };
  }
  return empty;
}

/**
 * Parse STXM .hdr file metadata including axis sizes and optional point arrays.
 */
export function readHdr(hdrPath: string): HdrMeta {
  const raw = fs.readFileSync(hdrPath, "utf8");
  const paxisCount = parsePointsCount(raw, "PAxis");
  const qaxisCount = parsePointsCount(raw, "QAxis");
  if (paxisCount === null || qaxisCount === null) {
    throw new Error("Could not find PAxis or QAxis Points in header");
  }
  const out: HdrMeta = {
    paxis_count: paxisCount,
    qaxis_count: qaxisCount,
    raw,
  };
  const pname = /PAxis\s*=\s*\{\s*Name\s*=\s*"([^"]*)"/.exec(raw);
  const qname = /QAxis\s*=\s*\{\s*Name\s*=\s*"([^"]*)"/.exec(raw);
  if (pname?.[1]) {
    out.paxis_name = pname[1];
  }
  if (qname?.[1]) {
    out.qaxis_name = qname[1];
  }
  const parr = parsePointsArray(raw, "PAxis");
  const qarr = parsePointsArray(raw, "QAxis");
  if (parr) {
    out.paxis_points = parr;
  }
  if (qarr) {
    out.qaxis_points = qarr;
  }
  Object.assign(out, extractScanEnergy(out));
  return out;
}

/**
 * Load STXM .xim ascii image as a 2D float array (rows x cols).
 */
export function readXim(ximPath: string, shape?: [number, number]): number[][] {
  const text = fs.readFileSync(ximPath, "utf8").trim();
  const rows = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const data = rows.map((line) =>
    line
      .trim()
      .split(/\s+/)
      .map((value) => Number.parseFloat(value)),
  );
  if (data.length === 1 && shape) {
    const [nRows, nCols] = shape;
    const flat = data[0] ?? [];
    if (flat.length !== nRows * nCols) {
      throw new Error(`xim flat length ${flat.length} != ${nRows * nCols}`);
    }
    const reshaped: number[][] = [];
    for (let row = 0; row < nRows; row += 1) {
      reshaped.push(flat.slice(row * nCols, (row + 1) * nCols));
    }
    return reshaped;
  }
  if (shape) {
    const [nRows, nCols] = shape;
    const total = data.reduce((sum, row) => sum + row.length, 0);
    if (total === nRows * nCols && (data.length !== nRows || (data[0]?.length ?? 0) !== nCols)) {
      const flat = data.flat();
      const reshaped: number[][] = [];
      for (let row = 0; row < nRows; row += 1) {
        reshaped.push(flat.slice(row * nCols, (row + 1) * nCols));
      }
      return reshaped;
    }
  }
  return data;
}

function resolveXimPath(hdrPath: string): string {
  const parent = path.dirname(hdrPath);
  const stem = path.basename(hdrPath, path.extname(hdrPath));
  let ximPath = path.join(parent, `${stem}_a.xim`);
  if (!fs.existsSync(ximPath)) {
    ximPath = path.join(parent, `${stem}.xim`);
  }
  if (!fs.existsSync(ximPath)) {
    throw new Error(`xim file not found: ${ximPath}`);
  }
  return ximPath;
}

/**
 * Load STXM scan: parse .hdr and load associated .xim ascii image.
 */
export function loadStxm(
  hdrPath: string,
  ximPath?: string,
): { meta: HdrMeta; image: number[][] } {
  const meta = readHdr(hdrPath);
  const resolvedXim = ximPath ?? resolveXimPath(hdrPath);
  const shape: [number, number] = [meta.qaxis_count, meta.paxis_count];
  const image = readXim(resolvedXim, shape);
  return { meta, image };
}

/**
 * Return true if the .hdr header contains Type = "NEXAFS Line Scan".
 */
export function isNexafsLineScanType(hdrPath: string): boolean {
  if (!hdrPath.toLowerCase().endsWith(".hdr") || !fs.existsSync(hdrPath)) {
    return false;
  }
  return fs.readFileSync(hdrPath, "utf8").includes(NEXAFS_LINE_SCAN_TYPE);
}

/**
 * Fast heuristic: header parses and .xim has sufficient byte size.
 */
export function isValidLineScanFast(hdrPath: string): boolean {
  try {
    const meta = readHdr(hdrPath);
    const n = meta.paxis_count * meta.qaxis_count;
    const ximPath = resolveXimPath(hdrPath);
    return fs.statSync(ximPath).size >= n * MIN_BYTES_PER_VALUE;
  } catch {
    return false;
  }
}

/**
 * Return true if loadStxm succeeds for the given header.
 */
export function isValidLineScan(hdrPath: string): boolean {
  try {
    loadStxm(hdrPath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Return true if header is NEXAFS Line Scan and loads with correct 2D shape.
 */
export function isNexafsLineScan(hdrPath: string): boolean {
  return isNexafsLineScanType(hdrPath) && isValidLineScan(hdrPath);
}

/**
 * Read the STXM Type field from a .hdr file.
 */
export function parseHdrScanType(hdrPath: string): string {
  if (!fs.existsSync(hdrPath)) {
    return "Unknown";
  }
  const match = HDR_TYPE_PATTERN.exec(fs.readFileSync(hdrPath, "utf8"));
  return match?.[1] ?? "Unknown";
}

/**
 * Map a header Type string to a stable UI grouping key.
 */
export function scanTypeCategory(scanType: string): ScanCategory {
  const lowered = scanType.toLowerCase();
  if (lowered.includes("nexafs line scan") || lowered.includes("line scan")) {
    return "line_scan";
  }
  if (lowered.includes("image scan")) {
    return "image_scan";
  }
  if (lowered.includes("focus scan")) {
    return "focus_scan";
  }
  if (lowered.includes("fixed point") || lowered.includes("fixed-point")) {
    return "fixed_point";
  }
  if (lowered.includes("stack")) {
    return "stack";
  }
  return "other";
}

/**
 * List .hdr files under an experiment directory tree, sorted by path.
 */
export function listExperimentHdrFiles(experimentPath: string): string[] {
  if (!fs.existsSync(experimentPath) || !fs.statSync(experimentPath).isDirectory()) {
    return [];
  }
  const paths: string[] = [];
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
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".hdr")) {
        try {
          const resolved = fs.realpathSync.native(full);
          if (fs.existsSync(resolved)) {
            paths.push(resolved);
          }
        } catch {
          continue;
        }
      }
    }
  };
  walk(experimentPath);
  return paths.sort();
}

/**
 * List NEXAFS line scans that pass isNexafsLineScan, sorted by name.
 */
export function listNexafsLineScans(experimentPath: string): string[] {
  if (!fs.existsSync(experimentPath) || !fs.statSync(experimentPath).isDirectory()) {
    return [];
  }
  return listExperimentHdrFiles(experimentPath)
    .filter((hdrPath) => isNexafsLineScan(hdrPath))
    .sort((a, b) => path.basename(a).localeCompare(path.basename(b)));
}
