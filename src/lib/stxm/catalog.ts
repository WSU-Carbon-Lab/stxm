import "server-only";

import path from "node:path";

import type { ExperimentCatalogPayload, ScanCatalogEntry } from "@/lib/stxm-types";
import {
  isNexafsLineScan,
  isNexafsLineScanType,
  isValidLineScanFast,
  listExperimentHdrFiles,
  loadStxm,
  parseHdrScanType,
  readHdr,
  scanTypeCategory,
} from "@/lib/stxm/io";
import { requireAllowedDirectory } from "@/lib/stxm/path-utils";
import { thumbnailPngBase64 } from "@/lib/stxm/thumbnail";

export type CatalogOptions = {
  thumbnailSize?: number;
  thumbnails?: boolean;
};

function applyEnergyMetadata(
  record: ScanCatalogEntry,
  meta: ReturnType<typeof readHdr>,
): void {
  record.paxis_count = meta.paxis_count;
  record.qaxis_count = meta.qaxis_count;
  for (const key of ["energy_eV", "energy_min_eV", "energy_max_eV", "num_energy_points"] as const) {
    const value = meta[key];
    if (value !== null && value !== undefined) {
      record[key] = Number(value);
    }
  }
}

function buildCatalogEntryFast(hdrPath: string): ScanCatalogEntry {
  const scanType = parseHdrScanType(hdrPath);
  const resolvedHdr = path.resolve(hdrPath);
  const record: ScanCatalogEntry = {
    basename: path.basename(resolvedHdr),
    hdr_path: resolvedHdr,
    scan_type: scanType,
    category: scanTypeCategory(scanType),
    is_nexafs_line_scan:
      isNexafsLineScanType(resolvedHdr) && isValidLineScanFast(resolvedHdr),
    shape: null,
  };
  try {
    const meta = readHdr(resolvedHdr);
    record.shape = [meta.qaxis_count, meta.paxis_count];
    applyEnergyMetadata(record, meta);
  } catch {
    record.shape = null;
  }
  return record;
}

async function buildCatalogEntryWithThumbnail(
  hdrPath: string,
  thumbnailSize: number,
): Promise<ScanCatalogEntry> {
  const scanType = parseHdrScanType(hdrPath);
  const resolvedHdr = path.resolve(hdrPath);
  const record: ScanCatalogEntry = {
    basename: path.basename(resolvedHdr),
    hdr_path: resolvedHdr,
    scan_type: scanType,
    category: scanTypeCategory(scanType),
    is_nexafs_line_scan: isNexafsLineScan(resolvedHdr),
    shape: null,
  };
  try {
    const { meta, image } = loadStxm(resolvedHdr);
    record.shape = [image.length, image[0]?.length ?? 0];
    applyEnergyMetadata(record, meta);
    const thumb = await thumbnailPngBase64(image, thumbnailSize);
    if (thumb) {
      record.thumbnail_png_base64 = thumb;
    }
  } catch {
    record.shape = null;
  }
  return record;
}

/**
 * Build a Finder-style scan catalog for an experiment directory.
 *
 * When ``thumbnails`` is false, entries are built from header metadata only so
 * large experiments become interactive quickly; callers can follow with a full
 * pass to attach preview images.
 */
export async function catalogExperiment(
  experimentDir: string,
  options: CatalogOptions = {},
): Promise<ExperimentCatalogPayload> {
  const { thumbnailSize = 128, thumbnails = true } = options;
  const experiment = requireAllowedDirectory(experimentDir);
  const hdrPaths = listExperimentHdrFiles(experiment);
  const entries: ScanCatalogEntry[] = [];

  if (!thumbnails) {
    for (const hdrPath of hdrPaths) {
      entries.push(buildCatalogEntryFast(hdrPath));
    }
    return { experiment_dir: experiment, entries };
  }

  for (const hdrPath of hdrPaths) {
    entries.push(await buildCatalogEntryWithThumbnail(hdrPath, thumbnailSize));
  }
  return { experiment_dir: experiment, entries };
}
