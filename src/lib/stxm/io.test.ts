import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { catalogExperiment } from "@/lib/stxm/catalog";
import { listExperiments } from "@/lib/stxm/experiments";
import {
  extractScanEnergy,
  isNexafsLineScan,
  listExperimentHdrFiles,
  loadStxm,
  parseHdrScanType,
  readHdr,
  scanTypeCategory,
} from "@/lib/stxm/io";
import { loadScan } from "@/lib/stxm/load-scan";
import { regionRawSpectra, reduceScan } from "@/lib/stxm/reduce";
import { loadScanRegions, saveScanRegions } from "@/lib/stxm/region-store";
import { thumbnailPngBase64 } from "@/lib/stxm/thumbnail";
import { writeImageScanFixture, writeLineScanFixture } from "@/lib/stxm/test-fixtures";

let tempRoot = "";
let previousAllowedRoots: string | undefined;

beforeEach(() => {
  tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-io-test-"));
  previousAllowedRoots = process.env.STXM_ALLOWED_ROOTS;
  process.env.STXM_ALLOWED_ROOTS = tempRoot;
});

afterEach(() => {
  if (previousAllowedRoots === undefined) {
    delete process.env.STXM_ALLOWED_ROOTS;
  } else {
    process.env.STXM_ALLOWED_ROOTS = previousAllowedRoots;
  }
  fs.rmSync(tempRoot, { recursive: true, force: true });
});

describe("readHdr and energy metadata", () => {
  test("extracts energy range from energy axis", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "line");
    const meta = readHdr(hdrPath);
    expect(meta.energy_min_eV).toBe(280);
    expect(meta.energy_max_eV).toBe(284);
    expect(meta.num_energy_points).toBe(5);
    expect(meta.energy_eV).toBeNull();
  });

  test("extracts scalar energy from image scan", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeImageScanFixture(experiment, "img", 285.4);
    const meta = readHdr(hdrPath);
    expect(meta.energy_eV).toBeCloseTo(285.4);
    expect(meta.num_energy_points).toBe(1);
  });

  test("extractScanEnergy handles start/end range", () => {
    const meta = extractScanEnergy({
      raw: [
        'Type = "Stack"',
        "StartEnergy = 278.0",
        "EndEnergy = 310.0",
        "NumEnergy = 45",
        'PAxis = { Name = "X" Points = ( 2 , 0 1 ) }',
        'QAxis = { Name = "Y" Points = ( 2 , 0 1 ) }',
      ].join("\n"),
    });
    expect(meta.energy_min_eV).toBe(278);
    expect(meta.energy_max_eV).toBe(310);
    expect(meta.num_energy_points).toBe(45);
  });
});

describe("catalog and experiments", () => {
  test("lists experiment subdirectories", () => {
    const parent = path.join(tempRoot, "beamtime");
    fs.mkdirSync(path.join(parent, "2024-02(Feb)"), { recursive: true });
    fs.mkdirSync(path.join(parent, "2023-12(Dec)"), { recursive: true });
    const result = listExperiments(parent);
    expect(result.experiments[0]).toBe("2024-02(Feb)");
    expect(result.experiments[1]).toBe("2023-12(Dec)");
  });

  test("catalogExperiment builds entries with thumbnails", async () => {
    const experiment = path.join(tempRoot, "2024-02(Feb)");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "nexafs");
    writeLineScanFixture(experiment, "focus", "Focus Scan");
    const catalog = await catalogExperiment(experiment);
    expect(catalog.entries).toHaveLength(2);
    const lineEntry = catalog.entries.find((entry) => entry.basename === "nexafs.hdr");
    expect(lineEntry?.is_nexafs_line_scan).toBe(true);
    expect(lineEntry?.category).toBe("line_scan");
    expect(lineEntry?.thumbnail_png_base64?.length ?? 0).toBeGreaterThan(100);
  });

  test("catalogExperiment fast path skips thumbnails", async () => {
    const experiment = path.join(tempRoot, "2024-02(Feb)-fast");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "nexafs");
    writeImageScanFixture(experiment, "image", 285.4);
    const catalog = await catalogExperiment(experiment, { thumbnails: false });
    expect(catalog.entries).toHaveLength(2);
    for (const entry of catalog.entries) {
      expect(entry.thumbnail_png_base64).toBeUndefined();
      expect(entry.shape).not.toBeNull();
      expect(entry.basename.endsWith(".hdr")).toBe(true);
    }
    const lineEntry = catalog.entries.find((entry) => entry.basename === "nexafs.hdr");
    expect(lineEntry?.is_nexafs_line_scan).toBe(true);
    const imageEntry = catalog.entries.find((entry) => entry.basename === "image.hdr");
    expect(imageEntry?.energy_eV).toBeCloseTo(285.4);
  });

  test("listExperimentHdrFiles finds nested hdr files", () => {
    const experiment = path.join(tempRoot, "beamtime");
    const subdir = path.join(experiment, "2025_10(October)");
    fs.mkdirSync(subdir, { recursive: true });
    writeLineScanFixture(subdir, "532_260313061");
    const paths = listExperimentHdrFiles(experiment);
    expect(paths).toHaveLength(1);
    expect(paths[0]).toContain("532_260313061.hdr");
  });
});

describe("loadScan and reduceScan", () => {
  test("loadScan returns image preview and default regions", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    const payload = loadScan(hdrPath);
    expect(payload.shape).toEqual([8, 5]);
    expect(payload.regions.length).toBeGreaterThan(0);
    expect(payload.image.length).toBeGreaterThan(0);
  });

  test("reduceScan produces izero and sample region spectra", async () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    const loaded = loadScan(hdrPath);
    const region = loaded.regions[0];
    expect(region).toBeDefined();
    const result = await reduceScan({
      hdrPath,
      regions: [region!],
      izero: loaded.izero_bounds,
    });
    expect(result.spectra).toHaveLength(2);
    expect(result.spectra[0]?.spot_label).toBe("izero");
    expect(result.spectra[0]?.color).toBe("#2563eb");
    expect(result.spectra[0]?.signal).toHaveLength(5);
    expect(result.spectra[1]?.kind).toBe("od");
    expect(result.spectra[1]?.energy_eV).toHaveLength(5);
    expect(result.spectra[1]?.OD).toHaveLength(5);
    expect(result.spectra[1]?.signal).toHaveLength(5);
    expect(result.spectra[1]?.beta).toHaveLength(5);
  });

  test("reduceScan with formula adds CXRO mass absorption columns", async () => {
    const experiment = path.join(tempRoot, "exp-mass");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan-mass");
    const loaded = loadScan(hdrPath);
    const region = loaded.regions[0];
    expect(region).toBeDefined();
    const result = await reduceScan({
      hdrPath,
      regions: [region!],
      izero: loaded.izero_bounds,
      formula: "C",
      bareAtomFitOffset: true,
    });
    const sample = result.spectra.find((spectrum) => spectrum.spot_label !== "izero");
    expect(sample?.mass_absorption?.length).toBe(loaded.paxis_points.length);
    expect(sample?.mass_absorption?.every((value) => Number.isFinite(value))).toBe(true);
  });

  test("regionRawSpectra produces izero and per-region mean signal without OD", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    const loaded = loadScan(hdrPath);
    const result = regionRawSpectra({
      hdrPath,
      regions: loaded.regions,
      izero: loaded.izero_bounds,
    });
    expect(result.spectra.length).toBeGreaterThan(1);
    expect(result.spectra[0]?.spot_label).toBe("izero");
    expect(result.spectra[0]?.color).toBe("#2563eb");
    for (const spectrum of result.spectra) {
      expect(spectrum.kind).toBe("raw");
      expect(spectrum.signal?.length).toBe(5);
      expect(spectrum.OD).toBeUndefined();
      expect(spectrum.signal?.every((value) => Number.isFinite(value))).toBe(true);
    }
  });

  test("regionRawSpectra rejects regions that miss the q-axis", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    const loaded = loadScan(hdrPath);
    expect(() =>
      regionRawSpectra({
        hdrPath,
        regions: [{ sample_lo: 100, sample_hi: 200, spot_label: "pure" }],
        izero: loaded.izero_bounds,
      }),
    ).toThrow(/No sample regions overlap/);
  });
});

describe("region store", () => {
  test("save and load scan regions round-trip", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    saveScanRegions(experiment, hdrPath, {
      izero_lo: 0,
      izero_hi: 2,
      regions: [{ sample_lo: 3, sample_hi: 7, spot_label: "spot_a" }],
    });
    const saved = loadScanRegions(experiment, hdrPath);
    expect(saved?.regions[0]?.spot_label).toBe("spot_a");
  });
});

describe("scan type helpers", () => {
  test("parse_hdr_scan_type and is_nexafs_line_scan", () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const line = writeLineScanFixture(experiment, "line");
    const image = writeImageScanFixture(experiment, "image");
    expect(parseHdrScanType(line.hdrPath)).toBe("NEXAFS Line Scan");
    expect(scanTypeCategory("Image Scan")).toBe("image_scan");
    expect(isNexafsLineScan(line.hdrPath)).toBe(true);
    expect(isNexafsLineScan(image.hdrPath)).toBe(false);
  });

  test("thumbnailPngBase64 encodes png", async () => {
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    const { hdrPath } = writeLineScanFixture(experiment, "scan");
    const { image } = loadStxm(hdrPath);
    const encoded = await thumbnailPngBase64(image, 64);
    expect(encoded.length).toBeGreaterThan(100);
  });
});
