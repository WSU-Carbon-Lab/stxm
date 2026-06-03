import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, test } from "bun:test";

import { loadStxm } from "@/lib/stxm/io";
import {
  buildInMemoryScanContext,
  izeroRawSpectrum,
  mergeRawSpectrumUpdate,
  regionRawSpectraFromContext,
  regionRawSpectraFromScanArrays,
  regionRawSpectrumSingle,
} from "@/lib/stxm/raw-spectrum";
import { writeLineScanFixture } from "@/lib/stxm/test-fixtures";

describe("raw-spectrum client helpers", () => {
  let tempRoot = "";

  afterEach(() => {
    if (tempRoot) {
      fs.rmSync(tempRoot, { recursive: true, force: true });
      tempRoot = "";
    }
  });

  test("regionRawSpectraFromScanArrays matches regionRawSpectraFromContext", () => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-raw-scan-arrays-"));
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "scan");
    const hdrPath = path.join(experiment, "scan.hdr");
    const { meta, image } = loadStxm(hdrPath);
    const qaxis = meta.qaxis_points ?? [];
    const paxis = meta.paxis_points ?? [];
    const izero = { izero_lo: 0, izero_hi: 2 };
    const regions = [{ sample_lo: 3, sample_hi: 7, spot_label: "pure" }];
    const ctx = buildInMemoryScanContext(image, paxis, qaxis, izero);
    const fromArrays = regionRawSpectraFromScanArrays(image, paxis, qaxis, regions, izero);
    const fromContext = regionRawSpectraFromContext(ctx, regions, izero);
    expect(fromArrays.map((s) => s.signal)).toEqual(fromContext.map((s) => s.signal));
  });

  test("regionRawSpectrumSingle matches one entry from regionRawSpectraFromContext", () => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-raw-spectrum-"));
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "scan");
    const hdrPath = path.join(experiment, "scan.hdr");
    const { meta, image } = loadStxm(hdrPath);
    const qaxis = meta.qaxis_points ?? [];
    const paxis = meta.paxis_points ?? [];
    const izero = { izero_lo: 0, izero_hi: 2 };
    const regions = [{ sample_lo: 3, sample_hi: 7, spot_label: "pure" }];
    const ctx = buildInMemoryScanContext(image, paxis, qaxis, izero);
    const all = regionRawSpectraFromContext(ctx, regions, izero);
    const single = regionRawSpectrumSingle(ctx, regions[0]!, 0, izero);
    expect(single).not.toBeNull();
    expect(single?.spot_label).toBe("pure");
    expect(single?.signal).toEqual(all[1]?.signal);
    expect(single?.signal_err).toEqual(all[1]?.signal_err);
  });

  test("mergeRawSpectrumUpdate replaces only the targeted region trace", () => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-raw-spectrum-"));
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "scan");
    const hdrPath = path.join(experiment, "scan.hdr");
    const { meta, image } = loadStxm(hdrPath);
    const qaxis = meta.qaxis_points ?? [];
    const paxis = meta.paxis_points ?? [];
    const izero = { izero_lo: 0, izero_hi: 2 };
    const regions = [
      { sample_lo: 3, sample_hi: 5, spot_label: "a" },
      { sample_lo: 6, sample_hi: 7, spot_label: "b" },
    ];
    const ctx = buildInMemoryScanContext(image, paxis, qaxis, izero);
    const spectra = regionRawSpectraFromContext(ctx, regions, izero);
    const shifted = regionRawSpectrumSingle(
      ctx,
      { sample_lo: 3, sample_hi: 6, spot_label: "a" },
      0,
      izero,
    );
    expect(shifted).not.toBeNull();
    const merged = mergeRawSpectrumUpdate(spectra, { kind: "region", index: 0 }, shifted!);
    expect(merged[1]?.signal).toEqual(shifted?.signal);
    expect(merged[2]?.signal).toEqual(spectra[2]?.signal);
    expect(merged[0]?.signal).toEqual(spectra[0]?.signal);
  });

  test("izeroRawSpectrum updates independently of sample regions", () => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-raw-spectrum-"));
    const experiment = path.join(tempRoot, "exp");
    fs.mkdirSync(experiment);
    writeLineScanFixture(experiment, "scan");
    const hdrPath = path.join(experiment, "scan.hdr");
    const { meta, image } = loadStxm(hdrPath);
    const qaxis = meta.qaxis_points ?? [];
    const paxis = meta.paxis_points ?? [];
    const izero = { izero_lo: 0, izero_hi: 2 };
    const shiftedIzero = { izero_lo: 0, izero_hi: 3 };
    const regions = [{ sample_lo: 4, sample_hi: 7, spot_label: "pure" }];
    const ctx = buildInMemoryScanContext(image, paxis, qaxis, izero);
    const spectra = regionRawSpectraFromContext(ctx, regions, izero);
    const ctxShifted = buildInMemoryScanContext(image, paxis, qaxis, shiftedIzero);
    const updatedIzero = izeroRawSpectrum(ctxShifted, shiftedIzero);
    const merged = mergeRawSpectrumUpdate(spectra, { kind: "izero" }, updatedIzero);
    expect(merged[0]?.signal).toEqual(updatedIzero.signal);
    expect(merged[1]?.signal).toEqual(spectra[1]?.signal);
  });
});
