import "server-only";

import type { IzeroBounds, SpectrumSeries, StxmRegion } from "@/lib/stxm-types";
import { type WeightingMode } from "@/lib/stxm/estimators";
import { loadStxm } from "@/lib/stxm/io";
import { nexafsBeerLambert } from "@/lib/stxm/nexafs";
import {
  normalizedMassAbsorptionAtEnergies,
  odErrToBetaErr,
  odToBeta,
} from "@/lib/stxm/absorption";
import { massAbsorptionCm2PerG } from "@/lib/stxm/cxro.server";
import { normalizeNexafsWithMetadata, type NormalizationMode } from "@/lib/stxm/normalization";
import { requireAllowedFile } from "@/lib/stxm/path-utils";
import {
  buildInMemoryScanContext,
  buildIzeroSpectrum,
  regionRawSpectraFromContext,
  regionRawSpectrumSingle,
} from "@/lib/stxm/raw-spectrum";
import { regionSeriesColor } from "@/lib/stxm/region-colors";
import { normalizeSpotLabel } from "@/lib/stxm/region-store";
import { sampleIzeroMasks } from "@/lib/stxm/sample-masks";

export type RegionSpectraInput = {
  hdrPath: string;
  regions: StxmRegion[];
  izero: IzeroBounds;
  weightingMode?: WeightingMode;
};

export type ReduceScanInput = RegionSpectraInput & {
  normalizationMode?: NormalizationMode;
  preEdge?: [number, number];
  postEdge?: [number, number];
  /** Chemical formula for CXRO bare-atom mass absorption (e.g. C8H8). */
  formula?: string;
  /** When true, include constant offset in bare-atom step-edge fit (first/last five points). */
  bareAtomFitOffset?: boolean;
};

const DEFAULT_THICKNESS_CM = 1e-4;

type LoadedScanContext = {
  hdrPath: string;
  image: number[][];
  paxis: number[];
  qaxis: number[];
  izeroMask: boolean[];
};

function loadScanContext(hdrPath: string, izero: IzeroBounds): LoadedScanContext {
  const resolved = requireAllowedFile(hdrPath);
  const { meta, image } = loadStxm(resolved);
  const qaxis = meta.qaxis_points ?? [];
  const paxis = meta.paxis_points ?? [];
  const memory = buildInMemoryScanContext(image, paxis, qaxis, izero);
  return { hdrPath: resolved, image, paxis, qaxis, izeroMask: memory.izeroMask };
}

function parseEdgeRange(spec: string | undefined, fallback: [number, number]): [number, number] {
  if (!spec) {
    return fallback;
  }
  const parts = spec.split(",").map((part) => Number.parseFloat(part.trim()));
  if (parts.length !== 2 || parts.some((value) => Number.isNaN(value))) {
    return fallback;
  }
  return [parts[0]!, parts[1]!];
}

/**
 * Compute per-region raw averaged detector signal vs energy without OD normalization.
 */
export function regionRawSpectra(
  input: RegionSpectraInput,
): { hdr_path: string; spectra: SpectrumSeries[] } {
  const ctx = loadScanContext(input.hdrPath, input.izero);
  const weightingMode = input.weightingMode ?? "poisson_mle";
  const memory = buildInMemoryScanContext(ctx.image, ctx.paxis, ctx.qaxis, input.izero);
  const spectra = regionRawSpectraFromContext(memory, input.regions, input.izero, weightingMode);
  return { hdr_path: ctx.hdrPath, spectra };
}

export { regionRawSpectrumSingle };

/**
 * Reduce a line scan to per-region NEXAFS spectra with OD, normalization, and optional CXRO mass absorption.
 */
export async function reduceScan(
  input: ReduceScanInput,
): Promise<{ hdr_path: string; spectra: SpectrumSeries[] }> {
  const ctx = loadScanContext(input.hdrPath, input.izero);
  const weightingMode = input.weightingMode ?? "poisson_mle";
  const normalizationMode = input.normalizationMode ?? "pre_edge_scale";
  const [preLo, preHi] = input.preEdge ?? [280, 283];
  const [postLo, postHi] = input.postEdge ?? [292, 310];
  const izLo = input.izero.izero_lo;
  const izHi = input.izero.izero_hi;
  const formula = input.formula?.trim() ?? "";
  let muRhoBare: number[] | null = null;
  if (formula) {
    muRhoBare = await massAbsorptionCm2PerG(formula, ctx.paxis);
  }
  const spectra: SpectrumSeries[] = [
    buildIzeroSpectrum(ctx, input.izero, weightingMode, "raw"),
  ];
  let regionIndex = 0;
  for (const reg of input.regions) {
    const { sampleMask } = sampleIzeroMasks(
      ctx.qaxis,
      reg.sample_lo,
      reg.sample_hi,
      izLo,
      izHi,
    );
    if (!sampleMask.some(Boolean)) {
      continue;
    }
    const { od, sigmaOd, intensity, sigmaI } = nexafsBeerLambert(
      ctx.image,
      sampleMask,
      ctx.izeroMask,
      1e-10,
      weightingMode,
    );
    const { normalized: odNorm } = normalizeNexafsWithMetadata(
      ctx.paxis,
      od,
      preLo,
      preHi,
      postLo,
      postHi,
      normalizationMode,
    );
    const beta = odToBeta(ctx.paxis, od, DEFAULT_THICKNESS_CM);
    const betaErr = odErrToBetaErr(ctx.paxis, sigmaOd, DEFAULT_THICKNESS_CM);
    let massAbsorption: number[] | undefined;
    let massAbsorptionErr: number[] | undefined;
    if (muRhoBare?.length === ctx.paxis.length) {
      const derived = normalizedMassAbsorptionAtEnergies(
        ctx.paxis,
        od,
        sigmaOd,
        muRhoBare,
        { fitOffset: input.bareAtomFitOffset !== false },
      );
      massAbsorption = derived.values;
      massAbsorptionErr = derived.errors;
    }
    spectra.push({
      kind: "od",
      spot_label: normalizeSpotLabel(reg.spot_label),
      sample_lo: reg.sample_lo,
      sample_hi: reg.sample_hi,
      energy_eV: ctx.paxis,
      signal: intensity,
      signal_err: sigmaI,
      OD: od,
      OD_err: sigmaOd,
      OD_normalized: odNorm,
      mass_absorption: massAbsorption,
      mass_absorption_err: massAbsorptionErr,
      beta,
      beta_err: betaErr,
      color: regionSeriesColor(regionIndex),
    });
    regionIndex += 1;
  }
  return { hdr_path: ctx.hdrPath, spectra };
}

export { parseEdgeRange };
