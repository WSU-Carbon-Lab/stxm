import type { IzeroBounds, SpectrumKind, SpectrumSeries, StxmRegion } from "@/lib/stxm-types";
import { regionMeanAndSigma, type WeightingMode } from "@/lib/stxm/estimators";
import { izeroSeriesColor, regionSeriesColor } from "@/lib/stxm/region-colors";
import { sampleIzeroMasks } from "@/lib/stxm/sample-masks";

/**
 * In-memory scan arrays used for client-side partial spectrum updates during region drags.
 */
export type InMemoryScanContext = {
  image: number[][];
  paxis: number[];
  qaxis: number[];
  izeroMask: boolean[];
};

function normalizeSpotLabel(raw: unknown): string {
  if (typeof raw === "string") {
    const text = raw.trim();
    return text || "pure";
  }
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return String(raw);
  }
  return "pure";
}

/**
 * Build an in-memory scan context from loaded image and axis arrays.
 *
 * Parameters
 * ----------
 * image : number[][]
 *     Detector counts with shape ``(n_rows, n_energy)``.
 * paxis : number[]
 *     Energy axis values in eV, length ``n_energy``.
 * qaxis : number[]
 *     Sample axis values, length ``n_rows``.
 * izero : IzeroBounds
 *     Current izero bar bounds on the sample axis.
 *
 * Returns
 * -------
 * InMemoryScanContext
 *     Context with a precomputed izero row mask.
 *
 * Raises
 * ------
 * Error
 *     When the izero mask selects no rows.
 */
export function buildInMemoryScanContext(
  image: number[][],
  paxis: number[],
  qaxis: number[],
  izero: IzeroBounds,
): InMemoryScanContext {
  const { izeroMask } = sampleIzeroMasks(qaxis, 0, 0, izero.izero_lo, izero.izero_hi);
  if (!izeroMask.some(Boolean)) {
    throw new Error("izero region selects no rows");
  }
  return { image, paxis, qaxis, izeroMask };
}

function buildIzeroSpectrum(
  ctx: InMemoryScanContext,
  izero: IzeroBounds,
  weightingMode: WeightingMode,
  kind: SpectrumKind,
): SpectrumSeries {
  const { mean, sigma } = regionMeanAndSigma(ctx.image, ctx.izeroMask, weightingMode);
  return {
    kind,
    spot_label: "izero",
    sample_lo: izero.izero_lo,
    sample_hi: izero.izero_hi,
    energy_eV: ctx.paxis,
    signal: mean,
    signal_err: sigma,
    color: izeroSeriesColor(),
  };
}

/**
 * Compute the izero raw mean signal spectrum from an in-memory scan.
 */
export function izeroRawSpectrum(
  ctx: InMemoryScanContext,
  izero: IzeroBounds,
  weightingMode: WeightingMode = "poisson_mle",
): SpectrumSeries {
  return buildIzeroSpectrum(ctx, izero, weightingMode, "raw");
}

/**
 * Compute one sample region's raw mean signal spectrum from an in-memory scan.
 *
 * Returns ``null`` when the region mask selects no rows.
 */
export function regionRawSpectrumSingle(
  ctx: InMemoryScanContext,
  region: StxmRegion,
  regionIndex: number,
  izero: IzeroBounds,
  weightingMode: WeightingMode = "poisson_mle",
): SpectrumSeries | null {
  const { sampleMask } = sampleIzeroMasks(
    ctx.qaxis,
    region.sample_lo,
    region.sample_hi,
    izero.izero_lo,
    izero.izero_hi,
  );
  if (!sampleMask.some(Boolean)) {
    return null;
  }
  const { mean, sigma } = regionMeanAndSigma(ctx.image, sampleMask, weightingMode);
  return {
    kind: "raw",
    spot_label: normalizeSpotLabel(region.spot_label),
    sample_lo: region.sample_lo,
    sample_hi: region.sample_hi,
    energy_eV: ctx.paxis,
    signal: mean,
    signal_err: sigma,
    color: regionSeriesColor(regionIndex),
  };
}

/**
 * Compute all raw spectra (izero plus sample regions) from an in-memory scan.
 *
 * Raises ``Error`` when no sample region overlaps the q-axis.
 */
/**
 * Compute all raw spectra from loaded scan arrays without calling the reduce API.
 *
 * Returns an empty list when the image has no rows or there are no sample regions.
 */
export function regionRawSpectraFromScanArrays(
  image: number[][],
  paxis: number[],
  qaxis: number[],
  regions: StxmRegion[],
  izero: IzeroBounds,
  weightingMode: WeightingMode = "poisson_mle",
): SpectrumSeries[] {
  if (!image.length || regions.length === 0) {
    return [];
  }
  const ctx = buildInMemoryScanContext(image, paxis, qaxis, izero);
  return regionRawSpectraFromContext(ctx, regions, izero, weightingMode);
}

export function regionRawSpectraFromContext(
  ctx: InMemoryScanContext,
  regions: StxmRegion[],
  izero: IzeroBounds,
  weightingMode: WeightingMode = "poisson_mle",
): SpectrumSeries[] {
  const spectra: SpectrumSeries[] = [izeroRawSpectrum(ctx, izero, weightingMode)];
  let regionIndex = 0;
  for (const region of regions) {
    const spectrum = regionRawSpectrumSingle(ctx, region, regionIndex, izero, weightingMode);
    if (spectrum) {
      spectra.push(spectrum);
      regionIndex += 1;
    }
  }
  if (spectra.length === 1) {
    throw new Error("No sample regions overlap the scan q-axis; adjust region bars.");
  }
  return spectra;
}

/**
 * Replace one spectrum series in a list by izero label or sample region index.
 *
 * Parameters
 * ----------
 * spectra : SpectrumSeries[]
 *     Existing spectra; index 0 is treated as izero when present.
 * target : { kind: "izero" } | { kind: "region"; index: number }
 *     Which trace to replace.
 * updated : SpectrumSeries
 *     New spectrum data for that trace.
 *
 * Returns
 * -------
 * SpectrumSeries[]
 *     Copy of ``spectra`` with the matching entry replaced, or appended when missing.
 */
export function mergeRawSpectrumUpdate(
  spectra: SpectrumSeries[],
  target: { kind: "izero" } | { kind: "region"; index: number },
  updated: SpectrumSeries,
): SpectrumSeries[] {
  if (target.kind === "izero") {
    const izeroIndex = spectra.findIndex((series) => series.spot_label === "izero");
    if (izeroIndex < 0) {
      return [updated, ...spectra];
    }
    const next = [...spectra];
    next[izeroIndex] = { ...updated, color: spectra[izeroIndex]?.color ?? updated.color };
    return next;
  }
  let regionCursor = 0;
  for (let index = 0; index < spectra.length; index += 1) {
    if (spectra[index]?.spot_label === "izero") {
      continue;
    }
    if (regionCursor === target.index) {
      const next = [...spectra];
      next[index] = { ...updated, color: spectra[index]?.color ?? updated.color };
      return next;
    }
    regionCursor += 1;
  }
  return [...spectra, updated];
}

export { buildIzeroSpectrum };
