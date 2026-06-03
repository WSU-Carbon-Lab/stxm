import type { IngestionYDisplayMode, SpectrumSeries } from "@/lib/stxm-types";

export type IngestionYModeOption = {
  value: IngestionYDisplayMode;
  label: string;
  shortLabel: string;
};

export const INGESTION_Y_MODE_OPTIONS: IngestionYModeOption[] = [
  { value: "signal", label: "Mean signal", shortLabel: "Signal" },
  { value: "signal_inverse", label: "Inverse mean signal (1/I)", shortLabel: "1/Signal" },
  { value: "od", label: "Optical density", shortLabel: "OD" },
  { value: "od_normalized", label: "Normalized OD", shortLabel: "Norm OD" },
  { value: "mass_absorption_cxro", label: "Norm. mass abs (CXRO)", shortLabel: "Mass abs" },
];

/** Minimum positive signal used as denominator for 1/I display. */
export const SIGNAL_INVERSE_MIN_DENOMINATOR = 1e-12;

/**
 * Whether the selected Y mode plots raw (or inverted) per-region mean signal without reduce.
 */
export function ingestionModeUsesRawSignal(mode: IngestionYDisplayMode): boolean {
  return mode === "signal" || mode === "signal_inverse";
}

/**
 * Whether log Y scale is valid for the current ingestion display mode (positive signal-like values).
 */
export function ingestionModeAllowsLogYScale(mode: IngestionYDisplayMode): boolean {
  return ingestionModeUsesRawSignal(mode);
}

/**
 * Whether the selected Y mode requires a full Beer-Lambert reduce (not raw signal only).
 */
export function ingestionModeNeedsReduce(mode: IngestionYDisplayMode): boolean {
  return !ingestionModeUsesRawSignal(mode);
}

/**
 * Whether the selected Y mode requires a chemical formula for CXRO bare-atom normalization.
 */
export function ingestionModeNeedsFormula(mode: IngestionYDisplayMode): boolean {
  return mode === "mass_absorption_cxro";
}

/**
 * Whether the ingestion spectrum chart includes the izero trace (raw mean I0 reference).
 *
 * OD-related modes use izero only as I0 during reduction; they plot sample regions only.
 */
export function ingestionModeShowsIzeroSpectrum(mode: IngestionYDisplayMode): boolean {
  return ingestionModeUsesRawSignal(mode);
}

/**
 * Filter reduced or raw spectra to those plotted for the selected ingestion Y mode.
 */
export function ingestionPlotSpectra(
  spectra: SpectrumSeries[],
  mode: IngestionYDisplayMode,
): SpectrumSeries[] {
  if (ingestionModeShowsIzeroSpectrum(mode)) {
    return spectra;
  }
  return spectra.filter((spectrum) => spectrum.spot_label !== "izero");
}

/**
 * Y-axis label for the ingestion spectrum chart.
 */
/**
 * Maps mean signal I to 1/I with a positive floor on I for display and log axes.
 */
export function ingestionSignalInverseValue(signal: number): number {
  if (!Number.isFinite(signal)) {
    return 0;
  }
  const denominator = Math.max(signal, SIGNAL_INVERSE_MIN_DENOMINATOR);
  return 1 / denominator;
}

/**
 * Propagates Gaussian uncertainty on I to 1/I via d(1/I)/dI = -1/I^2.
 */
export function ingestionSignalInverseErr(signal: number, signalErr: number): number | undefined {
  if (
    !Number.isFinite(signal) ||
    !Number.isFinite(signalErr) ||
    signalErr <= 0
  ) {
    return undefined;
  }
  const denominator = Math.max(signal, SIGNAL_INVERSE_MIN_DENOMINATOR);
  const invErr = signalErr / (denominator * denominator);
  return invErr > 0 && Number.isFinite(invErr) ? invErr : undefined;
}

export function ingestionYAxisLabel(mode: IngestionYDisplayMode): string {
  switch (mode) {
    case "signal":
      return "Mean signal";
    case "signal_inverse":
      return "1 / mean signal";
    case "od":
      return "OD (ln I0/I)";
    case "od_normalized":
      return "OD normalized";
    case "mass_absorption_cxro":
      return "Norm. mass abs (g/cm^2)";
    default:
      return "Mean signal";
  }
}

/**
 * Scalar spectrum value at one energy index for the ingestion plot.
 */
export function ingestionSpectrumValue(
  spectrum: SpectrumSeries,
  pointIndex: number,
  mode: IngestionYDisplayMode,
): number {
  if (mode === "signal_inverse") {
    return ingestionSignalInverseValue(spectrum.signal?.[pointIndex] ?? 0);
  }
  if (mode === "signal") {
    return spectrum.signal?.[pointIndex] ?? 0;
  }
  if (mode === "od") {
    return spectrum.OD?.[pointIndex] ?? 0;
  }
  if (mode === "od_normalized") {
    return spectrum.OD_normalized?.[pointIndex] ?? spectrum.OD?.[pointIndex] ?? 0;
  }
  if (mode === "mass_absorption_cxro") {
    const value = spectrum.mass_absorption?.[pointIndex];
    return value !== undefined && Number.isFinite(value) ? value : 0;
  }
  return spectrum.signal?.[pointIndex] ?? 0;
}

/**
 * Uncertainty at one energy index for error bars on the ingestion plot.
 */
export function ingestionSpectrumErr(
  spectrum: SpectrumSeries,
  pointIndex: number,
  mode: IngestionYDisplayMode,
): number | undefined {
  if (mode === "signal_inverse") {
    const signal = spectrum.signal?.[pointIndex] ?? 0;
    const err = spectrum.signal_err?.[pointIndex];
    if (err === undefined) {
      return undefined;
    }
    return ingestionSignalInverseErr(signal, err);
  }
  if (mode === "signal") {
    const err = spectrum.signal_err?.[pointIndex];
    return err !== undefined && Number.isFinite(err) && err > 0 ? err : undefined;
  }
  if (mode === "od" || mode === "od_normalized") {
    const err = spectrum.OD_err?.[pointIndex];
    return err !== undefined && Number.isFinite(err) && err > 0 ? err : undefined;
  }
  if (mode === "mass_absorption_cxro") {
    const err = spectrum.mass_absorption_err?.[pointIndex];
    return err !== undefined && Number.isFinite(err) && err > 0 ? err : undefined;
  }
  return undefined;
}

/**
 * Plotly tooltip value kind for one ingestion spectrum trace at the current Y mode.
 */
export function ingestionChartValueKind(
  mode: IngestionYDisplayMode,
): "signal" | "od" | "mass_absorption" {
  if (mode === "mass_absorption_cxro") {
    return "mass_absorption";
  }
  if (mode === "od" || mode === "od_normalized") {
    return "od";
  }
  return "signal";
}
