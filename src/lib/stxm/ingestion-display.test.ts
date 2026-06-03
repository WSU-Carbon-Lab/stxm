import { describe, expect, test } from "bun:test";

import {
  ingestionChartValueKind,
  ingestionModeAllowsLogYScale,
  ingestionModeNeedsReduce,
  ingestionModeShowsIzeroSpectrum,
  ingestionPlotSpectra,
  ingestionSignalInverseErr,
  ingestionSignalInverseValue,
  ingestionSpectrumErr,
  ingestionSpectrumValue,
  ingestionYAxisLabel,
  SIGNAL_INVERSE_MIN_DENOMINATOR,
} from "@/lib/stxm/ingestion-display";
import type { SpectrumSeries } from "@/lib/stxm-types";

const sampleSpectrum: SpectrumSeries = {
  spot_label: "A",
  sample_lo: 0,
  sample_hi: 1,
  energy_eV: [280, 290],
  signal: [100, 80],
  signal_err: [10, 8],
  OD: [0.5, 1.2],
  OD_err: [0.05, 0.06],
  OD_normalized: [0.0, 1.0],
  mass_absorption: [0.1, 0.4],
  mass_absorption_err: [0.01, 0.02],
};

const izeroSpectrum: SpectrumSeries = {
  spot_label: "izero",
  sample_lo: 2,
  sample_hi: 3,
  energy_eV: [280, 290],
  signal: [1000, 900],
  signal_err: [30, 25],
};

describe("ingestion display modes", () => {
  test("signal_inverse does not require reduce", () => {
    expect(ingestionModeNeedsReduce("signal_inverse")).toBe(false);
    expect(ingestionModeAllowsLogYScale("signal_inverse")).toBe(true);
    expect(ingestionModeAllowsLogYScale("od")).toBe(false);
  });

  test("izero trace shown only for raw signal modes", () => {
    expect(ingestionModeShowsIzeroSpectrum("signal")).toBe(true);
    expect(ingestionModeShowsIzeroSpectrum("signal_inverse")).toBe(true);
    expect(ingestionModeShowsIzeroSpectrum("od")).toBe(false);
    expect(ingestionModeShowsIzeroSpectrum("od_normalized")).toBe(false);
    expect(ingestionModeShowsIzeroSpectrum("mass_absorption_cxro")).toBe(false);
  });

  test("ingestionPlotSpectra drops izero for OD modes", () => {
    const spectra = [izeroSpectrum, sampleSpectrum];
    expect(ingestionPlotSpectra(spectra, "signal")).toHaveLength(2);
    expect(ingestionPlotSpectra(spectra, "od")).toEqual([sampleSpectrum]);
    expect(ingestionPlotSpectra(spectra, "mass_absorption_cxro")).toEqual([sampleSpectrum]);
  });

  test("inverts mean signal with positive floor", () => {
    expect(ingestionSignalInverseValue(100)).toBeCloseTo(0.01, 12);
    expect(ingestionSignalInverseValue(0)).toBeCloseTo(
      1 / SIGNAL_INVERSE_MIN_DENOMINATOR,
      6,
    );
    expect(ingestionYAxisLabel("signal_inverse")).toBe("1 / mean signal");
    expect(ingestionYAxisLabel("od")).toBe("OD (ln I0/I)");
    expect(ingestionYAxisLabel("mass_absorption_cxro")).toBe("Norm. mass abs (g/cm^2)");
  });

  test("propagates uncertainty for inverse signal", () => {
    const err = ingestionSignalInverseErr(100, 10);
    expect(err).toBeCloseTo(10 / 10000, 12);
    expect(
      ingestionSpectrumErr(sampleSpectrum, 0, "signal_inverse"),
    ).toBeCloseTo(0.001, 12);
  });

  test("spectrum value uses Beer-Lambert OD for sample regions", () => {
    expect(ingestionSpectrumValue(sampleSpectrum, 0, "signal")).toBe(100);
    expect(ingestionSpectrumValue(sampleSpectrum, 0, "signal_inverse")).toBeCloseTo(
      0.01,
      12,
    );
    expect(ingestionSpectrumValue(sampleSpectrum, 0, "od")).toBeCloseTo(0.5, 12);
    expect(ingestionSpectrumValue(sampleSpectrum, 1, "od_normalized")).toBeCloseTo(1.0, 12);
    expect(ingestionSpectrumValue(sampleSpectrum, 0, "mass_absorption_cxro")).toBeCloseTo(
      0.1,
      12,
    );
    expect(ingestionSpectrumValue(izeroSpectrum, 0, "signal")).toBe(1000);
    expect(ingestionSpectrumValue(izeroSpectrum, 0, "od")).toBe(0);
  });

  test("spectrum err follows display mode", () => {
    expect(ingestionSpectrumErr(sampleSpectrum, 0, "od")).toBeCloseTo(0.05, 12);
    expect(ingestionSpectrumErr(sampleSpectrum, 0, "mass_absorption_cxro")).toBeCloseTo(
      0.01,
      12,
    );
    expect(ingestionSpectrumErr(izeroSpectrum, 0, "od")).toBeUndefined();
  });

  test("chart value kind maps display modes", () => {
    expect(ingestionChartValueKind("signal")).toBe("signal");
    expect(ingestionChartValueKind("od")).toBe("od");
    expect(ingestionChartValueKind("od_normalized")).toBe("od");
    expect(ingestionChartValueKind("mass_absorption_cxro")).toBe("mass_absorption");
  });
});
