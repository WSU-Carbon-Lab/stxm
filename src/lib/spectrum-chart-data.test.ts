import { describe, expect, test } from "bun:test";

import type { ChartPoint } from "@/components/spectrum-chart";
import {
  combineTickValues,
  computePaddedDomain,
  isMajorTick,
  planLinearTicks,
  linearAxisTickOrigin,
  plotlyPublicationAxisTicks,
  plotlyPublicationLinearAxisTicks,
  spectrumEnergyTickformat,
  spectrumValueTickformat,
  energyFromChartPointer,
  energyFromGridClientX,
  energyFromPlotCoordinate,
  energyFromPlotlyClientX,
  energyLabelInDomain,
  plotHostPointFromPlotlyData,
  plotHostXFromPlotlyEnergy,
  plotHostYFromPlotlyValue,
  plotlyAxisDataToPixel,
  plotSvgPointFromPlotlyAxes,
  readPlotlyAxisRange,
  resolveTooltipEnergy,
  formatAsciiScientific,
  formatTooltipValueWithErr,
  formatPowerOfTenSuffix,
  formatScaledAxisTick,
  formatScientificAxisTick,
  formatSuperscriptExponent,
  formatYAxisTitle,
  interpolateChartPoint,
  plotXFromEnergy,
  toPlotAreaBounds,
  yAxisScaleFromValues,
  type PlotlyAxisLike,
  type PlotlyGraphLayout,
} from "@/lib/spectrum-chart-data";

describe("computePaddedDomain", () => {
  test("pads data extents without large nice-number expansion", () => {
    const domain = computePaddedDomain(280, 390, 0.02);
    expect(domain[0]).toBeGreaterThan(277);
    expect(domain[0]).toBeLessThan(280);
    expect(domain[1]).toBeGreaterThan(390);
    expect(domain[1]).toBeLessThan(393);
  });
});

describe("planLinearTicks", () => {
  test("plans round major ticks for typical NEXAFS energy span", () => {
    const plan = planLinearTicks(280, 390);
    expect(plan.major).toEqual([280, 300, 320, 340, 360, 380]);
    expect(plan.domain[0]).toBeLessThan(280);
    expect(plan.domain[1]).toBeGreaterThan(390);
  });

  test("plans even-step major ticks for small OD-like spans", () => {
    const plan = planLinearTicks(-0.05, 5.8, 5);
    expect(plan.major).toEqual([0, 2, 4]);
  });

  test("places minors every majorStep/4 between labeled majors", () => {
    const plan = planLinearTicks(280, 390);
    expect(plan.majorStep).toBe(20);
    expect(plan.minorDivisions).toBe(4);
    expect(plan.minor).toContain(285);
    expect(plan.minor).toContain(295);
    expect(isMajorTick(plan, 280)).toBe(true);
    expect(isMajorTick(plan, 285)).toBe(false);
    expect(combineTickValues(plan).length).toBe(plan.major.length + plan.minor.length);
  });
});

describe("plotlyPublicationLinearAxisTicks", () => {
  test("uses linear dtick and minor subdivisions for pan-safe ticks", () => {
    const plan = planLinearTicks(280, 390);
    const ticks = plotlyPublicationLinearAxisTicks(plan, {
      tickformat: spectrumEnergyTickformat(),
    });
    expect(ticks.tickmode).toBe("linear");
    expect(ticks.dtick).toBe(20);
    expect(ticks.tick0).toBe(linearAxisTickOrigin(plan));
    expect(ticks.tickformat).toBe(".1f");
    expect(ticks.ticks).toBe("inside");
    expect(ticks.minor.tickmode).toBe("linear");
    expect(ticks.minor.dtick).toBe(5);
    expect(ticks.minor.ticks).toBe("inside");
  });

  test("picks value tickformat from kind and scale", () => {
    expect(spectrumValueTickformat("od", false)).toBe(".4f");
    expect(spectrumValueTickformat("signal", true)).toBe(".2f");
    expect(spectrumValueTickformat("signal", false)).toBe(".4g");
  });
});

describe("plotlyPublicationAxisTicks", () => {
  test("maps majors to labels and minors to unlabeled inside ticks", () => {
    const plan = planLinearTicks(280, 390);
    const labeled = plotlyPublicationAxisTicks(plan, {
      formatMajorLabel: (value) => String(value),
    });
    expect(labeled.tickvals).toEqual(plan.major);
    expect(labeled.ticktext).toEqual(["280", "300", "320", "340", "360", "380"]);
    expect(labeled.ticks).toBe("inside");
    expect(labeled.ticklen).toBe(5);
    expect(labeled.minor.tickvals).toContain(285);
    expect(labeled.minor.ticks).toBe("inside");
    expect(labeled.minor.ticklen).toBe(3);
    expect(labeled.minor.showgrid).toBe(false);

    const mirror = plotlyPublicationAxisTicks(plan);
    expect(mirror.ticktext).toBeUndefined();
    expect(mirror.tickvals).toEqual(plan.major);
    expect(mirror.minor.tickvals).toEqual(plan.minor);
    expect(mirror.ticks).toBe("inside");
  });
});

describe("toPlotAreaBounds", () => {
  test("accepts left-width and legacy x-width shapes", () => {
    expect(toPlotAreaBounds({ left: 56, width: 400 })).toEqual({ left: 56, width: 400 });
    expect(toPlotAreaBounds({ x: 72, width: 380 })).toEqual({ left: 72, width: 380 });
  });
});

describe("energyFromGridClientX", () => {
  test("maps viewport clientX through grid rect", () => {
    const domain: [number, number] = [280, 390];
    const gridRect = { left: 100, width: 500 };
    expect(energyFromGridClientX(350, gridRect, domain)).toBeCloseTo(335, 0);
    expect(energyFromGridClientX(100, gridRect, domain)).toBeCloseTo(280, 6);
    expect(energyFromGridClientX(600, gridRect, domain)).toBeCloseTo(390, 6);
  });
});

function mockPlotlyLinearAxis(
  range: [number, number],
  length: number,
  offset: number,
  isY: boolean,
): PlotlyAxisLike {
  const rl0 = range[0];
  const rl1 = range[1];
  const slope = isY ? length / (rl0 - rl1) : length / (rl1 - rl0);
  const intercept = isY ? -slope * rl1 : -slope * rl0;
  const toPixel = (value: number) => intercept + slope * value;
  return {
    type: "linear",
    range,
    _offset: offset,
    _length: length,
    l2p: toPixel,
    c2p: toPixel,
  };
}

function mockPlotlyLogYAxis(range: [number, number], length: number, offset: number): PlotlyAxisLike {
  const rl0 = Math.log10(range[0]);
  const rl1 = Math.log10(range[1]);
  const slope = length / (rl0 - rl1);
  const intercept = -slope * rl1;
  return {
    type: "log",
    range,
    _offset: offset,
    _length: length,
    l2p: (linearValue: number) => intercept + slope * linearValue,
    c2p: (value: number) => {
      if (value <= 0) {
        return Number.NaN;
      }
      return intercept + slope * Math.log10(value);
    },
  };
}

function mockPlotlyGraphLayout(
  xRange: [number, number],
  yAxis: PlotlyAxisLike,
  size = { l: 72, t: 36, w: 400, h: 240 },
) {
  return {
    _fullLayout: {
      _size: size,
      xaxis: mockPlotlyLinearAxis(xRange, size.w, size.l, false),
      yaxis: yAxis,
    },
    getBoundingClientRect: () =>
      ({
        left: 100,
        top: 50,
        width: 520,
        height: 320,
        right: 620,
        bottom: 370,
        x: 100,
        y: 50,
        toJSON: () => ({}),
      }) as DOMRect,
  } as unknown as HTMLElement;
}

const mockHost = {
  getBoundingClientRect: () =>
    ({
      left: 100,
      top: 50,
      width: 520,
      height: 320,
      right: 620,
      bottom: 370,
      x: 100,
      y: 50,
      toJSON: () => ({}),
    }) as DOMRect,
} as unknown as HTMLElement;

describe("energyFromPlotlyClientX", () => {
  test("maps viewport clientX through Plotly plot area and live xaxis range", () => {
    const graphDiv = mockPlotlyGraphLayout([280, 390], mockPlotlyLinearAxis([0, 1], 240, 36, true));
    expect(energyFromPlotlyClientX(graphDiv, 100 + 72 + 200)).toBeCloseTo(335, 6);
    expect(energyFromPlotlyClientX(graphDiv, 100 + 72)).toBeCloseTo(280, 6);
    expect(energyFromPlotlyClientX(graphDiv, 100 + 72 + 400)).toBeCloseTo(390, 6);
  });

  test("uses zoomed xaxis range instead of initial domain", () => {
    const graphDiv = mockPlotlyGraphLayout([300, 360], mockPlotlyLinearAxis([0, 1], 240, 36, true));
    expect(energyFromPlotlyClientX(graphDiv, 100 + 72 + 200)).toBeCloseTo(330, 6);
  });
});

describe("plotlyAxisDataToPixel", () => {
  test("uses axis c2p when provided", () => {
    const axis: PlotlyAxisLike = {
      c2p: (value) => value * 2,
      _length: 100,
    };
    expect(plotlyAxisDataToPixel(axis, 25, false)).toBe(50);
  });

  test("falls back to Plotly-like linear mapping from range and length", () => {
    const axis = mockPlotlyLinearAxis([0, 10], 200, 0, true);
    expect(plotlyAxisDataToPixel(axis, 5, true)).toBeCloseTo(100, 6);
  });

  test("falls back to Plotly-like log mapping from range and length", () => {
    const axis = mockPlotlyLogYAxis([0.001, 0.01], 240, 0);
    const izero = plotlyAxisDataToPixel(axis, 0.001209, true)!;
    const pure = plotlyAxisDataToPixel(axis, 0.002018, true)!;
    expect(pure).toBeLessThan(izero);
    expect(plotlyAxisDataToPixel(axis, 0.001794, true)).toBeFinite();
  });
});

describe("plotSvgPointFromPlotlyAxes", () => {
  test("maps energy and trace value through axis offsets", () => {
    const layout: PlotlyGraphLayout = {
      _size: { l: 72, t: 36, w: 400, h: 240 },
      xaxis: mockPlotlyLinearAxis([280, 390], 400, 72, false),
      yaxis: mockPlotlyLinearAxis([0, 10], 240, 36, true),
    };
    const point = plotSvgPointFromPlotlyAxes(layout, 335, 5)!;
    expect(point.x).toBeCloseTo(72 + 200, 6);
    expect(point.y).toBeCloseTo(36 + 120, 6);
  });
});

describe("plotHostPointFromPlotlyData", () => {
  test("round-trips X with energyFromPlotlyClientX", () => {
    const graphDiv = mockPlotlyGraphLayout([280, 390], mockPlotlyLinearAxis([0, 10], 240, 36, true));
    const energy = 335;
    const point = plotHostPointFromPlotlyData(graphDiv, mockHost, energy, 5)!;
    expect(energyFromPlotlyClientX(graphDiv, 100 + point.x)).toBeCloseTo(energy, 6);
  });

  test("places log-scale signal markers mid-plot, not at the top", () => {
    const graphDiv = mockPlotlyGraphLayout(
      [280, 390],
      mockPlotlyLogYAxis([10, 10_000], 240, 36),
    );
    const point = plotHostPointFromPlotlyData(graphDiv, mockHost, 319, 716)!;
    expect(point.y).toBeGreaterThan(100);
    expect(point.y).toBeLessThan(220);
  });

  test("maps small reciprocal signal values on log axes", () => {
    const graphDiv = mockPlotlyGraphLayout(
      [280, 390],
      mockPlotlyLogYAxis([0.001, 0.01], 240, 36),
    );
    const point = plotHostPointFromPlotlyData(graphDiv, mockHost, 319, 0.001794)!;
    expect(point.y).toBeGreaterThan(100);
    expect(point.y).toBeLessThan(260);
  });
});

describe("plotHostXFromPlotlyEnergy", () => {
  test("matches plotHostPointFromPlotlyData X coordinate", () => {
    const graphDiv = mockPlotlyGraphLayout([280, 390], mockPlotlyLinearAxis([0, 10], 240, 36, true));
    const energy = 335;
    const hostX = plotHostXFromPlotlyEnergy(graphDiv, mockHost, energy)!;
    const pointX = plotHostPointFromPlotlyData(graphDiv, mockHost, energy, 5)!.x;
    expect(hostX).toBeCloseTo(pointX, 6);
  });
});

describe("plotHostYFromPlotlyValue", () => {
  test("matches plotHostPointFromPlotlyData Y coordinate on log axes", () => {
    const graphDiv = mockPlotlyGraphLayout(
      [280, 390],
      mockPlotlyLogYAxis([0.001, 0.01], 240, 36),
    );
    const hostY = plotHostYFromPlotlyValue(graphDiv, mockHost, 0.001794)!;
    const pointY = plotHostPointFromPlotlyData(graphDiv, mockHost, 319, 0.001794)!.y;
    expect(hostY).toBeCloseTo(pointY, 6);
  });
});

describe("readPlotlyAxisRange", () => {
  test("accepts numeric tuples and rejects invalid ranges", () => {
    expect(readPlotlyAxisRange([280, 390])).toEqual([280, 390]);
    expect(readPlotlyAxisRange([Number.NaN, 390])).toBeUndefined();
    expect(readPlotlyAxisRange(null)).toBeUndefined();
  });
});

describe("energyFromPlotCoordinate", () => {
  test("maps plot pixels to energy across domain", () => {
    const domain: [number, number] = [280, 390];
    const left = 56;
    const width = 400;
    const mid = energyFromPlotCoordinate(left + width / 2, left, width, domain);
    expect(mid).toBeCloseTo(335, 0);
    const end = energyFromPlotCoordinate(left + width, left, width, domain);
    expect(end).toBeCloseTo(390, 6);
  });
});

describe("energyLabelInDomain", () => {
  test("accepts labels inside domain and rejects out-of-range values", () => {
    const domain: [number, number] = [280, 390];
    expect(energyLabelInDomain(320, domain)).toBe(320);
    expect(energyLabelInDomain(200, domain)).toBeUndefined();
    expect(energyLabelInDomain("invalid", domain)).toBeUndefined();
  });
});

describe("resolveTooltipEnergy", () => {
  test("prefers chart pointer over snapped row label", () => {
    const domain: [number, number] = [280, 390];
    const plotArea = { left: 56, width: 400 };
    const energy = resolveTooltipEnergy(undefined, null, domain, 287, plotArea, 256);
    expect(energy).toBeCloseTo(335, 2);
  });

  test("falls back to in-domain label when pointer mapping is unavailable", () => {
    const domain: [number, number] = [280, 390];
    expect(resolveTooltipEnergy(undefined, null, domain, 305)).toBe(305);
  });
});

describe("energyFromChartPointer", () => {
  test("maps chart pointer X through measured plot area", () => {
    const domain: [number, number] = [280, 390];
    const plotArea = { x: 64, width: 512 };
    const energy = energyFromChartPointer(320, plotArea, domain);
    expect(energy).toBeCloseTo(335, 2);
  });

  test("round-trips with plotXFromEnergy", () => {
    const domain: [number, number] = [280, 390];
    const plotArea = { left: 56, width: 400 };
    const energy = 335;
    const x = plotXFromEnergy(energy, plotArea, domain);
    expect(x).toBeCloseTo(256, 6);
    expect(energyFromChartPointer(x, plotArea, domain)).toBeCloseTo(energy, 6);
  });
});

describe("interpolateChartPoint", () => {
  test("linearly interpolates between bracketing energies", () => {
    const points: ChartPoint[] = [
      { energy: 280, value: 0 },
      { energy: 290, value: 10 },
    ];
    const sample = interpolateChartPoint(points, 285);
    expect(sample?.value).toBeCloseTo(5, 6);
  });
});

describe("formatAsciiScientific", () => {
  test("uses signed two-digit exponent", () => {
    expect(formatAsciiScientific(14000, 2)).toBe("1.40e+04");
    expect(formatAsciiScientific(0.0012, 2)).toBe("1.20e-03");
  });
});

describe("formatTooltipValueWithErr", () => {
  test("uses scientific notation on log Y scale for small signal values", () => {
    expect(
      formatTooltipValueWithErr(0.001794, 0.000017, "signal", { yScale: "log" }),
    ).toBe("1.79e-03 ± 1.70e-05");
  });

  test("uses scientific notation on log Y scale for large signal counts", () => {
    expect(
      formatTooltipValueWithErr(717.4, 6.1, "signal", { yScale: "log" }),
    ).toBe("7.17e+02 ± 6.1e+00");
  });

  test("keeps linear decimal formatting for mid-range OD", () => {
    expect(
      formatTooltipValueWithErr(0.4523, 0.012, "od", { yScale: "linear" }),
    ).toBe("0.452 ± 0.012");
  });

  test("uses scientific notation on linear scale for very small signal", () => {
    expect(
      formatTooltipValueWithErr(0.00042, 0.000003, "signal", { yScale: "linear" }),
    ).toBe("4.20e-04 ± 3.00e-06");
  });
});

describe("formatScientificAxisTick", () => {
  test("formats large mean signal counts as plain integers when unscaled", () => {
    expect(formatScientificAxisTick(14250, "signal")).toBe("14250");
  });
});

describe("formatSuperscriptExponent", () => {
  test("maps digits to Unicode superscripts", () => {
    expect(formatSuperscriptExponent(3)).toBe("³");
    expect(formatSuperscriptExponent(12)).toBe("¹²");
    expect(formatSuperscriptExponent(-4)).toBe("⁻⁴");
  });

  test("returns empty for zero", () => {
    expect(formatSuperscriptExponent(0)).toBe("");
  });
});

describe("formatPowerOfTenSuffix", () => {
  test("builds multiplication and superscript exponent", () => {
    expect(formatPowerOfTenSuffix(3)).toBe(" ×10³");
    expect(formatPowerOfTenSuffix(0)).toBe("");
  });
});

describe("formatYAxisTitle", () => {
  test("appends scaled suffix to base label", () => {
    expect(formatYAxisTitle("Mean signal", formatPowerOfTenSuffix(3))).toBe("Mean signal ×10³");
  });
});

describe("yAxisScaleFromValues", () => {
  test("uses superscript title suffix for large signal counts", () => {
    const plan = yAxisScaleFromValues([14_250, 18_000], "signal");
    expect(plan.applyScale).toBe(true);
    expect(plan.exponent).toBe(4);
    expect(plan.titleSuffix).toBe(" ×10⁴");
    expect(plan.scale).toBe(10_000);
  });

  test("leaves OD spectra unscaled", () => {
    const plan = yAxisScaleFromValues([0.2, 1.5], "od");
    expect(plan.applyScale).toBe(false);
    expect(plan.titleSuffix).toBe("");
  });
});

describe("formatScaledAxisTick", () => {
  test("formats display-scaled tick values as decimals", () => {
    expect(formatScaledAxisTick(1.425)).toBe("1.43");
    expect(formatScaledAxisTick(142)).toBe("142");
  });
});
