import type { ChartPoint, ChartSeries, ChartValueKind } from "@/components/spectrum-chart";
import type { PlotScaleMode } from "@/lib/stxm/plot-scale";
import type { IngestionYDisplayMode } from "@/lib/stxm-types";

export type ChartRow = {
  energy: number;
  [key: string]: number | [number, number] | undefined;
};

export type PlotAreaBounds = {
  left: number;
  width: number;
};

export type PlotAreaLike = PlotAreaBounds | { x: number; width: number };

export type YAxisScalePlan = {
  scale: number;
  exponent: number;
  titleSuffix: string;
  applyScale: boolean;
};

export type PlotLayout = PlotAreaBounds & {
  top: number;
  height: number;
};

export type LinearTickPlan = {
  major: number[];
  minor: number[];
  domain: [number, number];
  majorStep: number;
  minorDivisions: number;
};

export type PlotlyPublicationAxisTicks = {
  tickmode: "array";
  tickvals: number[];
  ticktext?: string[];
  ticks: "inside";
  ticklen: number;
  tickwidth: number;
  tickcolor: string;
  minor: {
    ticks: "inside";
    tickmode: "array";
    tickvals: number[];
    ticklen: number;
    tickwidth: number;
    tickcolor: string;
    showgrid: false;
  };
};

export type PlotlyPublicationLinearAxisTicks = {
  tickmode: "linear";
  dtick: number;
  tick0: number;
  tickformat?: string;
  ticks: "inside";
  ticklen: number;
  tickwidth: number;
  tickcolor: string;
  minor: {
    tickmode: "linear";
    dtick: number;
    ticks: "inside";
    ticklen: number;
    tickwidth: number;
    tickcolor: string;
    showgrid: false;
  };
};

const DEFAULT_DOMAIN_PAD = 0.02;

const SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹" as const;

/**
 * Render an integer exponent with Unicode superscript digits (e.g. 3 -> "³", -4 -> "⁻⁴").
 */
export function formatSuperscriptExponent(exponent: number): string {
  if (!Number.isFinite(exponent) || exponent === 0) {
    return "";
  }
  const sign = exponent < 0 ? "⁻" : "";
  const digits = String(Math.abs(exponent))
    .split("")
    .map((digit) => SUPERSCRIPT_DIGITS[Number(digit)] ?? digit)
    .join("");
  return `${sign}${digits}`;
}

/**
 * Build a publication-style ``×10ⁿ`` suffix for scaled Y axis titles; returns empty when ``exponent`` is 0.
 */
export function formatPowerOfTenSuffix(exponent: number): string {
  const superscript = formatSuperscriptExponent(exponent);
  return superscript ? ` ×10${superscript}` : "";
}

/**
 * Expand numeric extents by a fractional margin for axis domains.
 */
export function computePaddedDomain(
  min: number,
  max: number,
  padFraction = DEFAULT_DOMAIN_PAD,
): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  if (lo === hi) {
    const pad = Math.abs(lo) > 0 ? Math.abs(lo) * 0.05 : 1;
    return [lo - pad, hi + pad];
  }
  const span = hi - lo;
  const pad = span * padFraction;
  return [snapTick(lo - pad), snapTick(hi + pad)];
}

function niceNumber(range: number, round: boolean): number {
  if (!Number.isFinite(range) || range <= 0) {
    return 1;
  }
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / 10 ** exponent;
  let niceFraction: number;
  if (round) {
    if (fraction < 1.5) {
      niceFraction = 1;
    } else if (fraction < 3) {
      niceFraction = 2;
    } else if (fraction < 7) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  } else if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

/**
 * Plan publication-style major ticks and a padded axis domain for a linear scale.
 */
export function planLinearTicks(
  min: number,
  max: number,
  targetMajor = 6,
  padFraction = DEFAULT_DOMAIN_PAD,
  minorDivisions = 4,
): LinearTickPlan {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return {
      major: [0, 1],
      minor: [],
      domain: [0, 1],
      majorStep: 1,
      minorDivisions,
    };
  }
  const domain = computePaddedDomain(min, max, padFraction);
  const [lo, hi] = domain;
  const rawRange = hi - lo;
  const majorStep = niceNumber(rawRange / Math.max(2, targetMajor - 1), true);
  const tickStart = Math.ceil(lo / majorStep - 1e-9) * majorStep;
  const tickEnd = Math.floor(hi / majorStep + 1e-9) * majorStep;
  const major: number[] = [];
  for (let tick = tickStart; tick <= tickEnd + majorStep * 0.25; tick += majorStep) {
    if (tick >= lo - majorStep * 0.01 && tick <= hi + majorStep * 0.01) {
      major.push(snapTick(tick));
    }
  }
  if (major.length === 0) {
    major.push(snapTick(lo), snapTick(hi));
  }
  const minor = planMinorTicks(lo, hi, majorStep, major, minorDivisions);
  return { major, minor, domain, majorStep, minorDivisions };
}

function planMinorTicks(
  lo: number,
  hi: number,
  majorStep: number,
  major: number[],
  minorDivisions: number,
): number[] {
  const divisions = Math.max(2, Math.floor(minorDivisions));
  const minorStep = majorStep / divisions;
  const minorStart = Math.ceil(lo / minorStep - 1e-9) * minorStep;
  const minorEnd = Math.floor(hi / minorStep + 1e-9) * minorStep;
  const minor: number[] = [];
  for (let tick = minorStart; tick <= minorEnd + minorStep * 0.25; tick += minorStep) {
    if (tick < lo - minorStep * 0.01 || tick > hi + minorStep * 0.01) {
      continue;
    }
    const snapped = snapTick(tick);
    if (!isMajorTickValue(snapped, major, majorStep)) {
      minor.push(snapped);
    }
  }
  return minor;
}

function isMajorTickValue(value: number, major: number[], majorStep: number): boolean {
  const tolerance = Math.max(Math.abs(majorStep) * 1e-6, 1e-9);
  return major.some((tick) => Math.abs(tick - value) <= tolerance);
}

/**
 * Merge major and minor tick positions in ascending order for axis rendering.
 */
export function combineTickValues(plan: LinearTickPlan): number[] {
  return [...plan.major, ...plan.minor].sort((left, right) => left - right);
}

/**
 * Return whether ``value`` is a planned major tick (not a minor subdivision).
 */
export function isMajorTick(plan: LinearTickPlan, value: number): boolean {
  return isMajorTickValue(value, plan.major, plan.majorStep);
}

/**
 * Build Plotly axis tick settings with labeled major ticks, unlabeled minor ticks, and major-only grid.
 */
export function plotlyPublicationAxisTicks(
  plan: LinearTickPlan,
  options?: {
    formatMajorLabel?: (value: number) => string;
    axisStroke?: string;
    majorTickLen?: number;
    minorTickLen?: number;
  },
): PlotlyPublicationAxisTicks {
  const axisStroke = options?.axisStroke ?? "#27272a";
  const majorTickLen = options?.majorTickLen ?? 5;
  const minorTickLen = options?.minorTickLen ?? 3;
  const major =
    plan.major.length > 0 ? plan.major : combineTickValues({ ...plan, major: plan.domain, minor: [] });
  const minor = plan.minor;
  const layers: PlotlyPublicationAxisTicks = {
    tickmode: "array",
    tickvals: major,
    ticks: "inside",
    ticklen: majorTickLen,
    tickwidth: 1,
    tickcolor: axisStroke,
    minor: {
      ticks: "inside",
      tickmode: "array",
      tickvals: minor,
      ticklen: minorTickLen,
      tickwidth: 1,
      tickcolor: axisStroke,
      showgrid: false,
    },
  };
  if (options?.formatMajorLabel) {
    layers.ticktext = major.map((tick) => options.formatMajorLabel!(tick));
  }
  return layers;
}

/**
 * Return the first major tick position used as ``tick0`` for linear Plotly axes.
 */
export function linearAxisTickOrigin(plan: LinearTickPlan): number {
  if (plan.major.length > 0) {
    return plan.major[0]!;
  }
  const lo = plan.domain[0];
  const step = plan.majorStep;
  return snapTick(Math.ceil(lo / step - 1e-9) * step);
}

/**
 * Build Plotly linear tick settings that stay valid after pan and zoom.
 *
 * Major and minor steps follow ``planLinearTicks``; tick positions regenerate for
 * the visible axis range instead of a fixed ``tickvals`` array.
 */
export function plotlyPublicationLinearAxisTicks(
  plan: LinearTickPlan,
  options?: {
    tickformat?: string;
    axisStroke?: string;
    majorTickLen?: number;
    minorTickLen?: number;
  },
): PlotlyPublicationLinearAxisTicks {
  const axisStroke = options?.axisStroke ?? "#27272a";
  const majorTickLen = options?.majorTickLen ?? 5;
  const minorTickLen = options?.minorTickLen ?? 3;
  const minorStep = plan.majorStep / Math.max(2, plan.minorDivisions);
  return {
    tickmode: "linear",
    dtick: plan.majorStep,
    tick0: linearAxisTickOrigin(plan),
    tickformat: options?.tickformat,
    ticks: "inside",
    ticklen: majorTickLen,
    tickwidth: 1,
    tickcolor: axisStroke,
    minor: {
      tickmode: "linear",
      dtick: minorStep,
      ticks: "inside",
      ticklen: minorTickLen,
      tickwidth: 1,
      tickcolor: axisStroke,
      showgrid: false,
    },
  };
}

/**
 * Plotly ``tickformat`` for energy (eV) axes on spectrum charts.
 */
export function spectrumEnergyTickformat(): string {
  return ".1f";
}

/**
 * Plotly ``tickformat`` for Y axes given value kind and whether values are display-scaled.
 */
export function spectrumValueTickformat(
  valueKind: ChartValueKind,
  applyScale: boolean,
): string {
  if (applyScale) {
    return ".2f";
  }
  if (valueKind === "od" || valueKind === "mass_absorption") {
    return ".4f";
  }
  return ".4g";
}

/**
 * Merge spectrum series into chart rows with optional `[low, high]` band keys for uncertainty shading.
 */
export function mergeSeriesForChart(series: ChartSeries[]): ChartRow[] {
  const byEnergy = new Map<number, ChartRow>();
  for (const entry of series) {
    for (const point of entry.points) {
      const row = byEnergy.get(point.energy) ?? { energy: point.energy };
      row[entry.id] = point.value;
      if (point.err !== undefined && Number.isFinite(point.err) && point.err > 0) {
        row[`${entry.id}__err`] = point.err;
        row[`${entry.id}__band`] = [point.value - point.err, point.value + point.err];
      }
      byEnergy.set(point.energy, row);
    }
  }
  return Array.from(byEnergy.values()).sort(
    (left, right) => (left.energy ?? 0) - (right.energy ?? 0),
  );
}

/**
 * Collect Y extents from merged rows, including uncertainty band bounds when present.
 */
export function collectYExtents(rows: ChartRow[], series: ChartSeries[]): [number, number] {
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;
  for (const row of rows) {
    for (const entry of series) {
      const value = row[entry.id];
      if (typeof value === "number" && Number.isFinite(value)) {
        yMin = Math.min(yMin, value);
        yMax = Math.max(yMax, value);
      }
      const band = row[`${entry.id}__band`];
      if (Array.isArray(band)) {
        yMin = Math.min(yMin, band[0], band[1]);
        yMax = Math.max(yMax, band[0], band[1]);
      }
    }
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    return [0, 1];
  }
  if (yMin === yMax) {
    const pad = Math.abs(yMin) > 0 ? Math.abs(yMin) * 0.05 : 1;
    return [yMin - pad, yMax + pad];
  }
  return [yMin, yMax];
}

/**
 * Collect finite Y samples from merged rows, including uncertainty band bounds.
 */
export function collectYValues(rows: ChartRow[], series: ChartSeries[]): number[] {
  const values: number[] = [];
  for (const row of rows) {
    for (const entry of series) {
      const value = row[entry.id];
      if (typeof value === "number" && Number.isFinite(value)) {
        values.push(value);
      }
      const band = row[`${entry.id}__band`];
      if (Array.isArray(band)) {
        for (const bound of band) {
          if (Number.isFinite(bound)) {
            values.push(bound);
          }
        }
      }
    }
  }
  return values;
}

/**
 * Divide series values (and bands) by ``scale`` for display on a scaled Y axis.
 */
export function scaleChartRows(rows: ChartRow[], series: ChartSeries[], scale: number): ChartRow[] {
  if (scale === 1) {
    return rows;
  }
  return rows.map((row) => {
    const scaled: ChartRow = { energy: row.energy };
    for (const entry of series) {
      const value = row[entry.id];
      if (typeof value === "number" && Number.isFinite(value)) {
        scaled[entry.id] = value / scale;
      }
      const errKey = `${entry.id}__err`;
      const err = row[errKey];
      if (typeof err === "number" && Number.isFinite(err)) {
        scaled[errKey] = err / scale;
      }
      const band = row[`${entry.id}__band`];
      if (Array.isArray(band)) {
        scaled[`${entry.id}__band`] = [band[0] / scale, band[1] / scale];
      }
    }
    return scaled;
  });
}

/**
 * Pick a power-of-ten Y display scale for large mean-signal counts; OD modes stay unscaled.
 */
export function yAxisScaleFromValues(
  values: number[],
  valueKind: ChartValueKind = "signal",
): YAxisScalePlan {
  if (valueKind !== "signal") {
    return { scale: 1, exponent: 0, titleSuffix: "", applyScale: false };
  }
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return { scale: 1, exponent: 0, titleSuffix: "", applyScale: false };
  }
  const maxAbs = Math.max(...finite.map((value) => Math.abs(value)));
  if (maxAbs < 1000 && (maxAbs === 0 || maxAbs >= 0.01)) {
    return { scale: 1, exponent: 0, titleSuffix: "", applyScale: false };
  }
  const exponent = Math.floor(Math.log10(maxAbs));
  const scale = 10 ** exponent;
  const titleSuffix = formatPowerOfTenSuffix(exponent);
  return { scale, exponent, titleSuffix, applyScale: true };
}

/**
 * Build the Y axis title with an optional ``×10ⁿ`` unit suffix.
 */
export function formatYAxisTitle(baseLabel: string, titleSuffix: string): string {
  return titleSuffix ? `${baseLabel}${titleSuffix}` : baseLabel;
}

/**
 * Format scaled-axis tick labels as plain decimals (values already divided by the axis scale).
 */
export function formatScaledAxisTick(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  if (abs >= 100) {
    return trimTrailingZeros(value.toFixed(0));
  }
  if (abs >= 10) {
    return trimTrailingZeros(value.toFixed(1));
  }
  if (abs >= 1) {
    return trimTrailingZeros(value.toFixed(2));
  }
  return trimTrailingZeros(value.toFixed(3));
}

/**
 * Collect energy extents from merged rows.
 */
export function collectXExtents(rows: ChartRow[]): [number, number] {
  if (rows.length === 0) {
    return [0, 1];
  }
  const energies = rows.map((row) => row.energy).filter((value) => Number.isFinite(value));
  if (energies.length === 0) {
    return [0, 1];
  }
  const xMin = Math.min(...energies);
  const xMax = Math.max(...energies);
  if (xMin === xMax) {
    return [xMin - 0.5, xMax + 0.5];
  }
  return [xMin, xMax];
}

function snapTick(value: number): number {
  return Number.parseFloat(value.toPrecision(12));
}

/**
 * Normalize plot-area or margin-derived bounds into ``{ left, width }``.
 */
export function toPlotAreaBounds(
  plotArea: PlotAreaLike | null | undefined,
): PlotAreaBounds | undefined {
  if (!plotArea || !Number.isFinite(plotArea.width) || plotArea.width <= 0) {
    return undefined;
  }
  const left = "left" in plotArea ? plotArea.left : plotArea.x;
  if (!Number.isFinite(left)) {
    return undefined;
  }
  return { left, width: plotArea.width };
}

export type GridClientRect = {
  left: number;
  width: number;
};

/**
 * Parse a tooltip label as energy when it lies inside the axis domain.
 */
export function energyLabelInDomain(
  label: string | number | undefined,
  domain: [number, number],
): number | undefined {
  const numeric = Number(label);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  const lo = Math.min(domain[0], domain[1]);
  const hi = Math.max(domain[0], domain[1]);
  if (numeric < lo - 1e-6 || numeric > hi + 1e-6) {
    return undefined;
  }
  return numeric;
}

/**
 * Map a viewport ``clientX`` to energy using the chart host's measured plot area.
 */
export function energyFromPlotClientX(
  clientX: number | undefined,
  host: HTMLElement | null,
  domain: [number, number],
): number | undefined {
  if (!host || clientX === undefined || !Number.isFinite(clientX)) {
    return undefined;
  }
  const plotEl = host.querySelector<PlotlyHostElement>(".js-plotly-plot");
  const fromPlotly = energyFromPlotlyClientX(plotEl, clientX);
  if (fromPlotly !== undefined && Number.isFinite(fromPlotly)) {
    return fromPlotly;
  }
  const plot = measurePlotLayout(host);
  if (!plot || !plotEl) {
    return undefined;
  }
  const plotRect = plotEl.getBoundingClientRect();
  return energyFromGridClientX(
    clientX,
    { left: plotRect.left + plot.left, width: plot.width },
    domain,
  );
}

/**
 * Resolve tooltip hover energy, preferring live pointer mapping over snapped row labels.
 */
export function resolveTooltipEnergy(
  clientX: number | undefined,
  host: HTMLElement | null,
  domain: [number, number],
  labelFallback: string | number | undefined,
  plotAreaFallback?: PlotAreaLike | null,
  chartPointerX?: number,
): number {
  const fromClient = energyFromPlotClientX(clientX, host, domain);
  if (fromClient !== undefined && Number.isFinite(fromClient)) {
    return fromClient;
  }
  const fromChartPointer = energyFromChartPointer(chartPointerX, plotAreaFallback, domain);
  if (fromChartPointer !== undefined && Number.isFinite(fromChartPointer)) {
    return fromChartPointer;
  }
  const fromLabel = energyLabelInDomain(labelFallback, domain);
  if (fromLabel !== undefined) {
    return fromLabel;
  }
  return Math.min(domain[0], domain[1]);
}

/**
 * Map a viewport ``clientX`` to energy using a measured Cartesian grid rectangle.
 */
export function energyFromGridClientX(
  clientX: number | undefined,
  gridRect: GridClientRect | null | undefined,
  domain: [number, number],
): number | undefined {
  if (clientX === undefined || !Number.isFinite(clientX) || !gridRect) {
    return undefined;
  }
  const width = gridRect.width;
  if (!Number.isFinite(width) || width <= 0) {
    return undefined;
  }
  const lo = Math.min(domain[0], domain[1]);
  const hi = Math.max(domain[0], domain[1]);
  const span = hi - lo;
  if (!Number.isFinite(span) || span <= 0) {
    return undefined;
  }
  const t = Math.min(1, Math.max(0, (clientX - gridRect.left) / width));
  return lo + t * span;
}

/**
 * Map a chart-container X pixel to energy using plot-area bounds and the X domain.
 */
export function energyFromChartPointer(
  chartX: number | undefined,
  plotArea: PlotAreaLike | null | undefined,
  domain: [number, number],
): number | undefined {
  const bounds = toPlotAreaBounds(plotArea);
  if (chartX === undefined || !Number.isFinite(chartX) || !bounds) {
    return undefined;
  }
  const lo = Math.min(domain[0], domain[1]);
  const hi = Math.max(domain[0], domain[1]);
  const span = hi - lo;
  if (!Number.isFinite(span) || span <= 0) {
    return undefined;
  }
  const t = Math.min(1, Math.max(0, (chartX - bounds.left) / bounds.width));
  return lo + t * span;
}

/**
 * Map energy back to a chart-container X pixel inside the plot area.
 */
export function plotXFromEnergy(
  energy: number,
  plotArea: PlotAreaLike | null | undefined,
  domain: [number, number],
): number | undefined {
  const bounds = toPlotAreaBounds(plotArea);
  if (!Number.isFinite(energy) || !bounds) {
    return undefined;
  }
  const lo = Math.min(domain[0], domain[1]);
  const hi = Math.max(domain[0], domain[1]);
  const span = hi - lo;
  if (!Number.isFinite(span) || span <= 0) {
    return undefined;
  }
  const t = (energy - lo) / span;
  return bounds.left + Math.min(1, Math.max(0, t)) * bounds.width;
}

/**
 * Map a horizontal chart pixel coordinate to energy using the plot layout and X domain.
 */
export function energyFromPlotCoordinate(
  chartX: number | undefined,
  plotLeft: number,
  plotWidth: number,
  domain: [number, number],
): number | undefined {
  return energyFromChartPointer(chartX, { left: plotLeft, width: plotWidth }, domain);
}

export type PlotlyAxisRange = [number, number];

export type PlotlyPlotSize = {
  l: number;
  t: number;
  w: number;
  h: number;
};

export type PlotlyAxisType = "linear" | "log" | string;

export type PlotlyAxisLike = {
  type?: PlotlyAxisType;
  range?: PlotlyAxisRange | number[];
  l2p?: (value: number) => number;
  c2p?: (value: number, clip?: boolean) => number;
  _offset?: number;
  _length?: number;
};

export type PlotlyGraphLayout = {
  _size?: PlotlyPlotSize;
  xaxis?: PlotlyAxisLike;
  yaxis?: PlotlyAxisLike;
};

export type PlotlyHostElement = HTMLElement & {
  _fullLayout?: PlotlyGraphLayout;
};

/**
 * Parse a Plotly axis range tuple, returning undefined when values are not finite.
 */
export function readPlotlyAxisRange(range: unknown): PlotlyAxisRange | undefined {
  if (!Array.isArray(range) || range.length < 2) {
    return undefined;
  }
  const lo = Number(range[0]);
  const hi = Number(range[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return undefined;
  }
  return [lo, hi];
}

/**
 * Map a viewport ``clientX`` to energy using Plotly ``_fullLayout._size`` and live ``xaxis.range``.
 */
export function energyFromPlotlyClientX(
  graphDiv: HTMLElement | null,
  clientX: number | undefined,
): number | undefined {
  if (!graphDiv || clientX === undefined || !Number.isFinite(clientX)) {
    return undefined;
  }
  const layout = (graphDiv as PlotlyHostElement)._fullLayout;
  const size = layout?._size;
  const xRange = readPlotlyAxisRange(layout?.xaxis?.range);
  if (!size || !xRange || !Number.isFinite(size.w) || size.w <= 0) {
    return undefined;
  }
  const rect = graphDiv.getBoundingClientRect();
  const plotX = clientX - rect.left - size.l;
  const t = Math.min(1, Math.max(0, plotX / size.w));
  const lo = Math.min(xRange[0], xRange[1]);
  const hi = Math.max(xRange[0], xRange[1]);
  return lo + t * (hi - lo);
}

function plotlyAxisR2l(axis: PlotlyAxisLike, dataValue: number): number | undefined {
  if (!Number.isFinite(dataValue)) {
    return undefined;
  }
  if (axis.type === "log") {
    if (dataValue <= 0) {
      return undefined;
    }
    return Math.log10(dataValue);
  }
  return dataValue;
}

/**
 * Map a trace data coordinate to axis-local pixels via Plotly ``c2p`` or a test fallback.
 */
export function plotlyAxisDataToPixel(
  axis: PlotlyAxisLike | undefined,
  value: number,
  isYAxis: boolean,
): number | undefined {
  if (!axis || !Number.isFinite(value)) {
    return undefined;
  }
  if (axis.c2p) {
    const pixel = axis.c2p(value, false);
    return Number.isFinite(pixel) ? pixel : undefined;
  }
  const range = readPlotlyAxisRange(axis.range);
  const length = axis._length;
  if (!range || length === undefined || !Number.isFinite(length) || length <= 0) {
    return undefined;
  }
  const rl0 = plotlyAxisR2l(axis, range[0]);
  const rl1 = plotlyAxisR2l(axis, range[1]);
  const linearValue = plotlyAxisR2l(axis, value);
  if (
    rl0 === undefined ||
    rl1 === undefined ||
    linearValue === undefined ||
    rl0 === rl1
  ) {
    return undefined;
  }
  const slope = isYAxis ? length / (rl0 - rl1) : length / (rl1 - rl0);
  const intercept = isYAxis ? -slope * rl1 : -slope * rl0;
  const pixel = intercept + slope * linearValue;
  return Number.isFinite(pixel) ? pixel : undefined;
}

/**
 * Map trace ``(energy, value)`` to pixel coordinates inside the Plotly graph div.
 */
export function plotSvgPointFromPlotlyAxes(
  layout: PlotlyGraphLayout | undefined,
  energy: number,
  value: number,
): { x: number; y: number } | null {
  const size = layout?._size;
  const xaxis = layout?.xaxis;
  const yaxis = layout?.yaxis;
  if (!size || !xaxis || !yaxis) {
    return null;
  }
  const xPixel = plotlyAxisDataToPixel(xaxis, energy, false);
  const yPixel = plotlyAxisDataToPixel(yaxis, value, true);
  if (xPixel === undefined || yPixel === undefined) {
    return null;
  }
  return {
    x: (xaxis._offset ?? size.l) + xPixel,
    y: (yaxis._offset ?? size.t) + yPixel,
  };
}

/**
 * Map trace ``(energy, value)`` to host-relative overlay coordinates using Plotly axis transforms.
 */
export function plotHostPointFromPlotlyData(
  graphDiv: HTMLElement | null,
  host: HTMLElement | null,
  energy: number,
  value: number,
): { x: number; y: number } | null {
  if (!graphDiv || !host || !Number.isFinite(energy) || !Number.isFinite(value)) {
    return null;
  }
  const svgPoint = plotSvgPointFromPlotlyAxes(
    (graphDiv as PlotlyHostElement)._fullLayout,
    energy,
    value,
  );
  if (!svgPoint) {
    return null;
  }
  const plotRect = graphDiv.getBoundingClientRect();
  const hostRect = host.getBoundingClientRect();
  return {
    x: plotRect.left - hostRect.left + svgPoint.x,
    y: plotRect.top - hostRect.top + svgPoint.y,
  };
}

/**
 * Map energy to a host-relative X pixel using Plotly ``xaxis.c2p`` and layout offsets.
 */
export function plotHostXFromPlotlyEnergy(
  graphDiv: HTMLElement | null,
  host: HTMLElement | null,
  energy: number,
): number | undefined {
  if (!graphDiv || !host || !Number.isFinite(energy)) {
    return undefined;
  }
  const layout = (graphDiv as PlotlyHostElement)._fullLayout;
  const size = layout?._size;
  const xaxis = layout?.xaxis;
  if (!size || !xaxis) {
    return undefined;
  }
  const xPixel = plotlyAxisDataToPixel(xaxis, energy, false);
  if (xPixel === undefined) {
    return undefined;
  }
  const plotRect = graphDiv.getBoundingClientRect();
  const hostRect = host.getBoundingClientRect();
  return plotRect.left - hostRect.left + (xaxis._offset ?? size.l) + xPixel;
}

/**
 * Map a trace Y value to a host-relative pixel using Plotly ``yaxis.c2p``.
 */
export function plotHostYFromPlotlyValue(
  graphDiv: HTMLElement | null,
  host: HTMLElement | null,
  value: number,
): number | undefined {
  if (!graphDiv || !host || !Number.isFinite(value)) {
    return undefined;
  }
  const layout = (graphDiv as PlotlyHostElement)._fullLayout;
  const size = layout?._size;
  const yaxis = layout?.yaxis;
  if (!size || !yaxis) {
    return undefined;
  }
  const yPixel = plotlyAxisDataToPixel(yaxis, value, true);
  if (yPixel === undefined) {
    return undefined;
  }
  const plotRect = graphDiv.getBoundingClientRect();
  const hostRect = host.getBoundingClientRect();
  return plotRect.top - hostRect.top + (yaxis._offset ?? size.t) + yPixel;
}

/**
 * Return host-relative plot-area bounds derived from a Plotly graph div.
 */
export function plotHostAreaFromPlotly(
  graphDiv: HTMLElement | null,
  host: HTMLElement | null,
): PlotLayout | null {
  if (!graphDiv || !host) {
    return null;
  }
  const size = (graphDiv as PlotlyHostElement)._fullLayout?._size;
  if (!size || !Number.isFinite(size.w) || size.w <= 0 || !Number.isFinite(size.h) || size.h <= 0) {
    return null;
  }
  const plotRect = graphDiv.getBoundingClientRect();
  const hostRect = host.getBoundingClientRect();
  return {
    left: plotRect.left - hostRect.left + size.l,
    top: plotRect.top - hostRect.top + size.t,
    width: size.w,
    height: size.h,
  };
}

/**
 * Measure the Cartesian plot area inside a Plotly chart host element.
 */
export function measurePlotLayout(host: HTMLElement | null): PlotLayout | null {
  if (!host) {
    return null;
  }
  const plotEl = host.querySelector<PlotlyHostElement>(".js-plotly-plot");
  const size = plotEl?._fullLayout?._size;
  if (!size || !Number.isFinite(size.w) || size.w <= 0 || !Number.isFinite(size.h) || size.h <= 0) {
    return null;
  }
  return {
    left: size.l,
    top: size.t,
    width: size.w,
    height: size.h,
  };
}

/**
 * Linearly interpolate a spectrum series at an arbitrary energy.
 */
export function interpolateChartPoint(
  points: ChartPoint[],
  energy: number,
): { value: number; err?: number } | null {
  if (points.length === 0 || !Number.isFinite(energy)) {
    return null;
  }
  const sorted = [...points].sort((left, right) => left.energy - right.energy);
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  if (!first || !last) {
    return null;
  }
  if (energy <= first.energy) {
    return { value: first.value, err: first.err };
  }
  if (energy >= last.energy) {
    return { value: last.value, err: last.err };
  }
  let rightIndex = 1;
  while (rightIndex < sorted.length && sorted[rightIndex]!.energy < energy) {
    rightIndex += 1;
  }
  const left = sorted[rightIndex - 1]!;
  const right = sorted[rightIndex]!;
  const span = right.energy - left.energy;
  if (!Number.isFinite(span) || span <= 0) {
    return { value: left.value, err: left.err };
  }
  const t = (energy - left.energy) / span;
  const value = left.value + t * (right.value - left.value);
  let err: number | undefined;
  if (left.err !== undefined && right.err !== undefined) {
    err = left.err + t * (right.err - left.err);
  } else {
    err = left.err ?? right.err;
  }
  return { value, err };
}

/**
 * Format axis tick labels with ASCII scientific notation for large mean-signal counts.
 */
export function formatScientificAxisTick(
  value: number,
  valueKind: ChartValueKind = "signal",
): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  if (valueKind === "signal") {
    if (abs > 0 && abs < 0.01) {
      return formatAsciiScientific(value, 2);
    }
    if (abs >= 100) {
      return String(Math.round(value));
    }
    if (abs >= 10) {
      return trimTrailingZeros(value.toFixed(1));
    }
    if (abs >= 1) {
      return trimTrailingZeros(value.toFixed(2));
    }
    return trimTrailingZeros(value.toFixed(3));
  }
  if (valueKind === "od" || valueKind === "mass_absorption") {
    if (abs >= 10000 || (abs > 0 && abs < 0.0001)) {
      return formatAsciiScientific(value, 2);
    }
    if (abs < 0.1) {
      return trimTrailingZeros(value.toFixed(4));
    }
    if (abs < 10) {
      return trimTrailingZeros(value.toFixed(3));
    }
    return trimTrailingZeros(value.toFixed(2));
  }
  if (abs >= 10000 || (abs > 0 && abs < 0.001)) {
    return formatAsciiScientific(value, 2);
  }
  if (abs >= 100) {
    return String(Math.round(value));
  }
  if (abs >= 10) {
    return trimTrailingZeros(value.toFixed(1));
  }
  if (abs >= 1) {
    return trimTrailingZeros(value.toFixed(2));
  }
  return trimTrailingZeros(value.toFixed(3));
}

/**
 * Format a number as ASCII scientific notation (e.g. `1.40e+04`).
 */
export function formatAsciiScientific(value: number, fractionDigits = 2): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (value === 0) {
    return "0.00e+00";
  }
  const raw = value.toExponential(fractionDigits);
  const match = /^(.+)e([+-]?)(\d+)$/.exec(raw);
  if (!match) {
    return raw;
  }
  const mantissa = match[1] ?? "0";
  const sign = match[2] ?? "+";
  const digits = match[3] ?? "0";
  const normalizedSign = sign === "-" ? "-" : "+";
  const padded = digits.padStart(2, "0");
  return `${mantissa}e${normalizedSign}${padded}`;
}

function trimTrailingZeros(text: string): string {
  return text.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function roundToSignificantFigures(value: number, sigFigs: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (value === 0) {
    return "0";
  }
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  const scale = 10 ** (sigFigs - 1 - magnitude);
  const rounded = Math.round(value * scale) / scale;
  const decimals = Math.max(0, sigFigs - 1 - magnitude);
  return rounded.toFixed(decimals);
}

export type SpectrumTooltipFormatOptions = {
  yScale?: PlotScaleMode;
  yDisplayMode?: IngestionYDisplayMode;
};

function tooltipUsesScientificNotation(
  value: number,
  kind: ChartValueKind,
  options: SpectrumTooltipFormatOptions,
): boolean {
  if (options.yScale === "log") {
    return true;
  }
  if (!Number.isFinite(value)) {
    return false;
  }
  const abs = Math.abs(value);
  if (kind === "od" || kind === "mass_absorption") {
    return abs >= 10000 || (abs > 0 && abs < 0.0001);
  }
  return abs >= 10000 || (abs > 0 && abs < 0.01);
}

/**
 * Format a spectrum hover tooltip scalar for the active Y scale and value kind.
 */
export function formatTooltipValue(
  value: number,
  kind: ChartValueKind = "signal",
  options: SpectrumTooltipFormatOptions = {},
): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (tooltipUsesScientificNotation(value, kind, options)) {
    return formatAsciiScientific(value, 2);
  }
  const abs = Math.abs(value);
  if (kind === "od" || kind === "mass_absorption") {
    if (abs < 1e-6) {
      return value.toExponential(2);
    }
    if (abs < 0.1) {
      return trimTrailingZeros(value.toFixed(4));
    }
    if (abs < 10) {
      return trimTrailingZeros(value.toFixed(3));
    }
    return trimTrailingZeros(value.toFixed(2));
  }
  if (abs >= 10000) {
    return formatScientificAxisTick(value, "signal");
  }
  return roundToSignificantFigures(value, 4);
}

/**
 * Format a spectrum hover tooltip uncertainty for the active Y scale and value kind.
 */
export function formatTooltipErr(
  err: number,
  kind: ChartValueKind,
  options: SpectrumTooltipFormatOptions = {},
): string {
  if (!Number.isFinite(err)) {
    return "";
  }
  if (tooltipUsesScientificNotation(err, kind, options)) {
    const fractionDigits = Math.abs(err) >= 1 ? 1 : 2;
    return formatAsciiScientific(err, fractionDigits);
  }
  const abs = Math.abs(err);
  if (kind === "od" || kind === "mass_absorption") {
    if (abs < 0.001) {
      return err.toExponential(1);
    }
    if (abs < 0.1) {
      return trimTrailingZeros(err.toFixed(4));
    }
    return trimTrailingZeros(err.toFixed(3));
  }
  if (abs >= 100) {
    return String(Math.round(err));
  }
  return roundToSignificantFigures(err, 2);
}

/**
 * Format a spectrum hover tooltip value with optional ``±`` uncertainty.
 */
export function formatTooltipValueWithErr(
  value: number,
  err: number | undefined,
  kind: ChartValueKind = "signal",
  options: SpectrumTooltipFormatOptions = {},
): string {
  const valStr = formatTooltipValue(value, kind, options);
  if (err === undefined || !Number.isFinite(err) || err <= 0) {
    return valStr;
  }
  return `${valStr} ± ${formatTooltipErr(err, kind, options)}`;
}

/**
 * Pick the dominant Y-axis value kind when series use mixed kinds.
 */
export function dominantValueKind(series: ChartSeries[]): ChartValueKind {
  if (series.some((entry) => entry.valueKind === "signal")) {
    return "signal";
  }
  if (series.some((entry) => entry.valueKind === "od")) {
    return "od";
  }
  if (series.some((entry) => entry.valueKind === "mass_absorption")) {
    return "mass_absorption";
  }
  return "signal";
}

