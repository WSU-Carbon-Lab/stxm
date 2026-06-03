/**
 * Shared intensity scaling helpers for STXM map and profile canvases.
 *
 * Line-scan heatmaps use ``paxis_points`` horizontally (energy columns) and
 * ``qaxis_points`` vertically (sample rows, row 0 at the top). High raw counts
 * (izero) map to bright tones; lower film counts map darker under both linear
 * and log display scaling.
 */

/** Colormap scaling mode for raw STXM intensity maps and compatible traces. */
export type PlotScaleMode = "linear" | "log";

/** Resolved display limits for a line-scan heatmap canvas. */
export type LineScanDisplayScale = {
  mode: PlotScaleMode;
  vmin: number;
  vmax: number;
  logFloor: number;
};

/**
 * Linearly interpolate a percentile from a sorted finite sample.
 *
 * Parameters
 * ----------
 * sorted : number[]
 *     Ascending finite values.
 * p : number
 *     Percentile in ``[0, 100]``.
 *
 * Returns
 * -------
 * number
 *     Interpolated percentile, or ``NaN`` when ``sorted`` is empty.
 */
export function percentileSorted(sorted: number[], p: number): number {
  if (sorted.length === 0) {
    return Number.NaN;
  }
  if (sorted.length === 1) {
    return sorted[0] ?? Number.NaN;
  }
  const fraction = p / 100;
  const index = fraction * (sorted.length - 1);
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) {
    return sorted[lo] ?? Number.NaN;
  }
  const weight = index - lo;
  return (sorted[lo] ?? 0) * (1 - weight) + (sorted[hi] ?? 0) * weight;
}

/**
 * Compute robust display limits from image pixel intensities.
 *
 * Parameters
 * ----------
 * values : number[] | number[][]
 *     Flat or 2D scan intensities; non-finite entries are ignored.
 * fallbackMin, fallbackMax : number
 *     Limits returned when no finite data exist or limits are degenerate.
 * pLow, pHigh : number, optional
 *     Lower and upper percentiles in ``[0, 100]`` (default 5 and 95).
 *
 * Returns
 * -------
 * tuple of (number, number)
 *     ``(vmin, vmax)`` for linear grayscale mapping.
 */
export function percentileLimits(
  values: number[] | number[][],
  fallbackMin: number,
  fallbackMax: number,
  pLow = 5,
  pHigh = 95,
): [number, number] {
  const flat = (Array.isArray(values[0])
    ? (values as number[][]).flat()
    : (values as number[])
  ).filter((value) => Number.isFinite(value));
  if (flat.length === 0) {
    return [fallbackMin, fallbackMax];
  }
  const sorted = [...flat].sort((a, b) => a - b);
  const vmin = percentileSorted(sorted, pLow);
  const vmax = percentileSorted(sorted, pHigh);
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin >= vmax) {
    return [fallbackMin, fallbackMax];
  }
  return [vmin, vmax];
}

function lineScanFinitePixels(image: number[][]): number[] {
  let finite = image.flat().filter((value) => Number.isFinite(value));
  const positive = finite.filter((value) => value > 0);
  if (positive.length >= Math.max(16, Math.floor(finite.length / 10))) {
    finite = positive;
  }
  return finite;
}

/**
 * Minimum positive count used as the log floor for line-scan log display scaling.
 */
export function lineScanLogFloor(positiveValues: number[]): number {
  const positive = positiveValues.filter((value) => value > 0);
  if (positive.length === 0) {
    return 1;
  }
  const minPositive = Math.min(...positive);
  if (!Number.isFinite(minPositive) || minPositive <= 0) {
    return 1;
  }
  return Math.max(1, minPositive * 1e-3);
}

/**
 * Map a raw detector count to the display domain used for heatmap normalization.
 *
 * Linear mode returns the raw count. Log mode returns ``log10(max(value, logFloor))``
 * so multiplicative izero-versus-film contrast becomes additive in display space.
 */
export function lineScanDisplayValue(
  value: number,
  mode: PlotScaleMode,
  logFloor = 1,
): number {
  if (!Number.isFinite(value)) {
    return Number.NaN;
  }
  if (mode === "linear") {
    return value;
  }
  const clamped = value > 0 ? Math.max(value, logFloor) : logFloor;
  return Math.log10(clamped);
}

/**
 * Compute grayscale display limits for a raw STXM line-scan intensity map.
 *
 * Matches ``apply_line_scan_image_clim`` in linear mode: 5th/95th percentiles on
 * positive finite pixels when enough exist. Log mode applies the same percentile
 * window to ``log10(count)`` values so izero rows appear brighter than film.
 */
export function lineScanImageDisplayLimits(
  image: number[][],
  fallbackMin: number,
  fallbackMax: number,
  pLow = 5,
  pHigh = 95,
  scaleMode: PlotScaleMode = "linear",
): [number, number] {
  const scale = lineScanImageDisplayScale(
    image,
    fallbackMin,
    fallbackMax,
    scaleMode,
    pLow,
    pHigh,
  );
  return [scale.vmin, scale.vmax];
}

/**
 * Resolve heatmap display limits and log floor for a line-scan intensity map.
 */
export function lineScanImageDisplayScale(
  image: number[][],
  fallbackMin: number,
  fallbackMax: number,
  scaleMode: PlotScaleMode = "linear",
  pLow = 5,
  pHigh = 95,
): LineScanDisplayScale {
  const finite = lineScanFinitePixels(image);
  const positive = finite.filter((value) => value > 0);
  const logFloor = lineScanLogFloor(positive);
  if (finite.length === 0) {
    const fallbackVmin =
      scaleMode === "log" ? lineScanDisplayValue(Math.max(fallbackMin, logFloor), "log", logFloor) : fallbackMin;
    const fallbackVmax =
      scaleMode === "log" ? lineScanDisplayValue(Math.max(fallbackMax, logFloor), "log", logFloor) : fallbackMax;
    return { mode: scaleMode, vmin: fallbackVmin, vmax: fallbackVmax, logFloor };
  }
  const displayValues = finite.map((value) => lineScanDisplayValue(value, scaleMode, logFloor));
  const sorted = [...displayValues].sort((a, b) => a - b);
  let vmin = percentileSorted(sorted, pLow);
  let vmax = percentileSorted(sorted, pHigh);
  if (!Number.isFinite(vmin)) {
    vmin = sorted[0] ?? fallbackMin;
  }
  if (!Number.isFinite(vmax)) {
    vmax = sorted[sorted.length - 1] ?? fallbackMax;
  }
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) {
    return {
      mode: scaleMode,
      vmin: scaleMode === "log" ? lineScanDisplayValue(fallbackMin, "log", logFloor) : fallbackMin,
      vmax: scaleMode === "log" ? lineScanDisplayValue(fallbackMax, "log", logFloor) : fallbackMax,
      logFloor,
    };
  }
  if (vmax <= vmin) {
    vmax = vmin + (scaleMode === "log" ? 0.25 : Math.max(Math.abs(vmin) * 1e-6, 1));
  }
  return { mode: scaleMode, vmin, vmax, logFloor };
}

/**
 * Map a raw pixel count to an 8-bit grayscale level for line-scan heatmaps.
 */
export function lineScanPixelGray(
  value: number,
  scale: LineScanDisplayScale,
): number {
  const displayValue = lineScanDisplayValue(value, scale.mode, scale.logFloor);
  const unit = normalizeToUnit(displayValue, scale.vmin, scale.vmax);
  return Math.round(unit * 255);
}

/**
 * Map a qaxis sample coordinate to vertical canvas pixels (origin upper).
 *
 * Row 0 of the scan image aligns with ``qaxisPoints[0]`` at the top edge,
 * matching Matplotlib ``imshow(..., origin="upper")`` with extent
 * ``[paxis[0], paxis[-1], qaxis[-1], qaxis[0]]``.
 *
 * Parameters
 * ----------
 * value : number
 *     Sample-axis coordinate in the same units as ``qaxisPoints``.
 * qaxisPoints : number[]
 *     Per-row qaxis coordinates, length equal to image row count when available.
 * height : number
 *     Canvas height in pixels.
 *
 * Returns
 * -------
 * number
 *     Vertical pixel coordinate with 0 at the top edge.
 */
export function qAxisValueToPx(value: number, qaxisPoints: number[], height: number): number {
  const qTop = qaxisPoints[0] ?? 0;
  const qBottom = qaxisPoints[qaxisPoints.length - 1] ?? 1;
  const qSpan = qTop - qBottom;
  if (!Number.isFinite(qSpan) || Math.abs(qSpan) < 1e-12) {
    return height / 2;
  }
  return ((qTop - value) / qSpan) * height;
}

/**
 * Convert a vertical canvas pixel to a qaxis sample coordinate.
 *
 * Inverse of ``qAxisValueToPx`` for drag hit-testing on region boundary lines.
 *
 * Parameters
 * ----------
 * clientY : number
 *     Pointer client Y in viewport coordinates.
 * canvasTop : number
 *     Canvas bounding rect top in viewport coordinates.
 * canvasHeight : number
 *     Canvas height in CSS pixels.
 * qaxisPoints : number[]
 *     Per-row qaxis coordinates from the scan payload.
 *
 * Returns
 * -------
 * number
 *     Sample-axis coordinate at ``clientY``.
 */
export function pxToQAxisValue(
  clientY: number,
  canvasTop: number,
  canvasHeight: number,
  qaxisPoints: number[],
): number {
  const qTop = qaxisPoints[0] ?? 0;
  const qBottom = qaxisPoints[qaxisPoints.length - 1] ?? 1;
  const qSpan = qTop - qBottom;
  const ratio = canvasHeight > 0 ? (clientY - canvasTop) / canvasHeight : 0;
  return qTop - ratio * qSpan;
}

/**
 * Return inclusive qaxis bounds for clamping region and izero lines.
 *
 * Parameters
 * ----------
 * qaxisPoints : number[]
 *     Per-row qaxis coordinates from the scan payload.
 *
 * Returns
 * -------
 * tuple of (number, number)
 *     ``(min, max)`` sample bounds regardless of axis point order.
 */
export function qAxisBounds(qaxisPoints: number[]): [number, number] {
  if (qaxisPoints.length === 0) {
    return [0, 1];
  }
  const lo = Math.min(...qaxisPoints);
  const hi = Math.max(...qaxisPoints);
  return lo <= hi ? [lo, hi] : [hi, lo];
}

export function computeRowSums(image: number[][]): number[] {
  return image.map((row) => {
    let sum = 0;
    for (const value of row) {
      sum += value;
    }
    return sum;
  });
}

export function normalizeToUnit(value: number, vmin: number, vmax: number): number {
  const span = vmax - vmin;
  if (!Number.isFinite(span) || span <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, (value - vmin) / span));
}

/**
 * Computes linear display limits for a row-sum profile trace.
 *
 * Uses the finite row-sum minimum and maximum with symmetric fractional padding
 * so the profile spans the trace width without arbitrary floors such as zero.
 *
 * Parameters
 * ----------
 * rowSums : number[]
 *     Per-row summed intensities.
 * marginFraction : number, optional
 *     Padding added below and above the data range as a fraction of span (default 0.05).
 *
 * Returns
 * -------
 * tuple of (number, number)
 *     ``(vmin, vmax)`` display limits for linear scaling.
 */
export function rowSumTraceLimits(
  rowSums: number[],
  marginFraction = 0.05,
): [number, number] {
  const finite = rowSums.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return [0, 1];
  }
  const dataMin = Math.min(...finite);
  const dataMax = Math.max(...finite);
  if (!Number.isFinite(dataMin) || !Number.isFinite(dataMax)) {
    return [0, 1];
  }
  const span = dataMax - dataMin;
  if (!Number.isFinite(span) || span <= 0) {
    const pad = Math.max(Math.abs(dataMin) * marginFraction, 1);
    return [dataMin - pad, dataMax + pad];
  }
  const margin = span * marginFraction;
  return [dataMin - margin, dataMax + margin];
}

/**
 * Maps a row-sum intensity to horizontal canvas coordinates for the profile trace.
 *
 * High values map toward ``plotLeft`` (adjacent to the heatmap); low values map
 * toward ``plotRight``. Values are clamped to the unit interval before mapping.
 *
 * Parameters
 * ----------
 * value : number
 *     Row-sum intensity to map.
 * vmin : number
 *     Lower display limit (maps to ``plotRight``).
 * vmax : number
 *     Upper display limit (maps toward ``plotLeft``).
 * plotLeft : number
 *     Left edge of the trace plot area in canvas pixels.
 * plotWidth : number
 *     Width of the trace plot area in canvas pixels.
 * widthFraction : number, optional
 *     Fraction of ``plotWidth`` used for the data range (default 1).
 *
 * Returns
 * -------
 * number
 *     Horizontal canvas coordinate for ``value``.
 */
export function rowSumToTraceX(
  value: number,
  vmin: number,
  vmax: number,
  plotLeft: number,
  plotWidth: number,
  widthFraction = 1,
): number {
  const padding = (plotWidth * (1 - widthFraction)) / 2;
  const innerLeft = plotLeft + padding;
  const innerWidth = plotWidth * widthFraction;
  const t = normalizeToUnit(value, vmin, vmax);
  return innerLeft + (1 - t) * innerWidth;
}
