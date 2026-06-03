import { describe, expect, test } from "bun:test";

import {
  computeRowSums,
  lineScanImageDisplayLimits,
  lineScanImageDisplayScale,
  lineScanLogFloor,
  lineScanPixelGray,
  normalizeToUnit,
  percentileLimits,
  percentileSorted,
  qAxisValueToPx,
  pxToQAxisValue,
  qAxisBounds,
  rowSumToTraceX,
  rowSumTraceLimits,
} from "@/lib/stxm/plot-scale";

describe("percentileSorted", () => {
  test("interpolates linearly between order statistics", () => {
    expect(percentileSorted([1, 2, 3, 4, 5], 0)).toBe(1);
    expect(percentileSorted([1, 2, 3, 4, 5], 100)).toBe(5);
    expect(percentileSorted([1, 2, 3, 4, 5], 50)).toBe(3);
  });
});

describe("percentileLimits", () => {
  test("uses 5th and 95th percentiles of image pixels by default", () => {
    const image = Array.from({ length: 100 }, (_, index) =>
      Array.from({ length: 10 }, () => index),
    );
    const [vmin, vmax] = percentileLimits(image, 0, 99);
    expect(vmin).toBeCloseTo(4.95, 5);
    expect(vmax).toBeCloseTo(94.05, 5);
    expect(vmin).toBeLessThan(vmax);
  });

  test("falls back when limits are degenerate", () => {
    expect(percentileLimits([[5, 5, 5]], 0, 10)).toEqual([0, 10]);
  });
});

describe("normalizeToUnit", () => {
  test("maps endpoints to zero and one", () => {
    expect(normalizeToUnit(0, 0, 10)).toBe(0);
    expect(normalizeToUnit(10, 0, 10)).toBe(1);
    expect(normalizeToUnit(5, 0, 10)).toBe(0.5);
  });
});

describe("rowSumTraceLimits", () => {
  test("spans the data range with symmetric padding", () => {
    const rowSums = [100, 200, 150];
    const [vmin, vmax] = rowSumTraceLimits(rowSums);
    expect(vmin).toBeCloseTo(95);
    expect(vmax).toBeCloseTo(205);
  });

  test("does not floor non-negative data at zero", () => {
    const rowSums = [1000, 1100, 1050];
    const [vmin, vmax] = rowSumTraceLimits(rowSums);
    expect(vmin).toBeGreaterThan(0);
    expect(vmax).toBeGreaterThan(vmin);
  });
});

describe("lineScanImageDisplayLimits", () => {
  test("ignores zeros when enough positive pixels exist", () => {
    const image = Array.from({ length: 20 }, () =>
      Array.from({ length: 20 }, () => 0),
    );
    for (let row = 5; row < 15; row += 1) {
      for (let col = 5; col < 15; col += 1) {
        image[row]![col] = 100;
      }
    }
    image[0]![0] = 1e6;
    const [vmin, vmax] = lineScanImageDisplayLimits(image, 0, 1e6);
    expect(vmin).toBeLessThan(vmax);
    expect(vmax).toBeLessThan(1e6);
  });

  test("log limits operate on log10 counts", () => {
    const image = Array.from({ length: 20 }, () =>
      Array.from({ length: 20 }, (_, col) => 100 * 10 ** (col % 3)),
    );
    const scale = lineScanImageDisplayScale(image, 0, 10000, "log");
    expect(scale.mode).toBe("log");
    expect(scale.vmin).toBeGreaterThanOrEqual(Math.log10(100));
    expect(scale.vmax).toBeLessThanOrEqual(Math.log10(10000));
    expect(scale.vmax).toBeGreaterThan(scale.vmin);
  });

  test("log scale maps izero counts brighter than film counts", () => {
    const image = Array.from({ length: 40 }, (_, row) =>
      Array.from({ length: 20 }, () => (row < 10 ? 5000 : 500)),
    );
    image[0]![0] = 50000;
    const logScale = lineScanImageDisplayScale(image, 0, 50000, "log");
    expect(lineScanPixelGray(5000, logScale)).toBeGreaterThan(lineScanPixelGray(500, logScale));
  });
});

describe("lineScanLogFloor", () => {
  test("returns at least one count", () => {
    expect(lineScanLogFloor([0.5, 2, 10])).toBeGreaterThanOrEqual(1);
    expect(lineScanLogFloor([])).toBe(1);
  });
});

describe("qAxisValueToPx", () => {
  test("maps first qaxis point to top and last to bottom", () => {
    const qaxis = [0, 1, 2, 3];
    expect(qAxisValueToPx(0, qaxis, 100)).toBeCloseTo(0);
    expect(qAxisValueToPx(3, qaxis, 100)).toBeCloseTo(100);
  });

  test("inverts when qaxis decreases", () => {
    const qaxis = [3, 2, 1, 0];
    expect(qAxisValueToPx(3, qaxis, 100)).toBeCloseTo(0);
    expect(qAxisValueToPx(0, qaxis, 100)).toBeCloseTo(100);
  });

  test("pxToQAxisValue inverts qAxisValueToPx", () => {
    const qaxis = [0.5, 1.5, 2.5];
    const height = 120;
    const value = 1.5;
    const px = qAxisValueToPx(value, qaxis, height);
    expect(pxToQAxisValue(px, 0, height, qaxis)).toBeCloseTo(value);
  });
});

describe("computeRowSums and rowSumToTraceX", () => {
  test("maps high row sums toward the heatmap edge", () => {
    const image = [
      [1, 1],
      [3, 3],
    ];
    const rowSums = computeRowSums(image);
    expect(rowSums).toEqual([2, 6]);
    const [vmin, vmax] = rowSumTraceLimits(rowSums);
    const lowX = rowSumToTraceX(rowSums[0]!, vmin, vmax, 4, 40);
    const highX = rowSumToTraceX(rowSums[1]!, vmin, vmax, 4, 40);
    expect(highX).toBeLessThan(lowX);
  });
});
