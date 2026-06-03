import { describe, expect, test } from "bun:test";

import { regionMeanAndSigma, type WeightingMode } from "@/lib/stxm/estimators";

describe("regionMeanAndSigma weighting modes", () => {
  const values2d = [
    [100, 200, 300],
    [110, 210, 290],
    [90, 190, 310],
    [105, 205, 295],
  ];
  const mask = [true, true, true, false];

  test("poisson_mle, inverse_count, and empirical yield different sigmas", () => {
    const modes: WeightingMode[] = ["poisson_mle", "inverse_count", "empirical"];
    const sigmas = modes.map((mode) => regionMeanAndSigma(values2d, mask, mode).sigma);
    expect(sigmas[0]).not.toEqual(sigmas[1]);
    expect(sigmas[1]).not.toEqual(sigmas[2]);
    expect(sigmas[0]).not.toEqual(sigmas[2]);
    for (const sigma of sigmas) {
      expect(sigma.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    }
  });
});
