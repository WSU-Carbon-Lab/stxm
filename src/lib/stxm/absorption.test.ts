import { describe, expect, test } from "bun:test";

import {
  fitBareAtomBackground,
  HC_EV_CM,
  normalizedMassAbsorptionAtEnergies,
  odToBeta,
} from "@/lib/stxm/absorption";

describe("fitBareAtomBackground", () => {
  test("recovers scale and offset on synthetic linear bare atom", () => {
    const energyEv = [270, 280, 290, 300, 310];
    const muRho = energyEv.map((energy) => energy * 0.01);
    const scaleTrue = 2.5;
    const offsetTrue = 0.4;
    const od = muRho.map((mu) => scaleTrue * mu + offsetTrue);
    const fit = fitBareAtomBackground(energyEv, od, muRho, 2, 2);
    expect(fit.scale).toBeCloseTo(scaleTrue, 5);
    expect(fit.offset).toBeCloseTo(offsetTrue, 5);
  });

  test("scale-only fit when nLow is zero", () => {
    const energyEv = [100, 200, 300];
    const muRho = [1, 2, 3];
    const od = [3, 6, 9];
    const fit = fitBareAtomBackground(energyEv, od, muRho, 0, 2);
    expect(fit.scale).toBeCloseTo(3, 8);
    expect(fit.offset).toBe(0);
  });
});

describe("normalizedMassAbsorptionAtEnergies", () => {
  test("matches manual (OD - const) / scale", () => {
    const energyEv = [275, 285, 295, 305, 315];
    const muRho = [0.5, 0.6, 0.8, 1.0, 1.1];
    const od = muRho.map((mu) => 2 * mu + 0.1);
    const odErr = od.map(() => 0.05);
    const result = normalizedMassAbsorptionAtEnergies(energyEv, od, odErr, muRho, {
      nLow: 2,
      nHigh: 2,
      fitOffset: true,
    });
    expect(result.values.length).toBe(energyEv.length);
    for (let index = 0; index < energyEv.length; index += 1) {
      const expected = (od[index]! - result.offset) / result.scale;
      expect(result.values[index]).toBeCloseTo(expected, 8);
      expect(result.errors[index]).toBeCloseTo(0.05 / result.scale, 8);
    }
  });
});

describe("odToBeta", () => {
  test("uses lambda = HC/E and beta = OD*lambda/(4*pi*t)", () => {
    const energyEv = [300];
    const od = [0.2];
    const thicknessCm = 1e-4;
    const beta = odToBeta(energyEv, od, thicknessCm)[0]!;
    const lamCm = HC_EV_CM / energyEv[0]!;
    expect(beta).toBeCloseTo((od[0]! * lamCm) / (4 * Math.PI * thicknessCm), 12);
  });
});
