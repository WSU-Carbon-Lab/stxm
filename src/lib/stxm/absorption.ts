/**
 * NEXAFS absorption and bare-atom normalization (Beer-Lambert OD, CXRO-backed mass absorption).
 *
 * Bare-atom tabulated optical constants follow the same convention as the Python widget:
 * `periodictable.xsf.index_of_refraction` (CXRO-based beta), with
 * mu/rho = (4*pi*beta/lambda) for unit mass density. This matches the STXM notebook pipeline in
 * `stxm.absorption` and differs from xray-atlas Henke f2 mixing
 * (mu/rho = 2*r_e*lambda*N_A*f2/M) only in the tabulation source, not in the step-edge fit:
 * OD ~ scale * mu_bare + const on pre/post-edge windows, then norm. mass abs = (OD - const) / scale.
 */

/** Planck constant times c in eV*cm (wavelength in cm = HC_EV_CM / energy_eV). */
export const HC_EV_CM = 1.2398e-4;

export type BareAtomFitResult = {
  scale: number;
  offset: number;
  odBare: number[];
  fitMask: boolean[];
};

/**
 * Mass absorption coefficient mu/rho (cm^2/g) from tabulated beta: mu = 4*pi*beta/lambda.
 */
export function massAbsorptionFromBeta(energyEv: number[], beta: number[]): number[] {
  return energyEv.map((energy, index) => {
    const lamCm = HC_EV_CM / energy;
    const b = beta[index] ?? 0;
    return (4 * Math.PI * b) / lamCm;
  });
}

/**
 * Fit OD = scale * mu_rho + offset using the lowest `nLow` and highest `nHigh` energy points.
 */
export function fitBareAtomBackground(
  energyEv: number[],
  od: number[],
  muRho: number[],
  nLow = 5,
  nHigh = 5,
): BareAtomFitResult {
  const n = energyEv.length;
  if (n === 0 || od.length !== n || muRho.length !== n) {
    throw new Error("energy_eV, OD, and mu_rho must have the same positive length");
  }
  const nLowUse = Math.min(nLow, n);
  const nHighUse = Math.min(nHigh, n);
  const idxLow = Array.from({ length: nLowUse }, (_, index) => index);
  const idxHigh = Array.from({ length: nHighUse }, (_, index) => n - nHighUse + index);
  const idxFit = [...idxLow, ...idxHigh.filter((index) => !idxLow.includes(index))];
  const fitMask = Array.from({ length: n }, () => false);
  for (const index of idxFit) {
    fitMask[index] = true;
  }

  const muFit = idxFit.map((index) => muRho[index]!);
  const odFit = idxFit.map((index) => od[index]!);

  let scale: number;
  let offset: number;
  if (nLowUse === 0) {
    scale = lstsqOneColumn(muFit, odFit);
    offset = 0;
  } else {
    [scale, offset] = lstsqTwoColumn(muFit, odFit);
  }

  const odBare = muRho.map((mu) => scale * mu + offset);
  return { scale, offset, odBare, fitMask };
}

/**
 * CXRO bare-atom normalized mass absorption (g/cm^2) and uncertainties from OD and tabulated mu/rho.
 */
export function normalizedMassAbsorptionAtEnergies(
  energyEv: number[],
  od: number[],
  odErr: number[],
  muRho: number[],
  options?: { nLow?: number; nHigh?: number; fitOffset?: boolean },
): { values: number[]; errors: number[]; scale: number; offset: number } {
  const nLow = options?.fitOffset === false ? 0 : (options?.nLow ?? 5);
  const nHigh = options?.nHigh ?? 5;
  const { scale, offset } = fitBareAtomBackground(energyEv, od, muRho, nLow, nHigh);
  const scaleSafe = scale !== 0 ? scale : 1;
  const values = od.map((value) => (value - offset) / scaleSafe);
  const errors = odErr.map((value) => Math.abs(value / scaleSafe));
  return { values, errors, scale: scaleSafe, offset };
}

/**
 * Convert optical density to beta (imaginary part of n) at thickness `thicknessCm`.
 */
export function odToBeta(energyEv: number[], od: number[], thicknessCm: number): number[] {
  if (thicknessCm <= 0) {
    throw new Error("thicknessCm must be positive");
  }
  return energyEv.map((energy, index) => {
    const lamCm = HC_EV_CM / energy;
    return ((od[index] ?? 0) * lamCm) / (4 * Math.PI * thicknessCm);
  });
}

/**
 * Uncertainty in beta propagated from OD uncertainty at fixed thickness.
 */
export function odErrToBetaErr(
  energyEv: number[],
  odErr: number[],
  thicknessCm: number,
): number[] {
  if (thicknessCm <= 0) {
    throw new Error("thicknessCm must be positive");
  }
  return energyEv.map((energy, index) => {
    const lamCm = HC_EV_CM / energy;
    return ((odErr[index] ?? 0) * lamCm) / (4 * Math.PI * thicknessCm);
  });
}

function lstsqOneColumn(x: number[], y: number[]): number {
  let sumXx = 0;
  let sumXy = 0;
  for (let index = 0; index < x.length; index += 1) {
    const xi = x[index]!;
    sumXx += xi * xi;
    sumXy += xi * (y[index] ?? 0);
  }
  if (sumXx === 0) {
    return 0;
  }
  return sumXy / sumXx;
}

function lstsqTwoColumn(x: number[], y: number[]): [number, number] {
  let s00 = 0;
  let s01 = 0;
  let s11 = 0;
  let t0 = 0;
  let t1 = 0;
  for (let index = 0; index < x.length; index += 1) {
    const xi = x[index]!;
    const yi = y[index] ?? 0;
    s00 += xi * xi;
    s01 += xi;
    s11 += 1;
    t0 += xi * yi;
    t1 += yi;
  }
  const det = s00 * s11 - s01 * s01;
  if (Math.abs(det) < 1e-30) {
    return [lstsqOneColumn(x, y), 0];
  }
  const scale = (s11 * t0 - s01 * t1) / det;
  const offset = (s00 * t1 - s01 * t0) / det;
  return [scale, offset];
}
