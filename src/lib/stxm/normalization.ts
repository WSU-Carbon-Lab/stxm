import "server-only";

export type NormalizationMode = "pre_edge_scale" | "scale_shift";

function energyRegionMask(energyPoints: number[], eLo: number, eHi: number): boolean[] {
  return energyPoints.map((energy) => energy >= eLo && energy <= eHi);
}

function preEdgeSubtract(
  energy: number[],
  od: number[],
  preLo: number,
  preHi: number,
): number[] {
  const mask = energyRegionMask(energy, preLo, preHi);
  const vals = od.filter((_, idx) => mask[idx]);
  const finite = vals.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return [...od];
  }
  const baseline = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  return od.map((value) => value - baseline);
}

function postEdgeNormalize(
  energy: number[],
  od: number[],
  postLo: number,
  postHi: number,
  target = 1.0,
): { od: number[]; scale: number } {
  const mask = energyRegionMask(energy, postLo, postHi);
  const vals = od.filter((_, idx) => mask[idx]);
  const finite = vals.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return { od: [...od], scale: 1.0 };
  }
  const meanPost = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  if (meanPost <= 0 || !Number.isFinite(meanPost)) {
    return { od: [...od], scale: 1.0 };
  }
  const scale = target / meanPost;
  return { od: od.map((value) => value * scale), scale };
}

function interpolateLinear(energy: number[], od: number[], queryEnergy: number): number {
  if (energy.length === 0) {
    return Number.NaN;
  }
  if (energy.length === 1) {
    return od[0] ?? Number.NaN;
  }
  const pairs = energy
    .map((e, idx) => ({ e, y: od[idx] ?? Number.NaN }))
    .sort((a, b) => a.e - b.e);
  const x = queryEnergy;
  if (x <= (pairs[0]?.e ?? 0)) {
    return pairs[0]?.y ?? Number.NaN;
  }
  const last = pairs[pairs.length - 1];
  if (last && x >= last.e) {
    return last.y;
  }
  for (let i = 0; i < pairs.length - 1; i += 1) {
    const left = pairs[i];
    const right = pairs[i + 1];
    if (!left || !right) {
      continue;
    }
    if (x >= left.e && x <= right.e) {
      const t = (x - left.e) / (right.e - left.e);
      return left.y + t * (right.y - left.y);
    }
  }
  return Number.NaN;
}

function shiftSpectrum(energy: number[], od: number[], deltaE: number): number[] {
  return energy.map((e) => interpolateLinear(energy, od, e + deltaE));
}

function postEdgeMean(energy: number[], od: number[], postLo: number, postHi: number): number {
  const mask = energyRegionMask(energy, postLo, postHi);
  const vals = od.filter((_, idx) => mask[idx]).filter((value) => Number.isFinite(value));
  if (vals.length === 0) {
    return Number.NaN;
  }
  return vals.reduce((sum, value) => sum + value, 0) / vals.length;
}

function fitEnergyShift(
  energy: number[],
  od: number[],
  preLo: number,
  preHi: number,
  postLo: number,
  postHi: number,
  postTarget = 1.0,
  shiftBounds: [number, number] = [-8, 8],
): { deltaE: number; scale: number } {
  const baselineRemoved = preEdgeSubtract(energy, od, preLo, preHi);
  const span = energy.length > 0 ? Math.max(...energy) - Math.min(...energy) : 1;
  const half = Math.min(Math.max(shiftBounds[1], Math.abs(shiftBounds[0])), 0.15 * span);

  const objective = (delta: number): number => {
    const shifted = shiftSpectrum(energy, baselineRemoved, delta);
    const { od: scaled } = postEdgeNormalize(energy, shifted, postLo, postHi, postTarget);
    const meanPost = postEdgeMean(energy, scaled, postLo, postHi);
    if (!Number.isFinite(meanPost)) {
      return 1e6;
    }
    return (meanPost - postTarget) ** 2;
  };

  let a = -half;
  let b = half;
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a);
  let d = a + gr * (b - a);
  let fc = objective(c);
  let fd = objective(d);
  for (let i = 0; i < 60; i += 1) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = objective(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = objective(d);
    }
  }
  const deltaE = fc < fd ? c : d;
  const shifted = shiftSpectrum(energy, baselineRemoved, deltaE);
  const { scale } = postEdgeNormalize(energy, shifted, postLo, postHi, postTarget);
  return { deltaE, scale };
}

/**
 * Normalize OD and return reproducibility metadata for provenance export.
 */
export function normalizeNexafsWithMetadata(
  energy: number[],
  od: number[],
  preLo: number,
  preHi: number,
  postLo: number,
  postHi: number,
  mode: NormalizationMode = "pre_edge_scale",
  postTarget = 1.0,
  shiftBounds: [number, number] = [-8, 8],
): {
  normalized: number[];
  metadata: { normalization_mode: string; energy_shift_eV: number; post_edge_scale: number };
} {
  const baselineRemoved = preEdgeSubtract(energy, od, preLo, preHi);
  let energyShift = 0;
  let working = baselineRemoved;
  if (mode === "scale_shift") {
    const fit = fitEnergyShift(
      energy,
      od,
      preLo,
      preHi,
      postLo,
      postHi,
      postTarget,
      shiftBounds,
    );
    energyShift = fit.deltaE;
    working = shiftSpectrum(energy, baselineRemoved, energyShift);
  }
  const { od: scaled, scale } = postEdgeNormalize(energy, working, postLo, postHi, postTarget);
  return {
    normalized: scaled,
    metadata: {
      normalization_mode: mode,
      energy_shift_eV: energyShift,
      post_edge_scale: scale,
    },
  };
}
