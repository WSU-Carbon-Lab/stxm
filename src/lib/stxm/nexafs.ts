import "server-only";

import { regionMeanAndSigma, type WeightingMode } from "@/lib/stxm/estimators";

/**
 * NEXAFS optical density and uncertainties via Beer-Lambert (OD = ln(I0/I)).
 */
export function nexafsBeerLambert(
  image: number[][],
  sampleMask: boolean[],
  izeroMask: boolean[],
  eps = 1e-10,
  mode: WeightingMode = "poisson_mle",
): {
  od: number[];
  sigmaOd: number[];
  i0: number[];
  sigmaI0: number[];
  intensity: number[];
  sigmaI: number[];
  nSample: number;
  nIzero: number;
} {
  const { mean: I0raw, sigma: sigmaI0, n: nIzero } = regionMeanAndSigma(image, izeroMask, mode, eps);
  const { mean: iSample, sigma: sigmaI, n: nSample } = regionMeanAndSigma(
    image,
    sampleMask,
    mode,
    eps,
  );
  const i0 = I0raw.map((value) => Math.max(value, eps));
  const iS = iSample.map((value) => Math.max(value, eps));
  const od = i0.map((i0Val, idx) => Math.log(i0Val / (iS[idx] ?? eps)));
  const sigmaOd = i0.map((i0Val, idx) => {
    const si0 = sigmaI0[idx] ?? 0;
    const si = sigmaI[idx] ?? 0;
    const isVal = iS[idx] ?? eps;
    return Math.sqrt((si0 / i0Val) ** 2 + (si / isVal) ** 2);
  });
  return {
    od,
    sigmaOd,
    i0,
    sigmaI0,
    intensity: iS,
    sigmaI,
    nSample,
    nIzero,
  };
}
