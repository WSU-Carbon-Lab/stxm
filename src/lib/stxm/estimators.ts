export type WeightingMode = "inverse_count" | "poisson_mle" | "empirical";

export const WEIGHTING_MODE_OPTIONS: ReadonlyArray<{
  value: WeightingMode;
  label: string;
}> = [
  { value: "poisson_mle", label: "Poisson MLE" },
  { value: "inverse_count", label: "Inverse count" },
  { value: "empirical", label: "Empirical" },
] as const;

/**
 * Compute per-energy column mean and standard error over masked rows.
 */
export function regionMeanAndSigma(
  values2d: number[][],
  mask: boolean[],
  mode: WeightingMode = "poisson_mle",
  eps = 1e-10,
): { mean: number[]; sigma: number[]; n: number } {
  if (values2d.length === 0 || (values2d[0]?.length ?? 0) === 0) {
    return { mean: [], sigma: [], n: 0 };
  }
  if (mask.length !== values2d.length) {
    throw new Error("mask length must match values2d row count");
  }
  const nEnergy = values2d[0]?.length ?? 0;
  const n = mask.filter(Boolean).length;
  if (n === 0) {
    const empty = new Array<number>(nEnergy).fill(Number.NaN);
    return { mean: empty, sigma: [...empty], n: 0 };
  }
  const block = values2d.filter((_, row) => mask[row]);
  if (mode === "inverse_count") {
    const mean = new Array<number>(nEnergy).fill(0);
    const sigma = new Array<number>(nEnergy).fill(0);
    for (let col = 0; col < nEnergy; col += 1) {
      let weightSum = 0;
      let weightedValSum = 0;
      for (const row of block) {
        const val = Math.max(row[col] ?? 0, eps);
        const weight = 1 / val;
        weightSum += weight;
        weightedValSum += val * weight;
      }
      mean[col] = weightSum > 0 ? weightedValSum / weightSum : Number.NaN;
      sigma[col] = weightSum > 0 ? 1 / Math.sqrt(weightSum) : Number.NaN;
    }
    return { mean, sigma, n };
  }
  const mean = new Array<number>(nEnergy).fill(0);
  for (let col = 0; col < nEnergy; col += 1) {
    let sum = 0;
    for (const row of block) {
      sum += row[col] ?? 0;
    }
    mean[col] = sum / n;
  }
  if (mode === "poisson_mle") {
    const sigma = mean.map((value) => Math.sqrt(Math.max(value, 0) / n));
    return { mean, sigma, n };
  }
  if (mode === "empirical") {
    const sigma = new Array<number>(nEnergy).fill(0);
    for (let col = 0; col < nEnergy; col += 1) {
      const values = block.map((row) => row[col] ?? 0);
      const colMean = mean[col] ?? 0;
      let varSum = 0;
      for (const value of values) {
        const diff = value - colMean;
        varSum += diff * diff;
      }
      const sampleVar = n > 1 ? varSum / (n - 1) : 0;
      sigma[col] = Math.sqrt(sampleVar / n);
    }
    return { mean, sigma, n };
  }
  throw new Error(`unsupported mode: ${String(mode)}`);
}
