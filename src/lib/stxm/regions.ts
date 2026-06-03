import "server-only";

/**
 * Region geometry helpers for line-scan sample/izero bar placement.
 */

function convolveSame(values: number[], kernel: number[]): number[] {
  const n = values.length;
  const kLen = kernel.length;
  const half = Math.floor(kLen / 2);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    let sum = 0;
    for (let j = 0; j < kLen; j += 1) {
      const idx = i + j - half;
      if (idx >= 0 && idx < n) {
        sum += (values[idx] ?? 0) * (kernel[j] ?? 0);
      }
    }
    out[i] = sum;
  }
  return out;
}

function kMeans1D(data: number[], k: number, randomState: number, maxIter = 100): number[] {
  const n = data.length;
  if (n === 0) {
    return [];
  }
  const sorted = [...data].sort((a, b) => a - b);
  const centroids: number[] = [];
  for (let i = 0; i < k; i += 1) {
    const idx = Math.min(n - 1, Math.floor(((i + 0.5) * n) / k));
    centroids.push(sorted[idx] ?? sorted[0] ?? 0);
  }
  const labels = new Array<number>(n).fill(0);
  let seed = randomState;
  const rand = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0x1_0000_0000;
  };
  for (let iter = 0; iter < maxIter; iter += 1) {
    let changed = false;
    for (let i = 0; i < n; i += 1) {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c += 1) {
        const dist = Math.abs((data[i] ?? 0) - (centroids[c] ?? 0));
        if (dist < bestDist) {
          bestDist = dist;
          best = c;
        }
      }
      if (labels[i] !== best) {
        labels[i] = best;
        changed = true;
      }
    }
    const newCentroids = new Array<number>(k).fill(0);
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < n; i += 1) {
      const label = labels[i] ?? 0;
      newCentroids[label] = (newCentroids[label] ?? 0) + (data[i] ?? 0);
      counts[label] = (counts[label] ?? 0) + 1;
    }
    for (let c = 0; c < k; c += 1) {
      if ((counts[c] ?? 0) > 0) {
        centroids[c] = (newCentroids[c] ?? 0) / (counts[c] ?? 1);
      } else {
        centroids[c] = sorted[Math.floor(rand() * n)] ?? 0;
      }
    }
    if (!changed) {
      break;
    }
  }
  return labels;
}

function segmentSpatialRegions(
  image: number[][],
  nRegions = 3,
  profileColumns?: number,
  randomState = 0,
): { rowLabels: number[]; labelNames: string[] } {
  const nRows = image.length;
  const nCols = image[0]?.length ?? 0;
  const cols = profileColumns ?? Math.min(20, nCols);
  const profile: number[] = [];
  for (let row = 0; row < nRows; row += 1) {
    let sum = 0;
    for (let col = Math.max(0, nCols - cols); col < nCols; col += 1) {
      sum += image[row]?.[col] ?? 0;
    }
    profile.push(sum / cols);
  }
  const rowLabels = new Array<number>(nRows).fill(0);
  const raw = kMeans1D(profile, 3, randomState);
  const extents: number[] = [0, 0, 0];
  const meansIntensity: number[] = [0, 0, 0];
  const meanRow: number[] = [0, 0, 0];
  for (let idx = 0; idx < 3; idx += 1) {
    let count = 0;
    let intensitySum = 0;
    let rowSum = 0;
    for (let row = 0; row < nRows; row += 1) {
      if ((raw[row] ?? 0) === idx) {
        count += 1;
        rowSum += row;
        for (let col = 0; col < nCols; col += 1) {
          intensitySum += image[row]?.[col] ?? 0;
        }
      }
    }
    extents[idx] = count;
    meansIntensity[idx] = count > 0 ? intensitySum / (count * nCols) : 0;
    meanRow[idx] = count > 0 ? rowSum / count : 0;
  }
  let edgeIdx = 0;
  for (let idx = 1; idx < 3; idx += 1) {
    if ((extents[idx] ?? 0) < (extents[edgeIdx] ?? 0)) {
      edgeIdx = idx;
    }
  }
  const other = [0, 1, 2].filter((i) => i !== edgeIdx);
  const leftIdx = (meanRow[other[0] ?? 0] ?? 0) < (meanRow[other[1] ?? 0] ?? 0) ? other[0]! : other[1]!;
  const rightIdx = leftIdx === other[0] ? other[1]! : other[0]!;
  const sampleIdx =
    (meansIntensity[leftIdx] ?? 0) < (meansIntensity[rightIdx] ?? 0) ? leftIdx : rightIdx;
  const izeroIdx = sampleIdx === leftIdx ? rightIdx : leftIdx;
  for (let row = 0; row < nRows; row += 1) {
    const label = raw[row] ?? 0;
    if (label === sampleIdx) {
      rowLabels[row] = 0;
    } else if (label === edgeIdx) {
      rowLabels[row] = 1;
    } else if (label === izeroIdx) {
      rowLabels[row] = 2;
    }
  }
  const labelNames = ["sample", "edge", "izero"];
  if (nRegions > 3) {
    return { rowLabels, labelNames };
  }
  return { rowLabels, labelNames };
}

/**
 * Infer sample and izero bar positions from image profile.
 */
export function autoSampleIzeroRegions(
  image: number[][],
  qaxis: number[],
): [number, number, number, number] {
  const n = image.length;
  const profile = image.map((row) => row[row.length - 1] ?? 0);
  if (n < 4 || qaxis.length !== n) {
    const yMin = Math.min(...qaxis);
    const yMax = Math.max(...qaxis);
    const span = yMax - yMin;
    const margin = span * 0.05;
    return [yMin + span * 0.45, yMax - margin, yMin + margin, yMin + span * 0.35];
  }
  const kernel = [0.2, 0.2, 0.2, 0.2, 0.2];
  const smoothed = convolveSame(profile, kernel);
  let cliffIdx = 0;
  let maxGrad = 0;
  for (let i = 0; i < n - 1; i += 1) {
    const grad = Math.abs((smoothed[i + 1] ?? 0) - (smoothed[i] ?? 0));
    if (grad > maxGrad) {
      maxGrad = grad;
      cliffIdx = i;
    }
  }
  cliffIdx = Math.max(0, Math.min(n - 2, cliffIdx));
  const leftWinStart = Math.max(0, cliffIdx - 2);
  const leftWinEnd = cliffIdx + 1;
  const rightWinStart = cliffIdx + 1;
  const rightWinEnd = Math.min(n, cliffIdx + 4);
  let leftSum = 0;
  let leftCount = 0;
  for (let i = leftWinStart; i < leftWinEnd; i += 1) {
    leftSum += smoothed[i] ?? 0;
    leftCount += 1;
  }
  let rightSum = 0;
  let rightCount = 0;
  for (let i = rightWinStart; i < rightWinEnd; i += 1) {
    rightSum += smoothed[i] ?? 0;
    rightCount += 1;
  }
  const leftMean = leftCount > 0 ? leftSum / leftCount : 0;
  const rightMean = rightCount > 0 ? rightSum / rightCount : 0;
  const izeroOnLeft = leftMean < rightMean;
  const bufferPixels = Math.max(1, Math.floor(n * 0.05));
  const minRegionPixels = Math.max(2, Math.floor(n * 0.08));
  let barSampleLo: number;
  let barSampleHi: number;
  let barIzeroLo: number;
  let barIzeroHi: number;
  if (izeroOnLeft) {
    let izeroEnd = cliffIdx - bufferPixels;
    let sampleStart = cliffIdx + 1 + bufferPixels;
    izeroEnd = Math.max(minRegionPixels - 1, Math.min(n - 2, izeroEnd));
    sampleStart = Math.max(1, Math.min(n - minRegionPixels, sampleStart));
    if (izeroEnd < sampleStart) {
      barIzeroLo = qaxis[0] ?? 0;
      barIzeroHi = qaxis[izeroEnd] ?? 0;
      barSampleLo = qaxis[sampleStart] ?? 0;
      barSampleHi = qaxis[n - 1] ?? 0;
    } else {
      const mid = Math.floor(n / 2);
      barSampleLo = qaxis[0] ?? 0;
      barSampleHi = qaxis[Math.max(0, mid - minRegionPixels)] ?? 0;
      barIzeroLo = qaxis[Math.min(n - 1, mid + minRegionPixels)] ?? 0;
      barIzeroHi = qaxis[n - 1] ?? 0;
    }
  } else {
    let sampleEnd = cliffIdx - bufferPixels;
    let izeroStart = cliffIdx + 1 + bufferPixels;
    sampleEnd = Math.max(minRegionPixels - 1, Math.min(n - 2, sampleEnd));
    izeroStart = Math.max(1, Math.min(n - minRegionPixels, izeroStart));
    if (sampleEnd < izeroStart) {
      barSampleLo = qaxis[0] ?? 0;
      barSampleHi = qaxis[sampleEnd] ?? 0;
      barIzeroLo = qaxis[izeroStart] ?? 0;
      barIzeroHi = qaxis[n - 1] ?? 0;
    } else {
      const mid = Math.floor(n / 2);
      barIzeroLo = qaxis[0] ?? 0;
      barIzeroHi = qaxis[Math.max(0, mid - minRegionPixels)] ?? 0;
      barSampleLo = qaxis[Math.min(n - 1, mid + minRegionPixels)] ?? 0;
      barSampleHi = qaxis[n - 1] ?? 0;
    }
  }
  const span = Math.max(...qaxis) - Math.min(...qaxis);
  const margin = span * 0.02;
  if (Math.abs(barSampleHi - barSampleLo) < margin) {
    const lo = Math.min(barSampleLo, barSampleHi);
    const hi = Math.max(barSampleLo, barSampleHi);
    barSampleLo = lo - margin;
    barSampleHi = hi + margin;
  }
  if (Math.abs(barIzeroHi - barIzeroLo) < margin) {
    const lo = Math.min(barIzeroLo, barIzeroHi);
    const hi = Math.max(barIzeroLo, barIzeroHi);
    barIzeroLo = lo - margin;
    barIzeroHi = hi + margin;
  }
  return [barSampleLo, barSampleHi, barIzeroLo, barIzeroHi];
}

/**
 * Set dividing bar positions from 3-region segmentation (sample, edge, izero).
 */
export function barBoundsFromThreeRegions(
  image: number[][],
  qaxis: number[],
  profileColumns?: number,
  randomState = 0,
): [number, number, number, number] {
  const nRows = image.length;
  if (nRows < 3 || qaxis.length !== nRows) {
    const yMin = Math.min(...qaxis);
    const yMax = Math.max(...qaxis);
    const span = yMax - yMin;
    const margin = span * 0.05;
    return [yMin + span * 0.45, yMax - margin, yMin + margin, yMin + span * 0.35];
  }
  const { rowLabels } = segmentSpatialRegions(image, 3, profileColumns, randomState);
  const sampleRows: number[] = [];
  const izeroRows: number[] = [];
  for (let row = 0; row < nRows; row += 1) {
    if (rowLabels[row] === 0) {
      sampleRows.push(row);
    } else if (rowLabels[row] === 2) {
      izeroRows.push(row);
    }
  }
  if (sampleRows.length === 0 || izeroRows.length === 0) {
    return autoSampleIzeroRegions(image, qaxis);
  }
  const qSample = sampleRows.map((row) => qaxis[row] ?? 0);
  const qIzero = izeroRows.map((row) => qaxis[row] ?? 0);
  let barSampleLo = Math.min(...qSample);
  let barSampleHi = Math.max(...qSample);
  let barIzeroLo = Math.min(...qIzero);
  let barIzeroHi = Math.max(...qIzero);
  const span = Math.max(...qaxis) - Math.min(...qaxis);
  const margin = Math.max(span * 0.01, 1e-9);
  if (Math.abs(barSampleHi - barSampleLo) < margin) {
    const lo = Math.min(barSampleLo, barSampleHi);
    const hi = Math.max(barSampleLo, barSampleHi);
    barSampleLo = lo - margin;
    barSampleHi = hi + margin;
  }
  if (Math.abs(barIzeroHi - barIzeroLo) < margin) {
    const lo = Math.min(barIzeroLo, barIzeroHi);
    const hi = Math.max(barIzeroLo, barIzeroHi);
    barIzeroLo = lo - margin;
    barIzeroHi = hi + margin;
  }
  return [barSampleLo, barSampleHi, barIzeroLo, barIzeroHi];
}

export { sampleIzeroMasks } from "@/lib/stxm/sample-masks";
