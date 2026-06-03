/**
 * Boolean masks for sample and izero regions from axis coordinates.
 */
export function sampleIzeroMasks(
  qaxisPoints: number[],
  sampleLo: number,
  sampleHi: number,
  izeroLo: number,
  izeroHi: number,
): { sampleMask: boolean[]; izeroMask: boolean[] } {
  const sampleMask = qaxisPoints.map((q) => q >= sampleLo && q <= sampleHi);
  const izeroMask = qaxisPoints.map((q) => q >= izeroLo && q <= izeroHi);
  return { sampleMask, izeroMask };
}
