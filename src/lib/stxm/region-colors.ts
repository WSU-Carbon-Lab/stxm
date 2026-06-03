export const IZERO_COLOR = "#2563eb";

export const REGION_COLORS = [
  "#16a34a",
  "#0891b2",
  "#ea580c",
  "#c026d3",
  "#65a30d",
  "#ca8a04",
] as const;

/**
 * Return the stroke color for a sample region at the given zero-based index.
 */
export function regionSeriesColor(regionIndex: number): string {
  return REGION_COLORS[regionIndex % REGION_COLORS.length] ?? "#16a34a";
}

/**
 * Return the stroke color for the izero reference series.
 */
export function izeroSeriesColor(): string {
  return IZERO_COLOR;
}
