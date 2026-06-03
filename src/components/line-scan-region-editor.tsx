"use client";

import { Minus } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

import { RowSumTrace, ROW_SUM_TRACE_WIDTH } from "@/components/row-sum-trace";
import type { IzeroBounds, StxmRegion } from "@/lib/stxm-types";
import {
  lineScanImageDisplayScale,
  lineScanPixelGray,
  pxToQAxisValue,
  qAxisBounds,
  qAxisValueToPx,
  type PlotScaleMode,
} from "@/lib/stxm/plot-scale";
import { IZERO_COLOR, REGION_COLORS } from "@/lib/stxm/region-colors";

type LineScanRegionEditorProps = {
  image: number[][];
  paxisPoints: number[];
  qaxisPoints: number[];
  regions: StxmRegion[];
  izero: IzeroBounds;
  imageScaleMode?: PlotScaleMode;
  onRegionsChange: (regions: StxmRegion[]) => void;
  onRegionChange: (index: number, region: StxmRegion) => void;
  onIzeroChange: (izero: IzeroBounds) => void;
  onDragStart?: (target: RegionDragTarget) => void;
  onDragEnd?: () => void;
};

/**
 * Interactive line-scan heatmap with draggable sample and izero region bars.
 *
 * Energy spans columns via ``paxisPoints``; sample position spans rows via
 * ``qaxisPoints`` with row 0 at the top (Matplotlib ``origin="upper"``).
 * High raw counts (izero) render brighter than lower film counts.
 */

/** Identifies which spectrum trace a drag operation should update live. */
export type RegionDragTarget =
  | { kind: "izero" }
  | { kind: "region"; index: number };

type RegionGap = {
  lo: number;
  hi: number;
};

type DragState =
  | { kind: "izero-lo" }
  | { kind: "izero-hi" }
  | { kind: "region"; index: number; edge: "lo" | "hi" }
  | null;

const PLOT_WIDTH = 280;
const HEATMAP_WIDTH = PLOT_WIDTH - ROW_SUM_TRACE_WIDTH;
const CANVAS_HEIGHT = 520;
const HIT_MARGIN_FRACTION = 0.015;

function regionDisplayLabel(region: StxmRegion, index: number): string {
  const label = region.spot_label.trim();
  return label.length > 0 ? label : `Region ${index + 1}`;
}

export function LineScanRegionEditor({
  image,
  paxisPoints,
  qaxisPoints,
  regions,
  izero,
  imageScaleMode = "log",
  onRegionsChange,
  onRegionChange,
  onIzeroChange,
  onDragStart,
  onDragEnd,
}: LineScanRegionEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverDragTarget, setHoverDragTarget] = useState<DragState>(null);
  const regionListId = useId();
  const [editingRegionIndex, setEditingRegionIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const editInputRef = useRef<HTMLInputElement | null>(null);

  const [sampleMin, sampleMax] = qAxisBounds(qaxisPoints);
  const sampleSpan = sampleMax - sampleMin || 1;
  const minGap = sampleSpan * 0.02;

  const yToPx = useCallback(
    (value: number, height: number) => qAxisValueToPx(value, qaxisPoints, height),
    [qaxisPoints],
  );

  const rowBandPx = useCallback(
    (row: number, rows: number, height: number) => {
      if (qaxisPoints.length !== rows || rows === 0) {
        const cellHeight = height / rows;
        return { top: row * cellHeight, bottom: (row + 1) * cellHeight };
      }
      const yAt = (index: number) => yToPx(qaxisPoints[index] ?? qaxisPoints[0] ?? sampleMin, height);
      const yRow = yAt(row);
      const top = row === 0 ? 0 : (yAt(row - 1) + yRow) / 2;
      const bottom = row === rows - 1 ? height : (yRow + yAt(row + 1)) / 2;
      return top <= bottom ? { top, bottom } : { top: bottom, bottom: top };
    },
    [qaxisPoints, sampleMin, yToPx],
  );

  const pxToSample = useCallback(
    (clientY: number, canvas: HTMLCanvasElement) => {
      const rect = canvas.getBoundingClientRect();
      return pxToQAxisValue(clientY, rect.top, rect.height, qaxisPoints);
    },
    [qaxisPoints],
  );

  const regionGaps = useMemo(() => computeRegionGaps(regions, izero, sampleMin, sampleMax, minGap), [
    regions,
    izero,
    sampleMin,
    sampleMax,
    minGap,
  ]);

  useEffect(() => {
    if (!isDragging) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      const sample = pxToSample(event.clientY, canvas);
      const drag = dragRef.current;
      if (!drag) {
        return;
      }
      if (drag.kind === "izero-lo") {
        onIzeroChange({
          ...izero,
          izero_lo: clamp(sample, sampleMin, izero.izero_hi - minGap),
        });
        return;
      }
      if (drag.kind === "izero-hi") {
        onIzeroChange({
          ...izero,
          izero_hi: clamp(sample, izero.izero_lo + minGap, sampleMax),
        });
        return;
      }
      const region = regions[drag.index];
      if (!region) {
        return;
      }
      if (drag.edge === "lo") {
        onRegionChange(drag.index, {
          ...region,
          sample_lo: clamp(sample, sampleMin, region.sample_hi - minGap),
        });
        return;
      }
      onRegionChange(drag.index, {
        ...region,
        sample_hi: clamp(sample, region.sample_lo + minGap, sampleMax),
      });
    };
    const handleUp = () => {
      dragRef.current = null;
      setIsDragging(false);
      onDragEnd?.();
    };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [
    isDragging,
    izero,
    minGap,
    onDragEnd,
    onIzeroChange,
    onRegionChange,
    pxToSample,
    regions,
    sampleMax,
    sampleMin,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || image.length === 0) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const width = canvas.width;
    const height = canvas.height;
    const finite = image.flat().filter((value) => Number.isFinite(value));
    const dataMin = finite.length > 0 ? Math.min(...finite) : 0;
    const dataMax = finite.length > 0 ? Math.max(...finite) : 1;
    const displayScale = lineScanImageDisplayScale(image, dataMin, dataMax, imageScaleMode);

    context.clearRect(0, 0, width, height);
    const rows = image.length;
    const cols = image[0]?.length ?? 0;
    for (let row = 0; row < rows; row += 1) {
      const { top, bottom } = rowBandPx(row, rows, height);
      const rowHeight = bottom - top;
      for (let col = 0; col < cols; col += 1) {
        const value = image[row]?.[col] ?? 0;
        const gray = lineScanPixelGray(value, displayScale);
        context.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        const x = (col / cols) * width;
        const cellWidth = width / cols + 1;
        context.fillRect(x, top, cellWidth, rowHeight + 1);
      }
    }

    context.font = "11px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";

    context.strokeStyle = IZERO_COLOR;
    context.lineWidth = 2;
    drawHorizontalLine(context, yToPx(izero.izero_lo, height), width);
    drawHorizontalLine(context, yToPx(izero.izero_hi, height), width);
    context.fillStyle = IZERO_COLOR;
    context.fillText(
      "izero",
      width / 2,
      yToPx((izero.izero_lo + izero.izero_hi) / 2, height),
    );

    regions.forEach((region, index) => {
      const color = REGION_COLORS[index % REGION_COLORS.length] ?? "#16a34a";
      context.strokeStyle = color;
      drawHorizontalLine(context, yToPx(region.sample_lo, height), width);
      drawHorizontalLine(context, yToPx(region.sample_hi, height), width);
    });
  }, [image, imageScaleMode, paxisPoints, qaxisPoints, regions, izero, rowBandPx, yToPx]);

  useEffect(() => {
    if (editingRegionIndex === null) {
      return;
    }
    if (editingRegionIndex >= regions.length) {
      setEditingRegionIndex(null);
      return;
    }
    editInputRef.current?.focus();
    editInputRef.current?.select();
  }, [editingRegionIndex, regions.length]);

  const commitRegionLabelEdit = useCallback(
    (index: number) => {
      const region = regions[index];
      if (!region) {
        setEditingRegionIndex(null);
        return;
      }
      onRegionChange(index, { ...region, spot_label: editDraft.trim() });
      setEditingRegionIndex(null);
    },
    [editDraft, onRegionChange, regions],
  );

  const cancelRegionLabelEdit = useCallback(() => {
    setEditingRegionIndex(null);
  }, []);

  const startRegionLabelEdit = useCallback(
    (index: number) => {
      const region = regions[index];
      if (!region) {
        return;
      }
      setEditDraft(region.spot_label);
      setEditingRegionIndex(index);
    },
    [regions],
  );

  const isInteractiveOverlayTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    return Boolean(
      target.closest("[data-gap-button]") ??
        target.closest("[data-region-label]") ??
        target.closest("[data-region-remove]") ??
        target.closest("[data-region-label-edit]"),
    );
  };

  const updateHoverFromEvent = useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      if (isInteractiveOverlayTarget(event.target)) {
        setHoverDragTarget(null);
        return;
      }
      const canvas = canvasRef.current;
      if (!canvas) {
        setHoverDragTarget(null);
        return;
      }
      const sample = pxToSample(event.clientY, canvas);
      const hitMargin = sampleSpan * HIT_MARGIN_FRACTION;
      setHoverDragTarget(findDragTarget(sample, hitMargin, izero, regions));
    },
    [izero, pxToSample, regions, sampleSpan],
  );

  const beginDrag = (event: React.MouseEvent<HTMLElement>) => {
    if (event.button !== 0) {
      return;
    }
    if (isInteractiveOverlayTarget(event.target)) {
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const sample = pxToSample(event.clientY, canvas);
    const hitMargin = sampleSpan * HIT_MARGIN_FRACTION;
    const dragTarget = findDragTarget(sample, hitMargin, izero, regions);
    if (dragTarget) {
      dragRef.current = dragTarget;
      setIsDragging(true);
      if (dragTarget.kind === "izero-lo" || dragTarget.kind === "izero-hi") {
        onDragStart?.({ kind: "izero" });
      } else {
        onDragStart?.({ kind: "region", index: dragTarget.index });
      }
    }
  };

  const endDrag = () => {
    dragRef.current = null;
    setIsDragging(false);
  };

  const addRegionInGap = (gap: RegionGap) => {
    onRegionsChange([
      ...regions,
      {
        sample_lo: gap.lo,
        sample_hi: gap.hi,
        spot_label: `spot${regions.length + 1}`,
      },
    ]);
  };

  const addRegionFallback = () => {
    if (regionGaps.length > 0) {
      addRegionInGap(regionGaps[0]!);
      return;
    }
    const last = regions.at(-1);
    onRegionsChange([
      ...regions,
      {
        sample_lo: last?.sample_hi ?? sampleMin,
        sample_hi: clamp((last?.sample_hi ?? sampleMin) + sampleSpan * 0.1, sampleMin, sampleMax),
        spot_label: `spot${regions.length + 1}`,
      },
    ]);
  };

  const removeRegion = (index: number) => {
    if (regions.length <= 1) {
      return;
    }
    if (editingRegionIndex === index) {
      setEditingRegionIndex(null);
    }
    onRegionsChange(regions.filter((_, rowIndex) => rowIndex !== index));
  };

  return (
    <div className="flex w-full max-w-[280px] flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white">
      <div
        className={`flex ${isDragging || hoverDragTarget ? "cursor-ns-resize" : "cursor-crosshair"}`}
        onMouseDown={beginDrag}
        onMouseMove={updateHoverFromEvent}
        onMouseLeave={() => setHoverDragTarget(null)}
        onMouseUp={endDrag}
      >
        <RowSumTrace
          image={image}
          height={CANVAS_HEIGHT}
          qaxisPoints={qaxisPoints}
          sampleMin={sampleMin}
          yToPx={yToPx}
          izero={izero}
          regions={regions}
        />
        <div className="relative min-w-0 flex-1">
          <canvas
            ref={canvasRef}
            width={HEATMAP_WIDTH}
            height={CANVAS_HEIGHT}
            className="pointer-events-none relative z-0 block h-auto w-full"
          />
        {!isDragging ? (
          <>
            {regions.map((region, index) => {
              const mid = (region.sample_lo + region.sample_hi) / 2;
              const topPct = (yToPx(mid, CANVAS_HEIGHT) / CANVAS_HEIGHT) * 100;
              const label = regionDisplayLabel(region, index);
              const color = REGION_COLORS[index % REGION_COLORS.length] ?? "#16a34a";
              const isEditing = editingRegionIndex === index;
              const canRemove = regions.length > 1;
              return (
                <div
                  key={`region-overlay-${index}`}
                  className="pointer-events-auto absolute left-1/2 z-10 flex -translate-x-1/2 -translate-y-1/2 items-center"
                  style={{ top: `${topPct}%` }}
                  onMouseDown={(event) => event.stopPropagation()}
                >
                  {isEditing ? (
                    <input
                      ref={editInputRef}
                      type="text"
                      data-region-label-edit=""
                      value={editDraft}
                      aria-label={`Edit label for ${label}`}
                      className="w-24 rounded-md border border-zinc-300 bg-white/95 px-1.5 py-0.5 text-center text-[11px] font-medium leading-none shadow-sm outline-none ring-1 ring-zinc-200 focus:ring-zinc-400"
                      style={{ color }}
                      onChange={(event) => setEditDraft(event.target.value)}
                      onBlur={() => commitRegionLabelEdit(index)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          commitRegionLabelEdit(index);
                        }
                        if (event.key === "Escape") {
                          event.preventDefault();
                          cancelRegionLabelEdit();
                        }
                      }}
                    />
                  ) : (
                    <div className="flex items-stretch overflow-hidden rounded-md border border-zinc-300/60 bg-white/90 shadow-sm">
                      <button
                        type="button"
                        data-region-label=""
                        className="cursor-pointer px-1.5 py-0.5 text-[11px] font-medium leading-none hover:underline"
                        style={{ color }}
                        onClick={() => startRegionLabelEdit(index)}
                      >
                        {label}
                      </button>
                      {canRemove ? (
                        <button
                          type="button"
                          data-region-remove=""
                          aria-label={`Remove region ${label}`}
                          className="flex cursor-pointer items-center justify-center self-stretch border-l border-red-300 bg-red-50 px-1 py-0.5 text-red-700 hover:bg-red-100"
                          onClick={(event) => {
                            event.stopPropagation();
                            removeRegion(index);
                          }}
                        >
                          <Minus className="h-3 w-3 shrink-0" aria-hidden="true" />
                        </button>
                      ) : null}
                    </div>
                  )}
                </div>
              );
            })}
            {regionGaps.map((gap) => {
              const mid = (gap.lo + gap.hi) / 2;
              const topPct = (yToPx(mid, CANVAS_HEIGHT) / CANVAS_HEIGHT) * 100;
              return (
                <button
                  key={`${gap.lo}-${gap.hi}`}
                  type="button"
                  data-gap-button=""
                  aria-label="Add region in gap"
                  className="absolute left-1/2 z-10 flex h-6 w-6 -translate-x-1/2 -translate-y-1/2 cursor-pointer items-center justify-center rounded-md border border-zinc-300/40 bg-white/50 text-sm font-semibold leading-none text-zinc-700 shadow-sm hover:border-zinc-400/60 hover:bg-white/70"
                  style={{ top: `${topPct}%` }}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={() => addRegionInGap(gap)}
                >
                  +
                </button>
              );
            })}
          </>
        ) : null}
        </div>
      </div>
      <div className="space-y-1 border-t border-zinc-200 p-2">
        <div className="flex items-center justify-between gap-1">
          <h3 className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
            Regions
          </h3>
          <button
            type="button"
            className="rounded border border-zinc-300 px-1.5 py-0.5 text-[10px] text-zinc-600 hover:bg-zinc-50"
            onClick={addRegionFallback}
          >
            Add region
          </button>
        </div>
        <p className="text-[10px] leading-snug text-zinc-400">
          Click a label on the scan to rename. Drag lines to adjust bounds.
        </p>
        <ul
          id={regionListId}
          className="max-h-28 space-y-0.5 overflow-y-auto"
          aria-label="Sample regions"
        >
          {regions.map((region, index) => {
            const color = REGION_COLORS[index % REGION_COLORS.length] ?? "#16a34a";
            const label = regionDisplayLabel(region, index);
            const canRemove = regions.length > 1;
            return (
              <li
                key={`region-${index}`}
                className="flex min-h-0 items-center gap-1 rounded px-1 py-0.5 hover:bg-zinc-50"
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: color }}
                  aria-hidden="true"
                />
                <span
                  className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-700"
                  title={label}
                >
                  {label}
                </span>
                {canRemove ? (
                  <button
                    type="button"
                    className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-zinc-500 hover:bg-zinc-100 hover:text-red-600"
                    aria-label={`Remove ${label}`}
                    onClick={() => removeRegion(index)}
                  >
                    <Minus className="h-4 w-4" aria-hidden="true" />
                  </button>
                ) : null}
              </li>
            );
          })}
        </ul>
        {regions.length > 0 ? (
          <details className="text-xs">
            <summary className="cursor-pointer list-none text-[10px] text-zinc-500 hover:text-zinc-700 [&::-webkit-details-marker]:hidden">
              Advanced bounds
            </summary>
            <div className="mt-1 space-y-1.5 border-t border-zinc-100 pt-1">
              {regions.map((region, index) => {
                const label = regionDisplayLabel(region, index);
                return (
                  <div
                    key={`bounds-${index}`}
                    className="grid grid-cols-[minmax(0,1fr)_1fr_1fr] items-end gap-1"
                  >
                    <span className="truncate text-[10px] text-zinc-500" title={label}>
                      {label}
                    </span>
                    <label className="min-w-0">
                      <span className="mb-0.5 block text-[9px] uppercase tracking-wide text-zinc-400">
                        Lo
                      </span>
                      <input
                        className="w-full rounded border border-zinc-300 px-1 py-0.5 text-[10px]"
                        type="number"
                        step="0.001"
                        value={region.sample_lo}
                        onChange={(event) => {
                          const next = [...regions];
                          next[index] = { ...region, sample_lo: Number(event.target.value) };
                          onRegionsChange(next);
                        }}
                      />
                    </label>
                    <label className="min-w-0">
                      <span className="mb-0.5 block text-[9px] uppercase tracking-wide text-zinc-400">
                        Hi
                      </span>
                      <input
                        className="w-full rounded border border-zinc-300 px-1 py-0.5 text-[10px]"
                        type="number"
                        step="0.001"
                        value={region.sample_hi}
                        onChange={(event) => {
                          const next = [...regions];
                          next[index] = { ...region, sample_hi: Number(event.target.value) };
                          onRegionsChange(next);
                        }}
                      />
                    </label>
                  </div>
                );
              })}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}

function findDragTarget(
  sample: number,
  hitMargin: number,
  izero: IzeroBounds,
  regions: StxmRegion[],
): NonNullable<DragState> | null {
  const candidates: Array<{ distance: number; drag: NonNullable<DragState> }> = [
    { distance: Math.abs(sample - izero.izero_lo), drag: { kind: "izero-lo" } },
    { distance: Math.abs(sample - izero.izero_hi), drag: { kind: "izero-hi" } },
  ];
  regions.forEach((region, index) => {
    candidates.push(
      { distance: Math.abs(sample - region.sample_lo), drag: { kind: "region", index, edge: "lo" } },
      { distance: Math.abs(sample - region.sample_hi), drag: { kind: "region", index, edge: "hi" } },
    );
  });
  candidates.sort((left, right) => left.distance - right.distance);
  const nearest = candidates[0];
  if (nearest && nearest.distance <= hitMargin) {
    return nearest.drag;
  }
  return null;
}

function computeRegionGaps(
  regions: StxmRegion[],
  izero: IzeroBounds,
  sampleMin: number,
  sampleMax: number,
  minGap: number,
): RegionGap[] {
  const boundaries = new Set<number>([
    sampleMin,
    sampleMax,
    izero.izero_lo,
    izero.izero_hi,
    ...regions.flatMap((region) => [region.sample_lo, region.sample_hi]),
  ]);
  const sorted = [...boundaries].sort((left, right) => left - right);
  const gaps: RegionGap[] = [];
  for (let index = 0; index < sorted.length - 1; index += 1) {
    const lo = sorted[index]!;
    const hi = sorted[index + 1]!;
    if (hi - lo < minGap) {
      continue;
    }
    const mid = (lo + hi) / 2;
    if (mid >= izero.izero_lo && mid <= izero.izero_hi) {
      continue;
    }
    const insideRegion = regions.some(
      (region) => mid > region.sample_lo && mid < region.sample_hi,
    );
    if (insideRegion) {
      continue;
    }
    gaps.push({ lo, hi });
  }
  return gaps;
}

function drawHorizontalLine(
  context: CanvasRenderingContext2D,
  y: number,
  width: number,
): void {
  context.beginPath();
  context.moveTo(0, y);
  context.lineTo(width, y);
  context.stroke();
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
