"use client";

import dynamic from "next/dynamic";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import type { Config, Data, Layout } from "plotly.js";

import { DraggableLegend } from "@/components/draggable-legend";
import type { PlotScaleMode } from "@/lib/stxm/plot-scale";
import type { IngestionYDisplayMode } from "@/lib/stxm-types";

import {
  collectXExtents,
  collectYExtents,
  collectYValues,
  dominantValueKind,
  planLinearTicks,
  energyFromPlotlyClientX,
  plotHostAreaFromPlotly,
  plotHostPointFromPlotlyData,
  plotHostXFromPlotlyEnergy,
  formatTooltipValueWithErr,
  formatYAxisTitle,
  interpolateChartPoint,
  mergeSeriesForChart,
  plotlyPublicationLinearAxisTicks,
  scaleChartRows,
  spectrumEnergyTickformat,
  spectrumValueTickformat,
  yAxisScaleFromValues,
} from "@/lib/spectrum-chart-data";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export type ChartPoint = {
  energy: number;
  value: number;
  err?: number;
};

export type ChartValueKind = "signal" | "od" | "mass_absorption";

export type ChartSeries = {
  id: string;
  label: string;
  color: string;
  points: ChartPoint[];
  valueKind?: ChartValueKind;
};

type SpectrumChartProps = {
  series: ChartSeries[];
  yLabel?: string;
  yScale?: PlotScaleMode;
  yDisplayMode?: IngestionYDisplayMode;
  height?: number;
  className?: string;
  emptyMessage?: string;
  loading?: boolean;
};

const UNCERTAINTY_FILL_OPACITY = 0.22;
const AXIS_STROKE = "#27272a";
const GRID_STROKE = "#e4e4e7";
const HOVER_GUIDE_STROKE = "#a1a1aa";
const CHART_MARGIN = { t: 36, r: 24, b: 44, l: 72 };

const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: true,
  displaylogo: false,
  responsive: true,
  scrollZoom: false,
  modeBarButtons: [
    ["zoom2d", "pan2d"],
    ["zoomIn2d", "zoomOut2d"],
    ["resetScale2d"],
  ],
};

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  if (normalized.length !== 6) {
    return `rgba(39, 39, 42, ${alpha})`;
  }
  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function buildPrimaryAxisStyle() {
  return {
    showline: true,
    linewidth: 1,
    linecolor: AXIS_STROKE,
    mirror: "allticks" as const,
    zeroline: false,
    showgrid: true,
    gridcolor: GRID_STROKE,
    gridwidth: 1,
    automargin: true,
    ticklabelposition: "outside" as const,
    tickfont: { size: 12, color: AXIS_STROKE },
  };
}

function sortedFinitePoints(points: ChartPoint[]): ChartPoint[] {
  return points
    .filter((point) => Number.isFinite(point.energy) && Number.isFinite(point.value))
    .sort((left, right) => left.energy - right.energy);
}

function buildBandTraces(entry: ChartSeries, scaleY: (value: number) => number): Data[] {
  const bandPoints = sortedFinitePoints(entry.points).filter(
    (point) => point.err !== undefined && Number.isFinite(point.err) && point.err > 0,
  );
  if (bandPoints.length === 0) {
    return [];
  }
  const energies = bandPoints.map((point) => point.energy);
  const lower = bandPoints.map((point) => scaleY(point.value - (point.err ?? 0)));
  const upper = bandPoints.map((point) => scaleY(point.value + (point.err ?? 0)));
  return [
    {
      x: energies,
      y: lower,
      type: "scatter",
      mode: "lines",
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
    },
    {
      x: energies,
      y: upper,
      type: "scatter",
      mode: "lines",
      fill: "tonexty",
      fillcolor: hexToRgba(entry.color, UNCERTAINTY_FILL_OPACITY),
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
    },
  ];
}

function buildLineTrace(entry: ChartSeries, scaleY: (value: number) => number): Data {
  const finite = sortedFinitePoints(entry.points);
  return {
    x: finite.map((point) => point.energy),
    y: finite.map((point) => scaleY(point.value)),
    type: "scatter",
    mode: "lines",
    name: entry.label,
    line: { color: entry.color, width: 1.8 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

/**
 * Multi-series spectrum chart with optional uncertainty bands and a draggable in-plot legend.
 */
export function SpectrumChart({
  series,
  yLabel = "OD",
  yScale = "linear",
  yDisplayMode,
  height = 320,
  className = "",
  emptyMessage = "Select spectra to plot.",
  loading = false,
}: SpectrumChartProps) {
  const plotHostRef = useRef<HTMLDivElement>(null);
  const graphDivRef = useRef<HTMLElement | null>(null);
  const graphReadyRef = useRef(false);
  const [graphReady, setGraphReady] = useState(false);
  const [hoverEnergy, setHoverEnergy] = useState<number | undefined>(undefined);
  const [tooltipAnchor, setTooltipAnchor] = useState<{ x: number; y: number } | null>(null);
  const [plotLayoutTick, setPlotLayoutTick] = useState(0);
  const merged = useMemo(() => mergeSeriesForChart(series), [series]);
  const yValueKind = useMemo(() => dominantValueKind(series), [series]);
  const yValues = useMemo(() => collectYValues(merged, series), [merged, series]);
  const useLogYAxis = useMemo(() => {
    if (yScale !== "log" || yValueKind !== "signal") {
      return false;
    }
    const finite = yValues.filter((value) => Number.isFinite(value));
    if (finite.length === 0) {
      return false;
    }
    return finite.every((value) => value > 0);
  }, [yScale, yValueKind, yValues]);
  const yScalePlan = useMemo(
    () => (useLogYAxis ? { scale: 1, exponent: 0, titleSuffix: "", applyScale: false } : yAxisScaleFromValues(yValues, yValueKind)),
    [useLogYAxis, yValueKind, yValues],
  );
  const chartRows = useMemo(
    () => scaleChartRows(merged, series, yScalePlan.scale),
    [merged, series, yScalePlan.scale],
  );
  const xPlan = useMemo(
    () => planLinearTicks(...collectXExtents(chartRows)),
    [chartRows],
  );
  const yPlan = useMemo(
    () => (useLogYAxis ? null : planLinearTicks(...collectYExtents(chartRows, series), 5)),
    [chartRows, series, useLogYAxis],
  );
  const xDomain = xPlan.domain;
  const yDomain = yPlan?.domain;
  const yAxisTitle = useMemo(
    () => formatYAxisTitle(yLabel, yScalePlan.titleSuffix),
    [yLabel, yScalePlan.titleSuffix],
  );
  const seriesById = useMemo(() => new Map(series.map((entry) => [entry.id, entry])), [series]);
  const chartYValue = useCallback(
    (rawValue: number) => (yScalePlan.applyScale ? rawValue / yScalePlan.scale : rawValue),
    [yScalePlan.applyScale, yScalePlan.scale],
  );
  const scaleY = useCallback(
    (rawValue: number) => chartYValue(rawValue),
    [chartYValue],
  );

  const plotData = useMemo(() => {
    const traces: Data[] = [];
    for (const entry of series) {
      traces.push(...buildBandTraces(entry, scaleY));
      traces.push(buildLineTrace(entry, scaleY));
    }
    return traces;
  }, [series, scaleY]);

  const staticPlotLayout = useMemo((): Partial<Layout> => {
    const primaryAxisStyle = buildPrimaryAxisStyle();
    const xTicks = plotlyPublicationLinearAxisTicks(xPlan, {
      tickformat: spectrumEnergyTickformat(),
      axisStroke: AXIS_STROKE,
    });
    const yTicks = useLogYAxis
      ? {}
      : plotlyPublicationLinearAxisTicks(yPlan!, {
          tickformat: spectrumValueTickformat(yValueKind, yScalePlan.applyScale),
          axisStroke: AXIS_STROKE,
        });
    return {
      autosize: true,
      uirevision: "spectrum-chart",
      margin: CHART_MARGIN,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      hovermode: false,
      showlegend: false,
      xaxis: {
        ...primaryAxisStyle,
        domain: [0, 1],
        range: xDomain,
        ...xTicks,
        title: { text: "Energy (eV)", standoff: 12, font: { size: 12, color: AXIS_STROKE } },
      },
      yaxis: {
        ...primaryAxisStyle,
        domain: [0, 1],
        type: useLogYAxis ? ("log" as const) : ("linear" as const),
        ...(useLogYAxis ? { autorange: true } : { range: yDomain, ...yTicks }),
        title: {
          text: yAxisTitle,
          font: { size: 12, color: AXIS_STROKE },
        },
      },
    };
  }, [xDomain, xPlan, yAxisTitle, yDomain, yPlan, yScalePlan.applyScale, yValueKind, useLogYAxis]);

  const captureGraphDiv = useCallback((_: unknown, graphDiv: HTMLElement) => {
    graphDivRef.current = graphDiv;
    if (!graphReadyRef.current) {
      graphReadyRef.current = true;
      setGraphReady(true);
    }
  }, []);

  const clearHover = useCallback(() => {
    setHoverEnergy(undefined);
    setTooltipAnchor(null);
  }, []);

  useEffect(() => {
    const graphDiv = graphDivRef.current;
    const host = plotHostRef.current;
    if (!graphDiv || !host) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const energy = energyFromPlotlyClientX(graphDiv, event.clientX);
      if (energy === undefined || !Number.isFinite(energy)) {
        return;
      }
      setHoverEnergy(energy);
      const hostRect = host.getBoundingClientRect();
      setTooltipAnchor({
        x: event.clientX - hostRect.left,
        y: event.clientY - hostRect.top,
      });
    };
    const bumpLayout = () => setPlotLayoutTick((tick) => tick + 1);
    graphDiv.addEventListener("mousemove", handleMove);
    graphDiv.addEventListener("mouseleave", clearHover);
    graphDiv.addEventListener("plotly_relayout", bumpLayout);
    graphDiv.addEventListener("plotly_relayouting", bumpLayout);
    return () => {
      graphDiv.removeEventListener("mousemove", handleMove);
      graphDiv.removeEventListener("mouseleave", clearHover);
      graphDiv.removeEventListener("plotly_relayout", bumpLayout);
      graphDiv.removeEventListener("plotly_relayouting", bumpLayout);
    };
  }, [clearHover, graphReady]);

  if (loading) {
    return (
      <div
        className={`flex items-center justify-center rounded-lg border border-dashed border-zinc-300 text-sm text-zinc-500 ${className}`.trim()}
        style={{ height }}
      >
        Computing spectra...
      </div>
    );
  }
  if (merged.length === 0) {
    return (
      <div
        className={`flex items-center justify-center rounded-lg border border-dashed border-zinc-300 text-sm text-zinc-500 ${className}`.trim()}
        style={{ height }}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      className={`flex min-h-0 flex-col rounded-lg border border-zinc-200 bg-white p-3 ${className}`.trim()}
      style={{ height }}
    >
      <div ref={plotHostRef} className="spectrum-plot-host relative min-h-0 w-full flex-1">
        <Plot
          data={plotData}
          layout={staticPlotLayout}
          config={PLOT_CONFIG}
          useResizeHandler
          className="h-full w-full"
          style={{ width: "100%", height: "100%" }}
          onInitialized={captureGraphDiv}
          onUpdate={captureGraphDiv}
        />
        {hoverEnergy != null && Number.isFinite(hoverEnergy) ? (
          <SpectrumHoverOverlay
            graphDivRef={graphDivRef}
            plotHostRef={plotHostRef}
            hoverEnergy={hoverEnergy}
            series={series}
            scaleY={scaleY}
            layoutTick={plotLayoutTick}
          />
        ) : null}
        {hoverEnergy != null && tooltipAnchor ? (
          <SpectrumTooltip
            hoverEnergy={hoverEnergy}
            anchor={tooltipAnchor}
            plotHostRef={plotHostRef}
            series={series}
            seriesById={seriesById}
            yScale={yScale}
            yDisplayMode={yDisplayMode}
          />
        ) : null}
        <DraggableLegend
          plotHostRef={plotHostRef}
          items={series.map((entry) => ({
            id: entry.id,
            label: entry.label,
            color: entry.color,
          }))}
        />
      </div>
    </div>
  );
}

function SpectrumHoverOverlay({
  graphDivRef,
  plotHostRef,
  hoverEnergy,
  series,
  scaleY,
  layoutTick,
}: {
  graphDivRef: RefObject<HTMLElement | null>;
  plotHostRef: RefObject<HTMLDivElement | null>;
  hoverEnergy: number;
  series: ChartSeries[];
  scaleY: (value: number) => number;
  layoutTick: number;
}) {
  void layoutTick;
  const graphDiv = graphDivRef.current;
  const host = plotHostRef.current;
  if (!graphDiv || !host) {
    return null;
  }
  const plotArea = plotHostAreaFromPlotly(graphDiv, host);
  if (!plotArea) {
    return null;
  }
  const markers = series
    .map((entry) => {
      const sample = interpolateChartPoint(entry.points, hoverEnergy);
      if (!sample || !Number.isFinite(sample.value)) {
        return null;
      }
      const traceY = scaleY(sample.value);
      const point = plotHostPointFromPlotlyData(graphDiv, host, hoverEnergy, traceY);
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
        return null;
      }
      return { id: entry.id, color: entry.color, x: point.x, y: point.y };
    })
    .filter((marker): marker is NonNullable<typeof marker> => marker !== null);
  const guideX = markers[0]?.x ?? plotHostXFromPlotlyEnergy(graphDiv, host, hoverEnergy);
  if (guideX === undefined || !Number.isFinite(guideX)) {
    return null;
  }
  return (
    <div className="pointer-events-none absolute inset-0 z-10" aria-hidden>
      <div
        className="absolute w-px"
        style={{
          left: guideX,
          top: plotArea.top,
          height: plotArea.height,
          backgroundColor: HOVER_GUIDE_STROKE,
        }}
      />
      {markers.map((marker) => (
        <span
          key={marker.id}
          className="absolute box-border h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white"
          style={{ left: marker.x, top: marker.y, backgroundColor: marker.color }}
        />
      ))}
    </div>
  );
}

function SpectrumTooltip({
  hoverEnergy,
  anchor,
  plotHostRef,
  series,
  seriesById,
  yScale,
  yDisplayMode,
}: {
  hoverEnergy: number;
  anchor: { x: number; y: number };
  plotHostRef: RefObject<HTMLDivElement | null>;
  series: ChartSeries[];
  seriesById: Map<string, ChartSeries>;
  yScale: PlotScaleMode;
  yDisplayMode?: IngestionYDisplayMode;
}) {
  const energy = hoverEnergy;
  if (!Number.isFinite(energy) || series.length === 0) {
    return null;
  }
  const rows = series
    .map((entry) => {
      const sample = interpolateChartPoint(entry.points, energy);
      if (!sample || !Number.isFinite(sample.value)) {
        return null;
      }
      const meta = seriesById.get(entry.id);
      const kind = meta?.valueKind ?? "signal";
      return {
        id: entry.id,
        name: entry.label,
        color: entry.color,
        kind,
        value: sample.value,
        err: sample.err,
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);
  if (rows.length === 0) {
    return null;
  }
  const host = plotHostRef.current;
  const hostWidth = host?.clientWidth ?? 0;
  const hostHeight = host?.clientHeight ?? 0;
  const offsetX = 14;
  const offsetY = 14;
  const tooltipWidth = 220;
  const tooltipHeight = 48 + rows.length * 18;
  const left =
    anchor.x + offsetX + tooltipWidth > hostWidth
      ? Math.max(8, anchor.x - tooltipWidth - offsetX)
      : anchor.x + offsetX;
  const top =
    anchor.y + offsetY + tooltipHeight > hostHeight
      ? Math.max(8, anchor.y - tooltipHeight - offsetY)
      : anchor.y + offsetY;
  const energyLabel = formatEnergy(energy);
  return (
    <div
      className="pointer-events-none absolute z-20 rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs shadow-md"
      style={{ left, top }}
    >
      <p className="mb-1 font-medium text-zinc-900">{energyLabel} eV</p>
      <ul className="space-y-0.5">
        {rows.map((entry) => (
          <li key={entry.id} className="flex items-center gap-2 text-zinc-700">
            <span
              className="inline-block h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="font-medium">{entry.name}:</span>
            <span className="tabular-nums">
              {formatTooltipValueWithErr(entry.value, entry.err, entry.kind, {
                yScale,
                yDisplayMode,
              })}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function formatEnergy(eV: number): string {
  if (!Number.isFinite(eV)) {
    return "—";
  }
  const rounded = Math.round(eV * 10) / 10;
  if (Math.abs(rounded - Math.round(rounded)) < 1e-9) {
    return String(Math.round(rounded));
  }
  return rounded.toFixed(1);
}

export {
  formatTooltipValue as formatValue,
  formatTooltipValueWithErr as formatValueWithErr,
} from "@/lib/spectrum-chart-data";
