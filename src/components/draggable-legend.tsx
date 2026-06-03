"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";

import { measurePlotLayout } from "@/lib/spectrum-chart-data";

export type LegendItem = {
  id: string;
  label: string;
  color: string;
};

type LegendPosition = {
  x: number;
  y: number;
};

type DraggableLegendProps = {
  items: LegendItem[];
  plotHostRef: RefObject<HTMLDivElement | null>;
  className?: string;
};

const LEGEND_INSET = 8;

function clampLegendPosition(
  position: LegendPosition,
  plotLeft: number,
  plotTop: number,
  plotWidth: number,
  plotHeight: number,
  legendWidth: number,
  legendHeight: number,
): LegendPosition {
  const minX = plotLeft + LEGEND_INSET;
  const minY = plotTop + LEGEND_INSET;
  const maxX = Math.max(minX, plotLeft + plotWidth - legendWidth - LEGEND_INSET);
  const maxY = Math.max(minY, plotTop + plotHeight - legendHeight - LEGEND_INSET);
  return {
    x: Math.min(Math.max(position.x, minX), maxX),
    y: Math.min(Math.max(position.y, minY), maxY),
  };
}

/**
 * Compact, in-plot legend with pointer drag repositioning clamped to the Cartesian grid bounds.
 */
export function DraggableLegend({ items, plotHostRef, className = "" }: DraggableLegendProps) {
  const legendRef = useRef<HTMLDivElement>(null);
  const dragOffsetRef = useRef<{ x: number; y: number } | null>(null);
  const userMovedRef = useRef(false);
  const [position, setPosition] = useState<LegendPosition | null>(null);

  const syncPosition = useCallback(() => {
    const host = plotHostRef.current;
    const legend = legendRef.current;
    if (!host || !legend) {
      return;
    }
    const plot = measurePlotLayout(host);
    if (!plot) {
      return;
    }
    const legendWidth = legend.offsetWidth;
    const legendHeight = legend.offsetHeight;
    if (legendWidth <= 0 || legendHeight <= 0) {
      return;
    }
    const defaultPosition: LegendPosition = {
      x: plot.left + plot.width - legendWidth - LEGEND_INSET,
      y: plot.top + LEGEND_INSET,
    };
    setPosition((current) => {
      const base = userMovedRef.current && current ? current : defaultPosition;
      return clampLegendPosition(
        base,
        plot.left,
        plot.top,
        plot.width,
        plot.height,
        legendWidth,
        legendHeight,
      );
    });
  }, [plotHostRef]);

  useEffect(() => {
    let cancelled = false;
    let frameCount = 0;
    const scheduleSync = () => {
      if (cancelled) {
        return;
      }
      syncPosition();
      frameCount += 1;
      if (frameCount < 16) {
        requestAnimationFrame(scheduleSync);
      }
    };
    scheduleSync();
    const host = plotHostRef.current;
    if (!host) {
      return () => {
        cancelled = true;
      };
    }
    const resizeObserver = new ResizeObserver(() => {
      syncPosition();
    });
    resizeObserver.observe(host);
    const mutationObserver = new MutationObserver(() => {
      syncPosition();
    });
    mutationObserver.observe(host, { childList: true, subtree: true });
    return () => {
      cancelled = true;
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [items, plotHostRef, syncPosition]);

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || !position) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      userMovedRef.current = true;
      dragOffsetRef.current = {
        x: event.clientX - event.currentTarget.getBoundingClientRect().left,
        y: event.clientY - event.currentTarget.getBoundingClientRect().top,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [position],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const dragOffset = dragOffsetRef.current;
      const host = plotHostRef.current;
      const legend = legendRef.current;
      if (!dragOffset || !host || !legend) {
        return;
      }
      event.stopPropagation();
      const plot = measurePlotLayout(host);
      if (!plot) {
        return;
      }
      const hostRect = host.getBoundingClientRect();
      const next: LegendPosition = {
        x: event.clientX - hostRect.left - dragOffset.x,
        y: event.clientY - hostRect.top - dragOffset.y,
      };
      setPosition(
        clampLegendPosition(
          next,
          plot.left,
          plot.top,
          plot.width,
          plot.height,
          legend.offsetWidth,
          legend.offsetHeight,
        ),
      );
    },
    [plotHostRef],
  );

  const handlePointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    dragOffsetRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      ref={legendRef}
      role="list"
      aria-label="Spectrum legend"
      className={`pointer-events-auto absolute z-10 cursor-grab select-none rounded-md border border-zinc-200 bg-white/90 px-2.5 py-1.5 text-[11px] leading-4 shadow-sm active:cursor-grabbing ${className}`.trim()}
      style={
        position
          ? { left: position.x, top: position.y }
          : { visibility: "hidden", left: 0, top: 0 }
      }
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <ul className="space-y-0.5">
        {items.map((entry) => (
          <li key={entry.id} className="flex items-center gap-2 text-zinc-800" role="listitem">
            <span
              className="inline-block h-0.5 w-3 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color }}
              aria-hidden
            />
            <span className="whitespace-nowrap">{entry.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
