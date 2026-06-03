"use client";

import { memo, useCallback, useMemo, useState } from "react";

import {
  SCAN_CATEGORY_ORDER,
  ScanTypeBadge,
  scanCategoryLabel,
} from "@/components/scan-type-badge";
import type { ScanCatalogEntry, ScanCategory } from "@/lib/stxm-types";

type ExperimentFileBrowserProps = {
  entries: ScanCatalogEntry[];
  selectedBasename: string;
  loading: boolean;
  error: string;
  onSelect: (entry: ScanCatalogEntry) => void;
  className?: string;
};

function thumbnailSrc(entry: ScanCatalogEntry): string | null {
  if (!entry.thumbnail_png_base64) {
    return null;
  }
  return `data:image/png;base64,${entry.thumbnail_png_base64}`;
}

function formatEnergyEv(value: number, compact: boolean): string {
  if (compact) {
    return Number.isInteger(value) ? `${value}` : value.toFixed(0);
  }
  return value.toFixed(1);
}

function formatEnergyLine(entry: ScanCatalogEntry): string | null {
  const compact = entry.category === "line_scan";

  if (entry.energy_eV != null) {
    return `${formatEnergyEv(entry.energy_eV, compact)} eV`;
  }
  if (entry.energy_min_eV != null && entry.energy_max_eV != null) {
    const lo = entry.energy_min_eV;
    const hi = entry.energy_max_eV;
    if (Math.abs(lo - hi) < 0.05) {
      return `${formatEnergyEv(lo, compact)} eV`;
    }
    return `${formatEnergyEv(lo, compact)}–${formatEnergyEv(hi, compact)} eV`;
  }
  return null;
}

function energyLineClassName(category: ScanCategory): string {
  if (category === "image_scan") {
    return "w-full truncate text-[11px] font-medium text-zinc-800";
  }
  if (category === "line_scan") {
    return "w-full truncate text-[10px] text-zinc-500";
  }
  return "w-full truncate text-[11px] text-zinc-600";
}

function formatDimensions(entry: ScanCatalogEntry): string | null {
  if (!entry.shape) {
    return null;
  }
  return `${entry.shape[0]} x ${entry.shape[1]}`;
}

function groupEntries(entries: ScanCatalogEntry[]): Map<ScanCategory, ScanCatalogEntry[]> {
  const grouped = new Map<ScanCategory, ScanCatalogEntry[]>();
  for (const category of SCAN_CATEGORY_ORDER) {
    grouped.set(category, []);
  }
  for (const entry of entries) {
    const bucket = grouped.get(entry.category) ?? grouped.get("other");
    if (bucket) {
      bucket.push(entry);
    }
  }
  return grouped;
}

type ScanCardProps = {
  entry: ScanCatalogEntry;
  selected: boolean;
  cardClassName?: string;
  onSelect: (entry: ScanCatalogEntry) => void;
};

const ScanCard = memo(function ScanCard({
  entry,
  selected,
  cardClassName = "",
  onSelect,
}: ScanCardProps) {
  const src = useMemo(() => thumbnailSrc(entry), [entry]);
  const energyLine = useMemo(() => formatEnergyLine(entry), [entry]);
  const dimensions = useMemo(() => formatDimensions(entry), [entry]);
  const isLineScan = entry.category === "line_scan";
  const energyClassName = energyLineClassName(entry.category);

  const handleClick = useCallback(() => {
    onSelect(entry);
  }, [entry, onSelect]);

  return (
    <button
      type="button"
      className={`group flex shrink-0 flex-col items-center gap-1.5 rounded-lg border p-2 text-left transition-colors ${cardClassName} ${
        selected
          ? "border-sky-500 bg-sky-50 ring-2 ring-sky-200"
          : "border-zinc-200 bg-zinc-50 hover:border-zinc-300 hover:bg-white"
      }`}
      onClick={handleClick}
      title={`${entry.basename}\n${entry.scan_type}${energyLine ? `\n${energyLine}` : ""}`}
    >
      <div className="relative flex h-24 w-full items-center justify-center overflow-hidden rounded-md bg-zinc-900">
        {src ? (
          <div
            role="img"
            aria-label={entry.basename}
            className="h-full w-full bg-contain bg-center bg-no-repeat"
            style={{ backgroundImage: `url(${src})` }}
          />
        ) : (
          <span className="px-2 text-center text-[10px] text-zinc-400">No preview</span>
        )}
        <ScanTypeBadge
          category={entry.category}
          className="absolute bottom-1 right-1 scale-90 opacity-90"
        />
      </div>
      <span className="w-full truncate font-mono text-[11px] leading-tight text-zinc-800">
        {entry.basename}
      </span>
      {!isLineScan && energyLine ? (
        <span className={energyClassName}>{energyLine}</span>
      ) : null}
      {dimensions ? (
        <span className="text-[10px] text-zinc-400">{dimensions}</span>
      ) : null}
      {isLineScan && energyLine ? (
        <span className={energyClassName}>{energyLine}</span>
      ) : null}
    </button>
  );
});

export const ExperimentFileBrowser = memo(function ExperimentFileBrowser({
  entries,
  selectedBasename,
  loading,
  error,
  onSelect,
  className = "",
}: ExperimentFileBrowserProps) {
  const grouped = useMemo(() => groupEntries(entries), [entries]);
  const [expandedGroups, setExpandedGroups] = useState<Partial<Record<ScanCategory, boolean>>>({});

  const toggleGroup = useCallback((category: ScanCategory) => {
    setExpandedGroups((current) => ({
      ...current,
      [category]: !current[category],
    }));
  }, []);

  if (loading && entries.length === 0) {
    return (
      <section
        className={`rounded-xl border border-zinc-200 bg-white p-6 shadow-sm ${className}`.trim()}
      >
        <p className="text-sm text-zinc-500">Loading experiment scans...</p>
      </section>
    );
  }

  if (error && entries.length === 0) {
    return (
      <section
        className={`rounded-xl border border-red-200 bg-red-50 p-6 shadow-sm ${className}`.trim()}
      >
        <p className="text-sm text-red-700" role="alert">
          {error}
        </p>
      </section>
    );
  }

  if (entries.length === 0) {
    return (
      <section
        className={`rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-6 shadow-sm ${className}`.trim()}
      >
        <p className="text-sm text-zinc-500">No .hdr scan files found in this experiment.</p>
      </section>
    );
  }

  return (
    <section
      className={`rounded-xl border border-zinc-200 bg-white p-4 shadow-sm ${className}`.trim()}
      aria-busy={loading || undefined}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Experiment files
        </h2>
        <p className="text-xs text-zinc-500" aria-live="polite">
          {entries.length} scans
          {loading ? " (refreshing...)" : ""}
        </p>
      </div>
      <div className="space-y-6">
        {SCAN_CATEGORY_ORDER.map((category) => {
          const sectionEntries = grouped.get(category) ?? [];
          if (sectionEntries.length === 0) {
            return null;
          }
          const expanded = expandedGroups[category] ?? false;
          const groupLabel = scanCategoryLabel(category).toUpperCase();
          return (
            <div key={category}>
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <h3 className="text-xs font-semibold tracking-wide text-zinc-700">{groupLabel}</h3>
                  <span className="text-xs text-zinc-400">{sectionEntries.length}</span>
                </div>
                <button
                  type="button"
                  className="shrink-0 text-xs font-medium text-sky-700 hover:text-sky-900 hover:underline"
                  onClick={() => toggleGroup(category)}
                >
                  {expanded ? "Show Less" : `Show All (${sectionEntries.length})`}
                </button>
              </div>
              {expanded ? (
                <ul className="grid grid-cols-[repeat(auto-fill,minmax(128px,1fr))] gap-3">
                  {sectionEntries.map((entry) => (
                    <li key={entry.hdr_path}>
                      <ScanCard
                        entry={entry}
                        selected={entry.basename === selectedBasename}
                        cardClassName="w-full"
                        onSelect={onSelect}
                      />
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="overflow-x-auto pb-1">
                  <ul className="flex snap-x snap-mandatory gap-3">
                    {sectionEntries.map((entry) => (
                      <li key={entry.hdr_path} className="w-[132px] shrink-0 snap-start">
                        <ScanCard
                          entry={entry}
                          selected={entry.basename === selectedBasename}
                          cardClassName="w-[132px]"
                          onSelect={onSelect}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
});
