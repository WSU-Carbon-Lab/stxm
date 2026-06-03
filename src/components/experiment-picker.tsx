"use client";

import { memo, useCallback } from "react";

export type ExperimentSummary = {
  name: string;
  scanCount?: number;
  lineScanCount?: number;
};

type ExperimentPickerProps = {
  experiments: ExperimentSummary[];
  selected: string;
  loading?: boolean;
  onSelect: (name: string) => void;
  className?: string;
};

function ExperimentTile({
  item,
  selected,
  onSelect,
}: {
  item: ExperimentSummary;
  selected: boolean;
  onSelect: (name: string) => void;
}) {
  const handleClick = useCallback(() => {
    onSelect(item.name);
  }, [item.name, onSelect]);

  const scanLabel =
    item.scanCount != null
      ? `${item.scanCount} scan${item.scanCount === 1 ? "" : "s"}`
      : "Browse scans";

  const lineScanLabel =
    item.lineScanCount != null && item.lineScanCount > 0
      ? `${item.lineScanCount} NEXAFS line`
      : null;

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`flex min-w-[168px] max-w-[220px] flex-col gap-1 rounded-lg border px-3 py-3 text-left transition-colors ${
        selected
          ? "border-sky-500 bg-sky-50 ring-2 ring-sky-200"
          : "border-zinc-200 bg-zinc-50 hover:border-zinc-300 hover:bg-white"
      }`}
    >
      <span className="truncate text-sm font-medium text-zinc-900">{item.name}</span>
      <span className="text-xs text-zinc-500">{scanLabel}</span>
      {lineScanLabel ? <span className="text-xs text-sky-700">{lineScanLabel}</span> : null}
    </button>
  );
}

/**
 * Visual experiment selector replacing the legacy dropdown; shows scan counts when known.
 */
export const ExperimentPicker = memo(function ExperimentPicker({
  experiments,
  selected,
  loading = false,
  onSelect,
  className = "",
}: ExperimentPickerProps) {
  if (loading && experiments.length === 0) {
    return (
      <section
        className={`rounded-xl border border-zinc-200 bg-white p-4 shadow-sm ${className}`.trim()}
      >
        <p className="text-sm text-zinc-500">Loading experiments...</p>
      </section>
    );
  }

  if (experiments.length === 0) {
    return (
      <section
        className={`rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-6 shadow-sm ${className}`.trim()}
      >
        <p className="text-sm text-zinc-500">
          No experiment folders found. Open a beamtime directory that contains dated subfolders.
        </p>
      </section>
    );
  }

  return (
    <section
      className={`rounded-xl border border-zinc-200 bg-white p-4 shadow-sm ${className}`.trim()}
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Experiments
        </h2>
        <p className="text-xs text-zinc-400">{experiments.length} folders</p>
      </div>
      <div className="overflow-x-auto pb-1">
        <ul className="flex snap-x snap-mandatory gap-3">
          {experiments.map((item) => (
            <li key={item.name} className="shrink-0 snap-start">
              <ExperimentTile item={item} selected={item.name === selected} onSelect={onSelect} />
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
});
