"use client";

import { WEIGHTING_MODE_OPTIONS, type WeightingMode } from "@/lib/stxm/estimators";
import {
  INGESTION_Y_MODE_OPTIONS,
  ingestionModeAllowsLogYScale,
} from "@/lib/stxm/ingestion-display";
import type { IngestionYDisplayMode } from "@/lib/stxm-types";
import type { PlotScaleMode } from "@/lib/stxm/plot-scale";
import { PlotScaleToggle } from "@/components/plot-scale-toggle";

export type IngestionToolbarProps = {
  scanLabel?: string;
  weightingMode: WeightingMode;
  onWeightingModeChange: (mode: WeightingMode) => void;
  yDisplayMode: IngestionYDisplayMode;
  onYDisplayModeChange: (mode: IngestionYDisplayMode) => void;
  plotScaleMode: PlotScaleMode;
  onPlotScaleModeChange: (mode: PlotScaleMode) => void;
  chemicalFormula: string;
  onChemicalFormulaChange: (value: string) => void;
  bareAtomFitOffset: boolean;
  onBareAtomFitOffsetChange: (value: boolean) => void;
  onRecompute: () => void;
  disabled?: boolean;
};

/**
 * Compact analysis controls for the ingestion tab: weighting, Y display mode, formula, and recompute.
 */
export function IngestionToolbar({
  scanLabel,
  weightingMode,
  onWeightingModeChange,
  yDisplayMode,
  onYDisplayModeChange,
  plotScaleMode,
  onPlotScaleModeChange,
  chemicalFormula,
  onChemicalFormulaChange,
  bareAtomFitOffset,
  onBareAtomFitOffsetChange,
  onRecompute,
  disabled = false,
}: IngestionToolbarProps) {
  const showFormula = yDisplayMode === "mass_absorption_cxro";
  const plotLogAllowed = ingestionModeAllowsLogYScale(yDisplayMode);

  return (
    <div
      className="flex flex-col gap-4 border-b border-zinc-200 pb-3 md:flex-row md:flex-wrap md:items-center md:justify-between"
      role="toolbar"
      aria-label="Ingestion analysis controls"
    >
      <div className="min-w-0 shrink-0">
        {scanLabel ? (
          <p className="truncate font-mono text-sm font-medium text-zinc-800" title={scanLabel}>
            {scanLabel}
          </p>
        ) : (
          <p className="text-sm text-zinc-500">No line scan loaded</p>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-center sm:justify-end">
        <fieldset className="min-w-0" disabled={disabled}>
          <legend className="sr-only">Region weighting mode</legend>
          <div
            className="inline-flex max-w-full flex-wrap rounded-lg border border-zinc-200 bg-zinc-50 p-0.5"
            role="radiogroup"
            aria-label="Weighting"
          >
            {WEIGHTING_MODE_OPTIONS.map((option) => {
              const selected = weightingMode === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  disabled={disabled}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors sm:text-sm ${
                    selected
                      ? "bg-white text-zinc-900 shadow-sm"
                      : "text-zinc-600 hover:text-zinc-900 disabled:opacity-50"
                  }`}
                  onClick={() => onWeightingModeChange(option.value)}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
        </fieldset>
        <fieldset className="min-w-0" disabled={disabled}>
          <legend className="sr-only">Spectrum Y display mode</legend>
          <div
            className="inline-flex max-w-full flex-wrap rounded-lg border border-zinc-200 bg-zinc-50 p-0.5"
            role="radiogroup"
            aria-label="Spectrum display"
          >
            {INGESTION_Y_MODE_OPTIONS.map((option) => {
              const selected = yDisplayMode === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  title={option.label}
                  disabled={disabled}
                  className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors sm:px-3 sm:text-sm ${
                    selected
                      ? "bg-white text-zinc-900 shadow-sm"
                      : "text-zinc-600 hover:text-zinc-900 disabled:opacity-50"
                  }`}
                  onClick={() => onYDisplayModeChange(option.value)}
                >
                  {option.shortLabel}
                </button>
              );
            })}
          </div>
        </fieldset>
        <PlotScaleToggle
          legend="Plot scale"
          ariaLabel="Plot scale"
          value={plotScaleMode}
          onChange={onPlotScaleModeChange}
          disabled={disabled}
          logDisabled={!plotLogAllowed}
          logDisabledTitle="Log scale applies to Signal and 1/Signal display modes"
        />
        {showFormula ? (
          <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center">
            <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-zinc-600">
              Formula
              <input
                type="text"
                className="w-full min-w-[8rem] rounded-md border border-zinc-300 px-2 py-1.5 font-mono text-sm disabled:opacity-50 sm:w-36"
                placeholder="e.g. C8H8"
                value={chemicalFormula}
                disabled={disabled}
                onChange={(event) => onChemicalFormulaChange(event.target.value)}
              />
            </label>
            <label
              className={`flex cursor-pointer items-center gap-2 text-sm text-zinc-800 ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
            >
              <input
                type="checkbox"
                className="rounded border-zinc-300"
                checked={bareAtomFitOffset}
                disabled={disabled}
                onChange={(event) => onBareAtomFitOffsetChange(event.target.checked)}
              />
              Bare-atom fit offset
            </label>
          </div>
        ) : null}
        <button
          type="button"
          className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={disabled}
          onClick={onRecompute}
        >
          Recompute spectra
        </button>
      </div>
    </div>
  );
}
