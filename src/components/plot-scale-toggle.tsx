"use client";

import type { PlotScaleMode } from "@/lib/stxm/plot-scale";

export type PlotScaleOption = {
  value: PlotScaleMode;
  label: string;
};

export const PLOT_SCALE_OPTIONS: PlotScaleOption[] = [
  { value: "linear", label: "Linear" },
  { value: "log", label: "Log" },
];

type PlotScaleToggleProps = {
  legend: string;
  ariaLabel: string;
  value: PlotScaleMode;
  onChange: (mode: PlotScaleMode) => void;
  disabled?: boolean;
  logDisabled?: boolean;
  logDisabledTitle?: string;
};

/**
 * Segmented Linear / Log control matching ingestion toolbar radiogroup styling.
 */
export function PlotScaleToggle({
  legend,
  ariaLabel,
  value,
  onChange,
  disabled = false,
  logDisabled = false,
  logDisabledTitle,
}: PlotScaleToggleProps) {
  return (
    <fieldset className="min-w-0" disabled={disabled}>
      <legend className="sr-only">{legend}</legend>
      <div
        className="inline-flex max-w-full flex-wrap rounded-lg border border-zinc-200 bg-zinc-50 p-0.5"
        role="radiogroup"
        aria-label={ariaLabel}
      >
        {PLOT_SCALE_OPTIONS.map((option) => {
          const selected = value === option.value;
          const optionDisabled = disabled || (option.value === "log" && logDisabled);
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              title={option.value === "log" && logDisabled ? logDisabledTitle : undefined}
              disabled={optionDisabled}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors sm:text-sm ${
                selected
                  ? "bg-white text-zinc-900 shadow-sm"
                  : "text-zinc-600 hover:text-zinc-900 disabled:opacity-50"
              }`}
              onClick={() => onChange(option.value)}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
