"use client";

import { RefreshCw } from "lucide-react";
import { useCallback, useId, useState, type ReactNode } from "react";

import {
  pickParentDirectory,
  resolveBeamtimeSelection,
  type BeamtimeSelection,
} from "@/lib/stxm-client";

type BreadcrumbSegment = {
  label: string;
  title?: string;
  onClick?: () => void;
};

type WorkspaceHeaderProps = {
  parentDir: string;
  breadcrumb: BreadcrumbSegment[];
  workspaceStatus: string;
  parquetFilename: string;
  storeRoot: string;
  parquetCustomized: boolean;
  directoryPickerEnabled?: boolean;
  refreshing?: boolean;
  recentWorkspaces: Array<{ parentDir: string; experiment: string; label: string }>;
  onParentDirChange: (value: string) => void;
  onBeamtimePicked?: (selection: BeamtimeSelection) => void;
  onParquetFilenameChange: (value: string) => void;
  onParquetCustomizedChange: (value: boolean) => void;
  onStoreRootChange: (value: string) => void;
  onRefresh: () => void;
  onOpenRecent: (parentDir: string, experiment: string) => void;
  children?: ReactNode;
};

function Cog6ToothIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-4 w-4"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M7.83922 1.80388C7.93271 1.33646 8.34312 1 8.81981 1H11.1802C11.6569 1 12.0673 1.33646 12.1608 1.80388L12.4913 3.45629C13.1956 3.72458 13.8454 4.10332 14.4196 4.57133L16.0179 4.03065C16.4694 3.8779 16.966 4.06509 17.2043 4.47791L18.3845 6.52207C18.6229 6.93489 18.5367 7.45855 18.1786 7.77322L16.9119 8.88645C16.9699 9.24909 17 9.62103 17 10C17 10.379 16.9699 10.7509 16.9119 11.1135L18.1786 12.2268C18.5367 12.5414 18.6229 13.0651 18.3845 13.4779L17.2043 15.5221C16.966 15.9349 16.4694 16.1221 16.0179 15.9693L14.4196 15.4287C13.8454 15.8967 13.1956 16.2754 12.4913 16.5437L12.1608 18.1961C12.0673 18.6635 11.6569 19 11.1802 19H8.81981C8.34312 19 7.93271 18.6635 7.83922 18.1961L7.50874 16.5437C6.80443 16.2754 6.1546 15.8967 5.58043 15.4287L3.98214 15.9694C3.5306 16.1221 3.03401 15.9349 2.79567 15.5221L1.61547 13.4779C1.37713 13.0651 1.4633 12.5415 1.82136 12.2268L3.08808 11.1135C3.03012 10.7509 3 10.379 3 10C3 9.62103 3.03012 9.2491 3.08808 8.88647L1.82136 7.77324C1.46331 7.45857 1.37713 6.93491 1.61547 6.52209L2.79567 4.47793C3.03401 4.06511 3.5306 3.87791 3.98214 4.03066L5.58042 4.57134C6.15459 4.10332 6.80442 3.72459 7.50874 3.45629L7.83922 1.80388ZM10 13C11.6569 13 13 11.6569 13 10C13 8.34315 11.6569 7 10 7C8.34315 7 7 8.34315 7 10C7 11.6569 8.34315 13 10 13Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/**
 * Compact workspace shell: beamtime picker, breadcrumb navigation, and advanced export paths.
 */
export function WorkspaceHeader({
  parentDir,
  breadcrumb,
  workspaceStatus,
  parquetFilename,
  storeRoot,
  parquetCustomized,
  directoryPickerEnabled = false,
  refreshing = false,
  recentWorkspaces,
  onParentDirChange,
  onBeamtimePicked,
  onParquetFilenameChange,
  onParquetCustomizedChange,
  onStoreRootChange,
  onRefresh,
  onOpenRecent,
  children,
}: WorkspaceHeaderProps) {
  const advancedId = useId();
  const [pickError, setPickError] = useState("");
  const [picking, setPicking] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [manualPathOpen, setManualPathOpen] = useState(false);

  const hasParent = parentDir.trim().length > 0;
  const openLabel = hasParent ? "Change location" : "Open data location";

  const handleBrowse = useCallback(async () => {
    setPickError("");
    setPicking(true);
    try {
      const result = await pickParentDirectory();
      if (!result.cancelled) {
        const selection = resolveBeamtimeSelection(result.path);
        if (onBeamtimePicked) {
          onBeamtimePicked(selection);
        } else {
          onParentDirChange(selection.parentDir);
        }
      }
    } catch (error) {
      setPickError(error instanceof Error ? error.message : "Failed to pick directory");
    } finally {
      setPicking(false);
    }
  }, [onBeamtimePicked, onParentDirChange]);

  return (
    <header className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            {directoryPickerEnabled ? (
              <button
                type="button"
                className="shrink-0 rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={() => void handleBrowse()}
                disabled={picking}
              >
                {picking ? "Opening..." : openLabel}
              </button>
            ) : null}
            <nav
              aria-label="Workspace location"
              className="flex min-w-0 flex-1 items-center gap-1 text-sm text-zinc-700"
            >
              {breadcrumb.length === 0 ? (
                <span className="text-zinc-500">Open beamtime root</span>
              ) : (
                breadcrumb.map((segment, index) => (
                  <span key={`${segment.label}-${index}`} className="flex min-w-0 items-center gap-1">
                    {index > 0 ? <span className="shrink-0 text-zinc-400">/</span> : null}
                    {segment.onClick ? (
                      <button
                        type="button"
                        className="max-w-[12rem] truncate rounded px-1 py-0.5 font-medium text-sky-800 hover:bg-sky-50 hover:underline sm:max-w-[16rem]"
                        title={segment.title ?? segment.label}
                        onClick={segment.onClick}
                      >
                        {segment.label}
                      </button>
                    ) : (
                      <span
                        className="max-w-[12rem] truncate sm:max-w-[16rem]"
                        title={segment.title ?? segment.label}
                      >
                        {segment.label}
                      </span>
                    )}
                  </span>
                ))
              )}
            </nav>
          </div>
          {pickError ? (
            <p className="text-xs text-red-600" role="alert">
              {pickError}
            </p>
          ) : null}
          {directoryPickerEnabled && !hasParent ? (
            <p className="text-xs text-zinc-500">
              Select your beamtime folder (e.g. BL5321 (New STXM))
            </p>
          ) : null}
          <p className="text-sm text-zinc-600">{workspaceStatus}</p>
          {recentWorkspaces.length > 0 ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                Recent
              </span>
              {recentWorkspaces.map((item) => (
                <button
                  key={`${item.parentDir}:${item.experiment}`}
                  type="button"
                  className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 hover:border-zinc-300 hover:bg-white"
                  title={item.parentDir}
                  onClick={() => onOpenRecent(item.parentDir, item.experiment)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-700 hover:bg-zinc-50 disabled:opacity-60"
            onClick={() => onRefresh()}
            disabled={refreshing || !hasParent}
            title="Reload experiments and scans"
          >
            <RefreshCw className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">{refreshing ? "Reloading..." : "Reload"}</span>
          </button>
          <button
            type="button"
            className={`inline-flex items-center justify-center rounded-md border px-3 py-2 text-sm ${
              advancedOpen
                ? "border-sky-300 bg-sky-50 text-sky-900"
                : "border-zinc-300 text-zinc-700 hover:bg-zinc-50"
            }`}
            aria-expanded={advancedOpen}
            aria-controls={advancedId}
            aria-label="Advanced settings"
            title="Advanced export paths"
            onClick={() => setAdvancedOpen((open) => !open)}
          >
            <Cog6ToothIcon />
          </button>
        </div>
      </div>

      {advancedOpen ? (
        <div
          id={advancedId}
          className="mt-4 space-y-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4"
        >
          <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Export paths
          </p>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-600">Parquet filename (inside experiment folder)</span>
            <input
              className="rounded-md border border-zinc-300 bg-white px-3 py-2 font-mono text-sm"
              value={parquetFilename}
              onChange={(event) => {
                onParquetFilenameChange(event.target.value);
                onParquetCustomizedChange(true);
              }}
            />
            {!parquetCustomized ? (
              <span className="text-xs text-zinc-500">
                Default per experiment: experiment.parquet
              </span>
            ) : null}
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-600">Spectrum store root (optional)</span>
            <input
              className="rounded-md border border-zinc-300 bg-white px-3 py-2 font-mono text-sm"
              value={storeRoot}
              onChange={(event) => onStoreRootChange(event.target.value)}
              placeholder="Leave empty to skip partitioned store"
            />
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="text-xs font-medium text-sky-700 hover:underline"
              onClick={() => setManualPathOpen((open) => !open)}
            >
              {manualPathOpen ? "Hide manual beamtime path" : "Edit beamtime path manually"}
            </button>
            {parquetCustomized ? (
              <button
                type="button"
                className="text-xs font-medium text-zinc-600 hover:underline"
                onClick={() => {
                  onParquetFilenameChange("experiment.parquet");
                  onParquetCustomizedChange(false);
                }}
              >
                Reset parquet to default
              </button>
            ) : null}
          </div>
          {manualPathOpen ? (
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-zinc-600">Beamtime root directory</span>
              <input
                className="rounded-md border border-zinc-300 bg-white px-3 py-2 font-mono text-sm"
                value={parentDir}
                title={parentDir}
                onChange={(event) => {
                  setPickError("");
                  onParentDirChange(event.target.value);
                }}
                aria-invalid={pickError.length > 0}
              />
            </label>
          ) : null}
        </div>
      ) : null}

      {children ? <div className="mt-3">{children}</div> : null}
    </header>
  );
}
