"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DataTable } from "@/components/data-table";
import { ExperimentFileBrowser } from "@/components/experiment-file-browser";
import { ExperimentPicker } from "@/components/experiment-picker";
import { IngestionToolbar } from "@/components/ingestion-toolbar";
import { LineScanRegionEditor, type RegionDragTarget } from "@/components/line-scan-region-editor";
import { ScanViewerShell } from "@/components/scan-viewer-shell";
import { SimpleImageViewer } from "@/components/simple-image-viewer";
import { SpectrumChart, type ChartSeries } from "@/components/spectrum-chart";
import { WorkspaceHeader } from "@/components/workspace-header";
import {
  DEFAULT_PARQUET_FILENAME,
  beamtimeBasename,
  deriveExperimentFromHdrPath,
  isConfiguredParentDir,
  parseBridgeResponse,
  resolveBeamtimeSelection,
  resolveExperimentDir,
  type BeamtimeSelection,
} from "@/lib/stxm-client";
import {
  loadWorkspacePersistence,
  pushRecentWorkspace,
  saveWorkspacePersistence,
  type RecentWorkspace,
} from "@/lib/workspace-storage";
import {
  StxmResourceCache,
  catalogCacheKey,
  scanCacheKey,
} from "@/lib/stxm-resource-cache";
import type {
  IngestionYDisplayMode,
  IzeroBounds,
  OverlaySeries,
  ScanCatalogEntry,
  ScanPayload,
  SpectrumSeries,
  StxmRegion,
} from "@/lib/stxm-types";
import {
  ingestionChartValueKind,
  ingestionModeNeedsFormula,
  ingestionModeAllowsLogYScale,
  ingestionModeNeedsReduce,
  ingestionPlotSpectra,
  ingestionSpectrumErr,
  ingestionSpectrumValue,
  ingestionYAxisLabel,
} from "@/lib/stxm/ingestion-display";

import type { WeightingMode } from "@/lib/stxm/estimators";
import type { PlotScaleMode } from "@/lib/stxm/plot-scale";
import { izeroSeriesColor, regionSeriesColor } from "@/lib/stxm/region-colors";
import {
  buildInMemoryScanContext,
  izeroRawSpectrum,
  mergeRawSpectrumUpdate,
  regionRawSpectraFromScanArrays,
  regionRawSpectrumSingle,
} from "@/lib/stxm/raw-spectrum";

const PARTIAL_SPECTRUM_THROTTLE_MS = 75;

type TabKey = "experiment" | "dashboard-preview" | "dashboard-lcf" | "ingestion";

function hdrBasename(hdrPath: string): string {
  const trimmed = hdrPath.replace(/\/+$/, "");
  const slash = trimmed.lastIndexOf("/");
  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed;
}

function hdrPathInExperiment(hdrPath: string, experimentDir: string): boolean {
  const base = experimentDir.replace(/\/+$/, "");
  return hdrPath === base || hdrPath.startsWith(`${base}/`);
}

type StxmWorkspaceProps = {
  initialParentDir?: string;
  directoryPickerEnabled?: boolean;
};

export function StxmWorkspace({
  initialParentDir = "",
  directoryPickerEnabled = false,
}: StxmWorkspaceProps) {
  const [parentDir, setParentDir] = useState(initialParentDir);
  const [hydrated, setHydrated] = useState(false);
  const [experiments, setExperiments] = useState<string[]>([]);
  const [experimentsLoading, setExperimentsLoading] = useState(false);
  const [experiment, setExperiment] = useState("");
  const [experimentCatalogCounts, setExperimentCatalogCounts] = useState<
    Record<string, { scanCount: number; lineScanCount: number }>
  >({});
  const [recentWorkspaces, setRecentWorkspaces] = useState<RecentWorkspace[]>([]);
  const [scans, setScans] = useState<string[]>([]);
  const [catalogEntries, setCatalogEntries] = useState<ScanCatalogEntry[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState("");
  const [selectedScan, setSelectedScan] = useState("");
  const [selectedHdrPath, setSelectedHdrPath] = useState("");
  const [previewScan, setPreviewScan] = useState<{
    basename: string;
    scanType: string;
    payload: ScanPayload;
  } | null>(null);
  const [parquetFilename, setParquetFilename] = useState(DEFAULT_PARQUET_FILENAME);
  const [parquetCustomized, setParquetCustomized] = useState(false);
  const [storeRoot, setStoreRoot] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("experiment");
  const [scanPayload, setScanPayload] = useState<ScanPayload | null>(null);
  const [regions, setRegions] = useState<StxmRegion[]>([]);
  const [izero, setIzero] = useState<IzeroBounds>({ izero_lo: 0, izero_hi: 0 });
  const [rawSpectra, setRawSpectra] = useState<SpectrumSeries[]>([]);
  const [reducedSpectra, setReducedSpectra] = useState<SpectrumSeries[]>([]);
  const [weightingMode, setWeightingMode] = useState<WeightingMode>("poisson_mle");
  const [useNormalized, setUseNormalized] = useState(false);
  const [yDisplayMode, setYDisplayMode] = useState<IngestionYDisplayMode>("signal");
  const [plotScaleMode, setPlotScaleMode] = useState<PlotScaleMode>("log");
  const [chemicalFormula, setChemicalFormula] = useState("");
  const [bareAtomFitOffset, setBareAtomFitOffset] = useState(true);
  const [reducedSpectraLoading, setReducedSpectraLoading] = useState(false);
  const [parquetPreview, setParquetPreview] = useState<{
    row_count: number;
    columns: string[];
    sample_names: string[];
    spot_labels: string[];
    scan_paths: string[];
  } | null>(null);
  const [overlaySeries, setOverlaySeries] = useState<OverlaySeries[]>([]);
  const [storeEntries, setStoreEntries] = useState<Array<Record<string, unknown>>>([]);
  const [status, setStatus] = useState<string>("");
  const [lcfResult, setLcfResult] = useState<{
    fractions: Record<string, number>;
    reduced_chi_square: number;
    energy_eV: number[];
    target: number[];
    model: number[];
    residual: number[];
  } | null>(null);

  const catalogCacheRef = useRef(new StxmResourceCache<ScanCatalogEntry[]>());
  const scanCacheRef = useRef(new StxmResourceCache<ScanPayload>());
  const catalogRequestIdRef = useRef(0);
  const regionDragTargetRef = useRef<RegionDragTarget | null>(null);
  const partialSpectrumTimerRef = useRef<number | null>(null);
  const partialSpectrumRafRef = useRef<number | null>(null);
  const regionsRef = useRef(regions);
  const izeroRef = useRef(izero);
  const weightingModeRef = useRef(weightingMode);
  const scanPayloadRef = useRef(scanPayload);

  regionsRef.current = regions;
  izeroRef.current = izero;
  weightingModeRef.current = weightingMode;
  scanPayloadRef.current = scanPayload;

  const experimentDir = useMemo(
    () => (experiment ? resolveExperimentDir(parentDir, experiment) : parentDir),
    [parentDir, experiment],
  );
  const hdrPath = selectedHdrPath;
  const resolvedParquetPath = useMemo(
    () =>
      parquetFilename.includes("/")
        ? parquetFilename
        : `${experimentDir}/${parquetFilename}`,
    [experimentDir, parquetFilename],
  );

  useEffect(() => {
    const stored = loadWorkspacePersistence();
    if (stored.parentDir) {
      setParentDir(stored.parentDir);
    } else if (initialParentDir) {
      setParentDir(initialParentDir);
    }
    if (stored.experiment) {
      setExperiment(stored.experiment);
    }
    setParquetFilename(stored.parquetFilename);
    setParquetCustomized(stored.parquetCustomized);
    setStoreRoot(stored.storeRoot);
    setRecentWorkspaces(stored.recent);
    setHydrated(true);
  }, [initialParentDir]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    saveWorkspacePersistence({
      parentDir,
      experiment,
      parquetFilename,
      storeRoot,
      parquetCustomized,
      recent: recentWorkspaces,
    });
  }, [
    hydrated,
    parentDir,
    experiment,
    parquetFilename,
    storeRoot,
    parquetCustomized,
    recentWorkspaces,
  ]);

  useEffect(() => {
    if (!parquetCustomized) {
      setParquetFilename(DEFAULT_PARQUET_FILENAME);
    }
  }, [experiment, parquetCustomized]);

  useEffect(() => {
    if (!hydrated || !experiment || !isConfiguredParentDir(parentDir)) {
      return;
    }
    setRecentWorkspaces((current) =>
      pushRecentWorkspace(current, { parentDir: parentDir.trim(), experiment }),
    );
  }, [hydrated, experiment, parentDir]);

  const refreshExperiments = useCallback(async () => {
    const trimmedParentDir = parentDir.trim();
    if (!isConfiguredParentDir(trimmedParentDir)) {
      setExperiments([]);
      setExperiment("");
      return;
    }
    setExperimentsLoading(true);
    try {
      const response = await fetch(
        `/api/experiments?parentDir=${encodeURIComponent(trimmedParentDir)}`,
      );
      const payload = await parseBridgeResponse<{ experiments: string[] }>(response);
      setExperiments(payload.experiments);
      if (payload.experiments.length > 0 && !payload.experiments.includes(experiment)) {
        setExperiment(payload.experiments[0] ?? "");
      } else if (payload.experiments.length === 0) {
        setExperiment("");
      }
    } finally {
      setExperimentsLoading(false);
    }
  }, [parentDir, experiment]);

  const applyCatalogEntries = useCallback((entries: ScanCatalogEntry[]) => {
    setCatalogEntries((current) => (current === entries ? current : entries));
    const lineEntries = entries.filter((entry) => entry.is_nexafs_line_scan);
    setScans(lineEntries.map((entry) => entry.basename));
    setSelectedHdrPath((current) => {
      if (lineEntries.length === 0) {
        return "";
      }
      if (lineEntries.some((entry) => entry.hdr_path === current)) {
        return current;
      }
      return lineEntries[0]?.hdr_path ?? "";
    });
    setSelectedScan((current) => {
      if (lineEntries.length === 0) {
        return "";
      }
      const match = lineEntries.find((entry) => entry.basename === current);
      if (match) {
        return current;
      }
      return lineEntries[0]?.basename ?? "";
    });
  }, []);

  const updateExperimentCatalogCounts = useCallback(
    (experimentName: string, entries: ScanCatalogEntry[]) => {
      setExperimentCatalogCounts((current) => ({
        ...current,
        [experimentName]: {
          scanCount: entries.length,
          lineScanCount: entries.filter((entry) => entry.is_nexafs_line_scan).length,
        },
      }));
    },
    [],
  );

  const refreshCatalog = useCallback(async (options?: { force?: boolean }) => {
    const force = options?.force ?? false;
    if (!experiment) {
      setCatalogEntries([]);
      setScans([]);
      setSelectedScan("");
      setSelectedHdrPath("");
      setCatalogError("");
      setCatalogLoading(false);
      return;
    }
    const cacheKey = catalogCacheKey(experimentDir);
    if (!force) {
      const cached = catalogCacheRef.current.get(cacheKey);
      if (cached) {
        applyCatalogEntries(cached);
        updateExperimentCatalogCounts(experiment, cached);
        setCatalogError("");
        return;
      }
    }

    const requestId = catalogRequestIdRef.current + 1;
    catalogRequestIdRef.current = requestId;
    const isCurrentRequest = () => requestId === catalogRequestIdRef.current;

    setCatalogEntries((current) => {
      if (current.length === 0) {
        return current;
      }
      if (current.every((entry) => hdrPathInExperiment(entry.hdr_path, experimentDir))) {
        return current;
      }
      return [];
    });

    setCatalogLoading(true);
    setCatalogError("");

    const catalogUrl = (thumbnails: boolean) =>
      `/api/experiment/catalog?experimentDir=${encodeURIComponent(experimentDir)}&thumbnails=${thumbnails ? "true" : "false"}`;

    try {
      const fastResponse = await fetch(catalogUrl(false));
      const fastPayload = await parseBridgeResponse<{ entries: ScanCatalogEntry[] }>(fastResponse);
      if (!isCurrentRequest()) {
        return;
      }
      applyCatalogEntries(fastPayload.entries);
      updateExperimentCatalogCounts(experiment, fastPayload.entries);

      const fullResponse = await fetch(catalogUrl(true));
      const fullPayload = await parseBridgeResponse<{ entries: ScanCatalogEntry[] }>(fullResponse);
      if (!isCurrentRequest()) {
        return;
      }
      catalogCacheRef.current.set(cacheKey, fullPayload.entries);
      applyCatalogEntries(fullPayload.entries);
      updateExperimentCatalogCounts(experiment, fullPayload.entries);
    } catch (error) {
      if (!isCurrentRequest()) {
        return;
      }
      catalogCacheRef.current.delete(cacheKey);
      setCatalogEntries((current) => {
        if (current.some((entry) => hdrPathInExperiment(entry.hdr_path, experimentDir))) {
          return current;
        }
        return [];
      });
      setScans([]);
      setCatalogError(error instanceof Error ? error.message : "Failed to load catalog");
    } finally {
      if (isCurrentRequest()) {
        setCatalogLoading(false);
      }
    }
  }, [applyCatalogEntries, experiment, experimentDir, updateExperimentCatalogCounts]);

  const applyScanPayload = useCallback((payload: ScanPayload) => {
    setScanPayload(payload);
    setRegions(payload.regions);
    setIzero(payload.izero_bounds);
    setReducedSpectra([]);
    try {
      const spectra = regionRawSpectraFromScanArrays(
        payload.image,
        payload.paxis_points,
        payload.qaxis_points,
        payload.regions,
        payload.izero_bounds,
        weightingModeRef.current,
      );
      setRawSpectra(spectra);
      if (spectra.length === 0 && payload.regions.length > 0) {
        setStatus("No raw spectra for the current sample regions; adjust region bars on the heatmap.");
      } else {
        setStatus(`Loaded ${payload.hdr_path}`);
      }
    } catch (error) {
      setRawSpectra([]);
      setStatus(error instanceof Error ? error.message : `Loaded ${payload.hdr_path}`);
    }
  }, []);

  const fetchScanPayload = useCallback(async (path: string): Promise<ScanPayload> => {
    const key = scanCacheKey(path);
    const cached = scanCacheRef.current.get(key);
    if (cached) {
      return cached;
    }
    const response = await fetch(`/api/scan?hdrPath=${encodeURIComponent(path)}`);
    const payload = await parseBridgeResponse<ScanPayload>(response);
    scanCacheRef.current.set(key, payload);
    return payload;
  }, []);

  const applyBeamtimeSelection = useCallback((selection: BeamtimeSelection) => {
    const trimmedParent = selection.parentDir.trim();
    if (trimmedParent) {
      setParentDir(trimmedParent);
    }
    if (selection.experiment) {
      setExperiment(selection.experiment);
      setStatus(
        `Using ${beamtimeBasename(trimmedParent)} with experiment ${selection.experiment}`,
      );
    }
  }, []);

  const openCatalogEntry = useCallback(
    async (entry: ScanCatalogEntry) => {
      const selection = resolveBeamtimeSelection(entry.hdr_path);
      const trimmedParent = parentDir.trim();
      if (selection.parentDir && selection.parentDir !== trimmedParent) {
        setParentDir(selection.parentDir);
      }
      const resolvedParent = selection.parentDir.trim() ? selection.parentDir : trimmedParent;
      const experimentName =
        selection.experiment ??
        (deriveExperimentFromHdrPath(resolvedParent, entry.hdr_path) ?? "");
      if (experimentName && experimentName !== experiment) {
        setExperiment(experimentName);
      }
      if (entry.is_nexafs_line_scan) {
        setPreviewScan(null);
        setSelectedScan(entry.basename);
        setSelectedHdrPath(entry.hdr_path);
        setActiveTab("ingestion");
        return;
      }
      setStatus(`Loading preview for ${entry.basename}`);
      const payload = await fetchScanPayload(entry.hdr_path);
      setPreviewScan({
        basename: entry.basename,
        scanType: entry.scan_type,
        payload,
      });
      setStatus(`Previewing ${entry.basename}`);
    },
    [experiment, fetchScanPayload, parentDir],
  );

  const handleBeamtimePicked = useCallback(
    (selection: BeamtimeSelection) => {
      applyBeamtimeSelection(selection);
      setActiveTab("experiment");
    },
    [applyBeamtimeSelection],
  );

  const loadScan = useCallback(async () => {
    if (!hdrPath) {
      setScanPayload(null);
      return;
    }
    const catalogEntry = catalogEntries.find((entry) => entry.hdr_path === hdrPath);
    if (!catalogEntry?.is_nexafs_line_scan) {
      setScanPayload(null);
      setStatus(
        catalogEntry
          ? `${catalogEntry.basename} is not a loadable NEXAFS line scan`
          : `Scan file not found in catalog: ${selectedScan || hdrPath}`,
      );
      return;
    }
    try {
      const payload = await fetchScanPayload(hdrPath);
      applyScanPayload(payload);
    } catch (error) {
      setScanPayload(null);
      setStatus(error instanceof Error ? error.message : "Failed to load scan");
    }
  }, [applyScanPayload, catalogEntries, fetchScanPayload, hdrPath, selectedScan]);

  const computeRawSpectra = useCallback(() => {
    if (!scanPayload || !hdrPath || regions.length === 0) {
      return;
    }
    if (hdrBasename(scanPayload.hdr_path) !== hdrBasename(hdrPath)) {
      return;
    }
    try {
      const spectra = regionRawSpectraFromScanArrays(
        scanPayload.image,
        scanPayload.paxis_points,
        scanPayload.qaxis_points,
        regions,
        izero,
        weightingMode,
      );
      setRawSpectra(spectra);
      if (spectra.length === 0) {
        setStatus("No raw spectra for the current sample regions; adjust region bars on the heatmap.");
      }
    } catch (error) {
      setRawSpectra([]);
      setStatus(error instanceof Error ? error.message : "Failed to compute raw spectra");
    }
  }, [hdrPath, izero, regions, scanPayload, weightingMode]);

  const clearPartialSpectrumSchedule = useCallback(() => {
    if (partialSpectrumTimerRef.current !== null) {
      window.clearTimeout(partialSpectrumTimerRef.current);
      partialSpectrumTimerRef.current = null;
    }
    if (partialSpectrumRafRef.current !== null) {
      window.cancelAnimationFrame(partialSpectrumRafRef.current);
      partialSpectrumRafRef.current = null;
    }
  }, []);

  const applyPartialSpectrumUpdate = useCallback((target: RegionDragTarget) => {
    const payload = scanPayloadRef.current;
    if (!payload?.image.length) {
      return;
    }
    try {
      const ctx = buildInMemoryScanContext(
        payload.image,
        payload.paxis_points,
        payload.qaxis_points,
        izeroRef.current,
      );
      if (target.kind === "izero") {
        const updated = izeroRawSpectrum(ctx, izeroRef.current, weightingModeRef.current);
        setRawSpectra((current) => mergeRawSpectrumUpdate(current, { kind: "izero" }, updated));
        return;
      }
      const region = regionsRef.current[target.index];
      if (!region) {
        return;
      }
      const updated = regionRawSpectrumSingle(
        ctx,
        region,
        target.index,
        izeroRef.current,
        weightingModeRef.current,
      );
      if (!updated) {
        return;
      }
      setRawSpectra((current) =>
        mergeRawSpectrumUpdate(current, { kind: "region", index: target.index }, updated),
      );
    } catch {
      return;
    }
  }, []);

  const schedulePartialSpectrumUpdate = useCallback(
    (target: RegionDragTarget) => {
      if (!scanPayloadRef.current?.image.length) {
        return;
      }
      if (partialSpectrumTimerRef.current !== null) {
        return;
      }
      partialSpectrumTimerRef.current = window.setTimeout(() => {
        partialSpectrumTimerRef.current = null;
        partialSpectrumRafRef.current = window.requestAnimationFrame(() => {
          partialSpectrumRafRef.current = null;
          applyPartialSpectrumUpdate(target);
        });
      }, PARTIAL_SPECTRUM_THROTTLE_MS);
    },
    [applyPartialSpectrumUpdate],
  );

  const handleRegionDragStart = useCallback((target: RegionDragTarget) => {
    regionDragTargetRef.current = target;
    clearPartialSpectrumSchedule();
  }, [clearPartialSpectrumSchedule]);

  const handleRegionDragEnd = useCallback(() => {
    regionDragTargetRef.current = null;
    clearPartialSpectrumSchedule();
    computeRawSpectra();
  }, [clearPartialSpectrumSchedule, computeRawSpectra]);

  const handleRegionChangeDuringSession = useCallback(
    (index: number, region: StxmRegion) => {
      setRegions((current) => {
        const next = [...current];
        next[index] = region;
        return next;
      });
      const dragTarget = regionDragTargetRef.current;
      if (dragTarget?.kind === "region" && dragTarget.index === index) {
        schedulePartialSpectrumUpdate(dragTarget);
      }
    },
    [schedulePartialSpectrumUpdate],
  );

  const handleIzeroChangeDuringSession = useCallback(
    (nextIzero: IzeroBounds) => {
      setIzero(nextIzero);
      const dragTarget = regionDragTargetRef.current;
      if (dragTarget?.kind === "izero") {
        schedulePartialSpectrumUpdate(dragTarget);
      }
    },
    [schedulePartialSpectrumUpdate],
  );

  useEffect(() => {
    return () => {
      clearPartialSpectrumSchedule();
    };
  }, [clearPartialSpectrumSchedule]);

  const reduceCurrentScan = useCallback(async () => {
    if (!hdrPath || regions.length === 0) {
      return;
    }
    const catalogEntry = catalogEntries.find((entry) => entry.hdr_path === hdrPath);
    if (!catalogEntry?.is_nexafs_line_scan) {
      return;
    }
    if (ingestionModeNeedsFormula(yDisplayMode) && !chemicalFormula.trim()) {
      setStatus("Enter a chemical formula for CXRO normalized mass absorption");
      return;
    }
    setReducedSpectraLoading(true);
    try {
      const response = await fetch("/api/reduce", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hdrPath,
          regions,
          izero,
          weightingMode,
          normalizationMode: "pre_edge_scale",
          preEdge: "280,283",
          postEdge: "292,310",
          formula: chemicalFormula.trim() || undefined,
          bareAtomFitOffset,
        }),
      });
      const payload = await parseBridgeResponse<{ spectra: SpectrumSeries[] }>(response);
      setReducedSpectra(payload.spectra);
      const sampleCount = payload.spectra.filter(
        (spectrum) => spectrum.spot_label !== "izero",
      ).length;
      setStatus(`Reduced ${sampleCount} region spectra`);
    } catch (error) {
      setReducedSpectra([]);
      setStatus(error instanceof Error ? error.message : "Failed to reduce scan");
    } finally {
      setReducedSpectraLoading(false);
    }
  }, [
    bareAtomFitOffset,
    catalogEntries,
    chemicalFormula,
    hdrPath,
    regions,
    izero,
    weightingMode,
    yDisplayMode,
  ]);

  const persistRegions = useCallback(async () => {
    if (!hdrPath || !experiment) {
      return;
    }
    await fetch("/api/regions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        experimentDir,
        hdrPath,
        izero_lo: izero.izero_lo,
        izero_hi: izero.izero_hi,
        regions,
      }),
    });
    scanCacheRef.current.delete(scanCacheKey(hdrPath));
  }, [experiment, experimentDir, hdrPath, izero, regions]);

  const refreshParquetPreview = useCallback(async () => {
    const response = await fetch(
      `/api/parquet/preview?parquetPath=${encodeURIComponent(resolvedParquetPath)}`,
    );
    const payload = await parseBridgeResponse<{
      row_count: number;
      columns: string[];
      sample_names: string[];
      spot_labels: string[];
      scan_paths: string[];
    }>(response);
    setParquetPreview(payload);
  }, [resolvedParquetPath]);

  const refreshParquetSpectra = useCallback(async () => {
    const response = await fetch(
      `/api/parquet/spectra?parquetPath=${encodeURIComponent(resolvedParquetPath)}&useNormalized=${useNormalized}`,
    );
    const payload = await parseBridgeResponse<{ series: OverlaySeries[] }>(response);
    setOverlaySeries(payload.series);
  }, [resolvedParquetPath, useNormalized]);

  const refreshStoreManifest = useCallback(async () => {
    if (!storeRoot) {
      setStoreEntries([]);
      return;
    }
    const response = await fetch(
      `/api/store/manifest?storeRoot=${encodeURIComponent(storeRoot)}`,
    );
    const payload = await parseBridgeResponse<{ entries: Array<Record<string, unknown>> }>(response);
    setStoreEntries(payload.entries);
  }, [storeRoot]);

  const refreshAll = useCallback(async () => {
    setRefreshing(true);
    setStatus("Reloading workspace...");
    try {
      await refreshExperiments();
      if (!isConfiguredParentDir(parentDir)) {
        setStatus("");
        return;
      }
      await refreshCatalog({ force: true });
      await loadScan();
      await refreshParquetPreview();
      await refreshParquetSpectra();
      await refreshStoreManifest();
      setStatus("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Reload failed");
    } finally {
      setRefreshing(false);
    }
  }, [
    loadScan,
    parentDir,
    refreshExperiments,
    refreshParquetPreview,
    refreshParquetSpectra,
    refreshCatalog,
    refreshStoreManifest,
  ]);

  const handleOpenRecent = useCallback((recentParentDir: string, recentExperiment: string) => {
    setParentDir(recentParentDir);
    setExperiment(recentExperiment);
    setActiveTab("experiment");
  }, []);

  const handleExperimentSelect = useCallback((name: string) => {
    setExperiment(name);
    setActiveTab("experiment");
  }, []);

  const experimentSummaries = useMemo(
    () =>
      experiments.map((name) => {
        const counts = experimentCatalogCounts[name];
        return {
          name,
          scanCount: counts?.scanCount,
          lineScanCount: counts?.lineScanCount,
        };
      }),
    [experimentCatalogCounts, experiments],
  );

  const breadcrumb = useMemo(() => {
    const segments: Array<{ label: string; title?: string; onClick?: () => void }> = [];
    const trimmedParent = parentDir.trim();
    if (isConfiguredParentDir(trimmedParent)) {
      segments.push({
        label: beamtimeBasename(trimmedParent),
        title: trimmedParent,
        onClick: () => setActiveTab("experiment"),
      });
    }
    if (experiment) {
      segments.push({
        label: experiment,
        title: resolveExperimentDir(trimmedParent, experiment),
        onClick: () => setActiveTab("experiment"),
      });
    }
    if (selectedScan) {
      segments.push({
        label: selectedScan,
        title: selectedHdrPath || selectedScan,
      });
    }
    return segments;
  }, [experiment, parentDir, selectedHdrPath, selectedScan]);

  const workspaceStatus = useMemo(() => {
    const trimmedParent = parentDir.trim();
    if (!isConfiguredParentDir(trimmedParent)) {
      return "Select your beamtime folder (e.g. BL5321 (New STXM)) to browse experiments and line scans.";
    }
    if (!experiment) {
      const folderLabel = experiments.length === 1 ? "folder" : "folders";
      return `${experiments.length} experiment ${folderLabel} in ${beamtimeBasename(trimmedParent)}. Select one below to browse scans.`;
    }
    const lineCount = catalogEntries.filter((entry) => entry.is_nexafs_line_scan).length;
    if (activeTab === "ingestion" && selectedScan) {
      return `Ingestion open for ${selectedScan} in ${experiment} (${lineCount} NEXAFS line scans available).`;
    }
    return `${experiment} contains ${catalogEntries.length} scans (${lineCount} NEXAFS line). Click a scan card to preview or open ingestion.`;
  }, [
    activeTab,
    catalogEntries,
    experiment,
    experiments.length,
    parentDir,
    selectedScan,
  ]);

  const recentWorkspaceChips = useMemo(
    () =>
      recentWorkspaces.map((item) => ({
        ...item,
        label: item.experiment || beamtimeBasename(item.parentDir),
      })),
    [recentWorkspaces],
  );

  const displayStatus = status || workspaceStatus;

  useEffect(() => {
    void refreshExperiments();
  }, [refreshExperiments]);

  useEffect(() => {
    void refreshCatalog();
  }, [refreshCatalog]);

  useEffect(() => {
    void loadScan();
  }, [loadScan]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void persistRegions();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [persistRegions, regions, izero]);

  useEffect(() => {
    if (regionDragTargetRef.current) {
      return;
    }
    computeRawSpectra();
  }, [computeRawSpectra, regions, izero, hdrPath, scanPayload, weightingMode]);

  useEffect(() => {
    if (regionDragTargetRef.current) {
      return;
    }
    if (!ingestionModeNeedsReduce(yDisplayMode)) {
      return;
    }
    if (reducedSpectra.length === 0) {
      return;
    }
    const timer = window.setTimeout(() => {
      void reduceCurrentScan();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [
    reduceCurrentScan,
    weightingMode,
    reducedSpectra.length,
    yDisplayMode,
    chemicalFormula,
    bareAtomFitOffset,
  ]);

  useEffect(() => {
    if (!ingestionModeNeedsReduce(yDisplayMode)) {
      return;
    }
    if (reducedSpectra.length > 0) {
      return;
    }
    if (!scanPayload || regions.length === 0) {
      return;
    }
    const timer = window.setTimeout(() => {
      void reduceCurrentScan();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [yDisplayMode, scanPayload, regions.length, reducedSpectra.length, reduceCurrentScan]);

  useEffect(() => {
    if (!ingestionModeAllowsLogYScale(yDisplayMode) && plotScaleMode === "log") {
      setPlotScaleMode("linear");
    }
  }, [plotScaleMode, yDisplayMode]);

  const ingestionUsesReduced = ingestionModeNeedsReduce(yDisplayMode);
  const activeIngestionSpectra = ingestionUsesReduced ? reducedSpectra : rawSpectra;
  const plotIngestionSpectra = useMemo(
    () => ingestionPlotSpectra(activeIngestionSpectra, yDisplayMode),
    [activeIngestionSpectra, yDisplayMode],
  );

  const ingestionChart = useMemo<ChartSeries[]>(() => {
    const valueKind = ingestionChartValueKind(yDisplayMode);
    let regionIndex = 0;
    return plotIngestionSpectra.map((spectrum, index) => {
      const isIzero = spectrum.spot_label === "izero";
      const color = spectrum.color ?? (isIzero ? izeroSeriesColor() : regionSeriesColor(regionIndex));
      if (!isIzero) {
        regionIndex += 1;
      }
      return {
        id: `ingest-${index}`,
        label: spectrum.spot_label,
        color,
        valueKind,
        points: spectrum.energy_eV.map((energy, pointIndex) => {
          const err = ingestionSpectrumErr(spectrum, pointIndex, yDisplayMode);
          return {
            energy,
            value: ingestionSpectrumValue(spectrum, pointIndex, yDisplayMode),
            ...(err !== undefined ? { err } : {}),
          };
        }),
      };
    });
  }, [plotIngestionSpectra, yDisplayMode]);

  const previewChart = useMemo<ChartSeries[]>(() => {
    return overlaySeries.map((series, index) => ({
      id: `preview-${index}`,
      label: series.label,
      color: regionSeriesColor(index),
      valueKind: "od" as const,
      points: series.energy_eV.map((energy, pointIndex) => {
        const err = series.y_err[pointIndex];
        return {
          energy,
          value: series.y[pointIndex] ?? 0,
          ...(err !== undefined && Number.isFinite(err) && err > 0 ? { err } : {}),
        };
      }),
    }));
  }, [overlaySeries]);

  const runDemoLcf = useCallback(async () => {
    if (reducedSpectra.length === 0 || overlaySeries.length === 0) {
      setStatus("Load a reduced spectrum and at least one reference spectrum first");
      return;
    }
    const target = reducedSpectra[0];
    if (!target) {
      return;
    }
    const components = overlaySeries.slice(0, 2).map((series, index) => ({
      name: series.label || `component-${index + 1}`,
      energy_eV: series.energy_eV,
      OD: series.y,
      initial: index === 0 ? 50 : 50,
      minimum: 0,
      maximum: 100,
      fixed: false,
    }));
    const response = await fetch("/api/lcf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target: {
          energy_eV: target.energy_eV,
          OD: useNormalized ? target.OD_normalized : target.OD,
        },
        components,
      }),
    });
    const payload = await parseBridgeResponse<{
      fractions: Record<string, number>;
      reduced_chi_square: number;
      energy_eV: number[];
      target: number[];
      model: number[];
      residual: number[];
    }>(response);
    setLcfResult(payload);
    setStatus("LCF fit complete");
  }, [overlaySeries, reducedSpectra, useNormalized]);

  return (
    <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-4 p-4 md:p-6">
      <WorkspaceHeader
        parentDir={parentDir}
        breadcrumb={breadcrumb}
        workspaceStatus={displayStatus}
        parquetFilename={parquetFilename}
        storeRoot={storeRoot}
        parquetCustomized={parquetCustomized}
        directoryPickerEnabled={directoryPickerEnabled}
        refreshing={refreshing}
        recentWorkspaces={recentWorkspaceChips}
        onParentDirChange={setParentDir}
        onBeamtimePicked={handleBeamtimePicked}
        onParquetFilenameChange={setParquetFilename}
        onParquetCustomizedChange={setParquetCustomized}
        onStoreRootChange={setStoreRoot}
        onRefresh={() => void refreshAll()}
        onOpenRecent={handleOpenRecent}
      />
      <ScanViewerShell
        open={previewScan !== null}
        title={previewScan?.basename ?? "Scan preview"}
        onClose={() => setPreviewScan(null)}
      >
        {previewScan ? (
          <SimpleImageViewer
            image={previewScan.payload.image}
            imageMin={previewScan.payload.image_min}
            imageMax={previewScan.payload.image_max}
            paxisName={previewScan.payload.paxis_name}
            qaxisName={previewScan.payload.qaxis_name}
            title={previewScan.basename}
            scanType={previewScan.scanType}
            shape={previewScan.payload.shape}
          />
        ) : null}
      </ScanViewerShell>
      <div className="flex flex-wrap gap-2">
        {(
          [
            ["experiment", "Experiment"],
            ["ingestion", "Ingestion"],
            ["dashboard-preview", "Dashboard / Preview spectra"],
            ["dashboard-lcf", "Dashboard / LC fitting"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`rounded-full px-4 py-2 text-sm font-medium ${
              activeTab === key ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-700"
            }`}
            onClick={() => setActiveTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {isConfiguredParentDir(parentDir) ? (
        <ExperimentPicker
          className={activeTab === "experiment" || !experiment ? "" : "hidden md:block"}
          experiments={experimentSummaries}
          selected={experiment}
          loading={experimentsLoading}
          onSelect={handleExperimentSelect}
        />
      ) : null}

      {experiment ? (
        <ExperimentFileBrowser
          className={activeTab === "experiment" ? "" : "hidden"}
          entries={catalogEntries}
          selectedBasename={selectedScan}
          loading={catalogLoading}
          error={catalogError}
          onSelect={openCatalogEntry}
        />
      ) : activeTab === "experiment" && isConfiguredParentDir(parentDir) ? (
        <p className="text-sm text-zinc-500">Choose an experiment above to browse scan files.</p>
      ) : null}

      {activeTab === "ingestion" ? (
        <section className="flex flex-col gap-4">
          {scans.length === 0 ? (
            <p className="text-sm text-zinc-500">
              Select a NEXAFS line scan from the Experiment tab to open the ingestion viewer.
            </p>
          ) : null}
          {selectedScan ? (
            <IngestionToolbar
              scanLabel={selectedScan}
              weightingMode={weightingMode}
              onWeightingModeChange={setWeightingMode}
              yDisplayMode={yDisplayMode}
              onYDisplayModeChange={setYDisplayMode}
              plotScaleMode={plotScaleMode}
              onPlotScaleModeChange={setPlotScaleMode}
              chemicalFormula={chemicalFormula}
              onChemicalFormulaChange={setChemicalFormula}
              bareAtomFitOffset={bareAtomFitOffset}
              onBareAtomFitOffsetChange={setBareAtomFitOffset}
              onRecompute={() => {
                computeRawSpectra();
                void reduceCurrentScan();
              }}
              disabled={!scanPayload}
            />
          ) : null}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[280px_minmax(0,1fr)] md:items-start">
            <div className="min-w-0">
              {scanPayload ? (
                <LineScanRegionEditor
                  image={scanPayload.image}
                  paxisPoints={scanPayload.paxis_points}
                  qaxisPoints={scanPayload.qaxis_points}
                  regions={regions}
                  izero={izero}
                  imageScaleMode={plotScaleMode}
                  onRegionsChange={setRegions}
                  onRegionChange={handleRegionChangeDuringSession}
                  onIzeroChange={handleIzeroChangeDuringSession}
                  onDragStart={handleRegionDragStart}
                  onDragEnd={handleRegionDragEnd}
                />
              ) : (
                <div className="flex h-[520px] w-full max-w-[280px] items-center justify-center rounded-lg border border-dashed border-zinc-300 text-sm text-zinc-500">
                  Load a line scan to view the heatmap.
                </div>
              )}
            </div>
            <div className="flex min-h-[520px] min-w-0 flex-col">
              <SpectrumChart
                series={ingestionChart}
                yLabel={ingestionYAxisLabel(yDisplayMode)}
                yScale={plotScaleMode}
                yDisplayMode={yDisplayMode}
                height={520}
                className="min-h-[520px]"
                loading={ingestionUsesReduced && reducedSpectraLoading}
                emptyMessage={
                  scanPayload
                    ? ingestionModeNeedsFormula(yDisplayMode) && !chemicalFormula.trim()
                      ? "Enter a chemical formula and click Recompute spectra for CXRO mass absorption."
                      : "No spectra for the current regions. Adjust sample bars or check the status line for errors."
                    : "Load a line scan to plot per-region spectra."
                }
              />
            </div>
          </div>
        </section>
      ) : null}

      {activeTab === "dashboard-preview" ? (
        <section className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
          <div className="space-y-4">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-800">
              <input
                type="checkbox"
                className="rounded border-zinc-300"
                checked={useNormalized}
                onChange={(event) => setUseNormalized(event.target.checked)}
              />
              Normalized OD (parquet overlay)
            </label>
            <button
              type="button"
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50"
              onClick={() => void refreshParquetPreview()}
            >
              Refresh parquet catalog
            </button>
            {parquetPreview ? (
              <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm">
                <p>{parquetPreview.row_count} rows</p>
                <p>{parquetPreview.sample_names.length} samples</p>
                <p>{parquetPreview.spot_labels.length} spot labels</p>
              </div>
            ) : null}
            <button
              type="button"
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50"
              onClick={() => void refreshParquetSpectra()}
            >
              Load overlay spectra
            </button>
            <button
              type="button"
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50"
              onClick={() => void refreshStoreManifest()}
            >
              Refresh store manifest
            </button>
            <DataTable
              columns={storeEntries[0] ? Object.keys(storeEntries[0]) : ["path"]}
              rows={storeEntries.slice(0, 20)}
              emptyMessage="Set a store root and refresh to browse stored spectra."
            />
          </div>
          <SpectrumChart series={previewChart} yLabel={useNormalized ? "OD normalized" : "OD"} />
        </section>
      ) : null}

      {activeTab === "dashboard-lcf" ? (
        <section className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700"
              onClick={() => void runDemoLcf()}
            >
              Run LCF
            </button>
          </div>
          {lcfResult ? (
            <>
              <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 text-sm">
                {Object.entries(lcfResult.fractions).map(([name, fraction]) => (
                  <p key={name}>
                    {name}: {(fraction * 100).toFixed(2)}%
                  </p>
                ))}
                <p>Reduced chi-square: {lcfResult.reduced_chi_square.toFixed(4)}</p>
              </div>
              <SpectrumChart
                series={[
                  {
                    id: "lcf-target",
                    label: "Target",
                    color: "#2563eb",
                    valueKind: "od",
                    points: lcfResult.energy_eV.map((energy, index) => ({
                      energy,
                      value: lcfResult.target[index] ?? 0,
                    })),
                  },
                  {
                    id: "lcf-model",
                    label: "Model",
                    color: "#16a34a",
                    valueKind: "od",
                    points: lcfResult.energy_eV.map((energy, index) => ({
                      energy,
                      value: lcfResult.model[index] ?? 0,
                    })),
                  },
                  {
                    id: "lcf-residual",
                    label: "Residual",
                    color: "#dc2626",
                    valueKind: "od",
                    points: lcfResult.energy_eV.map((energy, index) => ({
                      energy,
                      value: lcfResult.residual[index] ?? 0,
                    })),
                  },
                ]}
                height={420}
                yLabel="OD"
              />
            </>
          ) : (
            <p className="text-sm text-zinc-500">
              Reduce a scan in Ingestion, load reference spectra in Preview, then run LCF here.
            </p>
          )}
        </section>
      ) : null}
    </div>
  );
}
