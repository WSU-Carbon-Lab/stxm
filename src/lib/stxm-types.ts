export type StxmRegion = {
  sample_lo: number;
  sample_hi: number;
  spot_label: string;
};

export type IzeroBounds = {
  izero_lo: number;
  izero_hi: number;
};

export type ScanPayload = {
  ok: true;
  hdr_path: string;
  shape: number[];
  paxis_name: string;
  qaxis_name: string;
  paxis_points: number[];
  qaxis_points: number[];
  regions: StxmRegion[];
  izero_bounds: IzeroBounds;
  image: number[][];
  image_min: number;
  image_max: number;
};

export type SpectrumKind = "raw" | "od";

/** Ingestion spectrum Y-axis selection for the live line-scan plot. */
export type IngestionYDisplayMode =
  | "signal"
  | "signal_inverse"
  | "od"
  | "od_normalized"
  | "mass_absorption_cxro";

export type SpectrumSeries = {
  kind?: SpectrumKind;
  spot_label: string;
  sample_lo: number;
  sample_hi: number;
  energy_eV: number[];
  signal?: number[];
  signal_err?: number[];
  OD?: number[];
  OD_err?: number[];
  OD_normalized?: number[];
  mass_absorption?: number[];
  mass_absorption_err?: number[];
  beta?: number[];
  beta_err?: number[];
  color?: string;
};

export type OverlaySeries = {
  label: string;
  energy_eV: number[];
  y: number[];
  y_err: number[];
};

export type ScanCategory =
  | "line_scan"
  | "image_scan"
  | "fixed_point"
  | "focus_scan"
  | "stack"
  | "other";

export type ScanCatalogEntry = {
  basename: string;
  hdr_path: string;
  scan_type: string;
  category: ScanCategory;
  is_nexafs_line_scan: boolean;
  shape: [number, number] | null;
  paxis_count?: number;
  qaxis_count?: number;
  energy_eV?: number | null;
  energy_min_eV?: number | null;
  energy_max_eV?: number | null;
  num_energy_points?: number | null;
  thumbnail_png_base64?: string;
};

export type ExperimentCatalogPayload = {
  experiment_dir: string;
  entries: ScanCatalogEntry[];
};

export type BridgeError = {
  ok: false;
  error: string;
};

export type BridgeSuccess<T> = { ok: true } & T;

export type BridgeResponse<T> = BridgeSuccess<T> | BridgeError;

export type WorkspacePaths = {
  parentDir: string;
  experiment: string;
  parquetPath: string;
  storeRoot: string;
};
