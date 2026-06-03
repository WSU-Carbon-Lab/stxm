export class StxmBridgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StxmBridgeError";
  }
}

export const DEFAULT_PARQUET_FILENAME = "experiment.parquet";

export function isConfiguredParentDir(parentDir: string): boolean {
  return parentDir.trim().length > 0;
}

/** Matches experiment month folders such as `2025_10(October)` or `2026-03(March)`. */
export const EXPERIMENT_FOLDER_PATTERN = /^\d{4}[-_]\d{1,2}\([^)]+\)$/;

/**
 * Returns whether `name` is an STXM experiment subdirectory under a beamtime root.
 */
export function isExperimentFolderName(name: string): boolean {
  return EXPERIMENT_FOLDER_PATTERN.test(name.trim());
}

export type BeamtimeSelection = {
  parentDir: string;
  experiment: string;
};

function normalizePickedPath(pickedPath: string): string {
  const trimmed = pickedPath.trim();
  if (!trimmed) {
    return "";
  }
  const withoutTrailing = trimmed.replace(/\/$/, "");
  if (withoutTrailing.toLowerCase().endsWith(".hdr")) {
    const lastSlash = withoutTrailing.lastIndexOf("/");
    return lastSlash >= 0 ? withoutTrailing.slice(0, lastSlash) : withoutTrailing;
  }
  return withoutTrailing;
}

/**
 * Maps a directory-picker path to beamtime `parentDir` and optional `experiment`.
 *
 * Promotes month folders (e.g. `2026-03(March)`) to experiment with parent beamtime root,
 * and walks scan paths up to the deepest matching experiment segment.
 */
export function resolveBeamtimeSelection(pickedPath: string): BeamtimeSelection {
  const normalized = normalizePickedPath(pickedPath);
  if (!normalized) {
    return { parentDir: "", experiment: "" };
  }

  const parts = normalized.split("/");
  let experimentIndex = -1;
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    const segment = parts[index];
    if (segment && isExperimentFolderName(segment)) {
      experimentIndex = index;
      break;
    }
  }

  if (experimentIndex >= 0) {
    const experiment = parts[experimentIndex] ?? "";
    const parentDir = parts.slice(0, experimentIndex).join("/") || "/";
    return { parentDir, experiment };
  }

  return { parentDir: normalized, experiment: "" };
}

export function resolveExperimentDir(parentDir: string, experiment: string): string {
  const trimmedParent = parentDir.replace(/\/$/, "");
  return `${trimmedParent}/${experiment}`;
}

/** Returns the final path segment of `parentDir`, with trailing slashes removed. */
export function beamtimeBasename(parentDir: string): string {
  const trimmed = parentDir.replace(/\/$/, "");
  const segments = trimmed.split("/");
  return segments[segments.length - 1] ?? trimmed;
}

/**
 * Derives the experiment folder name for `hdrPath` relative to the beamtime root.
 *
 * Returns `null` when `hdrPath` is not under `parentDir` or sits directly in the root.
 */
export function deriveExperimentFromHdrPath(parentDir: string, hdrPath: string): string | null {
  const normalizedParent = parentDir.replace(/\/$/, "");
  const normalizedHdr = hdrPath.replace(/\/$/, "");
  if (!normalizedHdr.startsWith(`${normalizedParent}/`)) {
    return null;
  }
  const relative = normalizedHdr.slice(normalizedParent.length + 1);
  const slashIndex = relative.indexOf("/");
  if (slashIndex <= 0) {
    return null;
  }
  return relative.slice(0, slashIndex);
}

export async function parseBridgeResponse<T>(
  response: Response,
): Promise<T> {
  const payload = (await response.json()) as T & { ok?: boolean; error?: string };
  if (payload && typeof payload === "object" && "ok" in payload && payload.ok === false) {
    throw new StxmBridgeError(String(payload.error ?? "Bridge request failed"));
  }
  return payload;
}

export type PickDirectoryResult =
  | { cancelled: true }
  | { cancelled: false; path: string };

export async function pickParentDirectory(): Promise<PickDirectoryResult> {
  const response = await fetch("/api/pick-directory", { method: "POST" });
  const payload = (await response.json()) as {
    ok?: boolean;
    path?: string;
    cancelled?: boolean;
    error?: string;
  };
  if (!response.ok || payload.ok === false) {
    throw new StxmBridgeError(String(payload.error ?? "Failed to pick directory"));
  }
  if (payload.cancelled) {
    return { cancelled: true };
  }
  if (!payload.path) {
    throw new StxmBridgeError("Picker returned no path");
  }
  return { cancelled: false, path: payload.path };
}
