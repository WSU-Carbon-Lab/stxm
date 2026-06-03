const STORAGE_KEY = "stxm-workspace-v1";
const MAX_RECENT = 5;

export type RecentWorkspace = {
  parentDir: string;
  experiment: string;
};

export type WorkspacePersistence = {
  parentDir: string;
  experiment: string;
  parquetFilename: string;
  storeRoot: string;
  parquetCustomized: boolean;
  recent: RecentWorkspace[];
};

const DEFAULTS: WorkspacePersistence = {
  parentDir: "",
  experiment: "",
  parquetFilename: "experiment.parquet",
  storeRoot: "",
  parquetCustomized: false,
  recent: [],
};

function readRaw(): Partial<WorkspacePersistence> {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    return JSON.parse(raw) as Partial<WorkspacePersistence>;
  } catch {
    return {};
  }
}

/**
 * Loads persisted workspace fields from `localStorage`, merging with defaults.
 */
export function loadWorkspacePersistence(): WorkspacePersistence {
  const stored = readRaw();
  return {
    parentDir: stored.parentDir ?? DEFAULTS.parentDir,
    experiment: stored.experiment ?? DEFAULTS.experiment,
    parquetFilename: stored.parquetFilename ?? DEFAULTS.parquetFilename,
    storeRoot: stored.storeRoot ?? DEFAULTS.storeRoot,
    parquetCustomized: stored.parquetCustomized ?? DEFAULTS.parquetCustomized,
    recent: Array.isArray(stored.recent) ? stored.recent.slice(0, MAX_RECENT) : DEFAULTS.recent,
  };
}

/**
 * Merges `partial` into persisted workspace state and writes it to `localStorage`.
 */
export function saveWorkspacePersistence(partial: Partial<WorkspacePersistence>): void {
  if (typeof window === "undefined") {
    return;
  }
  const current = loadWorkspacePersistence();
  const next: WorkspacePersistence = {
    ...current,
    ...partial,
    recent: partial.recent ?? current.recent,
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
}

/**
 * Records a workspace visit and returns the updated recent list (newest first, deduped).
 */
export function pushRecentWorkspace(
  recent: RecentWorkspace[],
  entry: RecentWorkspace,
): RecentWorkspace[] {
  const normalized = {
    parentDir: entry.parentDir.trim(),
    experiment: entry.experiment.trim(),
  };
  if (!normalized.parentDir) {
    return recent;
  }
  const filtered = recent.filter(
    (item) =>
      !(
        item.parentDir === normalized.parentDir && item.experiment === normalized.experiment
      ),
  );
  return [normalized, ...filtered].slice(0, MAX_RECENT);
}
