/**
 * In-memory client-side cache for STXM workspace fetches.
 *
 * Keys catalog and scan payloads by filesystem path so tab switches and scan
 * selection do not repeat expensive bridge round-trips.
 */

/** Builds the cache key for an experiment catalog (absolute experiment directory). */
export function catalogCacheKey(experimentDir: string): string {
  return experimentDir.trim();
}

/** Builds the cache key for a loaded scan (absolute .hdr path). */
export function scanCacheKey(hdrPath: string): string {
  return hdrPath.trim();
}

/** Simple string-keyed map for workspace resource payloads. */
export class StxmResourceCache<T> {
  private readonly entries = new Map<string, T>();

  /** Returns a cached value when present; otherwise `undefined`. */
  get(key: string): T | undefined {
    return this.entries.get(key);
  }

  /** Stores `value` under `key`, replacing any prior entry. */
  set(key: string, value: T): void {
    this.entries.set(key, value);
  }

  /** Removes the entry for `key` when it exists. */
  delete(key: string): void {
    this.entries.delete(key);
  }

  /** Clears all cached entries. */
  clear(): void {
    this.entries.clear();
  }
}
