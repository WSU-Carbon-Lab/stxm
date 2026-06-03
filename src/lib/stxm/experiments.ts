import "server-only";

import fs from "node:fs";

import { listNexafsLineScans } from "@/lib/stxm/io";
import { requireAllowedDirectory } from "@/lib/stxm/path-utils";

/**
 * Sort key for experiment folder names (year, month, name) descending.
 */
export function experimentSortKey(name: string): [number, number, string] {
  let base = name.trim();
  if (base.includes("(")) {
    base = base.split("(")[0] ?? base;
  }
  base = base.replace(/_/g, "-");
  const parts = base.split("-");
  try {
    const year = Number.parseInt(parts[0] ?? "", 10);
    const month = parts.length > 1 ? Number.parseInt(parts[1] ?? "1", 10) : 1;
    if (Number.isNaN(year)) {
      return [0, 0, name];
    }
    return [year, Number.isNaN(month) ? 1 : month, name];
  } catch {
    return [0, 0, name];
  }
}

/**
 * List experiment subdirectories under a parent beamtime folder.
 */
export function listExperiments(parentDir: string): { parent_dir: string; experiments: string[] } {
  const parent = requireAllowedDirectory(parentDir);
  const names = fs
    .readdirSync(parent, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((a, b) => {
      const keyA = experimentSortKey(a);
      const keyB = experimentSortKey(b);
      if (keyA[0] !== keyB[0]) {
        return keyB[0] - keyA[0];
      }
      if (keyA[1] !== keyB[1]) {
        return keyB[1] - keyA[1];
      }
      return keyB[2].localeCompare(keyA[2]);
    });
  return { parent_dir: parent, experiments: names };
}

/**
 * List NEXAFS line scan basenames in an experiment directory.
 */
export function listScans(experimentDir: string): { experiment_dir: string; scans: string[] } {
  const experiment = requireAllowedDirectory(experimentDir);
  const scans = listNexafsLineScans(experiment).map((hdrPath) =>
    hdrPath.slice(hdrPath.lastIndexOf("/") + 1),
  );
  return { experiment_dir: experiment, scans };
}
