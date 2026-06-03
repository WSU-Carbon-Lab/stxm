import { describe, expect, test } from "bun:test";

import {
  DEFAULT_PARQUET_FILENAME,
  beamtimeBasename,
  deriveExperimentFromHdrPath,
  isExperimentFolderName,
  resolveBeamtimeSelection,
  resolveExperimentDir,
} from "@/lib/stxm-client";

describe("beamtimeBasename", () => {
  test("returns the last path segment", () => {
    expect(beamtimeBasename("/data/beamtime/2025")).toBe("2025");
  });

  test("strips trailing slashes", () => {
    expect(beamtimeBasename("/data/beamtime/2025/")).toBe("2025");
  });
});

describe("deriveExperimentFromHdrPath", () => {
  test("returns the experiment folder relative to the beamtime root", () => {
    expect(
      deriveExperimentFromHdrPath(
        "/data/beamtime",
        "/data/beamtime/2025_10(October)/scan.hdr",
      ),
    ).toBe("2025_10(October)");
  });

  test("returns null when the hdr path is outside the beamtime root", () => {
    expect(deriveExperimentFromHdrPath("/data/beamtime", "/other/scan.hdr")).toBeNull();
  });
});

describe("resolveExperimentDir", () => {
  test("joins parent and experiment without a duplicate slash", () => {
    expect(resolveExperimentDir("/data/beamtime/", "2025_10(October)")).toBe(
      "/data/beamtime/2025_10(October)",
    );
  });
});

describe("isExperimentFolderName", () => {
  test("accepts underscore and hyphen month folders", () => {
    expect(isExperimentFolderName("2025_10(October)")).toBe(true);
    expect(isExperimentFolderName("2026-03(March)")).toBe(true);
  });

  test("rejects beamtime and scan names", () => {
    expect(isExperimentFolderName("BL5321 (New STXM)")).toBe(false);
    expect(isExperimentFolderName("532_260313061.hdr")).toBe(false);
  });
});

describe("resolveBeamtimeSelection", () => {
  const beamtimeRoot = "/data/beamtime/BL5321 (New STXM)";

  test("promotes a picked experiment month folder", () => {
    expect(resolveBeamtimeSelection(`${beamtimeRoot}/2026-03(March)`)).toEqual({
      parentDir: beamtimeRoot,
      experiment: "2026-03(March)",
    });
  });

  test("keeps a picked beamtime root without an experiment", () => {
    expect(resolveBeamtimeSelection(beamtimeRoot)).toEqual({
      parentDir: beamtimeRoot,
      experiment: "",
    });
  });

  test("resolves scan hdr paths to beamtime root and experiment", () => {
    const hdrPath = `${beamtimeRoot}/2026-03(March)/532_260313061.hdr`;
    const selection = resolveBeamtimeSelection(hdrPath);
    expect(selection).toEqual({
      parentDir: beamtimeRoot,
      experiment: "2026-03(March)",
    });
    expect(deriveExperimentFromHdrPath(selection.parentDir, hdrPath)).toBe(
      selection.experiment,
    );
  });

  test("walks up from nested paths inside an experiment folder", () => {
    expect(
      resolveBeamtimeSelection(`${beamtimeRoot}/2025_10(October)/nested/deep`),
    ).toEqual({
      parentDir: beamtimeRoot,
      experiment: "2025_10(October)",
    });
  });
});

describe("DEFAULT_PARQUET_FILENAME", () => {
  test("matches the legacy widget default", () => {
    expect(DEFAULT_PARQUET_FILENAME).toBe("experiment.parquet");
  });
});
