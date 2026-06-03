import "server-only";

import path from "node:path";

import type { IzeroBounds, ScanPayload, StxmRegion } from "@/lib/stxm-types";
import { loadStxm } from "@/lib/stxm/io";
import { requireAllowedFile } from "@/lib/stxm/path-utils";
import { loadScanRegions } from "@/lib/stxm/region-store";
import { barBoundsFromThreeRegions } from "@/lib/stxm/regions";

function nanMinMax(image: number[][]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const row of image) {
    for (const value of row) {
      if (Number.isFinite(value)) {
        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  return { min, max };
}

/**
 * Load a scan header, preview image, and region bounds for the web UI.
 */
export function loadScan(hdrPath: string, downsample = 256): ScanPayload {
  const resolved = requireAllowedFile(hdrPath);
  const { meta, image } = loadStxm(resolved);
  const qaxis = meta.qaxis_points ?? [];
  const paxis = meta.paxis_points ?? [];
  const experimentDir = path.dirname(resolved);
  const saved = loadScanRegions(experimentDir, resolved);
  let izeroBounds: IzeroBounds;
  let regions: StxmRegion[];
  if (saved) {
    izeroBounds = { izero_lo: saved.izero_lo, izero_hi: saved.izero_hi };
    regions = saved.regions;
  } else {
    const [sampleLo, sampleHi, izeroLo, izeroHi] = barBoundsFromThreeRegions(image, qaxis);
    izeroBounds = { izero_lo: izeroLo, izero_hi: izeroHi };
    regions = [{ sample_lo: sampleLo, sample_hi: sampleHi, spot_label: "pure" }];
  }
  let preview = image;
  if (downsample > 0 && image.length > downsample) {
    const step = Math.max(1, Math.floor(image.length / downsample));
    preview = image.filter((_, idx) => idx % step === 0);
  }
  const { min: imageMin, max: imageMax } = nanMinMax(preview);
  return {
    ok: true,
    hdr_path: resolved,
    shape: [image.length, image[0]?.length ?? 0],
    paxis_name: meta.paxis_name ?? "Energy (eV)",
    qaxis_name: meta.qaxis_name ?? "Sample",
    paxis_points: paxis,
    qaxis_points: qaxis,
    regions,
    izero_bounds: izeroBounds,
    image: preview,
    image_min: imageMin,
    image_max: imageMax,
  };
}
