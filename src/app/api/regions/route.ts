import { NextResponse } from "next/server";

import { loadScanRegions, saveScanRegions } from "@/lib/stxm/region-store";
import { requireAllowedDirectory, requireAllowedFile } from "@/lib/stxm/path-utils";
import type { BridgeResponse } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const experimentDir = url.searchParams.get("experimentDir");
  const hdrPath = url.searchParams.get("hdrPath");
  if (!experimentDir || !hdrPath) {
    return NextResponse.json(
      { ok: false, error: "experimentDir and hdrPath are required" },
      { status: 400 },
    );
  }
  try {
    const experiment = requireAllowedDirectory(experimentDir);
    requireAllowedFile(hdrPath);
    const regions = loadScanRegions(experiment, hdrPath);
    const payload: BridgeResponse<{ regions: typeof regions }> = { ok: true, regions };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}

type SaveBody = {
  experimentDir: string;
  hdrPath: string;
  izero_lo: number;
  izero_hi: number;
  regions: Array<{ sample_lo: number; sample_hi: number; spot_label: string }>;
};

export async function POST(request: Request) {
  let body: SaveBody;
  try {
    body = (await request.json()) as SaveBody;
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.experimentDir || !body.hdrPath || !body.regions?.length) {
    return NextResponse.json(
      { ok: false, error: "experimentDir, hdrPath, and regions are required" },
      { status: 400 },
    );
  }
  try {
    const experiment = requireAllowedDirectory(body.experimentDir);
    requireAllowedFile(body.hdrPath);
    saveScanRegions(experiment, body.hdrPath, {
      izero_lo: body.izero_lo,
      izero_hi: body.izero_hi,
      regions: body.regions,
    });
    const payload: BridgeResponse<{ saved: boolean }> = { ok: true, saved: true };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}
