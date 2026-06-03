import { NextResponse } from "next/server";

import { catalogExperiment } from "@/lib/stxm/catalog";
import type { BridgeResponse, ExperimentCatalogPayload } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const experimentDir = url.searchParams.get("experimentDir");
  if (!experimentDir) {
    return NextResponse.json({ ok: false, error: "experimentDir is required" }, { status: 400 });
  }
  try {
    const thumbnailsParam = url.searchParams.get("thumbnails");
    const thumbnails =
      thumbnailsParam === null ? true : thumbnailsParam.toLowerCase() !== "false";
    const result = await catalogExperiment(experimentDir, { thumbnails });
    const payload: BridgeResponse<ExperimentCatalogPayload> = { ok: true, ...result };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: error instanceof Error && error.message.includes("outside allowed") ? 400 : 500 },
    );
  }
}
