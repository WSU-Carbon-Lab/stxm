import { NextResponse } from "next/server";

import { parquetSpectra } from "@/lib/stxm/parquet";
import type { BridgeResponse, OverlaySeries } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const parquetPath = url.searchParams.get("parquetPath");
  if (!parquetPath) {
    return NextResponse.json({ ok: false, error: "parquetPath is required" }, { status: 400 });
  }
  try {
    const result = await parquetSpectra(parquetPath, {
      sampleName: url.searchParams.get("sampleName") ?? undefined,
      spotLabel: url.searchParams.get("spotLabel") ?? undefined,
      scanPath: url.searchParams.get("scanPath") ?? undefined,
      useNormalized: url.searchParams.get("useNormalized") === "true",
    });
    const payload: BridgeResponse<{ series: OverlaySeries[] }> = { ok: true, ...result };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}
