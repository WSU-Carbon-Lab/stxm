import { NextResponse } from "next/server";

import { listScans } from "@/lib/stxm/experiments";
import type { BridgeResponse } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const experimentDir = url.searchParams.get("experimentDir");
  if (!experimentDir) {
    return NextResponse.json({ ok: false, error: "experimentDir is required" }, { status: 400 });
  }
  try {
    const result = listScans(experimentDir);
    const payload: BridgeResponse<{ scans: string[]; experiment_dir: string }> = {
      ok: true,
      ...result,
    };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: error instanceof Error && error.message.includes("outside allowed") ? 400 : 500 },
    );
  }
}
