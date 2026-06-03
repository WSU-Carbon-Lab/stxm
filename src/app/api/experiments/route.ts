import { NextResponse } from "next/server";

import { listExperiments } from "@/lib/stxm/experiments";
import type { BridgeResponse } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const parentDir = url.searchParams.get("parentDir");
  if (!parentDir) {
    return NextResponse.json({ ok: false, error: "parentDir is required" }, { status: 400 });
  }
  try {
    const result = listExperiments(parentDir);
    const payload: BridgeResponse<{ experiments: string[]; parent_dir: string }> = {
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
