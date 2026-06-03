import { NextResponse } from "next/server";

import { queryStoreSpectra } from "@/lib/stxm/store";
import type { BridgeResponse } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const storeRoot = url.searchParams.get("storeRoot");
  if (!storeRoot) {
    return NextResponse.json({ ok: false, error: "storeRoot is required" }, { status: 400 });
  }
  try {
    const result = await queryStoreSpectra(storeRoot, {
      sample: url.searchParams.get("sampleName") ?? undefined,
      region: url.searchParams.get("regionLabel") ?? undefined,
      edge: url.searchParams.get("edge") ?? undefined,
    });
    const payload: BridgeResponse<typeof result> = { ok: true, ...result };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}
