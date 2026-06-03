import { NextResponse } from "next/server";

import { listStoreManifest } from "@/lib/stxm/store";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const storeRoot = url.searchParams.get("storeRoot");
  if (!storeRoot) {
    return NextResponse.json({ ok: false, error: "storeRoot is required" }, { status: 400 });
  }
  try {
    const result = listStoreManifest(storeRoot);
    const payload = { ok: true as const, ...result };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}
