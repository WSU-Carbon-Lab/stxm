import { NextResponse } from "next/server";

import { parquetPreview } from "@/lib/stxm/parquet";
import type { BridgeResponse } from "@/lib/stxm-types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const parquetPath = url.searchParams.get("parquetPath");
  if (!parquetPath) {
    return NextResponse.json({ ok: false, error: "parquetPath is required" }, { status: 400 });
  }
  try {
    const result = await parquetPreview(parquetPath);
    const payload: BridgeResponse<typeof result> = { ok: true, ...result };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 400 },
    );
  }
}
