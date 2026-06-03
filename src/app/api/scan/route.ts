import { NextResponse } from "next/server";

import { loadScan } from "@/lib/stxm/load-scan";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const hdrPath = url.searchParams.get("hdrPath");
  if (!hdrPath) {
    return NextResponse.json({ ok: false, error: "hdrPath is required" }, { status: 400 });
  }
  try {
    const result = loadScan(hdrPath);
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = message.includes("not found") || message.includes("outside allowed") ? 400 : 500;
    return NextResponse.json({ ok: false, error: message }, { status });
  }
}
