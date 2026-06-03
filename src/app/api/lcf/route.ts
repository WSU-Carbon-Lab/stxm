import { NextResponse } from "next/server";

import { runStxmBridge } from "@/lib/python-bridge.server";
import type { BridgeResponse } from "@/lib/stxm-types";

type LcfBody = {
  target: { energy_eV: number[]; OD: number[]; OD_err?: number[] };
  components: Array<{
    name: string;
    energy_eV: number[];
    OD: number[];
    OD_err?: number[];
    initial?: number;
    minimum?: number;
    maximum?: number;
    fixed?: boolean;
  }>;
};

export async function POST(request: Request) {
  let body: LcfBody;
  try {
    body = (await request.json()) as LcfBody;
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.target || !body.components?.length) {
    return NextResponse.json(
      { ok: false, error: "target and components are required" },
      { status: 400 },
    );
  }
  try {
    const payload = await runStxmBridge<
      BridgeResponse<{
        fractions: Record<string, number>;
        fraction_errors: Record<string, number>;
        reduced_chi_square: number;
        energy_eV: number[];
        target: number[];
        model: number[];
        residual: number[];
      }>
    >("lcf-fit", [
      "--target-json",
      JSON.stringify(body.target),
      "--components-json",
      JSON.stringify(body.components),
    ]);
    return NextResponse.json(payload, { status: payload.ok ? 200 : 400 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 },
    );
  }
}
