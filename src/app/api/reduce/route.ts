import { NextResponse } from "next/server";

import { parseEdgeRange, regionRawSpectra, reduceScan } from "@/lib/stxm/reduce";
import type { BridgeResponse, SpectrumSeries } from "@/lib/stxm-types";
import type { NormalizationMode } from "@/lib/stxm/normalization";
import type { WeightingMode } from "@/lib/stxm/estimators";

type ReduceBody = {
  hdrPath: string;
  regions: Array<{ sample_lo: number; sample_hi: number; spot_label: string }>;
  izero: { izero_lo: number; izero_hi: number };
  raw?: boolean;
  weightingMode?: string;
  normalizationMode?: string;
  preEdge?: string;
  postEdge?: string;
  formula?: string;
  bareAtomFitOffset?: boolean;
};

export async function POST(request: Request) {
  let body: ReduceBody;
  try {
    body = (await request.json()) as ReduceBody;
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.hdrPath || !body.regions?.length || !body.izero) {
    return NextResponse.json(
      { ok: false, error: "hdrPath, regions, and izero are required" },
      { status: 400 },
    );
  }
  try {
    const shared = {
      hdrPath: body.hdrPath,
      regions: body.regions,
      izero: body.izero,
      weightingMode: (body.weightingMode ?? "poisson_mle") as WeightingMode,
    };
    const result = body.raw
      ? regionRawSpectra(shared)
      : await reduceScan({
          ...shared,
          normalizationMode: (body.normalizationMode ?? "pre_edge_scale") as NormalizationMode,
          preEdge: parseEdgeRange(body.preEdge, [280, 283]),
          postEdge: parseEdgeRange(body.postEdge, [292, 310]),
          formula: body.formula,
          bareAtomFitOffset: body.bareAtomFitOffset,
        });
    const payload: BridgeResponse<{ spectra: SpectrumSeries[]; hdr_path: string }> = {
      ok: true,
      ...result,
    };
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status =
      message.includes("not found") ||
      message.includes("selects no rows") ||
      message.includes("No sample regions overlap")
        ? 400
        : 500;
    return NextResponse.json({ ok: false, error: message }, { status });
  }
}
