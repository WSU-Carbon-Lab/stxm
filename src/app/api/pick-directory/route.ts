import { NextResponse } from "next/server";

import { validatePathUnderAllowedRoots } from "@/lib/allowed-paths.server";
import {
  isDirectoryPickerEnabled,
  pickDirectoryNative,
} from "@/lib/pick-directory.server";

export async function POST() {
  if (!isDirectoryPickerEnabled()) {
    return NextResponse.json(
      {
        ok: false,
        error:
          "Directory picker is disabled. Use development mode or set STXM_ENABLE_DIRECTORY_PICKER=true for local use.",
      },
      { status: 403 },
    );
  }

  try {
    const chosen = await pickDirectoryNative();
    if (!chosen) {
      return NextResponse.json({ ok: true, cancelled: true });
    }

    const validation = validatePathUnderAllowedRoots(chosen);
    if (!validation.ok) {
      return NextResponse.json({ ok: false, error: validation.error }, { status: 400 });
    }

    return NextResponse.json({ ok: true, path: validation.resolved });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
