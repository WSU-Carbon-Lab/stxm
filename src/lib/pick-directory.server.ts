import "server-only";

import { execFile } from "node:child_process";
import { platform } from "node:os";
import { promisify } from "node:util";

import { env } from "@/env.js";

const execFileAsync = promisify(execFile);

export function isDirectoryPickerEnabled(): boolean {
  if (env.NODE_ENV === "development") {
    return true;
  }
  return process.env.STXM_ENABLE_DIRECTORY_PICKER === "true";
}

function isPickerCancellation(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  const message = error.message.toLowerCase();
  return (
    message.includes("user canceled") ||
    message.includes("user cancelled") ||
    message.includes("(-128)")
  );
}

async function pickDirectoryDarwin(): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync("osascript", [
      "-e",
      'POSIX path of (choose folder with prompt "Select STXM beamtime folder (e.g. BL5321)")',
    ]);
    const chosen = stdout.trim();
    return chosen.length > 0 ? chosen : null;
  } catch (error) {
    if (isPickerCancellation(error)) {
      return null;
    }
    throw error;
  }
}

async function pickDirectoryLinux(): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync("zenity", ["--file-selection", "--directory"]);
    const chosen = stdout.trim();
    return chosen.length > 0 ? chosen : null;
  } catch (error) {
    if (isPickerCancellation(error)) {
      return null;
    }
    const code =
      error && typeof error === "object" && "code" in error ? String(error.code) : "";
    if (code === "ENOENT") {
      throw new Error("zenity is not installed; install it or set STXM_DEFAULT_PARENT_DIR");
    }
    throw error;
  }
}

export async function pickDirectoryNative(): Promise<string | null> {
  const system = platform();
  if (system === "darwin") {
    return pickDirectoryDarwin();
  }
  if (system === "linux") {
    return pickDirectoryLinux();
  }
  throw new Error(`Native directory picker is not supported on ${system}`);
}
