import { StxmWorkspace } from "@/components/stxm-workspace";
import { env } from "@/env.js";
import { isDirectoryPickerEnabled } from "@/lib/pick-directory.server";

export default function HomePage() {
  return (
    <StxmWorkspace
      initialParentDir={env.STXM_DEFAULT_PARENT_DIR ?? ""}
      directoryPickerEnabled={isDirectoryPickerEnabled()}
    />
  );
}
