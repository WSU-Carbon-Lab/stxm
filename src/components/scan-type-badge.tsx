import type { ScanCategory } from "@/lib/stxm-types";

const CATEGORY_LABELS: Record<ScanCategory, string> = {
  line_scan: "Line scans",
  image_scan: "Image scans",
  fixed_point: "Fixed point",
  focus_scan: "Focus scans",
  stack: "Stacks",
  other: "Other",
};

const CATEGORY_STYLES: Record<ScanCategory, string> = {
  line_scan: "bg-sky-100 text-sky-900 ring-sky-200",
  image_scan: "bg-emerald-100 text-emerald-900 ring-emerald-200",
  fixed_point: "bg-amber-100 text-amber-900 ring-amber-200",
  focus_scan: "bg-violet-100 text-violet-900 ring-violet-200",
  stack: "bg-zinc-200 text-zinc-800 ring-zinc-300",
  other: "bg-zinc-100 text-zinc-700 ring-zinc-200",
};

export const SCAN_CATEGORY_ORDER: ScanCategory[] = [
  "line_scan",
  "image_scan",
  "fixed_point",
  "focus_scan",
  "stack",
  "other",
];

export function scanCategoryLabel(category: ScanCategory): string {
  return CATEGORY_LABELS[category];
}

type ScanTypeBadgeProps = {
  category: ScanCategory;
  className?: string;
};

export function ScanTypeBadge({ category, className = "" }: ScanTypeBadgeProps) {
  return (
    <span
      className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-inset ${CATEGORY_STYLES[category]} ${className}`}
    >
      {scanCategoryLabel(category)}
    </span>
  );
}
