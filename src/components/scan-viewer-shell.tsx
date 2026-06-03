"use client";

import type { ReactNode } from "react";

type ScanViewerShellProps = {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
};

export function ScanViewerShell({ open, title, onClose, children }: ScanViewerShellProps) {
  if (!open) {
    return null;
  }
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/50 p-4 md:p-8"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
          <h2 className="text-sm font-semibold text-zinc-900">{title}</h2>
          <button
            type="button"
            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-50"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <div className="flex min-h-[320px] flex-1 flex-col overflow-auto p-4">{children}</div>
      </div>
    </div>
  );
}
