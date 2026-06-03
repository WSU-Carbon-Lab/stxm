"use client";

type DataTableProps = {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  emptyMessage?: string;
};

export function DataTable({
  columns,
  rows,
  emptyMessage = "No rows to display.",
}: DataTableProps) {
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-300 p-6 text-sm text-zinc-500">
        {emptyMessage}
      </div>
    );
  }
  return (
    <div className="overflow-auto rounded-lg border border-zinc-200">
      <table className="min-w-full divide-y divide-zinc-200 text-left text-sm">
        <thead className="bg-zinc-50">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 font-semibold text-zinc-700">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 bg-white">
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column} className="px-3 py-2 font-mono text-xs text-zinc-800">
                  {formatCell(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}
