# STXM

Local STXM line-scan toolkit with a Next.js App Router UI and Python processing backend.

## Layout

- `src/python/stxm/` — Python package (reduction, parquet, store, LCF, JSON bridge CLI)
- `src/app/` — Next.js App Router pages and API routes
- `src/components/` — Shared React UI (workspace header, heatmap, spectrum charts, tables)
- `tests/` — Python pytest suite

## Python

```bash
uv sync --all-groups
uv run pytest
uv run ruff check src/python tests
```

The bridge CLI exposes JSON commands for the web app:

```bash
uv run stxm-bridge list-experiments /path/to/beamtime/parent
```

## Next.js (create-t3-app)

Initialized with create-t3-app **7.40.0** (minimal stack: TypeScript, Tailwind CSS v4, App Router, ESLint; no auth/DB/tRPC):

```bash
bunx create-t3-app@latest stxm-t3-scaffold --CI --noGit --tailwind true --nextAuth false --prisma false --drizzle false --trpc false --appRouter true --eslint true -i "@/*"
```

The scaffold was generated in a temporary directory and merged into this repository root because the repo was already populated.

Install and run:

```bash
bun install
bun run dev
```

Build / typecheck:

```bash
bun run build
bun run check
```

## Environment

Copy `.env.example` to `.env` and set filesystem roots:

- `STXM_ALLOWED_ROOTS` — optional colon-separated allowed roots; when unset, defaults to home (and macOS `~/Library/CloudStorage` / OneDrive folders)
- `STXM_DEFAULT_PARENT_DIR` — default parent directory of experiment folders in the UI

## API routes

| Route | Purpose |
| --- | --- |
| `GET /api/experiments` | List experiment subdirectories |
| `GET /api/scans` | List NEXAFS line scans in an experiment |
| `GET /api/scan` | Load scan image, axes, default/saved regions |
| `POST /api/reduce` | Reduce current regions to spectra |
| `GET/POST /api/regions` | Load/save `regions.json` |
| `GET /api/parquet/preview` | Parquet catalog summary |
| `GET /api/parquet/spectra` | Filtered parquet overlay spectra |
| `GET /api/store/manifest` | Partitioned store manifest |
| `GET /api/store/query` | Store spectrum query |
| `POST /api/lcf` | Linear combination fit |

Python is invoked server-side via `uv run stxm-bridge ...` from `src/lib/python-bridge.ts`.
