"""JSON CLI bridge for the Next.js local app to invoke STXM processing.

Commands include ``list-experiments``, ``list-scans``, ``catalog-experiment`` (Finder-style
scan listing with thumbnails and type grouping), ``load-scan``, ``reduce-scan``, region
persistence, parquet preview, store queries, and ``lcf-fit``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stxm.allowed_paths import allowed_roots_from_env, resolve_path_under_roots
from stxm.estimators import WeightingMode
from stxm.experiment import load_experiment_parquet
from stxm.io import (
    is_nexafs_line_scan,
    list_experiment_hdr_files,
    list_nexafs_line_scans,
    load_stxm,
    parse_hdr_scan_type,
    scan_type_category,
    thumbnail_png_base64,
)
from stxm.lcf import Spectrum, fit_lcf, preview_lcf_model
from stxm.nexafs import nexafs_beer_lambert
from stxm.normalization import NormalizationMode, normalize_nexafs_with_metadata
from stxm.region_store import load_scan_regions, save_scan_regions
from stxm.regions import bar_bounds_from_three_regions, sample_izero_masks
from stxm.store import list_manifest, query_spectra
from stxm.absorption import mass_absorption_cm2_per_g
from stxm.transforms import normalize_spot_label


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, default=_json_default)
    sys.stdout.write("\n")


def _fail(message: str, *, code: int = 1) -> None:
    _emit({"ok": False, "error": message})
    raise SystemExit(code)


def _resolve_path(path: str, allowed_roots: list[str] | None) -> Path:
    try:
        return resolve_path_under_roots(path, allowed_roots)
    except ValueError as exc:
        _fail(str(exc))
        return Path(path)


def _require_hdr_path(path: str, allowed_roots: list[str] | None) -> Path:
    hdr_path = _resolve_path(path, allowed_roots)
    if not hdr_path.is_file():
        _fail(f"Scan header not found: {hdr_path}")
    return hdr_path.resolve()


def experiment_sort_key(name: str) -> tuple[int, int, str]:
    base = name.strip()
    if "(" in base:
        base = base.split("(", 1)[0]
    base = base.replace("_", "-")
    parts = base.split("-", 2)
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
    except Exception:
        return (0, 0, name)
    return (year, month, name)


def cmd_list_experiments(args: argparse.Namespace) -> None:
    parent = _resolve_path(args.parent_dir, args.allowed_root)
    if not parent.is_dir():
        _fail(f"Not a directory: {parent}")
    names = sorted(
        [entry.name for entry in parent.iterdir() if entry.is_dir()],
        key=experiment_sort_key,
        reverse=True,
    )
    _emit({"ok": True, "parent_dir": str(parent), "experiments": names})


def cmd_list_scans(args: argparse.Namespace) -> None:
    experiment = _resolve_path(args.experiment_dir, args.allowed_root)
    scans = [path.name for path in list_nexafs_line_scans(experiment)]
    _emit({"ok": True, "experiment_dir": str(experiment), "scans": scans})


def cmd_catalog_experiment(args: argparse.Namespace) -> None:
    experiment = _resolve_path(args.experiment_dir, args.allowed_root)
    if not experiment.is_dir():
        _fail(f"Not a directory: {experiment}")
    entries: list[dict[str, Any]] = []
    for hdr_path in list_experiment_hdr_files(experiment):
        scan_type = parse_hdr_scan_type(hdr_path)
        resolved_hdr = hdr_path.resolve()
        record: dict[str, Any] = {
            "basename": resolved_hdr.name,
            "hdr_path": str(resolved_hdr),
            "scan_type": scan_type,
            "category": scan_type_category(scan_type),
            "is_nexafs_line_scan": is_nexafs_line_scan(resolved_hdr),
        }
        try:
            meta, image = load_stxm(resolved_hdr)
            record["shape"] = [int(image.shape[0]), int(image.shape[1])]
            record["paxis_count"] = int(meta["paxis_count"])
            record["qaxis_count"] = int(meta["qaxis_count"])
            for key in ("energy_eV", "energy_min_eV", "energy_max_eV", "num_energy_points"):
                if key in meta and meta[key] is not None:
                    value = meta[key]
                    record[key] = float(value) if "eV" in key else int(value)
            thumb = thumbnail_png_base64(image, max_size=int(args.thumbnail_size))
            if thumb:
                record["thumbnail_png_base64"] = thumb
        except Exception:
            record["shape"] = None
        entries.append(record)
    _emit({"ok": True, "experiment_dir": str(experiment), "entries": entries})


def cmd_load_scan(args: argparse.Namespace) -> None:
    hdr_path = _require_hdr_path(args.hdr_path, args.allowed_root)
    meta, image = load_stxm(hdr_path)
    qaxis = np.asarray(meta["qaxis_points"], dtype=np.float64)
    paxis = np.asarray(meta["paxis_points"], dtype=np.float64)
    saved = load_scan_regions(hdr_path.parent, hdr_path)
    if saved is not None:
        izero_bounds = {
            "izero_lo": float(saved["izero_lo"]),
            "izero_hi": float(saved["izero_hi"]),
        }
        regions = saved["regions"]
    else:
        sample_lo, sample_hi, izero_lo, izero_hi = bar_bounds_from_three_regions(image, qaxis)
        izero_bounds = {"izero_lo": izero_lo, "izero_hi": izero_hi}
        regions = [
            {
                "sample_lo": sample_lo,
                "sample_hi": sample_hi,
                "spot_label": "pure",
            }
        ]
    preview = image
    if args.downsample and image.shape[0] > args.downsample:
        step = max(1, image.shape[0] // args.downsample)
        preview = image[::step, :]
    _emit(
        {
            "ok": True,
            "hdr_path": str(hdr_path),
            "shape": list(image.shape),
            "paxis_name": meta.get("paxis_name", "Energy (eV)"),
            "qaxis_name": meta.get("qaxis_name", "Sample"),
            "paxis_points": paxis,
            "qaxis_points": qaxis,
            "regions": regions,
            "izero_bounds": izero_bounds,
            "image": preview.tolist(),
            "image_min": float(np.nanmin(preview)),
            "image_max": float(np.nanmax(preview)),
        }
    )


def cmd_mass_absorption(args: argparse.Namespace) -> None:
    formula = (args.formula or "").strip()
    if not formula:
        _fail("formula is required")
    try:
        energy = np.asarray(json.loads(args.energy_json), dtype=np.float64)
    except json.JSONDecodeError as exc:
        _fail(f"Invalid energy JSON: {exc}")
    if energy.size == 0:
        _emit({"ok": True, "mu_rho_cm2_per_g": []})
        return
    mu = mass_absorption_cm2_per_g(formula, energy, None)
    _emit({"ok": True, "mu_rho_cm2_per_g": np.asarray(mu, dtype=np.float64).tolist()})


def cmd_reduce_scan(args: argparse.Namespace) -> None:
    hdr_path = _require_hdr_path(args.hdr_path, args.allowed_root)
    meta, image = load_stxm(hdr_path)
    qaxis = np.asarray(meta["qaxis_points"], dtype=np.float64)
    paxis = np.asarray(meta["paxis_points"], dtype=np.float64)
    regions_payload = json.loads(args.regions_json)
    izero_payload = json.loads(args.izero_json)
    iz_lo = float(izero_payload["izero_lo"])
    iz_hi = float(izero_payload["izero_hi"])
    _, izero_mask = sample_izero_masks(qaxis, 0.0, 0.0, iz_lo, iz_hi)
    if not np.any(izero_mask):
        _fail("izero region selects no rows")
    mode = WeightingMode(args.weighting_mode)
    norm_mode = NormalizationMode(args.normalization_mode)
    pre_lo, pre_hi = (float(x) for x in args.pre_edge.split(","))
    post_lo, post_hi = (float(x) for x in args.post_edge.split(","))
    out_rows: list[dict[str, Any]] = []
    for reg in regions_payload:
        sa_lo = float(reg["sample_lo"])
        sa_hi = float(reg["sample_hi"])
        sample_mask, _ = sample_izero_masks(qaxis, sa_lo, sa_hi, iz_lo, iz_hi)
        if not np.any(sample_mask):
            continue
        od, sigma_od, i0, sigma_i0, intensity, sigma_i, n_sample, n_izero = nexafs_beer_lambert(
            image,
            sample_mask,
            izero_mask,
            mode=mode,
        )
        od_norm, norm_meta = normalize_nexafs_with_metadata(
            paxis,
            od,
            pre_lo,
            pre_hi,
            post_lo,
            post_hi,
            mode=norm_mode,
        )
        spot_label = normalize_spot_label(reg.get("spot_label"))
        out_rows.append(
            {
                "spot_label": spot_label,
                "sample_lo": sa_lo,
                "sample_hi": sa_hi,
                "energy_eV": paxis,
                "OD": od,
                "OD_err": sigma_od,
                "OD_normalized": od_norm,
                "I0": i0,
                "I0_err": sigma_i0,
                "I": intensity,
                "I_err": sigma_i,
                "n_sample": int(n_sample),
                "n_izero": int(n_izero),
                "normalization": norm_meta,
            }
        )
    _emit({"ok": True, "hdr_path": str(hdr_path), "spectra": out_rows})


def cmd_load_regions(args: argparse.Namespace) -> None:
    experiment = _resolve_path(args.experiment_dir, args.allowed_root)
    hdr_path = _resolve_path(args.hdr_path, args.allowed_root)
    regions = load_scan_regions(experiment, hdr_path)
    _emit({"ok": True, "regions": regions})


def cmd_save_regions(args: argparse.Namespace) -> None:
    experiment = _resolve_path(args.experiment_dir, args.allowed_root)
    hdr_path = _resolve_path(args.hdr_path, args.allowed_root)
    payload = json.loads(args.regions_json)
    save_scan_regions(
        experiment,
        hdr_path,
        izero_lo=float(payload["izero_lo"]),
        izero_hi=float(payload["izero_hi"]),
        regions=payload["regions"],
    )
    _emit({"ok": True, "saved": True})


def cmd_parquet_preview(args: argparse.Namespace) -> None:
    parquet_path = _resolve_path(args.parquet_path, args.allowed_root)
    df = load_experiment_parquet(parquet_path)
    columns = list(df.columns)
    sample_names = (
        sorted(df["sample_name"].dropna().unique().tolist()) if "sample_name" in df else []
    )
    spot_labels = sorted(df["spot_label"].dropna().unique().tolist()) if "spot_label" in df else []
    scan_paths = sorted(df["scan_path"].dropna().unique().tolist()) if "scan_path" in df else []
    _emit(
        {
            "ok": True,
            "parquet_path": str(parquet_path),
            "row_count": int(len(df)),
            "columns": columns,
            "sample_names": sample_names,
            "spot_labels": spot_labels,
            "scan_paths": scan_paths,
        }
    )


def cmd_parquet_spectra(args: argparse.Namespace) -> None:
    parquet_path = _resolve_path(args.parquet_path, args.allowed_root)
    df = load_experiment_parquet(parquet_path)
    if args.sample_name:
        df = df[df["sample_name"] == args.sample_name]
    if args.spot_label:
        df = df[df["spot_label"] == args.spot_label]
    if args.scan_path:
        df = df[df["scan_path"] == args.scan_path]
    y_col = "OD_normalized" if args.use_normalized and "OD_normalized" in df.columns else "OD"
    y_err_col = "OD_err"
    groups = []
    group_cols = [
        column
        for column in ("sample_name", "spot_label", "scan_path", "film_region_name")
        if column in df.columns
    ]
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label_parts = [f"{col}={val}" for col, val in zip(group_cols, keys, strict=False)]
        groups.append(
            {
                "label": ", ".join(label_parts),
                "energy_eV": group["energy_eV"].to_numpy(dtype=float).tolist(),
                "y": group[y_col].to_numpy(dtype=float).tolist(),
                "y_err": (
                    group[y_err_col].to_numpy(dtype=float).tolist() if y_err_col in group else []
                ),
            }
        )
    _emit({"ok": True, "series": groups})


def cmd_store_manifest(args: argparse.Namespace) -> None:
    store_root = _resolve_path(args.store_root, args.allowed_root)
    manifest = list_manifest(store_root)
    entries = manifest.to_dict(orient="records") if not manifest.empty else []
    _emit({"ok": True, "store_root": str(store_root), "entries": entries})


def cmd_store_query(args: argparse.Namespace) -> None:
    store_root = _resolve_path(args.store_root, args.allowed_root)
    df = query_spectra(
        store_root,
        sample=args.sample_name or None,
        region=args.region_label or None,
        edge=args.edge or None,
    )
    groups = []
    if not df.empty and "sample_name" in df.columns and "region_label" in df.columns:
        grouped = df.groupby(["sample_name", "region_label"], dropna=False)
        for (sample_name, region_label), group in grouped:
            groups.append(
                {
                    "label": f"{sample_name} / {region_label}",
                    "energy_eV": group["energy_eV"].to_numpy(dtype=float).tolist(),
                    "OD": group["OD"].to_numpy(dtype=float).tolist(),
                    "OD_err": group["OD_err"].to_numpy(dtype=float).tolist(),
                }
            )
    _emit({"ok": True, "series": groups})


def cmd_lcf_fit(args: argparse.Namespace) -> None:
    target = json.loads(args.target_json)
    components = json.loads(args.components_json)
    target_spec = Spectrum(
        energy_eV=np.asarray(target["energy_eV"], dtype=float),
        OD=np.asarray(target["OD"], dtype=float),
        OD_err=np.asarray(target.get("OD_err", target["OD"]), dtype=float) * 0.0 + 0.01,
        label="target",
    )
    references: list[Spectrum] = []
    initial_fractions: list[float] = []
    fraction_bounds: list[tuple[float, float]] = []
    fixed: list[bool] = []
    for comp in components:
        references.append(
            Spectrum(
                energy_eV=np.asarray(comp["energy_eV"], dtype=float),
                OD=np.asarray(comp["OD"], dtype=float),
                OD_err=np.asarray(comp.get("OD_err", comp["OD"]), dtype=float) * 0.0 + 0.01,
                label=str(comp["name"]),
            )
        )
        initial_fractions.append(float(comp.get("initial", 0.0)) / 100.0)
        fraction_bounds.append(
            (
                float(comp.get("minimum", 0.0)) / 100.0,
                float(comp.get("maximum", 100.0)) / 100.0,
            )
        )
        fixed.append(bool(comp.get("fixed", False)))
    result = fit_lcf(
        target_spec,
        references,
        sum_to_one=True,
        initial_fractions=np.asarray(initial_fractions, dtype=float),
        fraction_bounds=fraction_bounds,
        fixed=fixed,
    )
    grid, model, target_on_grid = preview_lcf_model(
        target_spec,
        references,
        result.fractions,
        result.energy_grid,
    )
    fraction_errors = np.sqrt(np.maximum(np.diag(result.fraction_covariance), 0.0))
    _emit(
        {
            "ok": True,
            "fractions": {
                label: float(frac)
                for label, frac in zip(result.reference_labels, result.fractions, strict=True)
            },
            "fraction_errors": {
                label: float(err)
                for label, err in zip(result.reference_labels, fraction_errors, strict=True)
            },
            "reduced_chi_square": result.reduced_chi_square,
            "energy_eV": grid.tolist(),
            "target": target_on_grid.tolist(),
            "model": model.tolist(),
            "residual": result.residual.tolist(),
        }
    )


def _allowed_roots_from_env() -> list[str]:
    return allowed_roots_from_env()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STXM JSON bridge for local Next.js app")
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="Repeatable path root; defaults to STXM_ALLOWED_ROOTS or user home",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-experiments")
    p.add_argument("parent_dir")
    p.set_defaults(func=cmd_list_experiments)

    p = sub.add_parser("list-scans")
    p.add_argument("experiment_dir")
    p.set_defaults(func=cmd_list_scans)

    p = sub.add_parser("catalog-experiment")
    p.add_argument("experiment_dir")
    p.add_argument("--thumbnail-size", type=int, default=128)
    p.set_defaults(func=cmd_catalog_experiment)

    p = sub.add_parser("load-scan")
    p.add_argument("hdr_path")
    p.add_argument("--downsample", type=int, default=256)
    p.set_defaults(func=cmd_load_scan)

    p = sub.add_parser("mass-absorption")
    p.add_argument("--formula", required=True)
    p.add_argument("--energy-json", required=True)
    p.set_defaults(func=cmd_mass_absorption)

    p = sub.add_parser("reduce-scan")
    p.add_argument("hdr_path")
    p.add_argument("--regions-json", required=True)
    p.add_argument("--izero-json", required=True)
    p.add_argument("--weighting-mode", default=WeightingMode.POISSON_MLE.value)
    p.add_argument("--normalization-mode", default=NormalizationMode.PRE_EDGE_SCALE.value)
    p.add_argument("--pre-edge", default="280,283")
    p.add_argument("--post-edge", default="292,310")
    p.set_defaults(func=cmd_reduce_scan)

    p = sub.add_parser("load-regions")
    p.add_argument("experiment_dir")
    p.add_argument("hdr_path")
    p.set_defaults(func=cmd_load_regions)

    p = sub.add_parser("save-regions")
    p.add_argument("experiment_dir")
    p.add_argument("hdr_path")
    p.add_argument("--regions-json", required=True)
    p.set_defaults(func=cmd_save_regions)

    p = sub.add_parser("parquet-preview")
    p.add_argument("parquet_path")
    p.set_defaults(func=cmd_parquet_preview)

    p = sub.add_parser("parquet-spectra")
    p.add_argument("parquet_path")
    p.add_argument("--sample-name", default="")
    p.add_argument("--spot-label", default="")
    p.add_argument("--scan-path", default="")
    p.add_argument("--use-normalized", action="store_true")
    p.set_defaults(func=cmd_parquet_spectra)

    p = sub.add_parser("store-manifest")
    p.add_argument("store_root")
    p.set_defaults(func=cmd_store_manifest)

    p = sub.add_parser("store-query")
    p.add_argument("store_root")
    p.add_argument("--sample-name", default="")
    p.add_argument("--region-label", default="")
    p.add_argument("--edge", default="")
    p.set_defaults(func=cmd_store_query)

    p = sub.add_parser("lcf-fit")
    p.add_argument("--target-json", required=True)
    p.add_argument("--components-json", required=True)
    p.set_defaults(func=cmd_lcf_fit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.allowed_root is None:
        args.allowed_root = _allowed_roots_from_env()
    try:
        args.func(args)
    except SystemExit:
        raise
    except FileNotFoundError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(str(exc))


if __name__ == "__main__":
    main()
