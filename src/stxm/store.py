"""Append-only filesystem spectrum store with JSON provenance sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stxm.estimators import WeightingMode
from stxm.reduction import RegionSpectrum

logger = logging.getLogger(__name__)

SPECTRUM_COLUMNS = [
    "energy_eV",
    "OD",
    "OD_err",
    "sample_name",
    "region_label",
    "edge",
    "scan_id",
    "weighting_mode",
    "reduction_method",
]


@dataclass
class Provenance:
    """
    Metadata required to reproduce a stored spectrum from raw beamline files.

    Attributes
    ----------
    sample_name : str
        Sample identifier.
    region_label : str
        Film or spot region label.
    edge : str
        Absorption edge name (e.g. ``C_K``).
    sample_bounds : dict
        Region geometry in axis coordinates.
    pre_edge : tuple[float, float]
        Pre-edge normalization window in eV.
    post_edge : tuple[float, float]
        Post-edge normalization window in eV.
    weighting_mode : str
        ``WeightingMode`` value.
    reduction_method : str
        Reduction path identifier.
    hdr_path : str
        Path to the source ``.hdr`` file.
    hdr_sha256 : str
        Hex digest of the ``.hdr`` bytes.
    xim_sha256 : str
        Hex digest of the paired ``.xim`` bytes.
    package_version : str
        Installed ``stxm`` version string.
    created_utc : str
        ISO-8601 UTC timestamp.
    imported_legacy : bool
        True when migrated from monolithic parquet without full metadata.
    extra : dict
        Additional JSON-serializable fields.
    """

    sample_name: str
    region_label: str
    edge: str
    sample_bounds: dict[str, float]
    pre_edge: tuple[float, float]
    post_edge: tuple[float, float]
    weighting_mode: str
    reduction_method: str
    hdr_path: str
    hdr_sha256: str
    xim_sha256: str
    package_version: str
    created_utc: str
    imported_legacy: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("stxm")
    except Exception:
        return "unknown"


def _partition_dir(
    store_root: Path,
    *,
    sample: str,
    region: str,
    edge: str,
) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", sample.strip()) or "sample"
    region_safe = re.sub(r"[^\w.\-]+", "_", region.strip()) or "region"
    edge_safe = re.sub(r"[^\w.\-]+", "_", edge.strip()) or "edge"
    return store_root / f"sample={safe}" / f"region={region_safe}" / f"edge={edge_safe}"


def _scan_id_from_hdr(hdr_path: Path) -> str:
    return hdr_path.stem


def write_spectrum(
    store_root: str | Path,
    spectrum: RegionSpectrum,
    provenance: Provenance,
    *,
    sample_name: str | None = None,
    edge: str | None = None,
    scan_id: str | None = None,
    od_normalized: np.ndarray | None = None,
) -> Path:
    """
    Atomically write one spectrum parquet and provenance JSON sidecar.

    Parameters
    ----------
    store_root : Path
        Root of the partitioned store tree.
    spectrum : RegionSpectrum
        Reduced spectrum columns.
    provenance : Provenance
        Reproducibility metadata serialized to JSON.
    sample_name : str, optional
        Overrides ``provenance.sample_name`` for partitioning.
    edge : str, optional
        Overrides ``provenance.edge``.
    scan_id : str, optional
        Filename stem; inferred from ``provenance.hdr_path`` when omitted.
    od_normalized : np.ndarray, optional
        Normalized OD column stored alongside raw OD.

    Returns
    -------
    Path
        Path to the written parquet file.

    Raises
    ------
    ValueError
        If required provenance fields are empty.
    OSError
        If the atomic rename fails.
    """
    store_root = Path(store_root)
    sn = sample_name or provenance.sample_name
    rg = provenance.region_label
    ed = edge or provenance.edge
    if not sn or not rg or not ed:
        raise ValueError("sample_name, region_label, and edge are required")
    part = _partition_dir(store_root, sample=sn, region=rg, edge=ed)
    part.mkdir(parents=True, exist_ok=True)
    hdr = Path(provenance.hdr_path)
    sid = scan_id or _scan_id_from_hdr(hdr)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{sid}__{stamp}"
    df = pd.DataFrame({
        "energy_eV": spectrum.energy_eV,
        "OD": spectrum.OD,
        "OD_err": spectrum.OD_err,
        "sample_name": sn,
        "region_label": rg,
        "edge": ed,
        "scan_id": sid,
        "weighting_mode": spectrum.weighting_mode,
        "reduction_method": spectrum.reduction_method,
    })
    if od_normalized is not None:
        df["OD_normalized"] = od_normalized
    parquet_path = part / f"{base}.parquet"
    json_path = part / f"{base}.json"
    _atomic_write_parquet(df, parquet_path)
    payload = asdict(provenance)
    payload["weighting_mode"] = provenance.weighting_mode or spectrum.weighting_mode
    payload["reduction_method"] = provenance.reduction_method or spectrum.reduction_method
    _atomic_write_json(payload, json_path)
    return parquet_path


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target = Path(target)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".parquet.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(target)
    finally:
        if tmp.exists() and not target.samefile(tmp):
            tmp.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], target: Path) -> None:
    target = Path(target)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".json.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(target)
    finally:
        if tmp.exists() and not target.samefile(tmp):
            tmp.unlink(missing_ok=True)


def _iter_spectrum_parquets(store_root: Path) -> list[Path]:
    if not store_root.is_dir():
        return []
    return sorted(store_root.glob("sample=*/region=*/edge=*/*.parquet"))


def query_spectra(
    store_root: str | Path,
    *,
    sample: str | None = None,
    region: str | None = None,
    edge: str | None = None,
) -> pd.DataFrame:
    """
    Glob partitioned spectra and return a concatenated table.

    Parameters
    ----------
    store_root : Path
        Store root directory.
    sample, region, edge : str, optional
        Partition filters; omitted keys match all.

    Returns
    -------
    pd.DataFrame
        Combined spectra; empty when no files match.
    """
    store_root = Path(store_root)
    paths = _iter_spectrum_parquets(store_root)
    frames: list[pd.DataFrame] = []
    for path in paths:
        parts = path.relative_to(store_root).parts
        if len(parts) < 4:
            continue
        part_sample = parts[0].split("=", 1)[-1]
        part_region = parts[1].split("=", 1)[-1]
        part_edge = parts[2].split("=", 1)[-1]
        if sample is not None and part_sample != sample:
            continue
        if region is not None and part_region != region:
            continue
        if edge is not None and part_edge != edge:
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("skip unreadable spectrum %s: %s", path, exc)
    if not frames:
        return pd.DataFrame(columns=SPECTRUM_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def list_manifest(store_root: str | Path) -> pd.DataFrame:
    """
    List one row per stored spectrum file with partition keys and paths.

    Parameters
    ----------
    store_root : Path
        Store root directory.

    Returns
    -------
    pd.DataFrame
        Manifest with columns ``sample``, ``region``, ``edge``, ``parquet_path``,
        ``json_path``, ``scan_id``, ``created_utc``.
    """
    store_root = Path(store_root)
    rows: list[dict[str, str]] = []
    for parquet_path in _iter_spectrum_parquets(store_root):
        parts = parquet_path.relative_to(store_root).parts
        if len(parts) < 4:
            continue
        json_path = parquet_path.with_suffix(".json")
        stem = parquet_path.stem
        scan_id = stem.split("__", 1)[0] if "__" in stem else stem
        created = stem.split("__", 1)[1] if "__" in stem else ""
        rows.append({
            "sample": parts[0].split("=", 1)[-1],
            "region": parts[1].split("=", 1)[-1],
            "edge": parts[2].split("=", 1)[-1],
            "parquet_path": str(parquet_path),
            "json_path": str(json_path) if json_path.exists() else "",
            "scan_id": scan_id,
            "created_utc": created,
        })
    return pd.DataFrame(rows)


def provenance_from_hdr(
    hdr_path: str | Path,
    *,
    sample_name: str,
    region_label: str,
    edge: str,
    sample_bounds: dict[str, float],
    pre_edge: tuple[float, float],
    post_edge: tuple[float, float],
    weighting_mode: WeightingMode | str,
    reduction_method: str,
    xim_path: str | Path | None = None,
) -> Provenance:
    """
    Build provenance with content hashes for a beamline scan pair.

    Parameters
    ----------
    hdr_path : Path
        Source ``.hdr`` path.
    sample_name, region_label, edge : str
        Identity fields for partitioning.
    sample_bounds : dict
        Region bounds recorded for reproduction.
    pre_edge, post_edge : tuple of float
        Normalization windows in eV.
    weighting_mode : WeightingMode or str
        Averaging mode identifier.
    reduction_method : str
        Reduction path identifier.
    xim_path : Path, optional
        Explicit ``.xim`` path; inferred from ``hdr_path`` when omitted.

    Returns
    -------
    Provenance
        Populated provenance record.

    Raises
    ------
    FileNotFoundError
        If ``hdr_path`` or the resolved ``.xim`` is missing.
    """
    hdr_path = Path(hdr_path)
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)
    if xim_path is None:
        stem = hdr_path.stem
        parent = hdr_path.parent
        candidate = parent / f"{stem}_a.xim"
        if not candidate.exists():
            candidate = parent / f"{stem}.xim"
        xim_path = candidate
    xim_path = Path(xim_path)
    if not xim_path.exists():
        raise FileNotFoundError(xim_path)
    if isinstance(weighting_mode, WeightingMode):
        mode_val = weighting_mode.value
    else:
        mode_val = str(weighting_mode)
    return Provenance(
        sample_name=sample_name,
        region_label=region_label,
        edge=edge,
        sample_bounds=sample_bounds,
        pre_edge=pre_edge,
        post_edge=post_edge,
        weighting_mode=mode_val,
        reduction_method=reduction_method,
        hdr_path=str(hdr_path.resolve()),
        hdr_sha256=_file_sha256(hdr_path),
        xim_sha256=_file_sha256(xim_path),
        package_version=_package_version(),
        created_utc=datetime.now(UTC).isoformat(),
    )


def import_legacy_parquet(
    store_root: str | Path,
    parquet_path: str | Path,
    *,
    edge: str = "unknown",
) -> int:
    """
    Migrate a monolithic experiment parquet into partitioned store files.

    Parameters
    ----------
    store_root : Path
        Destination store root.
    parquet_path : Path
        Legacy ``experiment.parquet`` path.
    edge : str
        Edge label applied to every imported row group.

    Returns
    -------
    int
        Number of spectra written.

    Raises
    ------
    FileNotFoundError
        If ``parquet_path`` does not exist.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    table = pd.read_parquet(parquet_path)
    if table.empty:
        return 0
    group_cols = [
        c
        for c in (
            "sample_name",
            "spot_label",
            "film_region_name",
            "scan_path",
            FORMULA_COLUMN,
        )
        if c in table.columns
    ]
    if not group_cols:
        group_cols = ["scan_path"] if "scan_path" in table.columns else []
    count = 0
    store_root = Path(store_root)
    if group_cols:
        grouped = table.groupby(group_cols, dropna=False)
    else:
        grouped = [((), table)]
    for keys, frame in grouped:
        if group_cols:
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            key_map = dict(zip(group_cols, key_tuple, strict=False))
        else:
            key_map = {}
        sample = str(key_map.get("sample_name") or key_map.get("film_region_name") or "imported")
        region = str(key_map.get("spot_label") or key_map.get("film_region_name") or "legacy")
        scan_path = key_map.get("scan_path")
        hdr = Path(str(scan_path)) if scan_path else parquet_path
        energy = frame["energy_eV"].to_numpy(dtype=np.float64)
        spectrum = RegionSpectrum(
            energy_eV=energy,
            OD=frame["OD"].to_numpy(dtype=np.float64),
            OD_err=frame["OD_err"].to_numpy(dtype=np.float64),
            region_label=region,
            weighting_mode=WeightingMode.INVERSE_COUNT.value,
            reduction_method="legacy_import",
            n_pixels=int(frame["n_sample"].iloc[0]) if "n_sample" in frame.columns else 0,
        )
        bounds = {"sample_lo": 0.0, "sample_hi": 0.0, "izero_lo": 0.0, "izero_hi": 0.0}
        prov = Provenance(
            sample_name=sample,
            region_label=region,
            edge=edge,
            sample_bounds=bounds,
            pre_edge=(0.0, 0.0),
            post_edge=(0.0, 0.0),
            weighting_mode=WeightingMode.INVERSE_COUNT.value,
            reduction_method="legacy_import",
            hdr_path=str(hdr),
            hdr_sha256="",
            xim_sha256="",
            package_version=_package_version(),
            created_utc=datetime.now(UTC).isoformat(),
            imported_legacy=True,
        )
        od_norm = (
            frame["OD_normalized"].to_numpy()
            if "OD_normalized" in frame.columns
            else None
        )
        write_spectrum(
            store_root,
            spectrum,
            prov,
            sample_name=sample,
            edge=edge,
            od_normalized=od_norm,
        )
        count += 1
    return count


FORMULA_COLUMN = "formula"
