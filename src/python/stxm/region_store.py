"""Persist per-scan ROI bar bounds and spot labels in experiment directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stxm.regions import bar_bounds_from_three_regions
from stxm.transforms import normalize_spot_label

REGIONS_CONFIG_FILENAME = "regions.json"
REGIONS_SCHEMA_VERSION = 1


def regions_config_path(experiment_dir: str | Path) -> Path:
    """
    Resolve the regions config file inside an experiment directory.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment folder that contains beamline ``.hdr`` scans.

    Returns
    -------
    Path
        Path to ``regions.json`` under ``experiment_dir``.
    """
    return Path(experiment_dir) / REGIONS_CONFIG_FILENAME


def scan_key_from_path(scan_path: str | Path) -> str:
    """
    Derive the stable lookup key for a scan inside ``regions.json``.

    Parameters
    ----------
    scan_path : str or Path
        Absolute or relative path to a ``.hdr`` file.

    Returns
    -------
    str
        Basename of the scan path (for example ``scan001.hdr``).
    """
    return Path(scan_path).name


def normalize_region_entry(entry: dict[str, Any]) -> dict[str, float | str]:
    """
    Validate and normalize one saved sample-region record.

    Parameters
    ----------
    entry : dict
        Mapping with ``sample_lo``, ``sample_hi``, and optional ``spot_label``.

    Returns
    -------
    dict
        Normalized region with float bounds and a non-empty spot label.

    Raises
    ------
    ValueError
        If required bounds are missing or not numeric.
    """
    if "sample_lo" not in entry or "sample_hi" not in entry:
        raise ValueError("region entry requires sample_lo and sample_hi")
    sample_lo = float(entry["sample_lo"])
    sample_hi = float(entry["sample_hi"])
    if sample_lo > sample_hi:
        sample_lo, sample_hi = sample_hi, sample_lo
    return {
        "sample_lo": sample_lo,
        "sample_hi": sample_hi,
        "spot_label": normalize_spot_label(entry.get("spot_label")),
    }


def normalize_saved_scan(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Validate one scan entry from ``regions.json``.

    Parameters
    ----------
    raw : dict
        Raw scan payload with ``izero_lo``, ``izero_hi``, and ``regions``.

    Returns
    -------
    dict
        Normalized scan payload ready for widget state.

    Raises
    ------
    ValueError
        If izero bounds or the region list are invalid.
    """
    if "izero_lo" not in raw or "izero_hi" not in raw:
        raise ValueError("scan entry requires izero_lo and izero_hi")
    izero_lo = float(raw["izero_lo"])
    izero_hi = float(raw["izero_hi"])
    if izero_lo > izero_hi:
        izero_lo, izero_hi = izero_hi, izero_lo
    regions_raw = raw.get("regions")
    if not isinstance(regions_raw, list) or not regions_raw:
        raise ValueError("scan entry requires a non-empty regions list")
    regions = [normalize_region_entry(r) for r in regions_raw if isinstance(r, dict)]
    if not regions:
        raise ValueError("scan entry has no valid regions")
    return {
        "izero_lo": izero_lo,
        "izero_hi": izero_hi,
        "regions": regions,
    }


def default_regions_from_image(image, qaxis) -> dict[str, Any]:
    """
    Build default izero and sample bounds from three-region segmentation.

    Parameters
    ----------
    image : array-like
        2D scan array with rows along the sample axis.
    qaxis : array-like
        Sample-axis coordinates, one value per row.

    Returns
    -------
    dict
        Payload with ``izero_lo``, ``izero_hi``, and a single default region.
    """
    bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi = bar_bounds_from_three_regions(
        image, qaxis
    )
    return {
        "izero_lo": float(bar_izero_lo),
        "izero_hi": float(bar_izero_hi),
        "regions": [
            {
                "sample_lo": float(bar_sample_lo),
                "sample_hi": float(bar_sample_hi),
                "spot_label": "pure",
            }
        ],
    }


def load_regions_config(experiment_dir: str | Path) -> dict[str, Any]:
    """
    Load the full regions config for an experiment directory.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment folder containing ``regions.json``.

    Returns
    -------
    dict
        Parsed config with ``version`` and ``scans`` keys. Returns an empty
        scaffold when the file is missing or unreadable.
    """
    path = regions_config_path(experiment_dir)
    if not path.is_file():
        return {"version": REGIONS_SCHEMA_VERSION, "scans": {}}
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"version": REGIONS_SCHEMA_VERSION, "scans": {}}
    if not isinstance(raw, dict):
        return {"version": REGIONS_SCHEMA_VERSION, "scans": {}}
    scans = raw.get("scans")
    if not isinstance(scans, dict):
        scans = {}
    version = raw.get("version", REGIONS_SCHEMA_VERSION)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = REGIONS_SCHEMA_VERSION
    return {"version": version, "scans": scans}


def save_regions_config(experiment_dir: str | Path, config: dict[str, Any]) -> None:
    """
    Write the full regions config atomically to ``regions.json``.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment folder that receives ``regions.json``.
    config : dict
        Config with ``version`` and ``scans`` keys.

    Raises
    ------
    OSError
        If the config file cannot be written.
    ValueError
        If ``config`` is missing required top-level keys.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    scans = config.get("scans")
    if not isinstance(scans, dict):
        raise ValueError("config.scans must be a dict")
    version = config.get("version", REGIONS_SCHEMA_VERSION)
    path = regions_config_path(experiment_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": int(version), "scans": scans}
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_scan_regions(experiment_dir: str | Path, scan_path: str | Path) -> dict[str, Any] | None:
    """
    Load saved ROI bounds for one scan, if present and valid.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment folder containing ``regions.json``.
    scan_path : str or Path
        Path to the ``.hdr`` file whose regions should be restored.

    Returns
    -------
    dict or None
        Normalized payload with ``izero_lo``, ``izero_hi``, and ``regions``,
        or ``None`` when no valid saved entry exists.
    """
    key = scan_key_from_path(scan_path)
    config = load_regions_config(experiment_dir)
    scans = config.get("scans", {})
    raw = scans.get(key)
    if not isinstance(raw, dict):
        return None
    try:
        return normalize_saved_scan(raw)
    except ValueError:
        return None


def save_scan_regions(
    experiment_dir: str | Path,
    scan_path: str | Path,
    *,
    izero_lo: float,
    izero_hi: float,
    regions: list[dict[str, Any]],
) -> None:
    """
    Persist ROI bounds for one scan into ``regions.json``.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment folder that receives ``regions.json``.
    scan_path : str or Path
        Path to the ``.hdr`` file being edited.
    izero_lo, izero_hi : float
        Izero bar positions in sample-axis coordinates.
    regions : list of dict
        Sample regions with ``sample_lo``, ``sample_hi``, and ``spot_label``.

    Raises
    ------
    ValueError
        If ``regions`` is empty or any entry is invalid.
    OSError
        If the config file cannot be written.
    """
    if not regions:
        raise ValueError("regions must be non-empty")
    normalized = normalize_saved_scan(
        {
            "izero_lo": izero_lo,
            "izero_hi": izero_hi,
            "regions": regions,
        }
    )
    key = scan_key_from_path(scan_path)
    config = load_regions_config(experiment_dir)
    scans = dict(config.get("scans", {}))
    scans[key] = normalized
    save_regions_config(
        experiment_dir,
        {"version": config.get("version", REGIONS_SCHEMA_VERSION), "scans": scans},
    )
