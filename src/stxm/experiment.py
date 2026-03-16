from pathlib import Path

import numpy as np
import pandas as pd

from stxm.io import load_stxm
from stxm.nexafs import nexafs_beer_lambert
from stxm.normalization import normalize_nexafs
from stxm.regions import bar_bounds_from_three_regions, sample_izero_masks

NEXAFS_COLUMNS = [
    "energy_eV",
    "OD",
    "OD_err",
    "I0",
    "I0_err",
    "I",
    "I_err",
    "n_sample",
    "n_izero",
]
OPTIONAL_DERIVED_COLUMNS = [
    "mass_absorption",
    "mass_absorption_err",
    "beta",
    "beta_err",
]
OD_NORMALIZED_COLUMN = "OD_normalized"
FORMULA_COLUMN = "formula"
SCAN_PATH_COLUMN = "scan_path"
SAMPLE_NAME_COLUMN = "sample_name"
CHEMICAL_FORMULA_COLUMN = "chemical_formula"
SPOT_LABEL_COLUMN = "spot_label"
FILM_REGION_NAME_COLUMN = "film_region_name"
META_COLUMNS = [
    FORMULA_COLUMN,
    SCAN_PATH_COLUMN,
    SAMPLE_NAME_COLUMN,
    CHEMICAL_FORMULA_COLUMN,
    SPOT_LABEL_COLUMN,
    FILM_REGION_NAME_COLUMN,
]


def append_nexafs_to_experiment(
    parquet_path: str | Path,
    nexafs_df: pd.DataFrame,
    sample_name: str,
    chemical_formula: str | None,
    spot_label: str,
    film_region_name: str = "",
    scan_path: str | Path | None = None,
    formula: str | None = None,
) -> None:
    """
    Append a NEXAFS dataframe to the experiment parquet with sample metadata.

    Primary row identity: sample_name, chemical_formula (nullable for blends), spot_label,
    film_region_name. Legacy ``formula`` column is set to chemical_formula when not None.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the experiment parquet file.
    nexafs_df : pd.DataFrame
        DataFrame with NEXAFS_COLUMNS.
    sample_name : str
        Sample identifier (e.g. wafer or batch label).
    chemical_formula : str or None
        Stoichiometric formula; None for blend / multi-component fits.
    spot_label : str
        Domain or spot (e.g. crystal, matrix).
    film_region_name : str
        Film region label (e.g. sample bar name).
    scan_path : str or Path, optional
        Path to the .hdr file.
    formula : str, optional
        If given, overrides chemical_formula for legacy FORMULA_COLUMN when chemical_formula is None.
    """
    parquet_path = Path(parquet_path)
    required = set(NEXAFS_COLUMNS)
    missing = required - set(nexafs_df.columns)
    if missing:
        raise ValueError(f"nexafs_df missing columns: {missing}")
    out = nexafs_df[list(NEXAFS_COLUMNS)].copy()
    if OD_NORMALIZED_COLUMN in nexafs_df.columns:
        out[OD_NORMALIZED_COLUMN] = nexafs_df[OD_NORMALIZED_COLUMN]
    for col in OPTIONAL_DERIVED_COLUMNS:
        if col in nexafs_df.columns:
            out[col] = nexafs_df[col]
    cf = chemical_formula
    legacy_formula = cf if cf is not None else (formula or "")
    out[FORMULA_COLUMN] = legacy_formula if legacy_formula else None
    out[SCAN_PATH_COLUMN] = str(scan_path) if scan_path is not None else None
    out[SAMPLE_NAME_COLUMN] = sample_name
    out[CHEMICAL_FORMULA_COLUMN] = cf
    out[SPOT_LABEL_COLUMN] = spot_label
    out[FILM_REGION_NAME_COLUMN] = film_region_name or sample_name
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        for col in META_COLUMNS:
            if col not in existing.columns:
                existing[col] = None
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined.to_parquet(parquet_path, index=False)


def load_experiment_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """
    Load the full experiment parquet (all NEXAFS rows with formula and scan_path).

    Parameters
    ----------
    parquet_path : str or Path
        Path to the experiment parquet file.

    Returns
    -------
    pd.DataFrame
        Table with NEXAFS columns plus formula and scan_path.
    """
    return pd.read_parquet(parquet_path)


def process_experiment_folder(
    folder: str | Path,
    parquet_path: str | Path,
    formula: str | None,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    post_target: float = 1.0,
    sample_name: str | None = None,
    spot_label: str = "batch",
    film_region_name: str | None = None,
) -> pd.DataFrame:
    """
    Load every .hdr in the folder, compute OD with auto sample/izero, normalize with
    pre/post-edge, and write one experiment parquet (overwrites).

    Parameters
    ----------
    folder : str or Path
        Directory containing .hdr (and .xim) files.
    parquet_path : str or Path
        Output parquet path (overwritten).
    formula : str or None
        Chemical formula for all spectra; None for blends.
    pre_lo, pre_hi : float
        Pre-edge energy range [eV] for baseline subtraction.
    post_lo, post_hi : float
        Post-edge energy range [eV] for normalization.
    post_target : float
        Target mean OD in post-edge (default 1.0).

    Returns
    -------
    pd.DataFrame
        Combined table with NEXAFS columns, OD_normalized, formula, scan_path.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    sn = sample_name if sample_name else folder.name
    fr = film_region_name if film_region_name else sn
    cf = formula if (formula and str(formula).strip()) else None
    legacy_f = cf if cf is not None else ""
    hdr_files = sorted(folder.glob("*.hdr"))
    if not hdr_files:
        raise FileNotFoundError(f"No .hdr files in {folder}")

    rows = []
    for hdr_path in hdr_files:
        try:
            meta, image = load_stxm(hdr_path)
        except Exception:
            continue
        qaxis = np.asarray(meta["qaxis_points"])
        paxis = np.asarray(meta["paxis_points"])
        if qaxis.size != image.shape[0] or paxis.size != image.shape[1]:
            continue
        bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi = bar_bounds_from_three_regions(
            image, qaxis
        )
        sample_mask, izero_mask = sample_izero_masks(
            qaxis, bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi
        )
        if not (np.any(sample_mask) and np.any(izero_mask)):
            continue
        od, sigma_od, I0, sigma_I0, I, sigma_I, n_sample, n_izero = nexafs_beer_lambert(
            image, sample_mask, izero_mask
        )
        od_norm = normalize_nexafs(
            paxis, od, pre_lo, pre_hi, post_lo, post_hi, post_target=post_target
        )
        rows.append(
            pd.DataFrame({
                "energy_eV": paxis,
                "OD": od,
                "OD_err": sigma_od,
                "I0": I0,
                "I0_err": sigma_I0,
                "I": I,
                "I_err": sigma_I,
                "n_sample": n_sample,
                "n_izero": n_izero,
                OD_NORMALIZED_COLUMN: od_norm,
                FORMULA_COLUMN: legacy_f,
                SCAN_PATH_COLUMN: str(hdr_path),
                SAMPLE_NAME_COLUMN: sn,
                CHEMICAL_FORMULA_COLUMN: cf,
                SPOT_LABEL_COLUMN: spot_label,
                FILM_REGION_NAME_COLUMN: fr,
            })
        )
    if not rows:
        raise ValueError(f"No scans could be processed in {folder}")
    combined = pd.concat(rows, ignore_index=True)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False)
    return combined
