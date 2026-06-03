import re
from pathlib import Path

import numpy as np


def _parse_points_count(text: str, axis_name: str) -> int | None:
    pattern = rf"{axis_name}\s*=\s*\{{[^}}]*Points\s*=\s*\(\s*(\d+)"
    match = re.search(pattern, text, re.DOTALL)
    return int(match.group(1)) if match else None


def _parse_points_array(text: str, axis_name: str) -> np.ndarray | None:
    pattern = rf"{axis_name}\s*=\s*\{{[^}}]*Points\s*=\s*\(\s*\d+\s*,\s*([\d\s.,\-eE+]+)\)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).replace(",", " ").split()
    return np.array([float(x) for x in raw])


def read_hdr(path: str | Path) -> dict:
    """
    Parse STXM .hdr file. Returns dict with axis sizes and optional axis arrays.

    Returns
    -------
    dict
        Keys: paxis_count, qaxis_count (int); paxis_name, qaxis_name (str);
        paxis_points, qaxis_points (ndarray, optional); raw (str).
    """
    path = Path(path)
    raw = path.read_text()
    paxis_count = _parse_points_count(raw, "PAxis")
    qaxis_count = _parse_points_count(raw, "QAxis")
    if paxis_count is None or qaxis_count is None:
        raise ValueError("Could not find PAxis or QAxis Points in header")
    out = {
        "paxis_count": paxis_count,
        "qaxis_count": qaxis_count,
        "raw": raw,
    }
    pname = re.search(r'PAxis\s*=\s*\{\s*Name\s*=\s*"([^"]*)"', raw)
    qname = re.search(r'QAxis\s*=\s*\{\s*Name\s*=\s*"([^"]*)"', raw)
    if pname:
        out["paxis_name"] = pname.group(1)
    if qname:
        out["qaxis_name"] = qname.group(1)
    parr = _parse_points_array(raw, "PAxis")
    qarr = _parse_points_array(raw, "QAxis")
    if parr is not None:
        out["paxis_points"] = parr
    if qarr is not None:
        out["qaxis_points"] = qarr
    return out


def read_xim(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load STXM .xim ascii image (whitespace-separated values, one line per row).

    Parameters
    ----------
    path : str or Path
        Path to .xim file.
    shape : tuple of (n_rows, n_cols), optional
        If provided, ensure output has this shape (reshape by row-major if needed).
        If None, return 2D array as in file.

    Returns
    -------
    np.ndarray
        2D array of dtype float64. Shape (qaxis_count, paxis_count) when
        used with read_hdr dimensions.
    """
    path = Path(path)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        if shape is None:
            raise ValueError("xim is 1D; provide shape=(n_rows, n_cols)")
        data = data.reshape(shape)
    elif shape is not None and data.size == np.prod(shape):
        if data.shape != shape:
            data = data.reshape(shape)
    return data


def load_stxm(
    hdr_path: str | Path,
    xim_path: str | Path | None = None,
) -> tuple[dict, np.ndarray]:
    """
    Load STXM scan: parse .hdr and load associated .xim ascii image.

    Parameters
    ----------
    hdr_path : str or Path
        Path to .hdr file.
    xim_path : str or Path, optional
        Path to .xim file. If None, inferred as same stem with '_a.xim'.

    Returns
    -------
    header : dict
        Result of read_hdr(hdr_path).
    image : np.ndarray
        2D array shape (qaxis_count, paxis_count), i.e. (n_rows, n_cols).
    """
    hdr_path = Path(hdr_path)
    meta = read_hdr(hdr_path)
    if xim_path is None:
        stem = hdr_path.stem
        parent = hdr_path.parent
        xim_path = parent / f"{stem}_a.xim"
        if not xim_path.exists():
            xim_path = parent / f"{stem}.xim"
    xim_path = Path(xim_path)
    if not xim_path.exists():
        raise FileNotFoundError(f"xim file not found: {xim_path}")
    shape = (meta["qaxis_count"], meta["paxis_count"])
    image = read_xim(xim_path, shape=shape)
    return meta, image


MIN_BYTES_PER_VALUE = 3

NEXAFS_LINE_SCAN_TYPE = 'Type = "NEXAFS Line Scan"'


def is_nexafs_line_scan_type(hdr_path: str | Path) -> bool:
    """
    Return True if the .hdr header contains Type = "NEXAFS Line Scan".

    Parameters
    ----------
    hdr_path : str or Path
        Path to .hdr file.

    Returns
    -------
    bool
    """
    path = Path(hdr_path)
    if not path.suffix.lower() == ".hdr" or not path.exists():
        return False
    return NEXAFS_LINE_SCAN_TYPE in path.read_text()


def is_valid_line_scan_fast(hdr_path: str | Path) -> bool:
    """
    Fast heuristic: True if .hdr parses with PAxis/QAxis and the .xim file exists
    and has at least (paxis_count * qaxis_count) * MIN_BYTES_PER_VALUE bytes.
    Use to filter file lists without loading full .xim data. May have false
    negatives (smaller .xim files that still load). Prefer is_valid_line_scan
    when a definitive result is needed.

    Parameters
    ----------
    hdr_path : str or Path
        Path to .hdr file.

    Returns
    -------
    bool
        True if header format matches and .xim size suggests full data.
    """
    try:
        hdr_path = Path(hdr_path)
        meta = read_hdr(hdr_path)
        n = meta["paxis_count"] * meta["qaxis_count"]
        stem = hdr_path.stem
        parent = hdr_path.parent
        xim_path = parent / f"{stem}_a.xim"
        if not xim_path.exists():
            xim_path = parent / f"{stem}.xim"
        if not xim_path.exists():
            return False
        return xim_path.stat().st_size >= n * MIN_BYTES_PER_VALUE
    except Exception:
        return False


def is_valid_line_scan(hdr_path: str | Path) -> bool:
    """
    Return True if the .hdr has an associated .xim that loads as a valid line scan
    (correct format and shape). Use to filter the file dropdown to only show loadable scans.

    Parameters
    ----------
    hdr_path : str or Path
        Path to .hdr file.

    Returns
    -------
    bool
        True if load_stxm(hdr_path) succeeds.
    """
    try:
        load_stxm(hdr_path)
        return True
    except Exception:
        return False


def is_nexafs_line_scan(hdr_path: str | Path) -> bool:
    """
    Return True if the .hdr is a NEXAFS Line Scan (Type = "NEXAFS Line Scan")
    and has an associated .xim that loads with the correct 2D line shape.

    Parameters
    ----------
    hdr_path : str or Path
        Path to .hdr file.

    Returns
    -------
    bool
        True if header type is NEXAFS Line Scan and load_stxm(hdr_path) succeeds.
    """
    return is_nexafs_line_scan_type(hdr_path) and is_valid_line_scan(hdr_path)


def _axis_is_energy(name: str) -> bool:
    lowered = name.lower()
    return "energy" in lowered or lowered.endswith("ev") or "photon" in lowered


def _is_strictly_monotonic(axis: np.ndarray) -> bool:
    diffs = np.diff(np.asarray(axis, dtype=np.float64).ravel())
    return bool(diffs.size) and bool(np.all(diffs > 0) or np.all(diffs < 0))


def orient_scan(meta: dict, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(energy, spatial, image)`` with shape ``(n_spatial, n_energy)``.

    Parameters
    ----------
    meta : dict
        Header dict from ``read_hdr`` with ``paxis_points`` and ``qaxis_points``.
    image : np.ndarray
        Raw 2D array from ``load_stxm``.

    Returns
    -------
    energy : np.ndarray
        Energy axis in eV, strictly monotonic.
    spatial : np.ndarray
        Spatial axis coordinates, length ``image.shape[0]`` after orientation.
    image : np.ndarray
        Oriented intensities ``(n_spatial, n_energy)``.

    Raises
    ------
    ValueError
        If axis sizes disagree with the array or energy is not monotonic.
    """
    paxis = np.asarray(meta["paxis_points"], dtype=np.float64)
    qaxis = np.asarray(meta["qaxis_points"], dtype=np.float64)
    arr = np.asarray(image, dtype=np.float64)
    p_name = str(meta.get("paxis_name", "PAxis"))
    q_name = str(meta.get("qaxis_name", "QAxis"))
    energy_on_p = _axis_is_energy(p_name)
    energy_on_q = _axis_is_energy(q_name)
    if arr.shape == (qaxis.size, paxis.size):
        spatial, energy = qaxis, paxis
    elif arr.shape == (paxis.size, qaxis.size):
        arr = arr.T
        spatial, energy = qaxis, paxis
    else:
        raise ValueError(
            f"image shape {arr.shape} incompatible with qaxis={qaxis.size} paxis={paxis.size}"
        )
    if energy_on_q and not energy_on_p:
        arr = arr.T
        spatial, energy = paxis, qaxis
    if not _is_strictly_monotonic(energy):
        raise ValueError("energy axis must be strictly monotonic")
    if arr.shape != (spatial.size, energy.size):
        raise ValueError("oriented image shape does not match axis lengths")
    return energy, spatial, arr


def spectral_matrix(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    use_od: bool = True,
    eps: float = 1e-10,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Flatten an oriented scan to ``(n_pixels, n_energy)``.

    Parameters
    ----------
    image : np.ndarray
        Oriented scan ``(n_spatial, n_energy)``.
    mask : np.ndarray, optional
        Boolean spatial mask; all rows used when ``None``.
    use_od : bool
        If True, transform to ``-ln(I)``; otherwise use intensities.
    eps : float
        Intensity floor when ``use_od`` is True.

    Returns
    -------
    X : np.ndarray
        Data matrix for demixing or regression.
    spatial_shape : tuple of int
        ``(n_spatial, 1)`` for reshaping abundances back to a line scan.

    Raises
    ------
    ValueError
        If the mask selects no rows.
    """
    arr = np.asarray(image, dtype=np.float64)
    if mask is None:
        block = arr
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (arr.shape[0],):
            raise ValueError("mask length must match n_spatial")
        block = arr[mask, :]
    if block.size == 0:
        raise ValueError("mask selects no spatial rows")
    if use_od:
        block = -np.log(np.maximum(block, eps))
    return block, (arr.shape[0], 1)


def list_nexafs_line_scans(experiment_path: str | Path) -> list[Path]:
    """
    List .hdr files in the experiment directory that are NEXAFS Line Scan type
    and have a loadable .xim with correct 2D shape. Sorted by name.

    Parameters
    ----------
    experiment_path : str or Path
        Path to experiment folder (contains .hdr and .xim files).

    Returns
    -------
    list of Path
        Paths to .hdr files that pass is_nexafs_line_scan.
    """
    directory = Path(experiment_path)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.glob("*.hdr")
        if is_nexafs_line_scan(p)
    )
