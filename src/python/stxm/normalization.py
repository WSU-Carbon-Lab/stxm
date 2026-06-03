from enum import StrEnum

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class NormalizationMode(StrEnum):
    """NEXAFS OD normalization policies exposed to the widget and store."""

    PRE_EDGE_SCALE = "pre_edge_scale"
    SCALE_SHIFT = "scale_shift"


def energy_region_mask(
    energy_points: np.ndarray,
    e_lo: float,
    e_hi: float,
) -> np.ndarray:
    """
    Boolean mask for energy axis within [e_lo, e_hi] inclusive.

    Parameters
    ----------
    energy_points : np.ndarray
        Energy values in eV.
    e_lo, e_hi : float
        Lower and upper bounds (inclusive).

    Returns
    -------
    np.ndarray
        Boolean array, True where e_lo <= energy <= e_hi.
    """
    e = np.asarray(energy_points, dtype=float)
    return (e >= e_lo) & (e <= e_hi)


def pre_edge_subtract(
    energy: np.ndarray,
    OD: np.ndarray,
    pre_lo: float,
    pre_hi: float,
) -> np.ndarray:
    """
    Subtract the mean OD in the pre-edge region so pre-edge baseline is zero.

    Parameters
    ----------
    energy : np.ndarray
        Energy in eV, same length as OD.
    OD : np.ndarray
        Optical density (e.g. ln I0/I).
    pre_lo, pre_hi : float
        Pre-edge energy range [eV] over which to compute the mean.

    Returns
    -------
    np.ndarray
        OD with pre-edge mean subtracted. Same shape as OD.
    """
    energy = np.asarray(energy, dtype=float)
    OD = np.asarray(OD, dtype=float).copy()
    mask = energy_region_mask(energy, pre_lo, pre_hi)
    if not np.any(mask):
        return OD
    vals = OD[mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return OD
    baseline = np.mean(vals[finite])
    OD -= baseline
    return OD


def post_edge_normalize(
    energy: np.ndarray,
    OD: np.ndarray,
    post_lo: float,
    post_hi: float,
    target: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Scale OD so the mean value in the post-edge region equals target.

    Parameters
    ----------
    energy : np.ndarray
        Energy in eV, same length as OD.
    OD : np.ndarray
        Optical density (typically after pre-edge subtract).
    post_lo, post_hi : float
        Post-edge energy range [eV].
    target : float
        Desired mean OD in post-edge region (default 1.0).

    Returns
    -------
    OD_scaled : np.ndarray
        OD scaled by (target / mean_post).
    scale : float
        Scale factor applied (target / mean_post).
    """
    energy = np.asarray(energy, dtype=float)
    OD = np.asarray(OD, dtype=float).copy()
    mask = energy_region_mask(energy, post_lo, post_hi)
    if not np.any(mask):
        return OD, 1.0
    vals = OD[mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return OD, 1.0
    mean_post = np.mean(vals[finite])
    if mean_post <= 0 or not np.isfinite(mean_post):
        return OD, 1.0
    scale = target / mean_post
    OD *= scale
    return OD, scale


def _shift_spectrum(
    energy: np.ndarray,
    od: np.ndarray,
    delta_e: float,
) -> np.ndarray:
    e = np.asarray(energy, dtype=np.float64)
    y = np.asarray(od, dtype=np.float64)
    if e.size < 2:
        return y.copy()
    order = np.argsort(e)
    e_sorted = e[order]
    y_sorted = y[order]
    interpolator = interp1d(
        e_sorted,
        y_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(interpolator(e + float(delta_e)), dtype=np.float64)


def _post_edge_mean(
    energy: np.ndarray,
    od: np.ndarray,
    post_lo: float,
    post_hi: float,
) -> float:
    mask = energy_region_mask(energy, post_lo, post_hi)
    if not np.any(mask):
        return float("nan")
    vals = np.asarray(od, dtype=np.float64)[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def fit_energy_shift(
    energy: np.ndarray,
    od: np.ndarray,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    *,
    post_target: float = 1.0,
    shift_bounds: tuple[float, float] = (-8.0, 8.0),
) -> tuple[float, float]:
    """
    Estimate an energy shift that improves post-edge alignment after pre-edge subtraction.

    Parameters
    ----------
    energy : np.ndarray
        Energy axis in eV.
    od : np.ndarray
        Optical density before normalization.
    pre_lo, pre_hi : float
        Pre-edge window for baseline subtraction.
    post_lo, post_hi : float
        Post-edge window used to score alignment.
    post_target : float
        Desired post-edge mean after scaling.
    shift_bounds : tuple of float
        Search interval for ``delta_e`` in eV.

    Returns
    -------
    delta_e : float
        Energy shift applied as ``interp(energy + delta_e)``.
    scale : float
        Post-edge scale factor at the optimal shift.
    """
    baseline_removed = pre_edge_subtract(energy, od, pre_lo, pre_hi)
    span = float(np.nanmax(energy) - np.nanmin(energy)) if energy.size else 1.0
    half = min(max(shift_bounds[1], abs(shift_bounds[0])), 0.15 * span)

    def objective(delta: float) -> float:
        shifted = _shift_spectrum(energy, baseline_removed, delta)
        scaled, _ = post_edge_normalize(energy, shifted, post_lo, post_hi, target=post_target)
        mean_post = _post_edge_mean(energy, scaled, post_lo, post_hi)
        if not np.isfinite(mean_post):
            return 1e6
        return float((mean_post - post_target) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(-half, half),
        method="bounded",
    )
    delta_e = float(result.x)
    shifted = _shift_spectrum(energy, baseline_removed, delta_e)
    _, scale = post_edge_normalize(energy, shifted, post_lo, post_hi, target=post_target)
    return delta_e, scale


def normalize_nexafs(
    energy: np.ndarray,
    OD: np.ndarray,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    post_target: float = 1.0,
) -> np.ndarray:
    """
    Full NEXAFS normalization: subtract pre-edge baseline, then scale to post-edge = post_target.

    Parameters
    ----------
    energy : np.ndarray
        Energy in eV.
    OD : np.ndarray
        Raw optical density.
    pre_lo, pre_hi : float
        Pre-edge range [eV] for baseline subtraction.
    post_lo, post_hi : float
        Post-edge range [eV] for scaling.
    post_target : float
        Target value for mean OD in post-edge (default 1.0).

    Returns
    -------
    np.ndarray
        Normalized OD (pre-edge subtracted, then scaled so post-edge mean = post_target).
    """
    normalized, _ = normalize_nexafs_with_metadata(
        energy,
        OD,
        pre_lo,
        pre_hi,
        post_lo,
        post_hi,
        mode=NormalizationMode.PRE_EDGE_SCALE,
        post_target=post_target,
    )
    return normalized


def normalize_nexafs_with_metadata(
    energy: np.ndarray,
    OD: np.ndarray,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    *,
    mode: NormalizationMode | str = NormalizationMode.PRE_EDGE_SCALE,
    post_target: float = 1.0,
    shift_bounds: tuple[float, float] = (-8.0, 8.0),
) -> tuple[np.ndarray, dict[str, float | str]]:
    """
    Normalize OD and return reproducibility metadata for provenance export.

    Parameters
    ----------
    energy : np.ndarray
        Energy in eV.
    OD : np.ndarray
        Raw optical density.
    pre_lo, pre_hi : float
        Pre-edge window in eV.
    post_lo, post_hi : float
        Post-edge window in eV.
    mode : NormalizationMode or str
        ``pre_edge_scale`` (baseline + post-edge scale) or ``scale_shift`` (adds energy shift).
    post_target : float
        Target post-edge mean OD.
    shift_bounds : tuple of float
        Energy-shift search bounds for ``scale_shift`` mode.

    Returns
    -------
    normalized : np.ndarray
        Normalized OD on the input energy grid.
    metadata : dict
        Keys ``normalization_mode``, ``energy_shift_eV``, ``post_edge_scale``.
    """
    if isinstance(mode, str):
        mode = NormalizationMode(mode)
    energy = np.asarray(energy, dtype=np.float64)
    od = np.asarray(OD, dtype=np.float64)
    baseline_removed = pre_edge_subtract(energy, od, pre_lo, pre_hi)
    energy_shift = 0.0
    working = baseline_removed
    if mode is NormalizationMode.SCALE_SHIFT:
        energy_shift, _ = fit_energy_shift(
            energy,
            od,
            pre_lo,
            pre_hi,
            post_lo,
            post_hi,
            post_target=post_target,
            shift_bounds=shift_bounds,
        )
        working = _shift_spectrum(energy, baseline_removed, energy_shift)
    scaled, scale = post_edge_normalize(
        energy,
        working,
        post_lo,
        post_hi,
        target=post_target,
    )
    meta: dict[str, float | str] = {
        "normalization_mode": mode.value,
        "energy_shift_eV": float(energy_shift),
        "post_edge_scale": float(scale),
    }
    return scaled, meta


def apply_normalization_mode(
    energy: np.ndarray,
    OD: np.ndarray,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    mode: NormalizationMode | str,
    *,
    post_target: float = 1.0,
) -> np.ndarray:
    """
    Normalize OD using the selected mode (widget-facing alias).

    Parameters
    ----------
    energy : np.ndarray
        Energy in eV.
    OD : np.ndarray
        Raw optical density.
    pre_lo, pre_hi, post_lo, post_hi : float
        Normalization windows in eV.
    mode : NormalizationMode or str
        Normalization policy.
    post_target : float
        Target post-edge mean.

    Returns
    -------
    np.ndarray
        Normalized OD.
    """
    out, _ = normalize_nexafs_with_metadata(
        energy,
        OD,
        pre_lo,
        pre_hi,
        post_lo,
        post_hi,
        mode=mode,
        post_target=post_target,
    )
    return out
