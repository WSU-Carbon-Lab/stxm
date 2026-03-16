import numpy as np


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
    out = pre_edge_subtract(energy, OD, pre_lo, pre_hi)
    out, _ = post_edge_normalize(energy, out, post_lo, post_hi, target=post_target)
    return out
