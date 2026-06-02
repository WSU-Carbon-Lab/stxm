"""Shared energy-grid helpers for spectra interpolation."""

import numpy as np


def common_energy_grid(
    energies: list[np.ndarray],
    *,
    n_points: int | None = None,
) -> np.ndarray:
    """
    Build a monotonic energy grid spanning the overlap of input axes.

    Parameters
    ----------
    energies : list of np.ndarray
        One-dimensional energy axes in eV.
    n_points : int, optional
        If given, linearly resample the overlap interval to this many points.

    Returns
    -------
    np.ndarray
        Sorted grid over ``[max(lo), min(hi)]`` for all inputs.

    Raises
    ------
    ValueError
        If any axis is empty or the overlap interval is empty.
    """
    if not energies:
        raise ValueError("energies must be non-empty")
    lo = max(float(np.min(e)) for e in energies)
    hi = min(float(np.max(e)) for e in energies)
    if hi <= lo:
        raise ValueError("energy axes have no overlapping interval")
    if n_points is None:
        arrays = [np.asarray(e, dtype=np.float64).ravel() for e in energies]
        union = np.unique(np.concatenate(arrays))
        grid = union[(union >= lo) & (union <= hi)]
        return np.sort(grid)
    return np.linspace(lo, hi, int(n_points), dtype=np.float64)


def interpolate_spectrum(
    energy: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate ``values`` onto ``grid``.

    Parameters
    ----------
    energy : np.ndarray
        Source energy axis in eV, strictly increasing after sort.
    values : np.ndarray
        Spectrum samples aligned with ``energy``.
    grid : np.ndarray
        Target energy grid in eV.

    Returns
    -------
    np.ndarray
        Interpolated values; points outside ``energy`` range are ``nan``.

    Raises
    ------
    ValueError
        If ``energy`` and ``values`` differ in length.
    """
    energy = np.asarray(energy, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    grid = np.asarray(grid, dtype=np.float64).ravel()
    if energy.size != values.size:
        raise ValueError("energy and values must have the same length")
    if energy.size < 2:
        raise ValueError("energy must contain at least two points")
    order = np.argsort(energy)
    e_sorted = energy[order]
    v_sorted = values[order]
    return np.interp(grid, e_sorted, v_sorted, left=np.nan, right=np.nan)
