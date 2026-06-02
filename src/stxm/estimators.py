"""Region averaging strategies for NEXAFS line-scan intensities."""

from enum import StrEnum

import numpy as np


class WeightingMode(StrEnum):
    """Recorded region-mean weighting policy."""

    INVERSE_COUNT = "inverse_count"
    POISSON_MLE = "poisson_mle"
    EMPIRICAL = "empirical"


def region_mean_and_sigma(
    values_2d: np.ndarray,
    mask: np.ndarray,
    mode: WeightingMode = WeightingMode.POISSON_MLE,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-energy column mean and standard error over masked rows.

    Parameters
    ----------
    values_2d : np.ndarray
        Scan intensities with shape ``(n_spatial, n_energy)``.
    mask : np.ndarray
        Boolean mask along the spatial axis, length ``n_spatial``.
    mode : WeightingMode
        ``inverse_count`` reproduces the legacy harmonic-mean weighting;
        ``poisson_mle`` uses the arithmetic mean with ``sqrt(mean / n)`` error;
        ``empirical`` uses sample variance over ``n``.
    eps : float
        Floor applied before legacy inverse-count weighting.

    Returns
    -------
    mean : np.ndarray
        Length ``n_energy``; ``nan`` where the mask selects no rows.
    sigma : np.ndarray
        Standard error per energy column; ``nan`` where the mask is empty.
    n : int
        Number of masked spatial rows.

    Raises
    ------
    ValueError
        If ``values_2d`` is not two-dimensional or ``mask`` length mismatches.
    """
    values_2d = np.asarray(values_2d, dtype=np.float64)
    if values_2d.ndim != 2:
        raise ValueError("values_2d must be 2D (spatial, energy)")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (values_2d.shape[0],):
        raise ValueError("mask length must match values_2d.shape[0]")
    n_energy = values_2d.shape[1]
    n = int(np.sum(mask))
    if n == 0:
        empty = np.full(n_energy, np.nan, dtype=np.float64)
        return empty, empty.copy(), 0
    block = values_2d[mask, :]
    if mode is WeightingMode.INVERSE_COUNT:
        vals = np.maximum(block, eps)
        weights = 1.0 / vals
        weight_sum = np.sum(weights, axis=0)
        mean = np.sum(vals * weights, axis=0) / weight_sum
        sigma = 1.0 / np.sqrt(weight_sum)
        return mean, sigma, n
    mean = np.mean(block, axis=0)
    if mode is WeightingMode.POISSON_MLE:
        sigma = np.sqrt(np.maximum(mean, 0.0) / n)
        return mean, sigma, n
    if mode is WeightingMode.EMPIRICAL:
        sigma = np.std(block, axis=0, ddof=1) / np.sqrt(n)
        return mean, sigma, n
    raise ValueError(f"unsupported mode: {mode!r}")
