import numpy as np

from stxm.estimators import WeightingMode, region_mean_and_sigma


def _weighted_mean_and_sigma(
    values_2d: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    return region_mean_and_sigma(
        values_2d, mask, mode=WeightingMode.INVERSE_COUNT, eps=eps
    )


def nexafs_beer_lambert(
    image: np.ndarray,
    sample_mask: np.ndarray,
    izero_mask: np.ndarray,
    eps: float = 1e-10,
    mode: WeightingMode = WeightingMode.POISSON_MLE,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """
    NEXAFS optical density and uncertainties via Beer-Lambert (OD = ln(I0/I)).

    Parameters
    ----------
    image : np.ndarray
        2D scan (rows = sample axis, cols = energy).
    sample_mask : np.ndarray
        Boolean mask for sample region rows.
    izero_mask : np.ndarray
        Boolean mask for izero region rows.
    eps : float
        Minimum intensity to avoid log(0).
    mode : WeightingMode
        Region averaging strategy forwarded to ``region_mean_and_sigma``.

    Returns
    -------
    od : np.ndarray
        Optical density at each energy.
    sigma_od : np.ndarray
        Uncertainty in OD.
    I0, sigma_I0 : np.ndarray
        Izero mean and uncertainty per energy.
    I, sigma_I : np.ndarray
        Sample mean and uncertainty per energy.
    n_sample, n_izero : int
        Number of pixels in each region.
    """
    I0, sigma_I0, n_izero = region_mean_and_sigma(
        image, izero_mask, mode=mode, eps=eps
    )
    i_sample, sigma_i, n_sample = region_mean_and_sigma(
        image, sample_mask, mode=mode, eps=eps
    )
    i0 = np.maximum(I0, eps)
    i_s = np.maximum(i_sample, eps)
    od = np.log(i0 / i_s)
    sigma_od = np.sqrt((sigma_I0 / i0) ** 2 + (sigma_i / i_s) ** 2)
    return od, sigma_od, i0, sigma_I0, i_s, sigma_i, n_sample, n_izero
