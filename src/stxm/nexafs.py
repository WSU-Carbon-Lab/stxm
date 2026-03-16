import numpy as np


def _weighted_mean_and_sigma(
    values_2d: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_energy = values_2d.shape[1]
    mean_out = np.zeros(n_energy)
    sigma_out = np.zeros(n_energy)
    for j in range(n_energy):
        col = values_2d[:, j]
        vals = col[mask]
        vals = np.maximum(vals, eps)
        if vals.size == 0:
            mean_out[j] = np.nan
            sigma_out[j] = np.nan
            continue
        w = 1.0 / vals
        mean_out[j] = np.sum(vals * w) / np.sum(w)
        sigma_out[j] = 1.0 / np.sqrt(np.sum(w))
    return mean_out, sigma_out, int(np.sum(mask))


def nexafs_beer_lambert(
    image: np.ndarray,
    sample_mask: np.ndarray,
    izero_mask: np.ndarray,
    eps: float = 1e-10,
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
    I0, sigma_I0, n_izero = _weighted_mean_and_sigma(image, izero_mask, eps)
    I, sigma_I, n_sample = _weighted_mean_and_sigma(image, sample_mask, eps)
    I0 = np.maximum(I0, eps)
    I = np.maximum(I, eps)
    od = np.log(I0 / I)
    sigma_od = np.sqrt((sigma_I0 / I0) ** 2 + (sigma_I / I) ** 2)
    return od, sigma_od, I0, sigma_I0, I, sigma_I, n_sample, n_izero
