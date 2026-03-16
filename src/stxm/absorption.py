import numpy as np
import periodictable
from periodictable import xsf

HC_EV_CM = 1.2398e-4


def mass_absorption_cm2_per_g(
    formula_str: str,
    energy_eV: np.ndarray,
    density_g_cm3: float | None = 1,
) -> np.ndarray:
    """
    Mass absorption coefficient (cm^2/g) from periodictable beta (imaginary part of n).

    Parameters
    ----------
    formula_str : str
        Chemical formula, e.g. "C", "H2O", "CaCO3".
    energy_eV : np.ndarray
        Photon energies in eV.
    density_g_cm3 : float, optional
        Density in g/cm^3. If None, uses formula default.

    Returns
    -------
    np.ndarray
        Mass absorption coefficient mu/rho in cm^2/g at each energy.
    """
    compound = (
        periodictable.formula(formula_str, density=density_g_cm3)
        if density_g_cm3 is not None
        else periodictable.formula(formula_str)
    )
    rho = 1
    energy_keV = np.asarray(energy_eV, dtype=float) / 1000.0
    n = xsf.index_of_refraction(compound, energy=energy_keV, density=rho)
    n = np.atleast_1d(n)
    beta = -np.imag(n)
    lam_cm = HC_EV_CM / np.asarray(energy_eV, dtype=float)
    mu_per_cm = 4 * np.pi * beta / np.atleast_1d(lam_cm)
    return mu_per_cm / rho


def fit_bare_atom_background(
    energy_eV: np.ndarray,
    OD: np.ndarray,
    mu_rho_cm2_per_g: np.ndarray,
    n_low: int = 5,
    n_high: int = 5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fit OD = scale * mu_periodictable + const on first n_low and last n_high energies.

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy in eV (same length as OD and mu_rho).
    OD : np.ndarray
        Optical density (ln I0/I).
    mu_rho_cm2_per_g : np.ndarray
        Mass absorption from periodictable at same energies.
    n_low : int
        Number of points at lowest energies to use in fit.
    n_high : int
        Number of points at highest energies to use in fit.

    Returns
    -------
    scale : float
        Multiplicative scale factor.
    const : float
        Constant background.
    OD_bare : np.ndarray
        Fitted bare-atom curve: scale * mu_rho + const.
    mask_fit : np.ndarray
        Boolean mask of points used in the fit.
    """
    n = len(energy_eV)
    n_low = min(n_low, n)
    n_high = min(n_high, n)
    idx_low = np.arange(n_low) if n_low > 0 else np.array([], dtype=int)
    idx_high = np.arange(n - n_high, n) if n_high > 0 else np.array([], dtype=int)
    idx_fit = np.concatenate([idx_low, idx_high])
    mask_fit = np.zeros(n, dtype=bool)
    if idx_fit.size > 0:
        mask_fit[idx_fit] = True

    mu_fit = mu_rho_cm2_per_g[idx_fit]
    od_fit = OD[idx_fit]

    if n_low == 0:
        scale = np.linalg.lstsq(mu_fit[:, np.newaxis], od_fit, rcond=None)[0][0]
        const = 0.0
    else:
        A = np.column_stack([mu_fit, np.ones_like(mu_fit)])
        scale, const = np.linalg.lstsq(A, od_fit, rcond=None)[0]

    OD_bare = scale * mu_rho_cm2_per_g + const
    return scale, const, OD_bare, mask_fit


def od_to_beta(
    energy_eV: np.ndarray,
    OD: np.ndarray,
    thickness_cm: float,
) -> np.ndarray:
    """
    Convert optical density to beta (imaginary part of refractive index n = 1 - delta - i*beta).

    OD = mu * t, mu = 4*pi*beta/lambda, so beta = OD * lambda / (4*pi*t).

    Parameters
    ----------
    energy_eV : np.ndarray
        Photon energy in eV.
    OD : np.ndarray
        Optical density ln(I0/I).
    thickness_cm : float
        Sample thickness in cm.

    Returns
    -------
    np.ndarray
        Beta (dimensionless) at each energy.
    """
    if thickness_cm <= 0:
        raise ValueError("thickness_cm must be positive")
    lam_cm = HC_EV_CM / np.asarray(energy_eV, dtype=float)
    lam_cm = np.atleast_1d(lam_cm)
    od = np.atleast_1d(OD).astype(float)
    beta = od * lam_cm / (4 * np.pi * thickness_cm)
    return beta
