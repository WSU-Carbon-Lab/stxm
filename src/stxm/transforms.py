from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from stxm.absorption import HC_EV_CM, fit_bare_atom_background, mass_absorption_cm2_per_g, od_to_beta


def normalize_spot_label(raw: Any) -> str:
    """
    Normalize a raw spot label value.

    Parameters
    ----------
    raw : Any
        Raw label value, possibly None or empty.

    Returns
    -------
    str
        Normalized label string, defaulting to ``\"pure\"``.
    """
    text = str(raw).strip() if raw is not None else ""
    return text or "pure"


def compute_display_curve(
    mode: str,
    energy: np.ndarray,
    od: np.ndarray,
    sigma_od: np.ndarray,
    formula: str | None,
    fit_option: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Compute the displayed y-curve and uncertainties for a chosen mode.

    Parameters
    ----------
    mode : str
        Display mode: ``\"OD\"``, ``\"Norm. Mass Abs. (g/cm^2)\"``, or ``\"Beta\"``.
    energy : np.ndarray
        Energy axis in eV.
    od : np.ndarray
        Optical density values.
    sigma_od : np.ndarray
        Uncertainties in OD.
    formula : str or None
        Chemical formula for mass absorption, if available.
    fit_option : str
        Fit mode description, used to choose fitting windows.

    Returns
    -------
    y : np.ndarray
        Displayed y values.
    sigma_y : np.ndarray
        Uncertainties in y.
    label : str
        Y-axis label.
    """
    mode_clean = (mode or "OD").strip()
    energy = np.asarray(energy, dtype=float)
    od = np.asarray(od, dtype=float)
    sigma_od = np.asarray(sigma_od, dtype=float)
    if mode_clean == "OD":
        return od, sigma_od, "OD (ln I0/I)"
    if mode_clean == "Norm. Mass Abs. (g/cm^2)":
        if formula is None:
            return od, sigma_od, "OD (set sample in Reduction for mass abs)"
        try:
            mu_rho = mass_absorption_cm2_per_g(formula, energy, None)
            opt = fit_option or ""
            n_low = 5 if "offset" in opt.lower() else 0
            n_high = 5
            scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
            scale = scale if scale != 0 else 1.0
            y = (od - const) / scale
            sigma_y = np.abs(sigma_od / scale)
            return y, sigma_y, r"Norm. Mass Abs. (g/cm$^2$)"
        except Exception:
            return od, sigma_od, "OD (ln I0/I)"
    if mode_clean == "Beta":
        if formula is not None:
            try:
                mu_rho = mass_absorption_cm2_per_g(formula, energy, None)
                opt = fit_option or ""
                n_low = 5 if "offset" in opt.lower() else 0
                n_high = 5
                scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
                scale = scale if scale != 0 else 1.0
                mu_safe = np.where(mu_rho > 1e-30, mu_rho, 1e-30)
                norm_mu = (od - const) / scale / mu_safe
                lam_cm = HC_EV_CM / energy
                beta_bare = np.atleast_1d(mu_rho) * np.atleast_1d(lam_cm) / (4 * np.pi)
                y = norm_mu * beta_bare
                sigma_norm = np.abs(sigma_od / (scale * mu_safe))
                sigma_y = sigma_norm * np.atleast_1d(beta_bare)
                return y, sigma_y, "beta (from norm. mass abs)"
            except Exception:
                pass
        try:
            t_cm = 1e-4
            y = od_to_beta(energy, od, t_cm)
            lam_cm = HC_EV_CM / energy
            sigma_y = sigma_od * np.atleast_1d(lam_cm) / (4 * np.pi * t_cm)
            return y, sigma_y, "beta (Im n)"
        except Exception:
            return od, sigma_od, "OD (ln I0/I)"
    return od, sigma_od, "OD (ln I0/I)"


def add_derived_columns(
    df: pd.DataFrame,
    energy: np.ndarray,
    od: np.ndarray,
    od_err: np.ndarray,
    formula: str | None,
    fit_option: str,
) -> pd.DataFrame:
    """
    Add beta, beta_err, and mass absorption columns to a NEXAFS DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Base dataframe with NEXAFS columns.
    energy : np.ndarray
        Energy axis in eV.
    od : np.ndarray
        Optical density.
    od_err : np.ndarray
        Uncertainties in OD.
    formula : str or None
        Chemical formula for mass absorption; if None, mass absorption columns
        are filled with NaN.
    fit_option : str
        Fit mode description, used to choose fitting windows.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with added columns ``beta``, ``beta_err``, and
        optionally ``mass_absorption`` and ``mass_absorption_err``.
    """
    energy = np.asarray(energy, dtype=float)
    od = np.asarray(od, dtype=float)
    od_err = np.asarray(od_err, dtype=float)
    t_cm = 1e-4
    lam_cm = HC_EV_CM / energy
    beta = od_to_beta(energy, od, t_cm)
    beta_err = od_err * lam_cm / (4 * np.pi * t_cm)
    out = df.copy()
    out["beta"] = beta
    out["beta_err"] = beta_err
    if formula:
        try:
            mu_rho = mass_absorption_cm2_per_g(formula, energy, None)
            opt = fit_option or ""
            n_low = 5 if "offset" in opt.lower() else 0
            n_high = 5
            scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
            scale = scale if scale != 0 else 1.0
            out["mass_absorption"] = (od - const) / scale
            out["mass_absorption_err"] = np.abs(od_err / scale)
        except Exception:
            out["mass_absorption"] = np.nan
            out["mass_absorption_err"] = np.nan
    else:
        out["mass_absorption"] = np.nan
        out["mass_absorption_err"] = np.nan
    return out

