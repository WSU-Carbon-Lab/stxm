"""Constrained linear combination fitting of NEXAFS spectra."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from stxm.grid import common_energy_grid, interpolate_spectrum


@dataclass
class Spectrum:
    """
    Spectrum on an energy grid for LCF.

    Attributes
    ----------
    energy_eV : np.ndarray
        Energy axis in eV.
    OD : np.ndarray
        Optical density or absorption signal.
    OD_err : np.ndarray
        Per-point standard errors; zeros are replaced with a small floor.
    label : str
        Identifier for plotting and results.
    """

    energy_eV: np.ndarray
    OD: np.ndarray
    OD_err: np.ndarray
    label: str = ""


@dataclass
class LCFResult:
    """
    Outcome of a linear combination fit.

    Attributes
    ----------
    fractions : np.ndarray
        Component fractions aligned with ``reference_labels``.
    fraction_covariance : np.ndarray
        Approximate covariance of ``fractions``.
    reduced_chi_square : float
        Weighted residual sum of squares divided by degrees of freedom.
    residual : np.ndarray
        Target minus model on ``energy_grid``.
    energy_grid : np.ndarray
        Energy axis used for the fit.
    reference_labels : list of str
        Reference spectrum labels.
    """

    fractions: np.ndarray
    fraction_covariance: np.ndarray
    reduced_chi_square: float
    residual: np.ndarray
    energy_grid: np.ndarray
    reference_labels: list[str]


def _sigma_floor(sigma: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    out = np.asarray(sigma, dtype=np.float64).copy()
    out[~np.isfinite(out) | (out <= 0)] = floor
    return out


def _prepare_lcf_grid(
    target: Spectrum,
    references: list[Spectrum],
    energy_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = [target.energy_eV, *[r.energy_eV for r in references]]
    grid = (
        np.asarray(energy_grid, dtype=np.float64).ravel()
        if energy_grid is not None
        else common_energy_grid(grids)
    )
    y = interpolate_spectrum(target.energy_eV, target.OD, grid)
    sigma = _sigma_floor(interpolate_spectrum(target.energy_eV, target.OD_err, grid))
    valid = np.isfinite(y) & np.isfinite(sigma)
    for ref in references:
        valid &= np.isfinite(interpolate_spectrum(ref.energy_eV, ref.OD, grid))
    if not np.any(valid):
        raise ValueError("no finite points on the common energy grid")
    grid = grid[valid]
    y = y[valid]
    sigma = sigma[valid]
    design = np.column_stack([
        interpolate_spectrum(r.energy_eV, r.OD, grid) for r in references
    ])
    return grid, y, sigma, design


def _default_initial_fractions(n_ref: int, sum_to_one: bool) -> np.ndarray:
    if sum_to_one and n_ref > 0:
        return np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
    return np.zeros(n_ref, dtype=np.float64)


def _default_fraction_bounds(n_ref: int, non_negative: bool) -> list[tuple[float, float]]:
    if non_negative:
        return [(0.0, 1.0) for _ in range(n_ref)]
    return [(-np.inf, np.inf) for _ in range(n_ref)]


def _validate_lcf_inputs(
    n_ref: int,
    initial_fractions: np.ndarray,
    fraction_bounds: list[tuple[float, float]],
    fixed: list[bool],
) -> None:
    if initial_fractions.shape != (n_ref,):
        raise ValueError("initial_fractions length must match references")
    if len(fraction_bounds) != n_ref:
        raise ValueError("fraction_bounds length must match references")
    if len(fixed) != n_ref:
        raise ValueError("fixed length must match references")
    for idx, is_fixed in enumerate(fixed):
        lo, hi = fraction_bounds[idx]
        val = float(initial_fractions[idx])
        if lo is not None and val < lo - 1e-12:
            raise ValueError(
                f"initial_fractions[{idx}]={val} below bound minimum {lo}"
            )
        if hi is not None and val > hi + 1e-12:
            raise ValueError(
                f"initial_fractions[{idx}]={val} above bound maximum {hi}"
            )
        if is_fixed and not np.isfinite(val):
            raise ValueError(f"fixed component {idx} requires finite initial_fraction")


def preview_lcf_model(
    target: Spectrum,
    references: list[Spectrum],
    fractions: np.ndarray,
    energy_grid: np.ndarray | None = None,
    *,
    normalize_fractions: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a linear-combination model on the common energy grid for UI preview.

    Parameters
    ----------
    target : Spectrum
        Unknown spectrum to display alongside the model.
    references : list of Spectrum
        Basis spectra; must be non-empty.
    fractions : np.ndarray
        Component weights aligned with ``references``; values in ``[0, 1]`` or
        percent-scale (any positive scale when ``normalize_fractions`` is True).
    energy_grid : np.ndarray, optional
        Common grid; built from overlap when omitted.
    normalize_fractions : bool
        If True and the fraction sum is non-zero and not unity, scale fractions
        to sum to one before forming the model.

    Returns
    -------
    grid : np.ndarray
        Energy axis in eV.
    model : np.ndarray
        Weighted sum of reference spectra on ``grid``.
    target_on_grid : np.ndarray
        Target OD interpolated onto ``grid``.

    Raises
    ------
    ValueError
        If ``references`` is empty, ``fractions`` length mismatches, or grids
        do not overlap.
    """
    if not references:
        raise ValueError("references must be non-empty")
    fracs = np.asarray(fractions, dtype=np.float64).ravel()
    n_ref = len(references)
    if fracs.size != n_ref:
        raise ValueError("fractions length must match references")
    grid, y, _sigma, design = _prepare_lcf_grid(target, references, energy_grid)
    if normalize_fractions:
        total = float(np.sum(fracs))
        if total != 0.0 and abs(total - 1.0) > 1e-9:
            fracs = fracs / total
    model = design @ fracs
    return grid, model, y


def fit_lcf(
    target: Spectrum,
    references: list[Spectrum],
    *,
    non_negative: bool = True,
    sum_to_one: bool = False,
    energy_grid: np.ndarray | None = None,
    initial_fractions: np.ndarray | None = None,
    fraction_bounds: list[tuple[float, float]] | None = None,
    fixed: list[bool] | None = None,
) -> LCFResult:
    """
    Fit ``target`` as a weighted linear combination of ``references``.

    Parameters
    ----------
    target : Spectrum
        Unknown spectrum with uncertainties.
    references : list of Spectrum
        Basis spectra; must be non-empty.
    non_negative : bool
        If True, lower bounds are clipped to be at least zero.
    sum_to_one : bool
        If True, optimized fractions are constrained to sum to one; fixed
        components contribute their initial values toward that sum.
    energy_grid : np.ndarray, optional
        Common grid; built from overlap when omitted.
    initial_fractions : np.ndarray, optional
        Starting fractions in ``[0, 1]`` aligned with ``references``; used as
        the optimizer initial guess and held exactly for fixed components.
    fraction_bounds : list of tuple of float, optional
        Per-component ``(lower, upper)`` bounds in fraction units; ``None`` in
        a bound means unbounded on that side.
    fixed : list of bool, optional
        When ``fixed[i]`` is True, ``initial_fractions[i]`` is held during the
        fit and excluded from optimization.

    Returns
    -------
    LCFResult
        Fractions, covariance estimate, reduced chi-square, and residual.

    Raises
    ------
    ValueError
        If ``references`` is empty, grids do not overlap, array lengths
        mismatch, or a fixed component initial value lies outside its bounds.
    """
    if not references:
        raise ValueError("references must be non-empty")
    n_ref = len(references)
    grid, y, sigma, design = _prepare_lcf_grid(target, references, energy_grid)
    weights = 1.0 / sigma**2

    if initial_fractions is None:
        x_init = _default_initial_fractions(n_ref, sum_to_one)
    else:
        x_init = np.asarray(initial_fractions, dtype=np.float64).ravel().copy()
    bounds_list = (
        list(fraction_bounds)
        if fraction_bounds is not None
        else _default_fraction_bounds(n_ref, non_negative)
    )
    fixed_list = list(fixed) if fixed is not None else [False] * n_ref
    _validate_lcf_inputs(n_ref, x_init, bounds_list, fixed_list)

    free_mask = np.array([not f for f in fixed_list], dtype=bool)
    fixed_mask = ~free_mask
    fixed_sum = float(np.sum(x_init[fixed_mask])) if np.any(fixed_mask) else 0.0

    if sum_to_one and fixed_sum > 1.0 + 1e-9:
        raise ValueError("fixed fractions sum exceeds one under sum_to_one")

    def full_fractions(free_vals: np.ndarray) -> np.ndarray:
        out = x_init.copy()
        out[free_mask] = free_vals
        return out

    def objective(free_vals: np.ndarray) -> float:
        coeffs = full_fractions(free_vals)
        model = design @ coeffs
        resid = y - model
        return float(np.sum(weights * resid**2))

    opt_bounds: list[tuple[float | None, float | None]] = []
    for idx in range(n_ref):
        if not free_mask[idx]:
            continue
        lo, hi = bounds_list[idx]
        if non_negative and lo is not None:
            lo = max(0.0, float(lo))
        elif non_negative and lo is None:
            lo = 0.0
        opt_bounds.append((lo, hi))

    constraints: list[dict] = []
    if sum_to_one and np.any(free_mask):
        constraints.append({
            "type": "eq",
            "fun": lambda c: float(np.sum(c) + fixed_sum - 1.0),
        })

    if np.any(free_mask):
        x0_free = x_init[free_mask]
        result = minimize(
            objective,
            x0_free,
            method="SLSQP",
            bounds=opt_bounds,
            constraints=constraints,
        )
        fractions = full_fractions(np.asarray(result.x, dtype=np.float64))
    else:
        fractions = x_init.copy()

    model = design @ fractions
    resid = y - model
    n_free = int(np.sum(free_mask))
    dof = max(int(grid.size) - n_free, 1)
    chi2 = float(np.sum((resid / sigma) ** 2))
    reduced = chi2 / dof
    try:
        cov = np.linalg.inv(design.T @ (design * weights[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((n_ref, n_ref), np.nan, dtype=np.float64)
    labels = [r.label or f"ref_{i}" for i, r in enumerate(references)]
    return LCFResult(
        fractions=fractions,
        fraction_covariance=cov,
        reduced_chi_square=reduced,
        residual=resid,
        energy_grid=grid,
        reference_labels=labels,
    )


def rank_reference_subsets(
    target: Spectrum,
    references: list[Spectrum],
    subset_sizes: list[int],
    **fit_kwargs,
) -> list[tuple[tuple[int, ...], LCFResult]]:
    """
    Compare reference subsets by reduced chi-square.

    Parameters
    ----------
    target : Spectrum
        Unknown spectrum.
    references : list of Spectrum
        Full reference pool.
    subset_sizes : list of int
        Component counts to try (e.g. ``[2, 3]``).
    **fit_kwargs
        Forwarded to ``fit_lcf``.

    Returns
    -------
    list of tuple
        ``(index_tuple, LCFResult)`` sorted by increasing reduced chi-square.
    """
    from itertools import combinations

    results: list[tuple[tuple[int, ...], LCFResult]] = []
    n = len(references)
    for k in subset_sizes:
        if k < 1 or k > n:
            continue
        for indices in combinations(range(n), k):
            subset = [references[i] for i in indices]
            try:
                fit = fit_lcf(target, subset, **fit_kwargs)
            except ValueError:
                continue
            results.append((indices, fit))
    results.sort(key=lambda item: item[1].reduced_chi_square)
    return results
