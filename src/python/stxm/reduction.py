"""Region-resolved NEXAFS reduction: two-region Beer-Lambert and thickness regression."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from stxm.estimators import WeightingMode
from stxm.io import orient_scan
from stxm.nexafs import nexafs_beer_lambert
from stxm.regions import sample_izero_masks


class ReductionMethod(StrEnum):
    """Recorded reduction path for provenance."""

    TWO_REGION = "two_region"
    THICKNESS_REGRESSION = "thickness_regression"


@dataclass
class RegionSpectrum:
    """
    One reduced NEXAFS spectrum for a spatial region or regression fit.

    Attributes
    ----------
    energy_eV : np.ndarray
        Energy axis in eV.
    OD : np.ndarray
        Optical density or regression slope spectrum.
    OD_err : np.ndarray
        Standard error on ``OD``.
    region_label : str
        Human-readable region name.
    weighting_mode : str
        ``WeightingMode`` value used for averaging.
    reduction_method : str
        ``ReductionMethod`` value.
    n_pixels : int
        Number of spatial rows contributing.
    diagnostics : dict
        Optional regression or segmentation metadata.
    """

    energy_eV: np.ndarray
    OD: np.ndarray
    OD_err: np.ndarray
    region_label: str
    weighting_mode: str
    reduction_method: str
    n_pixels: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def reduce_two_region(
    image: np.ndarray,
    sample_mask: np.ndarray,
    izero_mask: np.ndarray,
    energy: np.ndarray,
    *,
    region_label: str = "sample",
    mode: WeightingMode = WeightingMode.POISSON_MLE,
    eps: float = 1e-10,
) -> RegionSpectrum:
    """
    Reduce a scan with the legacy two-region Beer-Lambert ratio.

    Parameters
    ----------
    image : np.ndarray
        Oriented scan ``(n_spatial, n_energy)``.
    sample_mask, izero_mask : np.ndarray
        Boolean masks along the spatial axis.
    energy : np.ndarray
        Energy axis aligned with image columns.
    region_label : str
        Label stored on the spectrum.
    mode : WeightingMode
        Region averaging mode.
    eps : float
        Intensity floor for logarithms.

    Returns
    -------
    RegionSpectrum
        Reduced spectrum with method ``two_region``.
    """
    od, sigma_od, _, _, _, _, n_sample, _ = nexafs_beer_lambert(
        image, sample_mask, izero_mask, eps=eps, mode=mode
    )
    return RegionSpectrum(
        energy_eV=np.asarray(energy, dtype=np.float64),
        OD=od,
        OD_err=sigma_od,
        region_label=region_label,
        weighting_mode=mode.value,
        reduction_method=ReductionMethod.TWO_REGION.value,
        n_pixels=n_sample,
    )


def reduce_by_regression(
    image: np.ndarray,
    film_mask: np.ndarray,
    thickness_proxy: np.ndarray,
    energy: np.ndarray,
    *,
    region_label: str = "film",
    mode: WeightingMode = WeightingMode.POISSON_MLE,
    eps: float = 1e-10,
) -> RegionSpectrum:
    """
    Regress ``-ln(I)`` against a thickness proxy at each energy.

    Parameters
    ----------
    image : np.ndarray
        Oriented scan ``(n_spatial, n_energy)``.
    film_mask : np.ndarray
        Boolean mask selecting film pixels.
    thickness_proxy : np.ndarray
        One thickness surrogate per spatial row (same length as mask axis).
    energy : np.ndarray
        Energy axis in eV.
    region_label : str
        Label stored on the spectrum.
    mode : WeightingMode
        Accepted for provenance; regression uses per-pixel intensities directly.
    eps : float
        Intensity floor before the logarithm.

    Returns
    -------
    RegionSpectrum
        Slope spectrum as ``OD`` with method ``thickness_regression``.

    Raises
    ------
    ValueError
        If fewer than two film pixels are selected or the design matrix is rank-deficient.
    """
    film_mask = np.asarray(film_mask, dtype=bool)
    proxy = np.asarray(thickness_proxy, dtype=np.float64).ravel()
    block = np.maximum(image[film_mask, :], eps)
    t = proxy[film_mask]
    n = int(t.size)
    if n < 2:
        raise ValueError("film_mask must select at least two spatial rows")
    y = -np.log(block)
    design = np.column_stack([t, np.ones(n, dtype=np.float64)])
    coeffs, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("thickness_proxy must vary across film pixels")
    slopes = coeffs[0]
    fitted = design @ coeffs
    od_err = np.std(y - fitted, axis=0, ddof=2) / np.sqrt(n)
    return RegionSpectrum(
        energy_eV=np.asarray(energy, dtype=np.float64),
        OD=slopes,
        OD_err=od_err,
        region_label=region_label,
        weighting_mode=mode.value,
        reduction_method=ReductionMethod.THICKNESS_REGRESSION.value,
        n_pixels=n,
        diagnostics={
            "thickness_proxy_min": float(np.min(t)),
            "thickness_proxy_max": float(np.max(t)),
        },
    )


def thickness_proxy_from_reference_od(
    image: np.ndarray,
    film_mask: np.ndarray,
    energy: np.ndarray,
    reference_energy: float,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """
  Build a per-row thickness surrogate from OD at one reference energy.

    Parameters
    ----------
    image : np.ndarray
        Oriented scan ``(n_spatial, n_energy)``.
    film_mask : np.ndarray
        Mask used only to validate row count; proxy is computed for all rows.
    energy : np.ndarray
        Energy axis in eV.
    reference_energy : float
        Energy at which ``-ln(I)`` defines relative thickness.
    eps : float
        Intensity floor.

    Returns
    -------
    np.ndarray
        Length ``n_spatial``; ``nan`` outside the film mask is not applied (full axis returned).
    """
    energy = np.asarray(energy, dtype=np.float64)
    idx = int(np.argmin(np.abs(energy - reference_energy)))
    column = np.maximum(image[:, idx], eps)
    proxy = -np.log(column)
    if not np.any(film_mask):
        return proxy
    return proxy


def reduce_regions(
    meta: dict,
    image: np.ndarray,
    row_labels: np.ndarray,
    label_names: list[str],
    *,
    method: ReductionMethod = ReductionMethod.TWO_REGION,
    izero_bounds: tuple[float, float, float, float] | None = None,
    reference_energy: float | None = None,
    mode: WeightingMode = WeightingMode.POISSON_MLE,
    eps: float = 1e-10,
) -> list[RegionSpectrum]:
    """
    Reduce one spectrum per segmented spatial label.

    Parameters
    ----------
    meta : dict
        STXM header dict from ``read_hdr`` / ``load_stxm``.
    image : np.ndarray
        Raw image from ``load_stxm``.
    row_labels : np.ndarray
        Integer label per spatial row from segmentation.
    label_names : list of str
        Name per label index.
    method : ReductionMethod
        ``two_region`` uses izero bounds; ``thickness_regression`` fits each sample label.
    izero_bounds : tuple of float, optional
        ``(sample_lo, sample_hi, izero_lo, izero_hi)`` in axis coordinates for
        ``two_region``; required when ``method`` is ``two_region``.
    reference_energy : float, optional
        Reference eV for the thickness proxy when ``method`` is ``thickness_regression``.
    mode : WeightingMode
        Region averaging mode for ``two_region``.
    eps : float
        Intensity floor.

    Returns
    -------
    list of RegionSpectrum
        One entry per sample-class label (excludes edge and izero names).

    Raises
    ------
    ValueError
        If ``izero_bounds`` is missing for ``two_region`` or labels are inconsistent.
    """
    energy, spatial, oriented = orient_scan(meta, image)
    row_labels = np.asarray(row_labels, dtype=int)
    spectra: list[RegionSpectrum] = []
    izero_name = "izero"
    sample_indices = [
        i for i, name in enumerate(label_names) if name not in (izero_name, "edge")
    ]
    if method is ReductionMethod.TWO_REGION:
        if izero_bounds is None:
            raise ValueError("izero_bounds required for two_region reduction")
        sa_lo, sa_hi, iz_lo, iz_hi = izero_bounds
        _, izero_mask = sample_izero_masks(spatial, 0.0, 0.0, iz_lo, iz_hi)
        for idx in sample_indices:
            sample_mask = row_labels == idx
            if not np.any(sample_mask):
                continue
            bounds_lo = float(np.min(spatial[sample_mask]))
            bounds_hi = float(np.max(spatial[sample_mask]))
            sample_mask, _ = sample_izero_masks(spatial, bounds_lo, bounds_hi, iz_lo, iz_hi)
            spectra.append(
                reduce_two_region(
                    oriented,
                    sample_mask,
                    izero_mask,
                    energy,
                    region_label=label_names[idx],
                    mode=mode,
                    eps=eps,
                )
            )
        return spectra
    ref_e = float(energy[len(energy) // 2] if reference_energy is None else reference_energy)
    film_mask = np.isin(row_labels, sample_indices)
    proxy_full = thickness_proxy_from_reference_od(oriented, film_mask, energy, ref_e, eps=eps)
    for idx in sample_indices:
        region_mask = row_labels == idx
        if not np.any(region_mask):
            continue
        spectra.append(
            reduce_by_regression(
                oriented,
                region_mask,
                proxy_full,
                energy,
                region_label=label_names[idx],
                mode=mode,
                eps=eps,
            )
        )
    return spectra


def reduce_loaded_scan_two_region(
    meta: dict,
    image: np.ndarray,
    sample_lo: float,
    sample_hi: float,
    izero_lo: float,
    izero_hi: float,
    *,
    region_label: str = "sample",
    mode: WeightingMode = WeightingMode.POISSON_MLE,
    eps: float = 1e-10,
) -> RegionSpectrum:
    """
    Orient a loaded scan and apply two-region Beer-Lambert reduction.

    Parameters
    ----------
    meta, image : dict, np.ndarray
        Outputs of ``load_stxm``.
    sample_lo, sample_hi, izero_lo, izero_hi : float
        Region bounds in spatial axis coordinates.
    region_label : str
        Stored region name.
    mode : WeightingMode
        Region averaging mode.
    eps : float
        Intensity floor.

    Returns
    -------
    RegionSpectrum
        Reduced spectrum.
    """
    energy, spatial, oriented = orient_scan(meta, image)
    sample_mask, izero_mask = sample_izero_masks(
        spatial, sample_lo, sample_hi, izero_lo, izero_hi
    )
    return reduce_two_region(
        oriented,
        sample_mask,
        izero_mask,
        energy,
        region_label=region_label,
        mode=mode,
        eps=eps,
    )


def region_spectrum_to_nexafs_df(spectrum: RegionSpectrum) -> dict[str, np.ndarray | int | str]:
    """
    Map a ``RegionSpectrum`` to legacy NEXAFS parquet column arrays.

    Parameters
    ----------
    spectrum : RegionSpectrum
        Reduced spectrum.

    Returns
    -------
    dict
        Keys compatible with ``experiment.NEXAFS_COLUMNS`` where available.
    """
    return {
        "energy_eV": spectrum.energy_eV,
        "OD": spectrum.OD,
        "OD_err": spectrum.OD_err,
        "I0": np.full_like(spectrum.OD, np.nan),
        "I0_err": np.full_like(spectrum.OD, np.nan),
        "I": np.full_like(spectrum.OD, np.nan),
        "I_err": np.full_like(spectrum.OD, np.nan),
        "n_sample": spectrum.n_pixels,
        "n_izero": 0,
    }
