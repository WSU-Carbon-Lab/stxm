"""Spatial-spectral demixing via SVD and NMF."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF


@dataclass
class Decomposition:
    """
    Factorization of a spatial-spectral matrix.

    Attributes
    ----------
    component_spectra : np.ndarray
        Shape ``(n_components, n_energy)``.
    abundances : np.ndarray
        Shape ``(n_spatial,)`` or ``(n_spatial, n_components)`` after reshape.
    spatial_shape : tuple of int
        Original spatial grid for reshaping abundances.
    explained_variance_ratio : np.ndarray | None
        Per-component variance fraction for SVD; ``None`` for NMF.
    reconstruction_residual : float
        Frobenius norm of ``X - reconstruction``.
    method : str
        ``svd`` or ``nmf``.
    """

    component_spectra: np.ndarray
    abundances: np.ndarray
    spatial_shape: tuple[int, ...]
    explained_variance_ratio: np.ndarray | None
    reconstruction_residual: float
    method: str


def scree_singular_values(X: np.ndarray) -> np.ndarray:
    """
    Return singular values of ``X`` for rank selection.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n_pixels, n_energy)``.

    Returns
    -------
    np.ndarray
        Singular values in descending order.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be 2D")
    return np.linalg.svd(matrix, compute_uv=False)


def demix_svd(X: np.ndarray, n_components: int) -> Decomposition:
    """
    PCA-style decomposition with signed component spectra.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n_pixels, n_energy)``.
    n_components : int
        Number of components to retain.

    Returns
    -------
    Decomposition
        Component spectra and spatial scores.

    Raises
    ------
    ValueError
        If ``n_components`` is out of range.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be 2D")
    n_pixels, n_energy = matrix.shape
    rank = min(n_components, n_pixels, n_energy)
    if rank < 1:
        raise ValueError("n_components must be at least 1")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:rank]
    scores = u[:, :rank] * singular[:rank]
    recon = scores @ components + np.mean(matrix, axis=0, keepdims=True)
    residual = float(np.linalg.norm(matrix - recon))
    total_var = float(np.sum(centered**2))
    if total_var > 0:
        evr = (singular[:rank] ** 2) / total_var
    else:
        evr = np.zeros(rank, dtype=np.float64)
    return Decomposition(
        component_spectra=components,
        abundances=scores,
        spatial_shape=(n_pixels, rank),
        explained_variance_ratio=evr,
        reconstruction_residual=residual,
        method="svd",
    )


def demix_nmf(
    X: np.ndarray,
    n_components: int,
    *,
    init: str = "nndsvda",
    max_iter: int = 500,
    random_state: int = 0,
) -> Decomposition:
    """
    Non-negative matrix factorization of spatial-spectral data.

    Recovered spectra are component-like: NMF is not unique up to scaling and
    rotation, so components should be confirmed against reference spectra.

    Parameters
    ----------
    X : np.ndarray
        Non-negative data matrix ``(n_pixels, n_energy)``.
    n_components : int
        Number of components.
    init : str
        sklearn NMF initialization method.
    max_iter : int
        Maximum iterations.
    random_state : int
        Random seed.

    Returns
    -------
    Decomposition
        Non-negative component spectra and abundances.

    Raises
    ------
    ValueError
        If ``X`` contains negative values or rank is invalid.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be 2D")
    if np.any(matrix < 0):
        raise ValueError("NMF requires non-negative X")
    n_pixels, n_energy = matrix.shape
    rank = min(n_components, n_pixels, n_energy)
    if rank < 1:
        raise ValueError("n_components must be at least 1")
    model = NMF(
        n_components=rank,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
    )
    abundances = model.fit_transform(matrix)
    components = model.components_
    recon = abundances @ components
    residual = float(np.linalg.norm(matrix - recon))
    return Decomposition(
        component_spectra=components,
        abundances=abundances,
        spatial_shape=(n_pixels, rank),
        explained_variance_ratio=None,
        reconstruction_residual=residual,
        method="nmf",
    )


def abundances_as_map(
    decomposition: Decomposition,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """
    Reshape abundances to ``(n_rows, n_cols, n_components)``.

    Parameters
    ----------
    decomposition : Decomposition
        Result from ``demix_svd`` or ``demix_nmf``.
    spatial_shape : tuple of int
        ``(n_spatial, n_other)``; only the row count is used.

    Returns
    -------
    np.ndarray
        Abundance map; for 1D line scans ``n_other`` is 1.
    """
    n_spatial = spatial_shape[0]
    abund = np.asarray(decomposition.abundances, dtype=np.float64)
    if abund.ndim == 1:
        return abund.reshape(n_spatial, 1)
    n_comp = abund.shape[1]
    return abund.reshape(n_spatial, n_comp)
