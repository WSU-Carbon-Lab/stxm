import numpy as np

from stxm.estimators import WeightingMode
from stxm.io import orient_scan
from stxm.nexafs import nexafs_beer_lambert
from stxm.reduction import reduce_by_regression, reduce_two_region


def _synthetic_film(
    n_spatial: int = 60,
    n_energy: int = 20,
    mu: float = 0.05,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    energy = np.linspace(280.0, 292.0, n_energy)
    spatial = np.linspace(0.0, 1.0, n_spatial)
    thickness = spatial
    mu_spec = mu * (energy - energy[0] + 1.0)
    image = np.zeros((n_spatial, n_energy), dtype=np.float64)
    for j, mu_j in enumerate(mu_spec):
        image[:, j] = np.exp(-mu_j * thickness)
    meta = {
        "paxis_points": energy,
        "qaxis_points": spatial,
        "paxis_name": "Energy",
        "qaxis_name": "Sample",
    }
    return meta, image, energy, thickness


def test_regression_recovers_absorption():
    meta, image, energy, thickness = _synthetic_film()
    film_mask = np.ones(image.shape[0], dtype=bool)
    spec = reduce_by_regression(image, film_mask, thickness, energy)
    expected = 0.05 * (energy - energy[0] + 1.0)
    np.testing.assert_allclose(spec.OD, expected, rtol=0.15, atol=0.02)


def test_regression_invariant_to_izero_scale():
    meta, image, energy, thickness = _synthetic_film()
    scaled = image * 3.0
    film_mask = np.ones(image.shape[0], dtype=bool)
    base = reduce_by_regression(image, film_mask, thickness, energy)
    scaled_spec = reduce_by_regression(scaled, film_mask, thickness, energy)
    np.testing.assert_allclose(base.OD, scaled_spec.OD, rtol=1e-6)


def test_two_region_matches_legacy_nexafs():
    meta, image, energy, _ = _synthetic_film()
    n = image.shape[0]
    sample_mask = np.zeros(n, dtype=bool)
    izero_mask = np.zeros(n, dtype=bool)
    sample_mask[10:40] = True
    izero_mask[:8] = True
    legacy = nexafs_beer_lambert(image, sample_mask, izero_mask, mode=WeightingMode.POISSON_MLE)
    spec = reduce_two_region(
        image, sample_mask, izero_mask, energy, mode=WeightingMode.POISSON_MLE
    )
    np.testing.assert_allclose(spec.OD, legacy[0])
    np.testing.assert_allclose(spec.OD_err, legacy[1])


def test_orient_scan_transposes_when_energy_on_rows():
    meta, image, energy, _ = _synthetic_film()
    transposed_meta = {
        "paxis_points": meta["qaxis_points"],
        "qaxis_points": meta["paxis_points"],
        "paxis_name": "Sample",
        "qaxis_name": "Energy",
    }
    transposed_image = image.T
    e1, _, img1 = orient_scan(meta, image)
    e2, _, img2 = orient_scan(transposed_meta, transposed_image)
    np.testing.assert_allclose(e1, e2)
    np.testing.assert_allclose(img1, img2)
