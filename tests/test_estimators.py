import numpy as np

from stxm.estimators import WeightingMode, region_mean_and_sigma
from stxm.nexafs import _weighted_mean_and_sigma


def test_inverse_count_matches_legacy(poisson_block):
    counts, mask, _ = poisson_block
    legacy_mean, legacy_sigma, legacy_n = _weighted_mean_and_sigma(counts, mask)
    mean, sigma, n = region_mean_and_sigma(counts, mask, mode=WeightingMode.INVERSE_COUNT)
    np.testing.assert_allclose(mean, legacy_mean)
    np.testing.assert_allclose(sigma, legacy_sigma)
    assert n == legacy_n


def test_poisson_mle_unbiased(poisson_block):
    counts, mask, rate = poisson_block
    mean, _, _ = region_mean_and_sigma(counts, mask, mode=WeightingMode.POISSON_MLE)
    assert abs(float(np.nanmean(mean)) - rate) < 2.0


def test_poisson_sigma_matches_empirical_scatter():
    rng = np.random.default_rng(0)
    rate = 8.0
    n_spatial, n_energy = 40, 1
    mask = np.ones(n_spatial, dtype=bool)
    estimates = []
    for _ in range(400):
        trial = rng.poisson(rate, size=(n_spatial, n_energy)).astype(np.float64)
        m, _, _ = region_mean_and_sigma(trial, mask, mode=WeightingMode.POISSON_MLE)
        estimates.append(float(m[0]))
    empirical_std = float(np.std(estimates, ddof=1))
    trial = rng.poisson(rate, size=(n_spatial, n_energy)).astype(np.float64)
    _, sigma_p, _ = region_mean_and_sigma(trial, mask, mode=WeightingMode.POISSON_MLE)
    _, sigma_i, _ = region_mean_and_sigma(trial, mask, mode=WeightingMode.INVERSE_COUNT)
    assert abs(float(sigma_p[0]) - empirical_std) < 0.25
    assert float(sigma_i[0]) < float(sigma_p[0])
