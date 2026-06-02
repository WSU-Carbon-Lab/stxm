import numpy as np

from stxm.demix import demix_nmf, demix_svd, scree_singular_values


def _two_component_blend():
    rng = np.random.default_rng(7)
    n_pixels, n_energy = 80, 25
    spec_a = np.linspace(0.2, 1.0, n_energy)
    spec_b = np.linspace(1.0, 0.3, n_energy)
    abund_a = rng.uniform(0.0, 1.0, n_pixels)
    abund_b = 1.0 - abund_a
    X = np.outer(abund_a, spec_a) + np.outer(abund_b, spec_b)
    X += rng.normal(0.0, 0.02, size=X.shape)
    X = np.maximum(X, 0.0)
    return X, spec_a, spec_b, abund_a, abund_b


def test_nmf_recovers_components():
    X, spec_a, spec_b, abund_a, abund_b = _two_component_blend()
    result = demix_nmf(X, 2, random_state=0)
    comps = result.component_spectra
    corr_a = max(
        abs(np.corrcoef(comps[0], spec_a)[0, 1]),
        abs(np.corrcoef(comps[1], spec_a)[0, 1]),
    )
    corr_b = max(
        abs(np.corrcoef(comps[0], spec_b)[0, 1]),
        abs(np.corrcoef(comps[1], spec_b)[0, 1]),
    )
    assert corr_a > 0.9
    assert corr_b > 0.9
    scores = result.abundances
    corr_map_a = abs(np.corrcoef(scores[:, 0], abund_a)[0, 1])
    corr_map_b = abs(np.corrcoef(scores[:, 1], abund_b)[0, 1])
    assert max(corr_map_a, corr_map_b) > 0.75


def test_svd_rank_two_clean_data():
    X, spec_a, spec_b, _, _ = _two_component_blend()
    X_clean = np.outer(np.linspace(0, 1, X.shape[0]), spec_a) + np.outer(
        1.0 - np.linspace(0, 1, X.shape[0]), spec_b
    )
    singular = scree_singular_values(X_clean)
    result = demix_svd(X_clean, 2)
    assert singular[1] > 1e-6
    assert result.component_spectra.shape[0] == 2
