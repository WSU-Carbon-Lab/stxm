import numpy as np
import pytest


@pytest.fixture
def poisson_block() -> tuple[np.ndarray, np.ndarray, float]:
    """Masked Poisson counts with known rate 50.0."""
    rng = np.random.default_rng(42)
    rate = 50.0
    n_spatial, n_energy = 40, 12
    counts = rng.poisson(rate, size=(n_spatial, n_energy)).astype(np.float64)
    mask = np.zeros(n_spatial, dtype=bool)
    mask[5:35] = True
    return counts, mask, rate
