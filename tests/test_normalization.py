import numpy as np

from stxm.normalization import (
    NormalizationMode,
    normalize_nexafs,
    normalize_nexafs_with_metadata,
    pre_edge_subtract,
)


def test_pre_edge_subtract_zeros_pre_region():
    energy = np.linspace(270.0, 330.0, 61)
    od = 0.2 + 0.01 * energy
    out = pre_edge_subtract(energy, od, 275.0, 285.0)
    mask = (energy >= 275.0) & (energy <= 285.0)
    assert np.allclose(np.mean(out[mask]), 0.0, atol=1e-10)


def test_normalize_nexafs_post_edge_mean():
    energy = np.linspace(270.0, 330.0, 121)
    od = np.where(energy < 290.0, 0.1, 2.0)
    out = normalize_nexafs(energy, od, 275.0, 285.0, 320.0, 330.0)
    post = (energy >= 320.0) & (energy <= 330.0)
    assert np.allclose(np.mean(out[post]), 1.0, atol=1e-6)


def test_scale_shift_metadata_records_shift():
    energy = np.linspace(270.0, 330.0, 121)
    od = np.where(energy < 290.0, 0.05, 1.8)
    normed, meta = normalize_nexafs_with_metadata(
        energy,
        od,
        275.0,
        285.0,
        320.0,
        330.0,
        mode=NormalizationMode.SCALE_SHIFT,
    )
    post = (energy >= 320.0) & (energy <= 330.0)
    mean_post = float(np.nanmean(normed[post]))
    assert np.isfinite(mean_post)
    assert abs(mean_post - 1.0) < 0.08
    assert meta["normalization_mode"] == NormalizationMode.SCALE_SHIFT.value
    assert isinstance(meta["energy_shift_eV"], float)
