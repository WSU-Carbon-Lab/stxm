import numpy as np
import pytest

from stxm.lcf import Spectrum, fit_lcf, preview_lcf_model


def _refs_and_target():
    energy = np.linspace(280.0, 290.0, 50)
    ref_a = Spectrum(energy, np.sin(energy * 0.4), np.full_like(energy, 0.02), label="a")
    ref_b = Spectrum(energy, np.cos(energy * 0.35), np.full_like(energy, 0.02), label="b")
    true_frac = np.array([0.35, 0.65])
    od = true_frac[0] * ref_a.OD + true_frac[1] * ref_b.OD
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.02, size=energy.size)
    target = Spectrum(energy, od + noise, np.full_like(energy, 0.02), label="target")
    return target, [ref_a, ref_b], true_frac


def test_lcf_recovers_fractions():
    target, refs, true_frac = _refs_and_target()
    result = fit_lcf(target, refs, non_negative=True, sum_to_one=True)
    np.testing.assert_allclose(result.fractions, true_frac, atol=0.08)
    assert np.all(result.fractions >= -1e-8)
    assert abs(float(np.sum(result.fractions)) - 1.0) < 1e-6


def test_non_negative_without_sum_constraint():
    target, refs, _ = _refs_and_target()
    result = fit_lcf(target, refs, non_negative=True, sum_to_one=False)
    assert np.all(result.fractions >= -1e-8)


def test_reduced_chi_square_near_one_for_correct_weights():
    target, refs, _ = _refs_and_target()
    result = fit_lcf(target, refs, non_negative=True, sum_to_one=True)
    assert 0.2 < result.reduced_chi_square < 5.0


def test_initial_fractions_used_as_starting_point():
    target, refs, true_frac = _refs_and_target()
    guess = np.array([0.5, 0.5])
    result = fit_lcf(
        target,
        refs,
        non_negative=True,
        sum_to_one=True,
        initial_fractions=guess,
    )
    np.testing.assert_allclose(result.fractions, true_frac, atol=0.08)


def test_per_component_bounds_respected():
    target, refs, _ = _refs_and_target()
    result = fit_lcf(
        target,
        refs,
        non_negative=True,
        sum_to_one=True,
        fraction_bounds=[(0.0, 0.4), (0.6, 1.0)],
        initial_fractions=np.array([0.2, 0.8]),
    )
    assert result.fractions[0] <= 0.4 + 1e-6
    assert result.fractions[1] >= 0.6 - 1e-6
    assert abs(float(np.sum(result.fractions)) - 1.0) < 1e-6


def test_fixed_component_holds_initial_value():
    target, refs, _ = _refs_and_target()
    fixed_frac = 0.25
    result = fit_lcf(
        target,
        refs,
        non_negative=True,
        sum_to_one=True,
        initial_fractions=np.array([fixed_frac, 0.75]),
        fixed=[True, False],
    )
    assert abs(result.fractions[0] - fixed_frac) < 1e-9
    assert abs(float(np.sum(result.fractions)) - 1.0) < 1e-6


def test_all_fixed_skips_optimization():
    target, refs, true_frac = _refs_and_target()
    result = fit_lcf(
        target,
        refs,
        sum_to_one=True,
        initial_fractions=true_frac,
        fixed=[True, True],
    )
    np.testing.assert_allclose(result.fractions, true_frac)


def test_fixed_initial_outside_bounds_raises():
    target, refs, _ = _refs_and_target()
    with pytest.raises(ValueError, match="below bound"):
        fit_lcf(
            target,
            refs,
            initial_fractions=np.array([0.05, 0.95]),
            fraction_bounds=[(0.1, 0.5), (0.5, 1.0)],
            fixed=[True, False],
            sum_to_one=True,
        )


def test_fixed_sum_exceeds_one_raises():
    target, refs, _ = _refs_and_target()
    with pytest.raises(ValueError, match="fixed fractions sum exceeds"):
        fit_lcf(
            target,
            refs,
            sum_to_one=True,
            initial_fractions=np.array([0.6, 0.5]),
            fixed=[True, True],
        )


def test_preview_lcf_model_normalizes_fractions():
    target, refs, true_frac = _refs_and_target()
    grid, model, y_t = preview_lcf_model(
        target,
        refs,
        np.array([35.0, 65.0]),
        normalize_fractions=True,
    )
    expected = true_frac[0] * refs[0].OD + true_frac[1] * refs[1].OD
    np.testing.assert_allclose(model, expected, atol=1e-10)
    np.testing.assert_allclose(y_t, target.OD, atol=1e-10)
    assert grid.size == target.OD.size


def test_preview_lcf_model_length_mismatch_raises():
    target, refs, _ = _refs_and_target()
    with pytest.raises(ValueError, match="fractions length"):
        preview_lcf_model(target, refs, np.array([0.5]))
