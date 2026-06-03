"""Tests for CXRO mass absorption and bare-atom fitting."""

import json

import numpy as np
import pytest

from stxm.absorption import fit_bare_atom_background, mass_absorption_cm2_per_g, od_to_beta


def test_mass_absorption_carbon_positive():
    energy = np.linspace(270.0, 320.0, 12)
    mu = mass_absorption_cm2_per_g("C", energy, None)
    assert mu.shape == energy.shape
    assert np.all(np.isfinite(mu))
    assert np.all(mu > 0)


def test_fit_bare_atom_background_recovers_linear_model():
    energy = np.linspace(270.0, 310.0, 9)
    mu = mass_absorption_cm2_per_g("C", energy, None)
    scale_true = 1.8
    const_true = 0.2
    od = scale_true * mu + const_true
    scale, const, _, _ = fit_bare_atom_background(energy, od, mu, n_low=3, n_high=3)
    assert scale == pytest.approx(scale_true, rel=1e-4)
    assert const == pytest.approx(const_true, rel=1e-4)


def test_od_to_beta_positive_thickness():
    energy = np.array([300.0])
    od = np.array([0.15])
    beta = od_to_beta(energy, od, thickness_cm=1e-4)
    assert beta.shape == (1,)
    assert beta[0] > 0


def test_bridge_mass_absorption_command():
    from stxm.bridge import cmd_mass_absorption

    class Args:
        formula = "C"
        energy_json = json.dumps([280.0, 290.0, 300.0])
        allowed_root = None

    import io
    import sys

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        cmd_mass_absorption(Args())
    finally:
        sys.stdout = old
    payload = json.loads(buf.getvalue().strip().split("\n")[-1])
    assert payload["ok"] is True
    assert len(payload["mu_rho_cm2_per_g"]) == 3
