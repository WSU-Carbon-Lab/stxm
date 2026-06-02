import json

import numpy as np
import pytest

from stxm.region_store import (
    REGIONS_CONFIG_FILENAME,
    default_regions_from_image,
    load_scan_regions,
    normalize_region_entry,
    normalize_saved_scan,
    regions_config_path,
    save_regions_config,
    save_scan_regions,
    scan_key_from_path,
)


def test_scan_key_from_path_uses_basename():
    assert scan_key_from_path("/data/exp/scan001.hdr") == "scan001.hdr"
    assert scan_key_from_path("nested/scan002.hdr") == "scan002.hdr"


def test_normalize_region_entry_orders_bounds_and_label():
    entry = normalize_region_entry({"sample_lo": 5.0, "sample_hi": 2.0, "spot_label": " edge "})
    assert entry["sample_lo"] == 2.0
    assert entry["sample_hi"] == 5.0
    assert entry["spot_label"] == "edge"


def test_normalize_saved_scan_requires_regions():
    with pytest.raises(ValueError, match="non-empty regions"):
        normalize_saved_scan({"izero_lo": 0.0, "izero_hi": 1.0, "regions": []})


def test_save_and_load_round_trip(tmp_path):
    exp_dir = tmp_path / "2026-06(June)"
    exp_dir.mkdir()
    scan_path = exp_dir / "line_scan.hdr"
    scan_path.write_text("stub")
    payload = {
        "izero_lo": 0.5,
        "izero_hi": 1.5,
        "regions": [
            {"sample_lo": 2.0, "sample_hi": 4.0, "spot_label": "pure"},
            {"sample_lo": 6.0, "sample_hi": 8.0, "spot_label": "matrix"},
        ],
    }
    save_scan_regions(
        exp_dir,
        scan_path,
        izero_lo=payload["izero_lo"],
        izero_hi=payload["izero_hi"],
        regions=payload["regions"],
    )
    cfg_path = regions_config_path(exp_dir)
    assert cfg_path.is_file()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert "line_scan.hdr" in raw["scans"]
    loaded = load_scan_regions(exp_dir, scan_path)
    assert loaded is not None
    assert loaded["izero_lo"] == pytest.approx(0.5)
    assert loaded["izero_hi"] == pytest.approx(1.5)
    assert len(loaded["regions"]) == 2
    assert loaded["regions"][1]["spot_label"] == "matrix"


def test_load_scan_regions_missing_returns_none(tmp_path):
    assert load_scan_regions(tmp_path, "missing.hdr") is None


def test_load_scan_regions_invalid_entry_returns_none(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    save_regions_config(
        exp_dir,
        {
            "version": 1,
            "scans": {"bad.hdr": {"izero_lo": 0.0, "izero_hi": 1.0, "regions": []}},
        },
    )
    assert load_scan_regions(exp_dir, exp_dir / "bad.hdr") is None


def test_default_regions_from_image_shape():
    qaxis = np.linspace(0.0, 10.0, 20)
    image = np.random.default_rng(0).random((20, 8))
    payload = default_regions_from_image(image, qaxis)
    assert "izero_lo" in payload
    assert "izero_hi" in payload
    assert len(payload["regions"]) == 1
    reg = payload["regions"][0]
    assert reg["sample_lo"] <= reg["sample_hi"]
    assert reg["spot_label"] == "pure"


def test_regions_config_filename_constant():
    assert REGIONS_CONFIG_FILENAME == "regions.json"
