from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from stxm.io import (
    extract_scan_energy,
    is_nexafs_line_scan,
    list_experiment_hdr_files,
    parse_hdr_scan_type,
    read_hdr,
    scan_type_category,
    thumbnail_png_base64,
)


def _write_line_scan(experiment: Path, stem: str, *, scan_type: str) -> None:
    n_energy = 5
    n_spatial = 8
    energy = np.linspace(280.0, 284.0, n_energy)
    spatial = np.linspace(0.0, 7.0, n_spatial)
    hdr = experiment / f"{stem}.hdr"
    xim = experiment / f"{stem}_a.xim"
    hdr.write_text(
        "\n".join(
            [
                f'Type = "{scan_type}"',
                'PAxis = { Name = "Energy (eV)" '
                f"Points = ( {n_energy} , {' '.join(f'{value:.1f}' for value in energy)} ) }}",
                'QAxis = { Name = "Sample" '
                f"Points = ( {n_spatial} , {' '.join(f'{value:.1f}' for value in spatial)} ) }}",
            ]
        )
    )
    image = np.arange(n_spatial * n_energy, dtype=np.float64).reshape(n_spatial, n_energy) + 1.0
    xim.write_text("\n".join(" ".join(f"{value:.6f}" for value in row) for row in image))


def _write_image_scan(experiment: Path, stem: str, *, energy: float = 284.2) -> None:
    n_x = 50
    n_y = 50
    x_axis = np.linspace(0.0, 49.0, n_x)
    y_axis = np.linspace(0.0, 49.0, n_y)
    hdr = experiment / f"{stem}.hdr"
    xim = experiment / f"{stem}_a.xim"
    hdr.write_text(
        "\n".join(
            [
                'Type = "Image Scan"',
                f"Energy = {energy:.1f}",
                'PAxis = { Name = "X (um)" '
                f"Points = ( {n_x} , {' '.join(f'{value:.1f}' for value in x_axis)} ) }}",
                'QAxis = { Name = "Y (um)" '
                f"Points = ( {n_y} , {' '.join(f'{value:.1f}' for value in y_axis)} ) }}",
            ]
        )
    )
    image = np.arange(n_y * n_x, dtype=np.float64).reshape(n_y, n_x) + 1.0
    xim.write_text("\n".join(" ".join(f"{value:.6f}" for value in row) for row in image))


def test_extract_scan_energy_from_energy_axis(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    experiment.mkdir()
    _write_line_scan(experiment, "line", scan_type="NEXAFS Line Scan")
    meta = read_hdr(experiment / "line.hdr")
    assert meta["energy_min_eV"] == pytest.approx(280.0)
    assert meta["energy_max_eV"] == pytest.approx(284.0)
    assert meta["num_energy_points"] == 5
    assert meta["energy_eV"] is None


def test_extract_scan_energy_single_scalar(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    experiment.mkdir()
    _write_image_scan(experiment, "img", energy=285.4)
    meta = read_hdr(experiment / "img.hdr")
    assert meta["energy_eV"] == pytest.approx(285.4)
    assert meta["energy_min_eV"] == pytest.approx(285.4)
    assert meta["energy_max_eV"] == pytest.approx(285.4)
    assert meta["num_energy_points"] == 1


def test_extract_scan_energy_start_end_range(tmp_path: Path) -> None:
    raw = "\n".join(
        [
            'Type = "Stack"',
            "StartEnergy = 278.0",
            "EndEnergy = 310.0",
            "NumEnergy = 45",
            'PAxis = { Name = "X" Points = ( 2 , 0 1 ) }',
            'QAxis = { Name = "Y" Points = ( 2 , 0 1 ) }',
        ]
    )
    meta = extract_scan_energy({"raw": raw})
    assert meta["energy_min_eV"] == pytest.approx(278.0)
    assert meta["energy_max_eV"] == pytest.approx(310.0)
    assert meta["num_energy_points"] == 45
    assert meta["energy_eV"] is None


def test_parse_hdr_scan_type_and_category(tmp_path: Path) -> None:
    experiment = tmp_path / "2024-01(Jan)"
    experiment.mkdir()
    _write_line_scan(experiment, "line_a", scan_type="NEXAFS Line Scan")
    _write_line_scan(experiment, "image_a", scan_type="Image Scan")
    line_hdr = experiment / "line_a.hdr"
    image_hdr = experiment / "image_a.hdr"
    assert parse_hdr_scan_type(line_hdr) == "NEXAFS Line Scan"
    assert scan_type_category("NEXAFS Line Scan") == "line_scan"
    assert scan_type_category("Image Scan") == "image_scan"
    assert scan_type_category("Unknown") == "other"
    assert is_nexafs_line_scan(line_hdr)
    assert not is_nexafs_line_scan(image_hdr)
    assert len(list_experiment_hdr_files(experiment)) == 2


def test_thumbnail_png_base64_non_empty(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    experiment.mkdir()
    _write_line_scan(experiment, "scan", scan_type="Image Scan")
    from stxm.io import load_stxm

    _, image = load_stxm(experiment / "scan.hdr")
    encoded = thumbnail_png_base64(image, max_size=64)
    assert len(encoded) > 100


def test_catalog_experiment_bridge_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = tmp_path / "2024-02(Feb)"
    experiment.mkdir()
    _write_line_scan(experiment, "nexafs", scan_type="NEXAFS Line Scan")
    _write_line_scan(experiment, "focus", scan_type="Focus Scan")
    monkeypatch.setenv("STXM_ALLOWED_ROOTS", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stxm.bridge",
            "--allowed-root",
            str(tmp_path),
            "catalog-experiment",
            str(experiment),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    basenames = {entry["basename"] for entry in payload["entries"]}
    assert basenames == {"nexafs.hdr", "focus.hdr"}
    line_entry = next(entry for entry in payload["entries"] if entry["basename"] == "nexafs.hdr")
    focus_entry = next(entry for entry in payload["entries"] if entry["basename"] == "focus.hdr")
    assert line_entry["is_nexafs_line_scan"] is True
    assert line_entry["category"] == "line_scan"
    assert focus_entry["category"] == "focus_scan"
    assert "thumbnail_png_base64" in line_entry
    assert "thumbnail_png_base64" in focus_entry
    assert line_entry["energy_min_eV"] == pytest.approx(280.0)
    assert line_entry["energy_max_eV"] == pytest.approx(284.0)
    assert line_entry["num_energy_points"] == 5


def test_list_experiment_hdr_files_in_subdirectory(tmp_path: Path) -> None:
    experiment = tmp_path / "beamtime"
    subdir = experiment / "2025_10(October)"
    subdir.mkdir(parents=True)
    _write_line_scan(subdir, "532_260313061", scan_type="NEXAFS Line Scan")
    paths = list_experiment_hdr_files(experiment)
    assert len(paths) == 1
    expected = (subdir / "532_260313061.hdr").resolve()
    assert paths[0] == expected
    assert paths[0].exists()


def test_catalog_experiment_subdirectory_hdr_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = tmp_path / "BL5321"
    subdir = experiment / "2025_10(October)"
    subdir.mkdir(parents=True)
    _write_line_scan(subdir, "532_260313061", scan_type="NEXAFS Line Scan")
    monkeypatch.setenv("STXM_ALLOWED_ROOTS", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stxm.bridge",
            "--allowed-root",
            str(tmp_path),
            "catalog-experiment",
            str(experiment),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(result.stdout)
    entry = next(
        item for item in payload["entries"] if item["basename"] == "532_260313061.hdr"
    )
    assert entry["hdr_path"] == str((subdir / "532_260313061.hdr").resolve())
    assert entry["is_nexafs_line_scan"] is True


def test_load_scan_missing_hdr_returns_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.hdr"
    monkeypatch.setenv("STXM_ALLOWED_ROOTS", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stxm.bridge",
            "--allowed-root",
            str(tmp_path),
            "load-scan",
            str(missing),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "not found" in payload["error"].lower()


def test_catalog_experiment_image_scan_energy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = tmp_path / "2024-03(Mar)"
    experiment.mkdir()
    _write_image_scan(experiment, "img01", energy=287.6)
    monkeypatch.setenv("STXM_ALLOWED_ROOTS", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stxm.bridge",
            "--allowed-root",
            str(tmp_path),
            "catalog-experiment",
            str(experiment),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(result.stdout)
    image_entry = next(entry for entry in payload["entries"] if entry["basename"] == "img01.hdr")
    assert image_entry["category"] == "image_scan"
    assert image_entry["energy_eV"] == pytest.approx(287.6)
    assert image_entry["energy_min_eV"] == pytest.approx(287.6)
    assert image_entry["energy_max_eV"] == pytest.approx(287.6)
