import threading
from pathlib import Path

import numpy as np
import pandas as pd

from stxm.estimators import WeightingMode
from stxm.reduction import RegionSpectrum
from stxm.store import (
    Provenance,
    import_legacy_parquet,
    list_manifest,
    provenance_from_hdr,
    query_spectra,
    write_spectrum,
)


def _spectrum() -> RegionSpectrum:
    energy = np.linspace(280.0, 285.0, 6)
    return RegionSpectrum(
        energy_eV=energy,
        OD=energy * 0.01,
        OD_err=np.full_like(energy, 0.001),
        region_label="pure",
        weighting_mode=WeightingMode.POISSON_MLE.value,
        reduction_method="two_region",
        n_pixels=10,
    )


def _provenance(tmp_path: Path) -> Provenance:
    hdr = tmp_path / "scan.hdr"
    xim = tmp_path / "scan_a.xim"
    hdr.write_text('PAxis = { Name = "Energy" Points = ( 6 , 280 281 282 283 284 285 ) }')
    xim.write_text("1\n" * 6)
    return provenance_from_hdr(
        hdr,
        sample_name="test_sample",
        region_label="pure",
        edge="C_K",
        sample_bounds={"sample_lo": 0.0, "sample_hi": 1.0, "izero_lo": 0.0, "izero_hi": 0.1},
        pre_edge=(275.0, 278.0),
        post_edge=(290.0, 292.0),
        weighting_mode=WeightingMode.POISSON_MLE,
        reduction_method="two_region",
        xim_path=xim,
    )


def test_write_and_query_round_trip(tmp_path):
    store = tmp_path / "store"
    prov = _provenance(tmp_path)
    path = write_spectrum(store, _spectrum(), prov)
    assert path.exists()
    table = query_spectra(store, sample="test_sample")
    assert len(table) == 6
    manifest = list_manifest(store)
    assert len(manifest) == 1


def test_concurrent_writes(tmp_path):
    store = tmp_path / "store"
    prov = _provenance(tmp_path)
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            spec = _spectrum()
            spec.region_label = f"region_{idx}"
            local = Provenance(**{**prov.__dict__, "region_label": spec.region_label})
            write_spectrum(store, spec, local, sample_name=f"sample_{idx}", edge="C_K")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(list_manifest(store)) == 4


def test_sidecar_fields(tmp_path):
    store = tmp_path / "store"
    prov = _provenance(tmp_path)
    parquet_path = write_spectrum(store, _spectrum(), prov)
    sidecar = parquet_path.with_suffix(".json")
    assert sidecar.exists()
    text = sidecar.read_text()
    for token in ("hdr_sha256", "weighting_mode", "reduction_method", "package_version"):
        assert token in text


def test_legacy_import_row_count(tmp_path):
    legacy = tmp_path / "experiment.parquet"
    energy = np.linspace(280.0, 282.0, 3)
    df = pd.DataFrame({
        "energy_eV": energy,
        "OD": energy * 0.01,
        "OD_err": np.full(3, 0.001),
        "I0": np.ones(3),
        "I0_err": np.ones(3),
        "I": np.ones(3),
        "I_err": np.ones(3),
        "n_sample": 5,
        "n_izero": 3,
        "sample_name": "s1",
        "spot_label": "pure",
        "scan_path": str(tmp_path / "a.hdr"),
    })
    df.to_parquet(legacy, index=False)
    store = tmp_path / "store"
    count = import_legacy_parquet(store, legacy, edge="C_K")
    assert count == 1
    assert len(list_manifest(store)) == 1
