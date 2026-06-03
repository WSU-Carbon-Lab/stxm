from __future__ import annotations

from pathlib import Path

import pytest
from stxm.allowed_paths import (
    allowed_roots_from_env,
    default_allowed_roots,
    resolve_path_under_roots,
)


def test_default_allowed_roots_includes_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STXM_ALLOWED_ROOTS", raising=False)
    roots = default_allowed_roots()
    home = str(Path.home().resolve())
    assert home in [str(Path(root).expanduser().resolve()) for root in roots]


def test_allowed_roots_from_env_uses_explicit_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "beamtime"
    custom.mkdir()
    monkeypatch.setenv("STXM_ALLOWED_ROOTS", f" {custom} ")
    assert allowed_roots_from_env() == [str(custom)]


def test_allowed_roots_from_env_falls_back_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STXM_ALLOWED_ROOTS", raising=False)
    assert allowed_roots_from_env() == default_allowed_roots()


def test_resolve_path_under_roots_accepts_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "parent"
    nested = root / "experiment" / "scan.hdr"
    nested.parent.mkdir(parents=True)
    nested.touch()
    resolved = resolve_path_under_roots(str(nested), [str(root)])
    assert resolved == nested.resolve()


def test_resolve_path_under_roots_rejects_outside_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    with pytest.raises(ValueError, match="outside allowed roots"):
        resolve_path_under_roots(str(outside), [str(allowed)])


def test_resolve_path_under_roots_trims_whitespace(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "child"
    root.mkdir()
    child.mkdir()
    resolved = resolve_path_under_roots(f"  {child}  ", [str(root)])
    assert resolved == child.resolve()
