"""Filesystem roots permitted for stxm-bridge path arguments."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _macos_cloud_storage_roots(home: Path) -> list[str]:
    cloud_storage = home / "Library" / "CloudStorage"
    if not cloud_storage.is_dir():
        return []
    roots = [str(cloud_storage)]
    for entry in cloud_storage.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("OneDrive-") or name.startswith("OneDrive@"):
            roots.append(str(entry))
    return roots


def _dedupe_roots(roots: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for root in roots:
        resolved = str(Path(root).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def default_allowed_roots() -> list[str]:
    """Return development-friendly path roots when STXM_ALLOWED_ROOTS is unset.

    Uses the user home directory and, on macOS, ``~/Library/CloudStorage`` plus
    detected OneDrive library folders (``OneDrive-*`` and ``OneDrive@*``).
    """
    home = Path.home()
    roots = [str(home)]
    if sys.platform == "darwin":
        roots.extend(_macos_cloud_storage_roots(home))
    return _dedupe_roots(roots)


def allowed_roots_from_env() -> list[str]:
    """Parse ``STXM_ALLOWED_ROOTS`` or return :func:`default_allowed_roots`."""
    env = os.environ.get("STXM_ALLOWED_ROOTS", "")
    configured = [part.strip() for part in env.split(":") if part.strip()]
    if configured:
        return configured
    return default_allowed_roots()


def resolve_path_under_roots(path: str, allowed_roots: list[str] | None) -> Path:
    """Resolve ``path`` and verify it lies under at least one allowed root.

    Parameters
    ----------
    path
        Filesystem path, optionally using a leading ``~``.
    allowed_roots
        Permitted root directories. When ``None`` or empty, any resolved path is
        accepted.

    Returns
    -------
    Path
        Resolved absolute path.

    Raises
    ------
    ValueError
        When ``path`` is outside every allowed root.
    """
    normalized = path.strip()
    resolved = Path(normalized).expanduser().resolve()
    if not allowed_roots:
        return resolved
    for root in allowed_roots:
        root_path = Path(root.strip()).expanduser().resolve()
        if resolved == root_path or root_path in resolved.parents:
            return resolved
    roots_label = ", ".join(allowed_roots)
    raise ValueError(f"Path is outside allowed roots ({roots_label}): {path}")
