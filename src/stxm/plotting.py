"""Matplotlib styling helpers for interactive STXM widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend import Legend

LINE_SCAN_DISPLAY_CMAP = "gray"


def use_science_style() -> bool:
    """
    Apply SciencePlots style sheets when the package is installed.

    Returns
    -------
    bool
        True when a science style was activated, False when falling back to defaults.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
        return True
    except ImportError:
        return False


def style_axes(ax: Axes) -> None:
    """
    Apply publication-oriented tick styling to one axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for tick and minor-tick configuration.
    """
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    ax.minorticks_on()


def image_display_limits(
    image: np.ndarray,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
    positive_only: bool = False,
) -> tuple[float, float]:
    """
    Compute robust display limits for a 2D intensity image.

    Parameters
    ----------
    image : np.ndarray
        Two-dimensional intensity array; non-finite values are ignored.
    p_low, p_high : float
        Lower and upper percentiles in ``[0, 100]`` used for ``vmin`` and ``vmax``.
    positive_only : bool
        When True and enough strictly positive finite pixels exist, percentiles use only
        those values so transmission maps (bright izero, darker sample) are not crushed
        by zeros or invalid pixels.

    Returns
    -------
    vmin, vmax : float
        Display limits; ``vmax`` is strictly greater than ``vmin`` when finite data exist.
    """
    data = np.asarray(image, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if positive_only:
        positive = finite[finite > 0]
        if positive.size >= max(16, finite.size // 10):
            finite = positive
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(finite))
    if not np.isfinite(hi):
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1.0)
    return lo, hi


def spectrum_legend_label(
    sample_name: str,
    spot_label: str,
    scan_path: str | None = None,
) -> str:
    """
    Build a distinct legend label for one spectrum trace.

    Parameters
    ----------
    sample_name : str
        Sample identifier shown as the legend prefix.
    spot_label : str
        Region or spot label; empty values become ``pure``.
    scan_path : str, optional
        Scan file path or basename; when set, the path stem is appended in parentheses.

    Returns
    -------
    str
        Label of the form ``sample:spot`` or ``sample:spot (scan_stem)``.
    """
    sample = (sample_name or "").strip() or "?"
    spot = (spot_label or "pure").strip() or "pure"
    base = f"{sample}:{spot}"
    if scan_path and str(scan_path).strip():
        stem = Path(str(scan_path).strip()).stem
        if stem:
            return f"{base} ({stem})"
    return base


def make_draggable_legend(ax: Axes, **legend_kwargs: Any) -> Legend | None:
    """
    Create a legend on ``ax`` and make it draggable in interactive backends.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already has labeled artists.
    **legend_kwargs
        Forwarded to ``Axes.legend``.

    Returns
    -------
    matplotlib.legend.Legend or None
        Legend instance when handles exist; otherwise None.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    defaults = {"loc": "best", "fontsize": 8, "framealpha": 0.92}
    defaults.update(legend_kwargs)
    leg = ax.legend(handles, labels, **defaults)
    leg.set_draggable(True)
    return leg


def apply_image_clim(
    artist: Any,
    image: np.ndarray,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
    positive_only: bool = False,
) -> tuple[float, float]:
    """
    Set color limits on an image artist from percentile limits.

    Parameters
    ----------
    artist
        Matplotlib image artist supporting ``set_clim``.
    image : np.ndarray
        Source array passed to ``image_display_limits``.
    p_low, p_high : float
        Percentile window forwarded to ``image_display_limits``.
    positive_only : bool
        Forwarded to ``image_display_limits`` for raw transmission line-scan maps.

    Returns
    -------
    vmin, vmax : float
        Limits applied to the artist.
    """
    vmin, vmax = image_display_limits(
        image, p_low=p_low, p_high=p_high, positive_only=positive_only
    )
    artist.set_clim(vmin, vmax)
    return vmin, vmax


def apply_line_scan_image_clim(
    artist: Any,
    image: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[float, float]:
    """
    Set grayscale display limits for a raw STXM line-scan intensity map.

    Uses ``LINE_SCAN_DISPLAY_CMAP`` conventions: high transmission (izero) maps to
    bright tones via percentile limits on positive finite detector counts.

    Parameters
    ----------
    artist
        Matplotlib image artist supporting ``set_clim``.
    image : np.ndarray
        Raw oriented intensity array from ``load_stxm``.
    p_low, p_high : float
        Percentile window on positive finite pixels.

    Returns
    -------
    vmin, vmax : float
        Limits applied to the artist.
    """
    return apply_image_clim(
        artist, image, p_low=p_low, p_high=p_high, positive_only=True
    )


def autoscale_y_with_margin(ax: Axes, *, margin: float = 0.08) -> None:
    """
    Relimit, autoscale the y-axis, and add fractional padding so traces do not touch the frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    margin : float
        Fractional y margin forwarded to ``Axes.margins``.
    """
    ax.relim()
    ax.autoscale(axis="y")
    ax.margins(y=margin)


def set_ylim_from_data_with_margin(
    ax: Axes,
    y_min: float,
    y_max: float,
    *,
    margin: float = 0.08,
) -> None:
    """
    Set y limits from finite data bounds and apply the same fractional margin as autoscale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    y_min, y_max : float
        Data extrema; no-op when non-finite or degenerate.
    margin : float
        Fractional padding applied after ``set_ylim``.
    """
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
        return
    ax.set_ylim(float(y_min), float(y_max))
    ax.margins(y=margin)
