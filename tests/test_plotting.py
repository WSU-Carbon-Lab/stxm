import matplotlib.pyplot as plt
import numpy as np

from stxm.plotting import (
    LINE_SCAN_DISPLAY_CMAP,
    apply_line_scan_image_clim,
    image_display_limits,
    spectrum_legend_label,
    style_axes,
)


def test_image_display_limits_percentile_span():
    image = np.zeros((10, 10))
    image[0, 0] = 1e6
    image[5, 5] = 50.0
    vmin, vmax = image_display_limits(image, p_low=2.0, p_high=98.0)
    assert vmin < vmax
    assert vmax < 1e6


def test_image_display_limits_positive_only_ignores_zeros():
    image = np.zeros((20, 20))
    image[5:15, 5:15] = 100.0
    image[0, 0] = 1e6
    vmin, vmax = image_display_limits(image, p_low=1.0, p_high=99.0, positive_only=True)
    assert vmin < vmax
    assert vmax < 1e6


def test_line_scan_display_cmap_is_grayscale():
    assert LINE_SCAN_DISPLAY_CMAP == "gray"


def test_apply_line_scan_image_clim_sets_artist_limits():
    fig, ax = plt.subplots()
    image = np.full((8, 8), 50.0)
    image[0, 0] = 200.0
    artist = ax.imshow(image, cmap=LINE_SCAN_DISPLAY_CMAP)
    vmin, vmax = apply_line_scan_image_clim(artist, image)
    assert artist.get_clim() == (vmin, vmax)
    assert vmin < vmax
    plt.close(fig)


def test_spectrum_legend_label_includes_scan_stem():
    label = spectrum_legend_label("Y6", "pure", "/data/exp/scan_001.hdr")
    assert label == "Y6:pure (scan_001)"
    assert spectrum_legend_label("Y6", "pure", "") == "Y6:pure"


def test_style_axes_enables_minor_ticks():
    fig, ax = plt.subplots()
    style_axes(ax)
    assert ax.xaxis.get_minor_locator() is not None
    plt.close(fig)
