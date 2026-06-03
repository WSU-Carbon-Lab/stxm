import numpy as np
from sklearn.mixture import GaussianMixture


def segment_spatial_regions(
    image: np.ndarray,
    n_regions: int = 3,
    profile_columns: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Segment the spatial (row) axis into sample, edge, and izero (and optionally more sample sub-regions).
    Always finds exactly three base regions first: sample, edge, izero in spatial order
    (sample | edge | izero or izero | edge | sample). The edge is the thinnest of the three.
    For n_regions > 3, additional regions are found only inside the sample; edge and izero stay single.

    Parameters
    ----------
    image : np.ndarray
        2D scan (rows = spatial axis, cols = energy).
    n_regions : int
        Total regions: 3 = sample, edge, izero; 4+ = sample split into sample_1, sample_2, ... plus edge, izero.
    profile_columns : int, optional
        Number of trailing columns to average for the profile. If None, uses 20 or all columns.
    random_state : int
        Random state for GMM.

    Returns
    -------
    row_labels : np.ndarray
        Integer label per row. 0..n_regions-3 = sample parts, n_regions-2 = edge, n_regions-1 = izero.
    label_names : list of str
        Names: sample_1, sample_2, ..., edge, izero.
    """
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    if profile_columns is None:
        profile_columns = min(20, n_cols)
    profile = np.mean(image[:, -profile_columns:], axis=1, dtype=np.float64).reshape(-1, 1)
    row_labels = np.zeros(n_rows, dtype=int)
    gm = GaussianMixture(n_components=3, random_state=random_state).fit(profile)
    raw = gm.predict(profile)
    extents = []
    means_intensity = []
    mean_row = []
    for idx in range(3):
        mask = raw == idx
        extents.append(np.sum(mask))
        means_intensity.append(float(np.mean(image[mask, :])))
        mean_row.append(float(np.mean(np.where(mask)[0])))
    edge_idx = int(np.argmin(extents))
    other = [i for i in range(3) if i != edge_idx]
    left_idx = other[0] if mean_row[other[0]] < mean_row[other[1]] else other[1]
    right_idx = other[1] if mean_row[other[0]] < mean_row[other[1]] else other[0]
    sample_idx = left_idx if means_intensity[left_idx] < means_intensity[right_idx] else right_idx
    izero_idx = right_idx if sample_idx == left_idx else left_idx
    row_labels[raw == sample_idx] = 0
    row_labels[raw == edge_idx] = 1
    row_labels[raw == izero_idx] = 2
    label_names = ["sample", "edge", "izero"]
    if n_regions > 3:
        n_sample_parts = n_regions - 2
        sample_mask = row_labels == 0
        n_sample = int(np.sum(sample_mask))
        if n_sample >= n_sample_parts * 2:
            profile_sample = profile[sample_mask].astype(np.float64)
            gm2 = GaussianMixture(n_components=n_sample_parts, random_state=random_state).fit(profile_sample)
            sub = gm2.predict(profile_sample)
            order = np.argsort(np.asarray(gm2.means_).ravel())
            remap = np.empty(n_sample_parts, dtype=int)
            remap[order] = np.arange(n_sample_parts)
            sub = remap[sub]
            row_labels[sample_mask] = sub
            row_labels[row_labels == 1] = n_regions - 2
            row_labels[row_labels == 2] = n_regions - 1
            label_names = [f"sample_{i+1}" for i in range(n_sample_parts)] + ["edge", "izero"]
    return row_labels, label_names


def bar_bounds_from_three_regions(
    image: np.ndarray,
    qaxis: np.ndarray,
    profile_columns: int | None = None,
    random_state: int = 0,
) -> tuple[float, float, float, float]:
    """
    Set the four dividing bar positions (sample_lo, sample_hi, izero_lo, izero_hi)
    from 3-region segmentation (sample, edge, izero). Uses segment_spatial_regions
    then maps sample and izero row indices to qaxis bounds.

    Parameters
    ----------
    image : np.ndarray
        2D scan (rows = spatial/qaxis axis, cols = energy).
    qaxis : np.ndarray
        Q-axis coordinates, length = image.shape[0].
    profile_columns : int, optional
        Passed to segment_spatial_regions.
    random_state : int
        Passed to segment_spatial_regions.

    Returns
    -------
    bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi : float
        Axis values for the four horizontal dividing bars.
    """
    qaxis = np.asarray(qaxis)
    n_rows = image.shape[0]
    if n_rows < 3 or len(qaxis) != n_rows:
        y_min, y_max = float(np.min(qaxis)), float(np.max(qaxis))
        span = y_max - y_min
        margin = span * 0.05
        return (
            y_min + span * 0.45,
            y_max - margin,
            y_min + margin,
            y_min + span * 0.35,
        )
    row_labels, _ = segment_spatial_regions(
        image,
        n_regions=3,
        profile_columns=profile_columns,
        random_state=random_state,
    )
    sample_rows = np.where(row_labels == 0)[0]
    izero_rows = np.where(row_labels == 2)[0]
    if sample_rows.size == 0 or izero_rows.size == 0:
        return auto_sample_izero_regions(image, qaxis)
    q_sample = qaxis[sample_rows]
    q_izero = qaxis[izero_rows]
    bar_sample_lo = float(np.min(q_sample))
    bar_sample_hi = float(np.max(q_sample))
    bar_izero_lo = float(np.min(q_izero))
    bar_izero_hi = float(np.max(q_izero))
    span = float(np.max(qaxis) - np.min(qaxis))
    margin = max(span * 0.01, 1e-9)
    if abs(bar_sample_hi - bar_sample_lo) < margin:
        lo, hi = min(bar_sample_lo, bar_sample_hi), max(bar_sample_lo, bar_sample_hi)
        bar_sample_lo = lo - margin
        bar_sample_hi = hi + margin
    if abs(bar_izero_hi - bar_izero_lo) < margin:
        lo, hi = min(bar_izero_lo, bar_izero_hi), max(bar_izero_lo, bar_izero_hi)
        bar_izero_lo = lo - margin
        bar_izero_hi = hi + margin
    return (bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi)


def sample_izero_masks(
    qaxis_points: np.ndarray,
    sample_lo: float,
    sample_hi: float,
    izero_lo: float,
    izero_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boolean masks for sample and izero regions from axis coordinates.

    Parameters
    ----------
    qaxis_points : np.ndarray
        Q-axis (sample axis) coordinates, one per row.
    sample_lo, sample_hi : float
        Sample region bounds (inclusive).
    izero_lo, izero_hi : float
        Izero region bounds (inclusive).

    Returns
    -------
    sample_mask : np.ndarray
        True where qaxis_points is in [sample_lo, sample_hi].
    izero_mask : np.ndarray
        True where qaxis_points is in [izero_lo, izero_hi].
    """
    sample = (qaxis_points >= sample_lo) & (qaxis_points <= sample_hi)
    izero = (qaxis_points >= izero_lo) & (qaxis_points <= izero_hi)
    return sample, izero


def auto_sample_izero_regions(
    image: np.ndarray,
    qaxis: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Infer sample and izero bar positions from image profile (sample_lo, sample_hi, izero_lo, izero_hi).

    Parameters
    ----------
    image : np.ndarray
        2D scan (rows = qaxis).
    qaxis : np.ndarray
        Q-axis coordinates, length = image.shape[0].

    Returns
    -------
    bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi : float
        Axis values for the four horizontal bars.
    """
    profile = np.asarray(image[:, -1], dtype=float)
    n = len(profile)
    qaxis = np.asarray(qaxis)
    if n < 4 or len(qaxis) != n:
        y_min, y_max = float(np.min(qaxis)), float(np.max(qaxis))
        span = y_max - y_min
        margin = span * 0.05
        return (
            y_min + span * 0.45,
            y_max - margin,
            y_min + margin,
            y_min + span * 0.35,
        )
    kernel = np.ones(5) / 5.0
    profile = np.convolve(profile, kernel, mode="same")
    grad = np.diff(profile)
    cliff_idx = int(np.argmax(np.abs(grad)))
    cliff_idx = np.clip(cliff_idx, 0, n - 2)
    left_win = slice(max(0, cliff_idx - 2), cliff_idx + 1)
    right_win = slice(cliff_idx + 1, min(n, cliff_idx + 4))
    left_mean = float(np.mean(profile[left_win]))
    right_mean = float(np.mean(profile[right_win]))
    izero_on_left = left_mean < right_mean
    buffer_pixels = max(1, int(n * 0.05))
    min_region_pixels = max(2, int(n * 0.08))
    if izero_on_left:
        izero_end = cliff_idx - buffer_pixels
        sample_start = cliff_idx + 1 + buffer_pixels
        izero_end = np.clip(izero_end, min_region_pixels - 1, n - 2)
        sample_start = np.clip(sample_start, 1, n - min_region_pixels)
        if izero_end < sample_start:
            bar_izero_lo = float(qaxis[0])
            bar_izero_hi = float(qaxis[izero_end])
            bar_sample_lo = float(qaxis[sample_start])
            bar_sample_hi = float(qaxis[n - 1])
        else:
            mid = n // 2
            bar_sample_lo = float(qaxis[0])
            bar_sample_hi = float(qaxis[max(0, mid - min_region_pixels)])
            bar_izero_lo = float(qaxis[min(n - 1, mid + min_region_pixels)])
            bar_izero_hi = float(qaxis[n - 1])
    else:
        sample_end = cliff_idx - buffer_pixels
        izero_start = cliff_idx + 1 + buffer_pixels
        sample_end = np.clip(sample_end, min_region_pixels - 1, n - 2)
        izero_start = np.clip(izero_start, 1, n - min_region_pixels)
        if sample_end < izero_start:
            bar_sample_lo = float(qaxis[0])
            bar_sample_hi = float(qaxis[sample_end])
            bar_izero_lo = float(qaxis[izero_start])
            bar_izero_hi = float(qaxis[n - 1])
        else:
            mid = n // 2
            bar_izero_lo = float(qaxis[0])
            bar_izero_hi = float(qaxis[max(0, mid - min_region_pixels)])
            bar_sample_lo = float(qaxis[min(n - 1, mid + min_region_pixels)])
            bar_sample_hi = float(qaxis[n - 1])
    span = float(np.max(qaxis) - np.min(qaxis))
    margin = span * 0.02
    if abs(bar_sample_hi - bar_sample_lo) < margin:
        lo, hi = min(bar_sample_lo, bar_sample_hi), max(bar_sample_lo, bar_sample_hi)
        bar_sample_lo = lo - margin
        bar_sample_hi = hi + margin
    if abs(bar_izero_hi - bar_izero_lo) < margin:
        lo, hi = min(bar_izero_lo, bar_izero_hi), max(bar_izero_lo, bar_izero_hi)
        bar_izero_lo = lo - margin
        bar_izero_hi = hi + margin
    return (bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi)
