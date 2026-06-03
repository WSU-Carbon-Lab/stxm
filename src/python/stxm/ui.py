import json
import threading
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from stxm.absorption import (
    HC_EV_CM,
    fit_bare_atom_background,
    mass_absorption_cm2_per_g,
    od_to_beta,
)
from stxm.estimators import WeightingMode
from stxm.experiment import (
    CHEMICAL_FORMULA_COLUMN,
    SAMPLE_NAME_COLUMN,
    SCAN_PATH_COLUMN,
    SPOT_LABEL_COLUMN,
    append_nexafs_to_experiment,
    load_experiment_parquet,
)
from stxm.grid import interpolate_spectrum
from stxm.io import list_nexafs_line_scans, load_stxm
from stxm.lcf import Spectrum, fit_lcf, preview_lcf_model
from stxm.nexafs import nexafs_beer_lambert
from stxm.normalization import NormalizationMode, normalize_nexafs_with_metadata
from stxm.plotting import (
    LINE_SCAN_DISPLAY_CMAP,
    apply_line_scan_image_clim,
    autoscale_y_with_margin,
    make_draggable_legend,
    set_ylim_from_data_with_margin,
    spectrum_legend_label,
    style_axes,
    use_science_style,
)
from stxm.reduction import RegionSpectrum
from stxm.region_store import (
    default_regions_from_image,
    load_scan_regions,
    save_scan_regions,
)
from stxm.regions import bar_bounds_from_three_regions, sample_izero_masks
from stxm.store import provenance_from_hdr, query_spectra, write_spectrum
from stxm.transforms import compute_display_curve, normalize_spot_label

REGION_COLORS = ["green", "cyan", "orange", "magenta", "lime", "yellow"]

_INGESTION_MAP_FIGSIZE = (2.8, 5.5)
_INGESTION_SPECTRUM_FIGSIZE = (8.0, 3.5)
_INGESTION_LEFT_WIDTH = "260px"
_INGESTION_MAP_CANVAS_MAX_WIDTH = "260px"
_SPECTRUM_FIGSIZE = (10.0, 3.25)
_LCF_FIGSIZE = (10.0, 5.0)
_DASH_HEADER_LABEL = (
    "font-size:10px;font-weight:600;color:#57606a;"
    "text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;display:block"
)
_DASH_REFRESH_BTN_LAYOUT: dict[str, str] = {
    "width": "90px",
    "min_width": "90px",
    "max_width": "90px",
    "flex": "0 0 90px",
    "height": "32px",
}


def _configure_mpl_canvas(
    fig: Any,
    *,
    min_height: str = "280px",
    min_width: str = "0",
) -> None:
    if hasattr(fig.canvas, "header_visible"):
        fig.canvas.header_visible = False
    canvas_layout = getattr(fig.canvas, "layout", None)
    if canvas_layout is not None:
        setattr(canvas_layout, "width", "100%")
        setattr(canvas_layout, "max_width", "100%")
        setattr(canvas_layout, "min_width", min_width)
        setattr(canvas_layout, "min_height", min_height)
        setattr(canvas_layout, "flex", "1 1 auto")
        overflow = getattr(canvas_layout, "overflow", None)
        if overflow is not None:
            setattr(canvas_layout, "overflow", "hidden")


def _panel_column(
    children: tuple[Any, ...] | list[Any],
    *,
    width: str = "240px",
    max_height: str = "520px",
) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        list(children),
        layout=widgets.Layout(
            width=width,
            min_width=width,
            flex=f"0 0 {width}",
            overflow_y="auto",
            max_height=max_height,
            padding="4px 8px 8px 0",
        ),
    )


def _figure_column(canvas: Any) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        [canvas],
        layout=widgets.Layout(
            flex="1 1 auto",
            min_width="0",
            width="100%",
            padding="4px 0 0 4px",
        ),
    )


def _split_row(left: Any, right: Any, *, gap: str = "12px") -> Any:
    import ipywidgets as widgets

    return widgets.HBox(
        [left, right],
        layout=widgets.Layout(
            width="100%",
            align_items="stretch",
            gap=gap,
        ),
    )


def _figure_row(
    control_vbox: Any,
    canvas: Any,
    *,
    gap: str = "12px",
) -> Any:
    return _split_row(control_vbox, _figure_column(canvas), gap=gap)


def _full_width_canvas_row(canvas: Any) -> Any:
    import ipywidgets as widgets

    return widgets.VBox([canvas], layout=widgets.Layout(width="100%"))


def _dashboard_path_row(text_w: Any, btn_w: Any) -> Any:
    import ipywidgets as widgets

    return widgets.HBox(
        [text_w, btn_w],
        layout=widgets.Layout(width="100%", align_items="center", gap="8px"),
    )


def _dashboard_labeled_field(caption: str, control: Any) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        [
            widgets.HTML(value=f"<span style='{_DASH_HEADER_LABEL}'>{caption}</span>"),
            control,
        ],
        layout=widgets.Layout(width="100%"),
    )


def _ingestion_compact_field(
    caption: str,
    control: Any,
    *,
    width: str = "auto",
    flex: str = "0 0 auto",
) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        [
            widgets.HTML(value=f"<span style='{_DASH_HEADER_LABEL}'>{caption}</span>"),
            control,
        ],
        layout=widgets.Layout(
            width=width,
            min_width=width if width != "auto" else "0",
            flex=flex,
        ),
    )


def _ingestion_controls_row(*fields: Any) -> Any:
    import ipywidgets as widgets

    return widgets.HBox(
        list(fields),
        layout=widgets.Layout(
            width="100%",
            flex_flow="row wrap",
            gap="8px",
            align_items="flex-end",
        ),
    )


def _dashboard_column(*children: Any) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        list(children),
        layout=widgets.Layout(
            flex="1 1 50%",
            min_width="0",
            width="auto",
            gap="10px",
        ),
    )


def _dashboard_store_edge_row(store_w: Any, edge_w: Any) -> Any:
    import ipywidgets as widgets

    store_block = _dashboard_labeled_field("Store root", store_w)
    edge_block = _dashboard_labeled_field("Edge", edge_w)
    store_block.layout = widgets.Layout(flex="1 1 auto", min_width="0", width="auto")
    edge_block.layout = widgets.Layout(width="72px", min_width="72px", flex="0 0 72px")
    return widgets.HBox(
        [store_block, edge_block],
        layout=widgets.Layout(width="100%", align_items="flex-start", gap="8px"),
    )


def _dashboard_header(*rows: Any) -> Any:
    import ipywidgets as widgets

    return widgets.VBox(
        list(rows),
        layout=widgets.Layout(width="100%", padding="8px 10px 6px 10px"),
    )


def _apply_image_and_region_lines(
    im_artist,
    ax_im,
    image: np.ndarray,
    meta: dict,
    izero_lo: float,
    izero_hi: float,
    regions: list[dict],
    line_c,
    line_d,
    region_lines: list[tuple],
    region_label_texts: list | None = None,
    izero_label_text=None,
) -> None:
    paxis = meta["paxis_points"]
    qaxis = meta["qaxis_points"]
    extent = [paxis[0], paxis[-1], qaxis[-1], qaxis[0]]
    im_artist.set_data(image)
    im_artist.set_extent(extent)
    apply_line_scan_image_clim(im_artist, image)
    ax_im.set_xlim(paxis[0], paxis[-1])
    ax_im.set_ylim(qaxis[-1], qaxis[0])
    line_c.set_ydata([izero_lo, izero_lo])
    line_d.set_ydata([izero_hi, izero_hi])
    for (r_lo, r_hi), reg in zip(region_lines, regions):
        r_lo.set_ydata([reg["sample_lo"], reg["sample_lo"]])
        r_hi.set_ydata([reg["sample_hi"], reg["sample_hi"]])
    x_center = (paxis[0] + paxis[-1]) / 2
    if izero_label_text is not None:
        izero_label_text.set_position((x_center, (izero_lo + izero_hi) / 2))
    if region_label_texts is not None:
        while len(region_label_texts) > len(regions):
            t = region_label_texts.pop()
            t.remove()
        for i, reg in enumerate(regions):
            label_str = normalize_spot_label(reg.get("spot_label"))
            if i >= len(region_label_texts):
                txt = ax_im.text(
                    x_center, (reg["sample_lo"] + reg["sample_hi"]) / 2,
                    label_str,
                    transform=ax_im.transData, fontsize=9, va="center", ha="center",
                    color=REGION_COLORS[i % len(REGION_COLORS)],
                )
                region_label_texts.append(txt)
            else:
                txt = region_label_texts[i]
                txt.set_position((x_center, (reg["sample_lo"] + reg["sample_hi"]) / 2))
                txt.set_text(label_str)
                txt.set_color(REGION_COLORS[i % len(REGION_COLORS)])


def line_scan_processor(
    parent_directory: str | Path,
    sample_config: dict | None = None,
):
    """
    Tabbed line-scan processor: experiment setup, OD reduction, views and export.
    sample_config: optional map sample_label -> chemical formula (str or null for blends).
    """
    import ipywidgets as widgets
    from IPython.display import display

    use_science_style()

    parent_dir = Path(parent_directory).resolve()
    if parent_dir.is_file():
        parent_dir = parent_dir.parent
    if not parent_dir.is_dir():
        raise ValueError(f"Not a directory: {parent_directory}")

    placeholder_meta = {
        "paxis_points": np.array([0.0, 1.0]),
        "qaxis_points": np.array([0.0, 1.0]),
        "paxis_name": "Energy (eV)",
        "qaxis_name": "Sample",
    }
    placeholder_image = np.zeros((2, 2), dtype=np.float64)
    data = {"meta": placeholder_meta, "image": placeholder_image}
    meta, image = placeholder_meta, placeholder_image

    def truncate_path(s, max_len=56):
        s = str(s)
        return s if len(s) <= max_len else "..." + s[-(max_len - 3) :]

    qaxis = meta["qaxis_points"]
    paxis = meta["paxis_points"]
    bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi = bar_bounds_from_three_regions(
        image, qaxis
    )

    extent = [paxis[0], paxis[-1], qaxis[-1], qaxis[0]]
    plt.ioff()
    fig, ax_im = plt.subplots(1, 1, figsize=_INGESTION_MAP_FIGSIZE, constrained_layout=True)
    im_artist = ax_im.imshow(
        image,
        aspect="auto",
        extent=extent,
        cmap=LINE_SCAN_DISPLAY_CMAP,
        interpolation="nearest",
        origin="upper",
    )
    apply_line_scan_image_clim(im_artist, image)
    style_axes(ax_im)
    ax_im.set_ylabel(meta.get("qaxis_name", "Sample"))
    ax_im.set_xlim(paxis[0], paxis[-1])
    ax_im.set_ylim(qaxis[-1], qaxis[0])

    line_c = ax_im.axhline(bar_izero_lo, color="blue", lw=2, picker=5)
    line_d = ax_im.axhline(bar_izero_hi, color="blue", lw=2, picker=5)
    region_lines: list[tuple] = []
    rl0 = ax_im.axhline(bar_sample_lo, color=REGION_COLORS[0], lw=2, picker=5)
    rl1 = ax_im.axhline(bar_sample_hi, color=REGION_COLORS[0], lw=2, picker=5)
    region_lines.append((rl0, rl1))
    x_center = (paxis[0] + paxis[-1]) / 2
    region_label_texts: list = []
    _tl = ax_im.text(
        x_center,
        (bar_sample_lo + bar_sample_hi) / 2,
        "pure",
        transform=ax_im.transData, fontsize=9, va="center", ha="center", color="green"
    )
    region_label_texts.append(_tl)
    izero_label_text = ax_im.text(
        x_center, (bar_izero_lo + bar_izero_hi) / 2, "izero",
        transform=ax_im.transData, fontsize=9, va="center", ha="center", color="blue"
    )

    last_nexafs = {}
    config_map = dict(sample_config) if sample_config else {}

    def current_chemical_formula():
        val = sample_dropdown.value if sample_dropdown else None
        if not val or val in ("(load config first)", "(no samples)"):
            return None
        v = config_map.get(val)
        return str(v).strip() if v is not None and str(v).strip() else None

    def apply_config_for_experiment(exp_name: str):
        if sample_dropdown is None:
            return
        opts = [k for k in config_map.keys() if k]
        sample_dropdown.options = opts if opts else ["(no samples)"]
        if exp_name and exp_name in config_map:
            sample_dropdown.value = exp_name
        elif opts:
            sample_dropdown.value = opts[0]

    def update_od():
        meta_u = data["meta"]
        image_u = data["image"]
        qaxis_u = meta_u["qaxis_points"]
        paxis_u = meta_u["paxis_points"]
        iz_lo = state["izero_lo"]
        iz_hi = state["izero_hi"]
        _, izero_mask = sample_izero_masks(qaxis_u, 0.0, 0.0, iz_lo, iz_hi)
        if not np.any(izero_mask):
            return
        energy = np.asarray(paxis_u)
        last_nexafs["energy"] = energy
        last_nexafs["izero_lo"] = iz_lo
        last_nexafs["izero_hi"] = iz_hi
        last_nexafs["regions"] = []
        sn = (sample_dropdown.value or "").strip()
        if sn in ("(load config first)", "(no samples)"):
            sn = "sample"
        if not sn:
            sn = "sample"
        for idx, reg in enumerate(state["regions"]):
            sa_lo, sa_hi = reg["sample_lo"], reg["sample_hi"]
            sample_mask, _ = sample_izero_masks(qaxis_u, sa_lo, sa_hi, iz_lo, iz_hi)
            if not np.any(sample_mask):
                continue
            mode = WeightingMode(weighting_dropdown.value)
            od, sigma_od, I0, sigma_I0, I, sigma_I, n_sample, n_izero = nexafs_beer_lambert(
                image_u, sample_mask, izero_mask, mode=mode
            )
            spot_label = normalize_spot_label(reg.get("spot_label"))
            label = f"{sn}:{spot_label}"
            last_nexafs["regions"].append({
                "OD": od,
                "OD_err": sigma_od,
                "I0": I0,
                "I0_err": sigma_I0,
                "I": I,
                "I_err": sigma_I,
                "n_sample": n_sample,
                "n_izero": n_izero,
                "spot_label": spot_label,
                "sample_lo": sa_lo,
                "sample_hi": sa_hi,
            })
        update_views_plot()

    def on_press(event):
        if event.inaxes != ax_im:
            return
        y = event.ydata
        candidates = [
            (abs(y - state["izero_lo"]), ("izero", "lo")),
            (abs(y - state["izero_hi"]), ("izero", "hi")),
        ]
        for i, reg in enumerate(state["regions"]):
            candidates.append((abs(y - reg["sample_lo"]), ("region", i, "lo")))  # type: ignore[arg-type]
            candidates.append((abs(y - reg["sample_hi"]), ("region", i, "hi")))  # type: ignore[arg-type]
        _, state["dragging"] = min(candidates, key=lambda x: x[0])

    def on_motion(event):
        if event.inaxes != ax_im or state["dragging"] is None:
            return
        qaxis_d = data["meta"]["qaxis_points"]
        y_lo, y_hi = float(qaxis_d.min()), float(qaxis_d.max())
        y = np.clip(event.ydata, y_lo, y_hi)
        margin = (y_hi - y_lo) * 0.02
        paxis_d = data["meta"]["paxis_points"]
        x_center = (paxis_d[0] + paxis_d[-1]) / 2
        drag = state["dragging"]
        if drag == ("izero", "lo"):
            state["izero_lo"] = np.clip(y, y_lo, state["izero_hi"] - margin)
            line_c.set_ydata([state["izero_lo"], state["izero_lo"]])
        elif drag == ("izero", "hi"):
            state["izero_hi"] = np.clip(y, state["izero_lo"] + margin, y_hi)
            line_d.set_ydata([state["izero_hi"], state["izero_hi"]])
        if isinstance(drag, tuple) and drag[0] == "izero" and izero_label_text is not None:
            izero_label_text.set_position(
                (x_center, (state["izero_lo"] + state["izero_hi"]) / 2)
            )
        elif isinstance(drag, tuple) and len(drag) == 3 and drag[0] == "region":
            i, which = drag[1], drag[2]
            if i < len(state["regions"]) and i < len(region_lines):
                reg = state["regions"][i]
                l_lo, l_hi = region_lines[i]
                if which == "lo":
                    reg["sample_lo"] = np.clip(y, y_lo, reg["sample_hi"] - margin)
                    l_lo.set_ydata([reg["sample_lo"], reg["sample_lo"]])
                else:
                    reg["sample_hi"] = np.clip(y, reg["sample_lo"] + margin, y_hi)
                    l_hi.set_ydata([reg["sample_hi"], reg["sample_hi"]])
        if isinstance(drag, tuple) and len(drag) == 3 and drag[0] == "region":
            i = drag[1]
            if i < len(state["regions"]) and i < len(region_label_texts):
                r = state["regions"][i]
                region_label_texts[i].set_position(
                    (x_center, (r["sample_lo"] + r["sample_hi"]) / 2)
                )
        fig.canvas.draw_idle()

    def on_release(event):
        if state["dragging"] is None:
            return
        state["dragging"] = None
        update_od()
        schedule_persist_regions()
        fig.canvas.draw_idle()

    state: dict[str, Any] = {
        "dragging": None,
        "dragging_edge": None,
        "izero_lo": bar_izero_lo,
        "izero_hi": bar_izero_hi,
        "regions": [
            {"sample_lo": bar_sample_lo, "sample_hi": bar_sample_hi, "spot_label": "pure"}
        ],
        "pre_lo": 280.0,
        "pre_hi": 285.0,
        "post_lo": 320.0,
        "post_hi": 330.0,
    }
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    parent_dir_holder = [parent_dir]

    def _experiment_sort_key(name: str) -> tuple[int, int, str]:
        base = name.strip()
        if "(" in base:
            base = base.split("(", 1)[0]
        base = base.replace("_", "-")
        parts = base.split("-", 2)
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
        except Exception:
            return (0, 0, name)
        return (year, month, name)

    experiment_names = (
        sorted(
            [d.name for d in parent_dir.iterdir() if d.is_dir()],
            key=_experiment_sort_key,
            reverse=True,
        )
        if parent_dir.is_dir()
        else []
    )
    experiment_dropdown = widgets.Dropdown(
        options=experiment_names if experiment_names else ["(no experiments)"],
        value=experiment_names[0] if experiment_names else None,
        description="",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )
    current_dir = [
        parent_dir_holder[0] / experiment_dropdown.value
        if experiment_names else parent_dir_holder[0]
    ]
    dir_text = widgets.Text(
        value=str(parent_dir_holder[0]),
        placeholder="Parent directory of experiments",
        layout=widgets.Layout(flex="1 1 auto", min_width="0", width="auto"),
    )
    dir_text.tooltip = str(parent_dir_holder[0])
    refresh_btn = widgets.Button(
        description="Refresh",
        tooltip="Refresh experiment and file lists",
        layout=widgets.Layout(**_DASH_REFRESH_BTN_LAYOUT),
    )

    def list_valid_line_scan_files(experiment_path):
        hdrs = list_nexafs_line_scans(experiment_path)
        return [p.name for p in hdrs]

    def refresh_file_list():
        opts = list_valid_line_scan_files(current_dir[0])
        file_dropdown.options = opts if opts else ["(empty)"]
        if opts:
            file_dropdown.value = opts[0]

    def refresh_experiment_list(_=None):
        p = Path(dir_text.value.strip()).resolve()
        if not p.is_dir():
            return
        parent_dir_holder[0] = p
        exp_names = sorted(
            [d.name for d in p.iterdir() if d.is_dir()],
            key=_experiment_sort_key,
            reverse=True,
        )
        experiment_dropdown.options = exp_names if exp_names else ["(no experiments)"]
        experiment_dropdown.value = exp_names[0] if exp_names else None
        if exp_names:
            current_dir[0] = parent_dir_holder[0] / exp_names[0]
        else:
            current_dir[0] = parent_dir_holder[0]
        dir_text.value = str(parent_dir_holder[0])
        dir_text.tooltip = str(parent_dir_holder[0])
        refresh_file_list()
        if exp_names:
            try_load_config_from_experiment_dir()

    def on_experiment_change(change: Any):
        new_val = change.get("new")
        if new_val is None or new_val == "(no experiments)" or new_val == "(empty)":
            return
        current_dir[0] = parent_dir_holder[0] / str(new_val)
        refresh_file_list()
        apply_config_for_experiment(str(new_val))
        if sample_dropdown.value in ("(load config first)", "(no samples)") and config_map and str(new_val) in config_map:
            sample_dropdown.value = str(new_val)
        try_load_config_from_experiment_dir()

    opts_init = list_valid_line_scan_files(current_dir[0])
    file_dropdown = widgets.Dropdown(
        options=opts_init if opts_init else ["(empty)"],
        description="Line scan",
        layout=widgets.Layout(flex="1", min_width="100px"),
        style={"description_width": "70px"},
    )
    if opts_init:
        file_dropdown.value = opts_init[0]

    sample_dropdown = widgets.Dropdown(
        options=["(load config first)"],
        value="(load config first)",
        description="Sample",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "60px"},
    )
    experiment_dropdown.observe(on_experiment_change, names="value")

    parquet_path_text = widgets.Text(
        value="experiment.parquet",
        placeholder="e.g. experiment.parquet",
        layout=widgets.Layout(flex="1 1 auto", min_width="0", width="auto"),
    )
    store_root_text = widgets.Text(
        value="spectrum_store",
        placeholder="e.g. spectrum_store (partitioned append-only)",
        layout=widgets.Layout(flex="1 1 auto", min_width="0", width="auto"),
    )
    edge_text = widgets.Text(
        value="C_K",
        placeholder="e.g. C_K",
        layout=widgets.Layout(width="100%", min_width="0"),
    )
    weighting_dropdown = widgets.Dropdown(
        options=[(m.value, m.value) for m in WeightingMode],
        value=WeightingMode.POISSON_MLE.value,
        description="Weighting",
        layout=widgets.Layout(min_width="180px"),
        style={"description_width": "70px"},
    )
    _ingestion_dropdown_style = {"description_width": "0px"}
    norm_mode_dropdown = widgets.Dropdown(
        options=[
            ("Pre-edge + scale", NormalizationMode.PRE_EDGE_SCALE.value),
            ("Scale + energy shift", NormalizationMode.SCALE_SHIFT.value),
        ],
        value=NormalizationMode.PRE_EDGE_SCALE.value,
        description="",
        layout=widgets.Layout(width="140px", min_width="140px"),
        style=_ingestion_dropdown_style,
    )
    spectrum_basis_dropdown = widgets.Dropdown(
        options=["Raw OD", "Normalized OD"],
        value="Raw OD",
        description="",
        layout=widgets.Layout(width="120px", min_width="120px"),
        style=_ingestion_dropdown_style,
    )
    pre_lo_widget = widgets.FloatText(
        value=state["pre_lo"],
        description="",
        layout=widgets.Layout(width="64px", min_width="64px"),
        style=_ingestion_dropdown_style,
    )
    pre_hi_widget = widgets.FloatText(
        value=state["pre_hi"],
        description="",
        layout=widgets.Layout(width="64px", min_width="64px"),
        style=_ingestion_dropdown_style,
    )
    post_lo_widget = widgets.FloatText(
        value=state["post_lo"],
        description="",
        layout=widgets.Layout(width="64px", min_width="64px"),
        style=_ingestion_dropdown_style,
    )
    post_hi_widget = widgets.FloatText(
        value=state["post_hi"],
        description="",
        layout=widgets.Layout(width="64px", min_width="64px"),
        style=_ingestion_dropdown_style,
    )

    def sync_norm_windows_from_widgets(_: Any = None):
        state["pre_lo"] = float(pre_lo_widget.value)
        state["pre_hi"] = float(pre_hi_widget.value)
        state["post_lo"] = float(post_lo_widget.value)
        state["post_hi"] = float(post_hi_widget.value)
        update_views_plot()

    def spectrum_for_display(reg: dict, energy_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        od = np.asarray(reg["OD"], dtype=np.float64)
        sigma = np.asarray(reg.get("OD_err", np.zeros_like(od)), dtype=np.float64)
        if (spectrum_basis_dropdown.value or "Raw OD").strip() != "Normalized OD":
            return od, sigma
        normed, _ = normalize_nexafs_with_metadata(
            energy_axis,
            od,
            float(state["pre_lo"]),
            float(state["pre_hi"]),
            float(state["post_lo"]),
            float(state["post_hi"]),
            mode=norm_mode_dropdown.value,
        )
        return normed, sigma
    config_json_text = widgets.Text(
        value="samples.json",
        placeholder="e.g. samples.json",
        layout=widgets.Layout(width="100%"),
    )
    load_config_btn = widgets.Button(description="Load config JSON", tooltip="Map sample_label -> formula")

    def _resolve_relative_path(path_str: str, base: Path | None) -> Path:
        pp = Path((path_str or "").strip())
        if not pp.is_absolute() and base is not None:
            return base / pp
        return pp

    def resolve_parquet_path(path_str: str) -> Path:
        base = current_dir[0] if current_dir else None
        return _resolve_relative_path(path_str, base)

    def resolve_config_path(path_str: str) -> Path:
        base = current_dir[0] if current_dir else None
        return _resolve_relative_path(path_str, base)

    def on_load_config(_):
        nonlocal config_map
        path_str = (config_json_text.value or "").strip()
        if not path_str:
            status_label.value = "<span style='font-size:11px;color:#c00'>Set JSON path.</span>"
            return
        path = resolve_config_path(path_str)
        try:
            with path.open() as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "samples" in raw:
                config_map = {str(k): v for k, v in raw["samples"].items()}
            else:
                config_map = {str(k): v for k, v in raw.items()}
            apply_config_for_experiment(experiment_dropdown.value or "")
            status_label.value = "<span style='font-size:11px;color:#066'>Config loaded.</span>"
            if sample_dropdown.options and sample_dropdown.options[0] != "(no samples)":
                sample_dropdown.value = sample_dropdown.options[0]
        except Exception as e:
            status_label.value = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    load_config_btn.on_click(on_load_config)

    new_sample_name = widgets.Text(
        value="",
        placeholder="New sample name",
        layout=widgets.Layout(width="200px"),
    )
    new_sample_formula = widgets.Text(
        value="",
        placeholder="Optional chemical formula",
        layout=widgets.Layout(width="200px"),
    )
    add_sample_btn = widgets.Button(
        description="Add sample",
        tooltip="Add sample to config JSON",
        button_style="info",
    )

    def on_add_sample(_):
        name = (new_sample_name.value or "").strip()
        if not name:
            status_label.value = "<span style='font-size:11px;color:#c00'>Set sample name.</span>"
            return
        formula_val = (new_sample_formula.value or "").strip()
        cfg_path = (config_json_text.value or "").strip()
        if not cfg_path:
            exp_dir = current_dir[0]
            if not isinstance(exp_dir, Path):
                status_label.value = "<span style='font-size:11px;color:#c00'>Set config path or experiment dir.</span>"
                return
            cfg = exp_dir / "samples.json"
            config_json_text.value = "samples.json"
        else:
            cfg = resolve_config_path(cfg_path)
        try:
            if cfg.is_file():
                with cfg.open() as f:
                    raw = json.load(f)
            else:
                raw = {"samples": {}}
            if isinstance(raw, dict) and "samples" in raw and isinstance(raw["samples"], dict):
                samples: dict[str, Any] = raw["samples"]
            else:
                samples = raw if isinstance(raw, dict) else {}
                raw = {"samples": samples}
            samples[name] = formula_val or None
            cfg.parent.mkdir(parents=True, exist_ok=True)
            with cfg.open("w") as f:
                json.dump(raw, f, indent=2)
            config_map[name] = formula_val or None
            apply_config_for_experiment(name)
            sample_dropdown.value = name
            status_label.value = "<span style='font-size:11px;color:#066'>Sample added.</span>"
            new_sample_name.value = ""
            new_sample_formula.value = ""
        except Exception as e:
            status_label.value = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    add_sample_btn.on_click(on_add_sample)

    def try_load_config_from_experiment_dir():
        exp_dir = current_dir[0]
        if not isinstance(exp_dir, Path) or not exp_dir.is_dir():
            return
        cfg_samples = exp_dir / "samples.json"
        cfg_legacy = exp_dir / "config.json"
        if cfg_samples.is_file():
            config_json_text.value = "samples.json"
            on_load_config(None)
        elif cfg_legacy.is_file():
            config_json_text.value = "config.json"
            on_load_config(None)

    export_btn = widgets.Button(description="Export", tooltip="Append current scan to parquet")
    refresh_plots_btn = widgets.Button(description="Refresh plots", tooltip="Redraw if canvas glitches")
    status_label = widgets.HTML(value="<span style='font-size:11px;color:#666'>Select a file.</span>")
    region_spot_widgets: list = []
    region_spot_rows: list = []
    add_region_btn = widgets.Button(
        description="+ Add region",
        tooltip="Add new region / spot",
        button_style="success",
        layout=widgets.Layout(width="100%", height="32px", margin="6px 0 0 0"),
    )

    def sync_spot_label(i: int, change: Any):
        if 0 <= i < len(state["regions"]):
            new_val = (change.get("new") or "").strip() or "pure"
            state["regions"][i]["spot_label"] = new_val
            if i < len(region_label_texts):
                region_label_texts[i].set_text(new_val)
            update_od()
            schedule_persist_regions()

    def make_spot_row(spot_value: str, idx: int):
        spot_w = widgets.Text(
            value=spot_value,
            placeholder="Spot label",
            layout=widgets.Layout(width="180px", min_width="120px"),
        )
        spot_w.observe(lambda c, i=idx: sync_spot_label(i, c), names="value")
        refresh_btn_row = widgets.Button(
            description="\u21bb",
            tooltip="Recompute automatic region boundaries (single region only)",
            button_style="info",
            layout=widgets.Layout(width="36px", min_width="36px"),
        )
        remove_btn = widgets.Button(
            description="\U0001f5d1",
            tooltip="Remove this region",
            button_style="danger",
            layout=widgets.Layout(width="36px", min_width="36px"),
        )
        def on_refresh(btn, i=idx):
            if i != 0 or len(state["regions"]) != 1:
                return
            if not data.get("meta") or data.get("image") is None:
                return
            qaxis_r = data["meta"]["qaxis_points"]
            img_r = data["image"]
            bar_sample_lo, bar_sample_hi, _, _ = bar_bounds_from_three_regions(img_r, qaxis_r)
            state["regions"][0]["sample_lo"] = float(bar_sample_lo)
            state["regions"][0]["sample_hi"] = float(bar_sample_hi)
            if region_lines:
                l_lo, l_hi = region_lines[0]
                l_lo.set_ydata([bar_sample_lo, bar_sample_lo])
                l_hi.set_ydata([bar_sample_hi, bar_sample_hi])
            _apply_image_and_region_lines(
                im_artist,
                ax_im,
                data["image"],
                data["meta"],
                state["izero_lo"],
                state["izero_hi"],
                state["regions"],
                line_c,
                line_d,
                region_lines,
                region_label_texts,
                izero_label_text,
            )
            update_od()
            schedule_persist_regions()
            fig.canvas.draw_idle()
        refresh_btn_row.on_click(on_refresh)
        def on_remove(btn, i=idx):
            if i < 0 or i >= len(state["regions"]) or len(state["regions"]) <= 1:
                return
            state["regions"].pop(i)
            l_lo, l_hi = region_lines.pop(i)
            l_lo.remove()
            l_hi.remove()
            if i < len(region_label_texts):
                region_label_texts.pop(i).remove()
            region_spot_widgets.clear()
            region_spot_rows.clear()
            for j, reg in enumerate(state["regions"]):
                spot_w, row = make_spot_row(reg.get("spot_label") or "pure", j)
                region_spot_widgets.append(spot_w)
                region_spot_rows.append(row)
            region_spot_box.children = tuple(region_spot_rows)
            if data.get("meta") and data.get("image") is not None:
                _apply_image_and_region_lines(
                    im_artist, ax_im, data["image"], data["meta"],
                    state["izero_lo"], state["izero_hi"], state["regions"],
                    line_c, line_d, region_lines, region_label_texts, izero_label_text,
                )
            update_od()
            schedule_persist_regions()
            fig.canvas.draw_idle()
        remove_btn.on_click(on_remove)
        row = widgets.HBox(
            [spot_w, refresh_btn_row, remove_btn],
            layout=widgets.Layout(align_items="center", width="100%"),
        )
        return spot_w, row

    def on_add_region(_):
        qaxis_a = data["meta"]["qaxis_points"]
        y_lo, y_hi = float(qaxis_a.min()), float(qaxis_a.max())
        if not state["regions"]:
            mid = (y_lo + y_hi) / 2
            delta = (y_hi - y_lo) * 0.1
            sample_lo, sample_hi = mid - delta, mid + delta
        else:
            r0 = state["regions"][0]
            sample_lo, sample_hi = r0["sample_lo"], r0["sample_hi"]
        state["regions"].append({
            "sample_lo": sample_lo,
            "sample_hi": sample_hi,
            "spot_label": "pure",
        })
        idx = len(region_lines)
        color = REGION_COLORS[idx % len(REGION_COLORS)]
        rl0 = ax_im.axhline(sample_lo, color=color, lw=2, picker=5)
        rl1 = ax_im.axhline(sample_hi, color=color, lw=2, picker=5)
        region_lines.append((rl0, rl1))
        spot_w, row = make_spot_row("pure", idx)
        region_spot_widgets.append(spot_w)
        region_spot_rows.append(row)
        region_spot_box.children = tuple(region_spot_rows)
        _apply_image_and_region_lines(
            im_artist, ax_im, data["image"], data["meta"],
            state["izero_lo"], state["izero_hi"], state["regions"], line_c, line_d, region_lines,
            region_label_texts, izero_label_text,
        )
        update_od()
        schedule_persist_regions()
        fig.canvas.draw_idle()

    add_region_btn.on_click(on_add_region)
    first_spot, first_row = make_spot_row("pure", 0)
    region_spot_widgets.append(first_spot)
    region_spot_rows.append(first_row)
    region_spot_box = widgets.VBox([first_row], layout=widgets.Layout(width="100%"))

    display_mode = widgets.Dropdown(
        options=[
            "OD",
            "Norm. Mass Abs. (g/cm^2)",
            "Beta",
            "I0 & transmission",
        ],
        value="OD",
        description="",
        layout=widgets.Layout(width="100px", min_width="100px"),
        style=_ingestion_dropdown_style,
    )
    views_region_dropdown = widgets.Dropdown(
        options=["All"],
        value="All",
        description="",
        layout=widgets.Layout(width="100px", min_width="100px"),
        style=_ingestion_dropdown_style,
    )
    mass_abs_fit_mode = widgets.Dropdown(
        options=[
            "Scale only (last 5 pts)",
            "Scale & offset (first+last 5)",
        ],
        value="Scale only (last 5 pts)",
        description="",
        layout=widgets.Layout(width="160px", min_width="160px"),
        style=_ingestion_dropdown_style,
    )
    show_bare_atom = widgets.Checkbox(
        value=True,
        description="Show bare-atom step edge",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    step_edge_hint = widgets.HTML(
        value="",
        layout=widgets.Layout(margin="0 0 0 4px", min_height="20px", flex="1 1 auto"),
    )

    plt.ioff()
    fig_views, ax_views = plt.subplots(
        1, 1, figsize=_INGESTION_SPECTRUM_FIGSIZE, constrained_layout=True
    )
    ax_views.set_xlabel(meta.get("paxis_name", "Energy (eV)"))
    ax_views.grid(True, alpha=0.3)
    ax_views.set_ylabel("OD (ln I0/I)")
    style_axes(ax_views)
    views_line_artists: list = []
    views_fill_artists: list = []
    views_bare_atom_artists: list = []
    views_twin_ax: list = []
    views_edge_artists: list = []
    edge_lines: dict[str, Any] = {}

    def update_views_plot():
        for line in views_line_artists[:]:
            line.remove()
        views_line_artists.clear()
        for coll in views_fill_artists[:]:
            coll.remove()
        views_fill_artists.clear()
        for line in views_bare_atom_artists[:]:
            line.remove()
        views_bare_atom_artists.clear()
        regions_data = last_nexafs.get("regions") or []
        energy = last_nexafs.get("energy")
        if not regions_data or energy is None:
            views_region_dropdown.options = ["All"]
            views_region_dropdown.value = "All"
            ax_views.set_ylabel("OD (ln I0/I)")
            fig_views.canvas.draw_idle()
            return
        spot_labels = sorted({normalize_spot_label(r.get("spot_label")) for r in regions_data})
        region_opts = ["All"] + spot_labels
        views_region_dropdown.options = region_opts
        current = (views_region_dropdown.value or "").strip()
        if current not in region_opts:
            views_region_dropdown.value = "pure" if "pure" in region_opts else "All"
        selected_region = (views_region_dropdown.value or "All").strip()
        if selected_region != "All":
            regions_data = [
                r for r in regions_data
                if normalize_spot_label(r.get("spot_label")) == selected_region
            ]
        if not regions_data:
            ax_views.set_ylabel("OD (ln I0/I)")
            fig_views.canvas.draw_idle()
            return
        energy = np.asarray(energy)
        y_min, y_max = np.inf, -np.inf
        y_label = "OD (ln I0/I)"
        sn = (sample_dropdown.value or "").strip()
        if sn in ("(load config first)", "(no samples)") or not sn:
            sn = "sample"
        mode = (display_mode.value or "OD").strip()
        mass_abs_mode = "Norm. Mass Abs. (g/cm^2)"
        cf_local = current_chemical_formula() if mode in (mass_abs_mode, "Beta") else None
        show_bare = bool(show_bare_atom.value) and cf_local is not None
        if views_twin_ax:
            views_twin_ax[0].remove()
            views_twin_ax.clear()
        if mode == "I0 & transmission":
            views_twin_ax.append(ax_views.twinx())
        for idx, reg in enumerate(regions_data):
            spot_label = normalize_spot_label(reg.get("spot_label"))
            label = f"{sn}:{spot_label}"
            color = REGION_COLORS[idx % len(REGION_COLORS)]
            if mode == "I0 & transmission":
                i0 = np.asarray(reg.get("I0", np.full_like(energy, np.nan)))
                i0_err = np.asarray(reg.get("I0_err", np.zeros_like(energy)))
                trans = np.asarray(reg.get("I", np.full_like(energy, np.nan)))
                trans_err = np.asarray(reg.get("I_err", np.zeros_like(energy)))
                f0 = ax_views.fill_between(
                    energy, i0 - i0_err, i0 + i0_err, alpha=0.25, color=color
                )
                views_fill_artists.append(f0)
                ln0, = ax_views.plot(energy, i0, color=color, lw=1.2, ls="-", label=f"{label} I0")
                views_line_artists.append(ln0)
                ft = views_twin_ax[0].fill_between(
                    energy, trans - trans_err, trans + trans_err, alpha=0.25, color=color
                )
                views_fill_artists.append(ft)
                lnt, = views_twin_ax[0].plot(energy, trans, color=color, lw=1.2, ls="--", label=f"{label} I")
                views_line_artists.append(lnt)
                if np.any(np.isfinite(i0)):
                    y_min = min(y_min, np.nanmin(i0 - i0_err))
                    y_max = max(y_max, np.nanmax(i0 + i0_err))
                if np.any(np.isfinite(trans)):
                    y_min = min(y_min, np.nanmin(trans - trans_err))
                    y_max = max(y_max, np.nanmax(trans + trans_err))
                continue
            od, sigma_od = spectrum_for_display(reg, energy)
            y, sig, y_label = compute_display_curve(
                mode,
                energy,
                od,
                sigma_od,
                cf_local,
                mass_abs_fit_mode.value or "",
            )
            sig = np.asarray(sig, dtype=float)
            fill = ax_views.fill_between(
                energy, y - sig, y + sig, alpha=0.25, color=color
            )
            views_fill_artists.append(fill)
            ln, = ax_views.plot(energy, y, color=color, lw=1.2, label=label)
            views_line_artists.append(ln)
            if np.any(np.isfinite(y)):
                y_min = min(y_min, np.nanmin(y - sig))
                y_max = max(y_max, np.nanmax(y + sig))
        if show_bare and cf_local is not None and mode == mass_abs_mode:
            try:
                mu_rho = mass_absorption_cm2_per_g(cf_local, energy, None)
                ln_bare, = ax_views.plot(
                    energy, mu_rho, color="black", lw=1, ls="--", alpha=0.9, label="Bare atom"
                )
                views_bare_atom_artists.append(ln_bare)
                if np.any(np.isfinite(mu_rho)):
                    y_min = min(y_min, np.nanmin(mu_rho))
                    y_max = max(y_max, np.nanmax(mu_rho))
            except Exception:
                pass
        if show_bare and cf_local is not None and mode == "Beta":
            try:
                mu_rho = mass_absorption_cm2_per_g(cf_local, energy, None)
                lam_cm = HC_EV_CM / np.asarray(energy, dtype=float)
                beta_bare = np.atleast_1d(mu_rho) * np.atleast_1d(lam_cm) / (4 * np.pi)
                ln_bare, = ax_views.plot(
                    energy, beta_bare, color="black", lw=1, ls="--", alpha=0.9, label="Bare atom"
                )
                views_bare_atom_artists.append(ln_bare)
                if np.any(np.isfinite(beta_bare)):
                    y_min = min(y_min, np.nanmin(beta_bare))
                    y_max = max(y_max, np.nanmax(beta_bare))
            except Exception:
                pass
        ax_views.set_ylabel(y_label)
        if mode == "I0 & transmission" and views_twin_ax:
            ax_views.set_ylabel("I0 (a.u.)")
            views_twin_ax[0].set_ylabel("Transmission I (a.u.)")
            h1, l1 = ax_views.get_legend_handles_labels()
            h2, l2 = views_twin_ax[0].get_legend_handles_labels()
            leg = ax_views.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            leg.set_draggable(True)
            for ax in (ax_views, views_twin_ax[0]):
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                set_ylim_from_data_with_margin(ax_views, y_min, y_max)
                set_ylim_from_data_with_margin(views_twin_ax[0], y_min, y_max)
        elif views_line_artists:
            make_draggable_legend(ax_views)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                set_ylim_from_data_with_margin(ax_views, y_min, y_max)
            ax_views.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax_views.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
        ax_views.set_xlim(energy.min(), energy.max())
        draw_edge_guides()
        fig_views.canvas.draw_idle()

    def draw_edge_guides() -> None:
        for artist in views_edge_artists:
            try:
                artist.remove()
            except Exception:
                pass
        views_edge_artists.clear()
        edge_lines.clear()
        if (display_mode.value or "OD").strip() == "I0 & transmission":
            return
        pre_lo, pre_hi = sorted((state["pre_lo"], state["pre_hi"]))
        post_lo, post_hi = sorted((state["post_lo"], state["post_hi"]))
        sp_pre = ax_views.axvspan(pre_lo, pre_hi, color="#1f77b4", alpha=0.10, zorder=0)
        sp_post = ax_views.axvspan(post_lo, post_hi, color="#d18b16", alpha=0.10, zorder=0)
        views_edge_artists.extend([sp_pre, sp_post])
        for key, color in (
            ("pre_lo", "#1f77b4"),
            ("pre_hi", "#1f77b4"),
            ("post_lo", "#d18b16"),
            ("post_hi", "#d18b16"),
        ):
            line = ax_views.axvline(
                state[key], color=color, lw=1.4, ls="--", alpha=0.9, picker=6, zorder=5
            )
            views_edge_artists.append(line)
            edge_lines[key] = line

    def _nearest_edge_key(x: float) -> str:
        keys = ("pre_lo", "pre_hi", "post_lo", "post_hi")
        return min(keys, key=lambda k: abs(x - state[k]))

    def on_press_views(event):
        if event.inaxes != ax_views or event.xdata is None or not edge_lines:
            return
        state["dragging_edge"] = _nearest_edge_key(float(event.xdata))

    def on_motion_views(event):
        key = state.get("dragging_edge")
        if key is None or event.inaxes != ax_views or event.xdata is None:
            return
        x0, x1 = ax_views.get_xlim()
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        x = float(np.clip(event.xdata, lo, hi))
        margin = (hi - lo) * 0.005
        if key == "pre_lo":
            x = min(x, state["pre_hi"] - margin)
        elif key == "pre_hi":
            x = max(x, state["pre_lo"] + margin)
        elif key == "post_lo":
            x = min(x, state["post_hi"] - margin)
        elif key == "post_hi":
            x = max(x, state["post_lo"] + margin)
        state[key] = x
        if key in edge_lines:
            edge_lines[key].set_xdata([x, x])
        fig_views.canvas.draw_idle()

    def on_release_views(event):
        key = state.get("dragging_edge")
        if key is None:
            return
        state["dragging_edge"] = None
        widget_for_key = {
            "pre_lo": pre_lo_widget,
            "pre_hi": pre_hi_widget,
            "post_lo": post_lo_widget,
            "post_hi": post_hi_widget,
        }
        target_widget = widget_for_key[key]
        new_value = round(float(state[key]), 3)
        if target_widget.value == new_value:
            sync_norm_windows_from_widgets()
        else:
            target_widget.value = new_value

    fig_views.canvas.mpl_connect("button_press_event", on_press_views)
    fig_views.canvas.mpl_connect("motion_notify_event", on_motion_views)
    fig_views.canvas.mpl_connect("button_release_event", on_release_views)

    def sync_step_edge_ui():
        mode = (display_mode.value or "OD").strip()
        step_edge_available = mode in ("Norm. Mass Abs. (g/cm^2)", "Beta")
        show_bare_atom.disabled = not step_edge_available
        if step_edge_available:
            cf = current_chemical_formula()
            if mode == "Norm. Mass Abs. (g/cm^2)" and cf:
                step_edge_hint.value = "<span style='font-size:10px;color:#666'>Theoretical mu/rho</span>"
            elif mode == "Beta" and cf:
                step_edge_hint.value = "<span style='font-size:10px;color:#666'>Theoretical beta</span>"
            else:
                step_edge_hint.value = "<span style='font-size:10px;color:#888'>Set sample formula in Reduction</span>"
        else:
            step_edge_hint.value = "<span style='font-size:10px;color:#999'>No step edge for OD or I0 & transmission</span>"

    def on_views_display_change(change: Any):
        sync_step_edge_ui()
        update_views_plot()

    display_mode.observe(on_views_display_change, names="value")
    mass_abs_fit_mode.observe(on_views_display_change, names="value")
    show_bare_atom.observe(on_views_display_change, names="value")
    sample_dropdown.observe(on_views_display_change, names="value")
    views_region_dropdown.observe(on_views_display_change, names="value")
    for w in (pre_lo_widget, pre_hi_widget, post_lo_widget, post_hi_widget):
        w.observe(sync_norm_windows_from_widgets, names="value")
    sync_step_edge_ui()

    current_scan_path: list[str | None] = [None]
    _persist_timer: list[threading.Timer | None] = [None]

    def _cancel_persist_timer() -> None:
        timer = _persist_timer[0]
        if timer is not None:
            timer.cancel()
            _persist_timer[0] = None

    def _persist_regions_now() -> None:
        path = current_scan_path[0]
        exp_dir = current_dir[0]
        if path is None or not isinstance(exp_dir, Path):
            return
        try:
            save_scan_regions(
                exp_dir,
                path,
                izero_lo=float(state["izero_lo"]),
                izero_hi=float(state["izero_hi"]),
                regions=state["regions"],
            )
        except Exception:
            pass

    def schedule_persist_regions() -> None:
        if current_scan_path[0] is None:
            return
        _cancel_persist_timer()
        timer = threading.Timer(0.4, _persist_regions_now)
        timer.daemon = True
        _persist_timer[0] = timer
        timer.start()

    def _sync_region_lines_to_state() -> None:
        while len(region_lines) > len(state["regions"]):
            l_lo, l_hi = region_lines.pop()
            l_lo.remove()
            l_hi.remove()
        while len(region_lines) < len(state["regions"]):
            idx = len(region_lines)
            color = REGION_COLORS[idx % len(REGION_COLORS)]
            rl0 = ax_im.axhline(0.0, color=color, lw=2, picker=5)
            rl1 = ax_im.axhline(0.0, color=color, lw=2, picker=5)
            region_lines.append((rl0, rl1))
        for i, reg in enumerate(state["regions"]):
            l_lo, l_hi = region_lines[i]
            l_lo.set_ydata([reg["sample_lo"], reg["sample_lo"]])
            l_hi.set_ydata([reg["sample_hi"], reg["sample_hi"]])

    def _rebuild_spot_widgets() -> None:
        region_spot_widgets.clear()
        region_spot_rows.clear()
        for j, reg in enumerate(state["regions"]):
            spot_w, row = make_spot_row(
                normalize_spot_label(reg.get("spot_label")), j
            )
            region_spot_widgets.append(spot_w)
            region_spot_rows.append(row)
        region_spot_box.children = tuple(region_spot_rows)

    def _apply_region_payload(payload: dict[str, Any]) -> None:
        state["izero_lo"] = float(payload["izero_lo"])
        state["izero_hi"] = float(payload["izero_hi"])
        state["regions"] = [
            {
                "sample_lo": float(r["sample_lo"]),
                "sample_hi": float(r["sample_hi"]),
                "spot_label": normalize_spot_label(r.get("spot_label")),
            }
            for r in payload["regions"]
        ]
        _sync_region_lines_to_state()
        _rebuild_spot_widgets()
        line_c.set_ydata([state["izero_lo"], state["izero_lo"]])
        line_d.set_ydata([state["izero_hi"], state["izero_hi"]])
        _apply_image_and_region_lines(
            im_artist,
            ax_im,
            data["image"],
            data["meta"],
            state["izero_lo"],
            state["izero_hi"],
            state["regions"],
            line_c,
            line_d,
            region_lines,
            region_label_texts,
            izero_label_text,
        )

    def do_load():
        val = file_dropdown.value
        if val is None or val == "(empty)":
            return
        path = str(current_dir[0] / str(val))
        try:
            meta_new, image_new = load_stxm(path)
        except Exception as e:
            print(f"Load failed: {e}")
            return
        data["meta"] = meta_new
        data["image"] = image_new
        qaxis = meta_new["qaxis_points"]
        paxis = meta_new["paxis_points"]
        current_scan_path[0] = path
        saved = load_scan_regions(current_dir[0], path)
        if saved is not None:
            payload = saved
        else:
            payload = default_regions_from_image(image_new, qaxis)
        _apply_region_payload(payload)
        paxis_pts = np.asarray(meta_new["paxis_points"])
        if paxis_pts.size >= 2:
            margin = 0.15 * float(paxis_pts[-1] - paxis_pts[0])
            state["pre_lo"] = float(paxis_pts[0])
            state["pre_hi"] = float(paxis_pts[0] + margin)
            state["post_lo"] = float(paxis_pts[-1] - margin)
            state["post_hi"] = float(paxis_pts[-1])
            pre_lo_widget.value = state["pre_lo"]
            pre_hi_widget.value = state["pre_hi"]
            post_lo_widget.value = state["post_lo"]
            post_hi_widget.value = state["post_hi"]
        status_label.value = f"<span style='font-size:11px;color:#066'>Loaded: {truncate_path(path)}</span>"
        update_od()
        refresh_lcf_catalog()
        fig.canvas.draw_idle()
        fig.canvas.draw()

    def _add_derived_columns(df: pd.DataFrame, energy: np.ndarray, od: np.ndarray, od_err: np.ndarray) -> pd.DataFrame:
        energy = np.asarray(energy)
        od = np.asarray(od)
        od_err = np.asarray(od_err)
        cf = current_chemical_formula()
        t_cm = 1e-4
        lam_cm = HC_EV_CM / energy
        beta = od_to_beta(energy, od, t_cm)
        beta_err = od_err * lam_cm / (4 * np.pi * t_cm)
        df = df.copy()
        df["beta"] = beta
        df["beta_err"] = beta_err
        if cf:
            try:
                mu_rho = mass_absorption_cm2_per_g(cf, energy, None)
                fit_opt = mass_abs_fit_mode.value or ""
                n_low = 5 if "offset" in fit_opt.lower() else 0
                n_high = 5
                scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
                scale = scale if scale != 0 else 1.0
                df["mass_absorption"] = (od - const) / scale
                df["mass_absorption_err"] = np.abs(od_err / scale)
            except Exception:
                df["mass_absorption"] = np.nan
                df["mass_absorption_err"] = np.nan
        else:
            df["mass_absorption"] = np.nan
            df["mass_absorption_err"] = np.nan
        return df

    def on_export(btn):
        regions_data = last_nexafs.get("regions") or []
        if not regions_data:
            status_label.value = "<span style='font-size:11px;color:#c00'>Load a file and reduce first.</span>"
            return
        parquet_p = (parquet_path_text.value or "").strip()
        if not parquet_p:
            status_label.value = "<span style='font-size:11px;color:#c00'>Set export file name.</span>"
            return
        sn = (sample_dropdown.value or "").strip() or (experiment_dropdown.value or "sample")
        if sn in ("(load config first)", "(no samples)"):
            sn = experiment_dropdown.value or "sample"
        parquet_path = resolve_parquet_path(parquet_p)
        fr = sn
        cf = current_chemical_formula()
        energy = last_nexafs["energy"]
        store_name = (store_root_text.value or "").strip()
        store_root = resolve_parquet_path(store_name) if store_name else None
        edge = (edge_text.value or "C_K").strip() or "C_K"
        mode = WeightingMode(weighting_dropdown.value)
        hdr_path = current_scan_path[0]
        try:
            for reg in regions_data:
                df = pd.DataFrame({
                    "energy_eV": energy,
                    "OD": reg["OD"],
                    "OD_err": reg["OD_err"],
                    "I0": reg["I0"],
                    "I0_err": reg["I0_err"],
                    "I": reg["I"],
                    "I_err": reg["I_err"],
                    "n_sample": reg["n_sample"],
                    "n_izero": reg["n_izero"],
                })
                df = _add_derived_columns(df, energy, reg["OD"], reg["OD_err"])
                od_norm, norm_meta = normalize_nexafs_with_metadata(
                    energy,
                    reg["OD"],
                    float(state["pre_lo"]),
                    float(state["pre_hi"]),
                    float(state["post_lo"]),
                    float(state["post_hi"]),
                    mode=norm_mode_dropdown.value,
                )
                df["OD_normalized"] = od_norm
                append_nexafs_to_experiment(
                    parquet_path,
                    df,
                    sample_name=sn,
                    chemical_formula=cf,
                    spot_label=reg["spot_label"],
                    film_region_name=fr,
                    scan_path=hdr_path,
                    formula=cf or "",
                )
                if store_root is not None and hdr_path is not None:
                    spectrum = RegionSpectrum(
                        energy_eV=np.asarray(energy, dtype=np.float64),
                        OD=np.asarray(reg["OD"], dtype=np.float64),
                        OD_err=np.asarray(reg["OD_err"], dtype=np.float64),
                        region_label=reg["spot_label"],
                        weighting_mode=mode.value,
                        reduction_method="two_region",
                        n_pixels=int(reg["n_sample"]),
                    )
                    bounds = {
                        "sample_lo": float(reg.get("sample_lo", 0.0)),
                        "sample_hi": float(reg.get("sample_hi", 0.0)),
                        "izero_lo": float(last_nexafs.get("izero_lo", 0.0)),
                        "izero_hi": float(last_nexafs.get("izero_hi", 0.0)),
                    }
                    prov = provenance_from_hdr(
                        hdr_path,
                        sample_name=sn,
                        region_label=reg["spot_label"],
                        edge=edge,
                        sample_bounds=bounds,
                        pre_edge=(float(state["pre_lo"]), float(state["pre_hi"])),
                        post_edge=(float(state["post_lo"]), float(state["post_hi"])),
                        weighting_mode=mode,
                        reduction_method="two_region",
                    )
                    prov.extra.update(norm_meta)
                    od_norm = df["OD_normalized"].to_numpy()
                    write_spectrum(
                        store_root,
                        spectrum,
                        prov,
                        od_normalized=od_norm,
                    )
            _persist_regions_now()
            msg = f"Appended {len(regions_data)} region(s) to {parquet_path.name}"
            if store_root is not None:
                msg += f" and store {store_root.name}"
            status_label.value = f"<span style='font-size:11px;color:#066'>{msg}</span>"
        except Exception as e:
            status_label.value = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    def on_file_select(change):
        if change.get("new") is None:
            return
        do_load()

    file_dropdown.observe(on_file_select, names="value")

    def on_export_and_refresh(btn: Any):
        on_export(btn)
        refresh_lcf_catalog()

    export_btn.on_click(on_export_and_refresh)
    refresh_btn.on_click(refresh_experiment_list)

    def on_refresh_plots(_):
        update_od()
        fig.canvas.draw_idle()
        fig.canvas.draw()

    refresh_plots_btn.on_click(on_refresh_plots)

    _configure_mpl_canvas(fig, min_height="300px", min_width="0")
    if getattr(fig.canvas, "layout", None) is not None:
        setattr(fig.canvas.layout, "max_width", _INGESTION_MAP_CANVAS_MAX_WIDTH)
        setattr(fig.canvas.layout, "flex", "0 0 auto")
    _configure_mpl_canvas(fig_views, min_height="280px")

    reduction_top = widgets.HBox(
        [
            file_dropdown,
            sample_dropdown,
            weighting_dropdown,
            new_sample_name,
            new_sample_formula,
            add_sample_btn,
        ],
        layout=widgets.Layout(
            width="100%",
            max_width="720px",
            align_items="center",
            flex_flow="wrap",
            gap="8px",
        ),
    )
    ingestion_norm_row = _ingestion_controls_row(
        _ingestion_compact_field("Norm", norm_mode_dropdown, width="140px"),
        _ingestion_compact_field("Basis", spectrum_basis_dropdown, width="120px"),
        _ingestion_compact_field("Pre lo", pre_lo_widget, width="64px"),
        _ingestion_compact_field("Pre hi", pre_hi_widget, width="64px"),
        _ingestion_compact_field("Post lo", post_lo_widget, width="64px"),
        _ingestion_compact_field("Post hi", post_hi_widget, width="64px"),
    )
    ingestion_views_row = _ingestion_controls_row(
        _ingestion_compact_field("Y-axis", display_mode, width="100px"),
        _ingestion_compact_field("Region", views_region_dropdown, width="100px"),
        _ingestion_compact_field("Bare-atom", mass_abs_fit_mode, width="160px"),
        _ingestion_compact_field(
            "Step edge",
            widgets.HBox(
                [show_bare_atom, step_edge_hint],
                layout=widgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    gap="4px",
                    width="100%",
                ),
            ),
            width="auto",
            flex="1 1 200px",
        ),
    )
    ingestion_spectrum_controls = widgets.VBox(
        [
            widgets.HTML(
                value=(
                    f"<span style='{_DASH_HEADER_LABEL}'>Spectrum</span>"
                    "<span style='font-size:10px;color:#6e7781;margin-left:8px'>"
                    "drag the dashed bars on the plot to set the pre-edge (blue) and "
                    "post-edge (amber) windows</span>"
                )
            ),
            ingestion_norm_row,
            ingestion_views_row,
        ],
        layout=widgets.Layout(width="100%", min_width="0"),
    )
    ingestion_regions_block = widgets.VBox(
        [
            region_spot_box,
            add_region_btn,
        ],
        layout=widgets.Layout(
            width="100%",
            max_height="220px",
            overflow_y="auto",
            gap="4px",
        ),
    )
    ingestion_left_column = widgets.VBox(
        [
            widgets.HTML(value=f"<span style='{_DASH_HEADER_LABEL}'>Line scan</span>"),
            fig.canvas,
            widgets.HTML(value=f"<span style='{_DASH_HEADER_LABEL}'>Regions</span>"),
            ingestion_regions_block,
            widgets.HTML(value=f"<span style='{_DASH_HEADER_LABEL}'>Export</span>"),
            export_btn,
        ],
        layout=widgets.Layout(
            width=_INGESTION_LEFT_WIDTH,
            min_width=_INGESTION_LEFT_WIDTH,
            flex=f"0 0 {_INGESTION_LEFT_WIDTH}",
            padding="4px 8px 8px 0",
        ),
    )
    ingestion_spectrum_figure = _figure_column(fig_views.canvas)
    ingestion_right_column = widgets.VBox(
        [
            ingestion_spectrum_controls,
            ingestion_spectrum_figure,
        ],
        layout=widgets.Layout(
            flex="1 1 auto",
            min_width="0",
            width="100%",
            overflow="hidden",
        ),
    )
    ingestion_main = _split_row(ingestion_left_column, ingestion_right_column, gap="14px")
    tab_ingestion = widgets.VBox(
        [
            reduction_top,
            ingestion_main,
            widgets.HBox(
                [refresh_plots_btn, status_label],
                layout=widgets.Layout(width="100%", margin="8px 0 0 0"),
            ),
        ],
        layout=widgets.Layout(width="100%", padding="8px"),
    )

    def load_browser_parquet() -> list[dict]:
        parquet_p = (parquet_path_text.value or "").strip()
        if not parquet_p:
            return []
        path = resolve_parquet_path(parquet_p)
        if not path.is_file():
            return []
        try:
            df = load_experiment_parquet(path)
        except Exception:
            return []
        if SAMPLE_NAME_COLUMN not in df.columns or SPOT_LABEL_COLUMN not in df.columns:
            return []
        if "energy_eV" not in df.columns or "OD" not in df.columns:
            return []
        scan_col = SCAN_PATH_COLUMN if SCAN_PATH_COLUMN in df.columns else "scan_path"
        if scan_col not in df.columns:
            df = df.copy()
            df[scan_col] = ""
        groups = df.groupby([SAMPLE_NAME_COLUMN, SPOT_LABEL_COLUMN, scan_col], dropna=False)
        spectra: list[dict] = []
        for key, g in groups:
            sample_name, spot_label, scan_path = cast(tuple[Any, Any, Any], key)
            g = g.sort_values("energy_eV")
            energy = np.asarray(g["energy_eV"])
            spectra.append({
                "sample_name": str(sample_name) if pd.notna(sample_name) else "",
                "spot_label": str(spot_label) if pd.notna(spot_label) else "pure",
                "scan_path": str(scan_path) if pd.notna(scan_path) else "",
                "energy": energy,
                "OD": np.asarray(g["OD"]),
                "OD_err": np.asarray(g.get("OD_err", np.zeros_like(energy))),
                "I0": np.asarray(g.get("I0", np.full_like(energy, np.nan))),
                "I0_err": np.asarray(g.get("I0_err", np.zeros_like(energy))),
                "I": np.asarray(g.get("I", np.full_like(energy, np.nan))),
                "I_err": np.asarray(g.get("I_err", np.zeros_like(energy))),
                "chemical_formula": g[CHEMICAL_FORMULA_COLUMN].iloc[0] if CHEMICAL_FORMULA_COLUMN in g.columns else None,
            })
        return sorted(spectra, key=lambda s: (s["sample_name"], s["spot_label"], s["scan_path"]))

    browser_spectra: list[dict] = []
    browser_selection: set[int] = set()
    browser_row_label = _DASH_HEADER_LABEL
    browser_refresh_btn = widgets.Button(
        description="Refresh",
        tooltip="Reload experiment.parquet from disk (e.g. when others export to the same experiment)",
        layout=widgets.Layout(**_DASH_REFRESH_BTN_LAYOUT),
    )
    browser_status = widgets.HTML(
        value="<span style='font-size:11px;color:#6e7781;'>No parquet loaded.</span>"
    )
    browser_sample_dropdown = widgets.Dropdown(
        options=[],
        value=None,
        description="",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )
    browser_region_dropdown = widgets.Dropdown(
        options=[],
        value=None,
        description="",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )
    browser_scan_box = widgets.VBox([], layout=widgets.Layout(width="100%"))
    browser_display_mode = widgets.Dropdown(
        options=["OD", "Norm. Mass Abs. (g/cm^2)", "Beta", "I0 & transmission"],
        value="OD",
        description="",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )
    browser_mass_abs_fit = widgets.Dropdown(
        options=["Scale only (last 5 pts)", "Scale & offset (first+last 5)"],
        value="Scale only (last 5 pts)",
        description="",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )
    browser_show_bare_atom = widgets.Checkbox(value=True, description="", indent=False, layout=widgets.Layout(width="100%"))
    plt.ioff()
    fig_browser, ax_browser = plt.subplots(
        1, 1, figsize=_SPECTRUM_FIGSIZE, constrained_layout=True
    )
    _configure_mpl_canvas(fig_browser, min_height="260px")
    ax_browser.set_xlabel(meta.get("paxis_name", "Energy (eV)"))
    ax_browser.grid(True, alpha=0.3)
    ax_browser.set_ylabel("OD (ln I0/I)")
    style_axes(ax_browser)
    browser_line_artists: list = []
    browser_fill_artists: list = []
    browser_bare_atom_artists: list = []
    browser_twin_ax: list = []
    browser_region_colors = REGION_COLORS

    def browser_formula_for_sample(sample_name: str, fallback: Any = None) -> str | None:
        out = config_map.get(sample_name) if sample_name else None
        if out is not None and str(out).strip():
            return str(out).strip()
        return str(fallback).strip() if fallback is not None and pd.notna(fallback) and str(fallback).strip() else None

    def browser_spectrum_od(sp: dict) -> tuple[np.ndarray, np.ndarray]:
        energy_axis = np.asarray(sp["energy"], dtype=np.float64)
        od = np.asarray(sp["OD"], dtype=np.float64)
        sigma = np.asarray(sp.get("OD_err", np.zeros_like(od)), dtype=np.float64)
        if (spectrum_basis_dropdown.value or "Raw OD").strip() != "Normalized OD":
            return od, sigma
        normed, _ = normalize_nexafs_with_metadata(
            energy_axis,
            od,
            float(state["pre_lo"]),
            float(state["pre_hi"]),
            float(state["post_lo"]),
            float(state["post_hi"]),
            mode=norm_mode_dropdown.value,
        )
        return normed, sigma

    def compute_browser_curve(energy: np.ndarray, od: np.ndarray, sigma_od: np.ndarray, formula: str | None) -> tuple[np.ndarray, np.ndarray, str]:
        mode = (browser_display_mode.value or "OD").strip()
        if mode == "Norm. Mass Abs. (g/cm^2)" and formula is None:
            return od, sigma_od, "OD (set formula)"
        return compute_display_curve(
            mode,
            energy,
            od,
            sigma_od,
            formula,
            browser_mass_abs_fit.value or "",
        )

    def refresh_browser(_: Any = None):
        nonlocal browser_spectra
        browser_spectra = load_browser_parquet()
        samples = sorted({s["sample_name"] for s in browser_spectra if s["sample_name"]})
        browser_sample_dropdown.options = samples if samples else ["(none)"]
        browser_sample_dropdown.value = samples[0] if samples else None
        sample = browser_sample_dropdown.value
        regions = sorted({(s.get("spot_label") or "pure").strip() or "pure" for s in browser_spectra if s.get("sample_name") == sample}) if sample else []
        browser_region_dropdown.options = regions if regions else ["(none)"]
        browser_region_dropdown.value = "pure" if regions and "pure" in regions else (regions[0] if regions else None)
        region = browser_region_dropdown.value
        pairs = spectra_for_sample_region(sample, region)
        browser_selection.clear()
        browser_selection.update(i for i, _ in pairs)
        browser_status.value = f"<span style='font-size:11px;color:#066'>Loaded {len(browser_spectra)} spectrum(a) from {resolve_parquet_path((parquet_path_text.value or '').strip()).name}</span>" if browser_spectra else "<span style='font-size:11px;color:#888'>No parquet or no data.</span>"
        update_browser_scan_list()
        update_browser_plot()
        refresh_lcf_catalog()

    def spectra_for_sample_region(sample: str | None, region: str | None) -> list[tuple[int, dict]]:
        if sample is None or region is None:
            return []
        norm = lambda s: (s.get("spot_label") or "pure").strip() or "pure"
        return [(i, s) for i, s in enumerate(browser_spectra) if s.get("sample_name") == sample and norm(s) == region]

    def update_browser_scan_list():
        sample = browser_sample_dropdown.value
        region = browser_region_dropdown.value
        pairs = spectra_for_sample_region(sample, region)
        children = []
        for idx, sp in pairs:
            label = Path(sp["scan_path"]).name if sp["scan_path"] else f"Scan {idx}"
            cb = widgets.Checkbox(value=idx in browser_selection, description=label[:60], indent=False, layout=widgets.Layout(width="100%"))
            def make_handler(i: int, c: widgets.Checkbox):
                def handler(change: Any):
                    if change.get("new"):
                        browser_selection.add(i)
                    else:
                        browser_selection.discard(i)
                    update_browser_plot()
                return handler
            cb.observe(make_handler(idx, cb), names="value")
            children.append(cb)
        browser_scan_box.children = tuple(children)

    def on_browser_sample_region_change(change: Any):
        sample = browser_sample_dropdown.value
        region = browser_region_dropdown.value
        regions = sorted({(s.get("spot_label") or "pure").strip() or "pure" for s in browser_spectra if s.get("sample_name") == sample})
        browser_region_dropdown.options = regions if regions else ["(none)"]
        if regions and (region not in regions or region is None):
            browser_region_dropdown.value = "pure" if "pure" in regions else regions[0]
        region = browser_region_dropdown.value
        pairs = spectra_for_sample_region(sample, region)
        browser_selection.clear()
        browser_selection.update(i for i, _ in pairs)
        sync_browser_step_edge_ui()
        update_browser_scan_list()
        update_browser_plot()

    browser_sample_dropdown.observe(on_browser_sample_region_change, names="value")
    browser_region_dropdown.observe(on_browser_sample_region_change, names="value")

    def update_browser_plot():
        for line in browser_line_artists[:]:
            line.remove()
        browser_line_artists.clear()
        for coll in browser_fill_artists[:]:
            coll.remove()
        browser_fill_artists.clear()
        for line in browser_bare_atom_artists[:]:
            line.remove()
        browser_bare_atom_artists.clear()
        if browser_twin_ax:
            browser_twin_ax[0].remove()
            browser_twin_ax.clear()
        selected = [browser_spectra[i] for i in sorted(browser_selection) if i < len(browser_spectra)]
        if not selected:
            ax_browser.set_ylabel("OD (ln I0/I)")
            fig_browser.canvas.draw_idle()
            return
        energy = selected[0]["energy"]
        y_min, y_max = np.inf, -np.inf
        y_label = "OD (ln I0/I)"
        mode = (browser_display_mode.value or "OD").strip()
        if mode == "I0 & transmission":
            browser_twin_ax.append(ax_browser.twinx())
        for idx, sp in enumerate(selected):
            color = browser_region_colors[idx % len(browser_region_colors)]
            sn, sl = sp["sample_name"], sp["spot_label"]
            label = spectrum_legend_label(sn, sl, sp.get("scan_path"))
            formula = browser_formula_for_sample(sn, sp.get("chemical_formula"))
            if mode == "I0 & transmission":
                i0, i0_err = sp["I0"], sp["I0_err"]
                trans, trans_err = sp["I"], sp["I_err"]
                f0 = ax_browser.fill_between(energy, i0 - i0_err, i0 + i0_err, alpha=0.25, color=color)
                browser_fill_artists.append(f0)
                ln0, = ax_browser.plot(energy, i0, color=color, lw=1.2, ls="-", label=f"{label} I0")
                browser_line_artists.append(ln0)
                ft = browser_twin_ax[0].fill_between(energy, trans - trans_err, trans + trans_err, alpha=0.25, color=color)
                browser_fill_artists.append(ft)
                lnt, = browser_twin_ax[0].plot(energy, trans, color=color, lw=1.2, ls="--", label=f"{label} I")
                browser_line_artists.append(lnt)
                if np.any(np.isfinite(i0)):
                    y_min, y_max = min(y_min, np.nanmin(i0 - i0_err)), max(y_max, np.nanmax(i0 + i0_err))
                if np.any(np.isfinite(trans)):
                    y_min, y_max = min(y_min, np.nanmin(trans - trans_err)), max(y_max, np.nanmax(trans + trans_err))
                continue
            od, od_err = browser_spectrum_od(sp)
            y, sig, y_label = compute_browser_curve(energy, od, od_err, formula)
            sig = np.asarray(sig, dtype=float)
            fill = ax_browser.fill_between(energy, y - sig, y + sig, alpha=0.25, color=color)
            browser_fill_artists.append(fill)
            ln, = ax_browser.plot(energy, y, color=color, lw=1.2, label=label)
            browser_line_artists.append(ln)
            if np.any(np.isfinite(y)):
                y_min, y_max = min(y_min, np.nanmin(y - sig)), max(y_max, np.nanmax(y + sig))
        if browser_show_bare_atom.value and mode in ("Norm. Mass Abs. (g/cm^2)", "Beta") and selected:
            cf = browser_formula_for_sample(selected[0]["sample_name"], selected[0].get("chemical_formula"))
            if cf:
                try:
                    mu_rho = mass_absorption_cm2_per_g(cf, energy, None)
                    if mode == "Norm. Mass Abs. (g/cm^2)":
                        ln_bare, = ax_browser.plot(energy, mu_rho, color="black", lw=1, ls="--", alpha=0.9, label="Bare atom")
                        browser_bare_atom_artists.append(ln_bare)
                        y_min, y_max = min(y_min, np.nanmin(mu_rho)), max(y_max, np.nanmax(mu_rho))
                    else:
                        lam_cm = HC_EV_CM / np.asarray(energy, dtype=float)
                        beta_bare = np.atleast_1d(mu_rho) * np.atleast_1d(lam_cm) / (4 * np.pi)
                        ln_bare, = ax_browser.plot(energy, beta_bare, color="black", lw=1, ls="--", alpha=0.9, label="Bare atom")
                        browser_bare_atom_artists.append(ln_bare)
                        y_min, y_max = min(y_min, np.nanmin(beta_bare)), max(y_max, np.nanmax(beta_bare))
                except Exception:
                    pass
        ax_browser.set_ylabel(y_label)
        if mode == "I0 & transmission" and browser_twin_ax:
            ax_browser.set_ylabel("I0 (a.u.)")
            browser_twin_ax[0].set_ylabel("Transmission I (a.u.)")
            h1, l1 = ax_browser.get_legend_handles_labels()
            h2, l2 = browser_twin_ax[0].get_legend_handles_labels()
            leg = ax_browser.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            leg.set_draggable(True)
            for ax in (ax_browser, browser_twin_ax[0]):
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                set_ylim_from_data_with_margin(ax_browser, y_min, y_max)
                set_ylim_from_data_with_margin(browser_twin_ax[0], y_min, y_max)
        elif browser_line_artists:
            make_draggable_legend(ax_browser)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                set_ylim_from_data_with_margin(ax_browser, y_min, y_max)
            ax_browser.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax_browser.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
        ax_browser.set_xlim(energy.min(), energy.max())
        fig_browser.canvas.draw_idle()

    browser_step_edge_hint = widgets.HTML(
        value="<span style='font-size:10px;color:#999'>No step edge for OD or I0 &amp; transmission</span>",
        layout=widgets.Layout(margin="0 0 0 6px"),
    )

    def sync_browser_step_edge_ui():
        mode = (browser_display_mode.value or "OD").strip()
        step_edge_available = mode in ("Norm. Mass Abs. (g/cm^2)", "Beta")
        browser_show_bare_atom.disabled = not step_edge_available
        if step_edge_available:
            sample = browser_sample_dropdown.value
            cf = browser_formula_for_sample(str(sample or ""), None) if sample else None
            if mode == "Norm. Mass Abs. (g/cm^2)" and cf:
                browser_step_edge_hint.value = (
                    "<span style='font-size:10px;color:#666'>Theoretical mu/rho</span>"
                )
            elif mode == "Beta" and cf:
                browser_step_edge_hint.value = (
                    "<span style='font-size:10px;color:#666'>Theoretical beta</span>"
                )
            else:
                browser_step_edge_hint.value = (
                    "<span style='font-size:10px;color:#888'>Set formula in samples config</span>"
                )
        else:
            browser_step_edge_hint.value = (
                "<span style='font-size:10px;color:#999'>No step edge for OD or I0 &amp; transmission</span>"
            )

    def on_browser_display_change(change: Any):
        sync_browser_step_edge_ui()
        update_browser_plot()

    lcf_catalog: list[dict[str, Any]] = []
    lcf_component_rows: list[dict[str, Any]] = []
    lcf_preview_timer: list[threading.Timer | None] = [None]
    lcf_last_fit: list[Any] = [None]

    def _bounded_float_text(
        value: float,
        minimum: float,
        maximum: float,
        *,
        width: str = "54px",
    ) -> widgets.FloatText:
        widget = widgets.FloatText(
            value=float(value),
            layout=widgets.Layout(width=width),
        )

        def _clamp(change: Any) -> None:
            raw = change.get("new")
            if raw is None:
                return
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                return
            if not np.isfinite(numeric):
                return
            clamped = min(max(numeric, minimum), maximum)
            if abs(clamped - numeric) > 1e-12:
                widget.unobserve(_clamp, names="value")
                widget.value = clamped
                widget.observe(_clamp, names="value")

        widget.observe(_clamp, names="value")
        return widget

    lcf_target_dropdown = widgets.Dropdown(
        options=[],
        description="Target",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "52px"},
    )
    lcf_components_box = widgets.VBox(
        [],
        layout=widgets.Layout(
            width="100%",
            max_height="280px",
            overflow_y="auto",
            border="1px solid #ddd",
            padding="4px",
        ),
    )
    lcf_add_component_btn = widgets.Button(description="+ Add component")
    lcf_sum_to_one = widgets.Checkbox(
        value=True,
        description="Sum to 1",
        indent=False,
    )
    lcf_nonneg = widgets.Checkbox(
        value=True,
        description="Non-negative",
        indent=False,
    )
    lcf_run_btn = widgets.Button(description="Run LCF", button_style="primary")
    lcf_status = widgets.HTML(
        value="<span style='font-size:11px;color:#888'>Select target and film components.</span>"
    )

    def _render_lcf_metrics(
        chi: str = "—", n_comp: str = "—", frac_sum: str = "—", rms: str = "—"
    ) -> str:
        def card(label: str, value: str) -> str:
            return (
                "<div style='background:#f1efe8;border-radius:8px;padding:10px 14px;"
                "min-width:120px;flex:1 1 120px'>"
                f"<div style='font-size:12px;color:#6e7781'>{label}</div>"
                f"<div style='font-size:22px;font-weight:500;color:#24292f'>{value}</div></div>"
            )
        return (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;width:100%;margin:6px 0'>"
            + card("reduced chi-square", chi)
            + card("components", n_comp)
            + card("fraction sum", frac_sum)
            + card("residual rms", rms)
            + "</div>"
        )

    lcf_metrics = widgets.HTML(value=_render_lcf_metrics())
    plt.ioff()
    fig_lcf, (ax_lcf_fit, ax_lcf_resid) = plt.subplots(
        2,
        1,
        figsize=_LCF_FIGSIZE,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    _configure_mpl_canvas(fig_lcf, min_height="360px")
    ax_lcf_fit.set_ylabel("OD")
    ax_lcf_resid.set_ylabel("Residual")
    ax_lcf_resid.set_xlabel("Energy (eV)")
    ax_lcf_fit.grid(True, alpha=0.25)
    ax_lcf_resid.grid(True, alpha=0.25)
    style_axes(ax_lcf_fit)
    style_axes(ax_lcf_resid)
    lcf_line_artists: list = []
    lcf_fill_artists: list = []

    def _catalog_od(energy_axis: np.ndarray, od: np.ndarray, od_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (spectrum_basis_dropdown.value or "Raw OD").strip() != "Normalized OD":
            return np.asarray(od, dtype=np.float64), np.asarray(od_err, dtype=np.float64)
        normed, _ = normalize_nexafs_with_metadata(
            energy_axis,
            od,
            float(state["pre_lo"]),
            float(state["pre_hi"]),
            float(state["post_lo"]),
            float(state["post_hi"]),
            mode=norm_mode_dropdown.value,
        )
        return normed, np.asarray(od_err, dtype=np.float64)

    def build_lcf_catalog() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        energy_cur = last_nexafs.get("energy")
        regions_cur = last_nexafs.get("regions") or []
        sn_cur = (sample_dropdown.value or "").strip()
        if sn_cur in ("(load config first)", "(no samples)"):
            sn_cur = experiment_dropdown.value or "sample"
        if energy_cur is not None and regions_cur:
            energy_axis = np.asarray(energy_cur, dtype=np.float64)
            for reg in regions_cur:
                spot = normalize_spot_label(reg.get("spot_label"))
                label = f"Current: {sn_cur}:{spot}"
                entries.append({
                    "id": f"current:{sn_cur}:{spot}",
                    "label": label,
                    "energy": energy_axis,
                    "od": np.asarray(reg["OD"], dtype=np.float64),
                    "od_err": np.asarray(reg.get("OD_err", np.zeros_like(energy_axis)), dtype=np.float64),
                })
        for idx, sp in enumerate(browser_spectra):
            energy_axis = np.asarray(sp["energy"], dtype=np.float64)
            sn = sp.get("sample_name") or ""
            spot = (sp.get("spot_label") or "pure").strip() or "pure"
            label = f"Parquet: {spectrum_legend_label(sn, spot, sp.get('scan_path'))}"
            entries.append({
                "id": f"parquet:{idx}",
                "label": label,
                "energy": energy_axis,
                "od": np.asarray(sp["OD"], dtype=np.float64),
                "od_err": np.asarray(sp.get("OD_err", np.zeros_like(energy_axis)), dtype=np.float64),
            })
        store_name = (store_root_text.value or "").strip()
        if store_name:
            store_path = resolve_parquet_path(store_name)
            if store_path.is_dir():
                store_df = query_spectra(store_path)
                if not store_df.empty and "energy_eV" in store_df.columns:
                    group_cols = ["sample_name", "region_label", "scan_id"]
                    for col in group_cols:
                        if col not in store_df.columns:
                            store_df[col] = ""
                    for key, grp in store_df.groupby(group_cols, dropna=False):
                        sample_name, region_label, scan_id = cast(tuple[Any, Any, Any], key)
                        grp = grp.sort_values("energy_eV")
                        energy_axis = np.asarray(grp["energy_eV"], dtype=np.float64)
                        od_vals = np.asarray(grp["OD"], dtype=np.float64)
                        od_err = (
                            np.asarray(grp["OD_err"], dtype=np.float64)
                            if "OD_err" in grp.columns
                            else np.zeros_like(od_vals)
                        )
                        scan = str(scan_id) if pd.notna(scan_id) else ""
                        label = f"Store: {sample_name}:{region_label}"
                        if scan:
                            label = f"{label} ({scan})"
                        entries.append({
                            "id": f"store:{sample_name}:{region_label}:{scan}",
                            "label": label,
                            "energy": energy_axis,
                            "od": od_vals,
                            "od_err": od_err,
                        })
        return entries

    def _lcf_ref_options(exclude_target: bool = True) -> list[str]:
        labels = [e["label"] for e in lcf_catalog]
        target_label = lcf_target_dropdown.value
        if exclude_target and target_label in labels:
            return [lbl for lbl in labels if lbl != target_label]
        return labels

    def _sync_lcf_component_dropdowns() -> None:
        ref_opts = _lcf_ref_options(exclude_target=True)
        for row in lcf_component_rows:
            dropdown = row["ref_dropdown"]
            current = dropdown.value
            dropdown.options = ref_opts
            if current in ref_opts:
                dropdown.value = current
            elif ref_opts:
                dropdown.value = ref_opts[0]
            else:
                dropdown.value = None

    def _rebuild_lcf_components_box() -> None:
        lcf_components_box.children = tuple(row["row_box"] for row in lcf_component_rows)

    def _remove_lcf_component_row(row_state: dict[str, Any]) -> None:
        if len(lcf_component_rows) <= 1:
            return
        if row_state in lcf_component_rows:
            lcf_component_rows.remove(row_state)
        _rebuild_lcf_components_box()
        _sync_lcf_remove_buttons()
        schedule_lcf_preview()

    def _sync_lcf_remove_buttons() -> None:
        disable_remove = len(lcf_component_rows) <= 1
        for row in lcf_component_rows:
            row["remove_btn"].disabled = disable_remove

    def _make_lcf_component_row(
        *,
        material: str = "",
        ref_label: str | None = None,
        initial_pct: float = 50.0,
        min_pct: float = 0.0,
        max_pct: float = 100.0,
        fixed: bool = False,
    ) -> dict[str, Any]:
        ref_opts = _lcf_ref_options(exclude_target=True)
        material_widget = widgets.Text(
            value=material,
            placeholder="Material",
            layout=widgets.Layout(width="72px"),
        )
        ref_dropdown = widgets.Dropdown(
            options=ref_opts,
            layout=widgets.Layout(width="100%", min_width="120px"),
        )
        if ref_label and ref_label in ref_opts:
            ref_dropdown.value = ref_label
        elif ref_opts:
            ref_dropdown.value = ref_opts[0]
        initial_widget = _bounded_float_text(initial_pct, 0.0, 100.0, width="48px")
        min_widget = _bounded_float_text(min_pct, 0.0, 100.0, width="44px")
        max_widget = _bounded_float_text(max_pct, 0.0, 100.0, width="44px")
        fixed_widget = widgets.Checkbox(value=fixed, description="Fix", indent=False)
        remove_btn = widgets.Button(description="Remove", layout=widgets.Layout(width="72px"))
        header = widgets.HBox(
            [
                widgets.HTML(
                    "<span style='font-size:10px;color:#666;width:72px'>Material</span>"
                ),
                widgets.HTML(
                    "<span style='font-size:10px;color:#666;flex:1;min-width:120px'>Reference</span>"
                ),
                widgets.HTML("<span style='font-size:10px;color:#666;width:48px'>Init%</span>"),
                widgets.HTML("<span style='font-size:10px;color:#666;width:44px'>Min%</span>"),
                widgets.HTML("<span style='font-size:10px;color:#666;width:44px'>Max%</span>"),
                widgets.HTML("<span style='font-size:10px;color:#666;width:52px'>Fixed</span>"),
                widgets.HTML("<span style='font-size:10px;color:#666;width:72px'></span>"),
            ],
            layout=widgets.Layout(width="100%"),
        )
        row_box = widgets.HBox(
            [
                material_widget,
                ref_dropdown,
                initial_widget,
                min_widget,
                max_widget,
                fixed_widget,
                remove_btn,
            ],
            layout=widgets.Layout(width="100%", align_items="center", gap="4px"),
        )
        row_state: dict[str, Any] = {
            "material": material_widget,
            "ref_dropdown": ref_dropdown,
            "initial_pct": initial_widget,
            "min_pct": min_widget,
            "max_pct": max_widget,
            "fixed": fixed_widget,
            "remove_btn": remove_btn,
            "row_box": widgets.VBox([row_box], layout=widgets.Layout(width="100%")),
        }
        if not lcf_component_rows:
            row_state["row_box"] = widgets.VBox(
                [header, row_box],
                layout=widgets.Layout(width="100%"),
            )

        def _on_row_change(_: Any) -> None:
            schedule_lcf_preview()

        for widget in (
            material_widget,
            ref_dropdown,
            initial_widget,
            min_widget,
            max_widget,
            fixed_widget,
        ):
            widget.observe(_on_row_change, names="value")

        def _on_remove(_: Any) -> None:
            _remove_lcf_component_row(row_state)

        remove_btn.on_click(_on_remove)
        return row_state

    def _ensure_lcf_component_rows(default_refs: list[str] | None = None) -> None:
        if lcf_component_rows:
            return
        refs = default_refs or _lcf_ref_options(exclude_target=True)
        if len(refs) >= 2:
            lcf_component_rows.append(
                _make_lcf_component_row(ref_label=refs[0], initial_pct=50.0)
            )
            lcf_component_rows.append(
                _make_lcf_component_row(ref_label=refs[1], initial_pct=50.0)
            )
        elif len(refs) == 1:
            lcf_component_rows.append(
                _make_lcf_component_row(ref_label=refs[0], initial_pct=100.0)
            )
        else:
            lcf_component_rows.append(_make_lcf_component_row(initial_pct=50.0))
        _rebuild_lcf_components_box()
        _sync_lcf_remove_buttons()

    def _collect_lcf_components() -> (
        tuple[list[Spectrum], list[str], np.ndarray, list[tuple[float, float]], list[bool]] | None
    ):
        if not lcf_component_rows:
            return None
        ref_specs: list[Spectrum] = []
        material_labels: list[str] = []
        initial_fracs: list[float] = []
        bounds: list[tuple[float, float]] = []
        fixed_flags: list[bool] = []
        for row in lcf_component_rows:
            ref_label = row["ref_dropdown"].value
            if not ref_label:
                return None
            entry = _entry_by_label(str(ref_label))
            if entry is None:
                return None
            material = (row["material"].value or "").strip() or str(ref_label)
            init_pct = float(row["initial_pct"].value or 0.0)
            min_pct = float(row["min_pct"].value or 0.0)
            max_pct = float(row["max_pct"].value or 100.0)
            if min_pct > max_pct:
                min_pct, max_pct = max_pct, min_pct
            ref_specs.append(_spectrum_from_entry(entry))
            material_labels.append(material)
            initial_fracs.append(init_pct / 100.0)
            bounds.append((min_pct / 100.0, max_pct / 100.0))
            fixed_flags.append(bool(row["fixed"].value))
        return (
            ref_specs,
            material_labels,
            np.asarray(initial_fracs, dtype=np.float64),
            bounds,
            fixed_flags,
        )

    def refresh_lcf_catalog(_: Any = None):
        nonlocal lcf_catalog
        lcf_catalog = build_lcf_catalog()
        labels = [e["label"] for e in lcf_catalog]
        lcf_target_dropdown.options = labels
        can_run = len(labels) >= 2
        lcf_run_btn.disabled = not can_run
        lcf_add_component_btn.disabled = not can_run
        if labels:
            if lcf_target_dropdown.value not in labels:
                lcf_target_dropdown.value = labels[0]
            default_refs = [lbl for lbl in labels if lbl != lcf_target_dropdown.value]
            _ensure_lcf_component_rows(default_refs[:2] if default_refs else None)
            _sync_lcf_component_dropdowns()
            lcf_status.value = (
                "<span style='font-size:11px;color:#066'>"
                f"{len(labels)} spectrum(a) available. Adjust components or Run LCF."
                "</span>"
            )
            schedule_lcf_preview()
        else:
            lcf_target_dropdown.value = None
            lcf_component_rows.clear()
            _rebuild_lcf_components_box()
            lcf_status.value = (
                "<span style='font-size:11px;color:#888'>"
                "Load parquet, export a scan, or set store root, then refresh."
                "</span>"
            )
            clear_lcf_plot()

    def _entry_by_label(label: str) -> dict[str, Any] | None:
        for entry in lcf_catalog:
            if entry["label"] == label:
                return entry
        return None

    def _spectrum_from_entry(entry: dict[str, Any]) -> Spectrum:
        energy_axis = np.asarray(entry["energy"], dtype=np.float64)
        od, od_err = _catalog_od(energy_axis, entry["od"], entry["od_err"])
        return Spectrum(
            energy_eV=energy_axis,
            OD=od,
            OD_err=od_err,
            label=str(entry["label"]),
        )

    def clear_lcf_plot() -> None:
        for artist in lcf_line_artists[:]:
            artist.remove()
        lcf_line_artists.clear()
        for coll in lcf_fill_artists[:]:
            coll.remove()
        lcf_fill_artists.clear()
        for ax in (ax_lcf_fit, ax_lcf_resid):
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.set_xlabel("")
        ax_lcf_resid.set_xlabel("Energy (eV)")
        fig_lcf.canvas.draw_idle()

    def _apply_lcf_axis_limits(grid: np.ndarray) -> None:
        if grid.size == 0:
            return
        e_lo, e_hi = float(np.nanmin(grid)), float(np.nanmax(grid))
        ax_lcf_fit.set_xlim(e_lo, e_hi)
        ax_lcf_resid.set_xlim(e_lo, e_hi)
        ax_lcf_fit.set_xlabel("Energy (eV)")
        ax_lcf_resid.set_xlabel("Energy (eV)")
        autoscale_y_with_margin(ax_lcf_fit)
        autoscale_y_with_margin(ax_lcf_resid)

    def _clear_lcf_axis_artists() -> None:
        for artist in lcf_line_artists[:]:
            artist.remove()
        lcf_line_artists.clear()
        for coll in lcf_fill_artists[:]:
            coll.remove()
        lcf_fill_artists.clear()
        for ax in (ax_lcf_fit, ax_lcf_resid):
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    def update_lcf_preview_plot() -> None:
        target_label = lcf_target_dropdown.value
        if not target_label or not lcf_catalog:
            clear_lcf_plot()
            return
        target_entry = _entry_by_label(str(target_label))
        if target_entry is None:
            clear_lcf_plot()
            return
        collected = _collect_lcf_components()
        if collected is None:
            clear_lcf_plot()
            return
        ref_specs, _material_labels, initial_fracs, _bounds, _fixed_flags = collected
        target_spec = _spectrum_from_entry(target_entry)
        try:
            grid, model, y_t = preview_lcf_model(
                target_spec,
                ref_specs,
                initial_fracs,
                normalize_fractions=True,
            )
        except ValueError:
            clear_lcf_plot()
            return
        _clear_lcf_axis_artists()
        if grid.size == 0:
            clear_lcf_plot()
            return
        ln_t, = ax_lcf_fit.plot(grid, y_t, color="black", lw=1.4, label="Target")
        ln_m, = ax_lcf_fit.plot(
            grid,
            model,
            color="darkorange",
            lw=1.2,
            ls="--",
            label="Preview",
        )
        lcf_line_artists.extend([ln_t, ln_m])
        ax_lcf_resid.cla()
        ax_lcf_resid.grid(True, alpha=0.25)
        style_axes(ax_lcf_resid)
        ax_lcf_resid.set_ylabel("Residual")
        make_draggable_legend(ax_lcf_fit)
        _apply_lcf_axis_limits(grid)
        fig_lcf.canvas.draw_idle()

    def schedule_lcf_preview(_: Any = None) -> None:
        timer = lcf_preview_timer[0]
        if timer is not None:
            timer.cancel()
        lcf_preview_timer[0] = threading.Timer(0.3, update_lcf_preview_plot)
        lcf_preview_timer[0].start()

    def update_lcf_plot(
        result: Any,
        target: Spectrum,
        references: list[Spectrum],
        material_labels: list[str] | None = None,
    ) -> None:
        _clear_lcf_axis_artists()
        grid = np.asarray(result.energy_grid, dtype=np.float64)
        if grid.size == 0:
            clear_lcf_plot()
            return
        y_t = interpolate_spectrum(target.energy_eV, target.OD, grid)
        model = np.zeros_like(grid, dtype=np.float64)
        display_labels = material_labels or [ref.label for ref in references]
        for frac, ref, label in zip(result.fractions, references, display_labels):
            y_r = interpolate_spectrum(ref.energy_eV, ref.OD, grid)
            model += float(frac) * y_r
            ln, = ax_lcf_fit.plot(grid, y_r, ls="--", lw=1.0, alpha=0.85, label=label)
            lcf_line_artists.append(ln)
        ln_t, = ax_lcf_fit.plot(grid, y_t, color="black", lw=1.4, label="Target")
        ln_m, = ax_lcf_fit.plot(grid, model, color="crimson", lw=1.3, label="Fit")
        lcf_line_artists.extend([ln_t, ln_m])
        resid = np.asarray(result.residual, dtype=np.float64)
        ax_lcf_resid.cla()
        ax_lcf_resid.grid(True, alpha=0.25)
        style_axes(ax_lcf_resid)
        ax_lcf_resid.set_ylabel("Residual")
        ln_r, = ax_lcf_resid.plot(grid, resid, color="0.35", lw=1.1, label="Residual")
        lcf_line_artists.append(ln_r)
        ax_lcf_resid.axhline(0.0, color="0.6", lw=0.8, ls=":")
        make_draggable_legend(ax_lcf_fit)
        _apply_lcf_axis_limits(grid)
        fig_lcf.canvas.draw_idle()

    def on_run_lcf(_: Any):
        target_label = lcf_target_dropdown.value
        if not target_label:
            lcf_status.value = "<span style='font-size:11px;color:#c00'>Select a target spectrum.</span>"
            return
        target_entry = _entry_by_label(str(target_label))
        if target_entry is None:
            lcf_status.value = "<span style='font-size:11px;color:#c00'>Target not in catalog.</span>"
            return
        collected = _collect_lcf_components()
        if collected is None:
            lcf_status.value = (
                "<span style='font-size:11px;color:#c00'>"
                "Each component needs a reference spectrum."
                "</span>"
            )
            return
        ref_specs, material_labels, initial_fracs, bounds, fixed_flags = collected
        all_free = not any(fixed_flags)
        sum_to_one = bool(lcf_sum_to_one.value) and all_free
        target_spec = _spectrum_from_entry(target_entry)
        try:
            fit = fit_lcf(
                target_spec,
                ref_specs,
                non_negative=bool(lcf_nonneg.value),
                sum_to_one=sum_to_one,
                initial_fractions=initial_fracs,
                fraction_bounds=bounds,
                fixed=fixed_flags,
            )
        except Exception as exc:
            lcf_status.value = f"<span style='font-size:11px;color:#c00'>{exc}</span>"
            return
        lcf_last_fit[0] = fit
        update_lcf_plot(fit, target_spec, ref_specs, material_labels)
        parts = [
            f"<b>{lbl}</b>: {100.0 * float(frac):.1f}%"
            for lbl, frac in zip(material_labels, fit.fractions)
        ]
        parts.append(f"reduced chi2 = {fit.reduced_chi_square:.4g}")
        lcf_status.value = "<span style='font-size:11px;color:#066'>" + "<br>".join(parts) + "</span>"
        frac_sum = float(np.sum(np.asarray(fit.fractions, dtype=np.float64)))
        resid = np.asarray(fit.residual, dtype=np.float64)
        rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
        lcf_metrics.value = _render_lcf_metrics(
            chi=f"{fit.reduced_chi_square:.3g}",
            n_comp=str(len(material_labels)),
            frac_sum=f"{frac_sum:.3f}",
            rms=f"{rms:.4f}",
        )

    def on_add_lcf_component(_: Any):
        refs = _lcf_ref_options(exclude_target=True)
        ref_label = refs[0] if refs else None
        lcf_component_rows.append(_make_lcf_component_row(ref_label=ref_label, initial_pct=0.0))
        _rebuild_lcf_components_box()
        _sync_lcf_remove_buttons()
        schedule_lcf_preview()

    def on_lcf_target_change(_: Any):
        _sync_lcf_component_dropdowns()
        schedule_lcf_preview()

    lcf_add_component_btn.on_click(on_add_lcf_component)
    lcf_target_dropdown.observe(on_lcf_target_change, names="value")
    lcf_run_btn.on_click(on_run_lcf)

    def on_norm_controls_change(_: Any = None):
        update_views_plot()
        update_browser_plot()
        refresh_lcf_catalog()

    norm_mode_dropdown.observe(on_norm_controls_change, names="value")
    spectrum_basis_dropdown.observe(on_norm_controls_change, names="value")
    browser_refresh_btn.on_click(refresh_browser)
    browser_display_mode.observe(on_browser_display_change, names="value")
    browser_mass_abs_fit.observe(lambda c: update_browser_plot(), names="value")
    browser_show_bare_atom.observe(lambda c: update_browser_plot(), names="value")
    sync_browser_step_edge_ui()

    bl = browser_row_label

    browser_controls_row = widgets.HBox(
        [
            widgets.VBox(
                [widgets.HTML(value=f"<span style='{bl}'>Y-axis</span>"), browser_display_mode],
                layout=widgets.Layout(min_width="180px", flex="1 1 180px"),
            ),
            widgets.VBox(
                [widgets.HTML(value=f"<span style='{bl}'>Bare-atom fit</span>"), browser_mass_abs_fit],
                layout=widgets.Layout(min_width="200px", flex="1 1 200px"),
            ),
            widgets.VBox(
                [
                    widgets.HTML(value=f"<span style='{bl}'>Step edge</span>"),
                    widgets.HBox(
                        [browser_show_bare_atom, browser_step_edge_hint],
                        layout=widgets.Layout(align_items="center", min_height="32px"),
                    ),
                ],
                layout=widgets.Layout(min_width="160px", flex="1 1 160px"),
            ),
        ],
        layout=widgets.Layout(width="100%", flex_wrap="wrap", align_items="flex-end", gap="10px"),
    )
    browser_scan_wrapper = widgets.VBox(
        [browser_scan_box],
        layout=widgets.Layout(
            width="100%",
            min_height="80px",
            max_height="240px",
            overflow_y="auto",
            padding="2px 0 0 0",
        ),
    )
    browser_top_bar = _dashboard_header(
        widgets.HTML(
            value=(
                "<span style='font-weight:600;font-size:11px;color:#24292f;"
                "letter-spacing:0.02em;'>Experiment dashboard</span>"
            )
        ),
        _split_row(
            _dashboard_column(
                _dashboard_labeled_field(
                    "Parent directory",
                    _dashboard_path_row(dir_text, refresh_btn),
                ),
                _dashboard_labeled_field("Experiment", experiment_dropdown),
            ),
            _dashboard_column(
                _dashboard_labeled_field(
                    "Parquet file",
                    _dashboard_path_row(parquet_path_text, browser_refresh_btn),
                ),
                _dashboard_store_edge_row(store_root_text, edge_text),
            ),
            gap="14px",
        ),
        widgets.VBox(
            [browser_status],
            layout=widgets.Layout(width="100%", margin="8px 0 0 0"),
        ),
    )
    preview_sidebar = _panel_column(
        [
            widgets.HTML(value=f"<span style='{bl}'>Sample</span>"),
            browser_sample_dropdown,
            widgets.HTML(value=f"<span style='{bl}'>Region</span>"),
            browser_region_dropdown,
            widgets.HTML(
                value="<span style='font-weight:600;font-size:10px;color:#24292f;margin-top:8px;'>Scans (select to plot)</span>"
            ),
            browser_scan_wrapper,
        ],
        width="240px",
        max_height="480px",
    )
    preview_figure_column = widgets.VBox(
        [browser_controls_row, fig_browser.canvas],
        layout=widgets.Layout(flex="1 1 auto", min_width="0", width="100%"),
    )
    tab_preview_spectra = widgets.VBox(
        [_split_row(preview_sidebar, preview_figure_column)],
        layout=widgets.Layout(width="100%", padding="0 10px 8px 10px"),
    )
    lcf_table_card = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.HTML(
                        value="<span style='font-size:11px;font-weight:600;color:#24292f'>Film components</span>"
                    ),
                    lcf_target_dropdown,
                ],
                layout=widgets.Layout(
                    width="100%", align_items="center", gap="12px", flex_flow="row wrap"
                ),
            ),
            lcf_components_box,
            widgets.HBox(
                [lcf_add_component_btn, lcf_nonneg, lcf_sum_to_one, lcf_run_btn],
                layout=widgets.Layout(
                    width="100%", align_items="center", gap="12px", flex_flow="row wrap"
                ),
            ),
            lcf_status,
        ],
        layout=widgets.Layout(
            width="100%",
            border="1px solid #e1e4e8",
            border_radius="8px",
            padding="10px 12px",
            gap="8px",
        ),
    )
    tab_lc_fitting = widgets.VBox(
        [lcf_table_card, lcf_metrics, _full_width_canvas_row(fig_lcf.canvas)],
        layout=widgets.Layout(width="100%", padding="0 10px 8px 10px", gap="6px"),
    )
    settings_accordion = widgets.Accordion(children=[browser_top_bar])
    settings_accordion.set_title(
        0, "Session settings · parent directory, experiment, store, edge"
    )
    settings_accordion.selected_index = None
    settings_accordion.layout = widgets.Layout(width="100%")
    def on_experiment_dashboard_refresh(change: Any):
        if change.get("new") in (None, "(no experiments)", "(empty)"):
            return
        refresh_browser()

    experiment_dropdown.observe(on_experiment_dashboard_refresh, names="value")
    store_root_text.observe(lambda c: refresh_lcf_catalog(), names="value")

    refresh_browser()
    update_od()
    if file_dropdown.value and file_dropdown.value != "(empty)":
        do_load()
    else:
        refresh_lcf_catalog()

    tabs = widgets.Tab(children=[tab_ingestion, tab_preview_spectra, tab_lc_fitting])
    tabs.set_title(0, "Ingest")
    tabs.set_title(1, "Browse")
    tabs.set_title(2, "Fit")

    main_col = widgets.VBox(
        [settings_accordion, tabs],
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )
    try_load_config_from_experiment_dir()
    display(main_col)
    plt.ion()
