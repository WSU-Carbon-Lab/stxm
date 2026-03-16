import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stxm.io import load_stxm, list_nexafs_line_scans
from stxm.regions import sample_izero_masks, bar_bounds_from_three_regions
from stxm.nexafs import nexafs_beer_lambert
from stxm.absorption import (
    HC_EV_CM,
    mass_absorption_cm2_per_g,
    fit_bare_atom_background,
    od_to_beta,
)
from stxm.experiment import append_nexafs_to_experiment


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
) -> None:
    paxis = meta["paxis_points"]
    qaxis = meta["qaxis_points"]
    extent = [paxis[0], paxis[-1], qaxis[-1], qaxis[0]]
    im_artist.set_data(image)
    im_artist.set_extent(extent)
    vmin = float(np.nanmin(image))
    vmax = float(np.nanmax(image))
    if vmax <= vmin:
        vmax = vmin + 1.0
    im_artist.set_clim(vmin, vmax)
    ax_im.set_xlim(paxis[0], paxis[-1])
    ax_im.set_ylim(qaxis[-1], qaxis[0])
    line_c.set_ydata([izero_lo, izero_lo])
    line_d.set_ydata([izero_hi, izero_hi])
    for (r_lo, r_hi), reg in zip(region_lines, regions):
        r_lo.set_ydata([reg["sample_lo"], reg["sample_lo"]])
        r_hi.set_ydata([reg["sample_hi"], reg["sample_hi"]])


def line_scan_processor(
    parent_directory: str | Path,
    sample_config: dict | None = None,
):
    """
    Tabbed line-scan processor (ipywidgets + matplotlib). Deprecated: use
    stxm.ui_panel.line_scan_processor or stxm.line_scan_processor for the Panel-based UI.
    sample_config: optional map sample_label -> chemical formula (str or null for blends).
    """
    from IPython.display import display
    import ipywidgets as widgets

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
    region_colors = ["green", "cyan", "orange", "magenta", "lime", "yellow"]

    extent = [paxis[0], paxis[-1], qaxis[-1], qaxis[0]]
    plt.ioff()
    fig, (ax_im, ax_od) = plt.subplots(
        2, 1, figsize=(8, 7), height_ratios=[1.2, 1], sharex=True
    )
    im_artist = ax_im.imshow(
        image,
        aspect="auto",
        extent=extent,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
    )
    ax_im.set_ylabel(meta.get("qaxis_name", "Sample"))
    ax_im.set_xlim(paxis[0], paxis[-1])
    ax_im.set_ylim(qaxis[-1], qaxis[0])

    line_c = ax_im.axhline(bar_izero_lo, color="blue", lw=2, picker=5)
    line_d = ax_im.axhline(bar_izero_hi, color="blue", lw=2, picker=5)
    region_lines: list[tuple] = []
    rl0 = ax_im.axhline(bar_sample_lo, color=region_colors[0], lw=2, picker=5)
    rl1 = ax_im.axhline(bar_sample_hi, color=region_colors[0], lw=2, picker=5)
    region_lines.append((rl0, rl1))

    ax_od.set_ylabel("OD (ln I0/I)")
    ax_od.set_xlabel(meta.get("paxis_name", "Energy (eV)"))
    ax_od.grid(True, alpha=0.3)

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
        for line in ax_od.lines[:]:
            line.remove()
        y_min, y_max = np.inf, -np.inf
        for idx, reg in enumerate(state["regions"]):
            sa_lo, sa_hi = reg["sample_lo"], reg["sample_hi"]
            sample_mask, _ = sample_izero_masks(qaxis_u, sa_lo, sa_hi, iz_lo, iz_hi)
            if not np.any(sample_mask):
                continue
            od, sigma_od, I0, sigma_I0, I, sigma_I, n_sample, n_izero = nexafs_beer_lambert(
                image_u, sample_mask, izero_mask
            )
            label = (reg.get("spot_label") or f"spot{idx + 1}").strip() or f"spot{idx + 1}"
            last_nexafs["regions"].append({
                "OD": od,
                "OD_err": sigma_od,
                "I0": I0,
                "I0_err": sigma_I0,
                "I": I,
                "I_err": sigma_I,
                "n_sample": n_sample,
                "n_izero": n_izero,
                "spot_label": label,
                "sample_lo": sa_lo,
                "sample_hi": sa_hi,
            })
            color = region_colors[idx % len(region_colors)]
            ax_od.plot(energy, od, color=color, lw=1, label=label)
            if np.any(np.isfinite(od)):
                y_min = min(y_min, np.nanmin(od))
                y_max = max(y_max, np.nanmax(od))
        if last_nexafs["regions"]:
            ax_od.legend(loc="best", fontsize=8)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.1 * (y_max - y_min + 1e-10)
                ax_od.set_ylim(y_min - pad, y_max + pad)
        ax_od.set_ylabel("OD (ln I0/I)")
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
        drag = state["dragging"]
        if drag == ("izero", "lo"):
            state["izero_lo"] = np.clip(y, y_lo, state["izero_hi"] - margin)
            line_c.set_ydata([state["izero_lo"], state["izero_lo"]])
        elif drag == ("izero", "hi"):
            state["izero_hi"] = np.clip(y, state["izero_lo"] + margin, y_hi)
            line_d.set_ydata([state["izero_hi"], state["izero_hi"]])
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
        fig.canvas.draw_idle()

    def on_release(event):
        if state["dragging"] is None:
            return
        state["dragging"] = None
        update_od()
        fig.canvas.draw_idle()

    state: dict[str, Any] = {
        "dragging": None,
        "izero_lo": bar_izero_lo,
        "izero_hi": bar_izero_hi,
        "regions": [
            {"sample_lo": bar_sample_lo, "sample_hi": bar_sample_hi, "spot_label": "spot1"}
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
    experiment_names = sorted(
        [d.name for d in parent_dir.iterdir() if d.is_dir()]
    ) if parent_dir.is_dir() else []
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
        layout=widgets.Layout(flex="1", min_width="120px"),
    )
    refresh_btn = widgets.Button(description="Refresh", tooltip="Refresh experiment and file lists")

    def list_valid_line_scan_files(experiment_path):
        hdrs = list_nexafs_line_scans(experiment_path)
        return [p.name for p in hdrs]

    def refresh_file_list():
        opts = list_valid_line_scan_files(current_dir[0])
        file_dropdown.options = opts if opts else ["(empty)"]
        if opts:
            file_dropdown.value = opts[0]

    def refresh_experiment_list():
        p = Path(dir_text.value.strip()).resolve()
        if not p.is_dir():
            return
        parent_dir_holder[0] = p
        exp_names = sorted([d.name for d in p.iterdir() if d.is_dir()])
        experiment_dropdown.options = exp_names if exp_names else ["(no experiments)"]
        experiment_dropdown.value = exp_names[0] if exp_names else None
        if exp_names:
            current_dir[0] = parent_dir_holder[0] / exp_names[0]
        else:
            current_dir[0] = parent_dir_holder[0]
        dir_text.value = str(parent_dir_holder[0])
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
        layout=widgets.Layout(width="100%"),
    )
    config_json_text = widgets.Text(
        value="",
        placeholder="Path to sample config JSON",
        layout=widgets.Layout(width="100%"),
    )
    load_config_btn = widgets.Button(description="Load config JSON", tooltip="Map sample_label -> formula")

    def resolve_parquet_path(path_str: str) -> Path:
        pp = Path((path_str or "").strip())
        if not pp.is_absolute() and current_dir:
            return current_dir[0] / pp
        return pp

    def on_load_config(_):
        nonlocal config_map
        path = (config_json_text.value or "").strip()
        if not path:
            status_label.value = "<span style='font-size:11px;color:#c00'>Set JSON path.</span>"
            return
        try:
            with open(path) as f:
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

    film_region_text = widgets.Text(
        value="",
        placeholder="Film region (defaults to sample name)",
        layout=widgets.Layout(width="100%"),
    )

    def try_load_config_from_experiment_dir():
        exp_dir = current_dir[0]
        if not isinstance(exp_dir, Path) or not exp_dir.is_dir():
            return
        config_path = exp_dir / "config.json"
        if config_path.is_file():
            config_json_text.value = str(config_path)
            on_load_config(None)

    export_btn = widgets.Button(description="Export", tooltip="Append current scan to parquet")
    refresh_plots_btn = widgets.Button(description="Refresh plots", tooltip="Redraw if canvas glitches")
    status_label = widgets.HTML(value="<span style='font-size:11px;color:#666'>Select a file.</span>")
    region_spot_widgets: list = []
    region_spot_rows: list = []
    add_region_btn = widgets.Button(description="Add region", tooltip="Add another sample region and plot its OD", button_style="primary")

    def sync_spot_label(i: int, change: Any):
        if 0 <= i < len(state["regions"]):
            state["regions"][i]["spot_label"] = (change.get("new") or "").strip() or f"spot{i + 1}"

    def make_spot_row(spot_value: str, idx: int):
        spot_w = widgets.Text(
            value=spot_value,
            placeholder="Spot label",
            layout=widgets.Layout(flex="1", min_width="80px"),
        )
        spot_w.observe(lambda c, i=idx: sync_spot_label(i, c), names="value")
        edit_btn = widgets.Button(
            description="\u270e",
            tooltip="Edit spot name",
            button_style="info",
            layout=widgets.Layout(width="36px", min_width="36px"),
        )
        remove_btn = widgets.Button(
            description="\U0001f5d1",
            tooltip="Remove this region",
            button_style="danger",
            layout=widgets.Layout(width="36px", min_width="36px"),
        )
        def on_remove(btn, i=idx):
            if i < 0 or i >= len(state["regions"]) or len(state["regions"]) <= 1:
                return
            state["regions"].pop(i)
            l_lo, l_hi = region_lines.pop(i)
            l_lo.remove()
            l_hi.remove()
            region_spot_widgets.clear()
            region_spot_rows.clear()
            for j, reg in enumerate(state["regions"]):
                spot_w, row = make_spot_row(reg.get("spot_label") or f"spot{j + 1}", j)
                region_spot_widgets.append(spot_w)
                region_spot_rows.append(row)
            region_spot_box.children = tuple(region_spot_rows)
            update_od()
            fig.canvas.draw_idle()
        remove_btn.on_click(on_remove)
        row = widgets.HBox(
            [spot_w, edit_btn, remove_btn],
            layout=widgets.Layout(width="100%", align_items="center"),
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
        n = len(state["regions"]) + 1
        state["regions"].append({
            "sample_lo": sample_lo,
            "sample_hi": sample_hi,
            "spot_label": f"spot{n}",
        })
        idx = len(region_lines)
        color = region_colors[idx % len(region_colors)]
        rl0 = ax_im.axhline(sample_lo, color=color, lw=2, picker=5)
        rl1 = ax_im.axhline(sample_hi, color=color, lw=2, picker=5)
        region_lines.append((rl0, rl1))
        spot_w, row = make_spot_row(f"spot{n}", idx)
        region_spot_widgets.append(spot_w)
        region_spot_rows.append(row)
        region_spot_box.children = tuple(region_spot_rows)
        _apply_image_and_region_lines(
            im_artist, ax_im, data["image"], data["meta"],
            state["izero_lo"], state["izero_hi"], state["regions"], line_c, line_d, region_lines,
        )
        update_od()
        fig.canvas.draw_idle()

    add_region_btn.on_click(on_add_region)
    first_spot, first_row = make_spot_row("spot1", 0)
    region_spot_widgets.append(first_spot)
    region_spot_rows.append(first_row)
    region_spot_box = widgets.VBox([first_row], layout=widgets.Layout(width="100%"))

    display_mode = widgets.Dropdown(
        options=["OD", "Mass absorption (cm^2/g)", "Beta"],
        value="OD",
        description="Y-axis:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )
    mass_abs_fit_mode = widgets.Dropdown(
        options=[
            "Scale only (last 5 pts)",
            "Scale & offset (first+last 5)",
        ],
        value="Scale & offset (first+last 5)",
        description="Fit:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "30px"},
    )

    plt.ioff()
    fig_views, ax_views = plt.subplots(1, 1, figsize=(7, 3.2))
    ax_views.set_xlabel(meta.get("paxis_name", "Energy (eV)"))
    ax_views.grid(True, alpha=0.3)
    ax_views.set_ylabel("OD (ln I0/I)")
    views_line_artists: list = []

    def compute_display_curve(energy: np.ndarray, od: np.ndarray, sigma_od: np.ndarray):
        mode = (display_mode.value or "OD").strip()
        if mode == "OD":
            return od, sigma_od, "OD (ln I0/I)"
        if mode == "Mass absorption (cm^2/g)":
            cf = current_chemical_formula()
            if cf is None:
                return od, sigma_od, "OD (set sample in Reduction for mass abs)"
            try:
                mu_rho = mass_absorption_cm2_per_g(cf, energy, None)
                fit_opt = mass_abs_fit_mode.value or ""
                n_low = 5 if "offset" in fit_opt.lower() else 0
                n_high = 5
                scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
                scale = scale if scale != 0 else 1.0
                return (od - const) / scale, np.abs(sigma_od / scale), "mu/rho (cm^2/g)"
            except Exception:
                return od, sigma_od, "OD (ln I0/I)"
        try:
            t_cm = 1e-4
            y = od_to_beta(energy, od, t_cm)
            lam_cm = HC_EV_CM / np.asarray(energy, dtype=float)
            sig = np.asarray(sigma_od, dtype=float) * np.atleast_1d(lam_cm) / (4 * np.pi * t_cm)
            return y, sig, "beta (Im n)"
        except Exception:
            return od, sigma_od, "OD (ln I0/I)"

    def update_views_plot():
        for line in views_line_artists[:]:
            line.remove()
        views_line_artists.clear()
        regions_data = last_nexafs.get("regions") or []
        energy = last_nexafs.get("energy")
        if not regions_data or energy is None:
            ax_views.set_ylabel("OD (ln I0/I)")
            fig_views.canvas.draw_idle()
            return
        energy = np.asarray(energy)
        y_min, y_max = np.inf, -np.inf
        y_label = "OD (ln I0/I)"
        for idx, reg in enumerate(regions_data):
            od = np.asarray(reg["OD"])
            sigma_od = np.asarray(reg.get("OD_err", np.zeros_like(od)))
            y, sig, y_label = compute_display_curve(energy, od, sigma_od)
            label = (reg.get("spot_label") or f"spot{idx + 1}").strip() or f"spot{idx + 1}"
            color = region_colors[idx % len(region_colors)]
            ln, = ax_views.plot(energy, y, color=color, lw=1.2, label=label)
            views_line_artists.append(ln)
            if np.any(np.isfinite(y)):
                y_min = min(y_min, np.nanmin(y))
                y_max = max(y_max, np.nanmax(y))
        ax_views.set_ylabel(y_label)
        if views_line_artists:
            ax_views.legend(loc="best", fontsize=8)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.1 * (y_max - y_min + 1e-10)
                ax_views.set_ylim(y_min - pad, y_max + pad)
        ax_views.set_xlim(energy.min(), energy.max())
        fig_views.canvas.draw_idle()

    def on_views_display_change(change: Any):
        update_views_plot()

    display_mode.observe(on_views_display_change, names="value")
    mass_abs_fit_mode.observe(on_views_display_change, names="value")
    sample_dropdown.observe(on_views_display_change, names="value")

    current_scan_path: list[Optional[str]] = [None]


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
        bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi = bar_bounds_from_three_regions(
            image_new, qaxis
        )
        state["izero_lo"] = bar_izero_lo
        state["izero_hi"] = bar_izero_hi
        while len(region_lines) > 1:
            l_lo, l_hi = region_lines.pop()
            l_lo.remove()
            l_hi.remove()
        state["regions"] = [
            {"sample_lo": bar_sample_lo, "sample_hi": bar_sample_hi, "spot_label": "spot1"}
        ]
        if region_spot_widgets:
            while len(region_spot_widgets) > 1:
                region_spot_widgets.pop()
                region_spot_rows.pop()
            region_spot_widgets[0].value = "spot1"
            region_spot_box.children = (region_spot_rows[0],)
        _apply_image_and_region_lines(
            im_artist,
            ax_im,
            image_new,
            meta_new,
            state["izero_lo"],
            state["izero_hi"],
            state["regions"],
            line_c,
            line_d,
            region_lines,
        )
        current_scan_path[0] = path
        paxis_pts = np.asarray(meta_new["paxis_points"])
        if paxis_pts.size >= 2:
            margin = 0.15 * float(paxis_pts[-1] - paxis_pts[0])
            state["pre_lo"] = float(paxis_pts[0])
            state["pre_hi"] = float(paxis_pts[0] + margin)
            state["post_lo"] = float(paxis_pts[-1] - margin)
            state["post_hi"] = float(paxis_pts[-1])
        status_label.value = f"<span style='font-size:11px;color:#066'>Loaded: {truncate_path(path)}</span>"
        update_od()
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
        fr = (film_region_text.value or "").strip() or sn
        cf = current_chemical_formula()
        energy = last_nexafs["energy"]
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
                append_nexafs_to_experiment(
                    parquet_path,
                    df,
                    sample_name=sn,
                    chemical_formula=cf,
                    spot_label=reg["spot_label"],
                    film_region_name=fr,
                    scan_path=current_scan_path[0],
                    formula=cf or "",
                )
            status_label.value = f"<span style='font-size:11px;color:#066'>Appended {len(regions_data)} region(s) to {parquet_path.name}</span>"
        except Exception as e:
            status_label.value = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    def on_file_select(change):
        if change.get("new") is None:
            return
        do_load()

    file_dropdown.observe(on_file_select, names="value")
    if file_dropdown.value and file_dropdown.value != "(empty)":
        do_load()
    export_btn.on_click(on_export)
    refresh_btn.on_click(refresh_experiment_list)

    def on_refresh_plots(_):
        update_od()
        fig.canvas.draw_idle()
        fig.canvas.draw()

    refresh_plots_btn.on_click(on_refresh_plots)

    ax_im.text(
        0.02, 0.98, "sample", transform=ax_im.transAxes, fontsize=10, va="top", color="green"
    )
    ax_im.text(
        0.02, 0.02, "izero", transform=ax_im.transAxes, fontsize=10, va="bottom", color="blue"
    )
    update_od()
    plt.tight_layout()
    canvas_layout = getattr(fig.canvas, "layout", None)
    if canvas_layout is not None:
        setattr(canvas_layout, "min_height", "420px")
        setattr(canvas_layout, "flex", "1")
        setattr(canvas_layout, "min_width", "480px")

    sec = "font-weight:600;font-size:12px;margin-top:8px;margin-bottom:4px;display:block"
    sec_sm = "font-size:11px;color:#555;margin-bottom:2px"
    tab_setup = widgets.VBox([
        widgets.HTML(value=f"<div style='{sec}'>Directory and experiment</div>"),
        widgets.HBox([dir_text, refresh_btn], layout=widgets.Layout(width="100%")),
        experiment_dropdown,
        widgets.HTML(value=f"<div style='{sec}'>Export file name</div>"),
        widgets.HTML(value=f"<span style='{sec_sm}'>Parquet path (default: experiment.parquet)</span>"),
        parquet_path_text,
        widgets.HTML(value=f"<div style='{sec}'>Sample config</div>"),
        widgets.HBox([config_json_text, load_config_btn], layout=widgets.Layout(width="100%")),
        widgets.HTML(value="<span style='font-size:11px'>JSON: {\"samples\": {\"ExpName\": \"C8H8\", \"Other\": null}}</span>"),
    ], layout=widgets.Layout(width="100%", padding="8px"))

    reduction_top = widgets.HBox([
        widgets.VBox([
            widgets.HTML(value=f"<div style='{sec_sm}'>Sample (from config)</div>"),
            sample_dropdown,
        ], layout=widgets.Layout(min_width="160px", flex="0 0 auto")),
        widgets.VBox([
            widgets.HTML(value=f"<span style='{sec_sm}'>Film region</span>"),
            film_region_text,
        ], layout=widgets.Layout(min_width="120px", flex="0 0 auto")),
        widgets.VBox([
            widgets.HTML(value=f"<div style='{sec_sm}'>Line scan</div>"),
            file_dropdown,
        ], layout=widgets.Layout(min_width="180px", flex="0 0 auto")),
        widgets.VBox([
            widgets.HTML(value=f"<div style='{sec_sm}'>Region spot labels</div>"),
            region_spot_box,
        ], layout=widgets.Layout(flex="1", min_width="120px")),
        widgets.VBox([
            widgets.HTML(value="<div style='height:20px'></div>"),
            add_region_btn,
        ], layout=widgets.Layout(flex="0 0 auto")),
    ], layout=widgets.Layout(width="100%", flex_wrap="wrap", align_items="flex-start"))
    tab_reduction = widgets.VBox([
        reduction_top,
        fig.canvas,
        widgets.HBox([refresh_plots_btn, status_label], layout=widgets.Layout(width="100%", margin="4px 0 0 0")),
    ], layout=widgets.Layout(width="100%", padding="8px"))

    views_canvas_layout = getattr(fig_views.canvas, "layout", None)
    if views_canvas_layout is not None:
        setattr(views_canvas_layout, "min_height", "280px")
        setattr(views_canvas_layout, "min_width", "400px")
    fig_views.tight_layout(pad=1.2)
    tab_views = widgets.VBox([
        widgets.HTML(value=f"<div style='{sec}'>Display</div>"),
        widgets.HBox([
            widgets.VBox([widgets.HTML(value=f"<span style='{sec_sm}'>Y-axis</span>"), display_mode], layout=widgets.Layout(min_width="140px")),
            widgets.VBox([widgets.HTML(value=f"<span style='{sec_sm}'>Mass abs fit</span>"), mass_abs_fit_mode], layout=widgets.Layout(min_width="160px")),
        ], layout=widgets.Layout(width="100%", flex_wrap="wrap")),
        fig_views.canvas,
        widgets.HTML(value=f"<div style='{sec}'>Export</div>"),
        export_btn,
    ], layout=widgets.Layout(width="100%", padding="8px"))

    tabs = widgets.Tab(children=[tab_setup, tab_reduction, tab_views])
    tabs.set_title(0, "Setup")
    tabs.set_title(1, "Reduction (OD)")
    tabs.set_title(2, "Views and export")

    main_col = widgets.VBox([tabs], layout=widgets.Layout(width="100%", align_items="stretch"))
    try_load_config_from_experiment_dir()
    display(main_col)
    plt.ion()

    def get_nexafs_dataframe() -> pd.DataFrame:
        regions_data = last_nexafs.get("regions") or []
        if not regions_data:
            return pd.DataFrame()
        reg = regions_data[0]
        energy = last_nexafs["energy"]
        return pd.DataFrame({
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

    def set_sample_config(d: dict) -> None:
        nonlocal config_map
        config_map = dict(d)
        apply_config_for_experiment(experiment_dropdown.value or "")

    setattr(get_nexafs_dataframe, "set_sample_config", set_sample_config)
    return get_nexafs_dataframe


interactive_izero_split = line_scan_processor
