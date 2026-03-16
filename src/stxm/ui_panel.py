import json
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from bokeh.models import ColumnDataSource, PointDrawTool, Range1d
from bokeh.plotting import figure

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


def line_scan_processor(
    parent_directory: str | Path,
    sample_config: dict | None = None,
):
    """
    Tabbed line-scan processor: experiment setup, OD reduction, views and export.
    Uses Panel for layout and widgets. sample_config: optional map sample_label -> chemical formula.
    Returns a callable that returns the current NEXAFS DataFrame (first region), with set_sample_config
    attached. In Jupyter the Panel UI is rendered as the cell output; call the return value for the dataframe.
    """
    pn.extension()
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
    data: dict[str, Any] = {"meta": placeholder_meta, "image": placeholder_image}
    meta, image = placeholder_meta, placeholder_image

    def truncate_path(s: str | Path, max_len: int = 56) -> str:
        s = str(s)
        return s if len(s) <= max_len else "..." + s[-(max_len - 3) :]

    qaxis = meta["qaxis_points"]
    paxis = meta["paxis_points"]
    bar_sample_lo, bar_sample_hi, bar_izero_lo, bar_izero_hi = bar_bounds_from_three_regions(
        image, qaxis
    )
    region_colors = ["green", "cyan", "orange", "magenta", "lime", "yellow"]

    last_nexafs: dict[str, Any] = {}
    config_map = dict(sample_config) if sample_config else {}

    parent_dir_holder = [parent_dir]
    experiment_names = sorted(
        [d.name for d in parent_dir.iterdir() if d.is_dir()]
    ) if parent_dir.is_dir() else []
    experiment_dropdown = pn.widgets.Select(
        name="",
        options=experiment_names if experiment_names else ["(no experiments)"],
        value=experiment_names[0] if experiment_names else None,
        sizing_mode="stretch_width",
    )
    _exp_val = experiment_dropdown.value
    current_dir: list[Path] = [
        parent_dir_holder[0] / _exp_val if _exp_val and experiment_names else parent_dir_holder[0]
    ]
    dir_text = pn.widgets.TextInput(
        value=str(parent_dir_holder[0]),
        placeholder="Parent directory of experiments",
        sizing_mode="stretch_width",
    )
    refresh_btn = pn.widgets.Button(name="Refresh", button_type="default")

    def list_valid_line_scan_files(experiment_path: Path) -> list[str]:
        hdrs = list_nexafs_line_scans(experiment_path)
        return [p.name for p in hdrs]

    def refresh_file_list() -> None:
        opts = list_valid_line_scan_files(current_dir[0])
        file_dropdown.options = opts if opts else ["(empty)"]
        if opts:
            file_dropdown.value = opts[0]

    def apply_config_for_experiment(exp_name: str) -> None:
        opts = [k for k in config_map.keys() if k]
        sample_dropdown.options = opts if opts else ["(no samples)"]
        if exp_name and exp_name in config_map:
            sample_dropdown.value = exp_name
        elif opts:
            sample_dropdown.value = opts[0]

    def current_chemical_formula() -> Optional[str]:
        val = sample_dropdown.value if sample_dropdown else None
        if not val or val in ("(load config first)", "(no samples)"):
            return None
        v = config_map.get(val)
        return str(v).strip() if v is not None and str(v).strip() else None

    opts_init = list_valid_line_scan_files(current_dir[0])
    file_dropdown = pn.widgets.Select(
        name="Line scan",
        options=opts_init if opts_init else ["(empty)"],
        value=opts_init[0] if opts_init else None,
        sizing_mode="stretch_width",
    )

    sample_dropdown = pn.widgets.Select(
        name="Sample",
        options=["(load config first)"],
        value="(load config first)",
        sizing_mode="stretch_width",
    )

    parquet_path_text = pn.widgets.TextInput(
        value="experiment.parquet",
        placeholder="e.g. experiment.parquet",
        sizing_mode="stretch_width",
    )
    config_json_text = pn.widgets.TextInput(
        value="",
        placeholder="Path to sample config JSON",
        sizing_mode="stretch_width",
    )
    load_config_btn = pn.widgets.Button(name="Load config JSON", button_type="default")

    def resolve_parquet_path(path_str: str) -> Path:
        pp = Path((path_str or "").strip())
        if not pp.is_absolute() and current_dir:
            return current_dir[0] / pp
        return pp

    status_label = pn.pane.HTML(
        "<span style='font-size:11px;color:#666'>Select a file.</span>",
        sizing_mode="stretch_width",
    )

    def on_experiment_change(event: Any) -> None:
        new_val = experiment_dropdown.value
        if new_val is None or new_val == "(no experiments)" or new_val == "(empty)":
            return
        current_dir[0] = parent_dir_holder[0] / str(new_val)
        refresh_file_list()
        apply_config_for_experiment(str(new_val))
        if sample_dropdown.value in ("(load config first)", "(no samples)") and config_map and str(new_val) in config_map:
            sample_dropdown.value = str(new_val)
        try_load_config_from_experiment_dir()
        if file_dropdown.value and file_dropdown.value != "(empty)":
            do_load()

    experiment_dropdown.param.watch(on_experiment_change, "value")

    state: dict[str, Any] = {
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

    data_trigger = pn.widgets.TextInput(value="0", visible=False)

    qaxis_min, qaxis_max = float(np.min(qaxis)), float(np.max(qaxis))
    izero_lo_slider = pn.widgets.FloatSlider(
        name="Izero lo",
        start=qaxis_min,
        end=qaxis_max,
        value=bar_izero_lo,
        sizing_mode="stretch_width",
    )
    izero_hi_slider = pn.widgets.FloatSlider(
        name="Izero hi",
        start=qaxis_min,
        end=qaxis_max,
        value=bar_izero_hi,
        sizing_mode="stretch_width",
    )

    def _bump_trigger() -> None:
        data_trigger.value = str(int(data_trigger.value or "0") + 1)

    def _sync_izero_lo(event: Any) -> None:
        state["izero_lo"] = event.new
        _bump_trigger()

    def _sync_izero_hi(event: Any) -> None:
        state["izero_hi"] = event.new
        _bump_trigger()

    izero_lo_slider.param.watch(_sync_izero_lo, "value")
    izero_hi_slider.param.watch(_sync_izero_hi, "value")

    def update_od() -> None:
        meta_u = data["meta"]
        image_u = data["image"]
        qaxis_u = meta_u["qaxis_points"]
        paxis_u = meta_u["paxis_points"]
        iz_lo = state["izero_lo"]
        iz_hi = state["izero_hi"]
        _, izero_mask = sample_izero_masks(qaxis_u, 0.0, 0.0, iz_lo, iz_hi)
        if not np.any(izero_mask):
            _bump_trigger()
            return
        energy = np.asarray(paxis_u)
        last_nexafs["energy"] = energy
        last_nexafs["izero_lo"] = iz_lo
        last_nexafs["izero_hi"] = iz_hi
        last_nexafs["regions"] = []
        for idx, reg in enumerate(state["regions"]):
            sa_lo, sa_hi = reg["sample_lo"], reg["sample_hi"]
            sample_mask, _ = sample_izero_masks(qaxis_u, sa_lo, sa_hi, iz_lo, iz_hi)
            if not np.any(sample_mask):
                continue
            od, sigma_od, I0, sigma_I0, I_sig, sigma_I_sig, n_sample, n_izero = nexafs_beer_lambert(
                image_u, sample_mask, izero_mask
            )
            label = (reg.get("spot_label") or f"spot{idx + 1}").strip() or f"spot{idx + 1}"
            last_nexafs["regions"].append({
                "OD": od,
                "OD_err": sigma_od,
                "I0": I0,
                "I0_err": sigma_I0,
                "I": I_sig,
                "I_err": sigma_I_sig,
                "n_sample": n_sample,
                "n_izero": n_izero,
                "spot_label": label,
                "sample_lo": sa_lo,
                "sample_hi": sa_hi,
            })
        _bump_trigger()

    def refresh_experiment_list(_event: Any = None) -> None:
        p = Path(str(dir_text.value or "").strip())
        if not p:
            return
        p = p.resolve()
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

    def on_load_config(event: Any) -> None:
        nonlocal config_map
        path = (config_json_text.value or "").strip()
        if not path:
            status_label.object = "<span style='font-size:11px;color:#c00'>Set JSON path.</span>"
            return
        try:
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "samples" in raw:
                config_map = {str(k): v for k, v in raw["samples"].items()}
            else:
                config_map = {str(k): v for k, v in raw.items()}
            apply_config_for_experiment(str(experiment_dropdown.value or ""))
            status_label.object = "<span style='font-size:11px;color:#066'>Config loaded.</span>"
            opts = getattr(sample_dropdown, "options", None) or []
            if opts and opts[0] != "(no samples)":
                sample_dropdown.value = opts[0]
        except Exception as e:
            status_label.object = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    load_config_btn.on_click(on_load_config)

    film_region_text = pn.widgets.TextInput(
        value="",
        placeholder="Film region (defaults to sample name)",
        sizing_mode="stretch_width",
    )

    def try_load_config_from_experiment_dir() -> None:
        exp_dir = current_dir[0]
        if not isinstance(exp_dir, Path) or not exp_dir.is_dir():
            return
        config_path = exp_dir / "config.json"
        if config_path.is_file():
            config_json_text.value = str(config_path)
            on_load_config(None)

    export_btn = pn.widgets.Button(name="Export", button_type="primary")
    refresh_plots_btn = pn.widgets.Button(name="Refresh plots", button_type="default")

    region_spot_rows: list[pn.Row] = []
    region_spot_widgets: list[pn.widgets.TextInput] = []
    add_region_btn = pn.widgets.Button(name="Add region", button_type="primary")

    def sync_spot_label(i: int, event: Any) -> None:
        if 0 <= i < len(state["regions"]):
            state["regions"][i]["spot_label"] = (event.new or "").strip() or f"spot{i + 1}"

    def make_spot_row(spot_value: str, idx: int, sample_lo: float, sample_hi: float) -> tuple[pn.widgets.TextInput, pn.Row]:
        qaxis_a = data["meta"]["qaxis_points"]
        y_lo, y_hi = float(np.min(qaxis_a)), float(np.max(qaxis_a))
        spot_w = pn.widgets.TextInput(
            value=spot_value,
            placeholder="Spot label",
            sizing_mode="stretch_width",
        )
        spot_w.param.watch(lambda e, i=idx: sync_spot_label(i, e), "value")
        sample_lo_slider = pn.widgets.FloatSlider(
            name="lo",
            start=y_lo,
            end=y_hi,
            value=sample_lo,
            width=80,
            sizing_mode="fixed",
        )
        sample_hi_slider = pn.widgets.FloatSlider(
            name="hi",
            start=y_lo,
            end=y_hi,
            value=sample_hi,
            width=80,
            sizing_mode="fixed",
        )

        def sync_region_lo(e: Any, i: int = idx) -> None:
            if 0 <= i < len(state["regions"]):
                state["regions"][i]["sample_lo"] = e.new
                _bump_trigger()

        def sync_region_hi(e: Any, i: int = idx) -> None:
            if 0 <= i < len(state["regions"]):
                state["regions"][i]["sample_hi"] = e.new
                _bump_trigger()

        sample_lo_slider.param.watch(sync_region_lo, "value")
        sample_hi_slider.param.watch(sync_region_hi, "value")

        remove_btn = pn.widgets.Button(name="X", button_type="danger", width=36)

        def on_remove(event: Any, i: int = idx) -> None:
            if i < 0 or i >= len(state["regions"]) or len(state["regions"]) <= 1:
                return
            state["regions"].pop(i)
            region_spot_widgets.clear()
            region_spot_rows.clear()
            for j, reg in enumerate(state["regions"]):
                spot_w_j, row_j = make_spot_row(
                    reg.get("spot_label") or f"spot{j + 1}", j, reg["sample_lo"], reg["sample_hi"]
                )
                region_spot_widgets.append(spot_w_j)
                region_spot_rows.append(row_j)
            region_spot_column.objects = list(region_spot_rows)
            update_od()

        remove_btn.on_click(lambda e, i=idx: on_remove(e, i))
        row = pn.Row(spot_w, sample_lo_slider, sample_hi_slider, remove_btn, sizing_mode="stretch_width")
        return spot_w, row

    def on_add_region(event: Any) -> None:
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
        idx = len(state["regions"]) - 1
        spot_w, row = make_spot_row(f"spot{n}", idx, sample_lo, sample_hi)
        region_spot_widgets.append(spot_w)
        region_spot_rows.append(row)
        region_spot_column.objects = list(region_spot_rows)
        update_od()

    add_region_btn.on_click(on_add_region)
    first_spot, first_row = make_spot_row("spot1", 0, bar_sample_lo, bar_sample_hi)
    region_spot_widgets.append(first_spot)
    region_spot_rows.append(first_row)
    region_spot_column = pn.Column(first_row, sizing_mode="stretch_width")

    display_mode = pn.widgets.Select(
        name="Y-axis",
        options=["OD", "Mass absorption (cm^2/g)", "Beta"],
        value="OD",
        sizing_mode="stretch_width",
    )
    mass_abs_fit_mode = pn.widgets.Select(
        name="Fit",
        options=[
            "Scale only (last 5 pts)",
            "Scale & offset (first+last 5)",
        ],
        value="Scale & offset (first+last 5)",
        sizing_mode="stretch_width",
    )

    def _curve_from_display(
        energy: np.ndarray,
        od: np.ndarray,
        sigma_od: np.ndarray,
        mode_str: str,
        fit_opt: str,
        formula_str: str | None,
    ) -> tuple[np.ndarray, str]:
        if mode_str == "OD":
            return od, "OD (ln I0/I)"
        if mode_str == "Mass absorption (cm^2/g)":
            if not formula_str or not formula_str.strip():
                return od, "OD (set sample in Reduction for mass abs)"
            try:
                mu_rho = mass_absorption_cm2_per_g(formula_str, energy, None)
                n_low = 5 if "offset" in (fit_opt or "").lower() else 0
                n_high = 5
                scale, const, _, _ = fit_bare_atom_background(energy, od, mu_rho, n_low=n_low, n_high=n_high)
                scale = scale if scale != 0 else 1.0
                return (od - const) / scale, "mu/rho (cm^2/g)"
            except Exception:
                return od, "OD (ln I0/I)"
        try:
            t_cm = 1e-4
            y = od_to_beta(energy, od, t_cm)
            return y, "beta (Im n)"
        except Exception:
            return od, "OD (ln I0/I)"

    def compute_views_plot(
        display_mode_val: str,
        mass_abs_fit_mode_val: str,
        sample_val: str,
        _trigger: str,
    ) -> hv.Curve | hv.Overlay:
        regions_data = last_nexafs.get("regions") or []
        energy_arr = last_nexafs.get("energy")
        empty: hv.Curve = cast(hv.Curve, hv.Curve([], label="").opts(
            width=600, height=280, xlabel="Energy (eV)", ylabel="OD (ln I0/I)"
        ))
        if not regions_data or energy_arr is None:
            return empty
        energy = np.asarray(energy_arr)
        formula_str = None
        if sample_val and sample_val not in ("(load config first)", "(no samples)"):
            v = config_map.get(sample_val)
            formula_str = str(v).strip() if v is not None and str(v).strip() else None
        curve_list: list[hv.Curve] = []
        for idx, reg in enumerate(regions_data):
            od = np.asarray(reg["OD"])
            sigma_od = np.asarray(reg.get("OD_err", np.zeros_like(od)))
            y, _ = _curve_from_display(
                energy, od, sigma_od,
                (display_mode_val or "OD").strip(),
                mass_abs_fit_mode_val or "",
                formula_str,
            )
            label = (reg.get("spot_label") or f"spot{idx + 1}").strip() or f"spot{idx + 1}"
            color = region_colors[idx % len(region_colors)]
            cur = cast(hv.Curve, hv.Curve((energy, y), "Energy (eV)", "y", label=label).opts(color=color))
            curve_list.append(cur)
        if not curve_list:
            return empty
        out: hv.Curve | hv.Overlay = curve_list[0]
        for c in curve_list[1:]:
            out = out * c
        return cast(hv.Curve | hv.Overlay, out.opts(width=600, height=280, show_legend=True))

    current_scan_path: list[Optional[str]] = [None]

    def do_load() -> None:
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
        bar_sample_lo_new, bar_sample_hi_new, bar_izero_lo_new, bar_izero_hi_new = bar_bounds_from_three_regions(
            image_new, qaxis
        )
        state["izero_lo"] = bar_izero_lo_new
        state["izero_hi"] = bar_izero_hi_new
        state["regions"] = [
            {"sample_lo": bar_sample_lo_new, "sample_hi": bar_sample_hi_new, "spot_label": "spot1"}
        ]
        q_min, q_max = float(np.min(qaxis)), float(np.max(qaxis))
        izero_lo_slider.start = q_min
        izero_lo_slider.end = q_max
        izero_lo_slider.value = bar_izero_lo_new
        izero_hi_slider.start = q_min
        izero_hi_slider.end = q_max
        izero_hi_slider.value = bar_izero_hi_new
        region_spot_widgets.clear()
        region_spot_rows.clear()
        spot_w, row = make_spot_row("spot1", 0, bar_sample_lo_new, bar_sample_hi_new)
        region_spot_widgets.append(spot_w)
        region_spot_rows.append(row)
        region_spot_column.objects = [row]
        current_scan_path[0] = path
        paxis_pts = np.asarray(meta_new["paxis_points"])
        if paxis_pts.size >= 2:
            margin = 0.15 * float(paxis_pts[-1] - paxis_pts[0])
            state["pre_lo"] = float(paxis_pts[0])
            state["pre_hi"] = float(paxis_pts[0] + margin)
            state["post_lo"] = float(paxis_pts[-1] - margin)
            state["post_hi"] = float(paxis_pts[-1])
        status_label.object = f"<span style='font-size:11px;color:#066'>Loaded: {truncate_path(path)}</span>"
        update_od()

    def _add_derived_columns(
        df: pd.DataFrame, energy: np.ndarray, od: np.ndarray, od_err: np.ndarray
    ) -> pd.DataFrame:
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

    def on_export(event: Any) -> None:
        regions_data = last_nexafs.get("regions") or []
        if not regions_data:
            status_label.object = "<span style='font-size:11px;color:#c00'>Load a file and reduce first.</span>"
            return
        parquet_p = (parquet_path_text.value or "").strip()
        if not parquet_p:
            status_label.object = "<span style='font-size:11px;color:#c00'>Set export file name.</span>"
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
            status_label.object = f"<span style='font-size:11px;color:#066'>Appended {len(regions_data)} region(s) to {parquet_path.name}</span>"
        except Exception as e:
            status_label.object = f"<span style='font-size:11px;color:#c00'>{e}</span>"

    def on_file_select(event: Any) -> None:
        if event.new is None:
            return
        do_load()

    file_dropdown.param.watch(on_file_select, "value")

    export_btn.on_click(on_export)
    refresh_btn.on_click(refresh_experiment_list)

    def on_refresh_plots(_event: Any = None) -> None:
        update_od()

    refresh_plots_btn.on_click(on_refresh_plots)

    reduction_plot_refs: list[Any] = []

    def build_reduction_bokeh(file_val: Optional[str]) -> Any:
        from bokeh.layouts import column as bokeh_column

        reduction_plot_refs.clear()
        meta_u = data["meta"]
        image_u = data["image"]
        paxis_u = np.asarray(meta_u["paxis_points"])
        qaxis_u = np.asarray(meta_u["qaxis_points"])
        if paxis_u.size < 2 or qaxis_u.size < 2:
            p = figure(width=600, height=250, x_axis_label=meta_u.get("paxis_name", "Energy (eV)"), y_axis_label=meta_u.get("qaxis_name", "Sample"), toolbar_location="above")
            return p
        left, right = float(paxis_u[0]), float(paxis_u[-1])
        bottom, top = float(qaxis_u[0]), float(qaxis_u[-1])
        img_array = np.asarray(image_u)
        img_rgba = _gray_to_rgba(np.flipud(img_array))
        p_img = figure(width=600, height=250, x_range=Range1d(left, right), y_range=Range1d(bottom, top), x_axis_label=meta_u.get("paxis_name", "Energy (eV)"), y_axis_label=meta_u.get("qaxis_name", "Sample"), toolbar_location="above", match_aspect=False)
        p_img.image_rgba(image=[img_rgba], x=left, y=bottom, dw=right - left, dh=top - bottom)

        handle_x: list[float] = []
        handle_y: list[float] = []
        line_ids: list[int] = []
        x_center = (left + right) / 2.0
        margin = (top - bottom) * 0.02

        izero_lo, izero_hi = state["izero_lo"], state["izero_hi"]
        izero_seg_src = ColumnDataSource(dict(x0=[left, left], y0=[izero_lo, izero_hi], x1=[right, right], y1=[izero_lo, izero_hi]))
        p_img.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=izero_seg_src, line_width=2, color="blue")
        handle_x.extend([x_center, x_center])
        handle_y.extend([izero_lo, izero_hi])
        line_ids.extend([0, 1])
        region_seg_sources: list[ColumnDataSource] = []
        for i, reg in enumerate(state["regions"]):
            color = region_colors[i % len(region_colors)]
            lo, hi = reg["sample_lo"], reg["sample_hi"]
            seg_src = ColumnDataSource(dict(x0=[left, left], y0=[lo, hi], x1=[right, right], y1=[lo, hi]))
            p_img.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=seg_src, line_width=2, color=color)
            region_seg_sources.append(seg_src)
            handle_x.extend([x_center, x_center])
            handle_y.extend([lo, hi])
            line_ids.extend([2 + i * 2, 2 + i * 2 + 1])

        handle_src = ColumnDataSource(dict(x=handle_x, y=handle_y, line_id=line_ids))

        def on_handle_change(attr: str, old: Any, new: Any) -> None:
            ys = list(handle_src.data["y"])
            lids = list(handle_src.data["line_id"])
            q_lo, q_hi = float(np.min(qaxis_u)), float(np.max(qaxis_u))
            for idx, (y_val, lid) in enumerate(zip(ys, lids)):
                y_clip = float(np.clip(y_val, q_lo, q_hi))
                if lid == 0:
                    state["izero_lo"] = np.clip(y_clip, q_lo, state["izero_hi"] - margin)
                    izero_lo_slider.value = state["izero_lo"]
                elif lid == 1:
                    state["izero_hi"] = np.clip(y_clip, state["izero_lo"] + margin, q_hi)
                    izero_hi_slider.value = state["izero_hi"]
                else:
                    r_idx = (lid - 2) // 2
                    which = (lid - 2) % 2
                    if r_idx < len(state["regions"]):
                        if which == 0:
                            state["regions"][r_idx]["sample_lo"] = np.clip(y_clip, q_lo, state["regions"][r_idx]["sample_hi"] - margin)
                        else:
                            state["regions"][r_idx]["sample_hi"] = np.clip(y_clip, state["regions"][r_idx]["sample_lo"] + margin, q_hi)
            _bump_trigger()

        handle_src.on_change("data", on_handle_change)
        radius_handle = (right - left) * 0.008
        cr = p_img.circle(x="x", y="y", source=handle_src, radius=radius_handle, color="orange", alpha=0.9)
        p_img.add_tools(PointDrawTool(renderers=[cr], add=False))  # type: ignore[arg-type]

        regions_data = last_nexafs.get("regions") or []
        energy_arr = last_nexafs.get("energy")
        p_od = figure(width=600, height=200, x_axis_label="Energy (eV)", y_axis_label="OD (ln I0/I)", toolbar_location="above")
        od_cds: ColumnDataSource | None = None
        if regions_data and energy_arr is not None:
            energy = np.asarray(energy_arr)
            xs = [energy.tolist()] * len(regions_data)
            ys = [np.asarray(reg["OD"]).tolist() for reg in regions_data]
            colors = [region_colors[i % len(region_colors)] for i in range(len(regions_data))]
            od_cds = ColumnDataSource(dict(xs=xs, ys=ys, line_color=colors))
            p_od.multi_line(xs="xs", ys="ys", source=od_cds, line_width=2, line_color="line_color")
            p_od.legend.location = "top_right"

        layout = bokeh_column(p_img, p_od)
        reduction_plot_refs.extend([layout, handle_src, izero_seg_src, region_seg_sources, p_od, od_cds, left, right, bottom, top, x_center])
        return layout

    def update_reduction_plot_in_place() -> None:
        if len(reduction_plot_refs) < 11:
            return
        handle_src = reduction_plot_refs[1]
        izero_seg_src = reduction_plot_refs[2]
        region_seg_sources = reduction_plot_refs[3]
        od_cds = reduction_plot_refs[5]
        left = reduction_plot_refs[6]
        right = reduction_plot_refs[7]
        x_center = reduction_plot_refs[10]
        handle_x = [x_center] * (2 + 2 * len(state["regions"]))
        handle_y = [state["izero_lo"], state["izero_hi"]]
        for reg in state["regions"]:
            handle_y.append(reg["sample_lo"])
            handle_y.append(reg["sample_hi"])
        handle_src.data = dict(x=handle_x, y=handle_y, line_id=list(range(len(handle_x))))
        izero_seg_src.data = dict(x0=[left, left], y0=[state["izero_lo"], state["izero_hi"]], x1=[right, right], y1=[state["izero_lo"], state["izero_hi"]])
        for i, reg in enumerate(state["regions"]):
            if i < len(region_seg_sources):
                region_seg_sources[i].data = dict(x0=[left, left], y0=[reg["sample_lo"], reg["sample_hi"]], x1=[right, right], y1=[reg["sample_lo"], reg["sample_hi"]])
        regions_data = last_nexafs.get("regions") or []
        energy_arr = last_nexafs.get("energy")
        if od_cds is not None and regions_data and energy_arr is not None:
            energy = np.asarray(energy_arr)
            colors = [region_colors[i % len(region_colors)] for i in range(len(regions_data))]
            od_cds.data = dict(xs=[energy.tolist()] * len(regions_data), ys=[np.asarray(reg["OD"]).tolist() for reg in regions_data], line_color=colors)

    data_trigger.param.watch(lambda e: update_reduction_plot_in_place(), "value")

    def _gray_to_rgba(gray: np.ndarray) -> np.ndarray:
        g = np.asarray(gray, dtype=np.float64)
        if g.size == 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        gmin, gmax = np.nanmin(g), np.nanmax(g)
        if gmax <= gmin:
            gmax = gmin + 1.0
        g = (255 * (g - gmin) / (gmax - gmin)).astype(np.uint8)
        return np.stack([g, g, g, np.full_like(g, 255)], axis=-1)

    reduction_plot_pane = pn.pane.Bokeh(
        pn.bind(build_reduction_bokeh, file_dropdown),
        height=500,
        sizing_mode="stretch_width",
    )

    def reduction_content_or_placeholder(file_val: Optional[str]) -> pn.pane.Markdown | pn.pane.Bokeh:
        if file_val is None or file_val == "(empty)":
            return pn.pane.Markdown("Select a line scan above to see the reduction plot and drag the orange handles to adjust izero and region bounds.")
        return reduction_plot_pane

    views_reactive_pane = pn.pane.HoloViews(
        pn.bind(compute_views_plot, display_mode, mass_abs_fit_mode, sample_dropdown, data_trigger),
        backend="bokeh",
        height=280,
        sizing_mode="stretch_width",
    )
    update_od()

    sec = "font-weight:600;font-size:12px;margin-top:8px;margin-bottom:4px;display:block"
    sec_sm = "font-size:11px;color:#555;margin-bottom:2px"

    tab_setup = pn.Column(
        pn.pane.HTML(f"<div style='{sec}'>Directory and experiment</div>"),
        pn.Row(dir_text, refresh_btn, sizing_mode="stretch_width"),
        experiment_dropdown,
        pn.pane.HTML(f"<div style='{sec}'>Export file name</div>"),
        pn.pane.HTML(f"<span style='{sec_sm}'>Parquet path (default: experiment.parquet)</span>"),
        parquet_path_text,
        pn.pane.HTML(f"<div style='{sec}'>Sample config</div>"),
        pn.Row(config_json_text, load_config_btn, sizing_mode="stretch_width"),
        pn.pane.HTML("<span style='font-size:11px'>JSON: {\"samples\": {\"ExpName\": \"C8H8\", \"Other\": null}}</span>"),
        sizing_mode="stretch_width",
        margin=(8, 8),
        min_height=320,
    )

    reduction_top = pn.Row(
        pn.Column(
            pn.pane.HTML(f"<div style='{sec_sm}'>Sample (from config)</div>"),
            sample_dropdown,
            sizing_mode="fixed",
            width=160,
        ),
        pn.Column(
            pn.pane.HTML(f"<span style='{sec_sm}'>Film region</span>"),
            film_region_text,
            sizing_mode="fixed",
            width=120,
        ),
        pn.Column(
            pn.pane.HTML(f"<div style='{sec_sm}'>Line scan</div>"),
            file_dropdown,
            sizing_mode="fixed",
            width=180,
        ),
        pn.Column(
            pn.pane.HTML(f"<div style='{sec_sm}'>Region spot labels</div>"),
            region_spot_column,
            sizing_mode="stretch_width",
        ),
        pn.Column(
            pn.Spacer(height=20),
            add_region_btn,
            sizing_mode="fixed",
        ),
        sizing_mode="stretch_width",
    )

    tab_reduction = pn.Column(
        reduction_top,
        pn.pane.HTML(f"<div style='{sec_sm}'>Izero region (sample position)</div>"),
        pn.Row(izero_lo_slider, izero_hi_slider, sizing_mode="stretch_width"),
        pn.bind(reduction_content_or_placeholder, file_dropdown),
        pn.Row(refresh_plots_btn, status_label, sizing_mode="stretch_width", margin=(4, 0)),
        sizing_mode="stretch_width",
        margin=(8, 8),
        min_height=520,
    )

    data_trigger.visible = False
    def views_content_or_placeholder(_trigger: str) -> pn.pane.Markdown | pn.pane.HoloViews:
        if not last_nexafs.get("regions"):
            return pn.pane.Markdown("Reduce a line scan in the **Reduction (OD)** tab first to see views here.")
        return views_reactive_pane

    tab_views = pn.Column(
        pn.pane.HTML(f"<div style='{sec}'>Display</div>"),
        pn.Row(
            pn.Column(pn.pane.HTML(f"<span style='{sec_sm}'>Y-axis</span>"), display_mode, sizing_mode="fixed", width=140),
            pn.Column(pn.pane.HTML(f"<span style='{sec_sm}'>Mass abs fit</span>"), mass_abs_fit_mode, sizing_mode="fixed", width=160),
            data_trigger,
            sizing_mode="stretch_width",
        ),
        pn.bind(views_content_or_placeholder, data_trigger),
        pn.pane.HTML(f"<div style='{sec}'>Export</div>"),
        export_btn,
        sizing_mode="stretch_width",
        margin=(8, 8),
        min_height=400,
    )

    tabs = pn.Tabs(
        ("Setup", tab_setup),
        ("Reduction (OD)", tab_reduction),
        ("Views and export", tab_views),
        sizing_mode="stretch_width",
    )

    main_col = pn.Column(tabs, sizing_mode="stretch_width")
    try_load_config_from_experiment_dir()
    if file_dropdown.value and file_dropdown.value != "(empty)":
        do_load()

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
        apply_config_for_experiment(str(experiment_dropdown.value or ""))

    setattr(get_nexafs_dataframe, "set_sample_config", set_sample_config)

    class _PanelWithGetter:
        def __init__(self, panel: pn.Column, get_df: Callable[[], pd.DataFrame]) -> None:
            self._panel = panel
            self._get_df = get_df

        def __call__(self) -> pd.DataFrame:
            return self._get_df()

        def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> Any:
            return self._panel._repr_mimebundle_(include=include, exclude=exclude)

    wrapper = _PanelWithGetter(main_col, get_nexafs_dataframe)
    setattr(wrapper, "set_sample_config", set_sample_config)
    setattr(wrapper, "get_nexafs_dataframe", get_nexafs_dataframe)
    return wrapper


interactive_izero_split = line_scan_processor
