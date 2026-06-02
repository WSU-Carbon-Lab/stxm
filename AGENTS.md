## Learned User Preferences

- Plot and figure must appear in the same output as the widget buttons and update when Load or other actions run; avoid separate plt.show() so the figure is embedded with controls (e.g. fig.canvas in VBox).
- Prefer production-grade, numerically sound code with no placeholder text, no emojis, and no inline comments; use NumPy-style docstrings for documentation.
- For Python projects use uv as the package manager; for TypeScript/JavaScript projects use bun instead of npm.

## Learned Workspace Facts

- STXM toolkit processes beamtime line-scan data from .hdr/.xim files in an experiment folder.
- Widget takes parent directory of experiments; experiment dropdown lists subdirs sorted by date (names like yyyy-mm(Month) or yyyy_mm(Month)), latest first. File dropdown lists only valid NEXAFS line scans (Type = "NEXAFS Line Scan" with loadable 2D .xim). Selecting a file loads it.
- Line-scan file list is filtered via list_nexafs_line_scans: only .hdr with Type = "NEXAFS Line Scan" and correct 2D shape; Image Scan, Focus Scan, and stacks are excluded.
- Sample and izero regions (A-B, C-D bars) are set per scan; defaults come from three-region segmentation (sample, edge, izero via bar_bounds_from_three_regions, edge thinnest); bars remain draggable. Pre-edge and post-edge (eV) define dataset normalization ranges.
- NEXAFS OD = ln(I0/I); normalization modes are `pre_edge_scale` (pre-edge baseline + post-edge scale to 1) and `scale_shift` (adds energy shift to align post-edge when samples drift).
- Experiment parquet stores NEXAFS columns plus formula, scan_path, optional OD_normalized, and optional derived columns mass_absorption, mass_absorption_err, beta, and beta_err; parquet and parent dirs are created if missing when appending. In Setup, parquet and sample config are filenames only (e.g. experiment.parquet, samples.json); resolved against the selected experiment directory. Auto-load prefers samples.json, then config.json.
- Interactive line-scan widget has two top-level tabs: Dashboard and Ingestion. Dashboard has a shared header (parent dir, experiment, parquet path, store root, refresh) and nested sub-tabs: Preview spectra (parquet/store browser, scan checkboxes, sample/region filters, overlay plot) and LC fitting (component-based LCF: target blend spectrum, film component rows with material name, reference spectrum, initial/min/max %, fixed checkbox, live preview, Run LCF, composition %, fit + residual panels). LCF uses spectra from the current reduction, loaded parquet, and store catalog; normalization basis follows Ingestion Raw vs Normalized OD settings. Ingestion tab: two-column layout (left: regions list with add button, narrow line scan map, export; right: spectrum controls and wide OD plot), line scan select, draggable region bars, normalization mode, SciencePlots-styled spectra with draggable legends; export to parquet and optional store.
- Process all batch processing is not available in the current widget; export works on the currently reduced scan and its defined regions only.
- When loading a new image, use a single consistent update path (set_data, set_extent, set_clim via plotting.apply_line_scan_image_clim with grayscale and percentile limits on raw counts) to avoid display corruption.
- File dropdown defaults to the first valid line scan so a selection is ready and changing selection loads that scan.
- line_scan_processor displays the widget and does not return a dataframe; read exported spectra with load_experiment_parquet.
- Export can write legacy experiment.parquet and an append-only partitioned spectrum store (store.py) when a store root is set; region averaging uses selectable WeightingMode (default POISSON_MLE).
- Region ROI bar bounds and spot labels persist per scan in experiment-dir `regions.json` (keyed by `.hdr` basename); the Ingestion tab restores saved regions on load and debounced auto-saves on drag, add/remove, label edit, and export.
