## Learned User Preferences

- Plot and figure must appear in the same output as the widget buttons and update when Load or other actions run; avoid separate plt.show() so the figure is embedded with controls (e.g. fig.canvas in VBox).

## Learned Workspace Facts

- STXM toolkit processes beamtime line-scan data from .hdr/.xim files in an experiment folder.
- Widget takes parent directory of experiments; use experiment dropdown to pick folder, then file dropdown lists only valid NEXAFS line scans (Type = "NEXAFS Line Scan" with loadable 2D .xim). Selecting a file loads it.
- Line-scan file list is filtered via list_nexafs_line_scans: only .hdr with Type = "NEXAFS Line Scan" and correct 2D shape; Image Scan, Focus Scan, and stacks are excluded.
- Sample and izero regions are set per scan via draggable bars; 3-region segmentation (sample, edge, izero; edge thinnest) initializes bar bounds via bar_bounds_from_three_regions. Multiple sample regions supported with add/remove and per-region spot labels.
- NEXAFS OD = ln(I0/I); dataset normalization is pre-edge baseline subtraction then scale so post-edge mean = 1.
- Experiment parquet stores NEXAFS columns plus formula, scan_path, sample_name, spot_label, film_region_name; optional OD_normalized and derived columns (mass_absorption, mass_absorption_err, beta, beta_err). Create parquet and parent dirs if missing when appending.
- Setup tab: directory, experiment dropdown, export file name (defaults to experiment.parquet), and sample config JSON. Config can be loaded from JSON path or auto-loaded when config.json exists in the selected experiment directory.
- Reduction tab: sample from config dropdown, film region, line-scan file, region spot labels with add region and per-row remove (trash) and edit (pencil) buttons, then image and OD plot. Sample choice sets formula for export from config map.
- Views tab: display mode (OD, mass absorption, beta), mass-abs fit option, spectrum plot, and Export button. No batch Process all in current widget.
- When loading a new image, use a single consistent update path (set_data, set_extent, set_clim, axes/line updates) to avoid display corruption.
- File dropdown defaults to the first valid line scan so a selection is ready and changing selection loads that scan.
