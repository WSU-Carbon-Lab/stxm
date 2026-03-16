# stxm

STXM line-scan toolkit: load .hdr/.xim, region selection, Beer-Lambert NEXAFS, mass absorption, experiment parquet.

The interactive line-scan processor (`line_scan_processor`) uses **Panel** for layout and widgets, with a reactive **HoloViews/Bokeh** plot on the Views tab. Use in Jupyter:

```python
from stxm import line_scan_processor

get_nexafs_dataframe = line_scan_processor(parent_dir)
# Use the UI; then:
df = get_nexafs_dataframe()
get_nexafs_dataframe.set_sample_config({"SampleName": "C8H8"})
```
