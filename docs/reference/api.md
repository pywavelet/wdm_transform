# API Reference

This page is generated from live docstrings with `mkdocstrings`, so signatures stay aligned with
the implementation. The sections below are curated around the public API rather than raw module
dumps.

## Core Objects

### `TimeSeries`

Public import: `from wdm_transform import TimeSeries`  
Implementation: [src/wdm_transform/datatypes/series.py](https://github.com/pywavelet/wdm_transform/blob/main/src/wdm_transform/datatypes/series.py)

::: wdm_transform.datatypes.series.TimeSeries
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `FrequencySeries`

Public import: `from wdm_transform import FrequencySeries`  
Implementation: [src/wdm_transform/datatypes/series.py](https://github.com/pywavelet/wdm_transform/blob/main/src/wdm_transform/datatypes/series.py)

::: wdm_transform.datatypes.series.FrequencySeries
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `WDM`

Public import: `from wdm_transform import WDM`  
Implementation: [src/wdm_transform/datatypes/wdm.py](https://github.com/pywavelet/wdm_transform/blob/main/src/wdm_transform/datatypes/wdm.py)

::: wdm_transform.datatypes.wdm.WDM
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

## Backend Utilities

### `Backend`

Implementation: [src/wdm_transform/backends/base.py](https://github.com/pywavelet/wdm_transform/blob/main/src/wdm_transform/backends/base.py)

::: wdm_transform.backends.base.Backend
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `get_backend`

::: wdm_transform.get_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `register_backend`

::: wdm_transform.register_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

## Plotting Helpers

These helpers power the datatype `.plot()` methods and remain available as standalone functions.

### `plot_time_series`

::: wdm_transform.plotting.plot_time_series
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `plot_frequency_series`

::: wdm_transform.plotting.plot_frequency_series
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `plot_periodogram`

::: wdm_transform.plotting.plot_periodogram
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `plot_spectrogram`

::: wdm_transform.plotting.plot_spectrogram
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

### `plot_wdm_grid`

::: wdm_transform.plotting.plot_wdm_grid
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
