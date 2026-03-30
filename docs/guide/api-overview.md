# API Overview

The package currently centers on three public objects.

## `TimeSeries`

`TimeSeries` represents one-dimensional sampled time-domain data and the sample spacing `dt`.

Typical operations:

- inspect `times`
- transform to `FrequencySeries`
- transform to `WDM`
- call `plot()`

## `FrequencySeries`

`FrequencySeries` represents one-dimensional FFT-domain data and the frequency spacing `df`.

Typical operations:

- inspect `freqs`
- convert back to `TimeSeries`
- transform to `WDM`
- call `plot()`

## `WDM`

`WDM` stores the packed coefficient matrix and the transform metadata:

- `dt`
- window parameter `a`
- auxiliary parameter `d`
- backend

Typical operations:

- `WDM.from_time_series(...)`
- `TimeSeries.to_wdm(...)`
- `FrequencySeries.to_wdm(...)`
- `to_time_series()`
- `to_frequency_series()`
- `plot()`

## Backend model

The backend system is intentionally small right now. A backend provides:

- an array namespace
- an FFT namespace
- `asarray(...)`

That is enough for the current NumPy implementation and sets up a clean insertion point for JAX and
CuPy later.

## Live API Reference

The sections below are generated from live docstrings with `mkdocstrings`, so signatures stay
aligned with the implementation.

## Core Objects Reference

### `TimeSeries`

Public import: `from wdm_transform import TimeSeries`

::: wdm_transform.datatypes.series.TimeSeries
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `FrequencySeries`

Public import: `from wdm_transform import FrequencySeries`

::: wdm_transform.datatypes.series.FrequencySeries
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `WDM`

Public import: `from wdm_transform import WDM`

::: wdm_transform.datatypes.wdm.WDM
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Backend Utilities Reference

### `Backend`

::: wdm_transform.backends.base.Backend
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `get_backend`

::: wdm_transform.get_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `register_backend`

::: wdm_transform.register_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Plotting Helpers Reference

These helpers power the datatype `.plot()` methods and remain available as standalone functions.

### `plot_time_series`

::: wdm_transform.plotting.plot_time_series
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `plot_frequency_series`

::: wdm_transform.plotting.plot_frequency_series
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `plot_periodogram`

::: wdm_transform.plotting.plot_periodogram
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `plot_spectrogram`

::: wdm_transform.plotting.plot_spectrogram
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `plot_wdm_grid`

::: wdm_transform.plotting.plot_wdm_grid
    options:
      show_root_heading: false
      show_root_toc_entry: false
