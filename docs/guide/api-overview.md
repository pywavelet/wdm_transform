# API Overview

The package currently centers on three public objects.

## `TimeSeries`

`TimeSeries` represents sampled time-domain data and the sample spacing `dt`.
It accepts either a single series with shape `(n,)` or a leading-batch layout
with shape `(batch, n)`, and stores data canonically as `(batch, n)`.

Typical operations:

- inspect `times`
- transform to `FrequencySeries`
- transform to `WDM`
- call `plot()`

## `FrequencySeries`

`FrequencySeries` represents FFT-domain data and the frequency spacing `df`.
It accepts either shape `(n,)` or `(batch, n)`, and stores data canonically as
`(batch, n)`.

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

The coefficient layout is stored canonically as `(batch, nt, nf + 1)`. Users
may still pass a single transform with shape `(nt, nf + 1)`, which is
normalized to a singleton batch internally.

```python
import jax.numpy as jnp
from wdm_transform import TimeSeries

aet = jnp.stack([a_channel, e_channel, t_channel], axis=0)
series = TimeSeries(aet, dt=dt, backend="jax")
wdm = series.to_wdm(nt=nt)

assert wdm.coeffs.shape == (3, nt, aet.shape[-1] // nt + 1)
```

## Backend model

The backend system is intentionally small right now. A backend provides:

- an array namespace
- an FFT namespace
- `asarray(...)`

That is enough for the current NumPy implementation and sets up a clean insertion point for JAX and
CuPy later.

## Sub-band transforms

The high-level objects transform complete time or frequency arrays. For narrow
frequency-domain workflows, `wdm_transform.transforms` also exposes compact
sub-band helpers that operate on only the Fourier bins and WDM channels touched
by a local band.

Use `from_freq_to_wdm_subband(...)` when you want the full overlapping WDM span
implied by a compact Fourier crop. It returns both the coefficient block and the
first WDM channel index for that block. Use `from_freq_to_wdm_band(...)` when
you already know the WDM channel range you want to keep. Use
`from_wdm_to_freq_subband(...)` to reconstruct the touched Fourier span from a
compact WDM block.

The span helpers are useful for allocating arrays and keeping local crops
aligned with the full transform grid:

```python
from wdm_transform.transforms import (
    from_freq_to_wdm_subband,
    fourier_span_from_wdm_span,
    wdm_span_from_fourier_span,
)

mmin, nf_sub_wdm = wdm_span_from_fourier_span(
    nfreqs_fourier=nfreqs_fourier,
    nfreqs_wdm=nfreqs_wdm,
    ntimes_wdm=ntimes_wdm,
    kmin=kmin,
    lendata=len(spectrum_crop),
)

coeffs, touched_mmin = from_freq_to_wdm_subband(
    spectrum_crop,
    df=df,
    nfreqs_fourier=nfreqs_fourier,
    kmin=kmin,
    nfreqs_wdm=nfreqs_wdm,
    ntimes_wdm=ntimes_wdm,
)

assert touched_mmin == mmin

kmin_recon, lendata_recon = fourier_span_from_wdm_span(
    nfreqs_fourier=nfreqs_fourier,
    nfreqs_wdm=nfreqs_wdm,
    ntimes_wdm=ntimes_wdm,
    mmin=touched_mmin,
    nf_sub_wdm=coeffs.shape[1],
)
```

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

## Transform Helpers Reference

### `from_freq_to_wdm_subband`

Public import: `from wdm_transform.transforms import from_freq_to_wdm_subband`

::: wdm_transform.transforms.from_freq_to_wdm_subband
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `from_freq_to_wdm_band`

Public import: `from wdm_transform.transforms import from_freq_to_wdm_band`

::: wdm_transform.transforms.from_freq_to_wdm_band
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `from_wdm_to_freq_subband`

Public import: `from wdm_transform.transforms import from_wdm_to_freq_subband`

::: wdm_transform.transforms.from_wdm_to_freq_subband
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `wdm_span_from_fourier_span`

Public import: `from wdm_transform.transforms import wdm_span_from_fourier_span`

::: wdm_transform.transforms.wdm_span_from_fourier_span
    options:
      show_root_heading: false
      show_root_toc_entry: false

### `fourier_span_from_wdm_span`

Public import: `from wdm_transform.transforms import fourier_span_from_wdm_span`

::: wdm_transform.transforms.fourier_span_from_wdm_span
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
