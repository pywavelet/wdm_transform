# API Overview

The public API centres on three data classes and a handful of
signal-processing helpers. The full auto-generated reference is below;
this page is a quick orientation.

## At A Glance

| Object | Purpose | Key methods |
|---|---|---|
| [`TimeSeries`](#wdm_transform.datatypes.TimeSeries) | Sampled time-domain data plus `dt` | `to_frequency_series()`, `to_wdm()`, `plot()` |
| [`FrequencySeries`](#wdm_transform.datatypes.FrequencySeries) | FFT-domain data plus `df` | `to_time_series()`, `to_wdm()`, `plot()` |
| [`WDM`](#wdm_transform.datatypes.WDM) | Packed $(N_t,\, N_f+1)$ real coefficients plus transform metadata | `to_time_series()`, `to_frequency_series()`, `plot()` |

All three accept either a single array `(n,)` or a leading-batch layout
`(batch, n)` and store data canonically as batched. The `WDM` class stores
coefficients as `(batch, nt, nf + 1)`.

## Window Selection And Reconstruction

Every full forward path accepts the same keyword-only window options:

```python
import numpy as np
from wdm_transform import TimeSeries

ts = TimeSeries(np.sin(2*np.pi*np.arange(8192)/128), dt=5.0)
meyer = ts.to_wdm(nt=256, window="meyer")
gaussian = ts.to_wdm(nt=256, window="gaussian", nu=0.5, support=16)
recovered = gaussian.to_time_series()
```

This applies to both series classes, `WDM.from_time_series`,
`WDM.from_frequency_series`, and the low-level full transform functions.
`WDM.window`, `WDM.nu`, and `WDM.support` retain the effective configuration.
Their fields follow the historical positional constructor arguments and
are keyword-only. Gaussian resolves omitted `nu` to 0.5 and omitted
`support` to `min(16,nf)`. `nu` is currently validated in `[0.3,0.8]`, and
explicit support must be an integer in `[1,nf]`.

`WDM` inverse methods automatically use these stored parameters. Inverse
functions receiving a bare coefficient array require the caller to preserve
and pass its window configuration. Identical shape does not make projections
in the two bases interchangeable: reconstruct and transform again to switch.

The full Fourier interface accepts the complete complex FFT grid, including
negative frequencies. Its existing projection onto a Hermitian spectrum is
preserved; it does not accept an rFFT half-spectrum as a substitute.

Compact subband functions and span helpers accept the same `window`, `nu` and
`support` options. Omitted Fourier bins or coefficient channels are treated
as zero. Reuse the originating settings for inverse synthesis; Gaussian
requires wider frequency margins. See [Choosing a window](choosing-a-window.md)
for support and noise-model
limitations and [full function signatures](../reference/api.md).

## Backend Selection

`wdm-transform` ships with a NumPy backend by default. The JAX backend is
opt-in:

```python
import wdm_transform as wt
wt.get_backend()                  # default NumPy backend
wt.get_backend("jax")             # opt in to the JAX backend (auto-loaded)
```

The `WDM_BACKEND` environment variable provides the same selection without
code changes (e.g. `WDM_BACKEND=jax`).

See [Reconstruction and Inference](reconstruction-and-inference.md) and
the [Benchmarks](../benchmarks.ipynb) page for a runtime comparison.

---

## Data Classes

::: wdm_transform.datatypes.TimeSeries
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

::: wdm_transform.datatypes.FrequencySeries
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

::: wdm_transform.datatypes.WDM
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

---

## Signal Processing Helpers

::: wdm_transform.signal_processing.matched_filter_snr_rfft
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.signal_processing.matched_filter_snr_wdm
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.signal_processing.noise_characteristic_strain
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.signal_processing.rfft_characteristic_strain
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.signal_processing.rfft_periodogram_characteristic_strain
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.signal_processing.wdm_noise_variance
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

---

## Backends

::: wdm_transform.backends.get_backend
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false

::: wdm_transform.backends.register_backend
    options:
      show_root_heading: true
      heading_level: 3
      show_source: false
