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
