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
