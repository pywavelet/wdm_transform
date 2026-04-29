# Reconstruction And Inference

WDM is useful only if it preserves the underlying signal content cleanly enough
to reconstruct and analyze it.

This package supports both.

## Reconstruction Paths

Starting from a `TimeSeries`, you can move through the WDM domain and back:

```python
from wdm_transform import TimeSeries

series = TimeSeries(data, dt=dt)
coeffs = series.to_wdm(nt=32)

recovered_time = coeffs.to_time_series()
recovered_freq = coeffs.to_frequency_series()
```

These two inverse paths answer slightly different questions:

- `to_time_series()`: do the coefficients preserve the original waveform?
- `to_frequency_series()`: do they preserve the original spectrum?

For a well-behaved transform, both reconstruction errors should be at the level
of floating-point roundoff.

## Why Work In WDM Space At All?

If your signal is localized in both time and frequency, WDM can be a more
natural analysis space than either raw time samples or a global FFT.

Typical motivations are:

- identifying localized narrow-band features
- separating signal-rich and noise-rich regions of the grid
- building likelihoods on a compressed or localized summary of the data

## Likelihoods In The WDM Domain

The study notebook compares two inference strategies for a sinusoid:

- a likelihood written directly in FFT space
- a likelihood written in WDM space

The WDM-domain version in the example is intentionally approximate:

- it keeps only the dominant WDM channel
- it uses an empirical diagonal noise variance estimate in that channel

That makes the model fast and easy to interpret, while still showing whether
the posterior is meaningfully shifted relative to the FFT-domain result.

## What To Look For In Posterior Comparisons

If the FFT and WDM posteriors are close, that suggests the WDM approximation is
retaining the information needed for that problem.

If they are noticeably different, common causes are:

- discarded channels still contain signal information
- the diagonal covariance approximation is too crude
- the chosen WDM resolution (`nt`, `nf`) does not align well with the feature
  of interest

## Next Step

For a full executed example with plots, go to:

- [WDM Walkthrough](../examples/wdm_walkthrough.py)
