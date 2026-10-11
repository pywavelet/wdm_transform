# Full Transform Functions

Object methods are documented in [API Overview](../guide/api-overview.md).
All four full functions select `window="meyer"` by default. Gaussian accepts
`nu` and `support`; `WDM` inverse methods pass their retained metadata automatically.
Bare coefficient arrays carry no metadata, so low-level inverse callers must
pass the matching originating configuration.

::: wdm_transform.transforms.from_time_to_wdm
    options:
      show_source: false

::: wdm_transform.transforms.from_freq_to_wdm
    options:
      show_source: false

::: wdm_transform.transforms.from_wdm_to_time
    options:
      show_source: false

::: wdm_transform.transforms.from_wdm_to_freq
    options:
      show_source: false

## Compact Subbands

The compact functions and aliases below support both window families.
Meyer retains its existing span convention; Gaussian expands spans according
to its finite frequency support. These APIs use a one-sided Fourier grid and
treat omitted bins or coefficient channels as zero. Bare coefficient arrays
carry no window metadata: pass the originating settings to inverse synthesis.

::: wdm_transform.transforms.from_freq_to_wdm_subband
    options:
      show_source: false

::: wdm_transform.transforms.from_freq_to_wdm_band
    options:
      show_source: false

::: wdm_transform.transforms.from_wdm_to_freq_subband
    options:
      show_source: false

::: wdm_transform.transforms.wdm_span_from_fourier_span
    options:
      show_source: false

::: wdm_transform.transforms.fourier_span_from_wdm_span
    options:
      show_source: false
