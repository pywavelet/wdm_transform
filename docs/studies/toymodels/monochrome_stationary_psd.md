# Sinusoid in Colored Noise

Executable script: [`monochrome_stationary_psd.py`](./monochrome_stationary_psd.py).


This study adapts the larger exploratory notebook into a docs-friendly,
executable example that uses the `wdm_transform` package wherever possible.

It shows how to

- generate a sinusoid plus stationary colored noise
- transform the signal with the package `TimeSeries` and `WDM` APIs
- check time-domain and frequency-domain reconstruction
- compare signal and noise energy in time, FFT, and WDM coordinates
Before diving in, it helps to keep the three representations straight:

- In the **time domain**, the data is just a 1D sampled waveform.
- In the **FFT domain**, the same data is described by global sine/cosine
  modes, which are excellent for stationary signals and noise models.
- In the **WDM domain**, the signal is stored on a time-frequency grid, so a
  localized oscillatory feature tends to occupy only a small part of that
  grid.

The point of this study is not to prove one representation is always
better. It is to show, on a simple example, how the same underlying signal
looks in each basis and how closely the inferred sinusoid parameters agree.

## Problem setup

We use `N = nt * nf = 1024` samples, a cadence of `dt = 0.1 s`, and a
sinusoid whose frequency lands near one WDM channel center.

The transform shape is controlled by two integers:

- `nt`: the number of WDM time bins
- `nf`: the number of interior frequency channels

Their product `N = nt * nf` is the total number of time samples. For this
study, `nt = 32` and `nf = 32`, so the WDM coefficients live on a
`(32, 33)` grid. The extra column comes from the two edge channels
(`m = 0` and `m = nf`) in the Wilson-style packing.

The "dominant channel" is the WDM frequency bin that contains the largest
share of the sinusoid's coefficient energy. Later, we use that channel as a
compact 1D summary of where the sinusoidal signal mostly lives in the WDM
representation.

## Basis sanity check

A WDM transform is built from a collection of basis atoms `g_{n,m}`. You can
think of each atom as a localized oscillatory template:

- `n` moves the template in time
- `m` moves the template in frequency

Each WDM coefficient answers the question:

"How much does the data look like this specific time-frequency atom?"

That interpretation only works cleanly if different atoms are nearly
orthogonal. If two different atoms overlap strongly, then one coefficient no
longer means "one localized feature" by itself, because neighboring
coefficients would be mixing the same content.

The two overlap maps below check that property in two directions:

- Left: fix one frequency channel `m` and compare atoms at different times
- Right: fix one time index `n` and compare atoms at different frequencies

A bright diagonal with dark off-diagonal entries means the basis is behaving
the way we want: each atom mostly overlaps with itself and not with the
others. That is why these plots matter. They justify reading the WDM grid as
a meaningful time-frequency decomposition rather than just a generic linear
change of coordinates.

![Basis overlap checks](../outdir_monochrome_stationary_psd/basis_overlap_checks.png)

In this example the overlaps are essentially diagonal, which is exactly what
we want. That means:

- moving the atom in time produces a different basis element
- moving the atom in frequency also produces a different basis element
- large WDM coefficients can be interpreted as localized signal content
  instead of leakage from many strongly correlated atoms

## Reconstruction checks

The `WDM` object can reconstruct both the time-domain signal and the
FFT-domain signal.

This is an important consistency check. If the transform and its inverse are
implemented correctly, converting

`time -> WDM -> time`

and

`time -> WDM -> FFT`

should reproduce the original data up to floating-point roundoff.

## Time, frequency, and WDM views

The three panels below show the same data from different angles:

- the raw waveform in time
- its global frequency content via the FFT
- its localized time-frequency content via the WDM grid

The WDM plot is the most useful one if you care about *where in time* a
narrow-band feature is active, not just *which frequency* it has.

![Time, frequency, and WDM views](../outdir_monochrome_stationary_psd/time_frequency_wdm_views.png)

A standard spectrogram gives a familiar comparison point for the same data.

![Reference spectrogram](../outdir_monochrome_stationary_psd/reference_spectrogram.png)

## Energy and SNR comparisons

Parseval's identity makes the time-domain and FFT-domain energies match
exactly under the package conventions. The WDM energy tracks the same
quantity closely, up to the transform's floating-point roundoff.

The next printout is a compact "same information in different coordinates"
check. If the transform is close to unitary, the signal and noise norms
should agree across the time, FFT, and WDM representations, modulo numerical
precision.

## Frequency reconstruction cross-check

This mirrors the atom-expansion check from the original prototype, but uses
the package method `WDM.to_frequency_series()` instead of manually summing
the basis functions.

The raw residual curves are not very informative here because the errors are
already at floating-point precision. A compact numerical summary is easier to
read than a nearly flat line sitting at `~1e-8` on the plot.

## Posterior comparison: FFT likelihood vs WDM likelihood

The original prototype compared posteriors in the frequency and WDM domains.
Here we do the same with `numpyro`, and we evaluate the WDM likelihood
through the package's JAX transform kernel.

We use the same noisy dataset in both likelihoods. The WDM-side noise scale
is approximated with an empirical diagonal covariance estimated from Monte
Carlo noise realizations. To compare the same local posterior mode, NUTS is
initialized near the injected parameters, matching the spirit of the original
`emcee` example where walkers started close to the fiducial values.

The inference comparison works as follows:

- In the **FFT likelihood**, we compare the observed FFT to the FFT of a
  trial sinusoid and weight residuals by the assumed noise PSD.
- In the **WDM likelihood**, we transform each trial sinusoid into WDM
  coefficients and compare it to the observed WDM coefficients in the
  dominant channel.

The WDM likelihood here is still an approximation: we use only one dominant
channel and a diagonal estimate of the noise covariance in that channel. That
makes the example fast and interpretable while still being close enough to
the FFT result to compare posteriors meaningfully.

We use broad uniform priors:

- amplitude in `(0, 0.3)`
- frequency in `(0.8, 1.4) Hz`
- phase in `[-π, π]`

`numpyro` handles the sampling, while `from_time_to_wdm(..., backend="jax")`
keeps the WDM likelihood differentiable and compatible with JAX-based NUTS.
That is why this section uses the lower-level transform function rather than
the higher-level `TimeSeries.to_wdm()` method inside the model definition.

The diagonal panels show the 1D marginals and the lower triangle shows
pairwise projections. In this setup the FFT and WDM posteriors land nearly on
top of each other.

When the two colored contour families overlap closely, it means the WDM
approximation is not noticeably biasing the recovered sinusoid parameters for
this example. Large shifts or very different widths would indicate that the
WDM-domain likelihood is throwing away too much information or mis-modelling
the noise.

![Posterior comparison](../outdir_monochrome_stationary_psd/posterior_comparison.png)

## Run log

This section is generated from the script's `print()` output.

<!-- BEGIN GENERATED RUN LOG -->
```text
WDM shape: (32, 251)
Dominant WDM channel: m=55
Channel center near the sinusoid: 1.10000 Hz
Relative time-domain reconstruction error: 1.956e-14
Relative FFT-domain reconstruction error: 1.956e-14
Signal energies
  time: 40.000000
  fft : 40.000000
  wdm : 40.000000

Noise energies
  time: 44.095749
  fft : 44.095749
  wdm : 44.082243

SNR estimates
  time-domain norm ratio: 0.952427
  fft-domain norm ratio : 0.952427
  wdm-domain norm ratio : 0.952573
FFT reconstruction summary
  signal max abs error     : 6.355e-14
  signal max relative error: 1.585e-03
  noise max abs error      : 6.633e-12
  noise max relative error : 4.207e-12
Using dominant WDM channel m=55
Median estimated WDM noise variance in that channel: 1.6468e-03

Posterior mean ± std
  FFT : A=0.10012±0.00064, f0=1.10000±0.00000, phi=0.50240±0.01409
  WDM : A=0.10007±0.00061, f0=1.10000±0.00000, phi=0.50473±0.01280
  Delta mean: dA=-5.33339e-05, df0=-6.79720e-07, dphi=2.32804e-03
```
<!-- END GENERATED RUN LOG -->
