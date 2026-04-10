# Deglitching with Thresholding

Executable script: [`deglitching.py`](./deglitching.py).


This study shows a deliberately simple WDM-domain cleanup workflow:

- generate a smooth signal plus stationary noise
- inject a few loud, short-duration artifacts
- transform the data into WDM coefficients
- build a blurred glitch score in the WDM grid
- threshold that score to obtain a soft time-bin mask
- attenuate the flagged WDM coefficients and reconstruct the cleaned series

The point is not that this is a production deglitcher. The point is that WDM
makes impulsive, broadband artifacts easy to localize in a way that is much
harder to express with a plain FFT.

## Synthetic data

We build a smooth underlying signal, add stationary Gaussian noise, and then
inject several loud glitch bursts. Each glitch has a short Gaussian envelope
but also a fast oscillatory carrier, so the contamination is both localized in
time and rich in high-frequency content. That is exactly the type of feature
WDM is good at isolating.

## What the contamination looks like

In the time domain the glitches appear as obvious short bursts. In the WDM
grid they show up as localized time bins with unusually large activity across
many channels. That is the key structural clue we use for cleanup.

![Observed contamination overview](../outdir_deglitching/contamination_overview.png)

## Build a simple glitch score

The cleanup rule below is intentionally simple:

1. Compute a robust per-channel scale from the coefficient magnitudes.
2. Whiten the WDM grid by that scale.
3. Blur the absolute whitened grid with a small Gaussian filter.
4. Collapse the blurred grid into one score per time bin.
5. Threshold the score and smooth the resulting binary mask.

The logic is that a glitch tends to light up many channels at once in a small
number of neighboring WDM time bins, whereas the underlying signal is more
structured and channel-localized.

The panels below show the intermediate representation:

- whitened coefficient magnitudes
- the blurred score used for detection
- the final 1D soft mask over WDM time bins

![Glitch score breakdown](../outdir_deglitching/glitch_score_breakdown.png)

## Reconstruct the cleaned series

We attenuate the WDM coefficients in the flagged time bins and reconstruct
back to the time domain. This simple version applies the same time-bin mask to
every channel.

## Iterative detect-clean-reconstruct loop

A natural refinement is to repeat the same operation a few times:

1. detect unusual WDM activity
2. attenuate the flagged bins
3. reconstruct the time series
4. transform the cleaned result back to WDM and detect again

The motivation is simple. A very loud glitch can partially hide weaker
neighbors on the first pass. After one reconstruction, the dominant artifact
is smaller, so the next pass can refine the score and the mask.

The next figure shows how the time-bin score evolves as we iterate. In a good
case the peaks caused by the glitches become less extreme after each pass, and
the cleaned waveform gets closer to the reference signal-plus-noise series.

In this toy problem the first pass already removes most of the artifact power.
Later passes still refine the score, but they also start to attenuate some
non-glitch structure. That is why iterative schemes usually need an explicit
stopping rule instead of a fixed number of passes.

![Iterative score and reconstruction](../outdir_deglitching/iterative_score_and_reconstruction.png)

The next figure compares the full time series and then zooms into the glitch
neighborhoods. This makes it easier to see both the suppression and the price
paid by such a simple mask.

![Deglitching time-domain result and zooms](../outdir_deglitching/deglitching_time_domain_zoom.png)

## PSD estimate before and after cleanup

A useful side effect of deglitching is that stationary spectral estimates are
often less biased by rare, loud artifacts. We compare Welch PSD estimates for
the observed data, the one-pass and iterative cleaned reconstructions, and
the reference data without glitches.

![Welch PSD estimate before and after cleanup](../outdir_deglitching/welch_psd_cleanup_comparison.png)

## Downstream signal inference with `numpyro`

A common question after deglitching is whether the cleaned data leads to
better parameter inference. To show that, we fit the dominant monochromatic
component in this synthetic dataset:

- amplitude `A`
- frequency `f0`
- phase `phi`
- residual scatter `sigma`

For this inference step we again use a *more selective* coefficient mask than
the waveform/PSD section above. We also apply it iteratively, so the dominant
artifact coefficients are removed first and the next WDM pass can refine the
mask on a cleaner background.

The comparison below is the main point: after selective WDM deglitching, the
posterior for the dominant sinusoid moves closer to the no-glitch reference
and the fitted residual scatter `sigma` drops substantially.

![Posterior comparison](../outdir_deglitching/posterior_comparison.png)

## Remarks

This example deliberately uses a crude cleanup rule. It is useful because the
logic is transparent:

- glitches are short in time and broad across channels
- the WDM grid makes that pattern easy to detect
- a soft time-bin mask already improves both the waveform and the PSD estimate
- iterating the same detect-reconstruct loop can refine the cleanup further,
  but it can also over-clean if you do not stop early enough

More realistic pipelines could use:

- channel-dependent masks
- stronger stopping criteria for iterative detection and reconstruction
- explicit glitch templates
- statistically calibrated thresholds instead of a hand-tuned MAD rule

## Run log

This section is generated from the script's `print()` output.

<!-- BEGIN GENERATED RUN LOG -->
```text
WDM shape: (32, 65)
Injected glitch sample indices: [150, 180, 430, 615, 760, 905]
Flagged WDM time bins: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Threshold on time-bin score: 1.055
MSE before cleanup: 0.1636
MSE after cleanup : 0.0710
Iterative deglitching summary
  iter 1: flagged bins=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], threshold=1.055, MSE=0.0699
  iter 2: flagged bins=[2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15], threshold=0.970, MSE=0.0489
  iter 3: flagged bins=[2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15], threshold=0.929, MSE=0.0481
Posterior mean ± std
  observed         : A=0.5530±0.0140, f0=0.17999±0.00007, phi=0.3125±0.0531, sigma=0.4415±0.0069
  inference-cleaned: A=0.5525±0.0074, f0=0.18001±0.00004, phi=0.3001±0.0261, sigma=0.2305±0.0036
  reference        : A=0.5542±0.0056, f0=0.17998±0.00003, phi=0.3155±0.0181, sigma=0.1894±0.0032
```
<!-- END GENERATED RUN LOG -->
