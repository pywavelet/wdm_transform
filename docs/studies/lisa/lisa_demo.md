# LISA Galactic-Binary Study

Executable scripts:
[`data_generation.py`](./data_generation.py),
[`lisa_freq_mcmc.py`](./lisa_freq_mcmc.py),
[`lisa_wdm_mcmc.py`](./lisa_wdm_mcmc.py),
[`pp_plot.py`](./pp_plot.py).

This study is organized as a markdown-first case study backed by three plain Python scripts.
The markdown page carries the narrative, the math, and the rendered figures. The scripts are
kept as standalone executables that generate the cached products and posterior diagnostics shown
here.

## Study structure

1. `data_generation.py` builds a toy anisotropic Galactic foreground, computes the sky-averaged
   LISA response, injects one seeded resolved Galactic binary, and writes
   `outdir_lisa/<mode>/seed_<LISA_SEED>/injection.npz`.
2. `lisa_freq_mcmc.py` loads that cache and performs one local frequency-domain fit with a
   narrow-band Whittle likelihood.
3. `lisa_wdm_mcmc.py` loads the same cache, transforms the injected data to WDM coefficients, and
   performs one per-source fit on a narrow WDM band (mirroring the frequency-domain
   approach). Uses $n_t = 32$ by default for inference.
4. `pp_plot.py` scans the seeded posterior archives for one mode and builds a multi-seed PP plot
   comparing the WDM and frequency-domain calibration for the sampled parameters
   $(f_0, \dot{f}, A, \phi_0)$.

## How To Run

Run the scripts from the repository root:

```bash
LISA_SEED=0 python docs/studies/lisa/data_generation.py
python docs/studies/lisa/lisa_freq_mcmc.py
python docs/studies/lisa/lisa_wdm_mcmc.py
```

To generate an instrument-only stationary-noise injection without the stochastic Galactic foreground:

```bash
LISA_SEED=0 LISA_INCLUDE_GALACTIC=0 python docs/studies/lisa/data_generation.py
```

Useful overrides:

```bash
LISA_SEED=3 python docs/studies/lisa/data_generation.py
LISA_N_WARMUP=400 LISA_N_DRAWS=600 python docs/studies/lisa/lisa_freq_mcmc.py
LISA_N_WARMUP=400 LISA_N_DRAWS=600 LISA_NT=32 python docs/studies/lisa/lisa_wdm_mcmc.py
python docs/studies/lisa/pp_plot.py --mode stationary_noise
```

`data_generation.py` is the prerequisite step. Both inference scripts read:

- `docs/studies/lisa/outdir_lisa/<mode>/seed_<LISA_SEED>/injection.npz`

That cache stores the A/E/T time series, the PSD grids, the injected source parameters, the
generation seed, and the SNR summary needed by the follow-on fits.

All outputs from the three scripts for a given run now live in the same directory:

- `docs/studies/lisa/outdir_lisa/stationary_noise/seed_1/` for `LISA_INCLUDE_GALACTIC=0 LISA_SEED=1`
- `docs/studies/lisa/outdir_lisa/galactic_background/seed_1/` for `LISA_INCLUDE_GALACTIC=1 LISA_SEED=1`

Typical files are:

- `injection.npz`
- `freq_posteriors.npz`
- `wdm_posteriors.npz`
- `posterior_marginals_compare.png`
- `posterior_interval_compare.png`
- `posterior_pp_compare.png`
- `corner_source_1.png`

The expensive response-tensor cache is now shared across runs:

- `docs/studies/lisa/outdir_lisa/_cache/Rtildeop_tf.npz`

## Data Generation

### Data Model

The A-channel strain is modeled as

$$
d_A(t) = n_A(t) + h_{\mathrm{gal}}(t) + h(t; \theta)
$$

where the instrumental noise term $n_A$, the stochastic Galactic foreground $h_{\mathrm{gal}}$, 
and the single resolved compact-binary signal $h$ are generated with a seed-controlled draw.

The foreground PSD is built from a sky map and a response tensor:

$$
S_A(f, t) = S_A^{\mathrm{inst}}(f) + |R_{AA}(f, t)| S_{\mathrm{gal}}(f)
$$

The expensive part is that we must compute and cache the time-dependent response tensor needed 
to mix the anisotropic sky model into the detector channels.

### Background Diagnostics

`data_generation.py` generates the following diagnostic plots:

**Galactic morphology and noise properties:**

![Galactic morphology](../../_static/lisa_background/galaxy_mollview.png)

![Galactic PSD](../../_static/lisa_background/galaxy_frequency_psd.png)

![LISA noise PSD](../../_static/lisa_background/lisa_noise_psd.png)

**Channel A: time-dependent components and injections:**

![Channel A total PSD](../../_static/lisa_background/channel_a_total_psd.png)

![Channel A noise versus galaxy](../../_static/lisa_background/channel_a_noise_vs_galaxy.png)

**Resolved binary injections relative to characteristic noise strain:**

![Resolved GB injections versus noise](../../_static/lisa_background/resolved_gb_vs_noise_characteristic_strain.png)

### Source Code: `data_generation.py`

```python
--8<-- "docs/studies/lisa/data_generation.py"
```

## Frequency-Domain MCMC

### Likelihood

For the frequency-domain MCMC, the injected source is fit in a narrow local band around its
carrier frequency.

If $\tilde{d}_k$ is the A-channel FFT and $\tilde{h}_k(\theta)$ is the template restricted to the
same band, the code uses the Whittle approximation:

$$
\log p(d \mid \theta) \propto
-\sum_{k \in \mathcal{B}} \left[
\log S_k + \frac{4 \Delta f |\Delta t (\tilde{d}_k - \tilde{h}_k(\theta))|^2}{S_k}
\right]
$$

The fitted parameters are $(f_0, \dot{f}, A, \phi_0)$ for the injected source. Sky position, polarization,
and inclination stay fixed at their injected values to isolate the likelihood machinery rather than
perform a full eight-parameter search.

To improve the sampler geometry, the scripts actually sample
$(\log f_0, \log \dot{f}, \log A, \phi_c)$ where

$$
\phi_c \equiv \phi_0 + 2 \pi f_0 t_c + \pi \dot{f} t_c^2,
\qquad t_c = T_{\rm obs}/2,
$$

and then reconstruct $\phi_0$ as a deterministic parameter. This largely removes the strongest
local $f_0$-$\phi_0$ degeneracy from the HMC coordinates while preserving the same physical model.

### Outputs

`lisa_freq_mcmc.py` now writes the frequency-domain posterior archive only. Plotting is handled by
the post-processing step after both inference runs finish.

### Source Code: `lisa_freq_mcmc.py`

```python
--8<-- "docs/studies/lisa/lisa_freq_mcmc.py"
```

## WDM-Domain MCMC

### Likelihood

The WDM run uses the same injected A-channel data after truncating it to a length compatible with
the $(n_t, n_f)$ tiling. The transform produces coefficients:

$$
w_{n,m} = \langle d, g_{n,m} \rangle
$$

where each $g_{n,m}$ is a localized Wilson-Daubechies-Meyer atom centered near time bin $n$ and
frequency bin $m$.

The full likelihood would require the covariance of those coefficients. This study uses a diagonal
approximation with analytic per-pixel variance $\Sigma_{n,m} = S_n(f_m) \cdot f_{\rm Nyq}$:

$$
\log p(w \mid \theta) \propto -\frac{1}{2}
\sum_{n,m \in \mathcal{B}} \left[
\frac{(w_{n,m} - h_{n,m}(\theta))^2}{\Sigma_{n,m}} + \log(2\pi \Sigma_{n,m})
\right]
$$

Like the frequency-domain run, the injected binary is fit on a narrow local band.

### Fast WDM forward model

The naive WDM template path would:

1. build the full rFFT template,
2. inverse FFT back to the time domain,
3. apply the full WDM transform,
4. then crop to the local band used for inference.

That is much more work than we need inside NUTS. The current script instead:

1. asks `JaxGB` only for the local FFT bins that contain the source,
2. embeds those bins into a small local FFT buffer,
3. applies only the WDM channels that intersect the local inference band.

In operator form, if $P_{\mathcal B}$ is the local FFT crop and $W_{\mathcal B}$ is the band-limited
WDM transform, the template path used by the script is

$$
h^{\rm WDM}_{n,m}(\theta) = W_{\mathcal B} \, P_{\mathcal B} \, \tilde h(\theta),
$$

which is algebraically equivalent to the full transform restricted to the same band, but much
cheaper to evaluate.

### Tiling choice

The WDM tiling parameter $n_t$ sets the frequency channel spacing
$\Delta f_{\rm wdm} = n_t / (2 T_{\rm obs})$ and the number of time bins.

For the current study, the default choice is $n_t = 32$, which gives

$$
\Delta f_{\rm wdm} \approx 5.1 \times 10^{-7}\ {\rm Hz}
$$

for a one-year observation. That is coarse compared to the FFT bin width, but still fine enough
for these narrow local fits once the likelihood normalization and sampler parameterization are set
up correctly.

### What We Fixed

The current WDM script is the result of a few debugging passes. The important fixes were:

1. Keep essentially the full observation time. Earlier versions threw away a large chunk of the
   year when forcing WDM-friendly lengths, which broadened the WDM posterior immediately.
2. Fit the injected source on its own narrow local band instead of carrying a multi-source loop.
3. Use the correct diagonal WDM noise variance,

   $$
   \mathrm{Var}[w_{n,m}] = \frac{S_n(f_m)}{2 \Delta t} = S_n(f_m)\,f_{\rm Nyq},
   $$

   not a naive `PSD × Δf_wdm` estimate.
4. Reparameterize the phase with $\phi_c$ at mid-observation time, exactly as in the frequency
   script.
5. Most importantly, use a straightforward model-based NumPyro path for the WDM sampler. Earlier
   WDM sampler variants produced prior-width $f_0$ and $\phi_0$ posteriors even when the underlying
   WDM likelihood agreed with the frequency-domain likelihood.

The final implementation now yields WDM and frequency-domain posteriors that overlap closely in the
full run.

### Outputs

`lisa_wdm_mcmc.py` writes the WDM posterior archive only. Plotting is handled centrally by the
post-processing step after both inference runs finish.

### Source Code: `lisa_wdm_mcmc.py`

```python
--8<-- "docs/studies/lisa/lisa_wdm_mcmc.py"
```

## Comparison of Methods

### Results

To assess the two inference approaches (frequency-domain vs. WDM-domain), the script
[`post_proc.py`](./post_proc.py) loads both posterior files and produces
side-by-side visualizations.

**Marginal posterior histograms:**

![Posterior marginals comparison](../../_static/lisa/outdir_lisa/galactic_background/seed_0/posterior_marginals_compare.png)

**Credible intervals (5th, median, 95th percentiles):**

![Posterior intervals comparison](../../_static/lisa/outdir_lisa/galactic_background/seed_0/posterior_interval_compare.png)

**Joint corner plot for the injected source (overlaid runs):**

![Comparison corner GB 1](../../_static/lisa/outdir_lisa/galactic_background/seed_0/corner_source_1.png)

The important final result is that the WDM and frequency-domain posteriors now overlap closely in
all four fitted source parameters and in the derived SNR. In particular, the earlier dramatic
WDM-only broadening in $f_0$ and $\phi_0$ was an implementation problem, not evidence that the
WDM representation was intrinsically much less informative for this toy study.

To generate these figures, run:

```bash
python docs/studies/lisa/post_proc.py
```

This compares the default WDM and frequency-domain posteriors. You can also pass custom paths:

```bash
python docs/studies/lisa/post_proc.py \
  --run-a path/to/wdm_posteriors.npz --name-a "WDM" \
  --run-b path/to/freq_posteriors.npz --name-b "Frequency"
```

## Notes

- `data_generation.py` is the expensive step because it computes and caches the response tensor
  before injecting the resolved binaries.
- JAX is imported inside `main()` in `data_generation.py` only after the multiprocessing pools
  finish, avoiding the JAX-plus-fork failure mode.
- `lisa_freq_mcmc.py` and `lisa_wdm_mcmc.py` are now ordinary scripts rather than notebook-style
  percent files.
- The shared helper file `lisa_common.py` now holds the common prior metadata, truth-vector, and
  posterior-output helpers used by both inference scripts.
- The page above includes the live source for the study scripts, so the docs build exposes the
  exact code used to produce the study outputs.
