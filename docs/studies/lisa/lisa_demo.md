# LISA Galactic-Binary Study

Executable scripts:
[`data_generation.py`](./data_generation.py),
[`lisa_freq_mcmc.py`](./lisa_freq_mcmc.py),
[`lisa_wdm_mcmc.py`](./lisa_wdm_mcmc.py).

This study is organized as a markdown-first case study backed by three plain Python scripts.
The markdown page carries the narrative, the math, and the rendered figures. The scripts are
kept as standalone executables that generate the cached products and posterior diagnostics shown
here.

## Study structure

1. `data_generation.py` builds a toy anisotropic Galactic foreground, computes the sky-averaged
   LISA response, injects two resolved Galactic binaries, and writes
   `outdir_gb_background/injection.npz`.
2. `lisa_freq_mcmc.py` loads that cache and performs two independent local frequency-domain fits
   with a narrow-band Whittle likelihood.
3. `lisa_wdm_mcmc.py` loads the same cache, transforms the injected data to WDM coefficients, and
   performs a joint two-source fit on a restricted time-frequency band.

## How To Run

Run the scripts from the repository root:

```bash
python docs/studies/lisa/data_generation.py
python docs/studies/lisa/lisa_freq_mcmc.py
python docs/studies/lisa/lisa_wdm_mcmc.py
```

`data_generation.py` is the prerequisite step. Both inference scripts read:

- `docs/studies/lisa/outdir_gb_background/injection.npz`

That cache stores the A/E-channel time series, the PSD grids, the injected source parameters, and
the per-source SNR summaries needed by the follow-on fits.

## Data Generation

### Data Model

The A-channel strain is modeled as

$$
d_A(t) = n_A(t) + h_{\mathrm{gal}}(t) + \sum_{i=1}^{2} h_i(t; \theta_i)
$$

where the instrumental noise term $n_A$, the stochastic Galactic foreground $h_{\mathrm{gal}}$, 
and the two resolved compact-binary signals $h_i$ are generated with `JaxGB`.

The foreground PSD is built from a sky map and a response tensor:

$$
S_A(f, t) = S_A^{\mathrm{inst}}(f) + |R_{AA}(f, t)| S_{\mathrm{gal}}(f)
$$

The expensive part is that we must compute and cache the time-dependent response tensor needed 
to mix the anisotropic sky model into the detector channels.

### Background Diagnostics

`data_generation.py` generates the following diagnostic plots:

**Galactic morphology and noise properties:**

![Galactic morphology](outdir_gb_background/galaxy_mollview.png)

![Galactic PSD](outdir_gb_background/galaxy_frequency_psd.png)

![LISA noise PSD](outdir_gb_background/lisa_noise_psd.png)

**Channel A: time-dependent components and injections:**

![Channel A total PSD](outdir_gb_background/channel_a_total_psd.png)

![Channel A noise versus galaxy](outdir_gb_background/channel_a_noise_vs_galaxy.png)

**Resolved binary injections relative to characteristic noise strain:**

![Resolved GB injections versus noise](outdir_gb_background/resolved_gb_vs_noise_characteristic_strain.png)

### Source Code: `data_generation.py`

```python
--8<-- "docs/studies/lisa/data_generation.py"
```

## Frequency-Domain MCMC

### Likelihood

For the frequency-domain MCMC, each source is fit in a narrow local band around its injected
carrier frequency. The two binaries are well separated in frequency, so they are treated as
independent local problems.

If $\tilde{d}_k$ is the A-channel FFT and $\tilde{h}_k(\theta)$ is the template restricted to the
same band, the code uses the Whittle approximation:

$$
\log p(d \mid \theta) \propto
-\sum_{k \in \mathcal{B}} \left[
\log S_k + \frac{4 \Delta f |\Delta t (\tilde{d}_k - \tilde{h}_k(\theta))|^2}{S_k}
\right]
$$

The fitted parameters are $(f_0, \dot{f}, A, \phi_0)$ for each source. Sky position, polarization,
and inclination stay fixed at their injected values to isolate the likelihood machinery rather than
perform a full eight-parameter search.

### Results

The local-band view checks that the posterior median template lands on top of the observed power in
each source neighborhood.

![Local frequency bands](lisa_freq_mcmc_assets/local_frequency_bands.png)

The posterior corner plots summarize the recovered local parameters for each binary.

![Frequency-domain GB 1 corner](lisa_freq_mcmc_assets/gb1_corner.png)

![Frequency-domain GB 2 corner](lisa_freq_mcmc_assets/gb2_corner.png)

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
approximation calibrated from synthetic stationary-noise draws:

$$
\Sigma_{n,m} \approx \mathrm{Var}[w_{n,m}]
\quad\text{with}\quad
\log p(w \mid \theta) \propto -\frac{1}{2}
\sum_{n,m \in \mathcal{B}} \left[
\frac{(w_{n,m} - h_{n,m}(\theta))^2}{\Sigma_{n,m}} + \log(2\pi \Sigma_{n,m})
\right]
$$

Unlike the FFT fit, the WDM run models both binaries jointly on one shared band because they occupy
the same localized time-frequency patch once projected into the WDM grid.

### Results

The first plot shows the injected data on the WDM grid and the selected analysis band.

![WDM overview](lisa_wdm_mcmc_assets/wdm_overview.png)

The band-limited comparison shows the observed WDM coefficients next to the posterior median model.

![Band-limited WDM fit](lisa_wdm_mcmc_assets/wdm_band_fit.png)

The joint fit still produces source-level posteriors, shown here as one corner plot per binary.

![WDM-domain GB 1 corner](lisa_wdm_mcmc_assets/gb1_corner.png)

![WDM-domain GB 2 corner](lisa_wdm_mcmc_assets/gb2_corner.png)

### Source Code: `lisa_wdm_mcmc.py`

```python
--8<-- "docs/studies/lisa/lisa_wdm_mcmc.py"
```

## Comparison of Methods

### Results

To assess the two inference approaches (frequency-domain vs. WDM-domain), the script
[`compare_mcmc_results.py`](./compare_mcmc_results.py) loads both posterior files and produces
side-by-side visualizations.

**Marginal posterior histograms:**

![Posterior marginals comparison](compare_mcmc_results_assets/posterior_marginals_compare.png)

**Credible intervals (5th, median, 95th percentiles):**

![Posterior intervals comparison](compare_mcmc_results_assets/posterior_interval_compare.png)

**Joint corner plots per source (overlaid runs):**

![Comparison corner GB 1](compare_mcmc_results_assets/corner_source_1.png)

![Comparison corner GB 2](compare_mcmc_results_assets/corner_source_2.png)

**Source-wise SNR comparison (if available):**

![SNR comparison](compare_mcmc_results_assets/snr_compare.png)

To generate these figures, run:

```bash
python docs/studies/lisa/compare_mcmc_results.py
```

This compares the default WDM and frequency-domain posteriors. You can also pass custom paths:

```bash
python docs/studies/lisa/compare_mcmc_results.py \
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
- The page above includes the live source for all three scripts, so the docs build exposes the
  exact code used to produce the study outputs.
