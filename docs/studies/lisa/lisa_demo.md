# LISA Galactic-Binary Study

Executable scripts:
[`lisa_study.py`](./lisa_study.py) (the per-seed runner) and
[`collect_all_results.py`](./collect_all_results.py) (the repeated-seed aggregator),
with shared helpers in [`lisa_common.py`](./lisa_common.py) and
[`gb_prior.py`](./gb_prior.py).

This study is organised as a markdown-first case study backed by two plain Python
scripts. The galactic confusion foreground has been removed: the only stochastic
component is **stationary instrument noise**, into which a single **resolved
galactic binary** is injected. The page carries the narrative, the math, and the
rendered figures; the scripts are standalone executables that generate the
products shown here.

## Study structure

`lisa_study.py` runs the whole per-seed pipeline in one process:

1. **Data generation** — draw a stationary A/E/T instrument-noise realization and
   inject one resolved galactic binary whose true carrier frequency is drawn from
   a small jitter around a fixed external reference `f_ref`. Print the
   matched-filter SNR and write `outdir_lisa/stationary_noise/seed_<seed>/injection.npz`.
2. **Frequency-domain inference** — fit the source in a narrow band with a Whittle
   likelihood (NumPyro/NUTS).
3. **WDM-domain inference** — fit the same data on a narrow WDM band with a
   diagonal Gaussian likelihood ($n_t = 32$ by default).
4. **Comparison** — overlay the two marginal posteriors against the injected
   truth and write `results.json` (per-parameter Jensen-Shannon divergence and
   the posterior rank of the truth, the latter feeding the PP plot).

`collect_all_results.py` then aggregates every seed's `results.json` into a PP
calibration plot, a JSD histogram, and a CSV/JSON summary.

## How To Run

Run from the study directory (or pass module-relative paths from the repo root):

```bash
python lisa_study.py --seed 0
```

Aggregate a batch of completed seeds:

```bash
python collect_all_results.py --start-seed 0 --end-seed 9
```

Useful environment overrides:

```bash
LISA_N_WARMUP=400 LISA_N_DRAWS=600 LISA_NUM_CHAINS=2 LISA_NT=32 \
  python lisa_study.py --seed 3
```

Each seed writes its products into `outdir_lisa/stationary_noise/seed_<seed>/`:

- `injection.npz` — A/E/T time series, PSD grids, injected source parameters,
  the reference frequency `f_ref`, the prior bounds, the seed, and the SNR summary.
- `data_overview.png` — frequency-domain characteristic strain (top) and WDM
  time-frequency power map (bottom).
- `freq_posterior.nc`, `wdm_posterior.nc` — posterior samples for
  `(f0, fdot, A, phi0)` (NetCDF written via `arviz_base`).
- `posterior_comparison.png` — frequency vs WDM marginal posteriors with the
  injected truth.
- `results.json` — per-parameter JSD and truth-rank diagnostics.

The aggregator writes into `outdir_lisa/stationary_noise/_summary/`:
`pp_plot.png`, `jsd_histogram.png`, `summary.csv`, and `summary.json`.

## Data Generation

### Data Model

The A-channel strain is modelled as

$$
d_A(t) = n_A(t) + h(t; \theta)
$$

where the stationary instrument-noise term $n_A$ and the single resolved
compact-binary signal $h$ are generated with a seed-controlled draw. The noise
PSD is the stationary TDI-1.5 instrument model $S_A(f) = S_A^{\mathrm{inst}}(f)$.

This is a conditional local-follow-up study, not a discovery search over broad
carrier frequency. The scripts assume an external matched-filter stage has
already localized the source near a fixed reference frequency $f_{\rm ref}$, and
the injected true carrier is drawn as a small log-jitter around that reference:

$$
f_0 = f_{\rm ref} \exp(\delta \log f_0),
\qquad
\delta \log f_0 \sim \mathrm{Uniform}[-w, +w].
$$

The same jitter width is then used in both the frequency-domain and WDM-domain
posteriors.

### Seed-0 data figure

`lisa_study.py` writes a two-panel A-channel figure: the frequency-domain
characteristic strain of the data, the instrument-noise model, and the injected
signal (top); and the WDM time-frequency power of the data with the injected
carrier overlaid (bottom).

![Seed-0 data overview](../../_static/lisa/data_overview.png)

## Frequency-Domain MCMC

For the frequency-domain MCMC, the injected source is fit in a narrow local band
around its carrier frequency. If $\tilde{d}_k$ is the A-channel FFT and
$\tilde{h}_k(\theta)$ is the band-limited template, the code uses the Whittle
approximation:

$$
\log p(d \mid \theta) \propto
-\sum_{k \in \mathcal{B}} \left[
\log S_k + \frac{4 \Delta f\, |\Delta t\,(\tilde{d}_k - \tilde{h}_k(\theta))|^2}{S_k}
\right]
$$

The sampled parameters are $(f_0, \dot{f}, A, \phi_0)$; sky position, polarization,
and inclination stay fixed at their injected values to isolate the local
likelihood machinery. `f0` is sampled as a narrow offset around the fixed external
reference `f_ref` (default $\delta f_0 \in [-3\times10^{-8}, 3\times10^{-8}]$ Hz
via `LISA_DELTA_F0_PRIOR_HALF_WIDTH`), parameterised as
`(delta_f0, logfdot, logA, phi0)`.

## WDM-Domain MCMC

### Likelihood

The WDM run uses the same injected A-channel data after truncating it to a length
compatible with the $(n_t, n_f)$ tiling. The transform produces coefficients
$w_{n,m} = \langle d, g_{n,m} \rangle$, where each $g_{n,m}$ is a localized
Wilson-Daubechies-Meyer atom. The study uses a diagonal Gaussian approximation
with analytic per-pixel variance:

$$
\log p(w \mid \theta) \propto -\frac{1}{2}
\sum_{n,m \in \mathcal{B}} \left[
\frac{(w_{n,m} - h_{n,m}(\theta))^2}{\Sigma_{n,m}} + \log(2\pi \Sigma_{n,m})
\right],
\qquad
\Sigma_{n,m} = \frac{S_n(f_m)}{2\,\Delta t} = S_n(f_m)\, f_{\rm Nyq}.
$$

### Fast WDM forward model

The naive WDM template path would build the full rFFT template, inverse-FFT to the
time domain, apply the full WDM transform, then crop to the local band. That is
far more work than NUTS needs. The runner instead (1) asks `JaxGB` only for the
local FFT bins containing the source, (2) embeds them into a small local FFT
buffer, and (3) applies only the WDM channels intersecting the inference band. In
operator form, with $P_{\mathcal B}$ the local FFT crop and $W_{\mathcal B}$ the
band-limited WDM transform,

$$
h^{\rm WDM}_{n,m}(\theta) = W_{\mathcal B}\, P_{\mathcal B}\, \tilde h(\theta),
$$

which is algebraically equivalent to the full transform restricted to the same
band but much cheaper to evaluate.

### Tiling choice

The tiling parameter $n_t$ sets the WDM frequency-channel spacing
$\Delta f_{\rm wdm} = n_t / (2 T_{\rm obs})$. The default $n_t = 32$ gives
$\Delta f_{\rm wdm} \approx 5.1 \times 10^{-7}$ Hz for a one-year observation —
coarse compared with the FFT bin width, but fine enough for these narrow local
fits once the likelihood normalization and sampler parameterization are correct.

## Comparison of Methods

The comparison step overlays the frequency- and WDM-domain marginal posteriors
against the injected truth (`arviz_plots.plot_dist`). The intended result is that
the two posteriors overlap closely in all four fitted parameters.

![Seed-0 posterior comparison](../../_static/lisa/posterior_comparison.png)

### Repeated-seed calibration

Single-run agreement is necessary but not sufficient. Across many seeds,
`collect_all_results.py` builds a PP plot of the posterior rank of the injected
truth (a well-calibrated sampler traces the diagonal within the shaded
binomial band) and a histogram of the per-parameter frequency-vs-WDM JSD.

![PP calibration plot](../../_static/lisa/pp_plot.png)

The checked-in figures are **snapshots**: if you change the sampler or
diagnostics, regenerate the per-seed outputs and re-aggregate before drawing
conclusions. Copy a representative seed's `data_overview.png` / `posterior_comparison.png`
and the `_summary/pp_plot.png` into `docs/_static/lisa/` to refresh this page.

## Source Code: `lisa_study.py`

```python
--8<-- "docs/studies/lisa/lisa_study.py"
```

## Notes

- `lisa_study.py` is a single, ordinary script: data generation, both inference
  runs, and the comparison all execute under one `--seed`.
- Heavy optional dependencies (`jax`, `numpyro`, `jaxgb`, `lisaorbits`,
  `arviz_base`, `arviz_plots`) are imported lazily inside the functions that need
  them, so `lisa_common.py` and the normalization tests import without them.
- `lisa_common.py` holds the run-directory conventions, the stationary noise PSD
  model, the injection (de)serialiser, and the posterior-rank diagnostics used by
  the PP plot. `gb_prior.py` holds the galactic-binary prior definitions.
