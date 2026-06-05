# LISA galactic-binary study: frequency domain vs WDM domain

This study answers one question: **does Bayesian parameter estimation for a
resolved LISA galactic binary (GB) give the same posterior whether the
likelihood is evaluated in the frequency domain or in the WDM (Wilson–
Daubechies–Meyer) time–frequency domain?**

The headline result, over 100 independent injections with SNR drawn uniformly in
[20, 45]:

- **Frequency and WDM posteriors are statistically identical** — the per-seed
  Jensen–Shannon divergence (JSD) between the two domains' marginals is
  ≈ 3 × 10⁻⁴ bits (median), ≤ 0.003 bits at the 95th percentile (0 = identical,
  1 = disjoint).
- **Runs converge reliably** — zero divergences in all 200 runs, and split-R̂ <
  1.05 in every coordinate on 99 of 100 seeds (both domains). The one exception
  is a stiff `f0–fdot` source that under-mixes *identically* in both domains, so
  its freq-vs-WDM JSD is still ~0 (the equivalence is unaffected). The
  posterior–vs–prior widths show the data, not the prior, drives `fdot`, `A`,
  `phi0` (and the joint `f0–fdot` ridge).

In short: the WDM likelihood is a faithful drop-in for the frequency-domain
Whittle likelihood for GB inference.

---

## What the pipeline does (per seed)

1. **Draw a source.** A chirping GB with parameters `(f0, fdot, A, phi0)` plus
   sky/orientation angles `(ra, dec, psi, iota)`. `f0` is drawn near a reference
   frequency, `fdot` is drawn to give a *few-Fourier-bin* frequency drift over
   the run (see "Why fdot is small but matters"), and the sky/phase are random.
2. **Set the SNR.** A target SNR is drawn from `[SNR_MIN, SNR_MAX]` and the
   amplitude `A` is rescaled to hit it.
3. **Inject.** The source is generated with **JaxGB** on the A/E/T TDI channels
   (full LISA orbital response), and stationary **colored LISA instrument
   noise** (the TDI-1.5 PSD) is added in each channel.
4. **Infer twice.** `(f0, fdot, A, phi0)` are inferred with NumPyro/NUTS using
   two likelihoods on the same data:
   - **Frequency domain:** a banded Whittle likelihood on the rFFT.
   - **WDM domain:** the library's `forward_wdm_band` sub-band transform of the
     same band, with a per-channel analytic WDM noise variance
     `σ²_{nm} = N · S(f_m) / (2 dt)`.
   Sky/orientation are held fixed at their injected values (a follow-up scenario
   where a prior search has localized the source); only `(f0, fdot, A, phi0)`
   are sampled.
5. **Compare.** Posterior marginals are compared between domains (JSD, PP plot),
   and convergence (R̂, ESS, divergences) is recorded for every run.

---

## Two design choices worth understanding

These are the parts of the code that look unusual at first glance. Both exist to
make NUTS sample a **sharp, strongly-correlated** GB posterior reliably on *every*
seed, not just most.

### 1. Why we sample `(g_c, g_s)` instead of `(A, phi0)`

The amplitude `A` and initial phase `phi0` are awkward coordinates for a
gradient sampler, for two reasons:

- **The amplitude–phase ridge is curved.** Holding the fit quality fixed,
  `(A, phi0)` trace out a *curve* (a circle/banana), not an ellipse. A sampler
  with a single fixed "mass matrix" (metric) can only precondition an
  *elliptical* (Gaussian) posterior; on a curved ridge it mixes poorly and can
  stall.
- **`phi0` is periodic.** It lives on `(-π, π]` with a hard wrap. When the true
  phase sits near ±π, that boundary both splits the posterior across the cut and
  (via NumPyro's bounded-parameter reparameterization) distorts the geometry the
  mass matrix sees, inflating R̂.

The fix uses a fact about the GB signal: the strain is **exactly bilinear** in a
Cartesian version of the amplitude/phase,

```
g_c = A cos(phi0),   g_s = A sin(phi0)
h   = g_c · h_c  +  g_s · h_s
```

where `h_c`, `h_s` are two *fixed* quadrature waveforms (the source evaluated at
`phi0 = 0` and `phi0 = π/2`). Because `h` is **linear** in `(g_c, g_s)`, the
Gaussian-noise likelihood is **exactly Gaussian** (a clean ellipse) in those
coordinates — no curvature, no periodic boundary. NUTS samples an ellipse
trivially. We then recover the physical parameters deterministically:
`A = √(g_c² + g_s²)`, `phi0 = atan2(g_s, g_c)`.

This is the same linear-amplitude idea behind the gravitational-wave
**F-statistic** (which *analytically marginalizes* these coordinates), except we
keep `(g_c, g_s)` as *sampled* coordinates. That choice is deliberate: it keeps a
full joint posterior and compares the two domains on identical footing, while
still removing the pathological geometry.

Two implementation notes:
- For a band-limited (analytic, positive-frequency) signal the π/2 quadrature is
  exactly `h_s = −i · h_c` in the rFFT domain, so the expensive waveform is
  evaluated **once** per likelihood call, not twice.
- `(g_c, g_s)` are **standardized** by the source amplitude (`g = A_ref · z_g`,
  with `z_g ~ N(0, 3)`), so they are O(1) like the frequency coordinates. This
  matters for the mass matrix — see below. As a side effect, an isotropic
  Gaussian on `(g_c, g_s)` is an exactly **uniform** prior on `phi0`.

### 2. Why we seed NUTS with an analytic Fisher mass matrix (instead of letting it adapt)

The remaining sampled parameters `(f0, fdot)` form a **razor-thin, strongly
correlated ridge**. Over a long baseline a sub-bin shift in `f0` changes the
template phase by many radians, and `f0`/`fdot` are ~98% anti-correlated (they
are the linear and quadratic coefficients of the same phase polynomial). The
posterior is a tilted needle whose width is a tiny fraction of a frequency bin.

NUTS needs a **mass matrix** (a metric) matched to this covariance to take
efficient steps. The standard approach is `adapt_mass_matrix=True`: let NUTS
*learn* the dense metric during warmup. On a near-singular ridge this hits a
chicken-and-egg failure — with a poor initial metric the chain barely moves, so
warmup cannot estimate the covariance, so the metric never improves. In practice
we observed exactly this: chains collapsed to sub-regions, posterior widths
flipped run-to-run, and R̂ blew up.

Instead we **compute the metric analytically**. For a locally-Gaussian
posterior, the **Fisher information matrix** — the Hessian of the negative
log-likelihood — *is* the inverse posterior covariance. We evaluate it once at
the reference parameters (the prior-search guess) and hand its inverse to NUTS as
a **frozen** dense mass matrix (`adapt_mass_matrix=False`). This gives a
correctly scaled, correctly correlated metric from the very first step, so the
chain moves along the ridge immediately — no warmup learning required.

So "why not just use a dense adaptive matrix?" — the dense matrix is exactly what
we want; the problem is *learning* it on this stiff geometry. The Fisher gives the
same matrix for free and sidesteps the warmup failure. Two details make it
robust:

- **Standardized coordinates.** The Fisher is built in `(z_f0, z_fdot, z_gc,
  z_gs)`, all O(1). Without this the amplitude coordinates (~10⁻²³) and the
  frequency coordinates (O(1)) would give a ~10⁴⁶ condition number, and the
  positive-definiteness floor would corrupt the matrix.
- Combined with the `(g_c, g_s)` reparameterization above, the posterior is close
  to Gaussian, so the single analytic Fisher is an excellent *global* metric, not
  just a local one. Chains are also initialized at the reference with a small
  Fisher-scaled jitter.

### Aside: why `fdot` is small but still matters

A GW-driven GB has `fdot ~ 10⁻¹⁷–10⁻¹⁵ Hz/s` — a frequency *drift* of only
~0.1–1 Fourier bins over a year (a few bins over the full mission). It is
**measured from the phase** (detectable once the drift exceeds ~`1/(π·SNR)`
bins), and does **not** need to move the source across WDM frequency channels. We
draw a physical few-bin drift: large enough to be measurable and to keep the
`f0–fdot` ridge short, small enough that the chirp stays inside the analysis
band (which is auto-widened to contain the drift).

---

## How to run

```bash
# one seed (writes outdir_gb/seed_0.json)
lisa_venv/bin/python lisa_gb_study.py --seed 0

# the full population study (per-seed processes, resumable) + summary + PP plot
./run_gb_study.sh 100

# re-aggregate / re-plot a finished study without rerunning
lisa_venv/bin/python summarize_gb_study.py

# paper figures for one seed (data overview, freq-vs-WDM corner) + PP + JSD table
lisa_venv/bin/python plot_gb_study.py --seed 0
```

`summarize_gb_study.py` prints the calibration table, the (KDE-estimated)
freq-vs-WDM JSD table, a **convergence certificate** (how many seeds reach
R̂ < 1.05 with 0 divergences, naming any that fail), and a config-consistency
check; it also regenerates the PP plot.

---

## Files

| file | role |
|---|---|
| `lisa_gb_study.py` | study engine: source draw, injection, freq + WDM likelihoods, the `(z_f0, z_fdot, z_gc, z_gs)` NUTS sampler with the Fisher mass matrix, and per-seed R̂/ESS/divergence diagnostics |
| `run_gb_study.sh` | one-command N-seed runner (one process per seed → bounded memory, crash-resistant, resumable), then runs the summary |
| `summarize_gb_study.py` | aggregate `outdir_gb/`: calibration + JSD + convergence + config check + PP plot |
| `plot_gb_study.py` | per-seed data-overview figure, freq-vs-WDM corner (with prior overlay), population PP plot, JSD table |
| `eval_seeds.py` | quick in-process convergence sweep over a seed range (development tool) |
| `lisa_common.py` | shared helpers: LISA TDI PSD, physical rFFT noise draw, posterior ranks |
| `gb_prior.py` | reference frequency `F0_REF` and prior helpers |
| `outdir_gb/` | per-seed `seed_N.json` (posterior summaries, thinned samples, diagnostics, config) and figures |

---

## Configuration (module constants)

Study settings live in module-level constants in `lisa_gb_study.py`.

| constant | default | meaning |
|---|---|---|
| `NBLOCKS` | 4096 | baseline length in WDM blocks (`2·NT` samples each); 4096 with the current grid gives the power-of-two study baseline used for production runs. |
| `SNR_MIN` / `SNR_MAX` | 20 / 45 | per-seed injected SNR range |
| `CHANNELS` | `(0, 1, 2)` | TDI channels analyzed jointly (`A,E,T`) |
| `N_WARMUP` / `N_DRAWS` / `NUM_CHAINS` | 1500 / 1000 / 2 | NUTS settings |
| `F0_PRIOR_HALF_BINS` / `FDOT_PRIOR_HALF_BINS` | 1 / 40 | prior box half-widths in resolution elements |
| `F0_PRIOR_SIGMA_BINS` / `FDOT_PRIOR_SIGMA_BINS` | 0.32 / 10 | standardized-prior scales in resolution elements |
| `FDOT_DRIFT_BINS` | `(1, 5)` | injected `fdot` drift range, in Fourier bins |

---

## Scope and caveats

- **Sky/orientation are fixed** at the injected values (follow-up after a search
  localizes the source). The study tests `(f0, fdot, A, phi0)` recovery and the
  freq-vs-WDM equivalence, **not** sky localization or marginalization over the
  four extrinsic angles — that is a larger inference problem.
- The default baseline (`NBLOCKS=4096`) is chosen as a power-of-two number of
  WDM blocks for faster FFT-heavy likelihood evaluations. The comparison is
  **scale-faithful** — priors
  are defined on the resolution scale (`1/T_obs`, `1/T_obs²`) and SNR is fixed —
  so the normalized freq-vs-WDM result is baseline-independent; a short baseline
  (`NBLOCKS=256`) is only for fast development.
- The `f0`/`fdot` priors are centered on the injected (search-guess) values and
  are deliberately narrow in `f0`; this is the realistic follow-up regime, not a
  blind all-sky search.
