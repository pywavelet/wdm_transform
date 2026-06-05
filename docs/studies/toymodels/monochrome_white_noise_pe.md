# Monochromatic White-Noise PE

Executable script: [`monochrome_white_noise_pe.py`](./monochrome_white_noise_pe.py).

This toy study is a small analogue of the LISA comparison. It injects one
chirping monochromatic signal,

```text
h(t) = A sin(2 pi (f t + 0.5 fdot t^2) + phi)
```

into white Gaussian noise and samples the four parameters `(A, f, fdot, phi)`
with two likelihoods:

- a full FFT-domain likelihood, using Parseval's theorem for white noise
- a diagonal WDM-domain likelihood, with per-coefficient noise variances
  calibrated by Monte Carlo white-noise draws through the WDM transform

Both domains are sampled with NumPyro/NUTS. The WDM model calls the package's
JAX WDM transform inside the NumPyro model, so the sampler sees a differentiable
WDM-domain likelihood.

Run a quick smoke test:

```bash
uv run --extra docs python docs/studies/toymodels/monochrome_white_noise_pe.py --quick
```

Run the default study:

```bash
uv run --extra docs python docs/studies/toymodels/monochrome_white_noise_pe.py
```

Outputs are written to `docs/studies/toymodels/outdir_monochrome_white_noise_pe/`:

- `summary.json`
- `posterior_samples.npz`
- `data_views.png`
- `posterior_comparison.png`

![Data views](outdir_monochrome_white_noise_pe/data_views.png)

![Posterior comparison](outdir_monochrome_white_noise_pe/posterior_comparison.png)
