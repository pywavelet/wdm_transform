# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---
# ruff: noqa: E402, I001

# %% [markdown]
# # Monochromatic Signal in Time-Varying Noise: Gibbs Sampler
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/studies/wdm_monochromatic_gibbs.py)
#
# This notebook demonstrates joint inference of a monochromatic signal embedded
# in locally stationary (time-varying) noise using a **block Gibbs sampler** on
# the WDM coefficient grid.
#
# The observation model is
#
# $$
# x(t) = h(t;\,A,f_0,\varphi) + n(t),
# $$
#
# where the signal is a pure sinusoid
#
# $$
# h(t) = A\sin(2\pi f_0 t + \varphi)
# $$
#
# and the noise $n(t)$ is a locally stationary process whose time-varying PSD
# $S[n,m]$ is **unknown** and must be estimated alongside the signal parameters.
#
# After transforming to the WDM domain the observation becomes
#
# $$
# x[n,m] \approx h[n,m] + w[n,m], \qquad w[n,m] \sim \mathcal{N}(0,\,S[n,m]),
# $$
#
# which decouples into two tractable conditional posteriors:
#
# | Block | Condition on | Target |
# |-------|-------------|--------|
# | **Signal** | current $S[n,m]$ | $(A, f_0, \varphi)$ via WDM Whittle |
# | **Noise** | current $(A,f_0,\varphi)$ | $S[n,m]$ via Gamma Whittle on residual |
#
# Each block is sampled with NUTS; the two blocks alternate in a standard block
# Gibbs schedule.

# %%
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

if "google.colab" in sys.modules:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "jax[cpu]>=0.4.30", "numpyro>=0.15", "corner>=2.2",
            "ipywidgets>=8.1",
            "git+https://github.com/pywavelet/wdm_transform.git",
        ],
        check=True,
    )

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from wdm_transform import TimeSeries
from wdm_transform.backends import get_backend
from wdm_transform.transforms.jax_backend import _from_spectrum_to_wdm_impl
from wdm_transform.windows import phi_window

if "__file__" in globals():
    NOTEBOOK_DIR = Path(__file__).resolve().parent
else:
    cwd = Path.cwd()
    docs_studies_dir = cwd / "docs" / "studies"
    NOTEBOOK_DIR = docs_studies_dir if docs_studies_dir.exists() else cwd

FIGURE_OUTPUT_DIR = NOTEBOOK_DIR / "wdm_monochromatic_gibbs_assets"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 160) -> Path:
    """Save a notebook figure to the docs static directory and close it."""
    path = FIGURE_OUTPUT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## Shared helpers
#
# `simulate_tv_arma`, `PSplineConfig`, `compute_true_tv_psd`,
# `monte_carlo_reference_wdm_psd`, and `run_wdm_psd_mcmc` are loaded from the
# sibling PSD notebook. Only the function-definition cells are executed —
# the experiment block is excluded so docs builds stay fast.

# %%
import sys as _sys, types as _types

_psd_path = NOTEBOOK_DIR / "wdm_time_varying_psd.py"
_psd_raw  = _psd_path.read_text()

# Slice off everything from the first experiment cell onward.
_cutoff   = _psd_raw.find("# ## Experiment")
_psd_defs = _psd_raw[:_cutoff] if _cutoff != -1 else _psd_raw

_psd_mod = _types.ModuleType("wdm_time_varying_psd")
_psd_mod.__file__ = str(_psd_path)
# Register before exec so @dataclass can resolve cls.__module__ at decoration time.
_sys.modules["wdm_time_varying_psd"] = _psd_mod
exec(compile(_psd_defs, str(_psd_path), "exec"), _psd_mod.__dict__)

from wdm_time_varying_psd import (  # noqa: E402
    PSplineConfig,
    compute_true_tv_psd,
    monte_carlo_reference_wdm_psd,
    run_wdm_psd_mcmc,
    simulate_tv_arma,
)

# %% [markdown]
# ## Problem setup

# %%
RNG = np.random.default_rng(7)
dt = 0.1
nt = 32
n_total = 2048
nf = n_total // nt
dgp = "LS2"

# True signal parameters — A must be large enough for reasonable per-pixel SNR.
# With 8000 samples and nt=32 (nf=250), increased data per pixel improves estimation.
# A=3.0 gives sufficient SNR for signal recovery with the larger dataset.
A_TRUE   = 3.0
F0_TRUE  = 1.5    # Hz — sits between LS2 noise peaks
PHI_TRUE = 0.6    # rad

times = np.arange(n_total) * dt

# Define the clean signal first (no dependencies that can fail).
signal_clean = A_TRUE * np.sin(2.0 * np.pi * F0_TRUE * times + PHI_TRUE)

# Simulate locally stationary noise and combine.
noise = simulate_tv_arma(n_total, dgp=dgp, rng=RNG)
data  = signal_clean + noise

print(f"Signal RMS:  {np.std(signal_clean):.4f}")
print(f"Noise RMS:   {np.std(noise):.4f}")
print(f"Data SNR:    {np.std(signal_clean)/np.std(noise):.2f}")

# %% [markdown]
# ## Data overview

# %%
wdm_data = TimeSeries(data, dt=dt).to_wdm(nt=nt)
wdm_signal = TimeSeries(signal_clean, dt=dt).to_wdm(nt=nt)
wdm_noise = TimeSeries(noise, dt=dt).to_wdm(nt=nt)

fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True, sharey=True)
for ax, coeffs, title in [
    (axes[0], np.asarray(wdm_data.coeffs),   "Data  x[n,m]"),
    (axes[1], np.asarray(wdm_signal.coeffs), "Signal  h[n,m]"),
    (axes[2], np.asarray(wdm_noise.coeffs),  "Noise  w[n,m]"),
]:
    mesh = ax.pcolormesh(
        np.asarray(wdm_data.time_grid),
        np.asarray(wdm_data.freq_grid),
        np.log(coeffs**2 + 1e-8).T,
        shading="nearest", cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    fig.colorbar(mesh, ax=ax, label="log local power")
axes[0].set_ylabel("Frequency [Hz]")
_ = save_figure(fig, "data_overview")

# %% [markdown]
# ![Data overview](../wdm_monochromatic_gibbs_assets/data_overview.png)

# %% [markdown]
# ## JAX-differentiable WDM signal template
#
# The signal block needs to evaluate $h[n,m] = \mathrm{WDM}(h(t; A, f_0,
# \varphi))$ inside a NumPyro model, so it must be JAX-traceable.  We use the
# JIT-compiled `_from_spectrum_to_wdm_impl` kernel directly with a precomputed
# phi-window.

# %%
_jax_backend = get_backend("jax")
_win_np = phi_window(_jax_backend, nt, nf, dt, a=1.0 / 3.0, d=1.0)
WINDOW_JAX = jnp.asarray(_win_np, dtype=jnp.complex128)
TIMES_JAX  = jnp.asarray(times, dtype=jnp.float64)


@jax.jit
def wdm_of_signal(A, f0, phi):
    """WDM of A·sin(2π f0 t + φ) — JAX-differentiable, shape (nt, nf+1)."""
    h = A * jnp.sin(2.0 * jnp.pi * f0 * TIMES_JAX + phi)
    spectrum = jnp.fft.fft(h.astype(jnp.complex128))
    return _from_spectrum_to_wdm_impl(spectrum, WINDOW_JAX, nt, nf)


# Sanity check: WDM of the known signal should have most energy near f0
_h_wdm_check = np.asarray(wdm_of_signal(A_TRUE, F0_TRUE, PHI_TRUE))
_dominant_ch = int(np.argmax(np.sum(_h_wdm_check**2, axis=0)))
print(f"Dominant WDM channel for signal: m={_dominant_ch} "
      f"({np.asarray(wdm_data.freq_grid)[_dominant_ch]:.3f} Hz)")

# %% [markdown]
# ## Gibbs sampler components
#
# ### Block 1 — Signal parameters $(A, f_0, \varphi)$ given $S[n,m]$
#
# The WDM Whittle log-likelihood with a fixed noise surface is
#
# $$
# \log p(x[n,m] \mid A,f_0,\varphi,S)
# = -\frac{1}{2}\sum_{n,m}
#   \left[\log(2\pi S[n,m]) + \frac{(x[n,m]-h[n,m])^2}{S[n,m]}\right].
# $$
#
# Because $h[n,m]$ is linear in $A$ and smoothly nonlinear in $f_0$ and
# $\varphi$, NUTS handles all three efficiently.

# %%
def signal_model(
    d_wdm_trim: jnp.ndarray,
    S_trim: jnp.ndarray,
    keep_time: np.ndarray,
    keep_freq: np.ndarray,
    f0_lo: float,
    f0_hi: float,
) -> None:
    """NumPyro model for (A, f0, phi) given a fixed noise surface S."""
    A   = numpyro.sample("A",   dist.HalfNormal(2.0))
    f0  = numpyro.sample("f0",  dist.Uniform(f0_lo, f0_hi))
    phi = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

    h_wdm_full = wdm_of_signal(A, f0, phi)
    h_trim = h_wdm_full[keep_time[0]:keep_time[-1] + 1,
                         keep_freq[0]:keep_freq[-1] + 1]

    log_like = -0.5 * jnp.sum(
        (d_wdm_trim - h_trim) ** 2 / S_trim
        + jnp.log(2.0 * jnp.pi * S_trim)
    )
    numpyro.factor("whittle_signal", log_like)


def run_signal_mcmc(
    d_wdm_trim: np.ndarray,
    S_trim: np.ndarray,
    keep_time: np.ndarray,
    keep_freq: np.ndarray,
    *,
    f0_lo: float = 0.5,
    f0_hi: float = 3.0,
    init_A: float,
    init_f0: float,
    init_phi: float,
    n_warmup: int = 300,
    n_samples: int = 400,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """NUTS on (A, f0, phi) with S fixed."""
    kernel = NUTS(
        signal_model,
        init_strategy=init_to_value(
            values={"A": init_A, "f0": init_f0, "phi": init_phi}
        ),
        target_accept_prob=0.85,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(seed),
        jnp.asarray(d_wdm_trim),
        jnp.asarray(S_trim),
        keep_time,
        keep_freq,
        f0_lo,
        f0_hi,
    )
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


# %% [markdown]
# ### Block 2 — Noise PSD $S[n,m]$ given signal parameters
#
# Subtract the current signal estimate from the raw data, transform the
# residual to the WDM domain, and run the Gamma Whittle spline smoother from
# `wdm_time_varying_psd.py`.  The previous $S$ estimate is used as the warm
# start to speed up convergence.

# %%
def run_noise_mcmc(
    data: np.ndarray,
    A: float,
    f0: float,
    phi: float,
    *,
    dt: float,
    nt: int,
    config: PSplineConfig,
    n_warmup: int = 200,
    n_samples: int = 200,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Fit S[n,m] to the residual x(t) - h(t; A, f0, phi)."""
    residual = data - A * np.sin(2.0 * np.pi * f0 * times + phi)
    return run_wdm_psd_mcmc(
        residual,
        dt=dt,
        nt=nt,
        config=config,
        n_warmup=n_warmup,
        n_samples=n_samples,
        num_chains=1,
        random_seed=seed,
    )


# %% [markdown]
# ## Initialisation
#
# Warm-start $S[n,m]$ by fitting the Gamma Whittle model to the raw data
# (signal + noise).  This is a conservative over-estimate of the noise PSD but
# gives NUTS a sensible starting point.

# %%
config_gamma = PSplineConfig(periodogram_freq_half_width=1)

print("Initialising S[n,m] from raw data (Gamma Whittle)…")
init_results = run_wdm_psd_mcmc(
    data, dt=dt, nt=nt, config=config_gamma,
    n_warmup=250, n_samples=250, num_chains=1, random_seed=0,
)

# Trim indices and trimmed data WDM
keep_time = init_results["keep_time"]
keep_freq = init_results["keep_freq"]
d_wdm_trim = init_results["coeffs_fit"]       # shape (nt_trim, nf_trim)
# Arithmetic mean of S (not geometric) — see the Gibbs loop for rationale.
_init_log_psd = np.asarray(init_results["samples"]["log_psd"])
S_current = np.mean(np.exp(_init_log_psd), axis=0)

print(f"Trimmed WDM grid: {d_wdm_trim.shape}")
print(f"Initial S range: [{S_current.min():.3f}, {S_current.max():.3f}]")

# %% [markdown]
# ## Gibbs iterations

# %%
N_GIBBS = 25         # number of full Gibbs sweeps
N_SIGNAL_WARMUP  = 200  # per iteration — reduced because init warms up fast after iter 1
N_SIGNAL_SAMPLES = 200
N_NOISE_WARMUP   = 150
N_NOISE_SAMPLES  = 150

# Frequency search range: two WDM bin widths around the dominant channel
delta_f = float(np.asarray(wdm_data.freq_grid)[1] - np.asarray(wdm_data.freq_grid)[0])
F0_LO = max(0.1, float(np.asarray(wdm_data.freq_grid)[_dominant_ch]) - 4 * delta_f)
F0_HI = float(np.asarray(wdm_data.freq_grid)[_dominant_ch]) + 4 * delta_f
print(f"Signal frequency search range: [{F0_LO:.3f}, {F0_HI:.3f}] Hz")

# Running state
A_current   = 0.5 * float(np.std(signal_clean))   # conservative amplitude start
f0_current  = float(np.asarray(wdm_data.freq_grid)[_dominant_ch])
phi_current = 0.0

all_signal_samples: list[dict[str, np.ndarray]] = []
gibbs_trace: list[dict[str, float]] = []

for gibbs_iter in range(N_GIBBS):
    print(f"\n── Gibbs iteration {gibbs_iter + 1}/{N_GIBBS} ──")

    # Block 1: Sample (A, f0, phi) | S
    print(f"  Signal block (NUTS)…", end=" ", flush=True)
    sig_samples = run_signal_mcmc(
        d_wdm_trim, S_current,
        keep_time, keep_freq,
        f0_lo=F0_LO, f0_hi=F0_HI,
        init_A=A_current,
        init_f0=f0_current,
        init_phi=phi_current,
        n_warmup=N_SIGNAL_WARMUP,
        n_samples=N_SIGNAL_SAMPLES,
        seed=gibbs_iter * 10,
    )
    all_signal_samples.append(sig_samples)

    A_current   = float(np.median(sig_samples["A"]))
    f0_current  = float(np.median(sig_samples["f0"]))
    phi_current = float(np.median(sig_samples["phi"]))
    print(f"A={A_current:.4f}, f0={f0_current:.4f}, phi={phi_current:.4f}")

    # Block 2: Sample S | (A, f0, phi)
    print(f"  Noise block (Gamma Whittle)…", end=" ", flush=True)
    noise_res = run_noise_mcmc(
        data, A_current, f0_current, phi_current,
        dt=dt, nt=nt, config=config_gamma,
        n_warmup=N_NOISE_WARMUP,
        n_samples=N_NOISE_SAMPLES,
        seed=gibbs_iter * 10 + 1,
    )
    # Use the arithmetic mean of S, not the geometric mean (psd_mean).
    # The signal likelihood involves 1/S, so Jensen's inequality makes the
    # geometric mean (= exp(E[log S])) a biased plug-in estimate.
    log_psd_samples = np.asarray(noise_res["samples"]["log_psd"])
    S_current = np.mean(np.exp(log_psd_samples), axis=0)
    print(f"S range: [{S_current.min():.3f}, {S_current.max():.3f}]")

    gibbs_trace.append({"A": A_current, "f0": f0_current, "phi": phi_current})

print("\nGibbs sampler complete.")

# %% [markdown]
# ## Convergence trace

# %%
iters = np.arange(1, N_GIBBS + 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, key, truth, label in [
    (axes[0], "A",   A_TRUE,   "Amplitude  A"),
    (axes[1], "f0",  F0_TRUE,  "Frequency  f₀ [Hz]"),
    (axes[2], "phi", PHI_TRUE, "Phase  φ [rad]"),
]:
    values = [row[key] for row in gibbs_trace]
    ax.plot(iters, values, marker="o", color="tab:blue", lw=2.0)
    ax.axhline(truth, color="tab:red", ls="--", lw=1.5, label="True value")
    ax.set_title(label)
    ax.set_xlabel("Gibbs iteration")
    ax.legend()
_ = save_figure(fig, "gibbs_trace")

# %% [markdown]
# ![Gibbs convergence trace](../wdm_monochromatic_gibbs_assets/gibbs_trace.png)

# %% [markdown]
# ## Posterior of signal parameters (pooled second half of chain)
#
# Discard the first half of Gibbs iterations as burn-in and pool the
# within-iteration NUTS samples from the remaining sweeps.

# %%
try:
    import corner
    _have_corner = True
except ImportError:
    _have_corner = False

_burn = N_GIBBS // 2
pooled = {
    k: np.concatenate([s[k] for s in all_signal_samples[_burn:]], axis=0)
    for k in ("A", "f0", "phi")
}
chain = np.column_stack([pooled["A"], pooled["f0"], pooled["phi"]])
print(f"Pooled {len(chain)} NUTS samples from Gibbs iterations {_burn+1}–{N_GIBBS}")

if _have_corner:
    fig = corner.corner(
        chain,
        labels=["A", "f₀ [Hz]", "φ [rad]"],
        truths=[A_TRUE, F0_TRUE, PHI_TRUE],
        truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
    )
    _ = save_figure(fig, "signal_posterior_corner")
else:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, samples, truth, label in [
        (axes[0], pooled["A"],   A_TRUE,   "A"),
        (axes[1], pooled["f0"],  F0_TRUE,  "f₀ [Hz]"),
        (axes[2], pooled["phi"], PHI_TRUE, "φ [rad]"),
    ]:
        ax.hist(samples, bins=40, density=True, color="tab:blue", alpha=0.7)
        ax.axvline(truth, color="tab:red", ls="--", lw=2, label="True")
        ax.axvline(np.median(samples), color="tab:blue", ls="-", lw=2, label="Median")
        ax.set_title(label)
        ax.legend()
    _ = save_figure(fig, "signal_posterior_corner")

# %% [markdown]
# ![Signal parameter posterior](../wdm_monochromatic_gibbs_assets/signal_posterior_corner.png)

# %% [markdown]
# ## Recovered noise PSD surface

# %%
# Compute ground-truth MC reference for the noise process alone
noise_reference = monte_carlo_reference_wdm_psd(
    n_draws=60, n_total=n_total, dt=dt, nt=nt, dgp=dgp, seed=99,
    config=config_gamma,
)
true_psd_grid = compute_true_tv_psd(
    dgp,
    noise_res["time_grid"],
    2.0 * np.pi * dt * noise_res["freq_grid"],
)

_vmin = np.log(noise_reference + 1e-8).min()
_vmax = np.log(noise_reference + 1e-8).max()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True, sharey=True)
for ax, surface, title in [
    (axes[0], np.log(noise_res["psd_mean"]  + 1e-8), "Gibbs noise estimate (final)"),
    (axes[1], np.log(noise_reference + 1e-8),         "MC reference  E[w²]"),
    (axes[2], np.log(true_psd_grid   + 1e-8),         "True PSD  S(e^{jω})"),
]:
    mesh = ax.pcolormesh(
        noise_res["time_grid"],
        noise_res["freq_grid"],
        surface.T,
        shading="nearest", cmap="viridis", vmin=_vmin, vmax=_vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Rescaled WDM Time")
    fig.colorbar(mesh, ax=ax, label="log local power")
axes[0].set_ylabel("Frequency [Hz]")
_ = save_figure(fig, "noise_psd_surface")

# %% [markdown]
# ![Recovered noise PSD surface](../wdm_monochromatic_gibbs_assets/noise_psd_surface.png)

# %% [markdown]
# ## Signal recovery

# %%
h_recovered = A_current * np.sin(2.0 * np.pi * f0_current * times + phi_current)

fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

ax = axes[0]
ax.plot(times, signal_clean, color="tab:green", lw=2.0, label="True signal")
ax.plot(times, h_recovered,  color="tab:blue",  lw=1.5, ls="--", label="Recovered signal")
ax.set_title("Signal recovery (time domain)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.legend()

# Residual after signal subtraction vs noise-only
residual_gibbs = data - h_recovered
ax = axes[1]
ax.plot(times, noise,          color="tab:orange", lw=1.0, alpha=0.7, label="True noise")
ax.plot(times, residual_gibbs, color="tab:blue",   lw=1.0, alpha=0.7, label="Data − recovered signal")
ax.set_title("Residual vs true noise")
ax.set_xlabel("Time [s]")
ax.legend()

_ = save_figure(fig, "signal_recovery")

# %% [markdown]
# ![Signal recovery](../wdm_monochromatic_gibbs_assets/signal_recovery.png)

# %% [markdown]
# ## Baseline: Frequency-domain Whittle with stationary noise
#
# As a baseline we run signal inference with the simplest possible noise model:
# a *time-stationary* PSD estimated once from the raw data using Welch's method,
# then fixed throughout.  The likelihood is the standard frequency-domain Whittle:
#
# $$
# \log p(x \mid A, f_0, \varphi)
#   \approx -\frac{1}{2}\sum_{k=0}^{N-1}
#   \frac{|X_k - H_k(A,f_0,\varphi)|^2}{\hat{S}(|f_k|)},
# $$
#
# where $X_k = \mathrm{FFT}(x)_k$, $H_k = \mathrm{FFT}(h)_k$, and
# $\hat{S}(f)$ is the Welch one-sided PSD converted to FFT-bin variance units
# $\hat{S}_{\rm bin}(f) = \hat{S}_{\rm Welch}(f) \cdot f_s / 2$.
#
# The LS2 noise is time-varying: the PSD at $f_0 = 1.5\,\mathrm{Hz}$ changes
# across time bins.  Welch averages over all time, so the stationary estimate
# misspecifies the noise structure — the Gibbs comparison shows the cost.
#
# > **Note:** the Welch estimate is run on the raw data (signal + noise), so it
# > carries a narrow spike at $f_0$ from the signal itself.  A refined version
# > would iterate (subtract signal, re-estimate PSD), but a single-shot estimate
# > is the realistic "naive" baseline.

# %%
from scipy.signal import welch as scipy_welch

# Welch PSD estimate from raw data — one-sided, units amplitude²/Hz
_nperseg = max(256, n_total // 8)
f_welch_stat, S_welch_stat = scipy_welch(
    data, fs=1.0 / dt, nperseg=_nperseg, scaling="density"
)

# Convert to FFT-bin variance: E[|X_k|²] = S_welch(|f_k|) * fs / 2
# where X_k = jnp.fft.fft(x)[k]  (unnormalized sum convention)
_freqs_full = np.fft.fftfreq(n_total, d=dt)
S_fft_stat_np = np.interp(np.abs(_freqs_full), f_welch_stat, S_welch_stat) / (2.0 * dt)
S_fft_stat_jax = jnp.asarray(S_fft_stat_np)
X_fft_data_jax = jnp.asarray(np.fft.fft(data))

# Quick sanity: the mean of S_fft_stat_np / n_total should match var(noise)
_psd_var_check = float(np.mean(S_fft_stat_np) / n_total)
print(f"PSD-implied variance: {_psd_var_check:.4f}  |  empirical var(data): {np.var(data):.4f}")

# %% [markdown]
# ### Stationary NUTS

# %%
def stationary_whittle_model(X_data, S_bins, f0_lo, f0_hi):
    """Standard FFT Whittle likelihood with a fixed stationary noise PSD."""
    A   = numpyro.sample("A",   dist.HalfNormal(2.0))
    f0  = numpyro.sample("f0",  dist.Uniform(f0_lo, f0_hi))
    phi = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

    h = A * jnp.sin(2.0 * jnp.pi * f0 * TIMES_JAX + phi)
    diff = X_data - jnp.fft.fft(h)
    numpyro.factor("whittle_fft", -0.5 * jnp.sum(jnp.abs(diff) ** 2 / S_bins))


kernel_stat = NUTS(
    stationary_whittle_model,
    init_strategy=init_to_value(
        values={"A": A_current, "f0": f0_current, "phi": phi_current}
    ),
    target_accept_prob=0.85,
)
mcmc_stat = MCMC(
    kernel_stat,
    num_warmup=500,
    num_samples=800,
    num_chains=1,
    progress_bar=False,
)
mcmc_stat.run(random.PRNGKey(999), X_fft_data_jax, S_fft_stat_jax, F0_LO, F0_HI)
stat_samples = {k: np.asarray(v) for k, v in mcmc_stat.get_samples().items()}
print("Stationary FFT Whittle NUTS complete.")
for key, truth in [("A", A_TRUE), ("f0", F0_TRUE), ("phi", PHI_TRUE)]:
    med = float(np.median(stat_samples[key]))
    q5, q95 = np.percentile(stat_samples[key], [5, 95])
    print(f"  {key}: median={med:.4f}  90% CI=[{q5:.4f}, {q95:.4f}]  (true={truth})")

# %% [markdown]
# ### Posterior comparison: Gibbs (time-varying) vs Stationary FFT Whittle

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
_labels = ["Amplitude  A", "Frequency  f₀ [Hz]", "Phase  φ [rad]"]
_keys   = ["A",             "f0",                  "phi"]
_truths = [A_TRUE,           F0_TRUE,               PHI_TRUE]

for ax, key, truth, label in zip(axes, _keys, _truths, _labels):
    ax.hist(
        pooled[key], bins=60, density=True,
        color="tab:blue", alpha=0.55, label="Gibbs (WDM, time-varying noise)",
    )
    ax.hist(
        stat_samples[key], bins=60, density=True,
        color="tab:orange", alpha=0.55, label="Stationary FFT Whittle",
    )
    ax.axvline(truth, color="tab:red", ls="--", lw=2.0, label="True value")
    ax.set_title(label)
    ax.set_xlabel(label)

axes[0].legend(fontsize=8)
_ = save_figure(fig, "stationary_vs_gibbs_comparison")

# %% [markdown]
# ![Stationary vs Gibbs comparison](../wdm_monochromatic_gibbs_assets/stationary_vs_gibbs_comparison.png)

# %% [markdown]
# ## Summary
#
# **What the Gibbs sampler does:**
#
# 1. *Initialisation* — fit a Gamma Whittle PSD surface to the raw data
#    (signal + noise) to get a conservative warm start for $S[n,m]$.
# 2. *Signal block* — NUTS on $(A, f_0, \varphi)$ using the WDM Whittle
#    likelihood $x[n,m] \sim \mathcal{N}(h[n,m],\,S[n,m])$ with $S$ fixed.
#    The JAX backend makes $h[n,m]$ fully differentiable in all three parameters.
# 3. *Noise block* — subtract the median signal estimate, transform the residual
#    to WDM, and run the Gamma Whittle spline smoother to update $S[n,m]$.
# 4. *Repeat* — alternate blocks until the signal parameters and noise surface
#    converge.
#
# **Limitations and next steps:**
#
# - Only a point estimate of $S$ is passed between Gibbs iterations (posterior
#   mean of the noise block). A fully Bayesian Gibbs sampler would propagate
#   uncertainty in $S$ into the signal block, which requires either storing
#   multiple $S$ draws per iteration or marginalising analytically.
# - The frequency prior `Uniform(F0_LO, F0_HI)` is deliberately narrow to
#   speed up the demo. For a blind search, broaden the range and increase NUTS
#   warmup steps.
# - For stronger signals or well-separated frequency content, the WDM tiling
#   (`nt`) and the smoothing config (`periodogram_freq_half_width`) are the
#   primary levers for trading time resolution against frequency resolution.
