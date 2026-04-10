# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Frequency-Domain GB Inference With NumPyro
#
# This study loads the cached TDI1.5 A-channel background from
# `data_generation.py`, injects two Galactic binaries with `JaxGB`, and performs
# local frequency-domain Bayesian inference in a Whittle approximation.
#
# The intended workflow is:
#
# 1. generate the background with `data_generation.py`
# 2. inject the resolved binaries with `JaxGB`
# 3. infer `(A, f0, fdot, phi0)` for each source on a narrow frequency window
# 4. sample each source independently with NumPyro NUTS
#
# The sky position, polarization, and inclination are held fixed at their
# injected values in this first version. Since the two injected GBs are
# separated in frequency, each source is fit independently and no Gibbs loop is
# used.

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import corner
import jax
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    BACKGROUND_REALIZATION_PATH,
    FREQ_ASSET_DIR,
    RESPONSE_TENSOR_PATH,
    freqs_gal,
    noise_tdi15_a_psd,
    save_figure,
    tdi15_factor,
    wrap_phase,
)
from numpyro.infer import MCMC, NUTS, init_to_value

jax.config.update("jax_enable_x64", True)


FIGURE_OUTPUT_DIR = FREQ_ASSET_DIR
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SourceConfig:
    label: str
    params: np.ndarray
    prior_A: tuple[float, float]
    prior_f0: tuple[float, float]
    prior_fdot: tuple[float, float]
    prior_phi0: tuple[float, float]


@dataclass(frozen=True)
class BandData:
    label: str
    idx: np.ndarray
    freqs: np.ndarray
    data: np.ndarray
    noise_psd: np.ndarray
    data_jax: jax.Array
    noise_psd_jax: jax.Array
    fixed_params: np.ndarray
    source_index: int
    band_kmin: int
    band_kmax: int
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logA_bounds: tuple[float, float]


BACKGROUND_PATH = BACKGROUND_REALIZATION_PATH
if not BACKGROUND_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached background at {BACKGROUND_PATH}. "
        "Run data_generation.py first."
    )

background = np.load(BACKGROUND_PATH)
dt = float(background["dt"])
t_obs = float(background["T_obs"])
freqs_full = np.asarray(background["freqs_all"], dtype=float)
background_A = np.asarray(background["data_Af"], dtype=np.complex128)
source_params = np.asarray(background["source_params"], dtype=float)
n_freq = len(freqs_full)
df = 1.0 / t_obs

RTILDE_PATH = RESPONSE_TENSOR_PATH
if not RTILDE_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached response tensor at {RTILDE_PATH}. "
        "Run data_generation.py first."
    )

rtilde = np.load(RTILDE_PATH)["Rtildeop_tf"]
freqs_response = freqs_gal(nfreqs=rtilde.shape[-1])
foreground_psd_time = np.abs(rtilde[0, 0]) * tdi15_factor(freqs_response)[None, :]
foreground_psd_mean = np.interp(
    freqs_full,
    freqs_response,
    np.mean(foreground_psd_time, axis=0),
    left=0.0,
    right=0.0,
)
noise_psd_full = np.maximum(noise_tdi15_a_psd(freqs_full) + foreground_psd_mean, 1e-60)

print(f"Loaded background from {BACKGROUND_PATH.name}")
print(f"Loaded response tensor from {RTILDE_PATH.name}")
print(f"T_obs = {t_obs / 86400:.2f} days, dt = {dt:.1f} s, N = {int(background['N'])}")
print(f"Loaded {len(source_params)} injected GB sources")

# %%
# Use SOURCE_PARAMS loaded from data file
SOURCE_PARAMS = source_params

SOURCE_CONFIGS = [
    SourceConfig(
        label="GB 1",
        params=SOURCE_PARAMS[0].copy(),
        prior_A=(0.3 * SOURCE_PARAMS[0, 2], 3.0 * SOURCE_PARAMS[0, 2]),
        prior_f0=(SOURCE_PARAMS[0, 0] - 8.0e-6, SOURCE_PARAMS[0, 0] + 8.0e-6),
        prior_fdot=(SOURCE_PARAMS[0, 1] - 8.0e-18, SOURCE_PARAMS[0, 1] + 8.0e-18),
        prior_phi0=(0.0, 2.0 * np.pi),
    ),
    SourceConfig(
        label="GB 2",
        params=SOURCE_PARAMS[1].copy(),
        prior_A=(0.3 * SOURCE_PARAMS[1, 2], 3.0 * SOURCE_PARAMS[1, 2]),
        prior_f0=(SOURCE_PARAMS[1, 0] - 8.0e-6, SOURCE_PARAMS[1, 0] + 8.0e-6),
        prior_fdot=(SOURCE_PARAMS[1, 1] - 8.0e-18, SOURCE_PARAMS[1, 1] + 8.0e-18),
        prior_phi0=(0.0, 2.0 * np.pi),
    ),
]

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)


def source_a_band(params: np.ndarray, kmin: int, kmax: int) -> np.ndarray:
    A, _, _ = jgb.sum_tdi(
        np.asarray(params, dtype=float).reshape(1, -1),
        kmin=int(kmin),
        kmax=int(kmax),
        tdi_combination="AET",
    )
    return np.asarray(A, dtype=np.complex128).reshape(-1)


def source_a_band_jax(params: jnp.ndarray, kmin: int, kmax: int) -> jnp.ndarray:
    A, _, _ = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=int(kmin),
        kmax=int(kmax),
        tdi_combination="AET",
    )
    return jnp.asarray(A, dtype=jnp.complex128).reshape(-1)


def source_a_full(params: np.ndarray) -> np.ndarray:
    return source_a_band(params, 0, n_freq)


# Data is already injected from data_generation.py
data_full = background_A


def build_band_data(config: SourceConfig, source_index: int) -> BandData:
    k_center = int(np.rint(config.params[0] * t_obs))
    band_half_width = jgb.n
    start = max(k_center - band_half_width, 0)
    stop = min(k_center + band_half_width, n_freq)
    idx = np.arange(start, stop)
    noise_band = np.interp(
        freqs_full[idx],
        freqs_full,
        noise_psd_full,
        left=noise_psd_full[0],
        right=noise_psd_full[-1],
    )
    t_c = t_obs / 2.0
    f_c_true = config.params[0] + config.params[1] * t_c
    phi_c_true = (
        config.params[7]
        + 2.0 * np.pi * config.params[0] * t_c
        + np.pi * config.params[1] * (t_c**2)
    ) % (2.0 * np.pi)

    prior_center = np.array(
        [f_c_true, config.params[1], phi_c_true],
        dtype=float,
    )
    prior_scale = np.array(
        [
            0.25 * (config.prior_f0[1] - config.prior_f0[0]),
            0.25 * (config.prior_fdot[1] - config.prior_fdot[0]),
            np.pi / 2.0,
        ],
        dtype=float,
    )
    return BandData(
        label=config.label,
        idx=idx,
        freqs=freqs_full[idx],
        data=data_full[idx],
        noise_psd=noise_band,
        data_jax=jnp.asarray(data_full[idx], dtype=jnp.complex128),
        noise_psd_jax=jnp.maximum(jnp.asarray(noise_band, dtype=jnp.float64), 1e-60),
        fixed_params=config.params.copy(),
        source_index=source_index,
        band_kmin=start,
        band_kmax=stop,
        prior_center=prior_center,
        prior_scale=prior_scale,
        logA_bounds=tuple(np.log(config.prior_A)),
    )


BANDS = [build_band_data(cfg, i) for i, cfg in enumerate(SOURCE_CONFIGS)]


def wrap_phase_jax(phi: jnp.ndarray) -> jnp.ndarray:
    return (phi + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def model_band(theta: np.ndarray, band: BandData) -> np.ndarray:
    params = band.fixed_params.copy()
    params[0] = theta[0]
    params[1] = theta[1]
    params[2] = theta[2]
    params[7] = wrap_phase(theta[3]) % (2.0 * np.pi)
    return source_a_band(params, band.band_kmin, band.band_kmax)


def model_band_jax(theta: jnp.ndarray, band: BandData) -> jnp.ndarray:
    params = jnp.asarray(band.fixed_params, dtype=jnp.float64)
    params = params.at[0].set(theta[0])
    params = params.at[1].set(theta[1])
    params = params.at[2].set(theta[2])
    params = params.at[7].set(jnp.mod(wrap_phase_jax(theta[3]), 2.0 * jnp.pi))
    return source_a_band_jax(params, band.band_kmin, band.band_kmax)


def whittle_loglike_jax(theta: jnp.ndarray, band: BandData) -> jnp.ndarray:
    model = model_band_jax(theta, band)
    residual = band.data_jax - model
    return -jnp.sum(
        jnp.log(band.noise_psd_jax) + (jnp.abs(residual) ** 2) / band.noise_psd_jax
    )


def numpyro_model(config: SourceConfig, band: BandData) -> None:
    delta_fc = numpyro.sample("delta_fc", dist.Normal(0.0, 1.0))
    delta_fdot = numpyro.sample("delta_fdot", dist.Normal(0.0, 1.0))
    phi_offset = numpyro.sample("phi_offset", dist.Normal(0.0, 1.0))
    logA = numpyro.sample("logA", dist.Uniform(*band.logA_bounds))

    f_c = band.prior_center[0] + band.prior_scale[0] * delta_fc
    fdot = numpyro.deterministic(
        "fdot",
        band.prior_center[1] + band.prior_scale[1] * delta_fdot,
    )
    amplitude = numpyro.deterministic("A", jnp.exp(logA))
    phi_c = jnp.mod(
        band.prior_center[2] + band.prior_scale[2] * phi_offset,
        2.0 * jnp.pi,
    )

    t_c = t_obs / 2.0
    f0 = numpyro.deterministic("f0", f_c - fdot * t_c)
    
    phi0_unwrapped = phi_c - 2.0 * jnp.pi * f_c * t_c + jnp.pi * fdot * t_c**2
    phi0 = numpyro.deterministic(
        "phi0",
        jnp.mod(phi0_unwrapped, 2.0 * jnp.pi),
    )

    numpyro.factor(
        "f0_prior_support",
        jnp.where(
            (f0 >= config.prior_f0[0]) & (f0 <= config.prior_f0[1]),
            0.0,
            -jnp.inf,
        ),
    )
    numpyro.factor(
        "fdot_prior_support",
        jnp.where(
            (fdot >= config.prior_fdot[0]) & (fdot <= config.prior_fdot[1]),
            0.0,
            -jnp.inf,
        ),
    )

    theta = jnp.array([f0, fdot, amplitude, phi0], dtype=jnp.float64)
    numpyro.factor("whittle", whittle_loglike_jax(theta, band))


def sample_single_source(
    config: SourceConfig,
    band: BandData,
    *,
    draws: int = 1200,
    tune: int = 800,
    chains: int = 1,
    seed: int = 0,
) -> MCMC:
    rng = np.random.default_rng(seed)
    init_values = {
        "delta_fc": float(rng.normal(0.0, 0.05)),
        "delta_fdot": float(rng.normal(0.0, 0.05)),
        "phi_offset": float(rng.normal(0.0, 0.05)),
        "logA": float(np.log(config.params[2]) + rng.normal(0.0, 0.05)),
    }
    kernel = NUTS(
        lambda: numpyro_model(config, band),
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=0.9,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def trace_to_array(mcmc: MCMC) -> np.ndarray:
    samples = mcmc.get_samples()
    return np.column_stack(
        [
            np.asarray(samples["f0"]),
            np.asarray(samples["fdot"]),
            np.asarray(samples["A"]),
            np.asarray(samples["phi0"]),
        ]
    )


def summarize_trace(label: str, samples: np.ndarray, truth: np.ndarray) -> None:
    med = np.median(samples, axis=0)
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    print(f"{label}")
    for name, median, low, high, true_value in zip(
        ["f0", "fdot", "A", "phi0"], med, lo, hi, truth, strict=True
    ):
        print(
            f"  {name:4s} median={median:.6e}  "
            f"90% CI=[{low:.6e}, {high:.6e}]  true={true_value:.6e}"
        )


def current_median_model(samples: np.ndarray, band: BandData) -> np.ndarray:
    theta = np.median(samples, axis=0)
    return model_band(theta, band)


DRAW_INDEP = 400
TUNE_INDEP = 200

print("\nIndependent local NumPyro runs")
independent_samples = []
posterior_models = []
for i, (cfg, band) in enumerate(zip(SOURCE_CONFIGS, BANDS, strict=True)):
    mcmc = sample_single_source(
        cfg,
        band,
        draws=DRAW_INDEP,
        tune=TUNE_INDEP,
        seed=10 + i,
    )
    samples = trace_to_array(mcmc)
    independent_samples.append(samples)
    posterior_models.append(current_median_model(samples, band))
    summarize_trace(
        f"{cfg.label}",
        samples,
        truth=np.array([cfg.params[0], cfg.params[1], cfg.params[2], cfg.params[7]]),
    )

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for ax, band, cfg, model in zip(
    axes, BANDS, SOURCE_CONFIGS, posterior_models, strict=True
):
    ax.plot(
        band.freqs,
        np.abs(band.data),
        color="tab:blue",
        lw=1.0,
        label="Injected data",
    )
    ax.plot(
        band.freqs,
        np.abs(model),
        color="tab:orange",
        lw=1.0,
        label="Posterior median model",
    )
    ax.set_title(f"{cfg.label} local frequency band")
    ax.set_ylabel(r"$|A(f)|$")
    ax.axvline(cfg.params[0], color="tab:red", ls="--", lw=1.5, label="True f0")
    ax.set_xlim(cfg.params[0] - 6.0e-6, cfg.params[0] + 6.0e-6)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Frequency [Hz]")
_ = save_figure(fig, FIGURE_OUTPUT_DIR, "local_frequency_bands")

# %% [markdown]
# ![Local frequency bands](../lisa_freq_mcmc_assets/local_frequency_bands.png)

# %%
corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$"]

for cfg, samples, stem in zip(
    SOURCE_CONFIGS,
    independent_samples,
    ["gb1_corner", "gb2_corner"],
    strict=True,
):
    fig = corner.corner(
        samples,
        labels=corner_labels,
        truths=[cfg.params[0], cfg.params[1], cfg.params[2], cfg.params[7]],
        truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
    )
    _ = save_figure(fig, FIGURE_OUTPUT_DIR, stem)

# %% [markdown]
# ![GB 1 corner](../lisa_freq_mcmc_assets/gb1_corner.png)
# ![GB 2 corner](../lisa_freq_mcmc_assets/gb2_corner.png)
