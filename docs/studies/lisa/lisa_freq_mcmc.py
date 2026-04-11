"""Frequency-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py``, truncates the
time series to a power-of-2 length, and performs frequency-domain Bayesian
inference for two Galactic binaries with a local Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Print per-source Whittle SNR in the A channel.
3. Run independent NumPyro NUTS chains on a narrow frequency band per source.
4. Save posterior summaries and diagnostic figures.

Sky position, polarisation, and inclination are held fixed at injected values.
The two injected GBs are well-separated in frequency, so each is fit independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    FREQ_ASSET_DIR,
    INJECTION_PATH,
    check_posterior_coverage,
    matched_filter_snr_rfft,
    print_posterior_summary,
    require_positive_fdot,
    save_figure,
    wrap_phase,
)
from numpy.fft import rfft, rfftfreq
from numpyro.infer import MCMC, NUTS, init_to_value

jax.config.update("jax_enable_x64", True)

FIGURE_OUTPUT_DIR = FREQ_ASSET_DIR
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))

# ── Load and truncate data ────────────────────────────────────────────────────
if not INJECTION_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached injection at {INJECTION_PATH}. "
        "Run data_generation.py first."
    )

_inj = np.load(INJECTION_PATH)
dt = float(_inj["dt"])
t_obs_saved = float(_inj["t_obs"])
data_At_full = np.asarray(_inj["data_At"], dtype=float)
noise_psd_saved = np.asarray(_inj["noise_psd_A"], dtype=float)
freqs_saved = np.asarray(_inj["freqs"], dtype=float)
SOURCE_PARAMS = require_positive_fdot(
    np.asarray(_inj["source_params"], dtype=float),
    context=str(INJECTION_PATH),
)
_inj.close()

data_At = data_At_full
t_obs = t_obs_saved
df = 1.0 / t_obs

freqs = rfftfreq(len(data_At), dt)
data_f = rfft(data_At)
n_freq = len(freqs)

noise_psd_full = np.maximum(
    np.interp(
        freqs,
        freqs_saved,
        noise_psd_saved,
        left=noise_psd_saved[0],
        right=noise_psd_saved[-1],
    ),
    1e-60,
)

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {len(data_At)}  df = {df:.3e} Hz"
)

# ── JaxGB generator ───────────────────────────────────────────────────────────
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


# ── Band data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BandData:
    label: str
    freqs: np.ndarray
    data: np.ndarray          # complex frequency-domain data in band
    noise_psd: np.ndarray     # one-sided noise PSD in band
    fixed_params: np.ndarray  # full 8-param vector (extrinsic held fixed)
    t_obs: float
    band_kmin: int
    band_kmax: int
    prior_center: np.ndarray  # [logf0, logfdot, logA]
    prior_scale: np.ndarray   # scales for [logf0, logfdot, logA, delta_phi_c]
    phase_ref: np.ndarray     # [phi0_ref, phi_c_ref]
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


def build_band(
    params: np.ndarray,
    label: str,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
) -> BandData:
    k_center = int(np.rint(params[0] * t_obs))
    kmin = max(k_center - jgb.n, 0)
    kmax = min(k_center + jgb.n, n_freq)

    band_freqs = freqs[kmin:kmax]
    noise_band = np.maximum(np.interp(band_freqs, freqs, noise_psd_full), 1e-60)

    phi0_ref = float(wrap_phase(params[7]))
    t_c = t_obs / 2.0
    phi_c_ref = float(
        wrap_phase(phi0_ref + 2 * np.pi * params[0] * t_c + np.pi * params[1] * t_c**2)
    )
    logf0_bounds = (float(np.log(prior_f0[0])), float(np.log(prior_f0[1])))
    logfdot_bounds = (float(np.log(prior_fdot[0])), float(np.log(prior_fdot[1])))
    logA_bounds = (float(np.log(prior_A[0])), float(np.log(prior_A[1])))

    return BandData(
        label=label,
        freqs=band_freqs,
        data=data_f[kmin:kmax],
        noise_psd=noise_band,
        fixed_params=params.copy(),
        t_obs=t_obs,
        band_kmin=kmin,
        band_kmax=kmax,
        prior_center=np.array([np.log(params[0]), np.log(params[1]), np.log(params[2])]),
        prior_scale=np.array([
            0.25 * (logf0_bounds[1] - logf0_bounds[0]),
            0.25 * (logfdot_bounds[1] - logfdot_bounds[0]),
            0.25 * (logA_bounds[1] - logA_bounds[0]),
            np.pi / 4.0,
        ]),
        phase_ref=np.array([phi0_ref, phi_c_ref]),
        logf0_bounds=logf0_bounds,
        logfdot_bounds=logfdot_bounds,
        logA_bounds=logA_bounds,
    )


BANDS = [
    build_band(
        SOURCE_PARAMS[0],
        label="GB 1",
        prior_f0=(SOURCE_PARAMS[0, 0] - 8e-6, SOURCE_PARAMS[0, 0] + 8e-6),
        prior_fdot=(0.25 * SOURCE_PARAMS[0, 1], 4.0 * SOURCE_PARAMS[0, 1]),
        prior_A=(0.3 * SOURCE_PARAMS[0, 2], 3.0 * SOURCE_PARAMS[0, 2]),
    ),
    build_band(
        SOURCE_PARAMS[1],
        label="GB 2",
        prior_f0=(SOURCE_PARAMS[1, 0] - 8e-6, SOURCE_PARAMS[1, 0] + 8e-6),
        prior_fdot=(0.25 * SOURCE_PARAMS[1, 1], 4.0 * SOURCE_PARAMS[1, 1]),
        prior_A=(0.3 * SOURCE_PARAMS[1, 2], 3.0 * SOURCE_PARAMS[1, 2]),
    ),
]

# ── Per-source SNR ────────────────────────────────────────────────────────────
snrs_optimal = []
print("\nPer-source matched-filter SNR (A channel):")
for i, src in enumerate(SOURCE_PARAMS):
    k_center = int(np.rint(src[0] * t_obs))
    kmin = max(k_center - jgb.n, 0)
    kmax = min(k_center + jgb.n, n_freq)
    band_freqs = freqs[kmin:kmax]
    noise_band = np.maximum(np.interp(band_freqs, freqs, noise_psd_full), 1e-60)
    
    h = source_a_band_jax(jnp.asarray(src, dtype=jnp.float64), int(kmin), int(kmax))
    snr = matched_filter_snr_rfft(
        np.asarray(h), noise_band, band_freqs, dt=dt
    )
    snrs_optimal.append(float(snr))
    print(f"  GB {i + 1}: SNR = {snr:.1f}")

# ── NUTS per source ───────────────────────────────────────────────────────────


def sample_source(band: BandData, *, seed: int = 0) -> MCMC:
    """Run NUTS for one source.

    ``data_j`` and ``psd_j`` are captured once as JAX constants to avoid
    redundant conversion inside the likelihood at every NUTS step.
    """
    data_j = jnp.asarray(band.data, dtype=jnp.complex128)
    psd_j = jnp.asarray(band.noise_psd, dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)

    def model():
        # Scale parameters to have O(1) posterior variance. This prevents NUTS mass-matrix 
        # adaptation from hitting regularization floors which causes max_tree_depth stalls.
        snr_guess = 300.0
        
        # logf0: analytical Fisher std on f0 is ~2e-10 Hz. For f0=1e-3, d(logf0) ~ 2e-7
        scale_logf0 = 2e-7
        # logA: fractional amplitude error is ~ 1/SNR
        scale_logA = 1.0 / snr_guess
        # phase: error is ~ 1/SNR
        scale_phi_c = 1.0 / snr_guess
        # logfdot: prior dominated, bounds span O(1).
        scale_logfdot = band.prior_scale[1]

        # Use unconstrained variables with unit Normal priors
        del_logf0 = numpyro.sample("del_logf0", dist.Normal(0.0, 1.0))
        del_logfdot = numpyro.sample("del_logfdot", dist.Normal(0.0, 1.0))
        del_logA = numpyro.sample("del_logA", dist.Normal(0.0, 1.0))
        del_phi_c = numpyro.sample("del_phi_c", dist.Normal(0.0, 1.0))

        logf0 = band.prior_center[0] + scale_logf0 * del_logf0
        logfdot = band.prior_center[1] + scale_logfdot * del_logfdot
        logA = band.prior_center[2] + scale_logA * del_logA
        delta_phi_c = scale_phi_c * del_phi_c

        # Add physical prior log-probabilities 
        numpyro.factor("prior_logf0", dist.TruncatedNormal(
            loc=band.prior_center[0], scale=band.prior_scale[0],
            low=band.logf0_bounds[0], high=band.logf0_bounds[1]).log_prob(logf0))
        numpyro.factor("prior_logfdot", dist.TruncatedNormal(
            loc=band.prior_center[1], scale=band.prior_scale[1],
            low=band.logfdot_bounds[0], high=band.logfdot_bounds[1]).log_prob(logfdot))
        numpyro.factor("prior_logA", dist.TruncatedNormal(
            loc=band.prior_center[2], scale=band.prior_scale[2],
            low=band.logA_bounds[0], high=band.logA_bounds[1]).log_prob(logA))

        f0 = numpyro.deterministic("f0", jnp.exp(logf0))
        fdot = numpyro.deterministic("fdot", jnp.exp(logfdot))
        A = numpyro.deterministic("A", jnp.exp(logA))

        t_c = band.t_obs / 2.0
        numpyro.deterministic(
            "phi_c",
            band.phase_ref[1] + band.prior_scale[3] * delta_phi_c,
        )
        phi0 = numpyro.deterministic(
            "phi0",
            band.phase_ref[0]
            + band.prior_scale[3] * delta_phi_c
            - 2 * jnp.pi * (f0 - jnp.exp(band.prior_center[0])) * t_c
            - jnp.pi * (fdot - jnp.exp(band.prior_center[1])) * t_c**2,
        )

        params = (
            jnp.asarray(band.fixed_params, dtype=jnp.float64)
            .at[0].set(f0)
            .at[1].set(fdot)
            .at[2].set(A)
            .at[7].set(phi0)
        )
        h = source_a_band_jax(params, band.band_kmin, band.band_kmax)
        residual = data_j - h
        residual_phys = dt_j * residual
        numpyro.factor(
            "whittle",
            -jnp.sum(jnp.log(psd_j) + 2.0 * df_j * jnp.abs(residual_phys) ** 2 / psd_j),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values={
            "del_logf0": 0.0,
            "del_logfdot": 0.0,
            "del_logA": 0.0,
            "del_phi_c": 0.0,
        }),
        dense_mass=True,
        target_accept_prob=0.9,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=1,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    return mcmc


print("\nRunning independent NUTS per source…")
all_samples: list[np.ndarray] = []

for i, band in enumerate(BANDS):
    print(f"\n─── {band.label} ───")
    mcmc = sample_source(band, seed=10 + i)

    n_div = int(mcmc.get_extra_fields()["diverging"].sum())
    print(f"  Divergences: {n_div}")
    mcmc.print_summary(exclude_deterministic=False)

    s = mcmc.get_samples()
    samples_i = np.column_stack([
        np.asarray(s["f0"]),
        np.asarray(s["fdot"]),
        np.asarray(s["A"]),
        np.asarray(s["phi0"]),
    ])

    # SNR computation for samples
    samples_i_full = np.tile(SOURCE_PARAMS[i], (samples_i.shape[0], 1))
    samples_i_full[:, 0] = samples_i[:, 0]
    samples_i_full[:, 1] = samples_i[:, 1]
    samples_i_full[:, 2] = samples_i[:, 2]
    samples_i_full[:, 7] = samples_i[:, 3]

    psd_j = jnp.asarray(band.noise_psd, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)
    kmin_stat = band.band_kmin
    kmax_stat = band.band_kmax

    @jax.jit
    def _get_snrs(ps):
        def _single_snr(p):
            h = source_a_band_jax(p.reshape(1, -1), kmin_stat, kmax_stat)
            h_tilde = dt * h
            snr2 = 4.0 * df_j * jnp.sum(jnp.abs(h_tilde)**2 / psd_j)
            return jnp.sqrt(snr2)
        return jax.vmap(_single_snr)(ps)

    snr_samples = np.asarray(_get_snrs(jnp.array(samples_i_full)))
    samples_i = np.column_stack([samples_i, snr_samples])
    
    all_samples.append(samples_i)

# ── Posterior summaries and coverage ─────────────────────────────────────────
PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"]

for i, (band, samples_i) in enumerate(zip(BANDS, all_samples, strict=True)):
    truth_i = np.array([
        band.fixed_params[0],
        band.fixed_params[1],
        band.fixed_params[2],
        band.fixed_params[7],
        snrs_optimal[i],
    ])
    print(f"\n{'═' * 56}  GB {i + 1}")
    print_posterior_summary(samples_i, truth_i, PARAM_NAMES)
    check_posterior_coverage(samples_i, truth_i, PARAM_NAMES)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for ax, band, samples in zip(axes, BANDS, all_samples, strict=True):
    theta_med = np.median(samples, axis=0)
    params_med = band.fixed_params.copy()
    params_med[[0, 1, 2, 7]] = [theta_med[0], theta_med[1], theta_med[2],
                                  wrap_phase(theta_med[3]) % (2 * np.pi)]
    model_med = source_a_band(params_med, band.band_kmin, band.band_kmax)

    ax.plot(band.freqs, np.abs(band.data), lw=1.0, label="Data")
    ax.plot(band.freqs, np.abs(model_med), lw=1.0, label="Posterior median model")
    ax.axvline(band.fixed_params[0], color="tab:red", ls="--", lw=1.5, label="True f0")
    ax.set_title(f"{band.label} local frequency band")
    ax.set_ylabel(r"$|A(f)|$")
    ax.set_xlim(band.fixed_params[0] - 6e-6, band.fixed_params[0] + 6e-6)
    ax.legend(fontsize=8)
axes[-1].set_xlabel("Frequency [Hz]")
save_figure(fig, FIGURE_OUTPUT_DIR, "local_frequency_bands")

corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$", "SNR"]
for i, (samples_i, stem) in enumerate(
    zip(all_samples, ["gb1_corner", "gb2_corner"], strict=True)
):
    truth_i = [SOURCE_PARAMS[i, 0], SOURCE_PARAMS[i, 1],
               SOURCE_PARAMS[i, 2], wrap_phase(SOURCE_PARAMS[i, 7]), snrs_optimal[i]]
    fig = corner.corner(
        samples_i, labels=corner_labels, truths=truth_i, truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95], show_titles=True, title_kwargs={"fontsize": 10}
    )
    save_figure(fig, FIGURE_OUTPUT_DIR, stem)

# ── Save posteriors ───────────────────────────────────────────────────────────
_out_path = FIGURE_OUTPUT_DIR / "posteriors.npz"
np.savez(
    _out_path,
    source_params=SOURCE_PARAMS,
    samples_gb1=all_samples[0],
    samples_gb2=all_samples[1],
    snr_optimal=snrs_optimal,
)
print(f"\nSaved posteriors to {_out_path}")
