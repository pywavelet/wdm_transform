"""Frequency-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py`` and performs
frequency-domain Bayesian inference for one Galactic binary with a local
three-channel Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Print per-source Whittle SNR in the A+E+T channels.
3. Run NumPyro NUTS on a narrow frequency band for the injected source.
4. Save posterior summaries and posterior samples.

Sky position, polarisation, and inclination are held fixed at injected values.
"""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass

_SCRIPT_START = time.perf_counter()


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[lisa_freq_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)

import jax
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import numpy as np
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    FREQ_POSTERIOR_PATH,
    INJECTION_PATH,
    RUN_DIR,
    build_local_prior_info,
    build_sampled_source_params,
    interp_psd_channels,
    load_injection,
    save_posterior_archive,
    setup_jax_and_matplotlib,
    source_truth_vector,
)
from numpyro.infer import MCMC, NUTS, init_to_value
from numpy.fft import rfft, rfftfreq
from wdm_transform.signal_processing import matched_filter_snr_rfft

setup_jax_and_matplotlib()
jax.config.update("jax_enable_x64", True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "1")


# ── Load and truncate data ────────────────────────────────────────────────────
if not INJECTION_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached injection at {INJECTION_PATH}. "
        "Run data_generation.py first."
    )

inj = load_injection(INJECTION_PATH)
dt = inj.dt
t_obs_saved = inj.t_obs
INJECTION_SEED = inj.seed
data_aet_full = np.stack([inj.data_At, inj.data_Et, inj.data_Tt], axis=0)
noise_psd_saved_aet = np.stack([inj.noise_psd_A, inj.noise_psd_E, inj.noise_psd_T], axis=0,)
freqs_saved = inj.freqs
SOURCE_PARAMS = inj.source_params
SOURCE_PARAM = SOURCE_PARAMS[0].copy()
F0_REF = inj.f0_ref
F0_JITTER_WIDTH = inj.f0_jitter_width
MCMC_SEED = int(os.getenv("LISA_MCMC_SEED", str(INJECTION_SEED + 10)))

data_aet = data_aet_full
t_obs = t_obs_saved
df = 1.0 / t_obs

freqs = rfftfreq(data_aet.shape[1], dt)
data_aet_f = rfft(data_aet, axis=1)
n_freq = len(freqs)

noise_psd_full_aet = interp_psd_channels(freqs, freqs_saved, noise_psd_saved_aet)

print(
    f"T_obs={t_obs/86400:.1f}d, N={data_aet.shape[1]}, "
    f"seed={INJECTION_SEED}, MCMC_seed={MCMC_SEED}"
)

# ── JaxGB generator ───────────────────────────────────────────────────────────
orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)


def source_aet_band(
    params: jnp.ndarray,
    kmin: int,
    kmax: int,
) -> jnp.ndarray:
    A, E, T = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=int(kmin),
        kmax=int(kmax),
        tdi_combination="AET",
        # should we specify TDI 1.5 for get_tdi_kwargs?
    )
    return jnp.stack(
        [
            jnp.asarray(A, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(E, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(T, dtype=jnp.complex128).reshape(-1),
        ],
        axis=0,
    )


# ── Band data ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BandData:
    label: str
    freqs: np.ndarray
    data_aet: np.ndarray
    noise_psd_aet: np.ndarray
    fixed_params: np.ndarray
    t_obs: float
    band_kmin: int
    band_kmax: int
    f0_init: float
    logf0_ref: float
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


def build_band(
    params: np.ndarray,
    label: str,
    f0_ref: float,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
) -> BandData:
    f0_init = float(f0_ref)

    kmin = max(int(np.floor(prior_f0[0] * t_obs)) - jgb.n, 0)
    kmax = min(int(np.ceil(prior_f0[1] * t_obs)) + jgb.n + 1, n_freq)

    band_freqs = freqs[kmin:kmax]
    noise_band_aet = noise_psd_full_aet[:, kmin:kmax].copy()

    prior_info = build_local_prior_info(
        prior_f0=prior_f0,
        prior_fdot=prior_fdot,
        prior_A=prior_A,
    )

    return BandData(
        label=label,
        freqs=band_freqs,
        data_aet=data_aet_f[:, kmin:kmax],
        noise_psd_aet=noise_band_aet,
        fixed_params=params.copy(),
        t_obs=t_obs,
        band_kmin=kmin,
        band_kmax=kmax,
        f0_init=f0_init,
        logf0_ref=float(np.log(f0_ref)),
        prior_center=prior_info.prior_center,
        prior_scale=prior_info.prior_scale,
        logf0_bounds=prior_info.logf0_bounds,
        logfdot_bounds=prior_info.logfdot_bounds,
        logA_bounds=prior_info.logA_bounds,
    )


BAND = build_band(
    SOURCE_PARAM,
    "Injected GB",
    F0_REF,
    inj.prior_f0,
    inj.prior_fdot,
    inj.prior_A,
)

h_aet = source_aet_band(
    jnp.asarray(SOURCE_PARAM, dtype=jnp.float64),
    int(BAND.band_kmin),
    int(BAND.band_kmax),
)
snr_channels = np.array(
    [
        matched_filter_snr_rfft(
            np.asarray(h_aet[i]),
            BAND.noise_psd_aet[i],
            BAND.freqs,
            dt=dt,
        )
        for i in range(3)
    ],
    dtype=float,
)
snr_optimal = float(np.linalg.norm(snr_channels))
print(f"Frequency band SNR={snr_optimal:.1f}")


# ── Diagnostic: data, signal, and priors ──────────────────────────────────
def make_diagnostic_plots(band: BandData, jgb_obj, true_params: np.ndarray) -> None:
    """Plot data + true signal + prior samples for sanity check."""

    def source_aet_band_diag(params, kmin_in, kmax_in):
        A, E, T = jgb_obj.sum_tdi(
            jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
            kmin=int(kmin_in),
            kmax=int(kmax_in),
            tdi_combination="AET",
        )
        return jnp.stack(
            [
                jnp.asarray(A, dtype=jnp.complex128).reshape(-1),
                jnp.asarray(E, dtype=jnp.complex128).reshape(-1),
                jnp.asarray(T, dtype=jnp.complex128).reshape(-1),
            ],
            axis=0,
        )

    # True signal
    h_true_aet = np.array(source_aet_band_diag(true_params, band.band_kmin, band.band_kmax))

    # Prior samples with f0_jitter model
    np.random.seed(42)
    n_prior = 100
    logf0_true = np.log(band.f0_init)
    delta_logf0_prior = np.random.uniform(-F0_JITTER_WIDTH, F0_JITTER_WIDTH, n_prior)
    logf0_prior = logf0_true + delta_logf0_prior  # Jitter around fixed f_ref
    logfdot_prior = np.random.normal(band.prior_center[1], band.prior_scale[1], n_prior)
    logA_prior = np.random.normal(band.prior_center[2], band.prior_scale[2], n_prior)
    phi0_prior = np.random.uniform(-np.pi, np.pi, n_prior)

    # Whittle NLL
    def nll_channel(logf0, logfdot, logA, phi0, ch_idx):
        params_eval = band.fixed_params.copy()
        params_eval[0] = np.exp(logf0)
        params_eval[1] = np.exp(logfdot)
        params_eval[2] = np.exp(logA)
        params_eval[7] = phi0

        h_aet_eval = np.array(source_aet_band_diag(params_eval, band.band_kmin, band.band_kmax))
        residual = dt * (band.data_aet[ch_idx] - h_aet_eval[ch_idx])
        nll = np.sum(np.log(band.noise_psd_aet[ch_idx]) + 2.0 * df * np.abs(residual)**2 / band.noise_psd_aet[ch_idx])
        return nll

    nll_truth = nll_channel(np.log(true_params[0]), np.log(true_params[1]), np.log(true_params[2]), true_params[7], 0)
    nlls_prior = [nll_channel(logf0_prior[i], logfdot_prior[i], logA_prior[i], phi0_prior[i], 0) for i in range(n_prior)]
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ch_names = ["A", "E", "T"]

    for ch in range(3):
        # Data + true signal
        ax = axes[ch, 0]
        power_data = np.abs(band.data_aet[ch])**2
        power_signal = np.abs(h_true_aet[ch])**2
        ax.semilogy(band.freqs, power_data, "k.", markersize=2, alpha=0.5, label="Data")
        ax.semilogy(band.freqs, power_signal, "r-", linewidth=1.5, label="True signal")
        ax.set_ylabel(f"{ch_names[ch]} power")
        ax.set_xlim(band.freqs[0], band.freqs[-1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Prior samples
        ax = axes[ch, 1]
        for i in range(n_prior):
            params_sample = band.fixed_params.copy()
            params_sample[0] = np.exp(logf0_prior[i])
            params_sample[1] = np.exp(logfdot_prior[i])
            params_sample[2] = np.exp(logA_prior[i])
            params_sample[7] = phi0_prior[i]
            h_sample = np.array(source_aet_band_diag(params_sample, band.band_kmin, band.band_kmax))
            ax.semilogy(band.freqs, np.abs(h_sample[ch])**2, "b-", alpha=0.1)

        ax.semilogy(band.freqs, power_signal, "r-", linewidth=2, label="True signal", zorder=10)
        ax.set_ylabel(f"{ch_names[ch]} power")
        ax.set_xlim(band.freqs[0], band.freqs[-1])
        ax.set_title(f"{ch_names[ch]}: {n_prior} prior samples")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"LISA GB Diagnostic: SNR={snr_optimal:.1f}, T_obs={band.t_obs/86400:.1f}d, "
        f"f0_jitter=±{F0_JITTER_WIDTH:.2e}",
        fontsize=12,
    )
    plt.tight_layout()
    diagnostic_path = RUN_DIR / "diagnostic_data_signal_priors.png"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(diagnostic_path, dpi=100, bbox_inches="tight")
    plt.close()


make_diagnostic_plots(BAND, jgb, SOURCE_PARAM)


def sample_source(band: BandData, *, seed: int = 0, snr_optimal: float = None):
    """Run NUTS for one source with f0_jitter model.

    Args:
        band: BandData with signal, priors, bounds.
        seed: Random seed for reproducibility.
        snr_optimal: Matched-filter SNR (for diagnostics).
    """
    init_start_time = time.perf_counter()

    data_aet_j = jnp.asarray(band.data_aet, dtype=jnp.complex128)
    psd_aet_j = jnp.asarray(band.noise_psd_aet, dtype=jnp.float64)

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)
    init_clip_eps = 1e-6
    fixed_params_j = jnp.asarray(band.fixed_params, dtype=jnp.float64)
    delta_logf0_true = float(np.log(band.fixed_params[0]) - band.logf0_ref)
    phi0_true = float(np.clip(band.fixed_params[7], -np.pi + init_clip_eps, np.pi - init_clip_eps))
    init_values = {
        "delta_logf0": float(np.clip(delta_logf0_true, -F0_JITTER_WIDTH + init_clip_eps, F0_JITTER_WIDTH - init_clip_eps)),
        "logfdot": float(np.clip(np.log(band.fixed_params[1]), band.logfdot_bounds[0] + init_clip_eps, band.logfdot_bounds[1] - init_clip_eps)),
        "logA": float(np.clip(np.log(band.fixed_params[2]), band.logA_bounds[0] + init_clip_eps, band.logA_bounds[1] - init_clip_eps)),
        "phi0": phi0_true,
    }

    def model():
        """NumPyro generative model with f0_jitter."""
        delta_logf0 = numpyro.sample(
            "delta_logf0",
            dist.Uniform(-F0_JITTER_WIDTH, F0_JITTER_WIDTH),
        )
        logf0 = band.logf0_ref + delta_logf0
        logfdot = numpyro.sample(
            "logfdot",
            dist.TruncatedNormal(
                loc=band.prior_center[1],
                scale=band.prior_scale[1],
                low=band.logfdot_bounds[0],
                high=band.logfdot_bounds[1],
            ),
        )
        logA = numpyro.sample(
            "logA",
            dist.TruncatedNormal(
                loc=band.prior_center[2],
                scale=band.prior_scale[2],
                low=band.logA_bounds[0],
                high=band.logA_bounds[1],
            ),
        )
        phi0 = numpyro.sample("phi0", dist.Uniform(-jnp.pi, jnp.pi))

        f0 = numpyro.deterministic("f0", jnp.exp(logf0))
        fdot = numpyro.deterministic("fdot", jnp.exp(logfdot))
        A = numpyro.deterministic("A", jnp.exp(logA))

        params = fixed_params_j.at[0].set(f0).at[1].set(fdot).at[2].set(A).at[7].set(phi0)

        h_aet = source_aet_band(params, band.band_kmin, band.band_kmax)

        residual_phys = dt_j * (data_aet_j - h_aet)
        numpyro.factor(
            "whittle",
            -jnp.sum(jnp.log(psd_aet_j) + 2.0 * df_j * jnp.abs(residual_phys) ** 2 / psd_aet_j),
        )

    init_strategy = init_to_value(
        values=init_values
    )

    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_DRAWS, progress_bar=bool(SHOW_PROGRESS))

    mcmc_start_time = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed))
    mcmc_elapsed = time.perf_counter() - mcmc_start_time

    # Extract samples
    samples = mcmc.get_samples()

    # Convert delta_logf0 back to logf0 around the fixed study reference.
    delta_logf0_samples = np.asarray(samples["delta_logf0"])
    logf0_samples = band.logf0_ref + delta_logf0_samples

    samples_dict = {
        "logf0": logf0_samples,
        "delta_logf0": delta_logf0_samples,
        "logfdot": np.asarray(samples["logfdot"]),
        "logA": np.asarray(samples["logA"]),
        "phi0": np.asarray(samples["phi0"]),
        "f0": np.asarray(samples["f0"]),
        "fdot": np.asarray(samples["fdot"]),
        "A": np.asarray(samples["A"]),
    }

    return {
        "samples": samples_dict,
        "mcmc": mcmc,
    }


mcmc_result = sample_source(BAND, seed=MCMC_SEED, snr_optimal=snr_optimal)

mcmc = mcmc_result["mcmc"]
nuts_extra = mcmc.get_extra_fields()
n_div = int(nuts_extra.get("diverging", [0]).sum()) if "diverging" in nuts_extra else 0
if n_div > 0:
    print(f"WARNING: {n_div} divergences")

s = mcmc_result["samples"]

# Since phi0 is sampled directly, phi0_waveform = phi0
samples_report = np.column_stack(
    [
        np.asarray(s["f0"]),
        np.asarray(s["fdot"]),
        np.asarray(s["A"]),
        np.asarray(s["phi0"]),
    ]
)

samples_waveform = np.column_stack(
    [
        np.asarray(s["f0"]),
        np.asarray(s["fdot"]),
        np.asarray(s["A"]),
        np.asarray(s["phi0"]),  # Same as phi0 now
    ]
)

samples_full = build_sampled_source_params(SOURCE_PARAM, samples_waveform)

psd_aet_j = jnp.asarray(BAND.noise_psd_aet, dtype=jnp.float64)
df_j = jnp.asarray(df, dtype=jnp.float64)
dt_j = jnp.asarray(dt, dtype=jnp.float64)
kmin_stat = BAND.band_kmin
kmax_stat = BAND.band_kmax


@jax.jit
def _get_snrs(ps: jnp.ndarray) -> jnp.ndarray:
    def _single_snr(p: jnp.ndarray) -> jnp.ndarray:
        h_aet_loc = source_aet_band(p, kmin_stat, kmax_stat)
        h_aet_tilde = dt_j * h_aet_loc
        snr2 = 4.0 * df_j * jnp.sum(jnp.abs(h_aet_tilde) ** 2 / psd_aet_j)
        return jnp.sqrt(snr2)

    return jax.vmap(_single_snr)(ps)


snr_eval_start = time.perf_counter()
snr_samples = np.asarray(_get_snrs(jnp.asarray(samples_full, dtype=jnp.float64)))
snr_eval_elapsed = time.perf_counter() - snr_eval_start

samples_to_save = np.column_stack([samples_report, snr_samples])

PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"]
truth = source_truth_vector(BAND.fixed_params, snr=snr_optimal)

_out_path = save_posterior_archive(
    FREQ_POSTERIOR_PATH,
    source_params=SOURCE_PARAMS,
    all_samples=[samples_to_save],
    snr_optimal=[snr_optimal],
    labels=PARAM_NAMES,
    truth=truth,
)
print(f"Saved frequency posteriors to {_out_path}")
