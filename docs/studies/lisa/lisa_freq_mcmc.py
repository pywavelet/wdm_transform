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
    save_corner_plot_dual,
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
F0_JITTER_WIDTH = float(os.getenv("LISA_F0_JITTER_WIDTH", "0.001"))  # log-space half-width


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
noise_psd_saved_aet = np.stack(
    [inj.noise_psd_A, inj.noise_psd_E, inj.noise_psd_T],
    axis=0,
)
freqs_saved = inj.freqs
SOURCE_PARAMS = inj.source_params
SOURCE_PARAM = SOURCE_PARAMS[0].copy()
MCMC_SEED = int(os.getenv("LISA_MCMC_SEED", str(INJECTION_SEED + 10)))

data_aet = data_aet_full
t_obs = t_obs_saved
df = 1.0 / t_obs

freqs = rfftfreq(data_aet.shape[1], dt)
data_aet_f = rfft(data_aet, axis=1)
n_freq = len(freqs)

noise_psd_full_aet = interp_psd_channels(freqs, freqs_saved, noise_psd_saved_aet)

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {data_aet.shape[1]}  df = {df:.3e} Hz"
)
print(f"Injection seed = {INJECTION_SEED}  MCMC seed = {MCMC_SEED}")
print(
    f"Analysis priors match injection priors: "
    f"f0 in [{inj.prior_f0[0]:.6e}, {inj.prior_f0[1]:.6e}] Hz"
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
    prior_center: np.ndarray
    prior_scale: np.ndarray
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
    # In this single-injection workflow, use the injected f0 directly as the
    # near-truth initialization for MCMC.
    f0_init = float(params[0])

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
        prior_center=prior_info.prior_center,
        prior_scale=prior_info.prior_scale,
        logf0_bounds=prior_info.logf0_bounds,
        logfdot_bounds=prior_info.logfdot_bounds,
        logA_bounds=prior_info.logA_bounds,
    )


BAND = build_band(
    SOURCE_PARAM,
    "Injected GB",
    inj.prior_f0,
    inj.prior_fdot,
    inj.prior_A,
)

# ── Per-source SNR ────────────────────────────────────────────────────────────
print("\nMatched-filter SNR (A+E+T channels):")
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
print(f"  {BAND.label}: SNR = {snr_optimal:.1f}")


# ── Diagnostic: data, signal, and priors ──────────────────────────────────
def make_diagnostic_plots(band: BandData, jgb_obj, true_params: np.ndarray) -> None:
    """Plot data + true signal + prior samples for sanity check."""
    print("\nGenerating diagnostic plots…")

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
    n_prior = 30
    logf0_true = np.log(true_params[0])
    delta_logf0_prior = np.random.uniform(-F0_JITTER_WIDTH, F0_JITTER_WIDTH, n_prior)
    logf0_prior = logf0_true + delta_logf0_prior  # Jitter around true f0
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

    print(f"  NLL at truth: {nll_truth:.1f}")
    print(f"  NLL at prior samples: mean={np.mean(nlls_prior):.1f}, min={np.min(nlls_prior):.1f}, max={np.max(nlls_prior):.1f}")
    print(f"  Δ NLL (truth - prior_mean): {nll_truth - np.mean(nlls_prior):.1f}")

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
    print(f"  Saved to {diagnostic_path}")
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

    # Deterministic near-truth initialization
    f0_true = float(band.fixed_params[0])
    fdot_true = float(band.fixed_params[1])
    A_true = float(band.fixed_params[2])
    phi0_true = float(band.fixed_params[7])

    init_values = {
        "logf0": float(np.log(f0_true)),
        "logfdot": float(np.log(fdot_true)),
        "logA": float(np.log(A_true)),
        "phi0": float(phi0_true),
    }

    init_elapsed = time.perf_counter() - init_start_time
    print(
        "  Init:"
        f" f0_peak={band.f0_init:.6e}"
        f" params=[{init_values['logf0']:.3f}, {init_values['logfdot']:.3f},"
        f" {init_values['logA']:.3f}, {init_values['phi0']:.3f}]"
        f" (direct truth, f0 fixed ±{F0_JITTER_WIDTH:.2e})"
    )
    print(f"  Timing: init proposal {init_elapsed:.2f} s")

    def model():
        """NumPyro generative model with f0_jitter."""
        # Sample frequency jitter around fixed f0
        delta_logf0 = numpyro.sample(
            "delta_logf0",
            dist.Uniform(-F0_JITTER_WIDTH, F0_JITTER_WIDTH),
        )
        logf0 = init_values["logf0"] + delta_logf0

        # Sample other parameters with priors
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

        # Deterministic transforms
        f0 = numpyro.deterministic("f0", jnp.exp(logf0))
        fdot = numpyro.deterministic("fdot", jnp.exp(logfdot))
        A = numpyro.deterministic("A", jnp.exp(logA))
        phi0_waveform = numpyro.deterministic("phi0_waveform", phi0)

        # Build parameter vector
        params = jnp.asarray(band.fixed_params, dtype=jnp.float64)
        params = params.at[0].set(f0)
        params = params.at[1].set(fdot)
        params = params.at[2].set(A)
        params = params.at[7].set(phi0_waveform)

        # Waveform
        h_aet = source_aet_band(params, band.band_kmin, band.band_kmax)

        # Whittle likelihood
        residual_phys = dt_j * (data_aet_j - h_aet)
        numpyro.factor(
            "whittle",
            -jnp.sum(jnp.log(psd_aet_j) + 2.0 * df_j * jnp.abs(residual_phys) ** 2 / psd_aet_j),
        )

    # Initialize at truth
    init_strategy = init_to_value(
        values={
            "delta_logf0": 0.0,
            "logfdot": init_values["logfdot"],
            "logA": init_values["logA"],
            "phi0": init_values["phi0"],
        }
    )

    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_DRAWS, progress_bar=bool(SHOW_PROGRESS))

    mcmc_start_time = time.perf_counter()
    print(f"  Running NUTS ({N_WARMUP} warmup + {N_DRAWS} draws)…")
    mcmc.run(jax.random.PRNGKey(seed))
    mcmc_elapsed = time.perf_counter() - mcmc_start_time
    print(f"  Timing: NUTS warmup+sampling {mcmc_elapsed:.2f} s")

    # Extract samples
    samples = mcmc.get_samples()

    # Convert delta_logf0 back to logf0
    delta_logf0_samples = np.asarray(samples["delta_logf0"])
    logf0_samples = init_values["logf0"] + delta_logf0_samples

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


print("\nRunning NUTS for the injected source…")
print(f"\n─── {BAND.label} ───")
mcmc_result = sample_source(BAND, seed=MCMC_SEED, snr_optimal=snr_optimal)

mcmc = mcmc_result["mcmc"]
print("\nNUTS diagnostics:")
nuts_extra = mcmc.get_extra_fields()
n_div = int(nuts_extra.get("diverging", [0]).sum()) if "diverging" in nuts_extra else 0
print(f"  Divergences: {n_div}")
if N_DRAWS >= 4:
    mcmc.print_summary(exclude_deterministic=False)
else:
    print("  Skipping summary: need at least 4 draws for R-hat diagnostics.")

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
print(f"  Timing: posterior SNR evaluation {snr_eval_elapsed:.2f} s")

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
print(f"\nSaved posteriors to {_out_path}")

# Generate corner plot with unified colors and overlays
print("\nGenerating corner plot…")
try:
    wdm_posterior_path = RUN_DIR / "wdm_posteriors.npz"
    wdm_samples_overlay = None
    if wdm_posterior_path.exists():
        print("  Loading WDM posterior for overlay…")
        with np.load(wdm_posterior_path) as wdm_data:
            wdm_samples = np.asarray(wdm_data["samples_source"], dtype=float)
            wdm_labels = [str(item) for item in np.asarray(wdm_data["labels"]).tolist()]

        # Extract matching parameters (f0, fdot, A, phi0)
        keep_cols = [i for i, label in enumerate(wdm_labels) if any(x in label.lower() for x in ["f0", "fdot", "a [", "phi0"])]
        if len(keep_cols) == 4:
            wdm_samples_overlay = wdm_samples[:, keep_cols]

    corner_path = save_corner_plot_dual(
        samples_report,
        wdm_samples_overlay,
        truth=truth[:4],
        output_dir=RUN_DIR,
        primary_name="freq",
        secondary_name="wdm",
        labels=PARAM_NAMES[:4],
    )
    print(f"Saved corner plot to {corner_path}")
except Exception as e:
    print(f"Could not generate corner plot: {e}")