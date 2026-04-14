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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

_SCRIPT_START = time.perf_counter()


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[lisa_freq_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)

import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    FREQ_POSTERIOR_PATH,
    INJECTION_PATH,
    build_local_prior_info,
    build_sampled_source_params,
    estimate_frequency_peak,
    load_injection,
    save_posterior_archive,
    source_truth_vector,
)
from numpy.fft import rfft, rfftfreq
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform.signal_processing import matched_filter_snr_rfft

jax.config.update("jax_enable_x64", True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "0").strip().lower() in {"1", "true", "yes", "on"}
INIT_JITTER_SCALE = float(os.getenv("LISA_INIT_JITTER_SCALE", "0.15"))

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
data_At_full = inj.data_At
data_Et_full = inj.data_Et
data_Tt_full = inj.data_Tt
noise_psd_A_saved = inj.noise_psd_A
noise_psd_E_saved = inj.noise_psd_E
noise_psd_T_saved = inj.noise_psd_T
freqs_saved = inj.freqs
SOURCE_PARAMS = inj.source_params
SOURCE_PARAM = SOURCE_PARAMS[0].copy()
MCMC_SEED = int(os.getenv("LISA_MCMC_SEED", str(INJECTION_SEED + 10)))

data_At = data_At_full
data_Et = data_Et_full
data_Tt = data_Tt_full
t_obs = t_obs_saved
df = 1.0 / t_obs

freqs = rfftfreq(len(data_At), dt)
data_Af = rfft(data_At)
data_Ef = rfft(data_Et)
data_Tf = rfft(data_Tt)
n_freq = len(freqs)

noise_psd_A_full = np.maximum(
    np.interp(
        freqs,
        freqs_saved,
        noise_psd_A_saved,
        left=noise_psd_A_saved[0],
        right=noise_psd_A_saved[-1],
    ),
    1e-60,
)
noise_psd_E_full = np.maximum(
    np.interp(
        freqs,
        freqs_saved,
        noise_psd_E_saved,
        left=noise_psd_E_saved[0],
        right=noise_psd_E_saved[-1],
    ),
    1e-60,
)
noise_psd_T_full = np.maximum(
    np.interp(
        freqs,
        freqs_saved,
        noise_psd_T_saved,
        left=noise_psd_T_saved[0],
        right=noise_psd_T_saved[-1],
    ),
    1e-60,
)

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {len(data_At)}  df = {df:.3e} Hz"
)
print(f"Injection seed = {INJECTION_SEED}  MCMC seed = {MCMC_SEED}")
print(f"Analysis priors match injection priors: f0 in [{inj.prior_f0[0]:.6e}, {inj.prior_f0[1]:.6e}] Hz")

# ── JaxGB generator ───────────────────────────────────────────────────────────
orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)


def source_aet_band(params: np.ndarray, kmin: int, kmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, E, T = jgb.sum_tdi(
        np.asarray(params, dtype=float).reshape(1, -1),
        kmin=int(kmin),
        kmax=int(kmax),
        tdi_combination="AET",
    )
    return (
        np.asarray(A, dtype=np.complex128).reshape(-1),
        np.asarray(E, dtype=np.complex128).reshape(-1),
        np.asarray(T, dtype=np.complex128).reshape(-1),
    )


def source_aet_band_jax(
    params: jnp.ndarray, kmin: int, kmax: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A, E, T = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=int(kmin),
        kmax=int(kmax),
        tdi_combination="AET",
    )
    return (
        jnp.asarray(A, dtype=jnp.complex128).reshape(-1),
        jnp.asarray(E, dtype=jnp.complex128).reshape(-1),
        jnp.asarray(T, dtype=jnp.complex128).reshape(-1),
    )


# ── Band data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BandData:
    label: str
    freqs: np.ndarray
    data_A: np.ndarray
    data_E: np.ndarray
    data_T: np.ndarray
    noise_psd_A: np.ndarray
    noise_psd_E: np.ndarray
    noise_psd_T: np.ndarray
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
    f0_init = estimate_frequency_peak(
        freqs,
        data_Af,
        data_Ef,
        data_Tf,
        noise_psd_A_full,
        noise_psd_E_full,
        noise_psd_T_full,
        prior_f0=prior_f0,
    )
    kmin = max(int(np.floor(prior_f0[0] * t_obs)) - jgb.n, 0)
    kmax = min(int(np.ceil(prior_f0[1] * t_obs)) + jgb.n + 1, n_freq)

    band_freqs = freqs[kmin:kmax]
    noise_band_A = np.maximum(np.interp(band_freqs, freqs, noise_psd_A_full), 1e-60)
    noise_band_E = np.maximum(np.interp(band_freqs, freqs, noise_psd_E_full), 1e-60)
    noise_band_T = np.maximum(np.interp(band_freqs, freqs, noise_psd_T_full), 1e-60)

    prior_info = build_local_prior_info(
        prior_f0=prior_f0,
        prior_fdot=prior_fdot,
        prior_A=prior_A,
    )

    return BandData(
        label=label,
        freqs=band_freqs,
        data_A=data_Af[kmin:kmax],
        data_E=data_Ef[kmin:kmax],
        data_T=data_Tf[kmin:kmax],
        noise_psd_A=noise_band_A,
        noise_psd_E=noise_band_E,
        noise_psd_T=noise_band_T,
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


BAND = build_band(SOURCE_PARAM, "Injected GB", inj.prior_f0, inj.prior_fdot, inj.prior_A)

# ── Per-source SNR ────────────────────────────────────────────────────────────
print("\nMatched-filter SNR (A+E+T channels):")
h_A, h_E, h_T = source_aet_band_jax(
    jnp.asarray(SOURCE_PARAM, dtype=jnp.float64),
    int(BAND.band_kmin),
    int(BAND.band_kmax),
)
snr_A = matched_filter_snr_rfft(np.asarray(h_A), BAND.noise_psd_A, BAND.freqs, dt=dt)
snr_E = matched_filter_snr_rfft(np.asarray(h_E), BAND.noise_psd_E, BAND.freqs, dt=dt)
snr_T = matched_filter_snr_rfft(np.asarray(h_T), BAND.noise_psd_T, BAND.freqs, dt=dt)
snr_optimal = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))
print(f"  {BAND.label}: SNR = {snr_optimal:.1f}")


def sample_source(band: BandData, *, seed: int = 0) -> MCMC:
    """Run NUTS for one source."""
    init_start_time = time.perf_counter()
    data_A_j = jnp.asarray(band.data_A, dtype=jnp.complex128)
    data_E_j = jnp.asarray(band.data_E, dtype=jnp.complex128)
    data_T_j = jnp.asarray(band.data_T, dtype=jnp.complex128)
    psd_A_j = jnp.asarray(band.noise_psd_A, dtype=jnp.float64)
    psd_E_j = jnp.asarray(band.noise_psd_E, dtype=jnp.float64)
    psd_T_j = jnp.asarray(band.noise_psd_T, dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)
    phi_low = jnp.asarray(-jnp.pi, dtype=jnp.float64)
    phi_high = jnp.asarray(jnp.pi, dtype=jnp.float64)
    t_c = jnp.asarray(band.t_obs / 2.0, dtype=jnp.float64)
    normal01 = dist.Normal(0.0, 1.0)
    z_logf0_bounds = (
        (band.logf0_bounds[0] - band.prior_center[0]) / band.prior_scale[0],
        (band.logf0_bounds[1] - band.prior_center[0]) / band.prior_scale[0],
    )
    z_logfdot_bounds = (
        (band.logfdot_bounds[0] - band.prior_center[1]) / band.prior_scale[1],
        (band.logfdot_bounds[1] - band.prior_center[1]) / band.prior_scale[1],
    )
    z_logA_bounds = (
        (band.logA_bounds[0] - band.prior_center[2]) / band.prior_scale[2],
        (band.logA_bounds[1] - band.prior_center[2]) / band.prior_scale[2],
    )

    def model():
        z_logf0 = numpyro.sample("z_logf0", normal01)
        z_logfdot = numpyro.sample("z_logfdot", normal01)
        z_logA = numpyro.sample("z_logA", normal01)
        u_phi_c = numpyro.sample("u_phi_c", dist.Uniform(-1.0, 1.0))

        logf0 = band.prior_center[0] + band.prior_scale[0] * z_logf0
        logfdot = band.prior_center[1] + band.prior_scale[1] * z_logfdot
        logA = band.prior_center[2] + band.prior_scale[2] * z_logA
        phi_c = numpyro.deterministic("phi_c", np.pi * u_phi_c)

        numpyro.factor(
            "prior_logf0",
            dist.TruncatedNormal(
                loc=band.prior_center[0],
                scale=band.prior_scale[0],
                low=band.logf0_bounds[0],
                high=band.logf0_bounds[1],
            ).log_prob(logf0)
            + jnp.log(band.prior_scale[0])
            - normal01.log_prob(z_logf0),
        )
        numpyro.factor(
            "prior_logfdot",
            dist.TruncatedNormal(
                loc=band.prior_center[1],
                scale=band.prior_scale[1],
                low=band.logfdot_bounds[0],
                high=band.logfdot_bounds[1],
            ).log_prob(logfdot)
            + jnp.log(band.prior_scale[1])
            - normal01.log_prob(z_logfdot),
        )
        numpyro.factor(
            "prior_logA",
            dist.TruncatedNormal(
                loc=band.prior_center[2],
                scale=band.prior_scale[2],
                low=band.logA_bounds[0],
                high=band.logA_bounds[1],
            ).log_prob(logA)
            + jnp.log(band.prior_scale[2])
            - normal01.log_prob(z_logA),
        )
        numpyro.factor("prior_phi_c", jnp.log(np.pi))

        f0 = numpyro.deterministic("f0", jnp.exp(logf0))
        fdot = numpyro.deterministic("fdot", jnp.exp(logfdot))
        A = numpyro.deterministic("A", jnp.exp(logA))
        phi0 = numpyro.deterministic(
            "phi0",
            (phi_c - 2 * jnp.pi * f0 * t_c - jnp.pi * fdot * t_c**2 + jnp.pi) % (2 * jnp.pi) - jnp.pi,
        )

        params = (
            jnp.asarray(band.fixed_params, dtype=jnp.float64)
            .at[0].set(f0)
            .at[1].set(fdot)
            .at[2].set(A)
            .at[7].set(phi0)
        )
        h_A, h_E, h_T = source_aet_band_jax(params, band.band_kmin, band.band_kmax)
        residual_A_phys = dt_j * (data_A_j - h_A)
        residual_E_phys = dt_j * (data_E_j - h_E)
        residual_T_phys = dt_j * (data_T_j - h_T)
        numpyro.factor(
            "whittle",
            -jnp.sum(jnp.log(psd_A_j) + 2.0 * df_j * jnp.abs(residual_A_phys) ** 2 / psd_A_j)
            -jnp.sum(jnp.log(psd_E_j) + 2.0 * df_j * jnp.abs(residual_E_phys) ** 2 / psd_E_j)
            -jnp.sum(jnp.log(psd_T_j) + 2.0 * df_j * jnp.abs(residual_T_phys) ** 2 / psd_T_j),
        )

    def _project(point: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        return {
            "z_logf0": jnp.clip(point["z_logf0"], z_logf0_bounds[0], z_logf0_bounds[1]),
            "z_logfdot": jnp.clip(point["z_logfdot"], z_logfdot_bounds[0], z_logfdot_bounds[1]),
            "z_logA": jnp.clip(point["z_logA"], z_logA_bounds[0], z_logA_bounds[1]),
            "u_phi_c": jnp.clip(point["u_phi_c"], -1.0, 1.0),
        }

    rng = np.random.default_rng(seed + 17)
    logf0_init = float(np.clip(np.log(band.f0_init), band.logf0_bounds[0], band.logf0_bounds[1]))
    z_logf0_init = (logf0_init - band.prior_center[0]) / band.prior_scale[0]
    init_values = {
        "z_logf0": float(np.asarray(jnp.clip(
            z_logf0_init + INIT_JITTER_SCALE * rng.standard_normal(),
            z_logf0_bounds[0],
            z_logf0_bounds[1],
        ))),
        "z_logfdot": float(np.asarray(jnp.clip(
            INIT_JITTER_SCALE * rng.standard_normal(),
            z_logfdot_bounds[0],
            z_logfdot_bounds[1],
        ))),
        "z_logA": float(np.asarray(jnp.clip(
            INIT_JITTER_SCALE * rng.standard_normal(),
            z_logA_bounds[0],
            z_logA_bounds[1],
        ))),
        "u_phi_c": float(np.asarray(jnp.clip(
            0.25 * rng.standard_normal(),
            -1.0,
            1.0,
        ))),
    }
    init_elapsed = time.perf_counter() - init_start_time
    print(
        "  Init:"
        f" f0_peak={band.f0_init:.6e}"
        f" z0=[{init_values['z_logf0']:.3f}, {init_values['z_logfdot']:.3f},"
        f" {init_values['z_logA']:.3f}, {init_values['u_phi_c']:.3f}]"
        f" jitter_scale={INIT_JITTER_SCALE:.2f}"
    )
    print(f"  Timing: init proposal {init_elapsed:.2f} s")

    def _make_mcmc(init_values: dict[str, float]) -> MCMC:
        kernel = NUTS(
            model,
            init_strategy=init_to_value(values=init_values),
            dense_mass=True,
            target_accept_prob=0.9,
        )
        return MCMC(
            kernel,
            num_warmup=N_WARMUP,
            num_samples=N_DRAWS,
            num_chains=1,
            progress_bar=SHOW_PROGRESS,
        )

    print("  NUTS init attempt: FFT/prior")
    mcmc = _make_mcmc(init_values)
    mcmc_start_time = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    mcmc_elapsed = time.perf_counter() - mcmc_start_time
    print(f"  Timing: NUTS warmup+sampling {mcmc_elapsed:.2f} s")
    return mcmc


print("\nRunning NUTS for the injected source…")
print(f"\n─── {BAND.label} ───")
mcmc = sample_source(BAND, seed=MCMC_SEED)

n_div = int(mcmc.get_extra_fields()["diverging"].sum())
print(f"  Divergences: {n_div}")
if N_DRAWS >= 4:
    mcmc.print_summary(exclude_deterministic=False)
else:
    print("  Skipping NumPyro summary: need at least 4 draws for R-hat diagnostics.")

s = mcmc.get_samples()
samples = np.column_stack([
    np.asarray(s["f0"]),
    np.asarray(s["fdot"]),
    np.asarray(s["A"]),
    np.asarray(s["phi0"]),
])

samples_full = build_sampled_source_params(SOURCE_PARAM, samples)

psd_A_j = jnp.asarray(BAND.noise_psd_A, dtype=jnp.float64)
psd_E_j = jnp.asarray(BAND.noise_psd_E, dtype=jnp.float64)
psd_T_j = jnp.asarray(BAND.noise_psd_T, dtype=jnp.float64)
df_j = jnp.asarray(df, dtype=jnp.float64)
kmin_stat = BAND.band_kmin
kmax_stat = BAND.band_kmax

@jax.jit
def _get_snrs(ps):
    def _single_snr(p):
        h_A_loc, h_E_loc, h_T_loc = source_aet_band_jax(p.reshape(1, -1), kmin_stat, kmax_stat)
        h_A_tilde = dt * h_A_loc
        h_E_tilde = dt * h_E_loc
        h_T_tilde = dt * h_T_loc
        snr2 = (
            4.0 * df_j * jnp.sum(jnp.abs(h_A_tilde) ** 2 / psd_A_j)
            + 4.0 * df_j * jnp.sum(jnp.abs(h_E_tilde) ** 2 / psd_E_j)
            + 4.0 * df_j * jnp.sum(jnp.abs(h_T_tilde) ** 2 / psd_T_j)
        )
        return jnp.sqrt(snr2)
    return jax.vmap(_single_snr)(ps)

snr_eval_start = time.perf_counter()
snr_samples = np.asarray(_get_snrs(jnp.array(samples_full)))
snr_eval_elapsed = time.perf_counter() - snr_eval_start
print(f"  Timing: posterior SNR evaluation {snr_eval_elapsed:.2f} s")
samples = np.column_stack([samples, snr_samples])

PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"]
truth = source_truth_vector(BAND.fixed_params, snr=snr_optimal)

_out_path = save_posterior_archive(
    FREQ_POSTERIOR_PATH,
    source_params=SOURCE_PARAMS,
    all_samples=[samples],
    snr_optimal=[snr_optimal],
    labels=PARAM_NAMES,
    truth=truth,
)
print(f"\nSaved posteriors to {_out_path}")
