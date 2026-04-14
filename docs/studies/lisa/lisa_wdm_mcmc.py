"""WDM-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py``, truncates the
time series to a length compatible with the WDM tiling, and performs WDM-domain
Bayesian inference for one Galactic binary with a diagonal three-channel
Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Build the WDM data representation and analytic noise variance
   S[n,m] = S(f_m) / (2·dt) = S(f_m)·f_Nyquist.
3. Print the injected-source SNR (WDM band, A+E+T channels).
4. Run NumPyro NUTS on a narrow WDM band for that source.
5. Print NUTS diagnostics and 90 % CI coverage; save posterior samples.

Sky position, polarisation, and inclination are held fixed at injected values.

Performance notes
-----------------
The forward model calls ``jgb.sum_tdi`` with static (Python-int) kmin/kmax covering
the rfft bins for each source's waveform.  This lets JAX use static slice
operations (no dynamic_update_slice) and avoids the full irfft + full WDM transform
at every NUTS step.  Only the narrow per-source ``band_width`` WDM channel IFFTs
are computed.
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
    print(f"\n[lisa_wdm_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)

import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    INJECTION_PATH,
    WDM_POSTERIOR_PATH,
    build_local_prior_info,
    build_sampled_source_params,
    estimate_frequency_peak,
    load_injection,
    save_posterior_archive,
    source_truth_vector,
    trim_frequency_band,
    wrap_phase,
)
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import log_density
from wdm_transform.signal_processing import matched_filter_snr_wdm, wdm_noise_variance
from wdm_transform.transforms import forward_wdm_band

jax.config.update("jax_enable_x64", True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
# NT controls the WDM time-frequency tiling used for inference.
#
# Frequency channel spacing: df_wdm = f_Nyquist / NF = NT / (2 * T_obs).
# With NT=128, df_wdm ≈ 2 µHz — the waveform bandwidth (±jgb.n/T_obs ≈ ±8 µHz)
# spans only ~4 WDM channels, giving weak inter-channel f0 discrimination.
# With NT=4, df_wdm ≈ 63 nHz — ~256 WDM channels cover the same bandwidth,
# matching the 512 rfft bins used by the frequency-domain inference and
# recovering the same f0 precision.
NT = int(os.getenv("LISA_NT", "32"))
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "0").strip().lower() in {"1", "true", "yes", "on"}
FM_JITTER_SCALE = float(os.getenv("LISA_FM_JITTER_SCALE", "0.5"))
A_WDM = 1.0 / 3.0
D_WDM = 1.0

if not INJECTION_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached injection at {INJECTION_PATH}. "
        "Run data_generation.py first."
    )

inj = load_injection(INJECTION_PATH)
dt = inj.dt
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

_lcm = 2 * NT
n_keep = (len(data_At_full) // _lcm) * _lcm
data_At = data_At_full[:n_keep]
data_Et = data_Et_full[:n_keep]
data_Tt = data_Tt_full[:n_keep]
t_obs = n_keep * dt
NF = n_keep // NT
n_freqs = n_keep // 2 + 1
rfft_freqs = np.fft.rfftfreq(n_keep, dt)

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {n_keep}  nt = {NT}  nf = {NF}  df_wdm = {0.5 / (dt * NF):.2e} Hz"
)
print(f"Injection seed = {INJECTION_SEED}  MCMC seed = {MCMC_SEED}")

freq_grid = np.linspace(0.0, 0.5 / dt, NF + 1)
time_grid = np.arange(NT) * (t_obs / NT)
_data_A_rfft = np.fft.rfft(data_At)
_data_E_rfft = np.fft.rfft(data_Et)
_data_T_rfft = np.fft.rfft(data_Tt)
noise_psd_A_rfft_full = np.maximum(
    np.interp(rfft_freqs, freqs_saved, noise_psd_A_saved, left=noise_psd_A_saved[0], right=noise_psd_A_saved[-1]),
    1e-60,
)
noise_psd_E_rfft_full = np.maximum(
    np.interp(rfft_freqs, freqs_saved, noise_psd_E_saved, left=noise_psd_E_saved[0], right=noise_psd_E_saved[-1]),
    1e-60,
)
noise_psd_T_rfft_full = np.maximum(
    np.interp(rfft_freqs, freqs_saved, noise_psd_T_saved, left=noise_psd_T_saved[0], right=noise_psd_T_saved[-1]),
    1e-60,
)
print(f"Analysis priors match injection priors: f0 in [{inj.prior_f0[0]:.6e}, {inj.prior_f0[1]:.6e}] Hz")

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)
half = NT // 2
DF_RFFT = 1.0 / t_obs


@dataclass(frozen=True)
class WdmBandData:
    label: str
    fixed_params: np.ndarray
    band_start: int
    band_stop: int
    kmin_rfft: int
    band_rfft_size: int
    src_kmin: int
    src_kmax: int
    data_band_A: np.ndarray
    data_band_E: np.ndarray
    data_band_T: np.ndarray
    noise_var_band_A: np.ndarray
    noise_var_band_E: np.ndarray
    noise_var_band_T: np.ndarray
    t_obs: float
    f0_init: float
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


def build_wdm_band(
    src: np.ndarray,
    label: str,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
) -> WdmBandData:
    f0_init = estimate_frequency_peak(
        freqs_saved,
        np.fft.rfft(data_At_full),
        np.fft.rfft(data_Et_full),
        np.fft.rfft(data_Tt_full),
        noise_psd_A_saved,
        noise_psd_E_saved,
        noise_psd_T_saved,
        prior_f0=prior_f0,
    )
    margin = jgb.n / t_obs
    band_sl = trim_frequency_band(
        freq_grid,
        prior_f0[0] - margin,
        prior_f0[1] + margin,
        pad_bins=2,
    )

    kmin_r = max((band_sl.start - 1) * half, 0)
    kmax_r = min(band_sl.stop * half, n_freqs)

    src_kmin = max(int(np.floor(prior_f0[0] * t_obs)) - jgb.n, 0)
    src_kmax = min(int(np.ceil(prior_f0[1] * t_obs)) + jgb.n + 1, n_freqs)

    prior_info = build_local_prior_info(
        prior_f0=prior_f0,
        prior_fdot=prior_fdot,
        prior_A=prior_A,
    )

    x_data_A_local = jnp.asarray(_data_A_rfft[kmin_r:kmax_r], dtype=jnp.complex128)
    x_data_E_local = jnp.asarray(_data_E_rfft[kmin_r:kmax_r], dtype=jnp.complex128)
    x_data_T_local = jnp.asarray(_data_T_rfft[kmin_r:kmax_r], dtype=jnp.complex128)
    data_band_A = forward_wdm_band(
        x_data_A_local,
        df=DF_RFFT,
        nfreqs_fourier=n_freqs,
        kmin=kmin_r,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=band_sl.start,
        nf_sub_wdm=band_sl.stop - band_sl.start,
        a=A_WDM,
        d=D_WDM,
        backend="jax",
    )
    data_band_E = forward_wdm_band(
        x_data_E_local,
        df=DF_RFFT,
        nfreqs_fourier=n_freqs,
        kmin=kmin_r,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=band_sl.start,
        nf_sub_wdm=band_sl.stop - band_sl.start,
        a=A_WDM,
        d=D_WDM,
        backend="jax",
    )
    data_band_T = forward_wdm_band(
        x_data_T_local,
        df=DF_RFFT,
        nfreqs_fourier=n_freqs,
        kmin=kmin_r,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=band_sl.start,
        nf_sub_wdm=band_sl.stop - band_sl.start,
        a=A_WDM,
        d=D_WDM,
        backend="jax",
    )
    data_band_A = np.asarray(data_band_A)
    data_band_E = np.asarray(data_band_E)
    data_band_T = np.asarray(data_band_T)

    band_freqs = freq_grid[band_sl.start:band_sl.stop]
    noise_psd_band_A = np.maximum(
        np.interp(band_freqs, freqs_saved, noise_psd_A_saved,
                  left=noise_psd_A_saved[0], right=noise_psd_A_saved[-1]),
        1e-60,
    )
    noise_psd_band_E = np.maximum(
        np.interp(band_freqs, freqs_saved, noise_psd_E_saved,
                  left=noise_psd_E_saved[0], right=noise_psd_E_saved[-1]),
        1e-60,
    )
    noise_psd_band_T = np.maximum(
        np.interp(band_freqs, freqs_saved, noise_psd_T_saved,
                  left=noise_psd_T_saved[0], right=noise_psd_T_saved[-1]),
        1e-60,
    )
    noise_var_band_A = wdm_noise_variance(noise_psd_band_A, nt=NT, dt=dt)
    noise_var_band_E = wdm_noise_variance(noise_psd_band_E, nt=NT, dt=dt)
    noise_var_band_T = wdm_noise_variance(noise_psd_band_T, nt=NT, dt=dt)

    return WdmBandData(
        label=label,
        fixed_params=src.copy(),
        band_start=band_sl.start,
        band_stop=band_sl.stop,
        kmin_rfft=kmin_r,
        band_rfft_size=kmax_r - kmin_r,
        src_kmin=src_kmin,
        src_kmax=src_kmax,
        data_band_A=data_band_A,
        data_band_E=data_band_E,
        data_band_T=data_band_T,
        noise_var_band_A=noise_var_band_A,
        noise_var_band_E=noise_var_band_E,
        noise_var_band_T=noise_var_band_T,
        t_obs=t_obs,
        f0_init=f0_init,
        prior_center=prior_info.prior_center,
        prior_scale=prior_info.prior_scale,
        logf0_bounds=prior_info.logf0_bounds,
        logfdot_bounds=prior_info.logfdot_bounds,
        logA_bounds=prior_info.logA_bounds,
    )


BAND = build_wdm_band(SOURCE_PARAM, "Injected GB", inj.prior_f0, inj.prior_fdot, inj.prior_A)

del _data_A_rfft
del _data_E_rfft
del _data_T_rfft

print(
    f"WDM band {BAND.label}: "
    f"[{freq_grid[BAND.band_start]:.4e}, {freq_grid[BAND.band_stop - 1]:.4e}] Hz  "
    f"({BAND.band_stop - BAND.band_start} channels)"
)


def generate_aet_wdm_for(
    params: jnp.ndarray, wband: WdmBandData
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a_loc, e_loc, t_loc = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=wband.src_kmin,
        kmax=wband.src_kmax,
        tdi_combination="AET",
    )
    return local_rfft_to_wdm(
        (
            jnp.asarray(a_loc, dtype=jnp.complex128),
            jnp.asarray(e_loc, dtype=jnp.complex128),
            jnp.asarray(t_loc, dtype=jnp.complex128),
        ),
        wband,
    )


def local_rfft_to_wdm(
    local_modes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    wband: WdmBandData,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    local_start = wband.src_kmin - wband.kmin_rfft
    local_end = wband.src_kmax - wband.kmin_rfft

    def _embed_and_transform(x_loc: jnp.ndarray) -> jnp.ndarray:
        x_local = (
            jnp.zeros(wband.band_rfft_size, dtype=jnp.complex128)
            .at[local_start:local_end]
            .set(jnp.asarray(x_loc, dtype=jnp.complex128))
        )
        return forward_wdm_band(
            x_local,
            df=DF_RFFT,
            nfreqs_fourier=n_freqs,
            kmin=wband.kmin_rfft,
            nfreqs_wdm=NF,
            ntimes_wdm=NT,
            mmin=wband.band_start,
            nf_sub_wdm=wband.band_stop - wband.band_start,
            a=A_WDM,
            d=D_WDM,
            backend="jax",
        )

    return tuple(_embed_and_transform(x_loc) for x_loc in local_modes)


print("\nMatched-filter SNR (A+E+T channels, WDM band):")
h_A_band, h_E_band, h_T_band = generate_aet_wdm_for(
    jnp.asarray(BAND.fixed_params, dtype=jnp.float64), BAND
)
snr_A = matched_filter_snr_wdm(np.asarray(h_A_band), BAND.noise_var_band_A)
snr_E = matched_filter_snr_wdm(np.asarray(h_E_band), BAND.noise_var_band_E)
snr_T = matched_filter_snr_wdm(np.asarray(h_T_band), BAND.noise_var_band_T)
snr_optimal = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))
print(f"  {BAND.label}: SNR = {snr_optimal:.1f}")


def sample_source_wdm(wband: WdmBandData, *, seed: int = 0) -> MCMC:
    init_start_time = time.perf_counter()
    data_A_j = jnp.asarray(wband.data_band_A, dtype=jnp.float64)
    data_E_j = jnp.asarray(wband.data_band_E, dtype=jnp.float64)
    data_T_j = jnp.asarray(wband.data_band_T, dtype=jnp.float64)
    noise_var_A_j = jnp.asarray(wband.noise_var_band_A, dtype=jnp.float64)
    noise_var_E_j = jnp.asarray(wband.noise_var_band_E, dtype=jnp.float64)
    noise_var_T_j = jnp.asarray(wband.noise_var_band_T, dtype=jnp.float64)
    fixed_params_j = jnp.asarray(wband.fixed_params, dtype=jnp.float64)
    phi_low = jnp.asarray(-jnp.pi, dtype=jnp.float64)
    phi_high = jnp.asarray(jnp.pi, dtype=jnp.float64)
    t_c = jnp.asarray(wband.t_obs / 2.0, dtype=jnp.float64)
    normal01 = dist.Normal(0.0, 1.0)
    z_logf0_bounds = (
        (wband.logf0_bounds[0] - wband.prior_center[0]) / wband.prior_scale[0],
        (wband.logf0_bounds[1] - wband.prior_center[0]) / wband.prior_scale[0],
    )
    z_logfdot_bounds = (
        (wband.logfdot_bounds[0] - wband.prior_center[1]) / wband.prior_scale[1],
        (wband.logfdot_bounds[1] - wband.prior_center[1]) / wband.prior_scale[1],
    )
    z_logA_bounds = (
        (wband.logA_bounds[0] - wband.prior_center[2]) / wband.prior_scale[2],
        (wband.logA_bounds[1] - wband.prior_center[2]) / wband.prior_scale[2],
    )

    def generate(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_loc, e_loc, t_loc = jgb.sum_tdi(
            jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
            kmin=wband.src_kmin,
            kmax=wband.src_kmax,
            tdi_combination="AET",
        )
        return local_rfft_to_wdm(
            (
                jnp.asarray(a_loc, dtype=jnp.complex128),
                jnp.asarray(e_loc, dtype=jnp.complex128),
                jnp.asarray(t_loc, dtype=jnp.complex128),
            ),
            wband,
        )

    def model():
        z_logf0 = numpyro.sample("z_logf0", normal01)
        z_logfdot = numpyro.sample("z_logfdot", normal01)
        z_logA = numpyro.sample("z_logA", normal01)
        u_phi_c = numpyro.sample("u_phi_c", dist.Uniform(-1.0, 1.0))

        logf0 = wband.prior_center[0] + wband.prior_scale[0] * z_logf0
        logfdot = wband.prior_center[1] + wband.prior_scale[1] * z_logfdot
        logA = wband.prior_center[2] + wband.prior_scale[2] * z_logA
        phi_c = numpyro.deterministic("phi_c", np.pi * u_phi_c)

        numpyro.factor(
            "prior_logf0",
            dist.TruncatedNormal(
                loc=wband.prior_center[0],
                scale=wband.prior_scale[0],
                low=wband.logf0_bounds[0],
                high=wband.logf0_bounds[1],
            ).log_prob(logf0)
            + jnp.log(wband.prior_scale[0])
            - normal01.log_prob(z_logf0),
        )
        numpyro.factor(
            "prior_logfdot",
            dist.TruncatedNormal(
                loc=wband.prior_center[1],
                scale=wband.prior_scale[1],
                low=wband.logfdot_bounds[0],
                high=wband.logfdot_bounds[1],
            ).log_prob(logfdot)
            + jnp.log(wband.prior_scale[1])
            - normal01.log_prob(z_logfdot),
        )
        numpyro.factor(
            "prior_logA",
            dist.TruncatedNormal(
                loc=wband.prior_center[2],
                scale=wband.prior_scale[2],
                low=wband.logA_bounds[0],
                high=wband.logA_bounds[1],
            ).log_prob(logA)
            + jnp.log(wband.prior_scale[2])
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
            fixed_params_j
            .at[0].set(f0)
            .at[1].set(fdot)
            .at[2].set(A)
            .at[7].set(phi0)
        )
        h_A, h_E, h_T = generate(params)
        diff_A = data_A_j - h_A
        diff_E = data_E_j - h_E
        diff_T = data_T_j - h_T
        numpyro.factor(
            "ll",
            -0.5 * jnp.sum(diff_A**2 / noise_var_A_j + jnp.log(2.0 * jnp.pi * noise_var_A_j))
            -0.5 * jnp.sum(diff_E**2 / noise_var_E_j + jnp.log(2.0 * jnp.pi * noise_var_E_j))
            -0.5 * jnp.sum(diff_T**2 / noise_var_T_j + jnp.log(2.0 * jnp.pi * noise_var_T_j)),
        )

    def _project(point: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        return {
            "z_logf0": jnp.clip(point["z_logf0"], z_logf0_bounds[0], z_logf0_bounds[1]),
            "z_logfdot": jnp.clip(point["z_logfdot"], z_logfdot_bounds[0], z_logfdot_bounds[1]),
            "z_logA": jnp.clip(point["z_logA"], z_logA_bounds[0], z_logA_bounds[1]),
            "u_phi_c": jnp.clip(point["u_phi_c"], -1.0, 1.0),
        }

    def _log_post(point: dict[str, jnp.ndarray]) -> jnp.ndarray:
        value, _ = log_density(model, (), {}, _project(point))
        return value

    def _pack(theta: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {
            "z_logf0": theta[0],
            "z_logfdot": theta[1],
            "z_logA": theta[2],
            "u_phi_c": theta[3],
        }

    def _flat_log_post(theta: jnp.ndarray) -> jnp.ndarray:
        return _log_post(_pack(theta))

    phi_c_truth = wrap_phase(
        float(SOURCE_PARAM[7])
        + 2.0 * np.pi * float(SOURCE_PARAM[0]) * (wband.t_obs / 2.0)
        + np.pi * float(SOURCE_PARAM[1]) * (wband.t_obs / 2.0) ** 2
    )
    theta_truth = jnp.asarray(
        [
            (np.log(float(SOURCE_PARAM[0])) - wband.prior_center[0]) / wband.prior_scale[0],
            (np.log(float(SOURCE_PARAM[1])) - wband.prior_center[1]) / wband.prior_scale[1],
            (np.log(float(SOURCE_PARAM[2])) - wband.prior_center[2]) / wband.prior_scale[2],
            float(np.clip(phi_c_truth / np.pi, -1.0, 1.0)),
        ],
        dtype=jnp.float64,
    )
    hessian = np.asarray(jax.hessian(_flat_log_post)(theta_truth), dtype=float)
    information = -0.5 * (hessian + hessian.T)
    eigvals, eigvecs = np.linalg.eigh(information)
    floor = max(1e-8, 1e-6 * float(np.max(np.abs(eigvals))))
    eigvals = np.clip(eigvals, floor, None)
    covariance = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rng = np.random.default_rng(seed + 17)
    theta_init = np.asarray(theta_truth, dtype=float) + FM_JITTER_SCALE * sigma * rng.standard_normal(4)
    init_values = {
        key: float(np.asarray(val))
        for key, val in _project(_pack(jnp.asarray(theta_init, dtype=jnp.float64))).items()
    }
    init_elapsed = time.perf_counter() - init_start_time
    print(
        "  FM init:"
        f" sigmas=[{sigma[0]:.3e}, {sigma[1]:.3e}, {sigma[2]:.3e}, {sigma[3]:.3e}]"
        f" jitter_scale={FM_JITTER_SCALE:.2f}"
    )
    print(f"  Timing: FM init {init_elapsed:.2f} s")

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

    print("  NUTS init attempt: FM truth-jitter")
    mcmc = _make_mcmc(init_values)
    mcmc_start_time = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    mcmc_elapsed = time.perf_counter() - mcmc_start_time
    print(f"  Timing: NUTS warmup+sampling {mcmc_elapsed:.2f} s")
    return mcmc


print("\nRunning NUTS for the injected source…")
PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]"]
print(f"\n─── {BAND.label} ───")
mcmc = sample_source_wdm(BAND, seed=MCMC_SEED)

print("  Stage: summarizing posterior")
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

truth = source_truth_vector(BAND.fixed_params)

samples_full = build_sampled_source_params(BAND.fixed_params, samples)

noise_var_A_j = jnp.asarray(BAND.noise_var_band_A, dtype=jnp.float64)
noise_var_E_j = jnp.asarray(BAND.noise_var_band_E, dtype=jnp.float64)
noise_var_T_j = jnp.asarray(BAND.noise_var_band_T, dtype=jnp.float64)

@jax.jit
def _get_snrs(ps, wb=BAND):
    def _single_snr(p):
        a_loc_s, e_loc_s, t_loc_s = jgb.sum_tdi(
            p.reshape(1, -1), kmin=wb.src_kmin, kmax=wb.src_kmax,
            tdi_combination="AET",
        )
        h_A_wdm, h_E_wdm, h_T_wdm = local_rfft_to_wdm(
            (a_loc_s, e_loc_s, t_loc_s),
            wb,
        )
        snr2 = (
            jnp.sum(h_A_wdm**2 / noise_var_A_j)
            + jnp.sum(h_E_wdm**2 / noise_var_E_j)
            + jnp.sum(h_T_wdm**2 / noise_var_T_j)
        )
        return jnp.sqrt(snr2)
    return jax.vmap(_single_snr)(ps)

print("  Stage: evaluating posterior SNR samples")
snr_eval_start = time.perf_counter()
snr_samples = np.asarray(_get_snrs(jnp.array(samples_full)))
snr_eval_elapsed = time.perf_counter() - snr_eval_start
print(f"  Timing: posterior SNR evaluation {snr_eval_elapsed:.2f} s")
samples = np.column_stack([samples, snr_samples])

PARAM_NAMES_SNR = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"]
truth_snr = source_truth_vector(BAND.fixed_params, snr=snr_optimal)

_out_path = save_posterior_archive(
    WDM_POSTERIOR_PATH,
    source_params=SOURCE_PARAMS,
    all_samples=[samples],
    snr_optimal=[snr_optimal],
    labels=PARAM_NAMES_SNR,
    truth=truth_snr,
)
print(f"\nSaved posteriors to {_out_path}")
