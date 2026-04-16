"""WDM-domain LISA GB inference with NumPyro."""

from __future__ import annotations

import atexit
import json
import os
import time

import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
import numpyro
import numpyro.distributions as dist
from gb_prior import build_local_prior_info
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    FREQ_POSTERIOR_PATH,
    INJECTION_PATH,
    WDM_POSTERIOR_PATH,
    build_sampled_source_params,
    interp_psd_channels,
    load_injection,
    load_posterior_samples_source,
    print_cross_domain_diagnostics,
    save_posterior_archive,
    setup_jax_and_matplotlib,
    source_truth_vector,
    trim_frequency_band,
)
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform.signal_processing import (
    matched_filter_snr_rfft,
    matched_filter_snr_wdm,
    wdm_noise_variance,
)
from wdm_transform.transforms import forward_wdm_band

setup_jax_and_matplotlib()
jax.config.update("jax_enable_x64", True)

_SCRIPT_START = time.perf_counter()
N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
NT = int(os.getenv("LISA_NT", "32"))
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "1").strip().lower() in {"1", "true", "yes", "on"}
NUTS_TARGET_ACCEPT = float(os.getenv("LISA_NUTS_TARGET_ACCEPT", "0.85"))
NUTS_MAX_TREE_DEPTH = int(os.getenv("LISA_NUTS_MAX_TREE_DEPTH", "10"))
NUTS_DENSE_MASS = os.getenv("LISA_NUTS_DENSE_MASS", "1").strip().lower() in {"1", "true", "yes", "on"}
INIT_JITTER_SCALE = float(os.getenv("LISA_INIT_JITTER_SCALE", "0.15"))
A_WDM = 1.0 / 3.0
D_WDM = 1.0


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[lisa_wdm_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)


def generate_aet_rfft(
    jgb: JaxGB,
    params: jnp.ndarray,
    kmin: int,
    kmax: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the local A/E/T FFT modes for one source."""
    a_loc, e_loc, t_loc = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=kmin,
        kmax=kmax,
        tdi_combination="AET",
        tdi_generation=1.5,
    )
    return (
        jnp.asarray(a_loc, dtype=jnp.complex128),
        jnp.asarray(e_loc, dtype=jnp.complex128),
        jnp.asarray(t_loc, dtype=jnp.complex128),
    )


def local_rfft_to_wdm(
    local_modes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    src_kmin: int,
    src_kmax: int,
    kmin_rfft: int,
    band_rfft_size: int,
    band_start: int,
    band_stop: int,
    df_rfft: float,
    n_freqs: int,
    nf: int,
    nt: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Embed local FFT modes into the WDM patch and transform them."""
    local_start = src_kmin - kmin_rfft
    local_end = src_kmax - kmin_rfft

    def embed_and_transform(x_loc: jnp.ndarray) -> jnp.ndarray:
        embedded = (
            jnp.zeros(band_rfft_size, dtype=jnp.complex128)
            .at[local_start:local_end]
            .set(jnp.asarray(x_loc, dtype=jnp.complex128))
        )
        return forward_wdm_band(
            embedded,
            df=df_rfft,
            nfreqs_fourier=n_freqs,
            kmin=kmin_rfft,
            nfreqs_wdm=nf,
            ntimes_wdm=nt,
            mmin=band_start,
            nf_sub_wdm=band_stop - band_start,
            a=A_WDM,
            d=D_WDM,
            backend="jax",
        )

    return tuple(embed_and_transform(mode) for mode in local_modes)


def build_wdm_inputs(
    *,
    source_param: np.ndarray,
    injection,
    jgb: JaxGB,
    data_a_rfft: np.ndarray,
    data_e_rfft: np.ndarray,
    data_t_rfft: np.ndarray,
    t_obs: float,
    n_freqs: int,
    nf: int,
    nt: int,
    df_rfft: float,
    freq_grid: np.ndarray,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Return the local WDM patch and prior metadata."""
    f0_init = float(injection.f0_ref)
    margin = jgb.n / t_obs
    band_slice = trim_frequency_band(
        freq_grid,
        injection.prior_f0[0] - margin,
        injection.prior_f0[1] + margin,
        pad_bins=2,
    )
    half = nt // 2
    kmin_rfft = max((band_slice.start - 1) * half, 0)
    kmax_rfft = min(band_slice.stop * half, n_freqs)
    src_kmin = max(int(np.floor(injection.prior_f0[0] * t_obs)) - jgb.n, 0)
    src_kmax = min(int(np.ceil(injection.prior_f0[1] * t_obs)) + jgb.n + 1, n_freqs)
    prior_info = build_local_prior_info(
        prior_f0=injection.prior_f0,
        prior_fdot=injection.prior_fdot,
        prior_A=injection.prior_A,
    )

    band_kwargs = {
        "df": df_rfft,
        "nfreqs_fourier": n_freqs,
        "kmin": kmin_rfft,
        "nfreqs_wdm": nf,
        "ntimes_wdm": nt,
        "mmin": band_slice.start,
        "nf_sub_wdm": band_slice.stop - band_slice.start,
        "a": A_WDM,
        "d": D_WDM,
        "backend": "jax",
    }
    band_freqs = freq_grid[band_slice.start:band_slice.stop]
    noise_psd_band_a = np.maximum(
        np.interp(band_freqs, injection.freqs, injection.noise_psd_A, left=injection.noise_psd_A[0], right=injection.noise_psd_A[-1]),
        1e-60,
    )
    noise_psd_band_e = np.maximum(
        np.interp(band_freqs, injection.freqs, injection.noise_psd_E, left=injection.noise_psd_E[0], right=injection.noise_psd_E[-1]),
        1e-60,
    )
    noise_psd_band_t = np.maximum(
        np.interp(band_freqs, injection.freqs, injection.noise_psd_T, left=injection.noise_psd_T[0], right=injection.noise_psd_T[-1]),
        1e-60,
    )

    return {
        "fixed_params": source_param.copy(),
        "t_obs": t_obs,
        "band_start": band_slice.start,
        "band_stop": band_slice.stop,
        "kmin_rfft": kmin_rfft,
        "band_rfft_size": kmax_rfft - kmin_rfft,
        "src_kmin": src_kmin,
        "src_kmax": src_kmax,
        "data_band_A": np.asarray(forward_wdm_band(jnp.asarray(data_a_rfft[kmin_rfft:kmax_rfft]), **band_kwargs)),
        "data_band_E": np.asarray(forward_wdm_band(jnp.asarray(data_e_rfft[kmin_rfft:kmax_rfft]), **band_kwargs)),
        "data_band_T": np.asarray(forward_wdm_band(jnp.asarray(data_t_rfft[kmin_rfft:kmax_rfft]), **band_kwargs)),
        "noise_var_band_A": wdm_noise_variance(noise_psd_band_a, nt=nt, dt=injection.dt),
        "noise_var_band_E": wdm_noise_variance(noise_psd_band_e, nt=nt, dt=injection.dt),
        "noise_var_band_T": wdm_noise_variance(noise_psd_band_t, nt=nt, dt=injection.dt),
        "f0_ref": float(injection.f0_ref),
        "f0_init": float(f0_init),
        "logf0_ref": float(np.log(f0_init)),
        "prior_center": prior_info.prior_center,
        "prior_scale": prior_info.prior_scale,
        "logfdot_bounds": prior_info.logfdot_bounds,
        "logA_bounds": prior_info.logA_bounds,
    }


def compute_truth_snrs(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
    rfft_freqs: np.ndarray,
    noise_psd_a_rfft: np.ndarray,
    noise_psd_e_rfft: np.ndarray,
    noise_psd_t_rfft: np.ndarray,
) -> tuple[float, float]:
    """Compute the injected-source truth SNR in frequency and WDM domains."""
    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    a_loc, e_loc, t_loc = generate_aet_rfft(
        jgb,
        fixed_params,
        int(band["src_kmin"]),
        int(band["src_kmax"]),
    )
    h_a_wdm, h_e_wdm, h_t_wdm = local_rfft_to_wdm(
        (a_loc, e_loc, t_loc),
        src_kmin=int(band["src_kmin"]),
        src_kmax=int(band["src_kmax"]),
        kmin_rfft=int(band["kmin_rfft"]),
        band_rfft_size=int(band["band_rfft_size"]),
        band_start=int(band["band_start"]),
        band_stop=int(band["band_stop"]),
        df_rfft=df_rfft,
        n_freqs=n_freqs,
        nf=nf,
        nt=nt,
    )
    snr_wdm = float(
        np.linalg.norm(
            [
                matched_filter_snr_wdm(np.asarray(h_a_wdm), np.asarray(band["noise_var_band_A"])),
                matched_filter_snr_wdm(np.asarray(h_e_wdm), np.asarray(band["noise_var_band_E"])),
                matched_filter_snr_wdm(np.asarray(h_t_wdm), np.asarray(band["noise_var_band_T"])),
            ]
        )
    )
    band_freqs_rfft = rfft_freqs[int(band["src_kmin"]):int(band["src_kmax"])]
    snr_freq = float(
        np.linalg.norm(
            [
                matched_filter_snr_rfft(
                    np.asarray(a_loc),
                    noise_psd_a_rfft[int(band["src_kmin"]):int(band["src_kmax"])],
                    band_freqs_rfft,
                    dt=dt,
                ),
                matched_filter_snr_rfft(
                    np.asarray(e_loc),
                    noise_psd_e_rfft[int(band["src_kmin"]):int(band["src_kmax"])],
                    band_freqs_rfft,
                    dt=dt,
                ),
                matched_filter_snr_rfft(
                    np.asarray(t_loc),
                    noise_psd_t_rfft[int(band["src_kmin"]):int(band["src_kmax"])],
                    band_freqs_rfft,
                    dt=dt,
                ),
            ]
        )
    )
    return snr_freq, snr_wdm


def evaluate_wdm_loglike(
    *,
    jgb: JaxGB,
    params: np.ndarray,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
) -> float:
    """Evaluate the diagonal WDM Gaussian log-likelihood."""
    h_a, h_e, h_t = local_rfft_to_wdm(
        generate_aet_rfft(jgb, params, int(band["src_kmin"]), int(band["src_kmax"])),
        src_kmin=int(band["src_kmin"]),
        src_kmax=int(band["src_kmax"]),
        kmin_rfft=int(band["kmin_rfft"]),
        band_rfft_size=int(band["band_rfft_size"]),
        band_start=int(band["band_start"]),
        band_stop=int(band["band_stop"]),
        df_rfft=df_rfft,
        n_freqs=n_freqs,
        nf=nf,
        nt=nt,
    )
    diff_a = np.asarray(band["data_band_A"]) - np.asarray(h_a)
    diff_e = np.asarray(band["data_band_E"]) - np.asarray(h_e)
    diff_t = np.asarray(band["data_band_T"]) - np.asarray(h_t)
    return float(
        -0.5
        * (
            np.sum(diff_a**2 / np.asarray(band["noise_var_band_A"]) + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_A"])))
            + np.sum(diff_e**2 / np.asarray(band["noise_var_band_E"]) + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_E"])))
            + np.sum(diff_t**2 / np.asarray(band["noise_var_band_T"]) + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_T"])))
        )
    )


def sample_source_wdm(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    f0_jitter_width: float,
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
    seed: int,
) -> tuple[MCMC, dict[str, float]]:
    """Run NUTS for the local WDM-domain model."""
    data_a_j = jnp.asarray(band["data_band_A"], dtype=jnp.float64)
    data_e_j = jnp.asarray(band["data_band_E"], dtype=jnp.float64)
    data_t_j = jnp.asarray(band["data_band_T"], dtype=jnp.float64)
    noise_var_a_j = jnp.asarray(band["noise_var_band_A"], dtype=jnp.float64)
    noise_var_e_j = jnp.asarray(band["noise_var_band_E"], dtype=jnp.float64)
    noise_var_t_j = jnp.asarray(band["noise_var_band_T"], dtype=jnp.float64)
    fixed_params_j = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    prior_center = jnp.asarray(band["prior_center"], dtype=jnp.float64)
    prior_scale = jnp.asarray(band["prior_scale"], dtype=jnp.float64)
    logfdot_bounds = tuple(np.asarray(band["logfdot_bounds"], dtype=float))
    logA_bounds = tuple(np.asarray(band["logA_bounds"], dtype=float))
    logf0_ref = float(band["logf0_ref"])

    def generate(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return local_rfft_to_wdm(
            generate_aet_rfft(jgb, params, int(band["src_kmin"]), int(band["src_kmax"])),
            src_kmin=int(band["src_kmin"]),
            src_kmax=int(band["src_kmax"]),
            kmin_rfft=int(band["kmin_rfft"]),
            band_rfft_size=int(band["band_rfft_size"]),
            band_start=int(band["band_start"]),
            band_stop=int(band["band_stop"]),
            df_rfft=df_rfft,
            n_freqs=n_freqs,
            nf=nf,
            nt=nt,
        )

    def model() -> None:
        delta_logf0 = numpyro.sample("delta_logf0", dist.Uniform(-f0_jitter_width, f0_jitter_width))
        logfdot = numpyro.sample(
            "logfdot",
            dist.TruncatedNormal(
                loc=prior_center[1],
                scale=prior_scale[1],
                low=logfdot_bounds[0],
                high=logfdot_bounds[1],
            ),
        )
        logA = numpyro.sample(
            "logA",
            dist.TruncatedNormal(
                loc=prior_center[2],
                scale=prior_scale[2],
                low=logA_bounds[0],
                high=logA_bounds[1],
            ),
        )
        phi0 = numpyro.sample("phi0", dist.Uniform(-jnp.pi, jnp.pi))
        f0 = jnp.exp(logf0_ref + delta_logf0)
        fdot = jnp.exp(logfdot)
        amplitude = jnp.exp(logA)
        params = (
            fixed_params_j
            .at[0].set(f0)
            .at[1].set(fdot)
            .at[2].set(amplitude)
            .at[7].set(phi0)
        )
        h_a, h_e, h_t = generate(params)
        numpyro.factor(
            "ll",
            -0.5 * jnp.sum((data_a_j - h_a) ** 2 / noise_var_a_j + jnp.log(2.0 * jnp.pi * noise_var_a_j))
            -0.5 * jnp.sum((data_e_j - h_e) ** 2 / noise_var_e_j + jnp.log(2.0 * jnp.pi * noise_var_e_j))
            -0.5 * jnp.sum((data_t_j - h_t) ** 2 / noise_var_t_j + jnp.log(2.0 * jnp.pi * noise_var_t_j)),
        )
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", amplitude)

    rng = np.random.default_rng(seed + 17)
    init_eps = 1e-6
    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    init_values = {
        "delta_logf0": float(
            np.clip(
                np.log(fixed_params[0]) - logf0_ref,
                -f0_jitter_width + init_eps,
                f0_jitter_width - init_eps,
            )
        ),
        "logfdot": float(
            np.clip(
                np.log(fixed_params[1]) + INIT_JITTER_SCALE * rng.standard_normal(),
                logfdot_bounds[0] + init_eps,
                logfdot_bounds[1] - init_eps,
            )
        ),
        "logA": float(
            np.clip(
                np.log(fixed_params[2]) + INIT_JITTER_SCALE * rng.standard_normal(),
                logA_bounds[0] + init_eps,
                logA_bounds[1] - init_eps,
            )
        ),
        "phi0": float(np.clip(fixed_params[7], -np.pi + init_eps, np.pi - init_eps)),
    }
    mcmc = MCMC(
        NUTS(
            model,
            init_strategy=init_to_value(values=init_values),
            dense_mass=NUTS_DENSE_MASS,
            target_accept_prob=NUTS_TARGET_ACCEPT,
            max_tree_depth=NUTS_MAX_TREE_DEPTH,
        ),
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=1,
        progress_bar=SHOW_PROGRESS,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    return mcmc, init_values


def compute_snr_samples(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
    samples_full: np.ndarray,
) -> np.ndarray:
    """Evaluate the WDM-domain optimal SNR for each posterior sample."""
    noise_var_a_j = jnp.asarray(band["noise_var_band_A"], dtype=jnp.float64)
    noise_var_e_j = jnp.asarray(band["noise_var_band_E"], dtype=jnp.float64)
    noise_var_t_j = jnp.asarray(band["noise_var_band_T"], dtype=jnp.float64)

    @jax.jit
    def get_snrs(params: jnp.ndarray) -> jnp.ndarray:
        def single_snr(source_params: jnp.ndarray) -> jnp.ndarray:
            h_a, h_e, h_t = local_rfft_to_wdm(
                generate_aet_rfft(jgb, source_params, int(band["src_kmin"]), int(band["src_kmax"])),
                src_kmin=int(band["src_kmin"]),
                src_kmax=int(band["src_kmax"]),
                kmin_rfft=int(band["kmin_rfft"]),
                band_rfft_size=int(band["band_rfft_size"]),
                band_start=int(band["band_start"]),
                band_stop=int(band["band_stop"]),
                df_rfft=df_rfft,
                n_freqs=n_freqs,
                nf=nf,
                nt=nt,
            )
            return jnp.sqrt(
                jnp.sum(h_a**2 / noise_var_a_j)
                + jnp.sum(h_e**2 / noise_var_e_j)
                + jnp.sum(h_t**2 / noise_var_t_j)
            )

        return jax.vmap(single_snr)(params)

    return np.asarray(get_snrs(jnp.asarray(samples_full, dtype=jnp.float64)))


def save_sampler_diagnostics(
    *,
    diagnostics_path,
    injection_seed: int,
    mcmc_seed: int,
    nt: int,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    samples: dict[str, np.ndarray],
    init_values: dict[str, float],
    n_divergences: int,
) -> None:
    """Write a compact JSON summary of the WDM sampler diagnostics."""
    latent_summary = numpyro_summary(
        {
            "delta_logf0": samples["delta_logf0"],
            "logfdot": samples["logfdot"],
            "logA": samples["logA"],
            "phi0": samples["phi0"],
        },
        group_by_chain=False,
    )
    payload = {
        "seed": injection_seed,
        "mcmc_seed": mcmc_seed,
        "num_warmup": N_WARMUP,
        "num_draws": N_DRAWS,
        "nt": nt,
        "divergences": n_divergences,
        "init_values": {
            **{key: float(value) for key, value in init_values.items()},
            "f0": float(np.exp(float(band["logf0_ref"]) + init_values["delta_logf0"])),
            "fdot": float(np.exp(init_values["logfdot"])),
            "A": float(np.exp(init_values["logA"])),
        },
        "latent_diagnostics": {
            key: {
                "n_eff": float(np.asarray(stats["n_eff"])),
                "r_hat": float(np.asarray(stats["r_hat"])),
            }
            for key, stats in latent_summary.items()
        },
    }
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    if not INJECTION_PATH.exists():
        raise FileNotFoundError(
            f"Expected cached injection at {INJECTION_PATH}. Run data_generation.py first."
        )

    injection = load_injection(INJECTION_PATH)
    source_params = injection.source_params
    source_param = source_params[0].copy()
    mcmc_seed = injection.seed + 10
    diagnostics_path = WDM_POSTERIOR_PATH.with_name("wdm_sampler_diagnostics.json")

    n_keep = (len(injection.data_At) // (2 * NT)) * (2 * NT)
    data_a = injection.data_At[:n_keep]
    data_e = injection.data_Et[:n_keep]
    data_t = injection.data_Tt[:n_keep]
    # Use the full injection baseline for the template (matches data_generation.py).
    # Truncation of the data array to n_keep is required by the WDM (2*NT) divisibility
    # constraint; it loses at most a few samples out of millions, so the sub-bin FFT
    # grid mismatch is negligible compared to the template-baseline bias it would
    # introduce if we set t_obs = n_keep * dt here.
    t_obs = injection.t_obs
    nf = n_keep // NT
    n_freqs = n_keep // 2 + 1
    df_rfft = 1.0 / t_obs
    rfft_freqs = np.fft.rfftfreq(n_keep, injection.dt)
    freq_grid = np.linspace(0.0, 0.5 / injection.dt, nf + 1)
    data_a_rfft = np.fft.rfft(data_a)
    data_e_rfft = np.fft.rfft(data_e)
    data_t_rfft = np.fft.rfft(data_t)
    noise_psd_rfft = interp_psd_channels(
        rfft_freqs,
        injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
    band = build_wdm_inputs(
        source_param=source_param,
        injection=injection,
        jgb=jgb,
        data_a_rfft=data_a_rfft,
        data_e_rfft=data_e_rfft,
        data_t_rfft=data_t_rfft,
        t_obs=t_obs,
        n_freqs=n_freqs,
        nf=nf,
        nt=NT,
        df_rfft=df_rfft,
        freq_grid=freq_grid,
    )
    truth_snr_frequency, truth_snr_wdm = compute_truth_snrs(
        jgb=jgb,
        band=band,
        dt=injection.dt,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
        rfft_freqs=rfft_freqs,
        noise_psd_a_rfft=noise_psd_rfft[0],
        noise_psd_e_rfft=noise_psd_rfft[1],
        noise_psd_t_rfft=noise_psd_rfft[2],
    )

    print(
        f"T_obs={t_obs / 86400.0:.1f}d, N={n_keep}, nt={NT}, nf={nf}, "
        f"seed={injection.seed}, MCMC_seed={mcmc_seed}"
    )
    print(f"Frequency truth-band SNR={truth_snr_frequency:.1f}")
    print(f"WDM band SNR={truth_snr_wdm:.1f}")

    mcmc, init_values = sample_source_wdm(
        jgb=jgb,
        band=band,
        f0_jitter_width=injection.f0_jitter_width,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
        seed=mcmc_seed,
    )
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    if n_divergences > 0:
        print(f"WARNING: {n_divergences} divergences")

    raw_samples = mcmc.get_samples()
    samples = {
        "delta_logf0": np.asarray(raw_samples["delta_logf0"]),
        "logfdot": np.asarray(raw_samples["logfdot"]),
        "logA": np.asarray(raw_samples["logA"]),
        "phi0": np.asarray(raw_samples["phi0"]),
        "f0": np.asarray(raw_samples["f0"]),
        "fdot": np.asarray(raw_samples["fdot"]),
        "A": np.asarray(raw_samples["A"]),
    }
    samples_base = np.column_stack([samples["f0"], samples["fdot"], samples["A"], samples["phi0"]])
    samples_full = build_sampled_source_params(source_param, samples_base)
    snr_samples = compute_snr_samples(
        jgb=jgb,
        band=band,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
        samples_full=samples_full,
    )
    samples_to_save = np.column_stack([samples_base, snr_samples])
    truth = source_truth_vector(source_param, snr=truth_snr_wdm)

    wdm_mean = np.mean(samples_to_save, axis=0)
    truth_wdm_loglike = evaluate_wdm_loglike(
        jgb=jgb,
        params=source_param,
        band=band,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
    )
    wdm_mean_params = build_sampled_source_params(source_param, wdm_mean[None, :4])[0]
    wdm_mean_loglike = evaluate_wdm_loglike(
        jgb=jgb,
        params=wdm_mean_params,
        band=band,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
    )

    freq_mean = None
    if FREQ_POSTERIOR_PATH.exists():
        freq_mean = np.mean(load_posterior_samples_source(FREQ_POSTERIOR_PATH), axis=0)

    print_cross_domain_diagnostics(
        labels=["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"],
        truth=truth,
        wdm_mean=wdm_mean,
        freq_mean=freq_mean,
        truth_snr_frequency=truth_snr_frequency,
        truth_snr_wdm=truth_snr_wdm,
        wdm_loglike_truth=truth_wdm_loglike,
        wdm_loglike_posterior_mean=wdm_mean_loglike,
    )

    output_path = save_posterior_archive(
        WDM_POSTERIOR_PATH,
        source_params=source_params,
        all_samples=[samples_to_save],
        snr_optimal=[truth_snr_wdm],
        labels=["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"],
        truth=truth,
    )
    print(f"Saved WDM posteriors to {output_path}")

    save_sampler_diagnostics(
        diagnostics_path=diagnostics_path,
        injection_seed=injection.seed,
        mcmc_seed=mcmc_seed,
        nt=NT,
        band=band,
        samples=samples,
        init_values=init_values,
        n_divergences=n_divergences,
    )
    print(f"Saved WDM sampler diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
