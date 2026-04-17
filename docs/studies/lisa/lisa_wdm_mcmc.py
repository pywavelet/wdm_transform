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
    check_template_injection_sanity,
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
NUM_CHAINS = int(os.getenv("LISA_NUM_CHAINS", "4"))
NT = int(os.getenv("LISA_NT", "32"))
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NUTS_TARGET_ACCEPT = float(os.getenv("LISA_NUTS_TARGET_ACCEPT", "0.85"))
NUTS_MAX_TREE_DEPTH = int(os.getenv("LISA_NUTS_MAX_TREE_DEPTH", "10"))
NUTS_DENSE_MASS = os.getenv("LISA_NUTS_DENSE_MASS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INIT_JITTER_SCALE = float(os.getenv("LISA_INIT_JITTER_SCALE", "0.15"))
MIN_EFFECTIVE_SAMPLES = float(os.getenv("LISA_MIN_EFFECTIVE_SAMPLES", "50"))
R_HAT_ALERT_THRESHOLD = float(os.getenv("LISA_R_HAT_ALERT_THRESHOLD", "1.1"))
LOGA_WIDTH_STUCK_RATIO = float(os.getenv("LISA_LOGA_WIDTH_STUCK_RATIO", "0.5"))
A_WDM = 1.0 / 3.0
D_WDM = 1.0

if NUM_CHAINS > 1:
    numpyro.set_host_device_count(NUM_CHAINS)


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


def _flatten_chain_samples(
    samples_by_chain: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Collapse chain and draw dimensions for downstream reporting."""
    return {
        name: np.asarray(values).reshape((-1,) + np.asarray(values).shape[2:])
        for name, values in samples_by_chain.items()
    }


def _build_init_values(
    *,
    fixed_params: np.ndarray,
    logf0_ref: float,
    delta_logf0_bounds: tuple[float, float],
    logfdot_bounds: tuple[float, float],
    logA_bounds: tuple[float, float],
    seed: int,
) -> dict[str, float]:
    """Build a single truth-centered initialization for single-chain runs."""
    rng = np.random.default_rng(seed + 17)
    delta_logf0_eps = min(
        1e-12, 1e-3 * (delta_logf0_bounds[1] - delta_logf0_bounds[0])
    )
    log_eps = 1e-6
    delta_logf0_center = np.clip(
        np.log(fixed_params[0]) - logf0_ref,
        delta_logf0_bounds[0] + delta_logf0_eps,
        delta_logf0_bounds[1] - delta_logf0_eps,
    )
    return {
        "delta_logf0": float(delta_logf0_center),
        "logfdot": float(
            np.clip(
                np.log(fixed_params[1]) + INIT_JITTER_SCALE * rng.standard_normal(),
                logfdot_bounds[0] + log_eps,
                logfdot_bounds[1] - log_eps,
            )
        ),
        "logA": float(
            np.clip(
                np.log(fixed_params[2]) + INIT_JITTER_SCALE * rng.standard_normal(),
                logA_bounds[0] + log_eps,
                logA_bounds[1] - log_eps,
            )
        ),
    }


def build_phase_basis_wdm(
    *,
    jgb: JaxGB,
    fixed_params_j: jnp.ndarray,
    f0: jnp.ndarray,
    fdot: jnp.ndarray,
    amplitude_ref: jnp.ndarray,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Return WDM cosine/sine phase bases for fixed f0/fdot and reference amplitude."""
    params_cos = (
        fixed_params_j.at[0]
        .set(f0)
        .at[1]
        .set(fdot)
        .at[2]
        .set(amplitude_ref)
        .at[7]
        .set(0.0)
    )
    params_sin = (
        fixed_params_j.at[0]
        .set(f0)
        .at[1]
        .set(fdot)
        .at[2]
        .set(amplitude_ref)
        .at[7]
        .set(0.5 * jnp.pi)
    )
    h_cos = local_rfft_to_wdm(
        generate_aet_rfft(jgb, params_cos, int(band["src_kmin"]), int(band["src_kmax"])),
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
    h_sin = local_rfft_to_wdm(
        generate_aet_rfft(jgb, params_sin, int(band["src_kmin"]), int(band["src_kmax"])),
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
    return h_cos, h_sin


def profile_phase_wdm(
    *,
    data_a_j: jnp.ndarray,
    data_e_j: jnp.ndarray,
    data_t_j: jnp.ndarray,
    noise_var_a_j: jnp.ndarray,
    noise_var_e_j: jnp.ndarray,
    noise_var_t_j: jnp.ndarray,
    h_cos: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_sin: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    amplitude: jnp.ndarray,
    amplitude_ref: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Profile over phase for fixed f0, fdot, and amplitude in WDM space."""
    scale = amplitude / jnp.maximum(amplitude_ref, 1e-30)
    h_cos_scaled = tuple(scale * component for component in h_cos)
    h_sin_scaled = tuple(scale * component for component in h_sin)

    proj_cos = (
        jnp.sum(h_cos_scaled[0] * data_a_j / noise_var_a_j)
        + jnp.sum(h_cos_scaled[1] * data_e_j / noise_var_e_j)
        + jnp.sum(h_cos_scaled[2] * data_t_j / noise_var_t_j)
    )
    proj_sin = (
        jnp.sum(h_sin_scaled[0] * data_a_j / noise_var_a_j)
        + jnp.sum(h_sin_scaled[1] * data_e_j / noise_var_e_j)
        + jnp.sum(h_sin_scaled[2] * data_t_j / noise_var_t_j)
    )
    phi0 = jnp.arctan2(proj_sin, proj_cos)
    h_a = jnp.cos(phi0) * h_cos_scaled[0] + jnp.sin(phi0) * h_sin_scaled[0]
    h_e = jnp.cos(phi0) * h_cos_scaled[1] + jnp.sin(phi0) * h_sin_scaled[1]
    h_t = jnp.cos(phi0) * h_cos_scaled[2] + jnp.sin(phi0) * h_sin_scaled[2]
    loglike = (
        -0.5
        * jnp.sum((data_a_j - h_a) ** 2 / noise_var_a_j + jnp.log(2.0 * jnp.pi * noise_var_a_j))
        - 0.5
        * jnp.sum((data_e_j - h_e) ** 2 / noise_var_e_j + jnp.log(2.0 * jnp.pi * noise_var_e_j))
        - 0.5
        * jnp.sum((data_t_j - h_t) ** 2 / noise_var_t_j + jnp.log(2.0 * jnp.pi * noise_var_t_j))
    )
    return loglike, phi0


def build_sampler_diagnostic_report(
    *,
    latent_summary: dict[str, dict[str, np.ndarray]],
    loga_samples: np.ndarray,
    snr_reference: float,
) -> dict[str, float | bool | list[str]]:
    """Build scalar convergence checks for console output and JSON diagnostics."""
    loga_std = float(np.std(np.asarray(loga_samples).reshape(-1), ddof=1))
    fisher_loga_std = float(1.0 / max(snr_reference, 1e-12))
    loga_std_ratio = (
        float(loga_std / fisher_loga_std) if fisher_loga_std > 0.0 else float("nan")
    )
    min_n_eff = min(
        float(np.min(np.asarray(stats["n_eff"]))) for stats in latent_summary.values()
    )
    max_r_hat = max(
        float(np.max(np.asarray(stats["r_hat"]))) for stats in latent_summary.values()
    )

    warnings: list[str] = []
    suspicious_loga_width = bool(loga_std_ratio < LOGA_WIDTH_STUCK_RATIO)
    if suspicious_loga_width:
        warnings.append(
            "logA posterior std is much smaller than the Fisher estimate; chains may be stuck."
        )
    if min_n_eff < MIN_EFFECTIVE_SAMPLES:
        warnings.append(
            f"Minimum n_eff={min_n_eff:.1f} is below {MIN_EFFECTIVE_SAMPLES:.0f}; mixing is poor."
        )
        if max_r_hat <= 1.01:
            warnings.append(
                "r_hat is near 1 while n_eff is tiny; chains may be stuck near isolated points."
            )
    if max_r_hat > R_HAT_ALERT_THRESHOLD:
        warnings.append(
            f"Maximum r_hat={max_r_hat:.3f} exceeds {R_HAT_ALERT_THRESHOLD:.2f}; chains disagree."
        )

    return {
        "snr_reference": float(snr_reference),
        "fisher_logA_std_estimate": fisher_loga_std,
        "posterior_logA_std": loga_std,
        "logA_std_to_fisher_ratio": loga_std_ratio,
        "suspicious_logA_width": suspicious_loga_width,
        "min_n_eff": min_n_eff,
        "max_r_hat": max_r_hat,
        "warnings": warnings,
    }


def print_sampler_diagnostic_report(
    *,
    title: str,
    latent_summary: dict[str, dict[str, np.ndarray]],
    report: dict[str, float | bool | list[str]],
) -> None:
    """Print the key chain-mixing diagnostics relevant to stuck-at-truth failures."""
    print(f"\n{title}:")
    print(
        "  "
        f"logA std={float(report['posterior_logA_std']):.4f} vs Fisher≈1/SNR="
        f"{float(report['fisher_logA_std_estimate']):.4f} "
        f"(ratio={float(report['logA_std_to_fisher_ratio']):.2f}, SNR={float(report['snr_reference']):.1f})"
    )
    for name in ("delta_logf0", "logfdot", "logA"):
        stats = latent_summary[name]
        print(
            f"  {name:<12} n_eff={float(np.asarray(stats['n_eff'])):8.1f} "
            f"r_hat={float(np.asarray(stats['r_hat'])):6.3f}"
        )
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")


def make_trace_plots(
    *,
    samples_by_chain: dict[str, np.ndarray],
    source_param: np.ndarray,
    logf0_ref: float,
    snr_optimal: float,
    output_prefix: str = "wdm",
) -> None:
    """Generate trace plots for delta_logf0, phi0, and logA by chain."""
    import matplotlib.pyplot as plt

    params_to_plot = ["delta_logf0", "phi0", "logA"]

    # Calculate true values for reference lines
    true_values = {
        "delta_logf0": np.log(source_param[0]) - logf0_ref,
        "phi0": source_param[7],
        "logA": np.log(source_param[2]),
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        samples = np.asarray(samples_by_chain[param])
        n_chains, n_samples = samples.shape

        # Plot each chain
        for chain_idx in range(n_chains):
            ax.plot(
                samples[chain_idx],
                color=colors[chain_idx % len(colors)],
                alpha=0.7,
                label=f'Chain {chain_idx + 1}',
                linewidth=0.8
            )

        # Add truth line
        ax.axhline(
            true_values[param],
            color='black',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label='Truth'
        )

        # Special handling for phi0 to show wrapping
        if param == "phi0":
            ax.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5, label='±π')
            ax.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
            ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
            ax.set_ylabel('phi0 [rad]')
        elif param == "delta_logf0":
            ax.set_ylabel('δlog(f0)')
        elif param == "logA":
            ax.set_ylabel('log(A)')

        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add summary statistics
        all_samples = samples.reshape(-1)
        mean_val = np.mean(all_samples)
        std_val = np.std(all_samples, ddof=1)

        # Calculate chain statistics
        chain_means = np.mean(samples, axis=1)
        chain_stds = np.std(samples, axis=1, ddof=1)
        between_chain_var = np.var(chain_means, ddof=1)

        ax.text(
            0.02, 0.98,
            f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nTruth: {true_values[param]:.4f}\nBetween-chain var: {between_chain_var:.6f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9
        )

    fig.suptitle(
        f'{output_prefix.upper()} MCMC Trace Plots (SNR={snr_optimal:.1f}, {n_chains} chains, {n_samples} samples)',
        fontsize=14
    )
    plt.tight_layout()

    # Import RUN_DIR from lisa_common
    from lisa_common import RUN_DIR
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        RUN_DIR / f"{output_prefix}_trace_plots.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Print chain-specific diagnostic summary
    print(f"\n{output_prefix.upper()} Chain-specific diagnostics:")
    for param in params_to_plot:
        samples = np.asarray(samples_by_chain[param])
        chain_means = np.mean(samples, axis=1)
        chain_stds = np.std(samples, axis=1, ddof=1)
        truth = true_values[param]

        print(f"  {param}:")
        print(f"    Truth: {truth:.6f}")
        for chain_idx in range(len(chain_means)):
            bias = chain_means[chain_idx] - truth
            print(f"    Chain {chain_idx + 1}: mean={chain_means[chain_idx]:.6f}, std={chain_stds[chain_idx]:.6f}, bias={bias:.6f}")

        # Check for potential mode separation
        if len(chain_means) > 1:
            max_separation = np.max(chain_means) - np.min(chain_means)
            typical_std = np.mean(chain_stds)
            if max_separation > 3 * typical_std:
                print(f"    WARNING: Large chain separation ({max_separation:.6f}) vs typical std ({typical_std:.6f})")

            # Special check for phi0 wrapping
            if param == "phi0":
                # Check if chains are on opposite sides of the ±π boundary
                chain_angles = np.array(chain_means)
                if np.any(chain_angles > 2.0) and np.any(chain_angles < -2.0):
                    print(f"    WARNING: phi0 chains may be separated by periodic boundary")
        print()


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
    band_freqs = freq_grid[band_slice.start : band_slice.stop]
    noise_psd_band_a = np.maximum(
        np.interp(
            band_freqs,
            injection.freqs,
            injection.noise_psd_A,
            left=injection.noise_psd_A[0],
            right=injection.noise_psd_A[-1],
        ),
        1e-60,
    )
    noise_psd_band_e = np.maximum(
        np.interp(
            band_freqs,
            injection.freqs,
            injection.noise_psd_E,
            left=injection.noise_psd_E[0],
            right=injection.noise_psd_E[-1],
        ),
        1e-60,
    )
    noise_psd_band_t = np.maximum(
        np.interp(
            band_freqs,
            injection.freqs,
            injection.noise_psd_T,
            left=injection.noise_psd_T[0],
            right=injection.noise_psd_T[-1],
        ),
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
        "data_band_A": np.asarray(
            forward_wdm_band(
                jnp.asarray(data_a_rfft[kmin_rfft:kmax_rfft]), **band_kwargs
            )
        ),
        "data_band_E": np.asarray(
            forward_wdm_band(
                jnp.asarray(data_e_rfft[kmin_rfft:kmax_rfft]), **band_kwargs
            )
        ),
        "data_band_T": np.asarray(
            forward_wdm_band(
                jnp.asarray(data_t_rfft[kmin_rfft:kmax_rfft]), **band_kwargs
            )
        ),
        "noise_var_band_A": wdm_noise_variance(
            noise_psd_band_a, nt=nt, dt=injection.dt
        ),
        "noise_var_band_E": wdm_noise_variance(
            noise_psd_band_e, nt=nt, dt=injection.dt
        ),
        "noise_var_band_T": wdm_noise_variance(
            noise_psd_band_t, nt=nt, dt=injection.dt
        ),
        "f0_ref": float(injection.f0_ref),
        "f0_init": float(f0_init),
        "logf0_ref": float(np.log(f0_init)),
        "delta_logf0_bounds": (
            float(prior_info.logf0_bounds[0] - np.log(f0_init)),
            float(prior_info.logf0_bounds[1] - np.log(f0_init)),
        ),
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
                matched_filter_snr_wdm(
                    np.asarray(h_a_wdm), np.asarray(band["noise_var_band_A"])
                ),
                matched_filter_snr_wdm(
                    np.asarray(h_e_wdm), np.asarray(band["noise_var_band_E"])
                ),
                matched_filter_snr_wdm(
                    np.asarray(h_t_wdm), np.asarray(band["noise_var_band_T"])
                ),
            ]
        )
    )
    band_freqs_rfft = rfft_freqs[int(band["src_kmin"]) : int(band["src_kmax"])]
    snr_freq = float(
        np.linalg.norm(
            [
                matched_filter_snr_rfft(
                    np.asarray(a_loc),
                    noise_psd_a_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
                    band_freqs_rfft,
                    dt=dt,
                ),
                matched_filter_snr_rfft(
                    np.asarray(e_loc),
                    noise_psd_e_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
                    band_freqs_rfft,
                    dt=dt,
                ),
                matched_filter_snr_rfft(
                    np.asarray(t_loc),
                    noise_psd_t_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
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
            np.sum(
                diff_a**2 / np.asarray(band["noise_var_band_A"])
                + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_A"]))
            )
            + np.sum(
                diff_e**2 / np.asarray(band["noise_var_band_E"])
                + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_E"]))
            )
            + np.sum(
                diff_t**2 / np.asarray(band["noise_var_band_T"])
                + np.log(2.0 * np.pi * np.asarray(band["noise_var_band_T"]))
            )
        )
    )


def sample_source_wdm(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    nt: int,
    nf: int,
    n_freqs: int,
    df_rfft: float,
    seed: int,
) -> tuple[MCMC, dict[str, float] | None]:
    """Run NUTS for the local WDM-domain model with profiled phase."""
    data_a_j = jnp.asarray(band["data_band_A"], dtype=jnp.float64)
    data_e_j = jnp.asarray(band["data_band_E"], dtype=jnp.float64)
    data_t_j = jnp.asarray(band["data_band_T"], dtype=jnp.float64)
    noise_var_a_j = jnp.asarray(band["noise_var_band_A"], dtype=jnp.float64)
    noise_var_e_j = jnp.asarray(band["noise_var_band_E"], dtype=jnp.float64)
    noise_var_t_j = jnp.asarray(band["noise_var_band_T"], dtype=jnp.float64)
    fixed_params_j = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    prior_center = jnp.asarray(band["prior_center"], dtype=jnp.float64)
    prior_scale = jnp.asarray(band["prior_scale"], dtype=jnp.float64)
    delta_logf0_bounds = tuple(np.asarray(band["delta_logf0_bounds"], dtype=float))
    logfdot_bounds = tuple(np.asarray(band["logfdot_bounds"], dtype=float))
    logA_bounds = tuple(np.asarray(band["logA_bounds"], dtype=float))
    logf0_ref = float(band["logf0_ref"])
    amplitude_ref_j = fixed_params_j[2]

    def model() -> None:
        delta_logf0 = numpyro.sample(
            "delta_logf0",
            dist.Uniform(delta_logf0_bounds[0], delta_logf0_bounds[1]),
        )
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
        f0 = jnp.exp(logf0_ref + delta_logf0)
        fdot = jnp.exp(logfdot)
        amplitude = jnp.exp(logA)
        h_cos, h_sin = build_phase_basis_wdm(
            jgb=jgb,
            fixed_params_j=fixed_params_j,
            f0=f0,
            fdot=fdot,
            amplitude_ref=amplitude_ref_j,
            band=band,
            nt=nt,
            nf=nf,
            n_freqs=n_freqs,
            df_rfft=df_rfft,
        )
        loglike, phi0 = profile_phase_wdm(
            data_a_j=data_a_j,
            data_e_j=data_e_j,
            data_t_j=data_t_j,
            noise_var_a_j=noise_var_a_j,
            noise_var_e_j=noise_var_e_j,
            noise_var_t_j=noise_var_t_j,
            h_cos=h_cos,
            h_sin=h_sin,
            amplitude=amplitude,
            amplitude_ref=amplitude_ref_j,
        )
        numpyro.factor("ll_profiled", loglike)
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", amplitude)
        numpyro.deterministic("phi0", phi0)

    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    init_values = _build_init_values(
        fixed_params=fixed_params,
        logf0_ref=logf0_ref,
        delta_logf0_bounds=delta_logf0_bounds,
        logfdot_bounds=logfdot_bounds,
        logA_bounds=logA_bounds,
        seed=seed,
    )
    nuts_kwargs = {
        "dense_mass": NUTS_DENSE_MASS,
        "target_accept_prob": NUTS_TARGET_ACCEPT,
        "max_tree_depth": NUTS_MAX_TREE_DEPTH,
    }
    if NUM_CHAINS == 1:
        nuts_kwargs["init_strategy"] = init_to_value(values=init_values)
    mcmc = MCMC(
        NUTS(
            model,
            **nuts_kwargs,
        ),
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=NUM_CHAINS,
        progress_bar=SHOW_PROGRESS,
    )
    if NUM_CHAINS > 1:
        per_chain_inits = [
            _build_init_values(
                fixed_params=fixed_params,
                logf0_ref=logf0_ref,
                delta_logf0_bounds=delta_logf0_bounds,
                logfdot_bounds=logfdot_bounds,
                logA_bounds=logA_bounds,
                seed=seed + chain_idx * 7,
            )
            for chain_idx in range(NUM_CHAINS)
        ]
        init_params = {
            key: jnp.array([iv[key] for iv in per_chain_inits])
            for key in per_chain_inits[0]
        }
        init_values = per_chain_inits[0]
        mcmc.run(jax.random.PRNGKey(seed), init_params=init_params, extra_fields=("diverging",))
    else:
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
                generate_aet_rfft(
                    jgb, source_params, int(band["src_kmin"]), int(band["src_kmax"])
                ),
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
    samples_by_chain: dict[str, np.ndarray],
    init_values: dict[str, float] | None,
    n_divergences: int,
    snr_reference: float,
) -> None:
    """Write a compact JSON summary of the WDM sampler diagnostics."""
    latent_summary = numpyro_summary(
        {
            "delta_logf0": samples_by_chain["delta_logf0"],
            "logfdot": samples_by_chain["logfdot"],
            "logA": samples_by_chain["logA"],
        },
        group_by_chain=True,
    )
    sampler_report = build_sampler_diagnostic_report(
        latent_summary=latent_summary,
        loga_samples=samples["logA"],
        snr_reference=snr_reference,
    )
    payload = {
        "seed": injection_seed,
        "mcmc_seed": mcmc_seed,
        "num_warmup": N_WARMUP,
        "num_draws": N_DRAWS,
        "num_chains": NUM_CHAINS,
        "nt": nt,
        "divergences": n_divergences,
        "init_values": (
            {
                **{key: float(value) for key, value in init_values.items()},
                "f0": float(
                    np.exp(float(band["logf0_ref"]) + init_values["delta_logf0"])
                ),
                "fdot": float(np.exp(init_values["logfdot"])),
                "A": float(np.exp(init_values["logA"])),
            }
            if init_values is not None
            else None
        ),
        "init_strategy": (
            "manual_truth_centered"
            if init_values is not None
            else "numpyro_random_per_chain"
        ),
        "latent_diagnostics": {
            key: {
                "n_eff": float(np.asarray(stats["n_eff"])),
                "r_hat": float(np.asarray(stats["r_hat"])),
            }
            for key, stats in latent_summary.items()
        },
        "sampler_report": sampler_report,
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
    t_obs = n_keep * injection.dt
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

    # ── Template-injection overlap sanity check in frequency domain ──────────────
    # Generate template and extract injection on the source frequency band for comparison
    h_true_aet_freq = generate_aet_rfft(
        jgb,
        source_param,
        int(band["src_kmin"]),
        int(band["src_kmax"]),
    )
    injection_aet_freq = (
        data_a_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
        data_e_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
        data_t_rfft[int(band["src_kmin"]) : int(band["src_kmax"])],
    )
    template_aet_freq = (
        np.asarray(h_true_aet_freq[0]),
        np.asarray(h_true_aet_freq[1]),
        np.asarray(h_true_aet_freq[2]),
    )
    band_freqs_rfft = rfft_freqs[int(band["src_kmin"]) : int(band["src_kmax"])]
    noise_psd_freq_band = (
        noise_psd_rfft[0][int(band["src_kmin"]) : int(band["src_kmax"])],
        noise_psd_rfft[1][int(band["src_kmin"]) : int(band["src_kmax"])],
        noise_psd_rfft[2][int(band["src_kmin"]) : int(band["src_kmax"])],
    )

    overlap_check_passed = check_template_injection_sanity(
        template_aet_freq,
        injection_aet_freq,
        noise_psd_freq_band,
        band_freqs_rfft,
        injection.dt,
        context="WDM analysis template-injection (freq domain)",
    )

    if (
        injection.source_Af is not None
        and injection.source_Ef is not None
        and injection.source_Tf is not None
    ):
        pure_aet_freq = (
            injection.source_Af[int(band["src_kmin"]) : int(band["src_kmax"])],
            injection.source_Ef[int(band["src_kmin"]) : int(band["src_kmax"])],
            injection.source_Tf[int(band["src_kmin"]) : int(band["src_kmax"])],
        )
        check_template_injection_sanity(
            template_aet_freq,
            pure_aet_freq,
            noise_psd_freq_band,
            band_freqs_rfft,
            injection.dt,
            context="Template vs pure source (no noise)",
        )
    else:
        print(
            "Skipping pure-source overlap check: injection archive does not include source_Af/source_Ef/source_Tf"
        )

    if not overlap_check_passed:
        print("WARNING: Template-injection overlap check FAILED")
        print("Consider investigating template generation or injection consistency")

    mcmc, init_values = sample_source_wdm(
        jgb=jgb,
        band=band,
        nt=NT,
        nf=nf,
        n_freqs=n_freqs,
        df_rfft=df_rfft,
        seed=mcmc_seed,
    )
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    if n_divergences > 0:
        print(f"WARNING: {n_divergences} divergences")

    samples_by_chain = {
        key: np.asarray(value)
        for key, value in mcmc.get_samples(group_by_chain=True).items()
    }
    samples = _flatten_chain_samples(samples_by_chain)
    latent_summary = numpyro_summary(
        {
            "delta_logf0": samples_by_chain["delta_logf0"],
            "logfdot": samples_by_chain["logfdot"],
            "logA": samples_by_chain["logA"],
        },
        group_by_chain=True,
    )
    sampler_report = build_sampler_diagnostic_report(
        latent_summary=latent_summary,
        loga_samples=samples["logA"],
        snr_reference=truth_snr_wdm,
    )
    print_sampler_diagnostic_report(
        title="Sampler diagnostics",
        latent_summary=latent_summary,
        report=sampler_report,
    )

    # Generate trace plots for key parameters
    make_trace_plots(
        samples_by_chain=samples_by_chain,
        source_param=source_param,
        logf0_ref=float(band["logf0_ref"]),
        snr_optimal=truth_snr_wdm,
        output_prefix="wdm",
    )

    samples_base = np.column_stack(
        [samples["f0"], samples["fdot"], samples["A"], samples["phi0"]]
    )
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
        samples_by_chain=samples_by_chain,
        init_values=init_values,
        n_divergences=n_divergences,
        snr_reference=truth_snr_wdm,
    )
    print(f"Saved WDM sampler diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
