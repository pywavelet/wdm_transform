"""Frequency-domain LISA GB inference with NumPyro."""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from gb_prior import build_local_prior_info
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    FREQ_POSTERIOR_PATH,
    INJECTION_PATH,
    RUN_DIR,
    build_sampled_source_params,
    check_template_injection_sanity,
    interp_psd_channels,
    load_injection,
    save_posterior_archive,
    setup_jax_and_matplotlib,
    source_truth_vector,
)
from numpy.fft import rfft, rfftfreq
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform.signal_processing import matched_filter_snr_rfft

setup_jax_and_matplotlib()
jax.config.update("jax_enable_x64", True)

_SCRIPT_START = time.perf_counter()
N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
NUM_CHAINS = int(os.getenv("LISA_NUM_CHAINS", "2"))
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

if NUM_CHAINS > 1:
    numpyro.set_host_device_count(NUM_CHAINS)


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[lisa_freq_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


def source_aet_band(
    jgb: JaxGB, params: jnp.ndarray, kmin: int, kmax: int
) -> jnp.ndarray:
    """Return A/E/T frequency modes on the requested local band."""
    a_mode, e_mode, t_mode = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=kmin,
        kmax=kmax,
        tdi_combination="AET",
        tdi_generation=1.5,
    )
    return jnp.stack(
        [
            jnp.asarray(a_mode, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(e_mode, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(t_mode, dtype=jnp.complex128).reshape(-1),
        ],
        axis=0,
    )


def build_band(
    *,
    source_param: np.ndarray,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
    f0_ref: float,
    freqs: np.ndarray,
    data_aet_f: np.ndarray,
    noise_psd_full_aet: np.ndarray,
    t_obs: float,
    jgb: JaxGB,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Return the local frequency band and prior metadata."""
    prior_info = build_local_prior_info(
        prior_f0=prior_f0,
        prior_fdot=prior_fdot,
        prior_A=prior_A,
    )
    kmin = max(int(np.floor(prior_f0[0] * t_obs)) - jgb.n, 0)
    kmax = min(int(np.ceil(prior_f0[1] * t_obs)) + jgb.n + 1, len(freqs))
    return {
        "freqs": freqs[kmin:kmax],
        "data_aet": data_aet_f[:, kmin:kmax],
        "noise_psd_aet": noise_psd_full_aet[:, kmin:kmax].copy(),
        "fixed_params": source_param.copy(),
        "t_obs": t_obs,
        "band_kmin": kmin,
        "band_kmax": kmax,
        "f0_ref": float(f0_ref),
        "prior_f0": tuple(float(x) for x in prior_f0),
        "delta_f0_bounds": (
            float(prior_f0[0] - f0_ref),
            float(prior_f0[1] - f0_ref),
        ),
        "prior_center": prior_info.prior_center,
        "prior_scale": prior_info.prior_scale,
        "logfdot_bounds": prior_info.logfdot_bounds,
        "logA_bounds": prior_info.logA_bounds,
    }


def compute_truth_snr(
    *,
    jgb: JaxGB,
    source_param: np.ndarray,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
) -> float:
    """Compute the injected-source A+E+T matched-filter SNR on the local band."""
    h_aet = source_aet_band(
        jgb,
        source_param,
        int(band["band_kmin"]),
        int(band["band_kmax"]),
    )
    return float(
        np.linalg.norm(
            [
                matched_filter_snr_rfft(
                    np.asarray(h_aet[channel]),
                    np.asarray(band["noise_psd_aet"])[channel],
                    np.asarray(band["freqs"]),
                    dt=dt,
                )
                for channel in range(3)
            ]
        )
    )


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
    f0_ref: float,
    delta_f0_bounds: tuple[float, float],
    logfdot_bounds: tuple[float, float],
    logA_bounds: tuple[float, float],
    seed: int,
) -> dict[str, float]:
    """Build a single truth-centered initialization for single-chain runs."""
    rng = np.random.default_rng(seed + 101)
    delta_f0_eps = min(1e-12, 1e-3 * (delta_f0_bounds[1] - delta_f0_bounds[0]))
    log_eps = 1e-6
    delta_f0_center = np.clip(
        fixed_params[0] - f0_ref,
        delta_f0_bounds[0] + delta_f0_eps,
        delta_f0_bounds[1] - delta_f0_eps,
    )
    return {
        "delta_f0": float(delta_f0_center),
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


def build_phase_basis_aet(
    *,
    jgb: JaxGB,
    fixed_params_j: jnp.ndarray,
    f0: jnp.ndarray,
    fdot: jnp.ndarray,
    amplitude_ref: jnp.ndarray,
    band: dict[str, np.ndarray | float | tuple[float, float]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return unit-amplitude cosine/sine phase basis templates on the local band."""
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
    h_cos = source_aet_band(
        jgb, params_cos, int(band["band_kmin"]), int(band["band_kmax"])
    )
    h_sin = source_aet_band(
        jgb, params_sin, int(band["band_kmin"]), int(band["band_kmax"])
    )
    return h_cos, h_sin


def profile_amplitude_phase(
    *,
    data_aet_j: jnp.ndarray,
    noise_psd_aet_j: jnp.ndarray,
    h_cos: jnp.ndarray,
    h_sin: jnp.ndarray,
    amplitude: jnp.ndarray,
    amplitude_ref: jnp.ndarray,
    dt_j: jnp.ndarray,
    df_j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Profile over phase for fixed f0, fdot, and amplitude."""
    weight = 2.0 * df_j * dt_j**2 / noise_psd_aet_j

    def inner(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(jnp.sum(weight * jnp.conj(lhs) * rhs))

    scale = amplitude / jnp.maximum(amplitude_ref, 1e-30)
    h_cos_scaled = scale * h_cos
    h_sin_scaled = scale * h_sin
    proj_cos = inner(h_cos_scaled, data_aet_j)
    proj_sin = inner(h_sin_scaled, data_aet_j)
    phi0 = jnp.arctan2(proj_sin, proj_cos)
    template = jnp.cos(phi0) * h_cos_scaled + jnp.sin(phi0) * h_sin_scaled
    residual = dt_j * (data_aet_j - template)
    loglike = -jnp.sum(
        jnp.log(noise_psd_aet_j)
        + 2.0 * df_j * jnp.abs(residual) ** 2 / noise_psd_aet_j
    )
    return loglike, amplitude, phi0, template


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
    for name in ("delta_f0", "logfdot", "logA"):
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
    f0_ref: float,
    snr_optimal: float,
    output_prefix: str = "freq",
) -> None:
    """Generate trace plots for delta_f0, phi0, and logA by chain."""
    params_to_plot = ["delta_f0", "phi0", "logA"]

    # Calculate true values for reference lines
    true_values = {
        "delta_f0": source_param[0] - f0_ref,
        "phi0": source_param[7],
        "logA": np.log(source_param[2]),
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes = axes.flatten()

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

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
                label=f"Chain {chain_idx + 1}",
                linewidth=0.8,
            )

        # Add truth line
        ax.axhline(
            true_values[param],
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Truth",
        )

        # Special handling for phi0 to show wrapping
        if param == "phi0":
            ax.axhline(-np.pi, color="gray", linestyle=":", alpha=0.5, label="±π")
            ax.axhline(np.pi, color="gray", linestyle=":", alpha=0.5)
            ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
            ax.set_ylabel("phi0 [rad]")
        elif param == "delta_f0":
            ax.set_ylabel("δf0 [Hz]")
        elif param == "logA":
            ax.set_ylabel("log(A)")

        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add summary statistics
        all_samples = samples.reshape(-1)
        mean_val = np.mean(all_samples)
        std_val = np.std(all_samples, ddof=1)

        # Calculate chain statistics
        chain_means = np.mean(samples, axis=1)
        chain_stds = np.std(samples, axis=1, ddof=1)
        between_chain_var = np.var(chain_means, ddof=1)

        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nTruth: {true_values[param]:.4f}\nBetween-chain var: {between_chain_var:.6f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

    fig.suptitle(
        f"{output_prefix.upper()} MCMC Trace Plots (SNR={snr_optimal:.1f}, {n_chains} chains, {n_samples} samples)",
        fontsize=14,
    )
    plt.tight_layout()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        RUN_DIR / f"{output_prefix}_trace_plots.png", dpi=150, bbox_inches="tight"
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
            print(
                f"    Chain {chain_idx + 1}: mean={chain_means[chain_idx]:.6f}, std={chain_stds[chain_idx]:.6f}, bias={bias:.6f}"
            )

        # Check for potential mode separation
        if len(chain_means) > 1:
            max_separation = np.max(chain_means) - np.min(chain_means)
            typical_std = np.mean(chain_stds)
            if max_separation > 3 * typical_std:
                print(
                    f"    WARNING: Large chain separation ({max_separation:.6f}) vs typical std ({typical_std:.6f})"
                )

            # Special check for phi0 wrapping
            if param == "phi0":
                # Check if chains are on opposite sides of the ±π boundary
                chain_angles = np.array(chain_means)
                if np.any(chain_angles > 2.0) and np.any(chain_angles < -2.0):
                    print(
                        f"    WARNING: phi0 chains may be separated by periodic boundary"
                    )
        print()


def sample_source(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
    df: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], MCMC, dict[str, float] | None]:
    """Run NUTS for the local frequency-domain model with profiled phase."""
    data_aet_j = jnp.asarray(band["data_aet"], dtype=jnp.complex128)
    psd_aet_j = jnp.asarray(band["noise_psd_aet"], dtype=jnp.float64)
    fixed_params_j = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    prior_center = jnp.asarray(band["prior_center"], dtype=jnp.float64)
    prior_scale = jnp.asarray(band["prior_scale"], dtype=jnp.float64)
    delta_f0_bounds = tuple(np.asarray(band["delta_f0_bounds"], dtype=float))
    logfdot_bounds = tuple(np.asarray(band["logfdot_bounds"], dtype=float))
    logA_bounds = tuple(np.asarray(band["logA_bounds"], dtype=float))
    f0_ref = float(band["f0_ref"])
    amplitude_ref_j = fixed_params_j[2]
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)

    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    init_values = _build_init_values(
        fixed_params=fixed_params,
        f0_ref=f0_ref,
        delta_f0_bounds=delta_f0_bounds,
        logfdot_bounds=logfdot_bounds,
        logA_bounds=logA_bounds,
        seed=seed,
    )

    def model() -> None:
        delta_f0 = numpyro.sample(
            "delta_f0", dist.Uniform(delta_f0_bounds[0], delta_f0_bounds[1])
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
        f0 = f0_ref + delta_f0
        fdot = jnp.exp(logfdot)
        amplitude = jnp.exp(logA)
        h_cos, h_sin = build_phase_basis_aet(
            jgb=jgb,
            fixed_params_j=fixed_params_j,
            f0=f0,
            fdot=fdot,
            amplitude_ref=amplitude_ref_j,
            band=band,
        )
        loglike, amplitude, phi0, _ = profile_amplitude_phase(
            data_aet_j=data_aet_j,
            noise_psd_aet_j=psd_aet_j,
            h_cos=h_cos,
            h_sin=h_sin,
            amplitude=amplitude,
            amplitude_ref=amplitude_ref_j,
            dt_j=dt_j,
            df_j=df_j,
        )
        numpyro.factor("whittle_profiled", loglike)
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", amplitude)
        numpyro.deterministic("phi0", phi0)

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
                f0_ref=f0_ref,
                delta_f0_bounds=delta_f0_bounds,
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
    samples_by_chain = {
        key: np.asarray(value)
        for key, value in mcmc.get_samples(group_by_chain=True).items()
    }
    samples = _flatten_chain_samples(samples_by_chain)
    return samples, samples_by_chain, mcmc, init_values


def compute_snr_samples(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
    df: float,
    samples_full: np.ndarray,
) -> np.ndarray:
    """Evaluate the optimal frequency-domain SNR for each posterior sample."""
    psd_aet_j = jnp.asarray(band["noise_psd_aet"], dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)

    @jax.jit
    def get_snrs(params: jnp.ndarray) -> jnp.ndarray:
        def single_snr(source_params: jnp.ndarray) -> jnp.ndarray:
            h_aet = source_aet_band(
                jgb, source_params, int(band["band_kmin"]), int(band["band_kmax"])
            )
            return jnp.sqrt(
                4.0 * df_j * jnp.sum(jnp.abs(dt_j * h_aet) ** 2 / psd_aet_j)
            )

        return jax.vmap(single_snr)(params)

    return np.asarray(get_snrs(jnp.asarray(samples_full, dtype=jnp.float64)))


def save_sampler_diagnostics(
    *,
    diagnostics_path,
    injection_seed: int,
    mcmc_seed: int,
    f0_ref: float,
    samples: dict[str, np.ndarray],
    samples_by_chain: dict[str, np.ndarray],
    init_values: dict[str, float] | None,
    n_divergences: int,
    snr_reference: float,
) -> None:
    """Write a compact JSON summary of the sampler diagnostics."""
    latent_summary = numpyro_summary(
        {
            "delta_f0": samples_by_chain["delta_f0"],
            "logfdot": samples_by_chain["logfdot"],
            "logA": samples_by_chain["logA"],
            "phi0": samples_by_chain["phi0"],
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
        "divergences": n_divergences,
        "init_values": (
            {
                **{key: float(value) for key, value in init_values.items()},
                "f0": float(f0_ref + init_values["delta_f0"]),
                "fdot": float(np.exp(init_values["logfdot"])),
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
    dt = injection.dt
    t_obs = injection.t_obs
    df = 1.0 / t_obs
    diagnostics_path = RUN_DIR / "freq_sampler_diagnostics.json"

    data_aet = np.stack(
        [injection.data_At, injection.data_Et, injection.data_Tt], axis=0
    )
    freqs = rfftfreq(data_aet.shape[1], dt)
    data_aet_f = rfft(data_aet, axis=1)
    noise_psd_full_aet = interp_psd_channels(
        freqs,
        injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
    band = build_band(
        source_param=source_param,
        prior_f0=injection.prior_f0,
        prior_fdot=injection.prior_fdot,
        prior_A=injection.prior_A,
        f0_ref=injection.f0_ref,
        freqs=freqs,
        data_aet_f=data_aet_f,
        noise_psd_full_aet=noise_psd_full_aet,
        t_obs=t_obs,
        jgb=jgb,
    )

    print(
        f"T_obs={t_obs / 86400.0:.1f}d, N={data_aet.shape[1]}, seed={injection.seed}, MCMC_seed={mcmc_seed}"
    )

    snr_optimal = compute_truth_snr(
        jgb=jgb, source_param=source_param, band=band, dt=dt
    )
    print(f"Frequency band SNR={snr_optimal:.1f}")

    # ── Template-injection overlap sanity check ──────────────────────────────────
    # Generate template at true parameters and compare with injection A/E/T
    h_true_aet = source_aet_band(
        jgb,
        source_param,
        int(band["band_kmin"]),
        int(band["band_kmax"]),
    )
    injection_aet_band = (
        data_aet_f[0, int(band["band_kmin"]) : int(band["band_kmax"])],
        data_aet_f[1, int(band["band_kmin"]) : int(band["band_kmax"])],
        data_aet_f[2, int(band["band_kmin"]) : int(band["band_kmax"])],
    )
    template_aet_band = (
        np.asarray(h_true_aet[0]),
        np.asarray(h_true_aet[1]),
        np.asarray(h_true_aet[2]),
    )
    noise_psd_band = (
        np.asarray(band["noise_psd_aet"])[0],
        np.asarray(band["noise_psd_aet"])[1],
        np.asarray(band["noise_psd_aet"])[2],
    )

    overlap_check_passed = check_template_injection_sanity(
        template_aet_band,
        injection_aet_band,
        noise_psd_band,
        np.asarray(band["freqs"]),
        dt,
        context="Frequency-domain template-injection",
    )

    if (
        injection.source_Af is not None
        and injection.source_Ef is not None
        and injection.source_Tf is not None
    ):
        pure_aet_band = (
            injection.source_Af[int(band["band_kmin"]) : int(band["band_kmax"])],
            injection.source_Ef[int(band["band_kmin"]) : int(band["band_kmax"])],
            injection.source_Tf[int(band["band_kmin"]) : int(band["band_kmax"])],
        )
        check_template_injection_sanity(
            template_aet_band,
            pure_aet_band,
            noise_psd_band,
            np.asarray(band["freqs"]),
            dt,
            context="Template vs pure source (no noise)",
        )
    else:
        print(
            "Skipping pure-source overlap check: injection archive does not include source_Af/source_Ef/source_Tf"
        )

    if not overlap_check_passed:
        print("WARNING: Template-injection overlap check FAILED")
        print("Consider investigating template generation or injection consistency")

    samples, samples_by_chain, mcmc, init_values = sample_source(
        jgb=jgb,
        band=band,
        dt=dt,
        df=df,
        seed=mcmc_seed,
    )
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    if n_divergences > 0:
        print(f"WARNING: {n_divergences} divergences")
    latent_summary = numpyro_summary(
        {
            "delta_f0": samples_by_chain["delta_f0"],
            "logfdot": samples_by_chain["logfdot"],
            "logA": samples_by_chain["logA"],
            "phi0": samples_by_chain["phi0"],
        },
        group_by_chain=True,
    )
    sampler_report = build_sampler_diagnostic_report(
        latent_summary=latent_summary,
        loga_samples=samples["logA"],
        snr_reference=snr_optimal,
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
        f0_ref=float(band["f0_ref"]),
        snr_optimal=snr_optimal,
        output_prefix="freq",
    )

    samples_report = np.column_stack(
        [samples["f0"], samples["fdot"], samples["A"], samples["phi0"]]
    )
    samples_full = build_sampled_source_params(source_param, samples_report)
    snr_samples = compute_snr_samples(
        jgb=jgb,
        band=band,
        dt=dt,
        df=df,
        samples_full=samples_full,
    )
    samples_to_save = np.column_stack([samples_report, snr_samples])
    truth = source_truth_vector(source_param, snr=snr_optimal)

    output_path = save_posterior_archive(
        FREQ_POSTERIOR_PATH,
        source_params=source_params,
        all_samples=[samples_to_save],
        snr_optimal=[snr_optimal],
        labels=["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"],
        truth=truth,
    )
    print(f"Saved frequency posteriors to {output_path}")

    save_sampler_diagnostics(
        diagnostics_path=diagnostics_path,
        injection_seed=injection.seed,
        mcmc_seed=mcmc_seed,
        f0_ref=float(band["f0_ref"]),
        samples=samples,
        samples_by_chain=samples_by_chain,
        init_values=init_values,
        n_divergences=n_divergences,
        snr_reference=snr_optimal,
    )
    print(f"Saved frequency sampler diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _print_runtime()
