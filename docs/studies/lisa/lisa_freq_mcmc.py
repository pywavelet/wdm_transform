"""Frequency-domain LISA GB inference with NumPyro."""

from __future__ import annotations

import atexit
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
SHOW_PROGRESS = os.getenv("LISA_PROGRESS_BAR", "1").strip().lower() in {"1", "true", "yes", "on"}
NUTS_TARGET_ACCEPT = float(os.getenv("LISA_NUTS_TARGET_ACCEPT", "0.85"))
NUTS_MAX_TREE_DEPTH = int(os.getenv("LISA_NUTS_MAX_TREE_DEPTH", "10"))
NUTS_DENSE_MASS = os.getenv("LISA_NUTS_DENSE_MASS", "1").strip().lower() in {"1", "true", "yes", "on"}
INIT_JITTER_SCALE = float(os.getenv("LISA_INIT_JITTER_SCALE", "0.15"))


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[lisa_freq_mcmc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)


def source_aet_band(jgb: JaxGB, params: jnp.ndarray, kmin: int, kmax: int) -> jnp.ndarray:
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
        "logf0_ref": float(np.log(f0_ref)),
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


def make_diagnostic_plots(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    source_param: np.ndarray,
    f0_jitter_width: float,
    snr_optimal: float,
) -> None:
    """Plot data, truth, and a cloud of prior draws on the local band."""
    h_true_aet = np.asarray(
        source_aet_band(jgb, source_param, int(band["band_kmin"]), int(band["band_kmax"]))
    )
    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    prior_center = np.asarray(band["prior_center"], dtype=float)
    prior_scale = np.asarray(band["prior_scale"], dtype=float)
    freqs = np.asarray(band["freqs"])
    data_aet = np.asarray(band["data_aet"])

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for channel, name in enumerate(("A", "E", "T")):
        power_data = np.abs(data_aet[channel]) ** 2
        power_truth = np.abs(h_true_aet[channel]) ** 2

        axes[channel, 0].semilogy(freqs, power_data, "k.", markersize=2, alpha=0.5, label="Data")
        axes[channel, 0].semilogy(freqs, power_truth, "r-", linewidth=1.5, label="Truth")
        axes[channel, 0].set_ylabel(f"{name} power")
        axes[channel, 0].grid(True, alpha=0.3)
        axes[channel, 0].legend()

        for _ in range(100):
            params = fixed_params.copy()
            params[0] = float(np.exp(np.log(float(band["f0_ref"])) + rng.uniform(-f0_jitter_width, f0_jitter_width)))
            params[1] = float(np.exp(rng.normal(prior_center[1], prior_scale[1])))
            params[2] = float(np.exp(rng.normal(prior_center[2], prior_scale[2])))
            params[7] = float(rng.uniform(-np.pi, np.pi))
            h_draw = np.asarray(
                source_aet_band(jgb, params, int(band["band_kmin"]), int(band["band_kmax"]))
            )
            axes[channel, 1].semilogy(freqs, np.abs(h_draw[channel]) ** 2, "b-", alpha=0.1)
        axes[channel, 1].semilogy(freqs, power_truth, "r-", linewidth=2, label="Truth")
        axes[channel, 1].set_ylabel(f"{name} power")
        axes[channel, 1].set_title(f"{name}: prior draws")
        axes[channel, 1].grid(True, alpha=0.3)
        axes[channel, 1].legend()

    fig.suptitle(
        f"LISA GB Diagnostic: SNR={snr_optimal:.1f}, T_obs={float(band['t_obs']) / 86400.0:.1f}d, "
        f"f0_jitter=±{f0_jitter_width:.2e}",
        fontsize=12,
    )
    plt.tight_layout()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RUN_DIR / "diagnostic_data_signal_priors.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def sample_source(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
    df: float,
    f0_jitter_width: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], MCMC, dict[str, float]]:
    """Run NUTS for the local frequency-domain model."""
    data_aet_j = jnp.asarray(band["data_aet"], dtype=jnp.complex128)
    psd_aet_j = jnp.asarray(band["noise_psd_aet"], dtype=jnp.float64)
    fixed_params_j = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    prior_center = jnp.asarray(band["prior_center"], dtype=jnp.float64)
    prior_scale = jnp.asarray(band["prior_scale"], dtype=jnp.float64)
    logfdot_bounds = tuple(np.asarray(band["logfdot_bounds"], dtype=float))
    logA_bounds = tuple(np.asarray(band["logA_bounds"], dtype=float))
    logf0_ref = float(band["logf0_ref"])
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)

    init_eps = 1e-6
    rng = np.random.default_rng(seed + 101)
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
        params = (
            fixed_params_j
            .at[0].set(jnp.exp(logf0_ref + delta_logf0))
            .at[1].set(jnp.exp(logfdot))
            .at[2].set(jnp.exp(logA))
            .at[7].set(phi0)
        )
        h_aet = source_aet_band(jgb, params, int(band["band_kmin"]), int(band["band_kmax"]))
        residual = dt_j * (data_aet_j - h_aet)
        numpyro.factor(
            "whittle",
            -jnp.sum(jnp.log(psd_aet_j) + 2.0 * df_j * jnp.abs(residual) ** 2 / psd_aet_j),
        )
        numpyro.deterministic("f0", jnp.exp(logf0_ref + delta_logf0))
        numpyro.deterministic("fdot", jnp.exp(logfdot))
        numpyro.deterministic("A", jnp.exp(logA))

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
        progress_bar=SHOW_PROGRESS,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    raw = mcmc.get_samples()
    samples = {
        "delta_logf0": np.asarray(raw["delta_logf0"]),
        "logfdot": np.asarray(raw["logfdot"]),
        "logA": np.asarray(raw["logA"]),
        "phi0": np.asarray(raw["phi0"]),
        "f0": np.asarray(raw["f0"]),
        "fdot": np.asarray(raw["fdot"]),
        "A": np.asarray(raw["A"]),
    }
    return samples, mcmc, init_values


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
            h_aet = source_aet_band(jgb, source_params, int(band["band_kmin"]), int(band["band_kmax"]))
            return jnp.sqrt(4.0 * df_j * jnp.sum(jnp.abs(dt_j * h_aet) ** 2 / psd_aet_j))

        return jax.vmap(single_snr)(params)

    return np.asarray(get_snrs(jnp.asarray(samples_full, dtype=jnp.float64)))


def save_sampler_diagnostics(
    *,
    diagnostics_path,
    injection_seed: int,
    mcmc_seed: int,
    logf0_ref: float,
    samples: dict[str, np.ndarray],
    init_values: dict[str, float],
    n_divergences: int,
) -> None:
    """Write a compact JSON summary of the sampler diagnostics."""
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
        "divergences": n_divergences,
        "init_values": {
            **{key: float(value) for key, value in init_values.items()},
            "f0": float(np.exp(logf0_ref + init_values["delta_logf0"])),
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
    dt = injection.dt
    t_obs = injection.t_obs
    df = 1.0 / t_obs
    diagnostics_path = RUN_DIR / "freq_sampler_diagnostics.json"

    data_aet = np.stack([injection.data_At, injection.data_Et, injection.data_Tt], axis=0)
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

    print(f"T_obs={t_obs / 86400.0:.1f}d, N={data_aet.shape[1]}, seed={injection.seed}, MCMC_seed={mcmc_seed}")

    snr_optimal = compute_truth_snr(jgb=jgb, source_param=source_param, band=band, dt=dt)
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
        data_aet_f[0, int(band["band_kmin"]):int(band["band_kmax"])],
        data_aet_f[1, int(band["band_kmin"]):int(band["band_kmax"])],
        data_aet_f[2, int(band["band_kmin"]):int(band["band_kmax"])],
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
            injection.source_Af[int(band["band_kmin"]):int(band["band_kmax"])],
            injection.source_Ef[int(band["band_kmin"]):int(band["band_kmax"])],
            injection.source_Tf[int(band["band_kmin"]):int(band["band_kmax"])],
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
        print("Skipping pure-source overlap check: injection archive does not include source_Af/source_Ef/source_Tf")

    if not overlap_check_passed:
        print("WARNING: Template-injection overlap check FAILED")
        print("Consider investigating template generation or injection consistency")

    make_diagnostic_plots(
        jgb=jgb,
        band=band,
        source_param=source_param,
        f0_jitter_width=injection.f0_jitter_width,
        snr_optimal=snr_optimal,
    )

    samples, mcmc, init_values = sample_source(
        jgb=jgb,
        band=band,
        dt=dt,
        df=df,
        f0_jitter_width=injection.f0_jitter_width,
        seed=mcmc_seed,
    )
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    if n_divergences > 0:
        print(f"WARNING: {n_divergences} divergences")

    samples_report = np.column_stack([samples["f0"], samples["fdot"], samples["A"], samples["phi0"]])
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
        logf0_ref=float(band["logf0_ref"]),
        samples=samples,
        init_values=init_values,
        n_divergences=n_divergences,
    )
    print(f"Saved frequency sampler diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
