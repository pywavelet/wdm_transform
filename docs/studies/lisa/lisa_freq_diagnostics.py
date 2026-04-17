"""Standalone frequency-domain likelihood diagnostics for the LISA GB study."""

from __future__ import annotations

import argparse

import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
from jaxgb.jaxgb import JaxGB
from numpy.fft import rfft, rfftfreq

from lisa_common import RUN_DIR, interp_psd_channels, load_injection
from lisa_freq_mcmc import (
    INJECTION_PATH,
    build_band,
    compute_truth_snr,
    setup_jax_and_matplotlib,
    source_aet_band,
)

setup_jax_and_matplotlib()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument(
        "--skip-2d-scans",
        action="store_true",
        help="Generate only the prior/data diagnostic plot and 1D slices.",
    )
    return parser.parse_args()


def make_diagnostic_plots(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    source_param: np.ndarray,
    snr_optimal: float,
) -> None:
    """Plot data, truth, and a cloud of prior draws on the local band."""
    h_true_aet = np.asarray(
        source_aet_band(
            jgb, source_param, int(band["band_kmin"]), int(band["band_kmax"])
        )
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

        axes[channel, 0].semilogy(
            freqs, power_data, "k.", markersize=2, alpha=0.5, label="Data"
        )
        axes[channel, 0].semilogy(
            freqs, power_truth, "r-", linewidth=1.5, label="Truth"
        )
        axes[channel, 0].set_ylabel(f"{name} power")
        axes[channel, 0].grid(True, alpha=0.3)
        axes[channel, 0].legend()

        for _ in range(100):
            params = fixed_params.copy()
            params[0] = float(
                float(band["f0_ref"])
                + rng.uniform(
                    float(np.asarray(band["delta_f0_bounds"])[0]),
                    float(np.asarray(band["delta_f0_bounds"])[1]),
                )
            )
            params[1] = float(np.exp(rng.normal(prior_center[1], prior_scale[1])))
            params[2] = float(np.exp(rng.normal(prior_center[2], prior_scale[2])))
            params[7] = float(rng.uniform(-np.pi, np.pi))
            h_draw = np.asarray(
                source_aet_band(
                    jgb, params, int(band["band_kmin"]), int(band["band_kmax"])
                )
            )
            axes[channel, 1].semilogy(
                freqs, np.abs(h_draw[channel]) ** 2, "b-", alpha=0.1
            )
        axes[channel, 1].semilogy(freqs, power_truth, "r-", linewidth=2, label="Truth")
        axes[channel, 1].set_ylabel(f"{name} power")
        axes[channel, 1].set_title(f"{name}: prior draws")
        axes[channel, 1].grid(True, alpha=0.3)
        axes[channel, 1].legend()

    fig.suptitle(
        f"LISA GB Diagnostic: SNR={snr_optimal:.1f}, T_obs={float(band['t_obs']) / 86400.0:.1f}d, "
        "delta_f0 prior="
        f"±{0.5 * (float(np.asarray(band['delta_f0_bounds'])[1]) - float(np.asarray(band['delta_f0_bounds'])[0])):.2e} Hz",
        fontsize=12,
    )
    fig.tight_layout()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUN_DIR / "diagnostic_data_signal_priors.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def evaluate_frequency_loglike(
    *,
    jgb: JaxGB,
    params: np.ndarray,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    dt: float,
    df: float,
) -> float:
    """Evaluate the local Whittle log-likelihood for one parameter vector."""
    h_aet = np.asarray(
        source_aet_band(jgb, params, int(band["band_kmin"]), int(band["band_kmax"]))
    )
    residual = dt * (np.asarray(band["data_aet"]) - h_aet)
    noise_psd = np.asarray(band["noise_psd_aet"])
    return float(
        -np.sum(np.log(noise_psd) + 2.0 * df * np.abs(residual) ** 2 / noise_psd)
    )


def make_likelihood_scan_plots(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    source_param: np.ndarray,
    dt: float,
    df: float,
    grid_size: int,
) -> None:
    """Generate 2D local likelihood scans around the injected source."""
    grid_size = max(grid_size, 16)
    delta_f0_bounds = tuple(np.asarray(band["delta_f0_bounds"], dtype=float))
    delta_f0_grid = np.linspace(delta_f0_bounds[0], delta_f0_bounds[1], grid_size)
    logA_grid = np.linspace(
        float(np.asarray(band["logA_bounds"])[0]),
        float(np.asarray(band["logA_bounds"])[1]),
        grid_size,
    )
    phi0_grid = np.linspace(-np.pi, np.pi, grid_size)

    scan_specs = [
        (
            "delta_f0_vs_phi0",
            delta_f0_grid,
            phi0_grid,
            "delta_f0",
            "phi0",
            "delta_f0 [Hz]",
            "phi0 [rad]",
        ),
        (
            "logA_vs_phi0",
            logA_grid,
            phi0_grid,
            "logA",
            "phi0",
            "log(A)",
            "phi0 [rad]",
        ),
        (
            "delta_f0_vs_logA",
            delta_f0_grid,
            logA_grid,
            "delta_f0",
            "logA",
            "delta_f0 [Hz]",
            "log(A)",
        ),
    ]

    truth_values = {
        "delta_f0": float(source_param[0] - float(band["f0_ref"])),
        "logA": float(np.log(source_param[2])),
        "phi0": float(source_param[7]),
    }

    fixed_truth = source_param.copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (stem, x_grid, y_grid, x_name, y_name, x_label, y_label) in zip(
        axes, scan_specs
    ):
        surface = np.empty((len(y_grid), len(x_grid)), dtype=float)
        for row_idx, y_val in enumerate(y_grid):
            for col_idx, x_val in enumerate(x_grid):
                params = fixed_truth.copy()
                params[0] = float(band["f0_ref"]) + (
                    x_val if x_name == "delta_f0" else truth_values["delta_f0"]
                )
                params[2] = float(
                    np.exp(
                        x_val
                        if x_name == "logA"
                        else y_val if y_name == "logA" else truth_values["logA"]
                    )
                )
                params[7] = float(
                    x_val
                    if x_name == "phi0"
                    else y_val if y_name == "phi0" else truth_values["phi0"]
                )
                surface[row_idx, col_idx] = evaluate_frequency_loglike(
                    jgb=jgb,
                    params=params,
                    band=band,
                    dt=dt,
                    df=df,
                )

        surface -= np.max(surface)
        image = ax.imshow(
            surface,
            extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax.axvline(truth_values[x_name], color="white", linestyle="--", linewidth=1.5)
        ax.axhline(truth_values[y_name], color="white", linestyle="--", linewidth=1.5)
        ax.set_title(stem.replace("_", " "))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.colorbar(image, ax=ax, label="Δ log-likelihood")

        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        image_single = ax_single.imshow(
            surface,
            extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax_single.axvline(
            truth_values[x_name], color="white", linestyle="--", linewidth=1.5
        )
        ax_single.axhline(
            truth_values[y_name], color="white", linestyle="--", linewidth=1.5
        )
        ax_single.set_title(stem.replace("_", " "))
        ax_single.set_xlabel(x_label)
        ax_single.set_ylabel(y_label)
        fig_single.colorbar(image_single, ax=ax_single, label="Δ log-likelihood")
        fig_single.tight_layout()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        fig_single.savefig(
            RUN_DIR / f"freq_likelihood_scan_{stem}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_single)

    fig.suptitle("Frequency-domain local likelihood scans", fontsize=14)
    fig.tight_layout()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUN_DIR / "freq_likelihood_scans.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_likelihood_slice_plots(
    *,
    jgb: JaxGB,
    band: dict[str, np.ndarray | float | tuple[float, float]],
    source_param: np.ndarray,
    dt: float,
    df: float,
    grid_size: int,
) -> None:
    """Generate 1D likelihood slices through the injected source."""
    grid_size = max(grid_size, 64)
    truth_values = {
        "delta_f0": float(source_param[0] - float(band["f0_ref"])),
        "logA": float(np.log(source_param[2])),
        "phi0": float(source_param[7]),
    }
    param_specs = [
        (
            "delta_f0",
            np.linspace(*tuple(np.asarray(band["delta_f0_bounds"], dtype=float)), grid_size),
            "delta_f0 [Hz]",
        ),
        (
            "logA",
            np.linspace(*tuple(np.asarray(band["logA_bounds"], dtype=float)), grid_size),
            "log(A)",
        ),
        ("phi0", np.linspace(-np.pi, np.pi, grid_size), "phi0 [rad]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fixed_truth = source_param.copy()

    for ax, (name, grid, label) in zip(axes, param_specs):
        values = np.empty_like(grid)
        for idx, value in enumerate(grid):
            params = fixed_truth.copy()
            if name == "delta_f0":
                params[0] = float(band["f0_ref"]) + float(value)
            elif name == "logA":
                params[2] = float(np.exp(value))
            else:
                params[7] = float(value)
            values[idx] = evaluate_frequency_loglike(
                jgb=jgb,
                params=params,
                band=band,
                dt=dt,
                df=df,
            )

        values -= np.max(values)
        ax.plot(grid, values, color="C0", linewidth=2.0)
        ax.axvline(truth_values[name], color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name} slice")
        ax.set_xlabel(label)
        ax.set_ylabel("Δ log-likelihood")
        ax.grid(True, alpha=0.3)

        fig_single, ax_single = plt.subplots(figsize=(6, 4.5))
        ax_single.plot(grid, values, color="C0", linewidth=2.0)
        ax_single.axvline(
            truth_values[name], color="black", linestyle="--", linewidth=1.5
        )
        ax_single.set_title(f"{name} slice")
        ax_single.set_xlabel(label)
        ax_single.set_ylabel("Δ log-likelihood")
        ax_single.grid(True, alpha=0.3)
        fig_single.tight_layout()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        fig_single.savefig(
            RUN_DIR / f"freq_likelihood_slice_{name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_single)

    fig.suptitle("Frequency-domain 1D likelihood slices", fontsize=14)
    fig.tight_layout()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUN_DIR / "freq_likelihood_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    injection = load_injection(INJECTION_PATH)
    source_param = injection.source_params[0].copy()
    dt = injection.dt
    t_obs = injection.t_obs
    df = 1.0 / t_obs

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
    snr_optimal = compute_truth_snr(
        jgb=jgb,
        source_param=source_param,
        band=band,
        dt=dt,
    )

    make_diagnostic_plots(jgb=jgb, band=band, source_param=source_param, snr_optimal=snr_optimal)
    if not args.skip_2d_scans:
        make_likelihood_scan_plots(
            jgb=jgb,
            band=band,
            source_param=source_param,
            dt=dt,
            df=df,
            grid_size=args.grid_size,
        )
    make_likelihood_slice_plots(
        jgb=jgb,
        band=band,
        source_param=source_param,
        dt=dt,
        df=df,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
