"""Toy PE for a chirping monochromatic signal in white noise.

This is a deliberately small analogue of the LISA frequency-vs-WDM study.  It
generates one real time series

    h(t) = A sin(2 pi (f t + 0.5 fdot t^2) + phi)

in white Gaussian noise, then samples the four parameters ``(A, f, fdot, phi)``
with two likelihoods:

* a full FFT-domain Gaussian likelihood, using Parseval's theorem for white
  noise;
* a diagonal WDM-domain Gaussian likelihood, with per-coefficient white-noise
  variances calibrated by Monte Carlo through the same WDM transform.

Both domains use NumPyro/NUTS for posterior sampling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wdm_transform import TimeSeries
from wdm_transform.transforms import from_time_to_wdm

STUDY_DIR = Path(__file__).resolve().parent
OUTDIR = STUDY_DIR / "outdir_monochrome_white_noise_pe"


@dataclass(frozen=True)
class ToyConfig:
    n: int = 1024
    nt: int = 32
    dt: float = 0.125
    sigma: float = 0.7
    seed: int = 11
    n_wdm_noise: int = 128
    warmup: int = 500
    draws: int = 1000
    num_chains: int = 1


TRUTH = {
    "A": 0.55,
    "f": 0.70,
    "fdot": 4.0e-4,
    "phi": 0.6,
}

PRIOR_BOUNDS = {
    "A": (0.05, 1.25),
    "f": (0.66, 0.74),
    "fdot": (-2.0e-4, 8.0e-4),
    "phi": (-np.pi, np.pi),
}

PARAM_NAMES = ("A", "f", "fdot", "phi")
LABELS = (r"$A$", r"$f$", r"$\dot f$", r"$\phi$")


def chirping_sinusoid(theta: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return ``A sin(2 pi (f t + 0.5 fdot t^2) + phi)``."""
    amp, freq, fdot, phase = theta
    phase_t = 2.0 * np.pi * (freq * times + 0.5 * fdot * times**2) + phase
    return amp * np.sin(phase_t)


def wrap_phase(phi: float | np.ndarray) -> float | np.ndarray:
    """Wrap phase to ``[-pi, pi)``."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def theta_from_mapping(values: dict[str, float]) -> np.ndarray:
    return np.array([values[name] for name in PARAM_NAMES], dtype=float)


def make_data(config: ToyConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    times = np.arange(config.n, dtype=float) * config.dt
    truth = theta_from_mapping(TRUTH)
    signal = chirping_sinusoid(truth, times)
    noise = rng.normal(0.0, config.sigma, size=config.n)
    return {
        "times": times,
        "truth": truth,
        "signal": signal,
        "noise": noise,
        "data": signal + noise,
    }


def to_wdm_array(values: np.ndarray, config: ToyConfig) -> np.ndarray:
    return np.asarray(TimeSeries(values, dt=config.dt).to_wdm(nt=config.nt).coeffs[0])


def calibrate_wdm_variance(config: ToyConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 100_000)
    coeffs = np.stack(
        [
            to_wdm_array(rng.normal(0.0, config.sigma, size=config.n), config)
            for _ in range(config.n_wdm_noise)
        ]
    )
    return np.maximum(coeffs.var(axis=0), 1e-10)


def run_numpyro_sampler(
    domain: str,
    *,
    data_fft: np.ndarray,
    data_wdm: np.ndarray,
    wdm_variance: np.ndarray,
    times: np.ndarray,
    start: np.ndarray,
    config: ToyConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(config.num_chains)

    times_j = jnp.asarray(times, dtype=jnp.float64)
    data_fft_j = jnp.asarray(data_fft, dtype=jnp.complex128)
    data_wdm_j = jnp.asarray(data_wdm, dtype=jnp.float64)
    wdm_variance_j = jnp.asarray(wdm_variance, dtype=jnp.float64)
    fft_norm = float(config.n * config.sigma**2)
    wdm_log_norm = jnp.log(2.0 * jnp.pi * wdm_variance_j)
    nf = config.n // config.nt

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(*PRIOR_BOUNDS["A"]))
        freq = numpyro.sample("f", dist.Uniform(*PRIOR_BOUNDS["f"]))
        fdot = numpyro.sample("fdot", dist.Uniform(*PRIOR_BOUNDS["fdot"]))
        phase = numpyro.sample("phi", dist.Uniform(*PRIOR_BOUNDS["phi"]))
        phase_t = 2.0 * jnp.pi * (freq * times_j + 0.5 * fdot * times_j**2) + phase
        signal = amp * jnp.sin(phase_t)

        if domain == "freq":
            residual = data_fft_j - jnp.fft.fft(signal)
            loglike = -0.5 * jnp.sum(jnp.abs(residual) ** 2) / fft_norm
        elif domain == "wdm":
            model_wdm = from_time_to_wdm(
                signal,
                nt=config.nt,
                nf=nf,
                a=1.0 / 3.0,
                d=1.0,
                dt=config.dt,
                backend="jax",
            )
            residual = data_wdm_j - model_wdm
            loglike = -0.5 * jnp.sum(wdm_log_norm + residual**2 / wdm_variance_j)
        else:
            raise ValueError(f"Unknown domain {domain!r}.")
        numpyro.factor(f"{domain}_loglike", loglike)

    init_values = {
        name: float(value) for name, value in zip(PARAM_NAMES, start, strict=True)
    }
    kernel = NUTS(
        model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=config.warmup,
        num_samples=config.draws,
        num_chains=config.num_chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "accept_prob"))
    samples = mcmc.get_samples()
    packed = np.column_stack(
        [np.asarray(samples[name]).reshape(-1) for name in PARAM_NAMES]
    )
    extra = mcmc.get_extra_fields()
    diagnostics = {
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
    }
    return packed, diagnostics


def summarize(samples: np.ndarray) -> dict[str, dict[str, float]]:
    summary = {}
    for i, name in enumerate(PARAM_NAMES):
        values = samples[:, i]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return summary


def plot_data_views(
    data: dict[str, np.ndarray],
    data_wdm: np.ndarray,
    config: ToyConfig,
    output_dir: Path,
) -> None:
    freqs = np.fft.fftfreq(config.n, d=config.dt)
    positive = freqs >= 0.0
    data_fft = np.fft.fft(data["data"])
    signal_fft = np.fft.fft(data["signal"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(data["times"], data["data"], color="0.55", lw=0.8, label="data")
    axes[0].plot(
        data["times"],
        data["signal"],
        color="tab:blue",
        lw=1.5,
        label="signal",
    )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Strain")
    axes[0].legend(frameon=False)

    axes[1].plot(
        freqs[positive],
        np.abs(data_fft[positive]),
        color="0.55",
        lw=0.8,
        label="data",
    )
    axes[1].plot(
        freqs[positive],
        np.abs(signal_fft[positive]),
        color="tab:blue",
        lw=1.5,
        label="signal",
    )
    axes[1].set_xlim(0.55, 0.85)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|FFT|")
    axes[1].legend(frameon=False)

    im = axes[2].imshow(
        data_wdm.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[2].set_xlabel("WDM time bin")
    axes[2].set_ylabel("WDM frequency channel")
    fig.colorbar(im, ax=axes[2], label="Coefficient")
    fig.tight_layout()
    fig.savefig(output_dir / "data_views.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pair_comparison(
    freq_samples: np.ndarray,
    wdm_samples: np.ndarray,
    truth: np.ndarray,
    output_dir: Path,
) -> None:
    n_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                ax.hist(
                    freq_samples[:, col],
                    bins=36,
                    density=True,
                    histtype="step",
                    color="tab:blue",
                )
                ax.hist(
                    wdm_samples[:, col],
                    bins=36,
                    density=True,
                    histtype="step",
                    color="tab:orange",
                )
                ax.axvline(truth[col], color="black", lw=1.0)
            elif row > col:
                ax.scatter(
                    freq_samples[:, col],
                    freq_samples[:, row],
                    s=2,
                    alpha=0.12,
                    color="tab:blue",
                )
                ax.scatter(
                    wdm_samples[:, col],
                    wdm_samples[:, row],
                    s=2,
                    alpha=0.12,
                    color="tab:orange",
                )
                ax.axvline(truth[col], color="black", lw=0.8)
                ax.axhline(truth[row], color="black", lw=0.8)
            else:
                ax.axis("off")
                continue

            if row == n_params - 1:
                ax.set_xlabel(LABELS[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(LABELS[row])
            elif row != col:
                ax.set_yticklabels([])

    axes[0, 0].legend(["FFT", "WDM", "truth"], frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(config: ToyConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_data(config)
    truth = data["truth"]
    data_fft = np.fft.fft(data["data"])
    data_wdm = to_wdm_array(data["data"], config)
    wdm_variance = calibrate_wdm_variance(config)

    snr = float(np.linalg.norm(data["signal"]) / config.sigma)
    print(f"Output directory: {output_dir}")
    print(
        f"Truth: A={truth[0]:.4f}, f={truth[1]:.6f}, "
        f"fdot={truth[2]:.3e}, phi={truth[3]:.4f}"
    )
    print(f"White-noise sigma={config.sigma:.3f}; matched-filter norm SNR={snr:.3f}")
    print(f"WDM data shape: {data_wdm.shape}")
    print(f"Median calibrated WDM noise variance: {np.median(wdm_variance):.4e}")

    start = truth + np.array([0.025, -5.0e-4, 1.0e-5, 0.10])
    start[-1] = wrap_phase(start[-1])

    freq_samples, freq_diagnostics = run_numpyro_sampler(
        "freq",
        data_fft=data_fft,
        data_wdm=data_wdm,
        wdm_variance=wdm_variance,
        times=data["times"],
        start=start,
        config=config,
        seed=config.seed + 1,
    )
    wdm_samples, wdm_diagnostics = run_numpyro_sampler(
        "wdm",
        data_fft=data_fft,
        data_wdm=data_wdm,
        wdm_variance=wdm_variance,
        times=data["times"],
        start=start,
        config=config,
        seed=config.seed + 2,
    )

    freq_summary = summarize(freq_samples)
    wdm_summary = summarize(wdm_samples)
    print(
        f"FFT mean accept prob: {freq_diagnostics['mean_accept_prob']:.3f}; "
        f"divergences={freq_diagnostics['divergences']}"
    )
    print(
        f"WDM mean accept prob: {wdm_diagnostics['mean_accept_prob']:.3f}; "
        f"divergences={wdm_diagnostics['divergences']}"
    )
    print("Posterior mean +/- std")
    for name in PARAM_NAMES:
        fsum = freq_summary[name]
        wsum = wdm_summary[name]
        print(
            f"  {name:4s} FFT={fsum['mean']:.6g} +/- {fsum['std']:.3g}; "
            f"WDM={wsum['mean']:.6g} +/- {wsum['std']:.3g}; "
            f"truth={TRUTH[name]:.6g}"
        )

    np.savez(
        output_dir / "posterior_samples.npz",
        freq=freq_samples,
        wdm=wdm_samples,
        truth=truth,
        param_names=np.asarray(PARAM_NAMES),
    )
    summary = {
        "config": asdict(config),
        "truth": {
            name: float(value) for name, value in zip(PARAM_NAMES, truth, strict=True)
        },
        "snr": snr,
        "diagnostics": {"freq": freq_diagnostics, "wdm": wdm_diagnostics},
        "posterior": {"freq": freq_summary, "wdm": wdm_summary},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_data_views(data, data_wdm, config, output_dir)
    plot_pair_comparison(freq_samples, wdm_samples, truth, output_dir)
    print(f"Saved posterior samples and figures under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--draws", type=int, default=ToyConfig.draws)
    parser.add_argument("--warmup", type=int, default=ToyConfig.warmup)
    parser.add_argument("--num-chains", type=int, default=ToyConfig.num_chains)
    parser.add_argument("--n-wdm-noise", type=int, default=ToyConfig.n_wdm_noise)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a shorter run for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ToyConfig(
        seed=args.seed,
        draws=200 if args.quick else args.draws,
        warmup=100 if args.quick else args.warmup,
        num_chains=args.num_chains,
        n_wdm_noise=32 if args.quick else args.n_wdm_noise,
    )
    run(config, args.outdir)


if __name__ == "__main__":
    main()
