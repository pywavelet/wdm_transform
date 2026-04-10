# ruff: noqa: B018, E402, I001

import atexit
import builtins
from pathlib import Path
import subprocess
import sys


import corner
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from matplotlib.lines import Line2D
from numpyro.infer import MCMC, NUTS, init_to_value

from wdm_transform import TimeSeries, get_backend
from wdm_transform.plotting import plot_spectrogram
from wdm_transform.transforms import from_time_to_wdm
from wdm_transform.windows import gnmf

if "__file__" in globals():
    NOTEBOOK_DIR = Path(__file__).resolve().parent
else:
    cwd = Path.cwd()
    docs_studies_dir = cwd / "docs" / "studies"
    NOTEBOOK_DIR = docs_studies_dir if docs_studies_dir.exists() else cwd

outdir_monochrome_stationary_psd = NOTEBOOK_DIR / "outdir_monochrome_stationary_psd"
outdir_monochrome_stationary_psd.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = outdir_monochrome_stationary_psd / "run_log.txt"
MARKDOWN_PATH = NOTEBOOK_DIR / "monochrome_stationary_psd.md"
RUN_LOG_START = "<!-- BEGIN GENERATED RUN LOG -->"
RUN_LOG_END = "<!-- END GENERATED RUN LOG -->"
_run_log_chunks: list[str] = []


def print(*args, sep=" ", end="\n", file=None, flush=False):
    target = sys.stdout if file is None else file
    builtins.print(*args, sep=sep, end=end, file=target, flush=flush)
    if target in (sys.stdout, sys.__stdout__):
        text = sep.join(str(arg) for arg in args) + end
        _run_log_chunks.append(text)
        RUN_LOG_PATH.write_text("".join(_run_log_chunks))


def _update_markdown_run_log() -> None:
    if not MARKDOWN_PATH.exists():
        return
    body = "".join(_run_log_chunks).rstrip()
    block = (
        f"{RUN_LOG_START}\n```text\n{body}\n```\n{RUN_LOG_END}"
        if body
        else f"{RUN_LOG_START}\n_No run output captured yet._\n{RUN_LOG_END}"
    )
    markdown = MARKDOWN_PATH.read_text()
    if RUN_LOG_START in markdown and RUN_LOG_END in markdown:
        start = markdown.index(RUN_LOG_START)
        end = markdown.index(RUN_LOG_END) + len(RUN_LOG_END)
        markdown = markdown[:start] + block + markdown[end:]
    else:
        markdown = markdown.rstrip() + (
            "\n\n## Run log\n\n"
            "This section is generated from the script's `print()` output.\n\n"
            f"{block}\n"
        )
    MARKDOWN_PATH.write_text(markdown)


atexit.register(_update_markdown_run_log)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 160) -> Path:
    """Save a study figure to a docs-local assets directory and close it."""
    path = outdir_monochrome_stationary_psd / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


RNG = np.random.default_rng(4)


def sinusoid(
    amplitude: float,
    frequency: float,
    phase: float,
    n: int,
    dt: float,
) -> np.ndarray:
    times = np.arange(n) * dt
    return amplitude * np.sin(2.0 * np.pi * frequency * times + phase)


def colored_noise_psd(freqs: np.ndarray) -> np.ndarray:
    """Simple stationary PSD: flat floor plus a broad bump near 3 Hz."""
    return 10.0 + 100.0 * np.exp(-((np.abs(freqs) - 3.0) ** 2))


def random_signal_from_psd(
    psd_func,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(n, d=dt)
    white = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    shaped = np.sqrt(psd_func(freqs)) * white / np.sqrt(2.0)
    return np.fft.irfft(shaped, n=n)


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / np.linalg.norm(reference))


def snr_from_vectors(signal_values: np.ndarray, noise_values: np.ndarray) -> float:
    return float(np.linalg.norm(signal_values) / np.linalg.norm(noise_values))


def run_nuts(model, seed: int) -> dict[str, np.ndarray]:
    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={"A": amplitude, "f0": frequency, "phi": phase}
        ),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=300,
        num_samples=500,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed))
    return {name: np.asarray(values) for name, values in mcmc.get_samples().items()}


def pack_samples(samples: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([samples["A"], samples["f0"], samples["phi"]])


def summarize_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(samples, axis=0), np.std(samples, axis=0)


nt = 32
n_total = 8000
dt = 0.1
nf = n_total // nt

amplitude = 0.1
frequency = 1.1
phase = 0.5

times = np.arange(n_total) * dt
freqs = np.fft.fftfreq(n_total, d=dt)
positive = freqs >= 0.0

signal = sinusoid(amplitude, frequency, phase, n_total, dt)
noise = random_signal_from_psd(colored_noise_psd, n_total, dt, RNG)
data = signal + noise

signal_series = TimeSeries(signal, dt=dt)
noise_series = TimeSeries(noise, dt=dt)
data_series = TimeSeries(data, dt=dt)

signal_fft = signal_series.to_frequency_series()
noise_fft = noise_series.to_frequency_series()
data_fft = data_series.to_frequency_series()

signal_wdm = signal_series.to_wdm(nt=nt)
noise_wdm = noise_series.to_wdm(nt=nt)
data_wdm = data_series.to_wdm(nt=nt)

channel_centers = np.arange(signal_wdm.nf + 1) / (2.0 * signal_wdm.nf * signal_wdm.dt)
dominant_channel = int(np.argmax(np.sum(np.asarray(signal_wdm.coeffs) ** 2, axis=0)))

print(f"WDM shape: {signal_wdm.shape}")
print(f"Dominant WDM channel: m={dominant_channel}")
print(f"Channel center near the sinusoid: {channel_centers[dominant_channel]:.5f} Hz")

backend = get_backend()
dt_block = nf * dt
normalization = nt / 2.0

fixed_m = 5
fixed_n = 7

n_overlap = (
    np.array(
        [
            [
                np.vdot(
                    np.asarray(
                        gnmf(backend, n1, fixed_m, freqs, dt_block, nf, 1.0 / 3.0, 1.0)
                    ),
                    np.asarray(
                        gnmf(backend, n2, fixed_m, freqs, dt_block, nf, 1.0 / 3.0, 1.0)
                    ),
                )
                for n2 in range(nt)
            ]
            for n1 in range(nt)
        ]
    )
    / normalization
)

m_overlap = (
    np.array(
        [
            [
                np.vdot(
                    np.asarray(
                        gnmf(backend, fixed_n, m1, freqs, dt_block, nf, 1.0 / 3.0, 1.0)
                    ),
                    np.asarray(
                        gnmf(backend, fixed_n, m2, freqs, dt_block, nf, 1.0 / 3.0, 1.0)
                    ),
                )
                for m2 in range(nf + 1)
            ]
            for m1 in range(nf + 1)
        ]
    )
    / normalization
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im0 = axes[0].imshow(np.abs(n_overlap), cmap="magma", aspect="auto")
im1 = axes[1].imshow(np.abs(m_overlap), cmap="magma", aspect="auto")
axes[0].set_title(rf"$|\langle g_{{n,{fixed_m}}}, g_{{n',{fixed_m}}} \rangle|$")
axes[1].set_title(rf"$|\langle g_{{{fixed_n},m}}, g_{{{fixed_n},m'}} \rangle|$")
axes[0].set_xlabel("n'")
axes[0].set_ylabel("n")
axes[1].set_xlabel("m'")
axes[1].set_ylabel("m")
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.tight_layout()
_ = save_figure(fig, "basis_overlap_checks")

recovered_time = data_wdm.to_time_series()
recovered_fft = data_wdm.to_frequency_series()

time_rel_error = relative_l2_error(
    np.asarray(data_series.data), np.asarray(recovered_time.data)
)
fft_rel_error = relative_l2_error(
    np.asarray(data_fft.data), np.asarray(recovered_fft.data)
)

print(f"Relative time-domain reconstruction error: {time_rel_error:.3e}")
print(f"Relative FFT-domain reconstruction error: {fft_rel_error:.3e}")

fig, axes = plt.subplots(3, 1, figsize=(11, 11))
signal_series.plot(ax=axes[0], color="tab:blue", label="signal")
data_series.plot(ax=axes[0], color="tab:gray", alpha=0.7, label="signal + noise")
axes[0].legend()
axes[0].set_title("Time-domain samples")

axes[1].plot(
    freqs[positive],
    np.abs(np.asarray(signal_fft.data))[positive],
    label="|FFT(signal)|",
)
axes[1].plot(
    freqs[positive],
    np.abs(np.asarray(noise_fft.data))[positive],
    alpha=0.7,
    label="|FFT(noise)|",
)
axes[1].plot(
    freqs[positive],
    np.sqrt(colored_noise_psd(freqs[positive])),
    label=r"$\sqrt{\mathrm{PSD}}$",
)
axes[1].set_xlim(0.0, 0.5 / dt)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Magnitude")
axes[1].legend()
axes[1].set_title("Frequency-domain magnitude")

data_wdm.plot(ax=axes[2], cmap="viridis")
axes[2].set_title("Packed WDM coefficients")

fig.tight_layout()
_ = save_figure(fig, "time_frequency_wdm_views")

fig, ax = plt.subplots(figsize=(11, 4))
plot_spectrogram(data_series, ax=ax, spec_kwargs={"nperseg": 64, "noverlap": 48})
ax.set_title("Reference spectrogram")
fig.tight_layout()
_ = save_figure(fig, "reference_spectrogram")

signal_time_energy = float(np.sum(np.asarray(signal_series.data) ** 2))
signal_fft_energy = float(np.sum(np.abs(np.asarray(signal_fft.data)) ** 2) / n_total)
signal_wdm_energy = float(np.sum(np.asarray(signal_wdm.coeffs) ** 2))

noise_time_energy = float(np.sum(np.asarray(noise_series.data) ** 2))
noise_fft_energy = float(np.sum(np.abs(np.asarray(noise_fft.data)) ** 2) / n_total)
noise_wdm_energy = float(np.sum(np.asarray(noise_wdm.coeffs) ** 2))

print("Signal energies")
print(f"  time: {signal_time_energy:.6f}")
print(f"  fft : {signal_fft_energy:.6f}")
print(f"  wdm : {signal_wdm_energy:.6f}")

print("\nNoise energies")
print(f"  time: {noise_time_energy:.6f}")
print(f"  fft : {noise_fft_energy:.6f}")
print(f"  wdm : {noise_wdm_energy:.6f}")

print("\nSNR estimates")
print(f"  time-domain norm ratio: {snr_from_vectors(signal, noise):.6f}")
print(
    "  fft-domain norm ratio : "
    f"{snr_from_vectors(np.asarray(signal_fft.data), np.asarray(noise_fft.data)):.6f}"
)
wdm_snr = snr_from_vectors(np.asarray(signal_wdm.coeffs), np.asarray(noise_wdm.coeffs))
print(f"  wdm-domain norm ratio : {wdm_snr:.6f}")

signal_fft_from_wdm = signal_wdm.to_frequency_series()
noise_fft_from_wdm = noise_wdm.to_frequency_series()

signal_fft_rel_error = np.abs(
    np.asarray(signal_fft.data)[positive]
    - np.asarray(signal_fft_from_wdm.data)[positive]
)
noise_fft_rel_error = np.abs(
    np.asarray(noise_fft.data)[positive] - np.asarray(noise_fft_from_wdm.data)[positive]
)

signal_fft_scale = np.maximum(np.abs(np.asarray(signal_fft.data)[positive]), 1e-12)
noise_fft_scale = np.maximum(np.abs(np.asarray(noise_fft.data)[positive]), 1e-12)

print("FFT reconstruction summary")
print(f"  signal max abs error     : {np.max(signal_fft_rel_error):.3e}")
print(
    "  signal max relative error: "
    f"{np.max(signal_fft_rel_error / signal_fft_scale):.3e}"
)
print(f"  noise max abs error      : {np.max(noise_fft_rel_error):.3e}")
print(
    f"  noise max relative error : {np.max(noise_fft_rel_error / noise_fft_scale):.3e}"
)

signal_coeffs = np.asarray(signal_wdm.coeffs)
inference_channel = int(np.argmax(np.sum(signal_coeffs**2, axis=0)))
jax_times = jnp.arange(n_total) * dt
jax_psd = jnp.asarray(colored_noise_psd(freqs))
jax_fft_data = jnp.asarray(np.asarray(data_fft.data))
observed_wdm_jax = from_time_to_wdm(
    data,
    nt=nt,
    nf=nf,
    a=1.0 / 3.0,
    d=1.0,
    dt=dt,
    backend="jax",
)

noise_realizations = np.stack(
    [
        # Estimate the WDM noise scale directly in coefficient space by pushing
        # Monte Carlo noise realizations through the same forward transform.
        np.asarray(
            from_time_to_wdm(
                random_signal_from_psd(colored_noise_psd, n_total, dt, RNG),
                nt=nt,
                nf=nf,
                a=1.0 / 3.0,
                d=1.0,
                dt=dt,
                backend="jax",
            )
        )
        for _ in range(256)
    ]
)
wdm_noise_variance = noise_realizations.var(axis=0) + 1e-12

print(f"Using dominant WDM channel m={inference_channel}")
print(
    "Median estimated WDM noise variance in that channel: "
    f"{np.median(wdm_noise_variance[:, inference_channel]):.4e}"
)


def numpyro_frequency_model() -> None:
    amp = numpyro.sample("A", dist.Uniform(0.0, 0.3))
    freq0 = numpyro.sample("f0", dist.Uniform(0.8, 1.4))
    phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

    model = amp * jnp.sin(2.0 * jnp.pi * freq0 * jax_times + phi0)
    diff = jax_fft_data - jnp.fft.fft(model)
    numpyro.factor("log_like", -0.5 * jnp.sum(jnp.abs(diff) ** 2 / jax_psd))


def numpyro_wdm_model() -> None:
    amp = numpyro.sample("A", dist.Uniform(0.0, 0.3))
    freq0 = numpyro.sample("f0", dist.Uniform(0.8, 1.4))
    phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

    model = amp * jnp.sin(2.0 * jnp.pi * freq0 * jax_times + phi0)
    coeffs = from_time_to_wdm(
        model,
        nt=nt,
        nf=nf,
        a=1.0 / 3.0,
        d=1.0,
        dt=dt,
        backend="jax",
    )
    diff = observed_wdm_jax[:, inference_channel] - coeffs[:, inference_channel]
    variance = jnp.asarray(wdm_noise_variance[:, inference_channel])
    numpyro.factor("log_like", -0.5 * jnp.sum(diff**2 / variance))


fiducial = np.array([amplitude, frequency, phase])
frequency_samples = pack_samples(run_nuts(numpyro_frequency_model, seed=0))
wdm_samples = pack_samples(run_nuts(numpyro_wdm_model, seed=1))

freq_mean, freq_std = summarize_samples(frequency_samples)
wdm_mean, wdm_std = summarize_samples(wdm_samples)

print("\nPosterior mean ± std")
print(
    f"  FFT : A={freq_mean[0]:.5f}±{freq_std[0]:.5f}, "
    f"f0={freq_mean[1]:.5f}±{freq_std[1]:.5f}, "
    f"phi={freq_mean[2]:.5f}±{freq_std[2]:.5f}"
)
print(
    f"  WDM : A={wdm_mean[0]:.5f}±{wdm_std[0]:.5f}, "
    f"f0={wdm_mean[1]:.5f}±{wdm_std[1]:.5f}, "
    f"phi={wdm_mean[2]:.5f}±{wdm_std[2]:.5f}"
)
print(
    f"  Delta mean: dA={wdm_mean[0] - freq_mean[0]:.5e}, "
    f"df0={wdm_mean[1] - freq_mean[1]:.5e}, "
    f"dphi={wdm_mean[2] - freq_mean[2]:.5e}"
)

fig = corner.corner(
    frequency_samples,
    labels=[r"$A$", r"$f_0$", r"$\phi$"],
    truths=fiducial,
    color="tab:blue",
    hist_kwargs={"density": True},
    plot_contours=True,
    fill_contours=False,
)
corner.corner(
    wdm_samples,
    fig=fig,
    labels=[r"$A$", r"$f_0$", r"$\phi$"],
    truths=fiducial,
    color="tab:orange",
    hist_kwargs={"density": True},
    plot_contours=True,
    fill_contours=False,
)
fig.legend(
    handles=[
        Line2D([], [], color="tab:blue", label="FFT likelihood"),
        Line2D([], [], color="tab:orange", label="WDM likelihood"),
    ],
    loc="upper right",
    frameon=False,
)
_ = save_figure(fig, "posterior_comparison")
