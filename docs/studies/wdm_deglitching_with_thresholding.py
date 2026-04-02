# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Deglitching with Thresholding
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/studies/wdm_deglitching_with_thresholding.py)
#
# This study shows a deliberately simple WDM-domain cleanup workflow:
#
# - generate a smooth signal plus stationary noise
# - inject a few loud, short-duration artifacts
# - transform the data into WDM coefficients
# - build a blurred glitch score in the WDM grid
# - threshold that score to obtain a soft time-bin mask
# - attenuate the flagged WDM coefficients and reconstruct the cleaned series
#
# The point is not that this is a production deglitcher. The point is that WDM
# makes impulsive, broadband artifacts easy to localize in a way that is much
# harder to express with a plain FFT.

# %%
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "corner>=2.2",
            "jax[cpu]>=0.4.30",
            "numpyro>=0.15",
            "ipywidgets>=8.1",
            "git+https://github.com/pywavelet/wdm_transform.git",
        ],
        check=True,
    )

import corner
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import chirp, welch

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from matplotlib.lines import Line2D
from numpyro.infer import MCMC, NUTS, init_to_value

from wdm_transform import TimeSeries, WDM


RNG = np.random.default_rng(12)


def build_clean_signal(times: np.ndarray) -> np.ndarray:
    """Signal with one dominant monochromatic component plus milder structure."""
    dominant = TARGET_AMPLITUDE * np.sin(
        2.0 * np.pi * TARGET_FREQUENCY * times + TARGET_PHASE
    )
    weak_background = 0.08 * np.cos(2.0 * np.pi * 0.045 * times + 1.1)
    return dominant + weak_background


def inject_glitches(
    times: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    widths: np.ndarray,
    carrier_frequencies: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    glitch = np.zeros_like(times)
    sample_index = np.arange(len(times))
    for center, amp, width, carrier_frequency, phase in zip(
        centers,
        amplitudes,
        widths,
        carrier_frequencies,
        phases,
        strict=True,
    ):
        envelope = np.exp(-0.5 * ((sample_index - center) / width) ** 2)
        carrier = np.cos(2.0 * np.pi * carrier_frequency * (times - times[center]) + phase)
        glitch += amp * envelope * carrier
    return glitch


def robust_channel_scale(coeffs: np.ndarray) -> np.ndarray:
    """Robust per-channel noise scale from the median absolute deviation."""
    scale = np.median(np.abs(coeffs), axis=0) / 0.6745
    return np.maximum(scale, 1e-6)


def welch_psd(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    fs = 1.0 / dt
    freqs, psd = welch(values, fs=fs, nperseg=128, noverlap=96, scaling="density")
    return freqs, psd


def keep_runs(mask: np.ndarray, *, min_width: int) -> np.ndarray:
    """Keep only contiguous runs of ones that are at least ``min_width`` long."""
    kept = np.zeros_like(mask)
    starts = np.where(np.diff(np.concatenate([[0], mask, [0]])) == 1)[0]
    stops = np.where(np.diff(np.concatenate([[0], mask, [0]])) == -1)[0]
    for start, stop in zip(starts, stops):
        if stop - start >= min_width:
            kept[start:stop] = 1.0
    return kept


def detect_glitches(
    coeffs: np.ndarray,
    *,
    dt: float,
    threshold_scale: float = 1.4,
    min_width: int = 2,
) -> dict[str, np.ndarray | float]:
    """Build the WDM-domain glitch score and corresponding soft mask."""
    channel_scale = robust_channel_scale(coeffs)
    whitened = np.abs(coeffs) / channel_scale[None, :]
    score_grid = gaussian_filter(whitened, sigma=(0.7, 1.2))
    time_score = score_grid.mean(axis=1)

    score_median = np.median(time_score)
    score_mad = np.median(np.abs(time_score - score_median))
    threshold = score_median + threshold_scale * score_mad

    mask_hard = (time_score > threshold).astype(float)
    mask_hard = keep_runs(mask_hard, min_width=min_width)
    mask_soft = gaussian_filter1d(mask_hard, sigma=0.9)
    mask_soft = np.clip(mask_soft, 0.0, 1.0)

    return {
        "channel_scale": channel_scale,
        "whitened": whitened,
        "score_grid": score_grid,
        "time_score": time_score,
        "threshold": float(threshold),
        "mask_hard": mask_hard,
        "mask_soft": mask_soft,
        "flagged_bins": np.flatnonzero(mask_hard),
    }


def iterative_deglitch(
    values: np.ndarray,
    *,
    dt: float,
    nt: int,
    n_iter: int,
) -> list[dict[str, np.ndarray | float]]:
    """Alternate WDM detection and reconstruction for a few staged iterations."""
    current = values.copy()
    history: list[dict[str, np.ndarray | float]] = []

    for iteration in range(n_iter):
        coeffs = np.asarray(TimeSeries(current, dt=dt).to_wdm(nt=nt).coeffs)
        threshold_scale = 1.4 - 0.10 * iteration
        attenuation_strength = 0.45 + 0.15 * iteration
        detection = detect_glitches(coeffs, dt=dt, threshold_scale=threshold_scale)
        coeff_mask = (np.asarray(detection["whitened"]) > 2.5).astype(float)
        attenuation = 1.0 - attenuation_strength * (
            np.asarray(detection["mask_soft"])[:, None] * coeff_mask
        )
        cleaned = np.asarray(WDM(coeffs * attenuation, dt=dt).to_time_series().data)

        history.append(
            {
                "iteration": iteration + 1,
                "input": current,
                "coeffs": coeffs,
                "coeff_mask": coeff_mask,
                "threshold_scale": threshold_scale,
                "attenuation": attenuation,
                "cleaned": cleaned,
                **detection,
            }
        )
        current = cleaned

    return history


def run_nuts(y: np.ndarray, times: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    jt = jnp.asarray(times)
    jy = jnp.asarray(y)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 1.0))
        freq0 = numpyro.sample("f0", dist.Uniform(0.1, 0.25))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        mean = amp * jnp.sin(2.0 * jnp.pi * freq0 * jt + phi0)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=jy)

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TARGET_AMPLITUDE,
                "f0": TARGET_FREQUENCY,
                "phi": TARGET_PHASE,
                "sigma": NOISE_SIGMA,
            }
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


def summarize(samples: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    return {
        name: (float(values.mean()), float(values.std()))
        for name, values in samples.items()
    }


# %% [markdown]
# ## Synthetic data
#
# We build a smooth underlying signal, add stationary Gaussian noise, and then
# inject several loud glitch bursts. Each glitch has a short Gaussian envelope
# but also a fast oscillatory carrier, so the contamination is both localized in
# time and rich in high-frequency content. That is exactly the type of feature
# WDM is good at isolating.

# %%
nt = 32
n_total = 1024
dt = 0.1
nf = n_total // nt
TARGET_AMPLITUDE = 0.55
TARGET_FREQUENCY = 0.18
TARGET_PHASE = 0.30
NOISE_SIGMA = 0.18

times = np.arange(n_total) * dt
clean_signal = build_clean_signal(times)
stationary_noise = NOISE_SIGMA * RNG.normal(size=n_total)

glitch_centers = np.array([150, 180, 430, 615, 760, 905])
glitch_amplitudes = np.array([4.2, -5.0, 3.8, -4.5, 5.4, 3.6])
glitch_widths = np.array([3.5, 2.2, 2.8, 3.0, 4.5, 2.4])
glitch_carrier_frequencies = np.array([2.4, 3.8, 2.9, 4.1, 2.2, 4.5])
glitch_phases = np.array([0.2, 1.1, -0.7, 2.4, -1.3, 0.8])
glitches = inject_glitches(
    times,
    glitch_centers,
    glitch_amplitudes,
    glitch_widths,
    glitch_carrier_frequencies,
    glitch_phases,
)

reference = clean_signal + stationary_noise
observed = reference + glitches

reference_series = TimeSeries(reference, dt=dt)
observed_series = TimeSeries(observed, dt=dt)
observed_wdm = observed_series.to_wdm(nt=nt)
coeffs = np.asarray(observed_wdm.coeffs)

print(f"WDM shape: {observed_wdm.shape}")
print(f"Injected glitch sample indices: {glitch_centers.tolist()}")

# %% [markdown]
# ## What the contamination looks like
#
# In the time domain the glitches appear as obvious short bursts. In the WDM
# grid they show up as localized time bins with unusually large activity across
# many channels. That is the key structural clue we use for cleanup.

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1.1, 1.0])
axes[0].plot(times, clean_signal, color="tab:blue", alpha=0.75, label="clean signal")
axes[0].plot(times, observed, color="tab:red", alpha=0.7, label="observed with glitches")
for center in glitch_centers:
    axes[0].axvline(center * dt, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
axes[0].set_title("Observed data with injected glitches")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].legend(frameon=False, loc="upper right")

observed_wdm.plot(ax=axes[1], cmap="viridis")
axes[1].set_title("Observed WDM coefficient grid")

fig.tight_layout()

# %% [markdown]
# ## Build a simple glitch score
#
# The cleanup rule below is intentionally simple:
#
# 1. Compute a robust per-channel scale from the coefficient magnitudes.
# 2. Whiten the WDM grid by that scale.
# 3. Blur the absolute whitened grid with a small Gaussian filter.
# 4. Collapse the blurred grid into one score per time bin.
# 5. Threshold the score and smooth the resulting binary mask.
#
# The logic is that a glitch tends to light up many channels at once in a small
# number of neighboring WDM time bins, whereas the underlying signal is more
# structured and channel-localized.

# %%
one_pass = detect_glitches(coeffs, dt=dt)
channel_scale = np.asarray(one_pass["channel_scale"])
whitened = np.asarray(one_pass["whitened"])
score_grid = np.asarray(one_pass["score_grid"])
time_score = np.asarray(one_pass["time_score"])
threshold = float(one_pass["threshold"])
mask_hard = np.asarray(one_pass["mask_hard"])
mask_soft = np.asarray(one_pass["mask_soft"])
attenuation = 1.0 - 0.92 * mask_soft[:, None]

flagged_bins = np.asarray(one_pass["flagged_bins"])
time_bin_size = nf * dt

print(f"Flagged WDM time bins: {flagged_bins.tolist()}")
print(f"Threshold on time-bin score: {threshold:.3f}")

# %% [markdown]
# The panels below show the intermediate representation:
#
# - whitened coefficient magnitudes
# - the blurred score used for detection
# - the final 1D soft mask over WDM time bins

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 10), height_ratios=[1.0, 1.0, 0.7])

WDM(whitened, dt=dt).plot(ax=axes[0], cmap="magma")
axes[0].set_title("Whitened WDM magnitude")

WDM(score_grid, dt=dt).plot(ax=axes[1], cmap="magma")
axes[1].set_title("Blurred glitch score in WDM space")

bin_times = np.arange(nt) * time_bin_size
axes[2].plot(bin_times, time_score, color="tab:blue", label="time-bin score")
axes[2].axhline(threshold, color="tab:red", linestyle="--", label="threshold")
axes[2].plot(bin_times, mask_soft * np.max(time_score), color="tab:orange", label="soft mask (scaled)")
axes[2].set_title("Collapsed score and soft WDM time-bin mask")
axes[2].set_xlabel("WDM time-bin coordinate [s]")
axes[2].set_ylabel("Score")
axes[2].legend(frameon=False, loc="upper right")

fig.tight_layout()

# %% [markdown]
# ## Reconstruct the cleaned series
#
# We attenuate the WDM coefficients in the flagged time bins and reconstruct
# back to the time domain. This simple version applies the same time-bin mask to
# every channel.

# %%
cleaned_coeffs = coeffs * attenuation
cleaned_series = WDM(cleaned_coeffs, dt=dt).to_time_series()
cleaned = np.asarray(cleaned_series.data)

observed_mse = float(np.mean((observed - reference) ** 2))
cleaned_mse = float(np.mean((cleaned - reference) ** 2))

print(f"MSE before cleanup: {observed_mse:.4f}")
print(f"MSE after cleanup : {cleaned_mse:.4f}")

# %% [markdown]
# ## Iterative detect-clean-reconstruct loop
#
# A natural refinement is to repeat the same operation a few times:
#
# 1. detect unusual WDM activity
# 2. attenuate the flagged bins
# 3. reconstruct the time series
# 4. transform the cleaned result back to WDM and detect again
#
# The motivation is simple. A very loud glitch can partially hide weaker
# neighbors on the first pass. After one reconstruction, the dominant artifact
# is smaller, so the next pass can refine the score and the mask.

# %%
iter_history = iterative_deglitch(observed, dt=dt, nt=nt, n_iter=3)
iter_cleaned = np.asarray(iter_history[-1]["cleaned"])
iter_mse = float(np.mean((iter_cleaned - reference) ** 2))

print("Iterative deglitching summary")
for step in iter_history:
    flagged = np.asarray(step["flagged_bins"]).tolist()
    mse = float(np.mean((np.asarray(step["cleaned"]) - reference) ** 2))
    print(
        f"  iter {int(step['iteration'])}: "
        f"flagged bins={flagged}, "
        f"threshold={float(step['threshold']):.3f}, "
        f"MSE={mse:.4f}"
    )

# %% [markdown]
# The next figure shows how the time-bin score evolves as we iterate. In a good
# case the peaks caused by the glitches become less extreme after each pass, and
# the cleaned waveform gets closer to the reference signal-plus-noise series.
#
# In this toy problem the first pass already removes most of the artifact power.
# Later passes still refine the score, but they also start to attenuate some
# non-glitch structure. That is why iterative schemes usually need an explicit
# stopping rule instead of a fixed number of passes.

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[0.9, 1.1])

colors = ["tab:red", "tab:orange", "tab:green"]
for step, color in zip(iter_history, colors, strict=True):
    axes[0].plot(
        bin_times,
        np.asarray(step["time_score"]),
        color=color,
        label=f"iter {int(step['iteration'])} score",
    )
    axes[0].axhline(
        float(step["threshold"]),
        color=color,
        linestyle="--",
        alpha=0.6,
    )
axes[0].set_title("Iterative WDM time-bin scores")
axes[0].set_xlabel("WDM time-bin coordinate [s]")
axes[0].set_ylabel("Score")
axes[0].legend(frameon=False, loc="upper right")

axes[1].plot(times, reference, color="tab:blue", alpha=0.7, label="reference")
axes[1].plot(times, observed, color="0.75", linewidth=1.0, label="observed")
axes[1].plot(times, cleaned, color="tab:orange", linewidth=1.1, label="one-pass cleaned")
axes[1].plot(times, iter_cleaned, color="tab:green", linewidth=1.3, label="iterative cleaned")
for center in glitch_centers:
    axes[1].axvline(center * dt, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
axes[1].set_title("One-pass vs iterative reconstruction")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Amplitude")
axes[1].legend(frameon=False, loc="upper right")

fig.tight_layout()

# %% [markdown]
# The next figure compares the full time series and then zooms into the glitch
# neighborhoods. This makes it easier to see both the suppression and the price
# paid by such a simple mask.

# %%
fig, axes = plt.subplots(4, 1, figsize=(11, 11), height_ratios=[1.3, 1.0, 1.0, 1.0])

axes[0].plot(times, reference, color="tab:blue", alpha=0.7, label="reference (signal + noise)")
axes[0].plot(times, observed, color="tab:red", alpha=0.55, label="observed")
axes[0].plot(times, cleaned, color="tab:orange", alpha=0.8, label="one-pass cleaned")
axes[0].plot(times, iter_cleaned, color="tab:green", alpha=0.9, label="iterative cleaned")
for bin_idx in flagged_bins:
    start = bin_idx * time_bin_size
    axes[0].axvspan(start, start + time_bin_size, color="tab:orange", alpha=0.08)
axes[0].set_title("Deglitching result in the time domain")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].legend(frameon=False, loc="upper right")

window_half_width = 3.0
for ax, center in zip(axes[1:], glitch_centers):
    center_time = center * dt
    mask = np.abs(times - center_time) <= window_half_width
    ax.plot(times[mask], reference[mask], color="tab:blue", alpha=0.7, label="reference")
    ax.plot(times[mask], observed[mask], color="tab:red", alpha=0.55, label="observed")
    ax.plot(times[mask], cleaned[mask], color="tab:orange", alpha=0.85, label="one-pass cleaned")
    ax.plot(times[mask], iter_cleaned[mask], color="tab:green", alpha=0.95, label="iterative cleaned")
    ax.axvline(center_time, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Zoom around glitch near t={center_time:.1f} s")

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()

# %% [markdown]
# ## PSD estimate before and after cleanup
#
# A useful side effect of deglitching is that stationary spectral estimates are
# often less biased by rare, loud artifacts. We compare Welch PSD estimates for
# the observed data, the one-pass and iterative cleaned reconstructions, and
# the reference data without glitches.

# %%
freqs_ref, psd_ref = welch_psd(reference, dt)
freqs_obs, psd_obs = welch_psd(observed, dt)
freqs_cleaned, psd_cleaned = welch_psd(cleaned, dt)
freqs_iter, psd_iter = welch_psd(iter_cleaned, dt)

fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.loglog(freqs_ref[1:], psd_ref[1:], color="tab:blue", alpha=0.8, label="reference")
ax.loglog(freqs_obs[1:], psd_obs[1:], color="tab:red", alpha=0.7, label="observed")
ax.loglog(freqs_cleaned[1:], psd_cleaned[1:], color="tab:orange", alpha=0.8, label="one-pass cleaned")
ax.loglog(freqs_iter[1:], psd_iter[1:], color="tab:green", alpha=0.85, label="iterative cleaned")
ax.set_title("Welch PSD estimate before and after WDM deglitching")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD")
ax.legend(frameon=False, loc="best")
fig.tight_layout()

# %% [markdown]
# ## Downstream signal inference with `numpyro`
#
# A common question after deglitching is whether the cleaned data leads to
# better parameter inference. To show that, we fit the dominant monochromatic
# component in this synthetic dataset:
#
# - amplitude `A`
# - frequency `f0`
# - phase `phi`
# - residual scatter `sigma`
#
# For this inference step we again use a *more selective* coefficient mask than
# the waveform/PSD section above. We also apply it iteratively, so the dominant
# artifact coefficients are removed first and the next WDM pass can refine the
# mask on a cleaner background.

# %%
inference_current = observed.copy()
for _ in range(2):
    inference_coeffs = np.asarray(TimeSeries(inference_current, dt=dt).to_wdm(nt=nt).coeffs)
    inference_detection = detect_glitches(inference_coeffs, dt=dt)
    inference_coeff_mask = (np.asarray(inference_detection["whitened"]) > 2.5).astype(float)
    inference_attenuation = 1.0 - 0.92 * (
        np.asarray(inference_detection["mask_soft"])[:, None] * inference_coeff_mask
    )
    inference_current = np.asarray(
        WDM(inference_coeffs * inference_attenuation, dt=dt).to_time_series().data
    )

inference_cleaned = inference_current

observed_samples = run_nuts(observed, times, seed=0)
inference_cleaned_samples = run_nuts(inference_cleaned, times, seed=1)
reference_samples = run_nuts(reference, times, seed=2)

observed_summary = summarize(observed_samples)
inference_summary = summarize(inference_cleaned_samples)
reference_summary = summarize(reference_samples)

print("Posterior mean ± std")
print(
    f"  observed         : A={observed_summary['A'][0]:.4f}±{observed_summary['A'][1]:.4f}, "
    f"f0={observed_summary['f0'][0]:.5f}±{observed_summary['f0'][1]:.5f}, "
    f"phi={observed_summary['phi'][0]:.4f}±{observed_summary['phi'][1]:.4f}, "
    f"sigma={observed_summary['sigma'][0]:.4f}±{observed_summary['sigma'][1]:.4f}"
)
print(
    f"  inference-cleaned: A={inference_summary['A'][0]:.4f}±{inference_summary['A'][1]:.4f}, "
    f"f0={inference_summary['f0'][0]:.5f}±{inference_summary['f0'][1]:.5f}, "
    f"phi={inference_summary['phi'][0]:.4f}±{inference_summary['phi'][1]:.4f}, "
    f"sigma={inference_summary['sigma'][0]:.4f}±{inference_summary['sigma'][1]:.4f}"
)
print(
    f"  reference        : A={reference_summary['A'][0]:.4f}±{reference_summary['A'][1]:.4f}, "
    f"f0={reference_summary['f0'][0]:.5f}±{reference_summary['f0'][1]:.5f}, "
    f"phi={reference_summary['phi'][0]:.4f}±{reference_summary['phi'][1]:.4f}, "
    f"sigma={reference_summary['sigma'][0]:.4f}±{reference_summary['sigma'][1]:.4f}"
)

# %% [markdown]
# The comparison below is the main point: after selective WDM deglitching, the
# posterior for the dominant sinusoid moves closer to the no-glitch reference
# and the fitted residual scatter `sigma` drops substantially.

# %%
truths = np.array([TARGET_AMPLITUDE, TARGET_FREQUENCY, TARGET_PHASE])
fig = corner.corner(
    pack_samples(observed_samples),
    labels=[r"$A$", r"$f_0$", r"$\phi$"],
    truths=truths,
    color="tab:red",
    hist_kwargs={"density": True},
    plot_contours=True,
    fill_contours=False,
)
corner.corner(
    pack_samples(inference_cleaned_samples),
    fig=fig,
    labels=[r"$A$", r"$f_0$", r"$\phi$"],
    truths=truths,
    color="tab:green",
    hist_kwargs={"density": True},
    plot_contours=True,
    fill_contours=False,
)
corner.corner(
    pack_samples(reference_samples),
    fig=fig,
    labels=[r"$A$", r"$f_0$", r"$\phi$"],
    truths=truths,
    color="tab:blue",
    hist_kwargs={"density": True},
    plot_contours=True,
    fill_contours=False,
)
fig.legend(
    handles=[
        Line2D([], [], color="tab:red", label="observed"),
        Line2D([], [], color="tab:green", label="inference-cleaned"),
        Line2D([], [], color="tab:blue", label="reference"),
    ],
    loc="upper right",
    frameon=False,
)

# %% [markdown]
# ## Remarks
#
# This example deliberately uses a crude cleanup rule. It is useful because the
# logic is transparent:
#
# - glitches are short in time and broad across channels
# - the WDM grid makes that pattern easy to detect
# - a soft time-bin mask already improves both the waveform and the PSD estimate
# - iterating the same detect-reconstruct loop can refine the cleanup further,
#   but it can also over-clean if you do not stop early enough
#
# More realistic pipelines could use:
#
# - channel-dependent masks
# - stronger stopping criteria for iterative detection and reconstruction
# - explicit glitch templates
# - statistically calibrated thresholds instead of a hand-tuned MAD rule
