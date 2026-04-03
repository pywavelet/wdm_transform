# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Sinusoid with Gaps
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/studies/wdm_monochromatic_signal_with_gap.py)
#
# This study upgrades the earlier toy gap example into a harder case:
#
# - one persistent sinusoid
# - the same stationary colored-noise PSD used in the colored-noise sinusoid study
# - several missing intervals
#
# The point is not that WDM magically contains more information than the time
# domain. The point is that gaps are local in WDM coordinates, while they
# become globally awkward in the FFT domain.
#
# We compare three inference strategies:
#
# - an **exact masked time-domain likelihood** for the colored-noise model
# - a **gap-ignorant FFT likelihood** that treats the zero-filled data as if it
#   were complete
# - a **localized WDM likelihood** that drops only the contaminated WDM time bins
#
# In this harder setting the WDM method should not beat the exact benchmark, but
# it should have a better chance of tracking it than the gap-ignorant FFT treatment.

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

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from matplotlib.lines import Line2D
from numpyro.infer import MCMC, NUTS, init_to_value

from wdm_transform import TimeSeries
from wdm_transform.transforms import forward_wdm


RNG = np.random.default_rng(21)


def sinusoid(
    amplitude: float,
    frequency: float,
    phase: float,
    times: np.ndarray,
) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * frequency * times + phase)


def stationary_noise_psd(freqs: np.ndarray) -> np.ndarray:
    """Same stationary PSD used in the colored-noise sinusoid study."""
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


def build_periodic_covariance_from_psd(psd: np.ndarray, n: int) -> np.ndarray:
    """Build the periodic time-domain covariance implied by a discrete PSD."""
    # Match the normalization used by `random_signal_from_psd`.
    #
    # For the convention in this notebook, E[|X_k|^2] = PSD(f_k) in the rFFT
    # domain, and the periodic time-domain covariance is the inverse rFFT of
    # that one-sided spectrum divided by the total number of samples.
    first_row = np.fft.irfft(psd, n=n) / n
    offsets = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    return first_row[offsets]


def whiten_operator(covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance + 1e-9 * np.eye(covariance.shape[0]))
    return np.linalg.solve(chol, np.eye(chol.shape[0]))


def run_exact_masked_time_nuts(seed: int) -> dict[str, np.ndarray]:
    j_whitener = jnp.asarray(obs_whitener)
    j_obs = jnp.asarray(obs_whitener @ observed_values)
    j_times = jnp.asarray(observed_times)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 0.3))
        freq0 = numpyro.sample("f0", dist.Uniform(0.8, 1.4))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

        mean = amp * jnp.sin(2.0 * jnp.pi * freq0 * j_times + phi0)
        mean_white = j_whitener @ mean
        numpyro.sample("obs", dist.Normal(mean_white, 1.0), obs=j_obs)

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TRUE_AMPLITUDE,
                "f0": TRUE_FREQUENCY,
                "phi": TRUE_PHASE,
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


def run_gap_ignorant_fft_nuts(seed: int) -> dict[str, np.ndarray]:
    j_times = jnp.asarray(times)
    j_observed = jnp.asarray(np.fft.rfft(gapped_data)[fft_keep])
    j_psd = jnp.asarray(noise_psd_rfft[fft_keep])
    j_scale_base = jnp.sqrt((n_total / 2.0) * j_psd)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 0.3))
        freq0 = numpyro.sample("f0", dist.Uniform(0.8, 1.4))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Deliberately gap-ignorant: pretend the zero-filled FFT came from a
        # complete stationary dataset with no missing samples.
        trial = amp * jnp.sin(2.0 * jnp.pi * freq0 * j_times + phi0)
        trial_fft = jnp.fft.rfft(trial)[fft_keep]
        scale = sigma * j_scale_base

        numpyro.sample(
            "obs_real",
            dist.Normal(jnp.real(trial_fft), scale),
            obs=jnp.real(j_observed),
        )
        numpyro.sample(
            "obs_imag",
            dist.Normal(jnp.imag(trial_fft), scale),
            obs=jnp.imag(j_observed),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TRUE_AMPLITUDE,
                "f0": TRUE_FREQUENCY,
                "phi": TRUE_PHASE,
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


def run_wdm_nuts(
    observed_coeffs: np.ndarray,
    variance: np.ndarray,
    keep_time_bins: np.ndarray,
    keep_channels: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    j_observed = jnp.asarray(observed_coeffs[np.ix_(keep_time_bins, keep_channels)])
    j_variance = jnp.asarray(variance[np.ix_(keep_time_bins, keep_channels)])
    j_keep_time_bins = jnp.asarray(keep_time_bins)
    j_keep_channels = jnp.asarray(keep_channels)
    j_times = jnp.asarray(times)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 0.3))
        freq0 = numpyro.sample("f0", dist.Uniform(0.8, 1.4))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Compare the gapped data to the ungapped signal model only in WDM bins
        # where the gap effect is small.
        trial = amp * jnp.sin(2.0 * jnp.pi * freq0 * j_times + phi0)
        trial_wdm = forward_wdm(
            trial,
            nt=nt,
            nf=nf,
            a=1.0 / 3.0,
            d=1.0,
            dt=dt,
            backend="jax",
        )
        trial_subset = trial_wdm[j_keep_time_bins][:, j_keep_channels]
        numpyro.sample(
            "obs",
            dist.Normal(trial_subset, sigma * jnp.sqrt(j_variance)),
            obs=j_observed,
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TRUE_AMPLITUDE,
                "f0": TRUE_FREQUENCY,
                "phi": TRUE_PHASE,
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
# ## Synthetic data with multiple gaps and stationary colored noise
#
# We use almost the same setup as the colored-noise sinusoid study, but now add
# several missing intervals and a longer total duration. This makes the
# comparison easier to interpret because the main difference is the gaps, not a
# completely different signal/noise regime.

# %%
nt = 48
n_total = 1536
dt = 0.1
nf = n_total // nt

TRUE_AMPLITUDE = 0.10
TRUE_FREQUENCY = 1.10
TRUE_PHASE = 0.50

times = np.arange(n_total) * dt
clean_signal = sinusoid(
    TRUE_AMPLITUDE,
    TRUE_FREQUENCY,
    TRUE_PHASE,
    times,
)

noise = random_signal_from_psd(stationary_noise_psd, n_total, dt, RNG)
full_data = clean_signal + noise

gap_intervals = [(270, 390), (705, 840), (1140, 1260)]
sample_mask = np.ones(n_total)
for start, stop in gap_intervals:
    sample_mask[start:stop] = 0.0
observed_mask = sample_mask.astype(bool)

gapped_data = full_data * sample_mask
gapped_clean_signal = clean_signal * sample_mask

full_series = TimeSeries(full_data, dt=dt)
gapped_series = TimeSeries(gapped_data, dt=dt)
clean_series = TimeSeries(clean_signal, dt=dt)
gapped_clean_series = TimeSeries(gapped_clean_signal, dt=dt)

full_fft = full_series.to_frequency_series()
gapped_fft = gapped_series.to_frequency_series()
clean_wdm = clean_series.to_wdm(nt=nt)
gapped_wdm = gapped_series.to_wdm(nt=nt)
gapped_clean_wdm = gapped_clean_series.to_wdm(nt=nt)

clean_coeffs = np.asarray(clean_wdm.coeffs)
gapped_coeffs = np.asarray(gapped_wdm.coeffs)
gap_effect = np.abs(clean_coeffs - np.asarray(gapped_clean_wdm.coeffs))

freq_rfft = np.fft.rfftfreq(n_total, d=dt)
noise_psd_rfft = stationary_noise_psd(freq_rfft)
full_covariance = build_periodic_covariance_from_psd(noise_psd_rfft, n_total)
obs_indices = np.flatnonzero(observed_mask)
observed_times = times[obs_indices]
observed_values = gapped_data[obs_indices]
obs_covariance = full_covariance[np.ix_(obs_indices, obs_indices)]
obs_whitener = whiten_operator(obs_covariance)

dominant_channel = int(np.argmax(np.sum(clean_coeffs**2, axis=0)))
selected_channels = np.arange(
    max(0, dominant_channel - 2),
    min(clean_wdm.nf, dominant_channel + 2) + 1,
)

excluded_mask = np.zeros(nt, dtype=bool)
bin_padding = 1
for start, stop in gap_intervals:
    gap_bin_start = start // nf
    gap_bin_stop = (stop - 1) // nf
    excluded_mask[
        max(0, gap_bin_start - bin_padding) : min(nt, gap_bin_stop + bin_padding + 1)
    ] = True
kept_time_bins = np.flatnonzero(~excluded_mask)
excluded_bins = np.flatnonzero(excluded_mask)

channel_centers = np.arange(clean_wdm.nf + 1) / (2.0 * clean_wdm.nf * dt)
outside_gap_mismatch = np.linalg.norm(gap_effect[kept_time_bins]) / np.linalg.norm(
    clean_coeffs[kept_time_bins]
)
inside_gap_mismatch = np.linalg.norm(gap_effect[excluded_bins]) / np.linalg.norm(
    clean_coeffs[excluded_bins]
)
fft_keep = np.arange(1, n_total // 2)

print(f"WDM shape: {gapped_wdm.shape}")
print(
    "Gap intervals: "
    + ", ".join(
        f"{start}:{stop} ({start * dt:.1f}s to {stop * dt:.1f}s)"
        for start, stop in gap_intervals
    )
)
print(f"Dominant signal channel: m={dominant_channel}")
print(
    "Selected channels for WDM inference: "
    f"{selected_channels.tolist()} "
    f"(centers near {channel_centers[selected_channels]})"
)
print(f"Excluded WDM time bins: {excluded_bins.tolist()}")
print(f"Relative clean-signal WDM mismatch outside excluded bins: {outside_gap_mismatch:.3e}")
print(f"Relative clean-signal WDM mismatch inside excluded bins : {inside_gap_mismatch:.3e}")

# %% [markdown]
# The mismatch numbers above are the key locality check:
#
# - outside the gap-adjacent WDM bins, zero-filling does not change the clean
#   signal very much
# - inside those bins, the effect is large
#
# That is why a localized WDM likelihood is a plausible approximation here.

# %% [markdown]
# ## Time, FFT, and WDM views
#
# With several gaps, the time-domain issue is obvious. In the FFT, however, the
# effect is global: the masked data no longer looks like a clean narrowband line
# plus stationary colored noise. In WDM, the disturbance is still concentrated
# in a limited set of time bins.

# %%
freqs = np.fft.fftfreq(n_total, d=dt)
positive = freqs >= 0.0

fig, axes = plt.subplots(3, 1, figsize=(12, 11), height_ratios=[1.0, 1.0, 1.15])

axes[0].plot(times, full_data, color="0.70", linewidth=1.0, label="full noisy data")
axes[0].plot(times, gapped_data, color="tab:blue", linewidth=1.1, label="zero-filled gapped data")
for start, stop in gap_intervals:
    axes[0].axvspan(start * dt, stop * dt, color="tab:red", alpha=0.10)
axes[0].set_title("Time-domain data with multiple missing intervals")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].legend(frameon=False, loc="upper right")

axes[1].plot(freqs[positive], np.abs(np.asarray(full_fft.data))[positive], label="full data")
axes[1].plot(
    freqs[positive],
    np.abs(np.asarray(gapped_fft.data))[positive],
    label="zero-filled gapped data",
)
axes[1].set_xlim(0.0, 0.5 / dt)
axes[1].set_title("FFT magnitude: multiple gaps spread power globally")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Magnitude")
axes[1].legend(frameon=False, loc="upper right")

im = axes[2].imshow(
    np.abs(gapped_coeffs).T,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
for idx in excluded_bins:
    axes[2].axvspan(idx - 0.5, idx + 0.5, color="white", alpha=0.10)
axes[2].set_title("WDM magnitude of the zero-filled gapped data")
axes[2].set_xlabel("WDM time bin n")
axes[2].set_ylabel("WDM channel m")
fig.colorbar(im, ax=axes[2], pad=0.01, label=r"$|w_{n,m}|$")

fig.tight_layout()

# %% [markdown]
# ## Gap locality in WDM space
#
# The plots below compare the clean sinusoid to the same signal after
# zero-filling the gaps. The contamination is concentrated near a subset of WDM
# time bins rather than being spread across the full grid.

# %%
gap_score = gap_effect.sum(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.0, 0.7])
im = axes[0].imshow(gap_effect.T, origin="lower", aspect="auto", cmap="magma")
for idx in excluded_bins:
    axes[0].axvspan(idx - 0.5, idx + 0.5, color="white", alpha=0.10)
axes[0].set_title("Absolute WDM change caused by the gaps in the clean signal")
axes[0].set_xlabel("WDM time bin n")
axes[0].set_ylabel("WDM channel m")
fig.colorbar(im, ax=axes[0], pad=0.01, label=r"$|\Delta w_{n,m}|$")

axes[1].plot(np.arange(nt), gap_score, color="tab:red")
for idx in excluded_bins:
    axes[1].axvspan(idx - 0.5, idx + 0.5, color="tab:gray", alpha=0.12)
axes[1].set_title("Gap-induced WDM disturbance aggregated over channel")
axes[1].set_xlabel("WDM time bin n")
axes[1].set_ylabel(r"$\sum_m |\Delta w_{n,m}|$")

fig.tight_layout()

# %% [markdown]
# ## Posterior comparison
#
# Here the exact benchmark is no longer trivial:
#
# - the stationary colored noise is diagonal in the *complete-data* frequency basis
# - once we remove samples, the exact observed-data likelihood becomes dense
#
# We therefore compare:
#
# - **exact masked time-domain**: whiten the observed samples with the true
#   colored-noise covariance submatrix
# - **gap-ignorant FFT**: fit the zero-filled spectrum as if the data were complete
# - **WDM kept-bins**: keep only the channels around the signal and drop the WDM
#   time bins touched by the gaps

# %%
noise_realizations = np.stack(
    [
        np.asarray(
            TimeSeries(
                random_signal_from_psd(stationary_noise_psd, n_total, dt, RNG) * sample_mask,
                dt=dt,
            ).to_wdm(nt=nt).coeffs
        )
        for _ in range(128)
    ]
)
wdm_variance = noise_realizations.var(axis=0) + 1e-8

exact_time_samples = run_exact_masked_time_nuts(seed=0)
fft_samples = run_gap_ignorant_fft_nuts(seed=1)
wdm_samples = run_wdm_nuts(
    gapped_coeffs,
    wdm_variance,
    kept_time_bins,
    selected_channels,
    seed=2,
)

exact_time_summary = summarize(exact_time_samples)
fft_summary = summarize(fft_samples)
wdm_summary = summarize(wdm_samples)

print("Posterior mean ± std")
print(
    f"  exact masked time : A={exact_time_summary['A'][0]:.4f}±{exact_time_summary['A'][1]:.4f}, "
    f"f0={exact_time_summary['f0'][0]:.5f}±{exact_time_summary['f0'][1]:.5f}, "
    f"phi={exact_time_summary['phi'][0]:.4f}±{exact_time_summary['phi'][1]:.4f}"
)
print(
    f"  gap-ignorant FFT : A={fft_summary['A'][0]:.4f}±{fft_summary['A'][1]:.4f}, "
    f"f0={fft_summary['f0'][0]:.5f}±{fft_summary['f0'][1]:.5f}, "
    f"phi={fft_summary['phi'][0]:.4f}±{fft_summary['phi'][1]:.4f}, "
    f"sigma={fft_summary['sigma'][0]:.4f}±{fft_summary['sigma'][1]:.4f}"
)
print(
    f"  WDM kept-bins    : A={wdm_summary['A'][0]:.4f}±{wdm_summary['A'][1]:.4f}, "
    f"f0={wdm_summary['f0'][0]:.5f}±{wdm_summary['f0'][1]:.5f}, "
    f"phi={wdm_summary['phi'][0]:.4f}±{wdm_summary['phi'][1]:.4f}, "
    f"sigma={wdm_summary['sigma'][0]:.4f}±{wdm_summary['sigma'][1]:.4f}"
)

# %% [markdown]
# The exact masked-time result is the benchmark. The question is not whether WDM
# beats it; it should not. The question is whether the WDM approximation lands
# nearer to that benchmark than the gap-ignorant FFT treatment while using a much more
# local nuisance model.
#
# In the current synthetic run, the WDM approximation tracks the benchmark much
# better in amplitude than the gap-ignorant FFT fit, but it
# still shows its own approximation error and needs a variance-inflation factor.

# %%
truths = np.array([TRUE_AMPLITUDE, TRUE_FREQUENCY, TRUE_PHASE])
labels = [r"$A$", r"$f_0$", r"$\phi$"]

fig = corner.corner(
    pack_samples(exact_time_samples),
    labels=labels,
    truths=truths,
    color="tab:green",
    truth_color="black",
    levels=(0.5, 0.9),
    plot_density=False,
    fill_contours=False,
)
corner.corner(
    pack_samples(fft_samples),
    fig=fig,
    color="tab:purple",
    levels=(0.5, 0.9),
    plot_density=False,
    fill_contours=False,
)
corner.corner(
    pack_samples(wdm_samples),
    fig=fig,
    color="tab:orange",
    levels=(0.5, 0.9),
    plot_density=False,
    fill_contours=False,
)
fig.legend(
    handles=[
        Line2D([], [], color="tab:green", label="exact masked time"),
        Line2D([], [], color="tab:purple", label="gap-ignorant FFT"),
        Line2D([], [], color="tab:orange", label="WDM kept-bins"),
    ],
    loc="upper right",
    frameon=False,
)
fig.suptitle("Posterior comparison for a sinusoid with multiple gaps", y=1.02)

# %% [markdown]
# ## Posterior-mean prediction in the frequency domain
#
# The posterior means below are shown in the FFT domain rather than the time
# domain. That view is more relevant here because the main failure mode of the
# gap-ignorant FFT treatment is spectral leakage and amplitude bias.

# %%
exact_mean = {key: value[0] for key, value in exact_time_summary.items()}
fft_mean = {key: value[0] for key, value in fft_summary.items()}
wdm_mean = {key: value[0] for key, value in wdm_summary.items()}

exact_fit = sinusoid(exact_mean["A"], exact_mean["f0"], exact_mean["phi"], times)
fft_fit = sinusoid(fft_mean["A"], fft_mean["f0"], fft_mean["phi"], times)
wdm_fit = sinusoid(wdm_mean["A"], wdm_mean["f0"], wdm_mean["phi"], times)

truth_fft = np.abs(np.fft.rfft(clean_signal))
gapped_fft_mag = np.abs(np.fft.rfft(gapped_data))
exact_fft_mag = np.abs(np.fft.rfft(exact_fit))
fft_fit_mag = np.abs(np.fft.rfft(fft_fit))
wdm_fit_mag = np.abs(np.fft.rfft(wdm_fit))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.0, 1.0])

axes[0].plot(freq_rfft, gapped_fft_mag, color="tab:blue", alpha=0.75, label="zero-filled gapped data")
axes[0].plot(freq_rfft, truth_fft, color="black", linestyle="--", linewidth=1.2, label="true signal")
axes[0].plot(freq_rfft, exact_fft_mag, color="tab:green", linewidth=1.3, label="exact masked time")
axes[0].plot(freq_rfft, fft_fit_mag, color="tab:purple", linewidth=1.3, label="gap-ignorant FFT")
axes[0].plot(freq_rfft, wdm_fit_mag, color="tab:orange", linewidth=1.3, label="WDM kept-bins")
axes[0].set_xlim(0.0, 0.5 / dt)
axes[0].set_title("Posterior-mean spectral prediction")
axes[0].set_xlabel("Frequency [Hz]")
axes[0].set_ylabel(r"$|\mathrm{FFT}|$")
axes[0].legend(frameon=False, loc="upper right", ncol=2)

axes[1].plot(freq_rfft, np.abs(gapped_fft_mag - truth_fft), color="tab:blue", alpha=0.7, label="gapped data error")
axes[1].plot(freq_rfft, np.abs(exact_fft_mag - truth_fft), color="tab:green", label="exact masked time error")
axes[1].plot(freq_rfft, np.abs(fft_fit_mag - truth_fft), color="tab:purple", label="gap-ignorant FFT error")
axes[1].plot(freq_rfft, np.abs(wdm_fit_mag - truth_fft), color="tab:orange", label="WDM kept-bins error")
axes[1].set_xlim(0.0, 0.5 / dt)
axes[1].set_yscale("log")
axes[1].set_title("Absolute spectral error relative to the true signal")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Absolute error")
axes[1].legend(frameon=False, loc="upper right", ncol=2)

fig.tight_layout()
