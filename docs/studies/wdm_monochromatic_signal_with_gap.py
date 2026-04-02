# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Monochromatic Signal with a Gap
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/studies/wdm_monochromatic_signal_with_gap.py)
#
# This study looks at a very common nuisance: a clean narrow-band signal is
# observed with stationary noise, but one chunk of the time series is missing.
#
# We use the package WDM transform to show two complementary facts:
#
# - a gap is awkward in the FFT domain because multiplying by a mask in time
#   spreads power across frequency
# - the same gap is much more local in WDM coordinates, so we can often handle
#   it by dropping only the WDM time bins whose atoms overlap the missing
#   interval
#
# The exact benchmark here is still a masked time-domain likelihood that uses
# only the observed samples. The WDM likelihood is an approximation, but it is
# a practical one: fit in WDM space while ignoring the bins that are visibly
# contaminated by the gap.

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


def run_time_nuts(
    values: np.ndarray,
    sample_times: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    jt = jnp.asarray(sample_times)
    jy = jnp.asarray(values)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 1.5))
        freq0 = numpyro.sample("f0", dist.Uniform(0.7, 1.2))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.6))

        mean = amp * jnp.sin(2.0 * jnp.pi * freq0 * jt + phi0)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=jy)

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TRUE_AMPLITUDE,
                "f0": TRUE_FREQUENCY,
                "phi": TRUE_PHASE,
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


def run_fft_nuts(
    values: np.ndarray,
    sample_mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    jt = jnp.asarray(times)
    j_mask = jnp.asarray(sample_mask)
    j_observed = jnp.asarray(np.fft.rfft(values)[fft_keep])
    j_fft_std_factor = jnp.sqrt(float(sample_mask.sum()) / 2.0)

    def model() -> None:
        amp = numpyro.sample("A", dist.Uniform(0.0, 1.5))
        freq0 = numpyro.sample("f0", dist.Uniform(0.7, 1.2))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.6))

        trial = amp * jnp.sin(2.0 * jnp.pi * freq0 * jt + phi0)
        trial_fft = jnp.fft.rfft(trial * j_mask)[fft_keep]
        fft_sigma = sigma * j_fft_std_factor

        numpyro.sample(
            "obs_real",
            dist.Normal(jnp.real(trial_fft), fft_sigma),
            obs=jnp.real(j_observed),
        )
        numpyro.sample(
            "obs_imag",
            dist.Normal(jnp.imag(trial_fft), fft_sigma),
            obs=jnp.imag(j_observed),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "A": TRUE_AMPLITUDE,
                "f0": TRUE_FREQUENCY,
                "phi": TRUE_PHASE,
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
        amp = numpyro.sample("A", dist.Uniform(0.0, 1.5))
        freq0 = numpyro.sample("f0", dist.Uniform(0.7, 1.2))
        phi0 = numpyro.sample("phi", dist.Uniform(-jnp.pi, jnp.pi))

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
        numpyro.sample("obs", dist.Normal(trial_subset, jnp.sqrt(j_variance)), obs=j_observed)

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
# ## Synthetic data with one missing interval
#
# We use a monochromatic signal plus white noise. Then we remove one block of
# samples and replace it with zeros in the stored time series. That zero-filling
# is not the statistical model; it is just a convenient way to put the gapped
# series through the FFT and WDM transforms.

# %%
nt = 32
n_total = 1024
dt = 0.1
nf = n_total // nt

TRUE_AMPLITUDE = 0.72
TRUE_FREQUENCY = 0.94
TRUE_PHASE = 0.80
NOISE_SIGMA = 0.22

times = np.arange(n_total) * dt
clean_signal = sinusoid(TRUE_AMPLITUDE, TRUE_FREQUENCY, TRUE_PHASE, times)
noise = NOISE_SIGMA * RNG.normal(size=n_total)
full_data = clean_signal + noise

gap_start = 420
gap_stop = 560
sample_mask = np.ones(n_total)
sample_mask[gap_start:gap_stop] = 0.0
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
fft_keep = np.arange(1, n_total // 2)

dominant_channel = int(np.argmax(np.sum(clean_coeffs**2, axis=0)))
selected_channels = np.arange(
    max(0, dominant_channel - 1),
    min(clean_wdm.nf, dominant_channel + 1) + 1,
)

gap_bin_start = gap_start // nf
gap_bin_stop = (gap_stop - 1) // nf
bin_padding = 2
excluded_mask = np.zeros(nt, dtype=bool)
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

print(f"WDM shape: {gapped_wdm.shape}")
print(
    "Gap interval: "
    f"samples {gap_start}:{gap_stop} "
    f"({gap_start * dt:.1f}s to {gap_stop * dt:.1f}s)"
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
# The two mismatch numbers above are the basic justification for the WDM
# approximation in this notebook:
#
# - outside the gap-adjacent WDM bins, the zero-filled and ungapped signals
#   look very similar in WDM space
# - inside those bins, the gap strongly perturbs the coefficients
#
# That is exactly the behavior we want if we plan to "handle the gap" by
# dropping only a localized set of coefficients.

# %% [markdown]
# ## Time, FFT, and WDM views of the gap
#
# The same missing interval has very different visual signatures in different
# representations.
#
# - In time, the problem is obvious: one interval is missing.
# - In the FFT, the zero-filled gap causes broadband leakage.
# - In WDM, the disturbance stays concentrated in a small range of WDM time
#   bins, which we highlight below.

# %%
freqs = np.fft.fftfreq(n_total, d=dt)
positive = freqs >= 0.0

fig, axes = plt.subplots(3, 1, figsize=(12, 11), height_ratios=[1.0, 1.0, 1.15])

axes[0].plot(times, full_data, color="0.70", linewidth=1.0, label="full noisy data")
axes[0].plot(times, gapped_data, color="tab:blue", linewidth=1.1, label="zero-filled gapped data")
axes[0].axvspan(gap_start * dt, gap_stop * dt, color="tab:red", alpha=0.12, label="missing interval")
axes[0].set_title("Time-domain data with one missing interval")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].legend(frameon=False, loc="upper right")

axes[1].plot(freqs[positive], np.abs(np.asarray(full_fft.data))[positive], label="full data")
axes[1].plot(
    freqs[positive],
    np.abs(np.asarray(gapped_fft.data))[positive],
    label="zero-filled gapped data",
)
axes[1].set_xlim(0.0, 2.0)
axes[1].set_title("FFT magnitude: the gap spreads power globally")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Magnitude")
axes[1].legend(frameon=False, loc="upper right")

im = axes[2].imshow(
    np.abs(gapped_coeffs).T,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
axes[2].axvspan(
    excluded_bins[0] - 0.5,
    excluded_bins[-1] + 0.5,
    color="white",
    alpha=0.15,
    label="excluded WDM time bins",
)
axes[2].set_title("WDM magnitude of the zero-filled gapped data")
axes[2].set_xlabel("WDM time bin n")
axes[2].set_ylabel("WDM channel m")
axes[2].legend(frameon=False, loc="upper right")
fig.colorbar(im, ax=axes[2], pad=0.01, label=r"$|w_{n,m}|$")

fig.tight_layout()

# %% [markdown]
# ## Locality of the gap in WDM space
#
# To see the localization more directly, the next plots compare the clean
# sinusoid to the same sinusoid after zero-filling the gap. The important point
# is that the disturbance is concentrated near the gap-adjacent WDM time bins,
# not spread uniformly across the whole coefficient grid.

# %%
gap_score = gap_effect.sum(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.0, 0.7])
im = axes[0].imshow(gap_effect.T, origin="lower", aspect="auto", cmap="magma")
axes[0].axvspan(excluded_bins[0] - 0.5, excluded_bins[-1] + 0.5, color="white", alpha=0.15)
axes[0].set_title("Absolute WDM change caused by inserting the gap into the clean sinusoid")
axes[0].set_xlabel("WDM time bin n")
axes[0].set_ylabel("WDM channel m")
fig.colorbar(im, ax=axes[0], pad=0.01, label=r"$|\Delta w_{n,m}|$")

axes[1].plot(np.arange(nt), gap_score, color="tab:red")
axes[1].axvspan(excluded_bins[0] - 0.5, excluded_bins[-1] + 0.5, color="tab:gray", alpha=0.15)
axes[1].set_title("Gap-induced WDM disturbance aggregated over channel")
axes[1].set_xlabel("WDM time bin n")
axes[1].set_ylabel(r"$\sum_m |\Delta w_{n,m}|$")

fig.tight_layout()

# %% [markdown]
# ## Posterior comparison
#
# We now compare three inference setups:
#
# - **full-data time-domain posterior**: a best-case reference with no missing
#   interval
# - **masked time-domain posterior**: the exact benchmark for the gapped data,
#   using only observed samples
# - **naive FFT posterior**: fit the zero-filled gapped FFT while pretending the
#   frequency bins are independent
# - **WDM posterior**: compute WDM coefficients of the zero-filled gapped data,
#   then ignore the WDM time bins whose atoms overlap the missing interval
#
# The WDM posterior is the key point of the example. It does not try to model
# the contaminated bins. It simply uses the locality of the WDM basis to throw
# them away and fit on the unaffected remainder.
#
# The FFT-side comparison is useful for a different reason. Frequency-domain
# analysis is not impossible here, but the exact treatment is no longer the
# simple diagonal one used for stationary complete data. The time-domain mask
# mixes Fourier bins, so the exact FFT likelihood would need a dense
# mask-induced covariance matrix. The "naive FFT" fit below intentionally
# ignores those correlations to show what that simplification does.

# %%
noise_realizations = np.stack(
    [
        np.asarray(TimeSeries(NOISE_SIGMA * RNG.normal(size=n_total) * sample_mask, dt=dt).to_wdm(nt=nt).coeffs)
        for _ in range(192)
    ]
)
wdm_variance = noise_realizations.var(axis=0) + 1e-8

full_samples = run_time_nuts(full_data, times, seed=0)
masked_samples = run_time_nuts(full_data[observed_mask], times[observed_mask], seed=1)
fft_samples = run_fft_nuts(gapped_data, sample_mask, seed=2)
wdm_samples = run_wdm_nuts(
    gapped_coeffs,
    wdm_variance,
    kept_time_bins,
    selected_channels,
    seed=3,
)

full_summary = summarize(full_samples)
masked_summary = summarize(masked_samples)
fft_summary = summarize(fft_samples)
wdm_summary = summarize(wdm_samples)

print("Posterior mean ± std")
print(
    f"  full-data time : A={full_summary['A'][0]:.4f}±{full_summary['A'][1]:.4f}, "
    f"f0={full_summary['f0'][0]:.5f}±{full_summary['f0'][1]:.5f}, "
    f"phi={full_summary['phi'][0]:.4f}±{full_summary['phi'][1]:.4f}, "
    f"sigma={full_summary['sigma'][0]:.4f}±{full_summary['sigma'][1]:.4f}"
)
print(
    f"  masked time    : A={masked_summary['A'][0]:.4f}±{masked_summary['A'][1]:.4f}, "
    f"f0={masked_summary['f0'][0]:.5f}±{masked_summary['f0'][1]:.5f}, "
    f"phi={masked_summary['phi'][0]:.4f}±{masked_summary['phi'][1]:.4f}, "
    f"sigma={masked_summary['sigma'][0]:.4f}±{masked_summary['sigma'][1]:.4f}"
)
print(
    f"  naive FFT      : A={fft_summary['A'][0]:.4f}±{fft_summary['A'][1]:.4f}, "
    f"f0={fft_summary['f0'][0]:.5f}±{fft_summary['f0'][1]:.5f}, "
    f"phi={fft_summary['phi'][0]:.4f}±{fft_summary['phi'][1]:.4f}, "
    f"sigma={fft_summary['sigma'][0]:.4f}±{fft_summary['sigma'][1]:.4f}"
)
print(
    f"  WDM kept-bins  : A={wdm_summary['A'][0]:.4f}±{wdm_summary['A'][1]:.4f}, "
    f"f0={wdm_summary['f0'][0]:.5f}±{wdm_summary['f0'][1]:.5f}, "
    f"phi={wdm_summary['phi'][0]:.4f}±{wdm_summary['phi'][1]:.4f}"
)

# %% [markdown]
# The full-data posterior is only a reference. The more important comparison is
# between the masked time-domain fit, the naive FFT fit, and the WDM fit:
#
# - the masked time-domain posterior is the exact answer for this synthetic
#   missing-data problem
# - the naive FFT posterior is what you get if you stay in frequency space but
#   ignore the gap-induced correlations between bins
# - the WDM posterior is the localized approximation
#
# If the masked time and WDM results agree reasonably well, then the WDM
# strategy is doing what we want: using the compact support of the basis to
# isolate the gap and keep the rest of the signal informative.
#
# In this particular toy problem the naive FFT posterior also lands close to
# the exact masked-time answer. That does not mean the gap is harmless in
# frequency space. It means that, for a single strong sinusoid in white noise,
# the diagonal FFT approximation happens to be good enough. The practical
# difference is still that the WDM-side handling is local and easy to express:
# exclude a few affected WDM time bins instead of constructing a dense
# frequency-domain covariance.

# %%
truths = np.array([TRUE_AMPLITUDE, TRUE_FREQUENCY, TRUE_PHASE])
labels = [r"$A$", r"$f_0$", r"$\phi$"]

fig = corner.corner(
    pack_samples(full_samples),
    labels=labels,
    truths=truths,
    color="tab:blue",
    truth_color="black",
    levels=(0.5, 0.9),
    plot_density=False,
    fill_contours=False,
)
corner.corner(
    pack_samples(masked_samples),
    fig=fig,
    color="tab:green",
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
        Line2D([], [], color="tab:blue", label="full-data time"),
        Line2D([], [], color="tab:green", label="masked time"),
        Line2D([], [], color="tab:purple", label="naive FFT"),
        Line2D([], [], color="tab:orange", label="WDM kept-bins"),
    ],
    loc="upper right",
    frameon=False,
)
fig.suptitle("Posterior comparison for a monochromatic signal with a gap", y=1.02)

# %% [markdown]
# ## Reconstructed signal from the posterior means
#
# The posterior means below give a simple visual summary. Both the exact masked
# fit and the WDM fit recover the oscillation through the missing interval,
# because both are fitting a global sinusoidal model even though part of the
# data is absent.

# %%
masked_mean = {
    key: value[0]
    for key, value in masked_summary.items()
}
wdm_mean = {
    key: value[0]
    for key, value in wdm_summary.items()
}
full_mean = {
    key: value[0]
    for key, value in full_summary.items()
}
fft_mean = {
    key: value[0]
    for key, value in fft_summary.items()
}

masked_fit = sinusoid(masked_mean["A"], masked_mean["f0"], masked_mean["phi"], times)
wdm_fit = sinusoid(wdm_mean["A"], wdm_mean["f0"], wdm_mean["phi"], times)
full_fit = sinusoid(full_mean["A"], full_mean["f0"], full_mean["phi"], times)
fft_fit = sinusoid(fft_mean["A"], fft_mean["f0"], fft_mean["phi"], times)

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(times, full_data, color="0.80", linewidth=1.0, label="full noisy data")
ax.plot(times[observed_mask], gapped_data[observed_mask], color="tab:blue", linewidth=1.1, label="observed samples")
ax.axvspan(gap_start * dt, gap_stop * dt, color="tab:red", alpha=0.10, label="missing interval")
ax.plot(times, clean_signal, color="black", linestyle="--", linewidth=1.2, label="true signal")
ax.plot(times, masked_fit, color="tab:green", linewidth=1.3, label="masked time posterior mean")
ax.plot(times, fft_fit, color="tab:purple", linewidth=1.3, label="naive FFT posterior mean")
ax.plot(times, wdm_fit, color="tab:orange", linewidth=1.3, label="WDM posterior mean")
ax.plot(times, full_fit, color="tab:blue", linestyle=":", linewidth=1.2, label="full-data posterior mean")
ax.set_title("Posterior-mean signal estimates across the gap")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
