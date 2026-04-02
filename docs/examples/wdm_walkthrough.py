# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # WDM Walkthrough
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/examples/wdm_walkthrough.py)
#
# This notebook provides a quick walkthrough of the WDM transform API. It demonstrates how to
#
# - create a `TimeSeries`
# - transform it into packed `WDM` coefficients
# - invert back to the time domain
# - compare the reconstruction error
# - show a few demo plots
# - report simple timing numbers

# %%
import subprocess
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

if "google.colab" in sys.modules:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/pywavelet/wdm_transform.git",
        ],
        check=True,
    )

from wdm_transform import TimeSeries, WDM
from wdm_transform.plotting import plot_spectrogram


nt = 32
nf = 16
dt = 0.1
n_total = nt * nf
times = np.arange(n_total) * dt

signal = np.sin(2 * np.pi * times * 0.08) + 0.6 * np.exp(-((times - times.max() / 2.0) ** 2) / (times.max() / 6.0) ** 2) * np.cos(2 * np.pi * times * (0.05 + 0.04 * times / times.max()))

series = TimeSeries(signal, dt=dt)
coeffs = WDM.from_time_series(series, nt=nt)
recovered = coeffs.to_time_series()

max_abs_error = np.max(np.abs(recovered.data - series.data))
coeff_shape = coeffs.shape

print(f"WDM coefficient shape: {coeff_shape}")
print(f"Max abs reconstruction error: {max_abs_error:.3e}")

# %% [markdown]
# The same packed coefficients can also be reconstructed back into the FFT domain:

# %%
original_fft = series.to_frequency_series()
reconstructed_fft = coeffs.to_frequency_series()
fft_error = np.max(np.abs(reconstructed_fft.data - original_fft.data))

print(f"Max abs FFT reconstruction error: {fft_error:.3e}")

# %% [markdown]
# ## Demo plots

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
series.plot(ax=axes[0], color="tab:blue")
axes[0].set_title("Input Time Series")

series.to_frequency_series().plot(ax=axes[1], color="tab:red")
axes[1].set_title("FFT Magnitude")

coeffs.plot(ax=axes[2])
axes[2].set_title("Packed WDM Grid")

fig.tight_layout()

# %% [markdown]
# For comparison, here is a standard spectrogram of the same signal.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
plot_spectrogram(series, ax=ax, spec_kwargs={"nperseg": 64, "noverlap": 48})
ax.set_title("Reference Spectrogram")
fig.tight_layout()

# %% [markdown]
# ## Timing snapshot
#
# These are small local timings from the docs build environment. They are just a quick smoke check,
# not a benchmark suite.

# %%
repeats = 20

start = perf_counter()
for _ in range(repeats):
    coeffs_bench = WDM.from_time_series(series, nt=nt)
forward_avg_ms = (perf_counter() - start) * 1e3 / repeats

start = perf_counter()
for _ in range(repeats):
    _ = coeffs_bench.to_time_series()
inverse_avg_ms = (perf_counter() - start) * 1e3 / repeats

print(f"Average forward transform time over {repeats} runs: {forward_avg_ms:.3f} ms")
print(f"Average inverse transform time over {repeats} runs: {inverse_avg_ms:.3f} ms")
