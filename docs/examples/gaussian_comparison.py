# %% [markdown]
# # Meyer And Modified-Gaussian Wilson Comparison
#
# This example compares the package's actual Meyer default (`a=1/3,d=1`)
# with the Gaussian default (`nu=0.5,support=16`). The signal combines a
# localized chirp and an impulse. Both coefficient images share one color
# logarithmic normalization with the same lower display floor.
# Reconstructing uses the originating window metadata automatically.
# See [Choosing a window](../guide/choosing-a-window.md) for the tradeoff and
# Noah Pearson et al.'s paper on the modified-Gaussian construction.

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from wdm_transform import TimeSeries

nt, nf, dt = 64, 128, 0.1
n = nt * nf
times = np.arange(n) * dt
center = 0.55 * n * dt
relative = times - center
chirp = np.exp(-((relative / 40.0) ** 2)) * np.cos(
    2 * np.pi * (0.6 * relative + 0.001 * relative**2)
)
signal = chirp.copy()
signal[n // 3] += 3.0
ts = TimeSeries(signal, dt=dt)

meyer = ts.to_wdm(nt=nt, window="meyer")
gaussian = ts.to_wdm(nt=nt, window="gaussian")
results = {"meyer": meyer, "gaussian": gaussian}
errors = {}
for name, coeffs in results.items():
    restored = np.asarray(coeffs.to_time_series().data[0])
    errors[name] = float(np.linalg.norm(restored - signal) / np.linalg.norm(signal))
    print(f"{name}: {coeffs.window=}, {coeffs.nu=}, {coeffs.support=}")
    print(f"  relative L2 reconstruction error: {errors[name]:.3e}")

# %% [markdown]
# The images show squared coefficients divided by N. They share shape and
# coordinates, but each projects onto its own basis. Compact subband functions
# accept the same window settings and use a wider frequency margin for Gaussian.
# Channel-center colored-noise PSD approximations require separate covariance
# validation for this wider frequency window.

# %%
powers = {
    name: np.asarray(coeffs.coeffs[0]) ** 2 / n for name, coeffs in results.items()
}
maximum = max(float(np.max(p)) for p in powers.values())
norm = LogNorm(vmin=maximum * 1e-5, vmax=maximum)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
axes[0].plot(times, signal, lw=0.8)
axes[0].set(xlabel="Time [s]", ylabel="Amplitude", title="Chirp and impulse")
for ax, (name, coeffs) in zip(axes[1:], results.items(), strict=True):
    mesh = ax.pcolormesh(
        np.asarray(coeffs.time_grid),
        np.asarray(coeffs.freq_grid),
        np.maximum(powers[name].T, norm.vmin),
        shading="nearest",
        norm=norm,
        cmap="viridis",
    )
    ax.set(
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
        ylim=(0.0, 1.5),
        title=f"{name.capitalize()} — relative L2 error {errors[name]:.2e}",
    )
fig.colorbar(mesh, ax=list(axes[1:]), label="Squared coefficient / N")

outdir = Path(os.environ.get("WDM_EXAMPLE_OUTDIR", "test_output/gaussian_comparison"))
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "comparison.png", dpi=160)
(outdir / "reconstruction_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
plt.show()
