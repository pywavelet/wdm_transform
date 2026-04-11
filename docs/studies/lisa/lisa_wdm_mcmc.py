"""WDM-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py``, truncates the
time series to a length compatible with the WDM tiling, and performs WDM-domain
Bayesian inference for two Galactic binaries with a diagonal Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Build the WDM data representation and analytic noise variance S[n,m] = S(f_m)·Δf.
3. Print per-source SNR (WDM band, A channel).
4. Run joint NumPyro NUTS over both sources on a shared WDM band.
5. Print NUTS diagnostics and 90 % CI coverage; save corner plots and posteriors.

Sky position, polarisation, and inclination are held fixed at injected values.
Both sources are fit jointly on a shared WDM band.

Performance notes
-----------------
The forward model calls ``jgb.sum_tdi`` with static (Python-int) kmin/kmax covering
the rfft bins needed by the WDM analysis band.  This lets JAX use static slice
operations (no dynamic_update_slice) and avoids the full irfft + full WDM transform
at every NUTS step.  Only the ``band_width`` WDM channel IFFTs are computed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import corner
import jax
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    INJECTION_PATH,
    WDM_ASSET_DIR,
    check_posterior_coverage,
    floor_pow2,
    matched_filter_snr_wdm,
    print_posterior_summary,
    require_positive_fdot,
    save_figure,
    trim_frequency_band,
    wdm_noise_variance,
    wrap_phase,
)
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform import TimeSeries
from wdm_transform.backends import get_backend
from wdm_transform.windows import phi_window

jax.config.update("jax_enable_x64", True)

FIGURE_OUTPUT_DIR = WDM_ASSET_DIR
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "800"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
NT = 128
A_WDM = 1.0 / 3.0
D_WDM = 1.0

# ── Load and truncate data ────────────────────────────────────────────────────
if not INJECTION_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached injection at {INJECTION_PATH}. "
        "Run data_generation.py first."
    )

_inj = np.load(INJECTION_PATH)
dt = float(_inj["dt"])
data_At_full = np.asarray(_inj["data_At"], dtype=float)
noise_psd_saved = np.asarray(_inj["noise_psd_A"], dtype=float)
freqs_saved = np.asarray(_inj["freqs"], dtype=float)
SOURCE_PARAMS = require_positive_fdot(
    np.asarray(_inj["source_params"], dtype=float),
    context=str(INJECTION_PATH),
)
_inj.close()

# Largest power-of-2 length that is also divisible by 2*NT
n_pow2 = floor_pow2(len(data_At_full))
n_keep = (n_pow2 // (2 * NT)) * (2 * NT)
data_At = data_At_full[:n_keep]
t_obs = n_keep * dt
NF = n_keep // NT
n_freqs = n_keep // 2 + 1  # rfft half-spectrum size

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {n_keep}  nt = {NT}  nf = {NF}"
)

# ── WDM data and analytic noise variance ──────────────────────────────────────
probe = TimeSeries(data_At, dt=dt).to_wdm(nt=NT)
data_wdm = np.asarray(probe.coeffs)
freq_grid = np.asarray(probe.freq_grid)
time_grid = np.asarray(probe.time_grid)

# Interpolate saved PSD onto the WDM frequency grid
noise_psd = np.maximum(
    np.interp(freq_grid, freqs_saved, noise_psd_saved,
               left=noise_psd_saved[0], right=noise_psd_saved[-1]),
    1e-60,
)
# Diagonal noise variance: S[n, m] = S_n(f_m) / (2·dt), tiled over NT time bins
noise_var = wdm_noise_variance(noise_psd, freq_grid, NT)

# WDM analysis band spanning both sources
band = trim_frequency_band(
    freq_grid,
    SOURCE_PARAMS[:, 0].min() - 1.5e-4,
    SOURCE_PARAMS[:, 0].max() + 1.5e-4,
    pad_bins=2,
)
print(
    f"WDM band: [{freq_grid[band.start]:.4e}, {freq_grid[band.stop - 1]:.4e}] Hz  "
    f"({band.stop - band.start} bins)"
)

# ── Band-limited WDM forward model ────────────────────────────────────────────
#
# Instead of irfft(full spectrum) → full WDM transform → slice band, we:
#   1. Call jgb.sum_tdi with static Python-int kmin/kmax for each source's
#      natural ±n-bin window.  Static indices → no dynamic_update_slice.
#   2. Place the result into a small local rfft array covering only the rfft
#      bins that the WDM band channels actually access.
#   3. Run a JIT-compiled kernel that does only band_width IFFTs of size NT
#      instead of the full nf-1 IFFTs.
#
# This eliminates the irfft(N) + fft(N) pair and reduces the per-step IFFT
# count from (nf-1) ≈ 1023 to (band.stop - band.start) ≈ 124.

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)

half = NT // 2  # = 64

# rfft bin range that covers all WDM channels in the analysis band:
#   channel m accesses rfft bins [(m-1)*half, (m+1)*half)
kmin_rfft: int = max((band.start - 1) * half, 0)
kmax_rfft: int = min(band.stop * half, n_freqs)
band_rfft_size: int = kmax_rfft - kmin_rfft

# Source-centered rfft windows — Python ints, static during NUTS
SRC_KMIN: list[int] = [
    max(int(np.rint(src[0] * t_obs)) - jgb.n, 0) for src in SOURCE_PARAMS
]
SRC_KMAX: list[int] = [min(k + 2 * jgb.n, n_freqs) for k in SRC_KMIN]

# WDM phi window (size NT, precomputed once on host)
_jax_backend = get_backend("jax")
_window_j = jnp.asarray(
    phi_window(_jax_backend, NT, NF, dt, A_WDM, D_WDM), dtype=jnp.complex128
)


@partial(jax.jit, static_argnames=("nt", "nf", "kmin_rfft", "band_start", "band_stop"))
def _wdm_band_from_local_rfft(
    x_local: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
    kmin_rfft: int,
    band_start: int,
    band_stop: int,
) -> jnp.ndarray:
    """WDM interior channels [band_start, band_stop) from a local rfft segment.

    ``x_local[i]`` corresponds to global rfft bin ``kmin_rfft + i``.
    Only interior channels lying entirely in the positive-frequency half are
    supported (band channels far from Nyquist), which holds for our sources.

    Returns shape ``(nt, band_stop - band_start)`` of real coefficients.
    """
    _half = nt // 2
    narr = jnp.arange(nt)
    band_m = jnp.arange(band_start, band_stop)              # (band_width,)

    # rfft indices for each band channel, shifted to local coords
    upper = band_m[:, None] * _half + jnp.arange(_half)[None, :]  # (bw, half)
    lower = (band_m[:, None] - 1) * _half + jnp.arange(_half)[None, :]
    mid_local = jnp.concatenate([upper, lower], axis=1) - kmin_rfft  # (bw, nt)

    mid_blocks = x_local[mid_local] * window[None, :]       # (bw, nt) complex
    mid_times = jnp.fft.ifft(mid_blocks, axis=1).T          # (nt, bw) complex

    # C_{n,m}^* phase factor
    parity = jnp.where((narr[:, None] + band_m[None, :]) % 2 == 0, 1.0, -1.0)
    mid_phase = jnp.conj(jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity)))

    return (jnp.sqrt(2.0) / nf) * jnp.real(mid_phase * mid_times)  # (nt, bw)


def generate_a_wdm_band(params: jnp.ndarray, src_idx: int) -> jnp.ndarray:
    """A-channel WDM coefficients in the analysis band for one source.

    Uses ``jgb.sum_tdi`` with static Python-int kmin/kmax — no dynamic
    indexing.  The small local rfft array has size ``band_rfft_size`` rather
    than the full ``n_freqs``.
    """
    kmin_i = SRC_KMIN[src_idx]
    kmax_i = SRC_KMAX[src_idx]
    a_loc, _, _ = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=kmin_i, kmax=kmax_i, tdi_combination="AET",
    )
    # Static slice indices (Python ints → no dynamic_update_slice)
    local_start = kmin_i - kmin_rfft
    local_end = kmax_i - kmin_rfft
    x_local = (
        jnp.zeros(band_rfft_size, dtype=jnp.complex128)
        .at[local_start:local_end]
        .set(jnp.asarray(a_loc, dtype=jnp.complex128))
    )
    return _wdm_band_from_local_rfft(
        x_local, _window_j, NT, NF, kmin_rfft, band.start, band.stop
    )


# ── Per-source SNR ────────────────────────────────────────────────────────────
noise_var_band = noise_var[:, band]

print("\nPer-source matched-filter SNR (A channel, WDM band):")
for i, src in enumerate(SOURCE_PARAMS):
    h_band = np.asarray(generate_a_wdm_band(jnp.asarray(src, dtype=jnp.float64), i))
    snr = matched_filter_snr_wdm(h_band, noise_var_band)
    print(f"  GB {i + 1}: SNR = {snr:.1f}")

# ── Inference setup ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InferenceSetup:
    fixed_params: np.ndarray       # (n_sources, 8) — full injected param vectors
    data_band_jax: jax.Array       # data_wdm[:, band] pre-converted to JAX
    noise_var_band_jax: jax.Array  # noise_var[:, band] pre-converted to JAX
    nt: int
    nf: int
    dt: float
    t_obs: float
    band: slice
    prior_center: np.ndarray       # (n_sources, 3): [logf0, logfdot, logA]
    prior_scale: np.ndarray        # (n_sources, 4): [logf0, logfdot, logA, delta_phi_c]
    phase_ref: np.ndarray          # (n_sources, 2): [phi0_ref, phi_c_ref]
    logA_bounds: np.ndarray        # (n_sources, 2): [lo, hi]
    logf0_bounds: np.ndarray       # (n_sources, 2): [lo, hi]
    logfdot_bounds: np.ndarray     # (n_sources, 2): [lo, hi]


t_c = t_obs / 2.0
phase_ref = []
logf0_bounds = []
logfdot_bounds = []
for src in SOURCE_PARAMS:
    phi0_ref = float(wrap_phase(src[7]))
    phi_c_ref = float(
        wrap_phase(phi0_ref + 2 * np.pi * src[0] * t_c + np.pi * src[1] * t_c**2)
    )
    phase_ref.append([phi0_ref, phi_c_ref])
    f0_lo, f0_hi = src[0] - 8.0e-6, src[0] + 8.0e-6
    fdot_lo, fdot_hi = 0.25 * src[1], 4.0 * src[1]
    logf0_bounds.append([float(np.log(f0_lo)), float(np.log(f0_hi))])
    logfdot_bounds.append([float(np.log(fdot_lo)), float(np.log(fdot_hi))])

setup = InferenceSetup(
    fixed_params=SOURCE_PARAMS,
    data_band_jax=jnp.asarray(data_wdm[:, band], dtype=jnp.float64),
    noise_var_band_jax=jnp.maximum(
        jnp.asarray(noise_var_band, dtype=jnp.float64), 1e-30
    ),
    nt=NT,
    nf=NF,
    dt=dt,
    t_obs=t_obs,
    band=band,
    prior_center=np.array(
        [[np.log(src[0]), np.log(src[1]), np.log(src[2])] for src in SOURCE_PARAMS],
        dtype=float,
    ),
    prior_scale=np.array(
        [
            [
                0.25 * (hi_f0 - lo_f0),
                0.25 * (hi_fdot - lo_fdot),
                0.25 * (np.log(3.0 / 0.3)),
                np.pi / 4.0,
            ]
            for (lo_f0, hi_f0), (lo_fdot, hi_fdot) in zip(
                logf0_bounds, logfdot_bounds, strict=True
            )
        ],
        dtype=float,
    ),
    phase_ref=np.array(phase_ref, dtype=float),
    logA_bounds=np.array(
        [[np.log(0.3 * src[2]), np.log(3.0 * src[2])] for src in SOURCE_PARAMS],
        dtype=float,
    ),
    logf0_bounds=np.array(logf0_bounds, dtype=float),
    logfdot_bounds=np.array(logfdot_bounds, dtype=float),
)

# ── Template and model ────────────────────────────────────────────────────────


def template_wdm_band(theta: jnp.ndarray, setup: InferenceSetup) -> jnp.ndarray:
    """Sum band-limited WDM templates for all sources.

    Args:
        theta: flat array [f0_0, fdot_0, A_0, phi0_0, f0_1, ...] length 4*n_sources
        setup: frozen InferenceSetup

    Returns:
        WDM coefficients in band, shape (nt, band_width).
    """
    n_sources = setup.fixed_params.shape[0]
    h = jnp.zeros((setup.nt, setup.band.stop - setup.band.start), dtype=jnp.float64)
    for i in range(n_sources):
        params_i = (
            jnp.asarray(setup.fixed_params[i], dtype=jnp.float64)
            .at[0].set(theta[4 * i])
            .at[1].set(theta[4 * i + 1])
            .at[2].set(theta[4 * i + 2])
            .at[7].set(theta[4 * i + 3])
        )
        h = h + generate_a_wdm_band(params_i, i)
    return h


def numpyro_wdm_model(setup: InferenceSetup) -> None:
    n_sources = setup.fixed_params.shape[0]
    theta_parts = []
    t_c = setup.t_obs / 2.0
    for i in range(n_sources):
        logf0_i = numpyro.sample(
            f"logf0_{i}",
            dist.TruncatedNormal(
                loc=setup.prior_center[i, 0],
                scale=setup.prior_scale[i, 0],
                low=setup.logf0_bounds[i, 0],
                high=setup.logf0_bounds[i, 1],
            ),
        )
        logfdot_i = numpyro.sample(
            f"logfdot_{i}",
            dist.TruncatedNormal(
                loc=setup.prior_center[i, 1],
                scale=setup.prior_scale[i, 1],
                low=setup.logfdot_bounds[i, 0],
                high=setup.logfdot_bounds[i, 1],
            ),
        )
        logA_i = numpyro.sample(
            f"logA_{i}",
            dist.TruncatedNormal(
                loc=setup.prior_center[i, 2],
                scale=setup.prior_scale[i, 2],
                low=setup.logA_bounds[i, 0],
                high=setup.logA_bounds[i, 1],
            ),
        )
        delta_phi_c_i = numpyro.sample(f"delta_phi_c_{i}", dist.Normal(0.0, 1.0))

        f0_i = numpyro.deterministic(f"f0_{i}", jnp.exp(logf0_i))
        fdot_i = numpyro.deterministic(f"fdot_{i}", jnp.exp(logfdot_i))
        numpyro.deterministic(
            f"phi_c_{i}",
            setup.phase_ref[i, 1] + setup.prior_scale[i, 3] * delta_phi_c_i,
        )
        phi0_i = numpyro.deterministic(
            f"phi0_{i}",
            setup.phase_ref[i, 0]
            + setup.prior_scale[i, 3] * delta_phi_c_i
            - 2 * jnp.pi * (f0_i - jnp.exp(setup.prior_center[i, 0])) * t_c
            - jnp.pi * (fdot_i - jnp.exp(setup.prior_center[i, 1])) * t_c**2,
        )
        A_i = numpyro.deterministic(f"A_{i}", jnp.exp(logA_i))
        theta_parts.extend([f0_i, fdot_i, A_i, phi0_i])

    h = template_wdm_band(jnp.stack(theta_parts), setup)
    diff = setup.data_band_jax - h
    numpyro.factor(
        "wdm_whittle",
        -0.5 * jnp.sum(
            diff ** 2 / setup.noise_var_band_jax
            + jnp.log(2.0 * jnp.pi * setup.noise_var_band_jax)
        ),
    )


# ── Run NUTS ──────────────────────────────────────────────────────────────────
n_sources = SOURCE_PARAMS.shape[0]
init_values = {
    name: value
    for i in range(n_sources)
    for name, value in (
        (f"logf0_{i}", float(setup.prior_center[i, 0])),
        (f"logfdot_{i}", float(setup.prior_center[i, 1])),
        (f"logA_{i}", float(setup.prior_center[i, 2])),
        (f"delta_phi_c_{i}", 0.0),
    )
}

print("\nRunning joint NUTS over all sources…")
kernel = NUTS(
    lambda: numpyro_wdm_model(setup),
    init_strategy=init_to_value(values=init_values),
    dense_mass=True,
    target_accept_prob=0.9,
)
mcmc = MCMC(
    kernel,
    num_warmup=N_WARMUP,
    num_samples=N_DRAWS,
    num_chains=1,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(42), extra_fields=("diverging",))

n_div = int(mcmc.get_extra_fields()["diverging"].sum())
print(f"\nDivergences: {n_div}")
mcmc.print_summary(exclude_deterministic=False)

# ── Posterior summaries and coverage ─────────────────────────────────────────
posterior = mcmc.get_samples()
PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]"]
all_samples: list[np.ndarray] = []

for i in range(n_sources):
    samples_i = np.column_stack([
        np.asarray(posterior[f"f0_{i}"]),
        np.asarray(posterior[f"fdot_{i}"]),
        np.asarray(posterior[f"A_{i}"]),
        np.asarray(posterior[f"phi0_{i}"]),
    ])
    truth_i = np.array([
        SOURCE_PARAMS[i, 0],
        SOURCE_PARAMS[i, 1],
        SOURCE_PARAMS[i, 2],
        wrap_phase(SOURCE_PARAMS[i, 7]),
    ])
    print(f"\n{'═' * 56}  GB {i + 1}")
    print_posterior_summary(samples_i, truth_i, PARAM_NAMES)
    check_posterior_coverage(samples_i, truth_i, PARAM_NAMES)
    all_samples.append(samples_i)

# ── Save posteriors ───────────────────────────────────────────────────────────
_out_path = FIGURE_OUTPUT_DIR / "posteriors.npz"
np.savez(
    _out_path,
    source_params=SOURCE_PARAMS,
    samples_gb1=all_samples[0],
    samples_gb2=all_samples[1],
)
print(f"\nSaved posteriors to {_out_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
mesh = ax.pcolormesh(
    time_grid, freq_grid, np.log(data_wdm ** 2 + 1e-30).T,
    shading="nearest", cmap="viridis",
)
ax.axhspan(freq_grid[band.start], freq_grid[band.stop - 1], color="white", alpha=0.08)
ax.set_title("Injected WDM data (A channel)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
fig.colorbar(mesh, ax=ax, label="log local power")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_overview")

# %% [markdown]
# ![WDM overview](../lisa_wdm_mcmc_assets/wdm_overview.png)

# %%
theta_med_parts = []
for i in range(n_sources):
    theta_med_parts.extend([
        float(np.median(posterior[f"f0_{i}"])),
        float(np.median(posterior[f"fdot_{i}"])),
        float(np.median(posterior[f"A_{i}"])),
        float(np.median(posterior[f"phi0_{i}"])),
    ])
map_wdm = np.asarray(
    template_wdm_band(jnp.array(theta_med_parts, dtype=jnp.float64), setup)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
for ax, coeffs, title in [
    (axes[0], data_wdm[:, band], "Data in fitted WDM band"),
    (axes[1], map_wdm, "Posterior median template"),
]:
    mesh = ax.pcolormesh(
        time_grid, freq_grid[band], np.log(coeffs ** 2 + 1e-30).T,
        shading="nearest", cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    fig.colorbar(mesh, ax=ax, label="log local power")
axes[0].set_ylabel("Frequency [Hz]")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_band_fit")

# %% [markdown]
# ![Band-limited WDM fit](../lisa_wdm_mcmc_assets/wdm_band_fit.png)

# %%
corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$"]
for i, (samples_i, stem) in enumerate(
    zip(all_samples, ["gb1_corner", "gb2_corner"], strict=True)
):
    truth_i = [SOURCE_PARAMS[i, 0], SOURCE_PARAMS[i, 1],
                SOURCE_PARAMS[i, 2], wrap_phase(SOURCE_PARAMS[i, 7])]
    fig = corner.corner(
        samples_i, labels=corner_labels, truths=truth_i, truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95], show_titles=True,
    )
    save_figure(fig, FIGURE_OUTPUT_DIR, stem)

# %% [markdown]
# ![GB 1 corner](../lisa_wdm_mcmc_assets/gb1_corner.png)
# ![GB 2 corner](../lisa_wdm_mcmc_assets/gb2_corner.png)
