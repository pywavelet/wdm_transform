"""WDM-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py``, truncates the
time series to a length compatible with the WDM tiling, and performs WDM-domain
Bayesian inference for two Galactic binaries with a diagonal Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Build the WDM representation and calibrate a diagonal noise-variance surface.
3. Print per-source SNR in the A channel.
4. Run joint NumPyro NUTS over both sources on a shared WDM band.
5. Save posterior summaries and diagnostic figures.

Sky position, polarisation, and inclination are held fixed at injected values.
Both sources are fit jointly because their signals overlap in the shared WDM band.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    save_figure,
    trim_frequency_band,
    wrap_phase,
)
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform import TimeSeries
from wdm_transform.transforms import from_time_to_wdm

jax.config.update("jax_enable_x64", True)

FIGURE_OUTPUT_DIR = WDM_ASSET_DIR
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = 800
N_DRAWS = 1000
NT = 128
A_WDM = 1.0 / 3.0
D_WDM = 1.0
N_NOISE_CAL = 64

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
SOURCE_PARAMS = np.asarray(_inj["source_params"], dtype=float)
_inj.close()

# Largest power-of-2 length that is also divisible by 2*NT
n_pow2 = floor_pow2(len(data_At_full))
n_keep = (n_pow2 // (2 * NT)) * (2 * NT)
data_At = data_At_full[:n_keep]
t_obs = n_keep * dt
NF = n_keep // NT

print(f"Loaded injection from {INJECTION_PATH.name}")
print(f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  N = {n_keep}  nt = {NT}  nf = {NF}")

# ── WDM transform and noise surface ──────────────────────────────────────────
probe = TimeSeries(data_At, dt=dt).to_wdm(nt=NT)
data_wdm = np.asarray(probe.coeffs)
freq_grid = np.asarray(probe.freq_grid)
time_grid = np.asarray(probe.time_grid)
freqs_full = np.fft.rfftfreq(n_keep, dt)
noise_psd_full = np.maximum(
    np.interp(freqs_full, freqs_saved, noise_psd_saved, left=noise_psd_saved[0], right=noise_psd_saved[-1]),
    1e-60,
)


def _draw_noise_time_series(noise_psd_rfft: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one real stationary Gaussian noise series with target one-sided PSD."""
    n_freq = len(noise_psd_rfft)
    n_time = 2 * (n_freq - 1)
    df = 1.0 / (n_time * dt)
    coeffs = np.zeros(n_freq, dtype=np.complex128)
    if n_freq > 2:
        amp = np.sqrt(np.maximum(noise_psd_rfft[1:-1], 0.0) / (4.0 * df)) / dt
        coeffs[1:-1] = amp * (
            rng.standard_normal(n_freq - 2) + 1j * rng.standard_normal(n_freq - 2)
        ) / np.sqrt(2.0)
    return np.fft.irfft(coeffs, n=n_time)


def _calibrated_wdm_noise_variance(
    noise_psd_rfft: np.ndarray,
    *,
    nt: int,
    nf: int,
    dt: float,
    a: float,
    d: float,
    n_realizations: int,
    cache_path,
) -> np.ndarray:
    """Estimate diagonal WDM noise variance from synthetic stationary noise draws."""
    if cache_path.exists():
        cached = np.load(cache_path)
        try:
            if (
                int(cached["nt"]) == nt
                and int(cached["nf"]) == nf
                and float(cached["dt"]) == dt
                and int(cached["n_realizations"]) == n_realizations
            ):
                return np.asarray(cached["noise_var"], dtype=float)
        finally:
            cached.close()

    print(f"Calibrating WDM noise variance from {n_realizations} synthetic draws...")
    rng = np.random.default_rng(0)
    var_accum = np.zeros(nf + 1, dtype=float)
    for _ in range(n_realizations):
        noise_t = _draw_noise_time_series(noise_psd_rfft, dt, rng)
        coeffs = np.asarray(
            from_time_to_wdm(noise_t, nt=nt, nf=nf, a=a, d=d, dt=dt, backend="numpy"),
            dtype=float,
        )
        var_accum += np.mean(coeffs**2, axis=0)
    var_row = np.maximum(var_accum / n_realizations, 1e-60)
    noise_var = np.broadcast_to(var_row[None, :], (nt, nf + 1)).copy()
    np.savez(
        cache_path,
        noise_var=noise_var,
        nt=nt,
        nf=nf,
        dt=dt,
        n_realizations=n_realizations,
    )
    return noise_var


noise_var = _calibrated_wdm_noise_variance(
    noise_psd_full,
    nt=NT,
    nf=NF,
    dt=dt,
    a=A_WDM,
    d=D_WDM,
    n_realizations=N_NOISE_CAL,
    cache_path=FIGURE_OUTPUT_DIR / f"noise_var_calibration_nt{NT}_nf{NF}.npz",
)

# Band spanning both sources with padding
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

# ── JaxGB generator ───────────────────────────────────────────────────────────
orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)
n_freqs = n_keep // 2 + 1


def place_local_tdi_jax(segment: jnp.ndarray, kmin, n_freqs: int) -> jnp.ndarray:
    """Place a band-limited TDI segment into a JAX zero-padded frequency array."""
    segment = jnp.asarray(segment, dtype=jnp.complex128).reshape(-1)
    full = jnp.zeros((n_freqs,), dtype=jnp.complex128)
    start = jnp.clip(
        jnp.asarray(kmin, dtype=jnp.int32),
        0,
        max(n_freqs - segment.shape[0], 0),
    )
    return jax.lax.dynamic_update_slice(full, segment, (start,))


def generate_a_time(params: jnp.ndarray) -> jnp.ndarray:
    """Return A-channel time series for a single GB source."""
    params = jnp.asarray(params, dtype=jnp.float64)
    a_loc, _, _ = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    kmin = jnp.asarray(jgb.get_kmin(params[None, 0:1])).reshape(-1)[0]
    a_full = place_local_tdi_jax(a_loc, kmin, n_freqs)
    return jnp.fft.irfft(a_full, n=n_keep)


# ── Per-source SNR ────────────────────────────────────────────────────────────
print("\nPer-source matched-filter SNR (A channel, WDM domain):")
for i, src in enumerate(SOURCE_PARAMS):
    h_t = np.asarray(generate_a_time(jnp.asarray(src, dtype=jnp.float64)))
    h_wdm = np.asarray(
        from_time_to_wdm(h_t, nt=NT, nf=NF, a=A_WDM, d=D_WDM, dt=dt, backend="jax"),
        dtype=float,
    )
    snr_full = matched_filter_snr_wdm(h_wdm, noise_var)
    snr_band = matched_filter_snr_wdm(h_wdm[:, band], noise_var[:, band])
    print(f"  GB {i + 1}: full-grid SNR = {snr_full:.3e}  band SNR = {snr_band:.3e}")

# ── Inference setup ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InferenceSetup:
    fixed_params: np.ndarray        # (n_sources, 8) — full injected param vectors
    data_band_jax: jax.Array        # data_wdm[:, band] pre-converted to JAX
    noise_var_band_jax: jax.Array   # noise_var[:, band] pre-converted to JAX
    nt: int
    nf: int
    dt: float
    t_obs: float
    band: slice
    prior_center: np.ndarray        # (n_sources, 4): [f0, fdot, logA, phi0]
    prior_scale: np.ndarray         # (n_sources, 4): Normal half-widths
    logA_bounds: np.ndarray         # (n_sources, 2): [lo, hi]
    f0_bounds: np.ndarray           # (n_sources, 2): [lo, hi]
    fdot_bounds: np.ndarray         # (n_sources, 2): [lo, hi]


setup = InferenceSetup(
    fixed_params=SOURCE_PARAMS,
    data_band_jax=jnp.asarray(data_wdm[:, band], dtype=jnp.float64),
    noise_var_band_jax=jnp.maximum(
        jnp.asarray(noise_var[:, band], dtype=jnp.float64), 1e-60
    ),
    nt=NT,
    nf=NF,
    dt=dt,
    t_obs=t_obs,
    band=band,
    prior_center=np.array(
        [[src[0], src[1], np.log(src[2]), wrap_phase(src[7])] for src in SOURCE_PARAMS],
        dtype=float,
    ),
    prior_scale=np.array(
        [[2.0e-6, 2.0e-18, 0.35, np.pi / 2.0] for _ in SOURCE_PARAMS],
        dtype=float,
    ),
    logA_bounds=np.array(
        [[np.log(0.3 * src[2]), np.log(3.0 * src[2])] for src in SOURCE_PARAMS],
        dtype=float,
    ),
    f0_bounds=np.array(
        [[src[0] - 8.0e-6, src[0] + 8.0e-6] for src in SOURCE_PARAMS],
        dtype=float,
    ),
    fdot_bounds=np.array(
        [[src[1] - 8.0e-18, src[1] + 8.0e-18] for src in SOURCE_PARAMS],
        dtype=float,
    ),
)

# ── Template and model ────────────────────────────────────────────────────────


def template_wdm_band(theta: jnp.ndarray, setup: InferenceSetup) -> jnp.ndarray:
    """Build WDM template in band by summing all sources.

    Args:
        theta: flat array [f0_0, fdot_0, A_0, phi0_0, f0_1, fdot_1, A_1, phi0_1, ...]
        setup: frozen InferenceSetup

    Returns:
        WDM coefficients in band, shape (nt, band_width).
    """
    n_sources = setup.fixed_params.shape[0]
    signal = jnp.zeros(setup.nt * setup.nf, dtype=jnp.float64)
    for i in range(n_sources):
        params_i = (
            jnp.asarray(setup.fixed_params[i], dtype=jnp.float64)
            .at[0].set(theta[4 * i])
            .at[1].set(theta[4 * i + 1])
            .at[2].set(theta[4 * i + 2])
            .at[7].set(jnp.mod(wrap_phase(theta[4 * i + 3]), 2 * jnp.pi))
        )
        signal = signal + generate_a_time(params_i)
    coeffs = from_time_to_wdm(
        signal, nt=setup.nt, nf=setup.nf, a=A_WDM, d=D_WDM, dt=setup.dt, backend="jax"
    )
    return coeffs[:, setup.band]


def numpyro_wdm_model(setup: InferenceSetup) -> None:
    n_sources = setup.fixed_params.shape[0]
    theta_parts = []
    for i in range(n_sources):
        delta_f = numpyro.sample(f"delta_f{i}", dist.Normal(0.0, 1.0))
        delta_fdot = numpyro.sample(f"delta_fdot{i}", dist.Normal(0.0, 1.0))
        delta_logA = numpyro.sample(f"delta_logA{i}", dist.Normal(0.0, 1.0))
        delta_phi = numpyro.sample(f"delta_phi{i}", dist.Normal(0.0, 1.0))

        f0_i = numpyro.deterministic(
            f"f0_{i}", setup.prior_center[i, 0] + setup.prior_scale[i, 0] * delta_f
        )
        fdot_i = numpyro.deterministic(
            f"fdot_{i}", setup.prior_center[i, 1] + setup.prior_scale[i, 1] * delta_fdot
        )
        logA_i = numpyro.deterministic(
            f"logA_{i}", setup.prior_center[i, 2] + setup.prior_scale[i, 2] * delta_logA
        )
        phi0_i = numpyro.deterministic(
            f"phi0_{i}", setup.prior_center[i, 3] + setup.prior_scale[i, 3] * delta_phi
        )
        A_i = numpyro.deterministic(f"A_{i}", jnp.exp(logA_i))

        numpyro.factor(
            f"f0_{i}_support",
            jnp.where(
                (f0_i >= setup.f0_bounds[i, 0]) & (f0_i <= setup.f0_bounds[i, 1]),
                0.0, -jnp.inf,
            ),
        )
        numpyro.factor(
            f"fdot_{i}_support",
            jnp.where(
                (fdot_i >= setup.fdot_bounds[i, 0]) & (fdot_i <= setup.fdot_bounds[i, 1]),
                0.0, -jnp.inf,
            ),
        )
        numpyro.factor(
            f"logA_{i}_support",
            jnp.where(
                (logA_i >= setup.logA_bounds[i, 0]) & (logA_i <= setup.logA_bounds[i, 1]),
                0.0, -jnp.inf,
            ),
        )
        theta_parts.extend([f0_i, fdot_i, A_i, phi0_i])

    h = template_wdm_band(jnp.stack(theta_parts), setup)
    diff = setup.data_band_jax - h
    numpyro.factor(
        "wdm_whittle",
        -0.5 * jnp.sum(
            diff ** 2 / setup.noise_var_band_jax + jnp.log(2.0 * jnp.pi * setup.noise_var_band_jax)
        ),
    )


# ── Run NUTS ──────────────────────────────────────────────────────────────────
n_sources = SOURCE_PARAMS.shape[0]
init_values = {
    f"delta_{v}{i}": 0.0
    for i in range(n_sources)
    for v in ("f", "fdot", "logA", "phi")
}

print("\nRunning joint NUTS over all sources…")
kernel = NUTS(
    lambda: numpyro_wdm_model(setup),
    init_strategy=init_to_value(values=init_values),
    target_accept_prob=0.95,
)
mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_DRAWS, num_chains=1, progress_bar=True)
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
    time_grid,
    freq_grid,
    np.log(data_wdm ** 2 + 1e-30).T,
    shading="nearest",
    cmap="viridis",
)
ax.axhspan(freq_grid[band.start], freq_grid[band.stop - 1], color="white", alpha=0.08)
ax.set_title("Injected WDM data (A channel)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
fig.colorbar(mesh, ax=ax, label="log local power")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_overview")

theta_med_parts = []
for i in range(n_sources):
    theta_med_parts.extend([
        float(np.median(posterior[f"f0_{i}"])),
        float(np.median(posterior[f"fdot_{i}"])),
        float(np.median(posterior[f"A_{i}"])),
        float(np.median(posterior[f"phi0_{i}"])),
    ])
map_wdm = np.asarray(template_wdm_band(jnp.array(theta_med_parts, dtype=jnp.float64), setup))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
for ax, coeffs, title in [
    (axes[0], data_wdm[:, band], "Data in fitted WDM band"),
    (axes[1], map_wdm, "Posterior median template"),
]:
    mesh = ax.pcolormesh(
        time_grid,
        freq_grid[band],
        np.log(coeffs ** 2 + 1e-30).T,
        shading="nearest",
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    fig.colorbar(mesh, ax=ax, label="log local power")
axes[0].set_ylabel("Frequency [Hz]")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_band_fit")

corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$"]
for i, (samples_i, stem) in enumerate(zip(all_samples, ["gb1_corner", "gb2_corner"])):
    truth_i = [SOURCE_PARAMS[i, 0], SOURCE_PARAMS[i, 1], SOURCE_PARAMS[i, 2], wrap_phase(SOURCE_PARAMS[i, 7])]
    fig = corner.corner(
        samples_i,
        labels=corner_labels,
        truths=truth_i,
        truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
    )
    save_figure(fig, FIGURE_OUTPUT_DIR, stem)
