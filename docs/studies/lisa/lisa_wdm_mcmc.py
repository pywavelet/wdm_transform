"""WDM-domain LISA GB inference with NumPyro.

Loads the cached ``injection.npz`` from ``data_generation.py``, truncates the
time series to a length compatible with the WDM tiling, and performs WDM-domain
Bayesian inference for two Galactic binaries with a diagonal Whittle likelihood.

Workflow:
1. Load ``injection.npz``.
2. Build the WDM data representation and analytic noise variance
   S[n,m] = S(f_m) / (2·dt) = S(f_m)·f_Nyquist.
3. Print per-source SNR (WDM band, A channel).
4. Run independent NumPyro NUTS chains on a narrow WDM band per source.
5. Print NUTS diagnostics and 90 % CI coverage; save corner plots and posteriors.

Sky position, polarisation, and inclination are held fixed at injected values.
The two injected GBs are well-separated in frequency, so each is fit independently.

Performance notes
-----------------
The forward model calls ``jgb.sum_tdi`` with static (Python-int) kmin/kmax covering
the rfft bins for each source's waveform.  This lets JAX use static slice
operations (no dynamic_update_slice) and avoids the full irfft + full WDM transform
at every NUTS step.  Only the narrow per-source ``band_width`` WDM channel IFFTs
are computed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    build_local_prior_info,
    build_sampled_source_params,
    check_posterior_coverage,
    default_local_priors,
    matched_filter_snr_wdm,
    print_posterior_summary,
    require_positive_fdot,
    save_figure,
    save_corner_plot,
    save_posterior_archive,
    source_truth_vector,
    trim_frequency_band,
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
# NT controls the WDM time-frequency tiling used for inference.
#
# Frequency channel spacing: df_wdm = f_Nyquist / NF = NT / (2 * T_obs).
# With NT=128, df_wdm ≈ 2 µHz — the waveform bandwidth (±jgb.n/T_obs ≈ ±8 µHz)
# spans only ~4 WDM channels, giving weak inter-channel f0 discrimination.
# With NT=4, df_wdm ≈ 63 nHz — ~256 WDM channels cover the same bandwidth,
# matching the 512 rfft bins used by the frequency-domain inference and
# recovering the same f0 precision.
NT = int(os.getenv("LISA_NT", "32"))
NT_PLOT = int(os.getenv("LISA_NT_PLOT", "128"))
A_WDM = 1.0 / 3.0
D_WDM = 1.0

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

_lcm = 2 * max(NT, NT_PLOT)
n_keep = (len(data_At_full) // _lcm) * _lcm
data_At = data_At_full[:n_keep]
t_obs = n_keep * dt
NF = n_keep // NT
n_freqs = n_keep // 2 + 1

print(f"Loaded injection from {INJECTION_PATH.name}")
print(
    f"T_obs = {t_obs / 86400:.1f} days  dt = {dt:.2f} s  "
    f"N = {n_keep}  nt = {NT}  nf = {NF}  df_wdm = {0.5 / (dt * NF):.2e} Hz"
)

freq_grid = np.linspace(0.0, 0.5 / dt, NF + 1)
time_grid = np.arange(NT) * (t_obs / NT)
_data_rfft = np.fft.rfft(data_At)

probe_plot = TimeSeries(data_At, dt=dt).to_wdm(nt=NT_PLOT)
data_wdm_plot = np.asarray(probe_plot.coeffs)
freq_grid_plot = np.asarray(probe_plot.freq_grid)
time_grid_plot = np.asarray(probe_plot.time_grid)

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)
half = NT // 2

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
    _half = nt // 2
    narr = jnp.arange(nt)
    band_m = jnp.arange(band_start, band_stop)

    upper = band_m[:, None] * _half + jnp.arange(_half)[None, :]
    lower = (band_m[:, None] - 1) * _half + jnp.arange(_half)[None, :]
    mid_local = jnp.concatenate([upper, lower], axis=1) - kmin_rfft

    mid_blocks = x_local[mid_local] * window[None, :]
    mid_times = jnp.fft.ifft(mid_blocks, axis=1).T

    parity = jnp.where((narr[:, None] + band_m[None, :]) % 2 == 0, 1.0, -1.0)
    mid_phase = jnp.conj(jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity)))

    return (jnp.sqrt(2.0) / nf) * jnp.real(mid_phase * mid_times)


@dataclass(frozen=True)
class WdmBandData:
    label: str
    src_idx: int
    fixed_params: np.ndarray
    band_start: int
    band_stop: int
    kmin_rfft: int
    kmax_rfft: int
    band_rfft_size: int
    src_kmin: int
    src_kmax: int
    data_band: np.ndarray
    noise_var_band: np.ndarray
    t_obs: float
    prior_center: np.ndarray
    prior_scale: np.ndarray
    phase_ref: np.ndarray
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


def build_wdm_band(
    src: np.ndarray,
    label: str,
    src_idx: int,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
) -> WdmBandData:
    margin = jgb.n / t_obs
    band_sl = trim_frequency_band(freq_grid, src[0] - margin, src[0] + margin, pad_bins=2)

    kmin_r = max((band_sl.start - 1) * half, 0)
    kmax_r = min(band_sl.stop * half, n_freqs)

    k_center = int(np.rint(src[0] * t_obs))
    src_kmin = max(k_center - jgb.n, 0)
    src_kmax = min(k_center + jgb.n, n_freqs)

    prior_info = build_local_prior_info(
        src,
        t_obs=t_obs,
        prior_f0=prior_f0,
        prior_fdot=prior_fdot,
        prior_A=prior_A,
    )

    x_data_local = jnp.asarray(_data_rfft[kmin_r:kmax_r], dtype=jnp.complex128)
    data_band = np.asarray(
        _wdm_band_from_local_rfft(
            x_data_local, _window_j, NT, NF, kmin_r, band_sl.start, band_sl.stop,
        )
    )

    band_freqs = freq_grid[band_sl.start:band_sl.stop]
    noise_psd_band = np.maximum(
        np.interp(band_freqs, freqs_saved, noise_psd_saved,
                  left=noise_psd_saved[0], right=noise_psd_saved[-1]),
        1e-60,
    )
    f_nyquist = float(freq_grid[-1])
    noise_var_band = np.tile(noise_psd_band * f_nyquist, (NT, 1))

    return WdmBandData(
        label=label,
        src_idx=src_idx,
        fixed_params=src.copy(),
        band_start=band_sl.start,
        band_stop=band_sl.stop,
        kmin_rfft=kmin_r,
        kmax_rfft=kmax_r,
        band_rfft_size=kmax_r - kmin_r,
        src_kmin=src_kmin,
        src_kmax=src_kmax,
        data_band=data_band,
        noise_var_band=noise_var_band,
        t_obs=t_obs,
        prior_center=prior_info.prior_center,
        prior_scale=prior_info.prior_scale,
        phase_ref=prior_info.phase_ref,
        logf0_bounds=prior_info.logf0_bounds,
        logfdot_bounds=prior_info.logfdot_bounds,
        logA_bounds=prior_info.logA_bounds,
    )


BANDS = [
    build_wdm_band(src, f"GB {i + 1}", i, *default_local_priors(src))
    for i, src in enumerate(SOURCE_PARAMS)
]

del _data_rfft

for wband in BANDS:
    print(
        f"WDM band {wband.label}: "
        f"[{freq_grid[wband.band_start]:.4e}, {freq_grid[wband.band_stop - 1]:.4e}] Hz  "
        f"({wband.band_stop - wband.band_start} channels)"
    )


def generate_a_wdm_for(params: jnp.ndarray, wband: WdmBandData) -> jnp.ndarray:
    a_loc, _, _ = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=wband.src_kmin, kmax=wband.src_kmax, tdi_combination="AET",
    )
    local_start = wband.src_kmin - wband.kmin_rfft
    local_end = wband.src_kmax - wband.kmin_rfft
    x_local = (
        jnp.zeros(wband.band_rfft_size, dtype=jnp.complex128)
        .at[local_start:local_end]
        .set(jnp.asarray(a_loc, dtype=jnp.complex128))
    )
    return _wdm_band_from_local_rfft(
        x_local, _window_j, NT, NF,
        wband.kmin_rfft, wband.band_start, wband.band_stop,
    )


snrs_optimal = []
print("\nPer-source matched-filter SNR (A channel, WDM band):")
for wband in BANDS:
    h_band = np.asarray(
        generate_a_wdm_for(jnp.asarray(wband.fixed_params, dtype=jnp.float64), wband)
    )
    snr = matched_filter_snr_wdm(h_band, wband.noise_var_band)
    snrs_optimal.append(float(snr))
    print(f"  {wband.label}: SNR = {snr:.1f}")


def sample_source_wdm(wband: WdmBandData, *, seed: int = 0) -> MCMC:
    data_j = jnp.asarray(wband.data_band, dtype=jnp.float64)
    noise_var_j = jnp.asarray(wband.noise_var_band, dtype=jnp.float64)
    fixed_params_j = jnp.asarray(wband.fixed_params, dtype=jnp.float64)
    src_kmin = int(wband.src_kmin)
    src_kmax = int(wband.src_kmax)
    local_start = int(wband.src_kmin - wband.kmin_rfft)
    local_end = int(wband.src_kmax - wband.kmin_rfft)
    band_rfft_size = int(wband.band_rfft_size)
    band_start = int(wband.band_start)
    band_stop = int(wband.band_stop)
    kmin_rfft = int(wband.kmin_rfft)

    def generate(params: jnp.ndarray) -> jnp.ndarray:
        a_loc, _, _ = jgb.sum_tdi(
            jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
            kmin=src_kmin,
            kmax=src_kmax,
            tdi_combination="AET",
        )
        x_local = (
            jnp.zeros(band_rfft_size, dtype=jnp.complex128)
            .at[local_start:local_end]
            .set(jnp.asarray(a_loc, dtype=jnp.complex128))
        )
        return _wdm_band_from_local_rfft(
            x_local,
            _window_j,
            NT,
            NF,
            kmin_rfft,
            band_start,
            band_stop,
        )

    def model():
        del_logf0 = numpyro.sample("del_logf0", dist.Normal(0.0, 1.0))
        del_logfdot = numpyro.sample("del_logfdot", dist.Normal(0.0, 1.0))
        del_logA = numpyro.sample("del_logA", dist.Normal(0.0, 1.0))
        del_phi_c = numpyro.sample("del_phi_c", dist.Normal(0.0, 1.0))

        logf0 = wband.prior_center[0] + 2e-7 * del_logf0
        logfdot = wband.prior_center[1] + wband.prior_scale[1] * del_logfdot
        logA = wband.prior_center[2] + (1.0 / 300.0) * del_logA
        dpc = (1.0 / 300.0) * del_phi_c

        numpyro.factor("p0", dist.TruncatedNormal(
            loc=wband.prior_center[0], scale=wband.prior_scale[0],
            low=wband.logf0_bounds[0], high=wband.logf0_bounds[1]).log_prob(logf0))
        numpyro.factor("p1", dist.TruncatedNormal(
            loc=wband.prior_center[1], scale=wband.prior_scale[1],
            low=wband.logfdot_bounds[0], high=wband.logfdot_bounds[1]).log_prob(logfdot))
        numpyro.factor("p2", dist.TruncatedNormal(
            loc=wband.prior_center[2], scale=wband.prior_scale[2],
            low=wband.logA_bounds[0], high=wband.logA_bounds[1]).log_prob(logA))

        f0 = numpyro.deterministic("f0", jnp.exp(logf0))
        fdot = numpyro.deterministic("fdot", jnp.exp(logfdot))
        A = numpyro.deterministic("A", jnp.exp(logA))

        phi0 = numpyro.deterministic(
            "phi0",
            wband.phase_ref[0]
            + wband.prior_scale[3] * dpc
            - 2 * jnp.pi * (f0 - jnp.exp(wband.prior_center[0])) * (wband.t_obs / 2.0)
            - jnp.pi * (fdot - jnp.exp(wband.prior_center[1])) * (wband.t_obs / 2.0) ** 2,
        )

        params = (
            fixed_params_j
            .at[0].set(f0)
            .at[1].set(fdot)
            .at[2].set(A)
            .at[7].set(phi0)
        )
        h = generate(params)
        diff = data_j - h
        numpyro.factor(
            "ll",
            -0.5 * jnp.sum(diff**2 / noise_var_j + jnp.log(2.0 * jnp.pi * noise_var_j)),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values={
            "del_logf0": 0.0,
            "del_logfdot": 0.0,
            "del_logA": 0.0,
            "del_phi_c": 0.0,
        }),
        dense_mass=True,
        target_accept_prob=0.9,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    return mcmc


print("\nRunning independent NUTS per source…")
PARAM_NAMES = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]"]
all_samples: list[np.ndarray] = []

for i, wband in enumerate(BANDS):
    print(f"\n─── {wband.label} ───")
    mcmc = sample_source_wdm(wband, seed=10 + i)

    n_div = int(mcmc.get_extra_fields()["diverging"].sum())
    print(f"  Divergences: {n_div}")
    mcmc.print_summary(exclude_deterministic=False)

    s = mcmc.get_samples()
    samples_i = np.column_stack([
        np.asarray(s["f0"]),
        np.asarray(s["fdot"]),
        np.asarray(s["A"]),
        np.asarray(s["phi0"]),
    ])

    truth_i = source_truth_vector(wband.fixed_params)
    print(f"\n{'═' * 56}  {wband.label}")
    print_posterior_summary(samples_i, truth_i, PARAM_NAMES)
    check_posterior_coverage(samples_i, truth_i, PARAM_NAMES)

    samples_i_full = build_sampled_source_params(wband.fixed_params, samples_i)

    noise_var_j = jnp.asarray(wband.noise_var_band, dtype=jnp.float64)

    @jax.jit
    def _get_snrs(ps, wb=wband):
        def _single_snr(p):
            a_loc_s, _, _ = jgb.sum_tdi(
                p.reshape(1, -1), kmin=wb.src_kmin, kmax=wb.src_kmax,
                tdi_combination="AET",
            )
            local_start = wb.src_kmin - wb.kmin_rfft
            local_end = wb.src_kmax - wb.kmin_rfft
            x_local = (
                jnp.zeros(wb.band_rfft_size, dtype=jnp.complex128)
                .at[local_start:local_end].set(a_loc_s)
            )
            h_wdm = _wdm_band_from_local_rfft(
                x_local, _window_j, NT, NF,
                wb.kmin_rfft, wb.band_start, wb.band_stop,
            )
            return jnp.sqrt(jnp.sum(h_wdm**2 / noise_var_j))
        return jax.vmap(_single_snr)(ps)

    snr_samples = np.asarray(_get_snrs(jnp.array(samples_i_full)))
    samples_i = np.column_stack([samples_i, snr_samples])
    all_samples.append(samples_i)

PARAM_NAMES_SNR = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]", "SNR"]
for i, (wband, samples_i) in enumerate(zip(BANDS, all_samples, strict=True)):
    truth_i = source_truth_vector(wband.fixed_params, snr=snrs_optimal[i])
    print(f"\n{'═' * 56}  {wband.label}")
    print_posterior_summary(samples_i, truth_i, PARAM_NAMES_SNR)
    check_posterior_coverage(samples_i, truth_i, PARAM_NAMES_SNR)

_out_path = save_posterior_archive(
    FIGURE_OUTPUT_DIR,
    source_params=SOURCE_PARAMS,
    all_samples=all_samples,
    snr_optimal=snrs_optimal,
)
print(f"\nSaved posteriors to {_out_path}")

fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
mesh = ax.pcolormesh(
    time_grid_plot, freq_grid_plot, np.log(data_wdm_plot ** 2 + 1e-30).T,
    shading="nearest", cmap="viridis",
)
for wband in BANDS:
    ax.axhspan(
        freq_grid[wband.band_start], freq_grid[wband.band_stop - 1],
        color="white", alpha=0.10, label=wband.label,
    )
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Injected WDM data (A channel)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
fig.colorbar(mesh, ax=ax, label="log local power")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_overview")

fig, axes = plt.subplots(
    len(BANDS), 2, figsize=(12, 4 * len(BANDS)),
    constrained_layout=True, sharey="row",
)
for row, (wband, samples_i) in enumerate(zip(BANDS, all_samples, strict=True)):
    theta_med = np.median(samples_i[:, :4], axis=0)
    params_med = wband.fixed_params.copy()
    params_med[[0, 1, 2, 7]] = [theta_med[0], theta_med[1], theta_med[2],
                                  wrap_phase(theta_med[3]) % (2 * np.pi)]
    map_wdm = np.asarray(
        generate_a_wdm_for(jnp.array(params_med, dtype=jnp.float64), wband)
    )
    band_freq = freq_grid[wband.band_start:wband.band_stop]
    for ax, coeffs, title in [
        (axes[row, 0], wband.data_band, f"{wband.label} — data"),
        (axes[row, 1], map_wdm, f"{wband.label} — posterior median"),
    ]:
        mesh = ax.pcolormesh(
            time_grid, band_freq, np.log(coeffs ** 2 + 1e-30).T,
            shading="nearest", cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        fig.colorbar(mesh, ax=ax, label="log local power")
    axes[row, 0].set_ylabel("Frequency [Hz]")
save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_band_fit")

corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$", "SNR"]
for i, (wband, samples_i, stem) in enumerate(
    zip(BANDS, all_samples, ["gb1_corner", "gb2_corner"], strict=True)
):
    save_corner_plot(
        samples_i,
        truth=source_truth_vector(wband.fixed_params, snr=snrs_optimal[i]),
        output_dir=FIGURE_OUTPUT_DIR,
        stem=stem,
        labels=corner_labels,
    )
