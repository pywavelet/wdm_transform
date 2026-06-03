"""End-to-end LISA galactic-binary study for a single seed.

Run the whole pipeline for one seed::

    python docs/studies/lisa/lisa_study.py --seed 0

Stages (all driven by ``--seed``):

1. **Data generation** — draw a stationary instrument-noise realization on the
   A/E/T TDI channels and inject one resolved galactic binary.  Print the
   matched-filter SNR and save a data figure with the frequency-domain
   characteristic strain on top and the WDM time-frequency map on the bottom.
2. **Inference** — fit the source in two domains with NumPyro/NUTS:
   a narrow-band frequency-domain Whittle likelihood and a WDM-domain Gaussian
   likelihood.  Both sample ``(delta_f0, delta_fdot, logA, phi0)`` where delta_f0
   and delta_fdot are offsets on the resolution scale around fixed reference
   ``(f0_ref, fdot_ref)`` values.
3. **Comparison** — overlay the two posteriors against the injected truth in a
   corner plot, and write ``results.json`` containing the per-parameter
   Jensen-Shannon divergence and the posterior rank of the truth (the latter
   feeds the PP plot built by ``collect_all_results.py``).

Useful environment overrides: ``LISA_N_WARMUP``, ``LISA_N_DRAWS``,
``LISA_NUM_CHAINS``, ``LISA_NT`` (WDM time bins for inference).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from gb_prior import (
    DELTA_F0_PRIOR_SIGMA,
    DELTA_FDOT_PRIOR_SIGMA,
    F0_JITTER_WIDTH,
    build_local_prior_info,
    draw_source_prior_and_params,
)
from lisa_common import (
    draw_rfft_from_psd,
    freqs_analysis,
    interp_psd_channels,
    lisa_run_dir,
    load_injection,
    noise_tdi15_psd,
    place_local_tdi,
    posterior_rank,
    setup_jax_and_matplotlib,
    trim_frequency_band,
    wrap_phase,
)
from plotting_utils import plot_data_overview, plot_marginal_comparison
from scipy.spatial.distance import jensenshannon

import numpyro
numpyro.set_host_device_count(2)

setup_jax_and_matplotlib()

# ── Configuration ─────────────────────────────────────────────────────────────

T_OBS = 365 * 24 * 3600  # one-year observation, seconds
INJECTION_NORMALIZATION_VERSION = "physical_psd_rfft_v1"

# Target injected SNR range.  The strain scales linearly with the amplitude A,
# so after drawing the source we rescale A to a target SNR drawn uniformly in
# this band.  This pins the injected SNR regardless of the random sky position /
# inclination, and keeps the likelihood peak broad enough for NUTS to sample
# (the sharp-peak stiffness only bites at high SNR).
SNR_MIN = float(os.getenv("LISA_SNR_MIN", "10.0"))
SNR_MAX = float(os.getenv("LISA_SNR_MAX", "30.0"))

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "4500"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
NUM_CHAINS = int(os.getenv("LISA_NUM_CHAINS", "2"))
NT = int(os.getenv("LISA_NT", "32"))  # WDM time bins used for inference
NT_SPECTROGRAM = int(os.getenv("LISA_NT_SPECTROGRAM", "64"))  # WDM time bins for the figure
NUTS_KWARGS = {"dense_mass": True, "target_accept_prob": 0.9, "max_tree_depth": 10}
INIT_JITTER_SCALE = 0.15
A_WDM = 1.0 / 3.0
D_WDM = 1.0

# Sampled / reported source parameters and their display labels.
POSTERIOR_VARS = ("f0", "fdot", "A", "phi0")
POSTERIOR_LABELS = ["log10(f0 / Hz)", "log10(fdot / Hz/s)", "log10(A)", "phi0 [rad]"]
LOG10_VARS = frozenset({"f0", "fdot", "A"})


def run_paths(seed: int) -> dict[str, Path]:
    run_dir = lisa_run_dir(seed=seed)
    return {
        "run_dir": run_dir,
        "injection": run_dir / "injection.npz",
        "freq_posterior": run_dir / "freq_posterior.nc",
        "wdm_posterior": run_dir / "wdm_posterior.nc",
        "results": run_dir / "results.json",
    }


# ── Stage 1: data generation ──────────────────────────────────────────────────


def generate_injection(seed: int, paths: dict[str, Path]) -> None:
    """Draw stationary noise + one resolved GB and write ``injection.npz``."""
    import jax
    import jax.numpy as jnp
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    from wdm_transform.signal_processing import matched_filter_snr_rfft

    jax.config.update("jax_enable_x64", True)
    run_dir = paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"[data] stationary instrument noise + resolved GB (seed={seed})")

    # Analysis band fixes the sampling grid: dt = 1 / (2 f_max).  The sample
    # count must be divisible by 2*nt for both the inference (NT) and figure
    # (NT_SPECTROGRAM) WDM tilings: otherwise the WDM domain truncates the time
    # series, and for red instrument noise that truncation leaks low-frequency
    # power across the analysis band (the frequency domain, which keeps all
    # samples, stays clean — so the two domains would disagree).
    frequencies = freqs_analysis()
    dt = 1.0 / (2.0 * float(np.max(frequencies)))
    wdm_block = int(np.lcm(2 * NT, 2 * NT_SPECTROGRAM))
    n_total = (int(T_OBS / dt) // wdm_block) * wdm_block
    t_obs = n_total * dt
    freqs_all = np.fft.rfftfreq(n_total, dt)
    df_full = 1.0 / t_obs
    n_freqs = len(freqs_all)

    # Stationary noise realization, per channel.
    def noise_fd(channel: int) -> np.ndarray:
        psd = noise_tdi15_psd(channel, freqs_all)
        channel_rng = np.random.default_rng(seed + 10_000 + channel)
        return draw_rfft_from_psd(psd, rng=channel_rng, df=df_full, dt=dt)

    nAf, nEf, nTf = noise_fd(0), noise_fd(1), noise_fd(2)
    psd_A = np.maximum(noise_tdi15_psd(0, freqs_all), 1e-60)
    psd_E = np.maximum(noise_tdi15_psd(1, freqs_all), 1e-60)
    psd_T = np.maximum(noise_tdi15_psd(2, freqs_all), 1e-60)

    # Inject one resolved GB drawn from the local follow-up prior.
    jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
    drawn = draw_source_prior_and_params(rng)
    f0_ref = drawn["f0_ref"]
    delta_logf0_true = drawn["delta_logf0_true"]
    fdot_ref = drawn["fdot_ref"]
    delta_fdot_true = drawn["delta_fdot_true"]
    prior_f0, prior_fdot, prior_A = drawn["prior_f0"], drawn["prior_fdot"], drawn["prior_A"]
    source_params = drawn["source"].reshape(1, -1)
    src = source_params[0]

    params_j = jnp.asarray(src, dtype=jnp.float64)
    a_loc, e_loc, t_loc = jgb.get_tdi(params_j, tdi_generation=1.5, tdi_combination="AET")
    kmin = int(jnp.asarray(jgb.get_kmin(params_j[None, 0:1])).reshape(-1)[0])
    source_Af = place_local_tdi(np.asarray(a_loc), kmin, n_freqs)
    source_Ef = place_local_tdi(np.asarray(e_loc), kmin, n_freqs)
    source_Tf = place_local_tdi(np.asarray(t_loc), kmin, n_freqs)

    snr_A = matched_filter_snr_rfft(source_Af, psd_A, freqs_all, dt=dt)
    snr_E = matched_filter_snr_rfft(source_Ef, psd_E, freqs_all, dt=dt)
    snr_T = matched_filter_snr_rfft(source_Tf, psd_T, freqs_all, dt=dt)
    snr_aet = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))
    if snr_aet <= 0.0:
        raise ValueError("Generated GB template has non-positive SNR.")

    # Rescale the amplitude (strain ∝ A) so the injected SNR lands in the target
    # band, independent of the drawn sky position / inclination.
    target_snr = float(rng.uniform(SNR_MIN, SNR_MAX))
    snr_scale = target_snr / snr_aet
    src[2] *= snr_scale
    source_Af *= snr_scale
    source_Ef *= snr_scale
    source_Tf *= snr_scale
    snr_A *= snr_scale
    snr_E *= snr_scale
    snr_T *= snr_scale
    snr_aet = target_snr
    snr_ae = float(np.hypot(snr_A, snr_E))
    print(
        f"[data] injected GB: f0={src[0]:.5e} Hz fdot={src[1]:.5e} Hz/s "
        f"A={src[2]:.5e} SNR(A+E+T)={snr_aet:.1f}"
    )

    # data = stationary instrument noise + resolved GB signal
    data_At = np.fft.irfft(nAf + source_Af, n=n_total)
    data_Et = np.fft.irfft(nEf + source_Ef, n=n_total)
    data_Tt = np.fft.irfft(nTf + source_Tf, n=n_total)

    np.savez(
        paths["injection"],
        dt=dt,
        t_obs=t_obs,
        seed=np.array(seed, dtype=int),
        freqs=freqs_all,
        noise_psd_A=psd_A,
        noise_psd_E=psd_E,
        noise_psd_T=psd_T,
        data_At=data_At,
        data_Et=data_Et,
        data_Tt=data_Tt,
        source_Af=source_Af,
        source_Ef=source_Ef,
        source_Tf=source_Tf,
        source_params=source_params,
        f0_ref=np.array(f0_ref, dtype=float),
        f0_jitter_width=np.array(F0_JITTER_WIDTH, dtype=float),
        delta_logf0_true=np.array(delta_logf0_true, dtype=float),
        fdot_ref=np.array(fdot_ref, dtype=float),
        delta_fdot_true=np.array(delta_fdot_true, dtype=float),
        prior_f0=np.asarray(prior_f0, dtype=float),
        prior_fdot=np.asarray(prior_fdot, dtype=float),
        prior_A=np.asarray(prior_A, dtype=float),
        source_snrs=np.array([snr_aet], dtype=float),
        source_snrs_ae=np.array([snr_ae], dtype=float),
        source_snrs_aet=np.array([snr_aet], dtype=float),
        normalization_version=np.array(INJECTION_NORMALIZATION_VERSION),
    )
    print(f"[data] saved injection to {paths['injection']}")
    plot_data_overview(
        run_dir,
        dt=dt,
        freqs=freqs_all,
        psd_A=psd_A,
        source_Af=source_Af,
        data_At=data_At,
        f0=float(src[0]),
        nt_spectrogram=NT_SPECTROGRAM,
        a_wdm=A_WDM,
        d_wdm=D_WDM,
    )


# ── Stage 2: inference ────────────────────────────────────────────────────────


def _prior_metadata(injection) -> dict:
    prior = build_local_prior_info(prior_A=injection.prior_A)
    return {
        "f0_ref": float(injection.f0_ref),
        "delta_f0_bounds": (
            float(injection.prior_f0[0] - injection.f0_ref),
            float(injection.prior_f0[1] - injection.f0_ref),
        ),
        "delta_f0_sigma": float(DELTA_F0_PRIOR_SIGMA),
        "fdot_ref": float(injection.fdot_ref),
        "delta_fdot_bounds": (
            float(injection.prior_fdot[0] - injection.fdot_ref),
            float(injection.prior_fdot[1] - injection.fdot_ref),
        ),
        "delta_fdot_sigma": float(DELTA_FDOT_PRIOR_SIGMA),
        "logA_center": prior.logA_center,
        "logA_scale": prior.logA_scale,
        "logA_bounds": prior.logA_bounds,
    }


def _aet_rfft(jgb, params, kmin: int, kmax: int):
    """Local A/E/T Fourier modes embedded into ``[kmin, kmax)`` as ``(3, n_band)``.

    Uses ``get_tdi`` — the same code path as the injection — so that the
    forward model is identical between data generation and inference.
    """
    import jax
    import jax.numpy as jnp

    params = jnp.asarray(params, dtype=jnp.float64)
    a_loc, e_loc, t_loc = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    kmin_nat = jax.lax.stop_gradient(
        jnp.asarray(jgb.get_kmin(params[None, 0:1]), dtype=jnp.int32).reshape(())
    )
    n_band = kmax - kmin
    local = jnp.stack([
        jnp.asarray(a_loc, dtype=jnp.complex128),
        jnp.asarray(e_loc, dtype=jnp.complex128),
        jnp.asarray(t_loc, dtype=jnp.complex128),
    ])
    out = jnp.zeros((3, n_band), dtype=jnp.complex128)
    start = kmin_nat - kmin
    return jax.lax.dynamic_update_slice(out, local, (jnp.zeros((), jnp.int32), start))


def _local_rfft_to_wdm(modes, *, src_kmin, src_kmax, kmin_rfft, band_rfft_size,
                       band_start, band_stop, df_rfft, n_freqs, nf, nt):
    import jax.numpy as jnp

    from wdm_transform.transforms import forward_wdm_band

    local_start = src_kmin - kmin_rfft
    local_end = src_kmax - kmin_rfft

    def one_channel(mode):
        embedded = (
            jnp.zeros(band_rfft_size, dtype=jnp.complex128)
            .at[local_start:local_end]
            .set(mode)
        )
        return forward_wdm_band(
            embedded, df=df_rfft, nfreqs_fourier=n_freqs, kmin=kmin_rfft,
            nfreqs_wdm=nf, ntimes_wdm=nt, mmin=band_start,
            nf_sub_wdm=band_stop - band_start, a=A_WDM, d=D_WDM, backend="jax",
        )

    return jnp.stack([one_channel(mode) for mode in modes])


def _build_frequency_band(injection, source_param, jgb) -> dict:
    data = np.stack([injection.data_At, injection.data_Et, injection.data_Tt])
    freqs = np.fft.rfftfreq(data.shape[1], injection.dt)
    kmin = max(int(np.floor(injection.prior_f0[0] * injection.t_obs)) - jgb.n, 0)
    kmax = min(int(np.ceil(injection.prior_f0[1] * injection.t_obs)) + jgb.n + 1, len(freqs))
    noise_psd = interp_psd_channels(
        freqs, injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    return {
        **_prior_metadata(injection),
        "domain": "freq",
        "fixed_params": source_param.copy(),
        "data": np.fft.rfft(data, axis=1)[:, kmin:kmax],
        "noise_psd": noise_psd[:, kmin:kmax],
        "whittle_weight": 2.0 * (1.0 / injection.t_obs) * injection.dt**2,
        "band_kmin": kmin,
        "band_kmax": kmax,
        "dt": injection.dt,
        "t_obs": injection.t_obs,
    }


def _build_wdm_band(injection, source_param, jgb) -> dict:
    import jax.numpy as jnp

    from wdm_transform.transforms import forward_wdm_band

    n_keep = (len(injection.data_At) // (2 * NT)) * (2 * NT)
    data = np.stack([
        injection.data_At[:n_keep],
        injection.data_Et[:n_keep],
        injection.data_Tt[:n_keep],
    ])
    t_obs = n_keep * injection.dt
    nf = n_keep // NT
    n_freqs = n_keep // 2 + 1
    df_rfft = 1.0 / t_obs
    freq_grid = np.linspace(0.0, 0.5 / injection.dt, nf + 1)
    margin = jgb.n / t_obs
    band_slice = trim_frequency_band(
        freq_grid, injection.prior_f0[0] - margin, injection.prior_f0[1] + margin, pad_bins=2
    )
    half = NT // 2
    kmin_rfft = max((band_slice.start - 1) * half, 0)
    kmax_rfft = min(band_slice.stop * half, n_freqs)
    band_kwargs = {
        "df": df_rfft, "nfreqs_fourier": n_freqs, "kmin": kmin_rfft,
        "nfreqs_wdm": nf, "ntimes_wdm": NT, "mmin": band_slice.start,
        "nf_sub_wdm": band_slice.stop - band_slice.start, "a": A_WDM, "d": D_WDM,
        "backend": "jax",
    }
    data_rfft = np.fft.rfft(data, axis=1)
    band_freqs = freq_grid[band_slice]
    noise_psd = interp_psd_channels(
        band_freqs, injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    N = 2 * (n_freqs - 1)
    wdm_psd = np.broadcast_to(
        N * noise_psd[:, None, :] / (2.0 * injection.dt),
        (noise_psd.shape[0], NT, noise_psd.shape[-1]),
    ).copy()
    return {
        **_prior_metadata(injection),
        "domain": "wdm",
        "fixed_params": source_param.copy(),
        "data": np.stack([
            np.asarray(forward_wdm_band(jnp.asarray(channel[kmin_rfft:kmax_rfft]), **band_kwargs))
            for channel in data_rfft
        ]),
        "noise_psd": wdm_psd,
        "src_kmin": max(int(np.floor(injection.prior_f0[0] * t_obs)) - jgb.n, 0),
        "src_kmax": min(int(np.ceil(injection.prior_f0[1] * t_obs)) + jgb.n + 1, n_freqs),
        "kmin_rfft": kmin_rfft,
        "band_rfft_size": kmax_rfft - kmin_rfft,
        "band_start": band_slice.start,
        "band_stop": band_slice.stop,
        "df_rfft": df_rfft,
        "n_freqs": n_freqs,
        "nf": nf,
        "nt": NT,
        "t_obs": t_obs,
    }


def _build_init_values(band: dict, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    delta_f0_sigma = float(band["delta_f0_sigma"])
    delta_fdot_sigma = float(band["delta_fdot_sigma"])
    zf0_bounds = np.asarray(band["delta_f0_bounds"], dtype=float) / max(delta_f0_sigma, 1e-30)
    zfdot_bounds = np.asarray(band["delta_fdot_bounds"], dtype=float) / max(delta_fdot_sigma, 1e-30)
    logA_bounds = np.asarray(band["logA_bounds"], dtype=float)
    # f0 and fdot are initialized at the truth in their standardized (z) coords,
    # where INIT_JITTER_SCALE is an O(1)-sigma perturbation on the resolution scale.
    return {
        "z_f0": float(np.clip(
            (fixed_params[0] - float(band["f0_ref"])) / max(delta_f0_sigma, 1e-30)
            + INIT_JITTER_SCALE * rng.standard_normal(),
            zf0_bounds[0] + 1e-6, zf0_bounds[1] - 1e-6)),
        "z_fdot": float(np.clip(
            (fixed_params[1] - float(band["fdot_ref"])) / max(delta_fdot_sigma, 1e-30)
            + INIT_JITTER_SCALE * rng.standard_normal(),
            zfdot_bounds[0] + 1e-6, zfdot_bounds[1] - 1e-6)),
        "logA": float(np.clip(
            np.log(fixed_params[2]) + INIT_JITTER_SCALE * rng.standard_normal(),
            logA_bounds[0] + 1e-6, logA_bounds[1] - 1e-6)),
        "phi0": float(np.clip(
            fixed_params[7] + INIT_JITTER_SCALE * rng.standard_normal(),
            -np.pi + 1e-6, np.pi - 1e-6)),
    }


def _domain_loglike(jgb, band, data, noise_psd, fixed_params, f0, fdot, amplitude, phi0):
    """Whittle log-likelihood for one domain at the given source parameters."""
    import jax.numpy as jnp

    params = fixed_params.at[0].set(f0).at[1].set(fdot).at[2].set(amplitude).at[7].set(phi0)
    if band["domain"] == "freq":
        template = _aet_rfft(jgb, params, int(band["band_kmin"]), int(band["band_kmax"]))
        residual = data - template
        residual_power = jnp.real(jnp.conj(residual) * residual)
        return -jnp.sum(jnp.log(noise_psd) + float(band["whittle_weight"]) * residual_power / noise_psd)
    wdm_kwargs = {key: int(band[key]) for key in (
        "src_kmin", "src_kmax", "kmin_rfft", "band_rfft_size",
        "band_start", "band_stop", "n_freqs", "nf", "nt")}
    wdm_kwargs["df_rfft"] = float(band["df_rfft"])
    template = _local_rfft_to_wdm(
        _aet_rfft(jgb, params, wdm_kwargs["src_kmin"], wdm_kwargs["src_kmax"]), **wdm_kwargs)
    residual = data - template
    residual_power = jnp.real(jnp.conj(residual) * residual)
    return -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * noise_psd) + residual_power / noise_psd)


def _fisher_inverse_mass_matrix(jgb, band, data, noise_psd, fixed_params):
    """Inverse mass matrix from the posterior precision at the reference source.

    For an SNR≳10 galactic binary the likelihood peak is far narrower than the
    priors (sub-bin in f0/fdot, ~1/SNR in logA).  Seeding NUTS' dense mass matrix
    with the inverse posterior precision — the likelihood Fisher plus the prior
    precision, evaluated at the reference parameters in the sampled
    ``(z_f0, z_fdot, logA, phi0)`` coordinates — scales its proposals to that
    narrow ridge so it locks onto the peak instead of the diffuse prior.

    The prior term regularises the (otherwise near-degenerate) f0/fdot/A ridge,
    and a small eigenvalue floor guarantees a positive-definite matrix even when
    the raw likelihood Fisher is not (e.g. the WDM domain).
    """
    import jax
    import jax.numpy as jnp

    f0_ref, fdot_ref = float(band["f0_ref"]), float(band["fdot_ref"])
    delta_f0_sigma, delta_fdot_sigma = float(band["delta_f0_sigma"]), float(band["delta_fdot_sigma"])

    def neg_loglike(u):
        loglike = _domain_loglike(
            jgb, band, data, noise_psd, fixed_params,
            f0_ref + delta_f0_sigma * u[0], fdot_ref + delta_fdot_sigma * u[1],
            jnp.exp(u[2]), u[3])
        return -loglike

    u0 = jnp.array([0.0, 0.0, jnp.log(fixed_params[2]), fixed_params[7]], dtype=jnp.float64)
    fisher = np.asarray(jax.hessian(neg_loglike)(u0))
    fisher = 0.5 * (fisher + fisher.T)
    # Floor the eigenvalues to keep the precision positive-definite (the raw WDM
    # Fisher can have a small negative mode) before inverting to a mass matrix.
    eigvals, eigvecs = np.linalg.eigh(fisher)
    eigvals = np.maximum(eigvals, 1e-3 * float(eigvals.max()))
    return np.linalg.inv((eigvecs * eigvals) @ eigvecs.T)


def _run_sampler(jgb, band: dict, seed: int):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    data = jnp.asarray(band["data"], dtype=jnp.complex128 if band["domain"] == "freq" else jnp.float64)
    noise_psd = jnp.asarray(band["noise_psd"], dtype=jnp.float64)
    fixed_params = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    logA_center = float(band["logA_center"])
    logA_scale = float(band["logA_scale"])
    delta_f0_sigma = float(band["delta_f0_sigma"])
    delta_fdot_sigma = float(band["delta_fdot_sigma"])
    zf0_low, zf0_high = np.asarray(band["delta_f0_bounds"], dtype=float) / max(delta_f0_sigma, 1e-30)
    zfdot_low, zfdot_high = np.asarray(band["delta_fdot_bounds"], dtype=float) / max(delta_fdot_sigma, 1e-30)
    logA_low, logA_high = np.asarray(band["logA_bounds"], dtype=float)

    def model() -> None:
        # f0 and fdot are sampled on their (standardized) resolution scales so the
        # razor-sharp chirp likelihood is O(1)-wide in the sampled coordinates.
        z_f0 = numpyro.sample("z_f0", dist.TruncatedNormal(0.0, 1.0, low=float(zf0_low), high=float(zf0_high)))
        delta_f0 = numpyro.deterministic("delta_f0", delta_f0_sigma * z_f0)
        z_fdot = numpyro.sample("z_fdot", dist.TruncatedNormal(0.0, 1.0, low=float(zfdot_low), high=float(zfdot_high)))
        delta_fdot = numpyro.deterministic("delta_fdot", delta_fdot_sigma * z_fdot)
        logA = numpyro.sample("logA", dist.TruncatedNormal(
            logA_center, logA_scale, low=float(logA_low), high=float(logA_high)))
        f0 = float(band["f0_ref"]) + delta_f0
        fdot = float(band["fdot_ref"]) + delta_fdot
        amplitude = jnp.exp(logA)
        phi0 = numpyro.sample("phi0", dist.Uniform(-jnp.pi, jnp.pi))
        loglike = _domain_loglike(jgb, band, data, noise_psd, fixed_params, f0, fdot, amplitude, phi0)
        numpyro.factor(f"{band['domain']}_loglike", loglike)
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", amplitude)

    # Seed the dense mass matrix with the posterior precision at the reference
    # source, then let NUTS refine it during warmup.  Without this the sampler
    # cannot find the sub-bin likelihood peak and collapses toward the prior.
    inverse_mass_matrix = _fisher_inverse_mass_matrix(jgb, band, data, noise_psd, fixed_params)
    nuts = NUTS(model, inverse_mass_matrix=inverse_mass_matrix, adapt_mass_matrix=False, **NUTS_KWARGS)
    mcmc = MCMC(nuts, num_warmup=N_WARMUP, num_samples=N_DRAWS,
                num_chains=NUM_CHAINS, progress_bar=True)
    inits = [_build_init_values(band, seed + 7 * i) for i in range(NUM_CHAINS)]
    mcmc.run(
        jax.random.PRNGKey(seed),
        init_params={key: jnp.array([init[key] for init in inits]) for key in inits[0]},
        extra_fields=("diverging",),
    )
    return mcmc


def run_domain(domain: str, injection, paths: dict[str, Path]) -> None:
    """Sample one domain and save the ``POSTERIOR_VARS`` posterior to NetCDF."""
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    start = time.perf_counter()
    source_param = injection.source_params[0].copy()
    t_obs = injection.t_obs if domain == "freq" else (
        (len(injection.data_At) // (2 * NT)) * (2 * NT) * injection.dt)
    jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
    band = (_build_frequency_band if domain == "freq" else _build_wdm_band)(injection, source_param, jgb)
    mcmc_seed = injection.seed + 10
    print(f"[{domain}] T_obs={float(band['t_obs']) / 86400.0:.1f}d, "
          f"seed={injection.seed}, MCMC_seed={mcmc_seed}, chains={NUM_CHAINS}")

    mcmc = _run_sampler(jgb, band, mcmc_seed)
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    if divergences:
        print(f"[{domain}] WARNING: {divergences} divergences")

    _save_posterior(mcmc, paths[f"{domain}_posterior"])
    print(f"[{domain}] saved posterior to {paths[f'{domain}_posterior']} "
          f"({time.perf_counter() - start:.1f}s)")


def _save_posterior(mcmc, output_path: Path) -> None:
    """Persist the ``POSTERIOR_VARS`` posterior to a NetCDF dataset via arviz_base."""
    import arviz_base as azb

    idata = azb.from_numpyro(mcmc)
    posterior = azb.convert_to_dataset(idata, group="posterior")[list(POSTERIOR_VARS)]
    posterior.attrs.clear()
    for variable in posterior.variables.values():
        variable.attrs.clear()
        variable.encoding.clear()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    posterior.to_netcdf(output_path)


# ── Stage 3: comparison ───────────────────────────────────────────────────────


def _load_posterior_dataset(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as posterior:
        return posterior.load()


def _stack_samples(posterior) -> np.ndarray:
    """Stack POSTERIOR_VARS from an xarray dataset into (n_samples, n_params).

    Columns are in display units: log10 for f0/fdot/A, wrapped phase for phi0.
    """
    cols = []
    for name, label in zip(POSTERIOR_VARS, POSTERIOR_LABELS, strict=True):
        col = np.asarray(posterior[name]).reshape(-1)
        if name in LOG10_VARS:
            col = np.log10(col)
        elif "phi" in label.lower():
            col = wrap_phase(col)
        cols.append(col)
    return np.column_stack(cols)


def _truth_vector(injection) -> np.ndarray:
    src = injection.source_params[0]
    return np.array([np.log10(src[0]), np.log10(src[1]), np.log10(src[2]), wrap_phase(src[7])])


def _jensen_shannon_bits(a: np.ndarray, b: np.ndarray) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    min_range = 64 * max(abs(np.spacing(lo)), abs(np.spacing(hi)))
    if not np.isfinite(lo + hi) or hi - lo <= min_range:
        return 0.0
    counts_a, edges = np.histogram(a, bins=64, range=(lo, hi))
    counts_b, _ = np.histogram(b, bins=edges)
    p = counts_a / max(float(np.sum(counts_a)), 1.0)
    q = counts_b / max(float(np.sum(counts_b)), 1.0)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def run_compare(injection, paths: dict[str, Path]) -> None:
    freq_ds = _load_posterior_dataset(paths["freq_posterior"])
    wdm_ds = _load_posterior_dataset(paths["wdm_posterior"])
    samples_freq = _stack_samples(freq_ds)
    samples_wdm = _stack_samples(wdm_ds)
    truth = _truth_vector(injection)

    per_param = []
    print("[compare] parameter / JSD(bits) / truth-rank (freq, wdm):")
    for idx, label in enumerate(POSTERIOR_LABELS):
        jsd = _jensen_shannon_bits(samples_freq[:, idx], samples_wdm[:, idx])
        rank_freq = posterior_rank(samples_freq[:, idx], truth[idx], label)
        rank_wdm = posterior_rank(samples_wdm[:, idx], truth[idx], label)
        per_param.append({
            "label": label,
            "jsd_bits": jsd,
            "truth": float(truth[idx]),
            "freq_mean": float(np.mean(samples_freq[:, idx])),
            "wdm_mean": float(np.mean(samples_wdm[:, idx])),
            "freq_rank": rank_freq,
            "wdm_rank": rank_wdm,
        })
        print(f"  {label:<22} JSD={jsd:.4e}  rank_freq={rank_freq:.3f} rank_wdm={rank_wdm:.3f}")

    # Carry the stored A+E+T SNR through to the summary.
    with np.load(paths["injection"]) as inj:
        snr_aet = float(np.asarray(inj["source_snrs_aet"]).reshape(-1)[0]) if "source_snrs_aet" in inj else None
    results = {
        "seed": int(injection.seed),
        "snr_aet": snr_aet,
        "parameters": per_param,
    }
    paths["results"].write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[compare] saved results to {paths['results']}")

    plot_marginal_comparison(
        freq_ds,
        wdm_ds,
        truth,
        paths["run_dir"],
        posterior_vars=POSTERIOR_VARS,
        log10_vars=LOG10_VARS,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for this realization.")
    args = parser.parse_args(argv)

    paths = run_paths(args.seed)
    generate_injection(args.seed, paths)

    injection = load_injection(paths["injection"])
    run_domain("freq", injection, paths)
    run_domain("wdm", injection, paths)
    run_compare(injection, paths)


if __name__ == "__main__":
    main()
