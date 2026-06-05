"""LISA galactic-binary study: frequency-domain vs WDM-domain Bayesian PE.

Per seed: draw a resolved chirping GB (f0, fdot, A, random sky, phi0) and a
target SNR, inject it via JaxGB on the A/E/T TDI channels with LISA colored
instrument noise, and infer (f0, fdot, A, phi0) with NumPyro/NUTS in BOTH the
frequency-domain Whittle likelihood and the WDM-domain likelihood (the library's
``forward_wdm_band`` sub-band code).  The goal is to show the two domains give
the same posterior (small freq-vs-WDM JSD) with reliable convergence on every
seed.

Sampling choices that make it robust:

  * Cartesian amplitude/phase: sample standardized (z_gc, z_gs) where
    g_c = A cos(phi0), g_s = A sin(phi0).  The GB strain is exactly bilinear in
    (g_c, g_s), so the likelihood is exactly Gaussian in them -- no curved
    amplitude/phase ridge, no phi0 +-pi boundary.  The pi/2 quadrature is
    -i * (phi0=0 template), so the waveform is evaluated only once.
  * f0/fdot sampled on the resolution scale (standardized z), dense Fisher mass
    matrix, tight init at the prior-search guess.
  * Injected fdot drifts a few Fourier bins (physical, measurable, keeps the
    f0-fdot ridge short); the analysis band auto-widens to contain the drift.

Run one seed:       python docs/studies/lisa/lisa_gb_study.py --seed 0
Run the population:  ./run_gb_study.sh 100   (then summarize_gb_study.py)

"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lisa_common import (
    draw_rfft_from_psd,
    freqs_analysis,
    noise_tdi15_psd,
    posterior_rank,
    setup_jax_and_matplotlib,
    trim_frequency_band,
    wrap_phase,
)

import numpyro

numpyro.set_host_device_count(2)
setup_jax_and_matplotlib()

# ── Configuration ──────────────

T_OBS = 365 * 24 * 3600  # one-year observation, seconds

N_WARMUP = 1500 
N_DRAWS = 1000
NUM_CHAINS = 2
NT = 32
NBLOCKS = 4096
NUTS_KWARGS = {
    "dense_mass": True,
    "target_accept_prob": 0.9,
    "max_tree_depth": 10,
}
# Tight, Fisher-scaled init: start every chain at truth + a small fraction of the
# (standardized) posterior width, so the chains map the local posterior well
# instead of having to hunt for the mode along the degenerate ridge.
INIT_JITTER_SCALE = 0.05
A_WDM = 1.0 / 3.0
D_WDM = 1.0

# Per-seed SNR is drawn uniformly in [SNR_MIN, SNR_MAX].  The validated clean
# regime is SNR <~ 45 (residual ridge stiffness onsets near ~50 before the
# reparam; the reparam pushes that much higher, but the band stays the default).
SNR_MIN = 20.0
SNR_MAX = 45.0
# Reference carrier frequency (the "prior search guess"): mean of two nearby
# mHz-band galactic-binary sources.  Each seed's injected f0 is drawn around it.
F0_REF = 1.385910e-3
FDOT_REF = 0.0
# Per-seed source f0 is drawn uniformly within +-F0_DRAW_BINS resolution elements
# of F0_REF; fdot within a fraction of its prior box.  The narrow f0 prior is then
# re-centered on each injected f0 (the "prior search gave a good guess" scenario).
F0_DRAW_BINS = 8.0
# Injected fdot drift over the run, in FOURIER bins (random sign).  A GW-driven
# GB has fdot ~ 1e-17..1e-15 Hz/s, i.e. a drift of ~0.1-1 bins over a year and a
# few bins over the full mission -- so a few-bin drift is physical.  fdot is
# measured from the PHASE (detectable at sub-bin drift, ~1/(pi*SNR)); it does NOT
# need to cross WDM channels (~NT/2 bins).  A few-bin drift stays well inside the
# GB band and keeps the f0-fdot ridge short enough for the sampler.
FDOT_DRIFT_BINS = (1.0, 5.0)

# JaxGB waveform settings.  Sky/orientation are drawn per seed and held fixed
# during inference (only f0, fdot, A, phi0 are sampled).
GB_N = 128  # half-bandwidth of the GB response, bins
GB_SKY = {"ra": 2.40, "dec": 0.31, "psi": 3.56, "iota": 0.52}  # fallback fixed angles

# TDI channels analyzed jointly (A/E/T by default).
_CHANNEL_INDEX = {"A": 0, "E": 1, "T": 2}
CHANNEL_NAMES = {0: "A", 1: "E", 2: "T"}
CHANNELS = (0, 1, 2)
# Prior box / standardized-z scales are derived from the baseline resolution
# (f0 ~ 1/T_obs, fdot ~ 1/T_obs^2) so the toy is well-posed at any T_obs.
PRIOR_SIGMA_BINS = 1.0   # standardized-z sigma, in resolution elements
PRIOR_HALF_BINS = 5.0    # default prior box half-width, in resolution elements
F0_PRIOR_HALF_BINS = 1.0
F0_PRIOR_SIGMA_BINS = 0.32
FDOT_PRIOR_HALF_BINS = 40.0
FDOT_PRIOR_SIGMA_BINS = 10.0
CENTER_F0_ON_TRUTH = True
CENTER_FDOT_ON_TRUTH = True


def _prior_scales(t_obs: float, f0_half_bins: float = F0_PRIOR_HALF_BINS) -> dict:
    """Resolution-scaled prior scales.  ``f0_half_bins`` sets the f0 prior box;
    sub-bin (<0.5) freezes the get_kmin placement, multi-bin lets it move.

    Widths are controlled by module constants in resolution-bin units.
    """
    res_f0 = 1.0 / t_obs
    res_fdot = 1.0 / t_obs ** 2
    f0_half = float(f0_half_bins)
    fdot_half = float(FDOT_PRIOR_HALF_BINS)
    # f0 sigma tracks the box so a sub-bin box stays an O(1)-sigma standardized prior.
    f0_sigma = float(min(F0_PRIOR_SIGMA_BINS, f0_half))
    fdot_sigma = float(FDOT_PRIOR_SIGMA_BINS)
    return {
        "delta_f0_sigma": f0_sigma * res_f0,
        "delta_fdot_sigma": fdot_sigma * res_fdot,
        "delta_f0_half": f0_half * res_f0,
        "delta_fdot_half": fdot_half * res_fdot,
    }

POSTERIOR_VARS = ("f0", "fdot", "A", "phi0")
POSTERIOR_LABELS = ["log10(f0 / Hz)", "fdot [Hz/s]", "log10(A)", "phi0 [rad]"]
LOG10_VARS = frozenset({"f0", "fdot", "A"})


def draw_source(rng: np.random.Generator, grid: dict) -> dict:
    """Draw one injected source: f0/fdot near the reference, random sky/phase.

    Sampled parameters (f0, fdot, A, phi0) get an injected truth; sky angles
    (ra, dec, psi, iota) are drawn per seed and held fixed during inference
    (not sampled).  A is a placeholder (1.0) and is
    rescaled later to the per-seed target SNR.
    """
    res_f0, res_fdot = 1.0 / grid["t_obs"], 1.0 / grid["t_obs"] ** 2
    # fdot magnitude = (drift in WDM channels) * (NT/2) resolution elements, random sign.
    drift_bins = rng.uniform(*FDOT_DRIFT_BINS) * float(rng.choice([-1.0, 1.0]))
    return {
        "f0": F0_REF + rng.uniform(-F0_DRAW_BINS, F0_DRAW_BINS) * res_f0,
        "fdot": FDOT_REF + drift_bins * res_fdot,
        "A": 1.0,
        "phi0": float(rng.uniform(-np.pi, np.pi)),
        "sky": {
            "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
            "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
            "psi": float(rng.uniform(0.0, np.pi)),
            "iota": float(np.arccos(rng.uniform(-1.0, 1.0))),
        },
    }


# ── Grid / data generation ─────────────────────────────────────────────────────


def _make_grid() -> dict:
    """Sampling grid.

    ``n_total`` is chosen as the largest power-of-two multiple of the WDM block
    ``2*NT`` that fits in ``T_OBS``.  This keeps ``dt`` and the banding identical
    to the real study while making the per-evaluation rFFT a pure power-of-two
    transform (the natural ``floor(T_OBS/dt)`` length carries a large prime
    factor that makes every gradient FFT several times slower).
    """
    frequencies = freqs_analysis()
    dt = 1.0 / (2.0 * float(np.max(frequencies)))
    wdm_block = 2 * NT
    max_blocks = int(T_OBS / dt) // wdm_block
    n_blocks = min(NBLOCKS, 1 << int(np.floor(np.log2(max_blocks))))
    n_total = n_blocks * wdm_block
    t_obs = n_total * dt
    n_freqs = n_total // 2 + 1
    return {
        "dt": dt,
        "n_total": n_total,
        "t_obs": t_obs,
        "n_freqs": n_freqs,
        "df": 1.0 / t_obs,
    }


# ── JaxGB A/E/T waveform ────────────────────────────────────────────────────────


def make_jgb(grid: dict):
    """Build a JaxGB instance on the analysis grid."""
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    return JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=grid["t_obs"], t0=0.0, n=GB_N)


def _gb_params(xp, f0, fdot, amplitude, phi0, sky=None):
    """Assemble the 8-vector [f0, fdot, A, ra, dec, psi, iota, phi0]."""
    s = sky if sky is not None else GB_SKY
    return xp.stack([f0, fdot, amplitude,
                     xp.asarray(s["ra"]), xp.asarray(s["dec"]),
                     xp.asarray(s["psi"]), xp.asarray(s["iota"]), phi0])


def gb_full_rfft(jgb, grid: dict, f0, fdot, amplitude, phi0, *, sky=None):
    """Per-channel GB response embedded into a full one-sided rFFT (JAX).

    Returns ``(n_chan, n_freqs)`` for the configured ``CHANNELS``.  The local
    modes are placed at ``get_kmin(f0)`` (stop_gradient'd, integer); the band is
    wide enough that the few-bin chirp drift stays inside it."""
    import jax
    import jax.numpy as jnp

    params = _gb_params(jnp, f0, fdot, amplitude, phi0, sky)
    aet = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    locs = jnp.stack([jnp.asarray(aet[ch], dtype=jnp.complex128) for ch in CHANNELS])
    start = jax.lax.stop_gradient(jnp.asarray(jgb.get_kmin(params[None, 0:1]), dtype=jnp.int32).reshape(()))
    full = jnp.zeros((len(CHANNELS), grid["n_freqs"]), dtype=jnp.complex128)
    return jax.lax.dynamic_update_slice(full, locs, (jnp.zeros((), jnp.int32), start))


def gb_full_rfft_np(jgb, grid: dict, f0, fdot, amplitude, phi0, sky=None):
    """NumPy embedding of the per-channel GB response (data generation / SNR)."""
    import jax.numpy as jnp

    from lisa_common import place_local_tdi

    params = jnp.asarray(np.asarray(_gb_params(np, f0, fdot, amplitude, phi0, sky), dtype=float))
    aet = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    kmin = int(np.asarray(jgb.get_kmin(params[None, 0:1])).reshape(-1)[0])
    full = np.stack([place_local_tdi(np.asarray(aet[ch]), kmin, grid["n_freqs"]) for ch in CHANNELS])
    return full, kmin


def _full_rfft(xp, band: dict, f0, fdot, amplitude, phi0):
    """GB per-channel full one-sided rFFT, ``(n_chan, n_freqs)``."""
    return gb_full_rfft(band["jgb"], band["grid"], f0, fdot, amplitude, phi0, sky=band["sky"])


def _optimal_snr_sq(template_rfft: np.ndarray, psd, grid: dict, band: slice | None = None) -> float:
    """One-sided matched-filter SNR^2 for a single channel (scalar or array PSD).

    ``band`` restricts the integral to the analysis bins the likelihood actually
    uses.  This matters because a real (cosine) chirp has a small DC/low-f
    spectral leakage, and the LISA PSD is floored near f=0 — without banding the
    DC bin's huge 1/PSD weight swamps the in-band signal and the recoverable SNR
    is mis-estimated by orders of magnitude.
    """
    weight = 4.0 * grid["df"] * grid["dt"] ** 2
    integrand = np.atleast_2d(np.abs(template_rfft) ** 2 / psd)  # (n_chan, n_freqs)
    integrand[..., 0] *= 0.5
    if grid["n_total"] % 2 == 0:
        integrand[..., -1] *= 0.5
    if band is not None:
        integrand = integrand[..., band]
    return float(weight * np.sum(integrand))  # summed over channels + freq


# ── Band builders (single channel) ──────────────────────────


def _signal_margin(grid: dict, *, fdot: float, fdot_half: float) -> float:
    """Half-width (Hz) the analysis band must add around the f0 prior.

    Covers the GB intrinsic bandwidth (~GB_N bins) PLUS the chirp's frequency
    excursion ``fdot * T_obs`` over the run, widened by the fdot prior range so a
    sampled fdot anywhere in the prior keeps the whole track inside the band.
    Without the drift term a fast chirp drifts out of the band mid-observation
    and the WDM likelihood is corrupted."""
    drift = (abs(fdot) + fdot_half) * grid["t_obs"]
    return (GB_N + 4) / grid["t_obs"] + drift


def _band_slices(grid: dict, f0_lo: float, f0_hi: float) -> dict:
    """WDM channel slice + matching rFFT crop for the prior f0 band."""
    nf = grid["n_total"] // NT
    freq_grid = np.linspace(0.0, 0.5 / grid["dt"], nf + 1)
    band_slice = trim_frequency_band(freq_grid, f0_lo, f0_hi, pad_bins=2)
    half = NT // 2
    kmin_rfft = max((band_slice.start - 1) * half, 0)
    kmax_rfft = min(band_slice.stop * half, grid["n_freqs"])
    return {
        "nf": nf,
        "band_start": band_slice.start,
        "band_stop": band_slice.stop,
        "kmin_rfft": kmin_rfft,
        "kmax_rfft": kmax_rfft,
    }


def _wdm_kwargs(grid: dict, bands: dict) -> dict:
    return {
        "df": grid["df"],
        "nfreqs_fourier": grid["n_freqs"],
        "kmin": bands["kmin_rfft"],
        "nfreqs_wdm": bands["nf"],
        "ntimes_wdm": NT,
        "mmin": bands["band_start"],
        "nf_sub_wdm": bands["band_stop"] - bands["band_start"],
        "a": A_WDM,
        "d": D_WDM,
        "backend": "jax",
    }


def _psd_full(grid: dict) -> np.ndarray:
    """Per-channel one-sided LISA TDI PSD on the full rFFT grid: ``(n_chan, n_freqs)``
    (A and E share a model; T differs)."""
    freqs = np.fft.rfftfreq(grid["n_total"], grid["dt"])
    return np.stack([np.maximum(noise_tdi15_psd(ch, freqs), 1e-60) for ch in CHANNELS])


def _wdm_transform(crop_rfft, wdm_kwargs):
    import jax.numpy as jnp

    from wdm_transform.transforms import forward_wdm_band

    return np.asarray(forward_wdm_band(jnp.asarray(crop_rfft), **wdm_kwargs))


def _draw_noise_rfft(psd_full, *, rng, df, dt):
    """Per-channel colored/white rFFT noise draw: ``(n_chan, n_freqs)``."""
    return np.stack([draw_rfft_from_psd(psd_ch, rng=rng, df=df, dt=dt) for psd_ch in psd_full])


def _analytic_wdm_variance(grid, bands, psd_full):
    """N * S_c(f_m) / (2 dt) at each WDM channel center, per channel."""
    nf = bands["nf"]
    freq_grid = np.linspace(0.0, 0.5 / grid["dt"], nf + 1)
    centers = freq_grid[bands["band_start"]:bands["band_stop"]]
    full_freqs = np.fft.rfftfreq(grid["n_total"], grid["dt"])
    out = []
    for psd_ch in psd_full:
        psd_centers = np.interp(centers, full_freqs, psd_ch)
        var_per_channel = grid["n_total"] * psd_centers / (2.0 * grid["dt"])
        out.append(np.broadcast_to(var_per_channel[None, :], (NT, centers.size)).copy())
    return np.stack(out)


def build_band(grid: dict, truth: dict, seed: int, jgb) -> dict:
    """Construct freq + WDM data, noise variances and band metadata."""
    scales = _prior_scales(grid["t_obs"])
    # The f0/fdot priors are centered on the injected (prior-search) values.
    f0_ref = truth["f0"] if CENTER_F0_ON_TRUTH else F0_REF
    fdot_ref = truth["fdot"] if CENTER_FDOT_ON_TRUTH else FDOT_REF
    prior_f0 = (truth["f0"] - scales["delta_f0_half"], truth["f0"] + scales["delta_f0_half"])
    psd_full = _psd_full(grid)

    # data = JaxGB AET signal + colored noise drawn via the physical PSD convention
    # (E[2 df dt^2 |X|^2] = S), round-tripped through the time domain, per channel.
    signal_rfft, _ = gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], truth["A"], truth["phi0"], truth.get("sky"))
    rng = np.random.default_rng(seed + 555)
    noise_rfft = _draw_noise_rfft(psd_full, rng=rng, df=grid["df"], dt=grid["dt"])
    data_rfft = np.fft.rfft(np.fft.irfft(signal_rfft + noise_rfft, n=grid["n_total"], axis=-1), axis=-1)

    # Band wide enough for the GB bandwidth + the chirp's frequency drift.
    sig_margin = _signal_margin(grid, fdot=truth["fdot"], fdot_half=scales["delta_fdot_half"])
    bands = _band_slices(grid, prior_f0[0] - sig_margin, prior_f0[1] + sig_margin)
    wdm_kwargs = _wdm_kwargs(grid, bands)
    f_lo_k, f_hi_k = bands["kmin_rfft"], bands["kmax_rfft"]

    wdm_data = np.stack([_wdm_transform(data_rfft[c, f_lo_k:f_hi_k], wdm_kwargs)
                         for c in range(data_rfft.shape[0])])
    wdm_var = _analytic_wdm_variance(grid, bands, psd_full)

    common = {
        "f0_ref": f0_ref,
        "fdot_ref": fdot_ref,
        "delta_f0_sigma": scales["delta_f0_sigma"],
        "delta_fdot_sigma": scales["delta_fdot_sigma"],
        "amp_scale": float(truth["A"]),
        "grid": grid,
        "truth": truth,
        "jgb": jgb,
        "sky": truth.get("sky"),
    }
    freq_band = {
        **common,
        "domain": "freq",
        "data": data_rfft[:, f_lo_k:f_hi_k],
        "band_kmin": f_lo_k,
        "band_kmax": f_hi_k,
        "noise_var": psd_full[:, f_lo_k:f_hi_k],
        "whittle_weight": 2.0 * grid["df"] * grid["dt"] ** 2,
    }
    wdm_band = {
        **common,
        "domain": "wdm",
        "data": wdm_data,
        "wdm_kwargs": wdm_kwargs,
        "kmin_rfft": bands["kmin_rfft"],
        "kmax_rfft": bands["kmax_rfft"],
        "noise_var": wdm_var,
    }
    return {"freq": freq_band, "wdm": wdm_band}


# ── Likelihoods (A/E/T, Cartesian amplitude/phase) ──────────────────────────────


def _quad_templates(band, f0, fdot):
    """The two quadrature templates W0=W(phi0=0), W90=W(phi0=pi/2), unit amplitude.

    The GB strain is exactly bilinear: W(A, phi0) = g_c*W0 + g_s*W90 with
    g_c=A*cos(phi0), g_s=A*sin(phi0).  Sampling (g_c, g_s) makes the likelihood
    exactly Gaussian in them (no curved amplitude/phase ridge, no phi0 boundary).

    The band template is analytic (positive-frequency only), so the pi/2
    quadrature is exactly ``W90 = -i * W0`` in the rFFT domain -- so we evaluate
    the (expensive) GB waveform ONCE and derive both quadratures."""
    import jax.numpy as jnp

    from wdm_transform.transforms import forward_wdm_band

    w0_full = _full_rfft(jnp, band, f0, fdot, 1.0, 0.0)  # (n_chan, n_freqs)
    if band["domain"] == "freq":
        t0 = w0_full[:, band["band_kmin"]:band["band_kmax"]]
        return t0, -1j * t0
    crop = w0_full[:, band["kmin_rfft"]:band["kmax_rfft"]]
    t0 = jnp.stack([forward_wdm_band(crop[c], **band["wdm_kwargs"]) for c in range(crop.shape[0])])
    t90 = jnp.stack([forward_wdm_band(-1j * crop[c], **band["wdm_kwargs"]) for c in range(crop.shape[0])])
    return t0, t90


def domain_loglike(band, f0, fdot, g_c, g_s):
    """Whittle log-likelihood summed over channels, bilinear amplitude/phase."""
    import jax.numpy as jnp

    t0, t90 = _quad_templates(band, f0, fdot)
    template = g_c * t0 + g_s * t90
    residual = band["data"] - template
    if band["domain"] == "freq":
        power = jnp.real(jnp.conj(residual) * residual)
        return -jnp.sum(float(band["whittle_weight"]) * power / band["noise_var"])
    return -0.5 * jnp.sum(residual * residual / band["noise_var"])


# ── Sampler (standardized z + Fisher dense mass matrix) ─────


def _fisher_inverse_mass_matrix(band):
    import jax
    import jax.numpy as jnp

    f0_ref, fdot_ref = float(band["f0_ref"]), float(band["fdot_ref"])
    s_f0, s_fdot = float(band["delta_f0_sigma"]), float(band["delta_fdot_sigma"])
    truth = band["truth"]

    amp = float(band["amp_scale"])  # standardize g_c,g_s by the source amplitude

    def neg_loglike(u):  # u = (z_f0, z_fdot, z_gc, z_gs), all O(1)
        return -domain_loglike(band, f0_ref + s_f0 * u[0], fdot_ref + s_fdot * u[1],
                               amp * u[2], amp * u[3])

    u0 = jnp.array(
        [
            (truth["f0"] - f0_ref) / s_f0,
            (truth["fdot"] - fdot_ref) / s_fdot,
            np.cos(truth["phi0"]),   # g_c/amp = A cos(phi0)/A
            np.sin(truth["phi0"]),   # g_s/amp = A sin(phi0)/A
        ],
        dtype=jnp.float64,
    )
    fisher = np.asarray(jax.hessian(neg_loglike)(u0))
    fisher = 0.5 * (fisher + fisher.T)
    eigvals, eigvecs = np.linalg.eigh(fisher)
    eigvals = np.maximum(eigvals, 1e-3 * float(eigvals.max()))
    return np.linalg.inv((eigvecs * eigvals) @ eigvecs.T)


def _init_values(band, seed: int) -> dict[str, float]:
    """Init every chain at truth + a small jitter, all in O(1) standardized coords."""
    rng = np.random.default_rng(seed)
    truth = band["truth"]
    s_f0, s_fdot = float(band["delta_f0_sigma"]), float(band["delta_fdot_sigma"])
    return {
        "z_f0": float((truth["f0"] - band["f0_ref"]) / s_f0 + INIT_JITTER_SCALE * rng.standard_normal()),
        "z_fdot": float((truth["fdot"] - band["fdot_ref"]) / s_fdot + INIT_JITTER_SCALE * rng.standard_normal()),
        "z_gc": float(np.cos(truth["phi0"]) + INIT_JITTER_SCALE * rng.standard_normal()),
        "z_gs": float(np.sin(truth["phi0"]) + INIT_JITTER_SCALE * rng.standard_normal()),
    }


def run_domain(band, seed: int):
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    s_f0, s_fdot = float(band["delta_f0_sigma"]), float(band["delta_fdot_sigma"])
    g_scale = float(band["amp_scale"])

    # Cartesian amplitude/phase: sample g_c = A cos(phi0), g_s = A sin(phi0)
    # instead of (logA, phi0).  The GB strain is exactly bilinear in (g_c, g_s),
    # so the likelihood is exactly Gaussian in them -- no curved amplitude/phase
    # ridge and no phi0 +-pi boundary.  z_f0, z_fdot keep unbounded Normal priors
    # so the unconstrained space equals the natural space the Fisher metric lives in.
    def model():
        z_f0 = numpyro.sample("z_f0", dist.Normal(0.0, 1.0))
        z_fdot = numpyro.sample("z_fdot", dist.Normal(0.0, 1.0))
        z_gc = numpyro.sample("z_gc", dist.Normal(0.0, 3.0))  # g_c = amp * z_gc
        z_gs = numpyro.sample("z_gs", dist.Normal(0.0, 3.0))
        f0 = band["f0_ref"] + s_f0 * z_f0
        fdot = band["fdot_ref"] + s_fdot * z_fdot
        g_c, g_s = g_scale * z_gc, g_scale * z_gs
        numpyro.factor(f"{band['domain']}_loglike", domain_loglike(band, f0, fdot, g_c, g_s))
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", g_scale * jnp.sqrt(z_gc ** 2 + z_gs ** 2))
        numpyro.deterministic("phi0", jnp.arctan2(z_gs, z_gc))

    inv_mass = _fisher_inverse_mass_matrix(band)
    nuts = NUTS(model, inverse_mass_matrix=inv_mass, adapt_mass_matrix=False, **NUTS_KWARGS)
    mcmc = MCMC(nuts, num_warmup=N_WARMUP, num_samples=N_DRAWS, num_chains=NUM_CHAINS, progress_bar=True)
    inits = [_init_values(band, seed + 7 * i) for i in range(NUM_CHAINS)]
    mcmc.run(
        jax.random.PRNGKey(seed),
        init_params={k: jnp.array([init[k] for init in inits]) for k in inits[0]},
        extra_fields=("diverging",),
    )
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    if div:
        print(f"[{band['domain']}] WARNING: {div} divergences")
    return mcmc


# ── Reporting ──────────────────────────────────────────────────────────────────


def _samples(mcmc) -> dict[str, np.ndarray]:
    s = mcmc.get_samples()
    return {k: np.asarray(s[k]).reshape(-1) for k in POSTERIOR_VARS}


def _transform_marginal(name: str, label: str, col: np.ndarray, t: float):
    """Map raw (param, truth) to the reported marginal: log10 for f0/A, linear
    fdot, wrapped phi0.  Used by both the summary and the saved samples so the
    stored marginals match the rank/JSD definitions exactly."""
    if name in LOG10_VARS and name != "fdot":
        return np.log10(col), float(np.log10(t))
    if "phi" in label.lower():
        return wrap_phase(col), float(wrap_phase(t))
    return col, float(t)


def _marginal_samples(samples: dict, truth: dict) -> dict[str, list]:
    """Per-label transformed posterior samples (thinned) for exact JSD."""
    out = {}
    for name, label in zip(POSTERIOR_VARS, POSTERIOR_LABELS):
        col, _ = _transform_marginal(name, label, samples[name], truth[name])
        step = max(1, col.size // 1000)  # cap at ~1000 samples per marginal
        out[label] = [float(v) for v in col[::step]]
    return out


def _summarize(domain: str, samples: dict, truth: dict) -> list[dict]:
    rows = []
    print(f"[{domain}] param / truth / mean / z-score / rank:")
    for name, label in zip(POSTERIOR_VARS, POSTERIOR_LABELS):
        col, t = _transform_marginal(name, label, samples[name], truth[name])
        mean, std = float(np.mean(col)), float(np.std(col))
        z = (mean - t) / std if std > 0 else 0.0
        # Circular rank for phi0 (correct near +-pi); plain CDF rank otherwise.
        rank = posterior_rank(col, t, label)
        rows.append({"label": label, "truth": float(t), "mean": mean, "std": std, "z": z, "rank": rank})
        print(f"  {label:<20} {t:+.4f}  {mean:+.4f}  z={z:+.2f}  rank={rank:.3f}")
    return rows


def run_one_seed(seed: int, *, grid: dict, jgb) -> dict:
    """Draw one source + noise realization, fit freq and WDM, return results.

    The source (f0, fdot, A, sky, phi0) and target SNR are drawn from ``seed``;
    A is rescaled to the per-seed target SNR; the f0/fdot priors are centered on
    the injected values (the prior-search-guess scenario) via build_band.
    """
    rng = np.random.default_rng(seed)
    truth = draw_source(rng, grid)
    target_snr = float(rng.uniform(SNR_MIN, SNR_MAX))

    psd_full = _psd_full(grid)
    scales = _prior_scales(grid["t_obs"])
    prior_f0 = (truth["f0"] - scales["delta_f0_half"], truth["f0"] + scales["delta_f0_half"])
    sig_margin = _signal_margin(grid, fdot=truth["fdot"], fdot_half=scales["delta_fdot_half"])
    bs = _band_slices(grid, prior_f0[0] - sig_margin, prior_f0[1] + sig_margin)
    ref_rfft, _ = gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], 1.0, truth["phi0"], truth["sky"])
    snr0 = np.sqrt(_optimal_snr_sq(ref_rfft, psd_full, grid, band=slice(bs["kmin_rfft"], bs["kmax_rfft"])))
    truth["A"] = target_snr / snr0

    print(f"[seed {seed}] SNR={target_snr:.1f} f0={truth['f0']:.6e} "
          f"fdot={truth['fdot']:+.2e} A={truth['A']:.2e} phi0={truth['phi0']:+.2f}")
    bands = build_band(grid, truth, seed, jgb)

    results = {"seed": seed, "snr": target_snr, "config": _run_config(),
               "truth": {k: float(truth[k]) for k in POSTERIOR_VARS}, "samples": {}, "diagnostics": {}}
    for domain in ("freq", "wdm"):
        t0 = time.perf_counter()
        mcmc = run_domain(bands[domain], seed + 10)
        samples = _samples(mcmc)
        results[domain] = _summarize(domain, samples, truth)
        results["samples"][domain] = _marginal_samples(samples, truth)  # for exact JSD
        results["diagnostics"][domain] = diag = _diagnostics(mcmc)
        worst = max(diag["rhat"].values())
        print(f"  [{domain}] {time.perf_counter() - t0:.1f}s  max R-hat={worst:.3f}  "
              f"min ESS={min(diag['ess'].values()):.0f}  div={diag['divergences']}"
              + ("  <-- NOT CONVERGED" if worst > 1.05 or diag["divergences"] else ""))
    return results


def _run_config() -> dict:
    """Snapshot the run configuration so summaries never mix incompatible runs."""
    return {"channels": list(CHANNELS), "nt": NT, "nblocks": NBLOCKS,
            "snr_min": SNR_MIN, "snr_max": SNR_MAX, "n_warmup": N_WARMUP, "n_draws": N_DRAWS,
            "num_chains": NUM_CHAINS, "max_tree_depth": NUTS_KWARGS["max_tree_depth"],
            "f0_half_bins": F0_PRIOR_HALF_BINS, "f0_sigma_bins": F0_PRIOR_SIGMA_BINS,
            "fdot_half_bins": FDOT_PRIOR_HALF_BINS, "fdot_sigma_bins": FDOT_PRIOR_SIGMA_BINS,
            "center_f0_on_truth": CENTER_F0_ON_TRUTH, "center_fdot_on_truth": CENTER_FDOT_ON_TRUTH,
            "fdot_drift_bins": list(FDOT_DRIFT_BINS)}


def _diagnostics(mcmc) -> dict:
    """Split-R-hat, ESS and divergence count per sampled parameter."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    by_chain = mcmc.get_samples(group_by_chain=True)
    rhat, ess = {}, {}
    # Diagnose the actual sampled sites (z_f0, z_fdot, g_c, g_s): they have no
    # circular boundary, so R-hat is not inflated by phi0 wrapping.
    for k in ("z_f0", "z_fdot", "z_gc", "z_gs"):
        x = np.asarray(by_chain[k])  # (chains, draws)
        rhat[k] = float(split_gelman_rubin(x))
        ess[k] = float(effective_sample_size(x))
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return {"rhat": rhat, "ess": ess, "divergences": div}


def injection_for_seed(seed: int, *, grid: dict, jgb) -> dict:
    """Regenerate the exact injected data for a seed (no MCMC), for plotting.

    Reproduces run_one_seed's source draw, SNR rescaling and data construction
    so the figure shows the same realization that was analyzed.  Returns the
    per-channel full rFFT data + signal, the per-channel PSD, the analysis band,
    and the source parameters."""
    rng = np.random.default_rng(seed)
    truth = draw_source(rng, grid)
    target_snr = float(rng.uniform(SNR_MIN, SNR_MAX))
    psd_full = _psd_full(grid)
    scales = _prior_scales(grid["t_obs"])
    prior_f0 = (truth["f0"] - scales["delta_f0_half"], truth["f0"] + scales["delta_f0_half"])
    sig_margin = _signal_margin(grid, fdot=truth["fdot"], fdot_half=scales["delta_fdot_half"])
    bs = _band_slices(grid, prior_f0[0] - sig_margin, prior_f0[1] + sig_margin)
    ref, kmin = gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], 1.0, truth["phi0"], truth["sky"])
    truth["A"] = target_snr / np.sqrt(
        _optimal_snr_sq(ref, psd_full, grid, band=slice(bs["kmin_rfft"], bs["kmax_rfft"])))
    signal_rfft, _ = gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], truth["A"], truth["phi0"], truth["sky"])
    noise_rfft = _draw_noise_rfft(psd_full, rng=np.random.default_rng(seed + 555), df=grid["df"], dt=grid["dt"])
    data_rfft = np.fft.rfft(np.fft.irfft(signal_rfft + noise_rfft, n=grid["n_total"], axis=-1), axis=-1)
    s = truth["sky"]
    return {
        "truth": truth, "snr": target_snr, "dt": grid["dt"], "t_obs": grid["t_obs"],
        "freqs": np.fft.rfftfreq(grid["n_total"], grid["dt"]),
        "data_rfft": data_rfft, "signal_rfft": signal_rfft, "psd_full": psd_full,
        "band": bs, "kmin": kmin,
        "source_params": np.array([truth["f0"], truth["fdot"], truth["A"],
                                   s["ra"], s["dec"], s["psi"], s["iota"], truth["phi0"]]),
    }


def configure_production_env() -> None:
    """Backward-compatible no-op.

    Production settings now live in module constants, so callers that still
    invoke this helper keep working without environment-variable side effects."""
    return None


def make_pp_plot(results_list: list[dict], out_path: Path) -> None:
    """PP plot: empirical CDF of truth-ranks vs uniform on one shared axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(0, 1, 200)
    n = len(results_list)
    colors = {label: f"C{i}" for i, label in enumerate(POSTERIOR_LABELS)}
    display_labels = {
        "log10(f0 / Hz)": r"$\log_{10} f_0$",
        "fdot [Hz/s]": r"$\dot f$ [Hz/s]",
        "log10(A)": r"$\log_{10} A$",
        "phi0 [rad]": r"$\phi_0$ [rad]",
    }

    sigma = np.sqrt(x * (1 - x) / n)
    for level, color in ((3, "0.92"), (2, "0.85"), (1, "0.76")):
        ax.fill_between(x, np.clip(x - level * sigma, 0, 1),
                        np.clip(x + level * sigma, 0, 1),
                        color=color, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.9, ls=":", zorder=1)
    for label in POSTERIOR_LABELS:
        for domain, linestyle in (("wdm", "-"), ("freq", "--")):
            ranks = np.sort([next(r["rank"] for r in res[domain] if r["label"] == label)
                             for res in results_list])
            cdf = np.searchsorted(ranks, x, side="right") / n
            ax.plot(x, cdf, color=colors[label], ls=linestyle, lw=1.6,
                    label=display_labels[label] if domain == "wdm" else None)

    parameter_legend = ax.legend(loc="upper left", frameon=False, title="parameter")
    ax.add_artist(parameter_legend)
    style_legend = ax.legend(loc="lower right", frameon=False, handles=[
        plt.Line2D([0], [0], color="0.3", ls="-", label="WDM"),
        plt.Line2D([0], [0], color="0.3", ls="--", label="Frequency"),
    ])
    ax.add_artist(style_legend)
    ax.legend(loc="center right", frameon=False, handles=[
        plt.Rectangle((0, 0), 1, 1, color=color, label=fr"${level}\sigma$")
        for level, color in ((1, "0.76"), (2, "0.85"), (3, "0.92"))
    ])
    ax.set(xlabel="credible level", ylabel="fraction of truths below",
           xlim=(0, 1), ylim=(0, 1), title=f"PP plot ({n} seeds)")
    ax.set_aspect("equal")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _study_summary(results_list: list[dict]) -> None:
    """Print rank calibration and freq-vs-WDM agreement over all seeds."""
    print(f"\n===== STUDY SUMMARY ({len(results_list)} seeds) =====")
    for label in POSTERIOR_LABELS:
        line = f"  {label:<18}"
        for domain in ("freq", "wdm"):
            ranks = np.array([next(r["rank"] for r in res[domain] if r["label"] == label)
                              for res in results_list])
            line += f" | {domain} rank={ranks.mean():.3f}"
        deltas = []
        for res in results_list:
            fr = next(r for r in res["freq"] if r["label"] == label)
            wd = next(r for r in res["wdm"] if r["label"] == label)
            s = 0.5 * (fr["std"] + wd["std"])
            if s > 0:
                deltas.append(abs(fr["mean"] - wd["mean"]) / s)
        line += f" | freq-wdm Δ={np.mean(deltas):.2f}σ (max {np.max(deltas):.2f})"
        print(line)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="Single-seed run (ignored if --nseeds given).")
    parser.add_argument("--nseeds", type=int, default=None, help="Run a population study over seeds [0, nseeds).")
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    configure_production_env()  # AET + the f0/fdot prior defaults used by the study

    grid = _make_grid()
    jgb = make_jgb(grid)
    outdir = Path(__file__).resolve().parent / "outdir_gb"
    print(f"[gb] N={grid['n_total']} T_obs={grid['t_obs'] / 86400:.1f}d dt={grid['dt']:.2f}s "
          f"channels={CHANNELS} SNR~U[{SNR_MIN:.0f},{SNR_MAX:.0f}]")

    seeds = range(args.nseeds) if args.nseeds is not None else [args.seed]
    results_list = []
    for seed in seeds:
        res = run_one_seed(seed, grid=grid, jgb=jgb)
        (outdir / f"seed_{seed}.json").parent.mkdir(parents=True, exist_ok=True)
        (outdir / f"seed_{seed}.json").write_text(json.dumps(res, indent=2) + "\n")
        results_list.append(res)
        # Each seed has a different f0 -> different static band indices, so JAX
        # recompiles per seed; clear the compilation cache to keep memory bounded.
        jax.clear_caches()

    if args.nseeds is not None:
        _study_summary(results_list)
        pp_path = outdir / "pp_plot.png"
        make_pp_plot(results_list, pp_path)
        print(f"\n[gb] saved {len(results_list)} results + PP plot to {pp_path}")


if __name__ == "__main__":
    main()
