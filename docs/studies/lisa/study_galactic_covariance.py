"""Galactic foreground spectral estimation with frequency-bin covariance.

Compare posteriors for (alp_gal, fknee) from:
  1. WDM diagonal        – time-varying per-pixel chi-sq(1) likelihood
  2. DFT full covariance – banded annual-harmonic covariance in Fourier bins
  3. DFT diagonal        – same time-averaged PSD, but ignores bin covariance

The useful comparison is between Methods 2 and 3: they use the same
time-averaged diagonal PSD, but Method 2 also includes the annual-harmonic
off-diagonal covariance.  The printed width diagnostics quantify the impact of
dropping those off-diagonal terms.

Run from the lisa study directory:
    python study_galactic_covariance.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import cholesky_banded, solveh_banded
from scipy.special import iv, logsumexp

# ── Local imports (no JAX required for this study) ────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).parent))

INJECTION_NORMALIZATION_VERSION = "physical_psd_rfft_v1"

from lisa_common import (
    INJECTION_PATH,
    RUN_DIR,
    ensure_output_dir,
    noise_tdi15_psd,
    save_figure,
    trim_frequency_band,
)

# ── WDM transform (numpy backend only) ───────────────────────────────────────
from wdm_transform.transforms import forward_wdm_band

# ── Study constants ────────────────────────────────────────────────────────────
CHANNEL = 0           # TDI channel A
NT = 32               # WDM time bins (matches lisa_mcmc default)
A_WDM = 1.0 / 3.0    # WDM window parameter
D_WDM = 1.0

# Keep the exact DFT covariance cheap enough for an example while staying close
# enough to the fiducial knee to constrain both alpha and fknee.
F_LO = 2.0e-3
F_HI = 3.6e-3

# The response is sampled at 90 orbital phases.  The annual harmonics therefore
# produce a banded covariance with offsets |k-k'| <= 45 full-year DFT bins.
MAX_ANNUAL_HARMONIC = 45
COVARIANCE_JITTER = 1.0e-10

# Fiducial galactic PSD spectral parameters (galactic_psd defaults, T_obs=2 yr)
_TOBSYR = 2.0
_AK = -0.27;  _BK = -2.47
_A1 = -0.25;  _B1 = -2.7
ALPHA_TRUE    = 1.8
FKNEE_FIDUCIAL = 10 ** (_AK * np.log10(_TOBSYR) + _BK)   # ≈ 2.81 mHz
F1_FIDUCIAL    = 10 ** (_A1 * np.log10(_TOBSYR) + _B1)   # ≈ 1.68 mHz
F2_FIDUCIAL    = 10 ** -3.5                                # ≈ 0.316 mHz
A_GAL_FIDUCIAL = 10 ** -43.0                               # bright foreground

# Parameter grid
N_GRID = 20
ALPHA_GRID       = np.linspace(1.2, 2.5, N_GRID)
LOG_FKNEE_FACTOR = np.linspace(-0.4, 0.4, N_GRID)   # log10 of fknee / fknee_fiducial

# Synthetic controlled-covariance demo.  This is the default run because it
# isolates the frequency-bin covariance effect from the current WDM modelling
# mismatch in the full LISA injection.
SYNTHETIC_N_GRID = 60
SYNTHETIC_N_BINS = 900
SYNTHETIC_MAX_HARMONIC = 20
SYNTHETIC_RESPONSE_KAPPA = 3.0
SYNTHETIC_SEED = 5
SYNTHETIC_ALPHA_GRID = np.linspace(1.25, 2.35, SYNTHETIC_N_GRID)
SYNTHETIC_LOG_FKNEE_FACTOR = np.linspace(-0.18, 0.18, SYNTHETIC_N_GRID)

# Real-response synthetic demo: draw data from the saved LISA A-channel response
# covariance, then fit amplitude, alpha, and fknee on a coarse 3-D grid.
REAL_RESPONSE_N_BINS = 700
REAL_RESPONSE_MAX_HARMONIC = 30
REAL_RESPONSE_SEED = 13
REAL_RESPONSE_LOG_AMP_GRID = np.linspace(-0.35, 0.35, 13)
REAL_RESPONSE_ALPHA_GRID = np.linspace(1.35, 2.25, 31)
REAL_RESPONSE_LOG_FKNEE_FACTOR = np.linspace(-0.18, 0.18, 31)

BAND_SCAN_BANDS = (
    (0.5e-3, 1.0e-3),
    (1.0e-3, 1.6e-3),
    (1.6e-3, 2.4e-3),
    (2.0e-3, 3.6e-3),
    (0.5e-3, 3.0e-3),
)
BAND_SCAN_N_BINS = 450
BAND_SCAN_MAX_HARMONIC = 30
BAND_SCAN_LOG_AMP_GRID = np.linspace(-0.25, 0.25, 7)
BAND_SCAN_ALPHA_GRID = np.linspace(1.45, 2.15, 17)
BAND_SCAN_LOG_FKNEE_FACTOR = np.linspace(-0.14, 0.14, 17)
BAND_SCAN_SEED = 21

PSD_DIAGNOSTIC_SMOOTH_BINS = 300


# ── Spectral model ─────────────────────────────────────────────────────────────

def _galactic_psd(
    f: np.ndarray,
    alpha: float,
    fknee_factor: float,
    log_amp_factor: float = 0.0,
) -> np.ndarray:
    """One-sided galactic foreground PSD for given spectral parameters."""
    fknee = FKNEE_FIDUCIAL * (10 ** fknee_factor)
    f_safe = np.where(f > 0, f, 1.0)
    return (
        A_GAL_FIDUCIAL
        * (10 ** log_amp_factor)
        * f_safe ** (-7.0 / 3.0)
        * np.exp(-((f_safe / F1_FIDUCIAL) ** alpha))
        * (1.0 + np.tanh((fknee - f_safe) / F2_FIDUCIAL))
    )


def _interp_complex(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation for one complex-valued response curve."""
    return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data(path: Path) -> dict:
    """Load injection, subtract source, return precomputed arrays for all methods."""
    with np.load(path, allow_pickle=False) as raw:
        inj = {k: np.asarray(v) for k, v in raw.items()}

    ver = str(np.asarray(inj["normalization_version"]).item())
    if ver != INJECTION_NORMALIZATION_VERSION:
        raise RuntimeError(
            f"Stale injection (version={ver}). Regenerate with data_generation.py."
        )

    dt     = float(inj["dt"])
    data_A = inj[f"data_At"]  # full time series channel A
    N_full = len(data_A)

    # Subtract injected source (reconstruct in time domain first)
    source_f = np.asarray(inj["source_Af"], dtype=np.complex128)
    source_t = irfft(source_f, n=N_full)
    residual  = data_A - source_t

    # ── Truncate only for the WDM path.  The DFT covariance path keeps the
    # full year so annual harmonics land on integer Fourier-bin offsets.
    n_keep_wdm = (N_full // (2 * NT)) * (2 * NT)
    residual_wdm = residual[:n_keep_wdm]
    nf = n_keep_wdm // NT
    t_obs_wdm = n_keep_wdm * dt
    n_freqs_wdm_full = n_keep_wdm // 2 + 1

    # ── WDM transform in the galactic band ───────────────────────────────────
    df_rfft     = 1.0 / t_obs_wdm
    freq_grid   = np.linspace(0.0, 0.5 / dt, nf + 1)
    band_slice  = trim_frequency_band(freq_grid, F_LO, F_HI, pad_bins=2)
    wdm_freq    = freq_grid[band_slice]                           # (n_band,)
    wdm_time    = (np.arange(NT) + 0.5) * t_obs_wdm / NT          # (NT,)
    n_band_wdm  = band_slice.stop - band_slice.start

    half    = NT // 2
    kmin    = max((band_slice.start - 1) * half, 0)
    kmax    = min(band_slice.stop * half, n_freqs_wdm_full)

    data_rfft_wdm = rfft(residual_wdm)

    coeffs_wdm = np.asarray(forward_wdm_band(
        data_rfft_wdm[kmin:kmax],
        df=df_rfft,
        nfreqs_fourier=n_freqs_wdm_full,
        kmin=kmin,
        nfreqs_wdm=nf,
        ntimes_wdm=NT,
        mmin=band_slice.start,
        nf_sub_wdm=n_band_wdm,
        a=A_WDM,
        d=D_WDM,
        backend="numpy",
    ))  # (NT, n_band_wdm)

    # ── Full-year DFT data in the narrow galactic band ────────────────────────
    data_rfft = rfft(residual)
    freqs_rfft = rfftfreq(N_full, d=dt)
    dft_band   = (freqs_rfft >= F_LO) & (freqs_rfft <= F_HI)
    f_dft      = freqs_rfft[dft_band]                            # (N_dft,)
    X_dft      = data_rfft[dft_band]                             # complex (N_dft,)
    # One-sided periodogram S_hat ≈ S(f): E[P] = S for P = 2dt/N |X|²
    periodogram_dft = (2.0 * dt / N_full) * np.abs(X_dft) ** 2  # (N_dft,)

    # ── Pure antenna response R(t, f) ─────────────────────────────────────────
    gal_tf    = inj["gal_psd_A_tf"]   # (n_gal_t, n_gal_f)
    gal_freqs = inj["gal_psd_freqs"]  # (n_gal_f,)
    gal_times = inj["gal_psd_times"]  # (n_gal_t,)

    S_gal_fid = _galactic_psd(gal_freqs, ALPHA_TRUE, 0.0)        # fiducial model
    R_tf      = gal_tf / np.maximum(S_gal_fid, 1e-300)           # (n_gal_t, n_gal_f)
    R_mean    = R_tf.mean(axis=0)                                 # (n_gal_f,)
    R_harmonics = np.fft.fft(R_tf, axis=0) / R_tf.shape[0]
    max_harmonic = min(MAX_ANNUAL_HARMONIC, R_tf.shape[0] // 2)

    # Interpolate R(t_n, f_m) at WDM time/freq centres
    R_wdm = np.stack([
        np.interp(wdm_freq, gal_freqs, R_tf[np.argmin(np.abs(gal_times - t))])
        for t in wdm_time
    ])   # (NT, n_band_wdm)

    # Interpolate R_mean at DFT frequencies
    R_mean_dft = np.interp(f_dft, gal_freqs, R_mean)             # (N_dft,)
    R_harmonics_dft = np.empty((max_harmonic + 1, len(f_dft)), dtype=np.complex128)
    for harmonic in range(max_harmonic + 1):
        R_harmonics_dft[harmonic] = _interp_complex(
            f_dft,
            gal_freqs,
            R_harmonics[harmonic],
        )

    # Instrument noise at WDM and DFT frequencies
    S_inst_wdm = noise_tdi15_psd(CHANNEL, wdm_freq)              # (n_band_wdm,)
    S_inst_dft = noise_tdi15_psd(CHANNEL, f_dft)                 # (N_dft,)

    return dict(
        dt=dt,
        n_full=N_full,
        n_keep_wdm=n_keep_wdm,
        nf=nf,
        n_freqs_wdm_full=n_freqs_wdm_full,
        t_obs_wdm=t_obs_wdm,
        # WDM data
        coeffs_wdm=coeffs_wdm,
        wdm_freq=wdm_freq,
        wdm_time=wdm_time,
        R_wdm=R_wdm,
        S_inst_wdm=S_inst_wdm,
        # DFT data
        f_dft=f_dft,
        X_dft=X_dft,
        periodogram_dft=periodogram_dft,
        R_mean_dft=R_mean_dft,
        R_harmonics_dft=R_harmonics_dft,
        max_harmonic=max_harmonic,
        S_inst_dft=S_inst_dft,
    )


# ── Likelihood functions ───────────────────────────────────────────────────────

def _log_like_wdm(alpha: float, fknee_factor: float, d: dict) -> float:
    """Method 1: WDM diagonal chi-sq(1) likelihood with time-varying sigma²."""
    S_gal = _galactic_psd(d["wdm_freq"], alpha, fknee_factor)  # (n_band,)
    # sigma²[n, m] = N * (S_inst + S_gal * R(t_n, f_m)) / (2*dt)
    N = d["n_freqs_wdm_full"] * 2 - 2
    sigma2 = (N / (2.0 * d["dt"])) * (d["S_inst_wdm"] + S_gal * d["R_wdm"])
    sigma2 = np.maximum(sigma2, 1e-60)
    w2 = d["coeffs_wdm"] ** 2
    return float(-0.5 * np.sum(w2 / sigma2 + np.log(sigma2)))


def _build_lower_banded_covariance(
    alpha: float,
    fknee_factor: float,
    d: dict,
    log_amp_factor: float = 0.0,
) -> np.ndarray:
    """Return lower-banded covariance for full-year complex rFFT coefficients."""
    S_gal = _galactic_psd(d["f_dft"], alpha, fknee_factor, log_amp_factor)
    n = len(S_gal)
    bw = min(int(d["max_harmonic"]), n - 1)
    scale = d["n_full"] / (2.0 * d["dt"])
    ab = np.zeros((bw + 1, n), dtype=np.complex128)

    diagonal_psd = d["S_inst_dft"] + S_gal * d["R_harmonics_dft"][0].real
    diagonal_psd = np.maximum(diagonal_psd, 1e-60)
    jitter = COVARIANCE_JITTER * np.median(diagonal_psd)
    ab[0] = scale * (diagonal_psd + jitter)

    for harmonic in range(1, bw + 1):
        s_mid = np.sqrt(S_gal[harmonic:] * S_gal[:-harmonic])
        response_mid = 0.5 * (
            d["R_harmonics_dft"][harmonic, harmonic:]
            + d["R_harmonics_dft"][harmonic, :-harmonic]
        )
        ab[harmonic, :-harmonic] = scale * s_mid * response_mid
    return ab


def _log_like_dft_full_covariance(alpha: float, fknee_factor: float, d: dict) -> float:
    """Method 2: complex Gaussian likelihood with annual-harmonic covariance."""
    cov_ab = _build_lower_banded_covariance(alpha, fknee_factor, d)
    x = d["X_dft"]
    try:
        chol = cholesky_banded(cov_ab, lower=True, check_finite=False)
        solved = solveh_banded(cov_ab, x, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf

    quad = np.vdot(x, solved).real
    logdet = 2.0 * np.sum(np.log(np.maximum(chol[0].real, 1e-300)))
    return float(-(quad + logdet))


def _log_like_dft_diagonal(alpha: float, fknee_factor: float, d: dict) -> float:
    """Method 3: diagonal Whittle likelihood with the time-averaged response."""
    S_gal = _galactic_psd(d["f_dft"], alpha, fknee_factor)
    S_model = np.maximum(d["S_inst_dft"] + S_gal * d["R_mean_dft"], 1e-60)
    P = d["periodogram_dft"]
    return float(-np.sum(P / S_model + np.log(S_model)))


def _log_like_dft_full_covariance_amp(
    log_amp_factor: float,
    alpha: float,
    fknee_factor: float,
    d: dict,
) -> float:
    cov_ab = _build_lower_banded_covariance(
        alpha,
        fknee_factor,
        d,
        log_amp_factor,
    )
    x = d["X_dft"]
    try:
        chol = cholesky_banded(cov_ab, lower=True, check_finite=False)
        solved = solveh_banded(cov_ab, x, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf
    quad = np.vdot(x, solved).real
    logdet = 2.0 * np.sum(np.log(np.maximum(chol[0].real, 1e-300)))
    return float(-(quad + logdet))


def _log_like_dft_diagonal_amp(
    log_amp_factor: float,
    alpha: float,
    fknee_factor: float,
    d: dict,
) -> float:
    S_gal = _galactic_psd(d["f_dft"], alpha, fknee_factor, log_amp_factor)
    S_model = np.maximum(d["S_inst_dft"] + S_gal * d["R_mean_dft"], 1e-60)
    P = d["periodogram_dft"]
    return float(-np.sum(P / S_model + np.log(S_model)))


def _matmul_lower_banded(lower_ab: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return L @ vector for SciPy lower-banded storage."""
    bw = lower_ab.shape[0] - 1
    out = lower_ab[0] * vector
    for offset in range(1, bw + 1):
        out[offset:] += lower_ab[offset, :-offset] * vector[:-offset]
    return out


def _load_synthetic_covariance_data() -> dict:
    """Draw a controlled foreground realization from a known banded covariance.

    The synthetic response is positive by construction:

        R(t) ∝ exp(kappa cos(2πt/T)).

    Its Fourier coefficients are I_k(kappa) / I_0(kappa), so we can place them
    directly into the annual-harmonic covariance matrix.  This creates a clear
    test case where the diagonal covariance approximation has the right mean
    PSD but the wrong likelihood geometry.
    """
    f_dft = np.linspace(F_LO, F_HI, SYNTHETIC_N_BINS)
    harmonics = np.zeros(
        (SYNTHETIC_MAX_HARMONIC + 1, SYNTHETIC_N_BINS),
        dtype=np.complex128,
    )
    norm = iv(0, SYNTHETIC_RESPONSE_KAPPA)
    for harmonic in range(SYNTHETIC_MAX_HARMONIC + 1):
        harmonics[harmonic] = iv(harmonic, SYNTHETIC_RESPONSE_KAPPA) / norm

    data = dict(
        dt=1.0,
        n_full=2,
        f_dft=f_dft,
        R_harmonics_dft=harmonics,
        R_mean_dft=np.ones_like(f_dft),
        max_harmonic=SYNTHETIC_MAX_HARMONIC,
        S_inst_dft=np.zeros_like(f_dft),
        synthetic_response_kappa=SYNTHETIC_RESPONSE_KAPPA,
    )

    cov_ab = _build_lower_banded_covariance(ALPHA_TRUE, 0.0, data)
    chol_ab = cholesky_banded(cov_ab, lower=True, check_finite=False)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    white = (
        rng.standard_normal(SYNTHETIC_N_BINS)
        + 1j * rng.standard_normal(SYNTHETIC_N_BINS)
    ) / np.sqrt(2.0)
    x_dft = _matmul_lower_banded(chol_ab, white)
    data["X_dft"] = x_dft
    data["periodogram_dft"] = np.abs(x_dft) ** 2
    return data


def _load_real_response_arrays(path: Path = INJECTION_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return response-grid frequencies and pure A-channel response R(t, f)."""
    with np.load(path, allow_pickle=False) as raw:
        inj = {key: np.asarray(value) for key, value in raw.items()}

    ver = str(np.asarray(inj["normalization_version"]).item())
    if ver != INJECTION_NORMALIZATION_VERSION:
        raise RuntimeError(
            f"Stale injection (version={ver}). Regenerate with data_generation.py."
        )

    gal_tf = np.asarray(inj["gal_psd_A_tf"], dtype=float)
    gal_freqs = np.asarray(inj["gal_psd_freqs"], dtype=float)
    s_fid = _galactic_psd(gal_freqs, ALPHA_TRUE, 0.0)
    r_tf = gal_tf / np.maximum(s_fid, 1e-300)
    return gal_freqs, r_tf


def _build_real_response_data_from_arrays(
    gal_freqs: np.ndarray,
    r_tf: np.ndarray,
    *,
    f_lo: float,
    f_hi: float,
    n_bins: int,
    max_harmonic: int,
    seed: int,
) -> dict:
    """Draw synthetic data from real response arrays for one frequency band."""
    r_harmonics = np.fft.fft(r_tf, axis=0) / r_tf.shape[0]
    max_harmonic = min(max_harmonic, r_tf.shape[0] // 2)
    f_dft = np.linspace(f_lo, f_hi, n_bins)
    harmonics = np.empty((max_harmonic + 1, n_bins), dtype=np.complex128)
    for harmonic in range(max_harmonic + 1):
        harmonics[harmonic] = _interp_complex(f_dft, gal_freqs, r_harmonics[harmonic])

    data = dict(
        dt=1.0,
        n_full=2,
        f_dft=f_dft,
        R_harmonics_dft=harmonics,
        R_mean_dft=harmonics[0].real,
        max_harmonic=max_harmonic,
        S_inst_dft=np.zeros_like(f_dft),
    )

    cov_ab = _build_lower_banded_covariance(ALPHA_TRUE, 0.0, data)
    chol_ab = cholesky_banded(cov_ab, lower=True, check_finite=False)
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal(n_bins) + 1j * rng.standard_normal(n_bins)) / np.sqrt(2.0)
    x_dft = _matmul_lower_banded(chol_ab, white)
    data["X_dft"] = x_dft
    data["periodogram_dft"] = np.abs(x_dft) ** 2
    return data


def _load_real_response_synthetic_data(path: Path = INJECTION_PATH) -> dict:
    """Draw synthetic data from the saved LISA A-channel response covariance."""
    gal_freqs, r_tf = _load_real_response_arrays(path)
    return _build_real_response_data_from_arrays(
        gal_freqs,
        r_tf,
        f_lo=F_LO,
        f_hi=F_HI,
        n_bins=REAL_RESPONSE_N_BINS,
        max_harmonic=REAL_RESPONSE_MAX_HARMONIC,
        seed=REAL_RESPONSE_SEED,
    )


# ── Grid evaluation ────────────────────────────────────────────────────────────

def _run_grid(log_like_fn, alpha_grid, fknee_grid, d: dict) -> tuple[np.ndarray, float]:
    """Evaluate log-likelihood on an (N_alpha × N_fknee) grid.

    Returns:
        log_L: (N_alpha, N_fknee) array
        elapsed: wall time in seconds
    """
    N_a, N_k = len(alpha_grid), len(fknee_grid)
    log_L = np.zeros((N_a, N_k))
    t0 = time.perf_counter()
    for i, alpha in enumerate(alpha_grid):
        for j, fknee_fac in enumerate(fknee_grid):
            log_L[i, j] = log_like_fn(alpha, fknee_fac, d)
    elapsed = time.perf_counter() - t0
    return log_L, elapsed


def _run_grid_3d(
    log_like_fn,
    log_amp_grid: np.ndarray,
    alpha_grid: np.ndarray,
    fknee_grid: np.ndarray,
    d: dict,
) -> tuple[np.ndarray, float]:
    """Evaluate a likelihood on (log_amp, alpha, log_fknee_factor)."""
    log_l = np.zeros((len(log_amp_grid), len(alpha_grid), len(fknee_grid)))
    t0 = time.perf_counter()
    for ia, log_amp in enumerate(log_amp_grid):
        for i, alpha in enumerate(alpha_grid):
            for j, fknee_fac in enumerate(fknee_grid):
                log_l[ia, i, j] = log_like_fn(log_amp, alpha, fknee_fac, d)
    return log_l, time.perf_counter() - t0


# ── Posterior normalisation ────────────────────────────────────────────────────

def _to_posterior(log_L: np.ndarray) -> np.ndarray:
    log_L = log_L - log_L.max()
    return np.exp(log_L)


def _profile_over_amplitude(log_l_3d: np.ndarray) -> np.ndarray:
    """Profile a 3-D log-likelihood over amplitude."""
    return np.max(log_l_3d, axis=0)


def _marginalize_over_amplitude(log_l_3d: np.ndarray) -> np.ndarray:
    """Marginalize a 3-D log-likelihood over a uniform log-amplitude grid."""
    return logsumexp(log_l_3d, axis=0)


def _posterior_summary(
    posterior: np.ndarray,
    alpha_grid: np.ndarray,
    fknee_grid: np.ndarray,
) -> dict[str, float]:
    weights = np.asarray(posterior, dtype=float)
    weights = weights / np.sum(weights)
    alpha = alpha_grid[:, None]
    log_fknee = fknee_grid[None, :]
    mean_alpha = float(np.sum(weights * alpha))
    mean_log_fknee = float(np.sum(weights * log_fknee))
    std_alpha = float(np.sqrt(np.sum(weights * (alpha - mean_alpha) ** 2)))
    std_log_fknee = float(np.sqrt(np.sum(weights * (log_fknee - mean_log_fknee) ** 2)))

    flat = np.sort(weights.ravel())[::-1]
    cumulative = np.cumsum(flat)
    threshold_68 = flat[min(np.searchsorted(cumulative, 0.68), flat.size - 1)]
    hpd_cells_68 = float(np.count_nonzero(weights >= threshold_68))
    effective_cells = float(1.0 / np.sum(weights**2))
    return {
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "mean_log_fknee": mean_log_fknee,
        "std_log_fknee": std_log_fknee,
        "hpd_cells_68": hpd_cells_68,
        "effective_cells": effective_cells,
    }


def _response_harmonic_summary(d: dict) -> dict[str, float]:
    r0 = np.maximum(np.abs(d["R_harmonics_dft"][0]), 1e-300)
    ratios = np.abs(d["R_harmonics_dft"][1:]) / r0[None, :]
    if ratios.size == 0:
        return {"median_ratio": 0.0, "max_ratio": 0.0, "rss_ratio": 0.0}
    return {
        "median_ratio": float(np.median(ratios)),
        "max_ratio": float(np.max(ratios)),
        "rss_ratio": float(np.median(np.sqrt(np.sum(ratios**2, axis=0)))),
    }


def _log_bin_average(
    freqs: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freqs > 0.0) & np.isfinite(values) & (values > 0.0)
    freqs = np.asarray(freqs[mask], dtype=float)
    values = np.asarray(values[mask], dtype=float)
    edges = np.geomspace(freqs.min(), freqs.max(), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    averaged = np.full(n_bins, np.nan)
    for i in range(n_bins):
        keep = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(keep):
            averaged[i] = np.mean(values[keep])
    good = np.isfinite(averaged)
    return centers[good], averaged[good]


def _smooth_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_dense: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a grid surface for display-only contour smoothing."""
    x_dense = np.linspace(float(x[0]), float(x[-1]), n_dense)
    y_dense = np.linspace(float(y[0]), float(y[-1]), n_dense)
    spline = RectBivariateSpline(y, x, z, kx=3, ky=3)
    return x_dense, y_dense, spline(y_dense, x_dense)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_comparison(
    log_surfaces: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float]],
    timings: dict[str, float],
    alpha_grid: np.ndarray,
    fknee_grid: np.ndarray,
    run_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    fknee_mhz = FKNEE_FIDUCIAL * 10 ** fknee_grid * 1e3   # in mHz

    method_styles = {
        "Method 1\n(WDM diagonal)": {
            "short": "1. WDM domain",
            "subtitle": "time-localized diagonal pixels",
            "color": "C0",
        },
        "Method 2\n(DFT full covariance)": {
            "short": "2. Frequency domain",
            "subtitle": "full banded covariance",
            "color": "C2",
        },
        "Method 3\n(DFT diagonal)": {
            "short": "3. Frequency domain",
            "subtitle": "diagonal covariance approximation",
            "color": "C3",
        },
    }
    method_keys = list(method_styles.keys())
    reference_key = "Method 2\n(DFT full covariance)"
    reference_delta = log_surfaces[reference_key] - np.max(log_surfaces[reference_key])
    chi2_levels = np.array([2.30, 6.18])
    contour_levels = -0.5 * chi2_levels[::-1]
    image = None

    for ax_idx, key in enumerate(method_keys):
        ax = axes[ax_idx]
        style = method_styles[key]
        delta = log_surfaces[key] - np.max(log_surfaces[key])
        clipped = np.clip(delta, -12.0, 0.0)
        image = ax.pcolormesh(
            fknee_mhz,
            alpha_grid,
            clipped,
            shading="auto",
            cmap="viridis",
            vmin=-12.0,
            vmax=0.0,
        )
        ax.contour(
            fknee_mhz,
            alpha_grid,
            delta,
            levels=contour_levels,
            colors=[style["color"]],
            linewidths=2.0,
        )
        if key != reference_key:
            ax.contour(
                fknee_mhz,
                alpha_grid,
                reference_delta,
                levels=contour_levels,
                colors="white",
                linewidths=1.5,
                linestyles="--",
            )

        best_i, best_j = np.unravel_index(np.argmax(log_surfaces[key]), log_surfaces[key].shape)
        best_fknee_mhz = fknee_mhz[best_j]
        best_alpha = alpha_grid[best_i]

        # Mark true parameters
        ax.axhline(ALPHA_TRUE, color="k", lw=0.7, ls=":")
        ax.axvline(FKNEE_FIDUCIAL * 1e3, color="k", lw=0.7, ls=":")
        ax.plot(FKNEE_FIDUCIAL * 1e3, ALPHA_TRUE, marker="+", ms=9, mew=1.6, color="k", label="truth")
        ax.plot(best_fknee_mhz, best_alpha, marker="x", ms=8, mew=1.8, color="white", label="best grid point")
        ax.text(
            0.03,
            0.97,
            f"best: α={best_alpha:.2f}\n"
            f"$f_\\mathrm{{knee}}$={best_fknee_mhz:.2f} mHz",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

        t = timings.get(key, float("nan"))
        summary = summaries[key]
        ax.set_title(
            f"{style['short']}\n{style['subtitle']}\n"
            f"{t:.2f}s grid,  σ_α={summary['std_alpha']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("$f_\\mathrm{knee}$ (mHz)")
        ax.set_ylim(alpha_grid[0], alpha_grid[-1])
        ax.grid(color="white", alpha=0.18, lw=0.5)

    axes[0].set_ylabel("$\\alpha$")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.colorbar(image, ax=axes, shrink=0.86, label="$\\Delta \\log L$ (clipped at -12)")

    fig.suptitle(
        "Galactic foreground spectral inference: posterior surfaces\n"
        "Solid contours are each method's 68% and 95% likelihood-ratio levels; "
        "white dashed contours show Method 2 for comparison.  "
        f"DFT band: {F_LO*1e3:.1f}-{F_HI*1e3:.1f} mHz  |  "
        f"{N_GRID}x{N_GRID} grid",
        fontsize=9,
    )
    save_figure(fig, run_dir, "study_galactic_covariance_posteriors")
    print(f"Saved to {run_dir / 'study_galactic_covariance_posteriors.png'}")

    # ── 1D marginals ──────────────────────────────────────────────────────────
    fig2, (ax_a, ax_fk) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for key in method_keys:
        color = method_styles[key]["color"]
        label = method_styles[key]["short"]
        Z = _to_posterior(log_surfaces[key])
        marg_alpha  = Z.sum(axis=1);  marg_alpha  /= marg_alpha.sum()
        marg_fknee  = Z.sum(axis=0);  marg_fknee  /= marg_fknee.sum()
        ax_a.plot(alpha_grid, marg_alpha, color=color, label=label)
        ax_fk.plot(fknee_mhz, marg_fknee, color=color, label=label)

    ax_a.axvline(ALPHA_TRUE, color="k", lw=0.8, ls=":")
    ax_a.set_xlabel("$\\alpha$");   ax_a.set_title("Marginal over $\\alpha$")
    ax_a.legend(fontsize=7)
    ax_fk.axvline(FKNEE_FIDUCIAL * 1e3, color="k", lw=0.8, ls=":")
    ax_fk.set_xlabel("$f_\\mathrm{knee}$ (mHz)");  ax_fk.set_title("Marginal over $f_\\mathrm{knee}$")
    ax_fk.legend(fontsize=7)
    save_figure(fig2, run_dir, "study_galactic_covariance_marginals")
    print(f"Saved to {run_dir / 'study_galactic_covariance_marginals.png'}")


def _plot_synthetic_comparison(
    log_surfaces: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float]],
    timings: dict[str, float],
    alpha_grid: np.ndarray,
    fknee_grid: np.ndarray,
    run_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharex=True, sharey=True, constrained_layout=True)
    fknee_mhz = FKNEE_FIDUCIAL * 10 ** fknee_grid * 1e3
    keys = [
        "Full frequency covariance",
        "Diagonal frequency covariance",
    ]
    styles = {
        "Full frequency covariance": ("C2", "correct banded covariance"),
        "Diagonal frequency covariance": ("C3", "drops off-diagonal bins"),
    }
    reference_delta = log_surfaces[keys[0]] - np.max(log_surfaces[keys[0]])
    fknee_dense, alpha_dense, reference_delta_dense = _smooth_surface(
        fknee_mhz,
        alpha_grid,
        reference_delta,
    )
    contour_levels = -0.5 * np.array([6.18, 2.30])
    image = None

    for ax, key in zip(axes, keys):
        color, subtitle = styles[key]
        delta = log_surfaces[key] - np.max(log_surfaces[key])
        fknee_dense, alpha_dense, delta_dense = _smooth_surface(
            fknee_mhz,
            alpha_grid,
            delta,
        )
        image = ax.contourf(
            fknee_dense,
            alpha_dense,
            np.clip(delta_dense, -10.0, 0.0),
            levels=np.linspace(-10.0, 0.0, 18),
            cmap="viridis",
        )
        ax.contour(
            fknee_dense,
            alpha_dense,
            delta_dense,
            levels=contour_levels,
            colors=[color],
            linewidths=2.2,
        )
        if key != keys[0]:
            ax.contour(
                fknee_dense,
                alpha_dense,
                reference_delta_dense,
                levels=contour_levels,
                colors="white",
                linewidths=1.6,
                linestyles="--",
            )
        best_i, best_j = np.unravel_index(np.argmax(log_surfaces[key]), log_surfaces[key].shape)
        best_alpha = alpha_grid[best_i]
        best_fknee_mhz = fknee_mhz[best_j]
        ax.axhline(ALPHA_TRUE, color="k", lw=0.8, ls=":")
        ax.axvline(FKNEE_FIDUCIAL * 1e3, color="k", lw=0.8, ls=":")
        ax.plot(FKNEE_FIDUCIAL * 1e3, ALPHA_TRUE, marker="+", ms=10, mew=1.7, color="k")
        ax.plot(best_fknee_mhz, best_alpha, marker="x", ms=8, mew=1.8, color="white")
        ax.text(
            0.04,
            0.96,
            f"best: α={best_alpha:.2f}\n"
            f"$f_\\mathrm{{knee}}$={best_fknee_mhz:.2f} mHz\n"
            f"σ_α={summaries[key]['std_alpha']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
        ax.set_title(f"{key}\n{subtitle}\n{timings[key]:.2f}s for {SYNTHETIC_N_GRID}x{SYNTHETIC_N_GRID} grid", fontsize=9)
        ax.set_xlabel("$f_\\mathrm{knee}$ (mHz)")
        ax.grid(color="white", alpha=0.2, lw=0.5)

    axes[0].set_ylabel("$\\alpha$")
    fig.colorbar(image, ax=axes, shrink=0.86, label="$\\Delta \\log L$ (clipped at -10)")
    fig.suptitle(
        "Controlled synthetic covariance test\n"
        "Data are drawn from the full banded covariance.  White dashed contours in the right panel are the correct covariance reference.",
        fontsize=10,
    )
    save_figure(fig, run_dir, "study_galactic_covariance_synthetic_posteriors")
    print(f"Saved to {run_dir / 'study_galactic_covariance_synthetic_posteriors.png'}")


def _plot_real_response_comparison(
    log_surfaces: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float]],
    timings: dict[str, float],
    alpha_grid: np.ndarray,
    fknee_grid: np.ndarray,
    run_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharex=True, sharey=True, constrained_layout=True)
    fknee_mhz = FKNEE_FIDUCIAL * 10 ** fknee_grid * 1e3
    keys = ["Full covariance", "Diagonal covariance"]
    colors = {"Full covariance": "C2", "Diagonal covariance": "C3"}
    reference_delta = log_surfaces["Full covariance"] - np.max(log_surfaces["Full covariance"])
    fknee_dense, alpha_dense, reference_delta_dense = _smooth_surface(
        fknee_mhz,
        alpha_grid,
        reference_delta,
    )
    contour_levels = -0.5 * np.array([6.18, 2.30])
    image = None

    for ax, key in zip(axes, keys):
        delta = log_surfaces[key] - np.max(log_surfaces[key])
        fknee_dense, alpha_dense, delta_dense = _smooth_surface(fknee_mhz, alpha_grid, delta)
        image = ax.contourf(
            fknee_dense,
            alpha_dense,
            np.clip(delta_dense, -10.0, 0.0),
            levels=np.linspace(-10.0, 0.0, 18),
            cmap="viridis",
        )
        ax.contour(
            fknee_dense,
            alpha_dense,
            delta_dense,
            levels=contour_levels,
            colors=[colors[key]],
            linewidths=2.2,
        )
        if key != "Full covariance":
            ax.contour(
                fknee_dense,
                alpha_dense,
                reference_delta_dense,
                levels=contour_levels,
                colors="white",
                linewidths=1.6,
                linestyles="--",
            )

        best_i, best_j = np.unravel_index(np.argmax(log_surfaces[key]), log_surfaces[key].shape)
        best_alpha = alpha_grid[best_i]
        best_fknee_mhz = fknee_mhz[best_j]
        ax.axhline(ALPHA_TRUE, color="k", lw=0.8, ls=":")
        ax.axvline(FKNEE_FIDUCIAL * 1e3, color="k", lw=0.8, ls=":")
        ax.plot(FKNEE_FIDUCIAL * 1e3, ALPHA_TRUE, marker="+", ms=10, mew=1.7, color="k")
        ax.plot(best_fknee_mhz, best_alpha, marker="x", ms=8, mew=1.8, color="white")
        ax.text(
            0.04,
            0.96,
            f"best: alpha={best_alpha:.2f}\n"
            f"$f_\\mathrm{{knee}}$={best_fknee_mhz:.2f} mHz\n"
            f"std alpha={summaries[key]['std_alpha']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
        ax.set_title(f"{key}\nprofiled over amplitude\n{timings[key]:.2f}s grid", fontsize=9)
        ax.set_xlabel("$f_\\mathrm{knee}$ (mHz)")
        ax.grid(color="white", alpha=0.2, lw=0.5)

    axes[0].set_ylabel("$\\alpha$")
    fig.colorbar(image, ax=axes, shrink=0.86, label="$\\Delta \\log L$ (clipped at -10)")
    fig.suptitle(
        "Real LISA response, synthetic covariance draw\n"
        "Foreground-only data drawn from saved A-channel response harmonics; "
        "white dashed contours in right panel are the full-covariance reference.",
        fontsize=10,
    )
    save_figure(fig, run_dir, "study_galactic_covariance_real_response_posteriors")
    print(f"Saved to {run_dir / 'study_galactic_covariance_real_response_posteriors.png'}")


def _plot_band_scan(rows: list[dict[str, float]], run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['f_lo_mhz']:.1f}-{row['f_hi_mhz']:.1f}" for row in rows]
    x = np.arange(len(rows))
    sigma_ratio = np.array([row["sigma_alpha_ratio"] for row in rows])
    alpha_shift = np.array([row["diag_minus_full_alpha"] for row in rows])
    rss_ratio = np.array([row["rss_harmonic_ratio"] for row in rows])
    max_ratio = np.array([row["max_harmonic_ratio"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True, constrained_layout=True)
    axes[0].bar(x, sigma_ratio, color="C3", alpha=0.8)
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("diag/full\nstd(alpha)")
    axes[0].set_title("Real-response band scan: diagonal covariance impact")

    axes[1].bar(x, alpha_shift, color="C1", alpha=0.8)
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_ylabel("diag - full\nbest alpha")

    axes[2].plot(x, rss_ratio, marker="o", color="C2", label="median RSS |R_k|/R_0")
    axes[2].plot(x, max_ratio, marker="s", color="C0", label="max |R_k|/R_0")
    axes[2].set_ylabel("harmonic\nstrength")
    axes[2].set_xlabel("frequency band (mHz)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)

    save_figure(fig, run_dir, "study_galactic_covariance_band_scan")
    print(f"Saved to {run_dir / 'study_galactic_covariance_band_scan.png'}")


def _plot_psd_sanity_check(
    injection_path: Path = INJECTION_PATH,
    run_dir: Path = RUN_DIR,
) -> None:
    import matplotlib.pyplot as plt

    with np.load(injection_path, allow_pickle=False) as raw:
        inj = {key: np.asarray(value) for key, value in raw.items()}

    dt = float(inj["dt"])
    data_a = np.asarray(inj["data_At"], dtype=float)
    source_f = np.asarray(inj["source_Af"], dtype=np.complex128)
    source_t = irfft(source_f, n=len(data_a))
    residual = data_a - source_t
    n = len(residual)
    freqs = rfftfreq(n, d=dt)
    residual_f = rfft(residual)
    periodogram = (2.0 * dt / n) * np.abs(residual_f) ** 2

    gal_freqs, r_tf = _load_real_response_arrays(injection_path)
    r_mean = r_tf.mean(axis=0)
    gal_intrinsic = _galactic_psd(gal_freqs, ALPHA_TRUE, 0.0)
    gal_mean = gal_intrinsic * r_mean
    gal_mean_interp = np.interp(freqs, gal_freqs, gal_mean, left=np.nan, right=np.nan)
    r_mean_interp = np.interp(freqs, gal_freqs, r_mean, left=np.nan, right=np.nan)
    gal_intrinsic_interp = np.interp(freqs, gal_freqs, gal_intrinsic, left=np.nan, right=np.nan)
    inst = noise_tdi15_psd(CHANNEL, freqs)
    total = inst + np.nan_to_num(gal_mean_interp, nan=0.0)

    f_avg, p_avg = _log_bin_average(freqs[1:], periodogram[1:], n_bins=PSD_DIAGNOSTIC_SMOOTH_BINS)
    r_avg = np.interp(f_avg, freqs, r_mean_interp, left=np.nan, right=np.nan)
    p_avg_strain = p_avg / np.maximum(r_avg, 1e-300)
    positive = (freqs > 0.0) & (freqs >= 1.0e-4) & (freqs <= 1.0e-2)
    ratio_band = (freqs >= 5.0e-4) & (freqs <= 3.0e-3) & np.isfinite(gal_mean_interp)
    gal_to_inst_median = float(np.median(gal_mean_interp[ratio_band] / inst[ratio_band]))
    gal_to_inst_max = float(np.max(gal_mean_interp[ratio_band] / inst[ratio_band]))

    strain_positive = positive & np.isfinite(r_mean_interp) & (r_mean_interp > 0.0)
    inst_strain = inst / np.maximum(r_mean_interp, 1e-300)
    total_strain = total / np.maximum(r_mean_interp, 1e-300)
    strain_ratio_band = ratio_band & strain_positive
    intrinsic_to_inst_median = float(
        np.median(gal_intrinsic_interp[strain_ratio_band] / inst_strain[strain_ratio_band])
    )
    intrinsic_to_inst_max = float(
        np.max(gal_intrinsic_interp[strain_ratio_band] / inst_strain[strain_ratio_band])
    )

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax.loglog(freqs[positive], inst[positive], color="k", ls="--", lw=1.8, label="Instrument noise")
    ax.loglog(
        freqs[positive],
        total[positive],
        color="magenta",
        lw=1.8,
        label="Instrument + mean foreground",
    )
    gal_positive = positive & np.isfinite(gal_mean_interp) & (gal_mean_interp > 0.0)
    ax.loglog(
        freqs[gal_positive],
        gal_mean_interp[gal_positive],
        color="C0",
        lw=1.6,
        label="Mean galactic foreground",
    )
    ax.loglog(f_avg, p_avg, color="0.65", lw=1.0, alpha=0.9, label="Residual periodogram (log-binned)")
    ax.axvspan(5.0e-4, 3.0e-3, color="C2", alpha=0.08, label="0.5-3.0 mHz")
    ax.axvspan(F_LO, F_HI, color="C3", alpha=0.08, label=f"default {F_LO*1e3:.1f}-{F_HI*1e3:.1f} mHz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("One-sided PSD [1/Hz]")
    ax.set_xlim(1.0e-4, 1.0e-2)
    ax.set_ylim(1.0e-44, 3.0e-39)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(
        "A-channel TDI PSD\n"
        f"median foreground/instrument in 0.5-3.0 mHz = {gal_to_inst_median:.2f}, "
        f"max = {gal_to_inst_max:.2f}"
    )

    ax_s.loglog(
        freqs[strain_positive],
        inst_strain[strain_positive],
        color="k",
        ls="--",
        lw=1.8,
        label="Instrument noise / mean response",
    )
    ax_s.loglog(
        freqs[strain_positive],
        total_strain[strain_positive],
        color="magenta",
        lw=1.8,
        label="Total / mean response",
    )
    intrinsic_positive = strain_positive & np.isfinite(gal_intrinsic_interp) & (gal_intrinsic_interp > 0.0)
    ax_s.loglog(
        freqs[intrinsic_positive],
        gal_intrinsic_interp[intrinsic_positive],
        color="C0",
        lw=1.6,
        label="Intrinsic foreground model",
    )
    good_avg = np.isfinite(p_avg_strain) & (p_avg_strain > 0.0)
    ax_s.loglog(
        f_avg[good_avg],
        p_avg_strain[good_avg],
        color="0.65",
        lw=1.0,
        alpha=0.9,
        label="Residual periodogram / mean response",
    )
    ax_s.axvspan(5.0e-4, 3.0e-3, color="C2", alpha=0.08, label="0.5-3.0 mHz")
    ax_s.axvspan(F_LO, F_HI, color="C3", alpha=0.08, label=f"default {F_LO*1e3:.1f}-{F_HI*1e3:.1f} mHz")
    ax_s.set_xlabel("Frequency [Hz]")
    ax_s.set_ylabel("Response-divided PSD [1/Hz]")
    ax_s.set_xlim(1.0e-4, 1.0e-2)
    visible = strain_positive & (freqs >= 1.0e-4) & (freqs <= 1.0e-2)
    y_candidates = [
        inst_strain[visible],
        total_strain[visible],
        gal_intrinsic_interp[intrinsic_positive],
        p_avg_strain[good_avg],
    ]
    y_all = np.concatenate([
        np.asarray(y, dtype=float).reshape(-1)
        for y in y_candidates
        if np.asarray(y).size
    ])
    y_all = y_all[np.isfinite(y_all) & (y_all > 0.0)]
    if y_all.size:
        ax_s.set_ylim(np.percentile(y_all, 1.0) / 3.0, np.percentile(y_all, 99.0) * 3.0)
    ax_s.grid(True, which="both", alpha=0.25)
    ax_s.legend(fontsize=8, loc="best")
    ax_s.set_title(
        "Response-divided strain-equivalent PSD\n"
        f"median foreground/instrument = {intrinsic_to_inst_median:.2f}, "
        f"max = {intrinsic_to_inst_max:.2f}"
    )

    fig.suptitle("LISA A-channel foreground normalization sanity check", fontsize=12)
    save_figure(fig, run_dir, "study_galactic_covariance_psd_sanity")
    print(f"Saved to {run_dir / 'study_galactic_covariance_psd_sanity.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main_synthetic(run_dir: Path = RUN_DIR) -> None:
    """Run the controlled synthetic covariance demonstration."""
    print("Running controlled synthetic covariance test …")
    d = _load_synthetic_covariance_data()
    print(
        f"Synthetic DFT band: {len(d['f_dft'])} bins in "
        f"[{F_LO*1e3:.1f}, {F_HI*1e3:.1f}] mHz, "
        f"half-bandwidth={d['max_harmonic']}, "
        f"response kappa={SYNTHETIC_RESPONSE_KAPPA}"
    )

    methods = {
        "Full frequency covariance": _log_like_dft_full_covariance,
        "Diagonal frequency covariance": _log_like_dft_diagonal,
    }
    log_surfaces: dict[str, np.ndarray] = {}
    posteriors: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}

    for name, fn in methods.items():
        print(
            f"Running {name} on {SYNTHETIC_N_GRID}x{SYNTHETIC_N_GRID} grid …",
            end=" ",
            flush=True,
        )
        log_l, elapsed = _run_grid(
            fn,
            SYNTHETIC_ALPHA_GRID,
            SYNTHETIC_LOG_FKNEE_FACTOR,
            d,
        )
        log_surfaces[name] = log_l
        posteriors[name] = _to_posterior(log_l)
        timings[name] = elapsed
        summaries[name] = _posterior_summary(
            posteriors[name],
            SYNTHETIC_ALPHA_GRID,
            SYNTHETIC_LOG_FKNEE_FACTOR,
        )
        best_i, best_j = np.unravel_index(log_l.argmax(), log_l.shape)
        fknee_best = FKNEE_FIDUCIAL * 10 ** SYNTHETIC_LOG_FKNEE_FACTOR[best_j]
        print(
            f"{elapsed:.2f}s  ->  alpha_hat={SYNTHETIC_ALPHA_GRID[best_i]:.3f} "
            f"(true={ALPHA_TRUE}), fknee_hat={fknee_best*1e3:.3f} mHz "
            f"(true={FKNEE_FIDUCIAL*1e3:.3f} mHz)"
        )

    full = summaries["Full frequency covariance"]
    diag = summaries["Diagonal frequency covariance"]
    print("\nSynthetic posterior diagnostics:")
    print(
        f"  Full covariance: std(alpha)={full['std_alpha']:.3f}, "
        f"std(log10 fknee factor)={full['std_log_fknee']:.3f}"
    )
    print(
        f"  Diagonal covariance: std(alpha)={diag['std_alpha']:.3f}, "
        f"std(log10 fknee factor)={diag['std_log_fknee']:.3f}"
    )
    print(
        f"  Diagonal/full sigma_alpha ratio: "
        f"{diag['std_alpha'] / max(full['std_alpha'], 1e-12):.2f}"
    )

    ensure_output_dir(run_dir)
    _plot_synthetic_comparison(
        log_surfaces,
        summaries,
        timings,
        SYNTHETIC_ALPHA_GRID,
        SYNTHETIC_LOG_FKNEE_FACTOR,
        run_dir,
    )


def main_real_response_synthetic(
    injection_path: Path = INJECTION_PATH,
    run_dir: Path = RUN_DIR,
) -> None:
    """Run the real-LISA-response synthetic covariance demonstration."""
    print("Running real-response synthetic covariance test …")
    d = _load_real_response_synthetic_data(injection_path)
    print(
        f"Real-response synthetic band: {len(d['f_dft'])} bins in "
        f"[{F_LO*1e3:.1f}, {F_HI*1e3:.1f}] mHz, "
        f"half-bandwidth={d['max_harmonic']}"
    )

    methods = {
        "Full covariance": _log_like_dft_full_covariance_amp,
        "Diagonal covariance": _log_like_dft_diagonal_amp,
    }
    log_surfaces: dict[str, np.ndarray] = {}
    posteriors: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}
    best_amp: dict[str, float] = {}

    for name, fn in methods.items():
        print(
            f"Running {name} on "
            f"{len(REAL_RESPONSE_LOG_AMP_GRID)}x{len(REAL_RESPONSE_ALPHA_GRID)}x"
            f"{len(REAL_RESPONSE_LOG_FKNEE_FACTOR)} grid …",
            end=" ",
            flush=True,
        )
        log_l_3d, elapsed = _run_grid_3d(
            fn,
            REAL_RESPONSE_LOG_AMP_GRID,
            REAL_RESPONSE_ALPHA_GRID,
            REAL_RESPONSE_LOG_FKNEE_FACTOR,
            d,
        )
        profiled = _profile_over_amplitude(log_l_3d)
        marginalized = _marginalize_over_amplitude(log_l_3d)
        log_surfaces[name] = profiled
        posteriors[name] = _to_posterior(marginalized)
        timings[name] = elapsed
        summaries[name] = _posterior_summary(
            posteriors[name],
            REAL_RESPONSE_ALPHA_GRID,
            REAL_RESPONSE_LOG_FKNEE_FACTOR,
        )
        best = np.unravel_index(np.argmax(log_l_3d), log_l_3d.shape)
        best_amp[name] = REAL_RESPONSE_LOG_AMP_GRID[best[0]]
        fknee_best = FKNEE_FIDUCIAL * 10 ** REAL_RESPONSE_LOG_FKNEE_FACTOR[best[2]]
        print(
            f"{elapsed:.2f}s  ->  logAfac={best_amp[name]:+.3f}, "
            f"alpha_hat={REAL_RESPONSE_ALPHA_GRID[best[1]]:.3f}, "
            f"fknee_hat={fknee_best*1e3:.3f} mHz"
        )

    full = summaries["Full covariance"]
    diag = summaries["Diagonal covariance"]
    print("\nReal-response synthetic diagnostics:")
    print(
        f"  Full covariance: std(alpha)={full['std_alpha']:.3f}, "
        f"std(log10 fknee factor)={full['std_log_fknee']:.3f}, "
        f"best logA factor={best_amp['Full covariance']:+.3f}"
    )
    print(
        f"  Diagonal covariance: std(alpha)={diag['std_alpha']:.3f}, "
        f"std(log10 fknee factor)={diag['std_log_fknee']:.3f}, "
        f"best logA factor={best_amp['Diagonal covariance']:+.3f}"
    )
    print(
        f"  Diagonal/full sigma_alpha ratio: "
        f"{diag['std_alpha'] / max(full['std_alpha'], 1e-12):.2f}"
    )

    ensure_output_dir(run_dir)
    _plot_real_response_comparison(
        log_surfaces,
        summaries,
        timings,
        REAL_RESPONSE_ALPHA_GRID,
        REAL_RESPONSE_LOG_FKNEE_FACTOR,
        run_dir,
    )


def _evaluate_real_response_band(
    d: dict,
    *,
    band_label: str,
) -> dict[str, float]:
    methods = {
        "full": _log_like_dft_full_covariance_amp,
        "diag": _log_like_dft_diagonal_amp,
    }
    results: dict[str, dict[str, float]] = {}
    for name, fn in methods.items():
        log_l_3d, elapsed = _run_grid_3d(
            fn,
            BAND_SCAN_LOG_AMP_GRID,
            BAND_SCAN_ALPHA_GRID,
            BAND_SCAN_LOG_FKNEE_FACTOR,
            d,
        )
        marginalized = _marginalize_over_amplitude(log_l_3d)
        posterior = _to_posterior(marginalized)
        summary = _posterior_summary(posterior, BAND_SCAN_ALPHA_GRID, BAND_SCAN_LOG_FKNEE_FACTOR)
        best = np.unravel_index(np.argmax(log_l_3d), log_l_3d.shape)
        results[name] = {
            "elapsed": elapsed,
            "best_log_amp": float(BAND_SCAN_LOG_AMP_GRID[best[0]]),
            "best_alpha": float(BAND_SCAN_ALPHA_GRID[best[1]]),
            "best_log_fknee": float(BAND_SCAN_LOG_FKNEE_FACTOR[best[2]]),
            "best_fknee_mhz": float(FKNEE_FIDUCIAL * 10 ** BAND_SCAN_LOG_FKNEE_FACTOR[best[2]] * 1e3),
            "std_alpha": summary["std_alpha"],
            "std_log_fknee": summary["std_log_fknee"],
        }

    harmonic = _response_harmonic_summary(d)
    f_lo_mhz = float(d["f_dft"][0] * 1e3)
    f_hi_mhz = float(d["f_dft"][-1] * 1e3)
    row = {
        "f_lo_mhz": f_lo_mhz,
        "f_hi_mhz": f_hi_mhz,
        "full_best_alpha": results["full"]["best_alpha"],
        "diag_best_alpha": results["diag"]["best_alpha"],
        "diag_minus_full_alpha": results["diag"]["best_alpha"] - results["full"]["best_alpha"],
        "full_best_fknee_mhz": results["full"]["best_fknee_mhz"],
        "diag_best_fknee_mhz": results["diag"]["best_fknee_mhz"],
        "full_std_alpha": results["full"]["std_alpha"],
        "diag_std_alpha": results["diag"]["std_alpha"],
        "sigma_alpha_ratio": results["diag"]["std_alpha"] / max(results["full"]["std_alpha"], 1e-12),
        "full_elapsed": results["full"]["elapsed"],
        "diag_elapsed": results["diag"]["elapsed"],
        "median_harmonic_ratio": harmonic["median_ratio"],
        "max_harmonic_ratio": harmonic["max_ratio"],
        "rss_harmonic_ratio": harmonic["rss_ratio"],
    }
    print(
        f"{band_label:>11s} mHz | "
        f"full alpha={row['full_best_alpha']:.3f}, diag alpha={row['diag_best_alpha']:.3f}, "
        f"ratio std_alpha={row['sigma_alpha_ratio']:.2f}, "
        f"rss |Rk|/R0={row['rss_harmonic_ratio']:.3f}, "
        f"max |Rk|/R0={row['max_harmonic_ratio']:.3f}"
    )
    return row


def main_band_scan(
    injection_path: Path = INJECTION_PATH,
    run_dir: Path = RUN_DIR,
) -> None:
    """Scan bands to see where the real response makes diagonal covariance fail."""
    print("Running real-response band scan …")
    gal_freqs, r_tf = _load_real_response_arrays(injection_path)
    rows: list[dict[str, float]] = []
    for idx, (f_lo, f_hi) in enumerate(BAND_SCAN_BANDS):
        d = _build_real_response_data_from_arrays(
            gal_freqs,
            r_tf,
            f_lo=f_lo,
            f_hi=f_hi,
            n_bins=BAND_SCAN_N_BINS,
            max_harmonic=BAND_SCAN_MAX_HARMONIC,
            seed=BAND_SCAN_SEED + idx,
        )
        rows.append(
            _evaluate_real_response_band(
                d,
                band_label=f"{f_lo*1e3:.1f}-{f_hi*1e3:.1f}",
            )
        )

    ensure_output_dir(run_dir)
    table_path = run_dir / "study_galactic_covariance_band_scan.csv"
    header = list(rows[0].keys())
    with table_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{row[key]:.8g}" for key in header) + "\n")
    print(f"Saved to {table_path}")
    _plot_band_scan(rows, run_dir)


def main_injection(injection_path: Path = INJECTION_PATH, run_dir: Path = RUN_DIR) -> None:
    print(f"Loading injection from {injection_path} …")
    d = _load_data(injection_path)
    print(
        f"WDM: NT={NT}, n_band={d['coeffs_wdm'].shape[1]} freq channels, "
        f"t_obs={d['t_obs_wdm']/86400:.1f} days"
    )
    print(
        f"DFT covariance band: {len(d['f_dft'])} bins in "
        f"[{F_LO*1e3:.1f}, {F_HI*1e3:.1f}] mHz, "
        f"half-bandwidth={d['max_harmonic']}"
    )

    methods = {
        "Method 1\n(WDM diagonal)":   _log_like_wdm,
        "Method 2\n(DFT full covariance)": _log_like_dft_full_covariance,
        "Method 3\n(DFT diagonal)":  _log_like_dft_diagonal,
    }

    log_surfaces: dict[str, np.ndarray] = {}
    posteriors: dict[str, np.ndarray] = {}
    timings:    dict[str, float]      = {}
    summaries:  dict[str, dict[str, float]] = {}

    for name, fn in methods.items():
        print(f"Running {name.replace(chr(10), ' ')} on {N_GRID}×{N_GRID} grid …", end=" ", flush=True)
        log_L, elapsed = _run_grid(fn, ALPHA_GRID, LOG_FKNEE_FACTOR, d)
        log_surfaces[name] = log_L
        posteriors[name] = _to_posterior(log_L)
        timings[name]    = elapsed
        summaries[name]   = _posterior_summary(posteriors[name], ALPHA_GRID, LOG_FKNEE_FACTOR)
        best_i, best_j   = np.unravel_index(log_L.argmax(), log_L.shape)
        fknee_best = FKNEE_FIDUCIAL * 10 ** LOG_FKNEE_FACTOR[best_j]
        print(
            f"{elapsed:.2f}s  →  α̂={ALPHA_GRID[best_i]:.3f} "
            f"(true={ALPHA_TRUE}),  f̂knee={fknee_best*1e3:.3f} mHz "
            f"(true={FKNEE_FIDUCIAL*1e3:.3f} mHz)"
        )

    print("\nPosterior width diagnostics:")
    for name, summary in summaries.items():
        print(
            f"  {name.replace(chr(10), ' ')}: "
            f"std(alpha)={summary['std_alpha']:.3f}, "
            f"std(log10 fknee factor)={summary['std_log_fknee']:.3f}, "
            f"68% HPD cells={summary['hpd_cells_68']:.0f}, "
            f"effective cells={summary['effective_cells']:.1f}"
        )

    full_key = "Method 2\n(DFT full covariance)"
    wdm_key = "Method 1\n(WDM diagonal)"
    diag_key = "Method 3\n(DFT diagonal)"
    print(
        "\nMethod comparison:"
        f"\n  WDM speedup vs full covariance: {timings[full_key] / max(timings[wdm_key], 1e-12):.1f}x"
        f"\n  Diagonal DFT speedup vs full covariance: {timings[full_key] / max(timings[diag_key], 1e-12):.1f}x"
        f"\n  WDM/full Δmean alpha: {summaries[wdm_key]['mean_alpha'] - summaries[full_key]['mean_alpha']:+.3f}"
        f"\n  Diagonal/full σ_alpha ratio: {summaries[diag_key]['std_alpha'] / max(summaries[full_key]['std_alpha'], 1e-12):.2f}"
    )

    ensure_output_dir(run_dir)
    _plot_comparison(log_surfaces, summaries, timings, ALPHA_GRID, LOG_FKNEE_FACTOR, run_dir)


def main() -> None:
    ensure_output_dir(RUN_DIR)
    _plot_psd_sanity_check(INJECTION_PATH, RUN_DIR)
    main_band_scan()


if __name__ == "__main__":
    main()
