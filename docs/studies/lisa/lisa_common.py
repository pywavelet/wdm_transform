"""Shared constants, paths, PSD models, and utilities for the LISA GB study."""
from __future__ import annotations

from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
BACKGROUND_DIR = STUDY_DIR / "outdir_gb_background"
FREQ_ASSET_DIR = STUDY_DIR / "lisa_freq_mcmc_assets"
WDM_ASSET_DIR = STUDY_DIR / "lisa_wdm_mcmc_assets"

RESPONSE_TENSOR_PATH = BACKGROUND_DIR / "Rtildeop_tf.npz"
INJECTION_PATH = BACKGROUND_DIR / "injection.npz"

c = 299792458.0
L_LISA = 2.5e9

# Canonical injected GB source parameters: [f0, fdot, A, ra, dec, psi, iota, phi0]
SOURCE_PARAMS = np.array(
    [
        [1.35962e-3, 8.94581279e-19, 1.07345e-22, 2.40, 0.31, 3.56, 0.52, 3.06],
        [1.41220e-3, -2.30000000e-18, 8.20000000e-23, 2.15, 0.18, 1.20, 0.93, 1.40],
    ],
    dtype=float,
)


# ── Filesystem helpers ────────────────────────────────────────────────────────


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_dir: Path, stem: str, *, dpi: int = 160) -> Path:
    ensure_output_dir(output_dir)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return path


# ── Signal utilities ─────────────────────────────────────────────────────────


def wrap_phase(phi):
    """Wrap angle to (-π, π].  Works on scalars, NumPy, and JAX arrays."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def floor_pow2(n: int) -> int:
    """Largest power of 2 that is ≤ n."""
    return 1 << (int(n).bit_length() - 1)


def place_local_tdi(segment, kmin: int, n_freqs: int) -> np.ndarray:
    """Place a band-limited TDI segment into a zero-padded full-length frequency array."""
    full = np.zeros(n_freqs, dtype=np.complex128)
    seg = np.asarray(segment, dtype=np.complex128).reshape(-1)
    end = min(kmin + seg.size, n_freqs)
    if end > kmin:
        full[kmin:end] = seg[: end - kmin]
    return full


def rfft_characteristic_strain(
    coeffs: np.ndarray,
    freqs: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Convert NumPy rFFT coefficients into characteristic strain."""
    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)
    freqs_arr = np.asarray(freqs, dtype=float)
    h_c = np.zeros_like(freqs_arr, dtype=float)
    pos = freqs_arr > 0.0
    h_c[pos] = 2.0 * freqs_arr[pos] * np.abs(dt * coeffs_arr[pos])
    return h_c


def noise_characteristic_strain(noise_psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Convert one-sided PSD to characteristic noise strain."""
    noise_psd_arr = np.asarray(noise_psd, dtype=float)
    freqs_arr = np.asarray(freqs, dtype=float)
    h_n = np.zeros_like(freqs_arr, dtype=float)
    pos = freqs_arr > 0.0
    h_n[pos] = np.sqrt(freqs_arr[pos] * np.maximum(noise_psd_arr[pos], 0.0))
    return h_n


def matched_filter_snr_rfft(
    coeffs: np.ndarray,
    noise_psd: np.ndarray,
    freqs: np.ndarray,
    *,
    dt: float,
) -> float:
    """Matched-filter SNR using one-sided PSD and NumPy rFFT coefficients."""
    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)
    noise_psd_arr = np.asarray(noise_psd, dtype=float)
    freqs_arr = np.asarray(freqs, dtype=float)
    pos = freqs_arr > 0.0
    if pos.sum() < 2:
        return 0.0
    df = float(freqs_arr[pos][1] - freqs_arr[pos][0])
    h_tilde = dt * coeffs_arr[pos]
    snr2 = 4.0 * df * np.sum(np.abs(h_tilde) ** 2 / np.maximum(noise_psd_arr[pos], 1e-60))
    return float(np.sqrt(max(float(np.real(snr2)), 0.0)))


def matched_filter_snr_wdm(
    coeffs: np.ndarray,
    noise_var: np.ndarray,
) -> float:
    """Matched-filter SNR for real WDM coefficients with diagonal noise variance."""
    coeffs_arr = np.asarray(coeffs, dtype=float)
    noise_var_arr = np.asarray(noise_var, dtype=float)
    snr2 = np.sum(coeffs_arr**2 / np.maximum(noise_var_arr, 1e-60))
    return float(np.sqrt(max(float(snr2), 0.0)))


# ── Frequency-axis models ─────────────────────────────────────────────────────


def freqs_gal(
    nfreqs: int = 500,
    fmin_gal: float = 1e-4,
    fmax_gal: float = 3e-3,
) -> np.ndarray:
    return np.linspace(fmin_gal, fmax_gal, nfreqs)


def galactic_psd(
    f,
    Tobsyr: float = 2.0,
    A_gal: float = 10**-43.9,
    alp_gal: float = 1.8,
    a1_gal: float = -0.25,
    b1_gal: float = -2.7,
    ak_gal: float = -0.27,
    bk_gal: float = -2.47,
    f2_gal: float = 10**-3.5,
):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0.0, f, 1.0)
    f1_gal = 10 ** (a1_gal * np.log10(Tobsyr) + b1_gal)
    fknee_gal = 10 ** (ak_gal * np.log10(Tobsyr) + bk_gal)
    return (
        A_gal
        * f_safe ** (-7.0 / 3.0)
        * np.exp(-((f_safe / f1_gal) ** alp_gal))
        * (1.0 + np.tanh((fknee_gal - f_safe) / f2_gal))
    )


def omega_gw(f, Sh):
    H0 = 2.2e-18
    return (4 * np.pi**2) / (3 * H0**2) * np.asarray(Sh) * np.asarray(f) ** 3


def _ntilda_e(f, A: float = 3.0, P: float = 15.0, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0, f, 1.0)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return (
        0.5
        * (2.0 + np.cos(f_safe / fstar))
        * (P / L) ** 2
        * 1e-24
        * (1.0 + (0.002 / f_safe) ** 4)
        + 2.0
        * (1.0 + np.cos(f_safe / fstar) + np.cos(f_safe / fstar) ** 2)
        * (A / L) ** 2
        * 1e-30
        * (1.0 + (0.0004 / f_safe) ** 2)
        * (1.0 + (f_safe / 0.008) ** 4)
        * (1.0 / (2.0 * np.pi * f_safe)) ** 4
    )


def _ntilda_t(f, A: float = 3.0, P: float = 15.0, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0, f, 1.0)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return (1.0 - np.cos(f_safe / fstar)) * (P / L) ** 2 * 1e-24 * (
        1.0 + (0.002 / f_safe) ** 4
    ) + 2.0 * (1.0 - np.cos(f_safe / fstar)) ** 2 * (A / L) ** 2 * 1e-30 * (
        1.0 + (0.0004 / f_safe) ** 2
    ) * (1.0 + (f_safe / 0.008) ** 4) * (1.0 / (2.0 * np.pi * f_safe)) ** 4


def tdi15_factor(f, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return 4.0 * np.sin(f / fstar) * f / fstar


def noise_tdi15_psd(channel: int, f, L: float = L_LISA):
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros_like(f_arr, dtype=float)
    pos = f_arr > 0.0
    if np.any(pos):
        base = _ntilda_t if channel == 2 else _ntilda_e
        out[pos] = base(f_arr[pos], L=L) * tdi15_factor(f_arr[pos], L=L)
    return float(out) if np.isscalar(f) else out


def noise_tdi15_a_psd(f, L: float = L_LISA):
    return noise_tdi15_psd(0, f, L=L)


def build_total_noise_psd(
    Rtildeop_tf: np.ndarray,
    freqs_response: np.ndarray,
    target_freqs: np.ndarray,
    channel: int = 0,
) -> np.ndarray:
    """Instrument noise + time-averaged galactic foreground PSD on *target_freqs*."""
    fg_time = np.abs(Rtildeop_tf[channel, channel]) * tdi15_factor(freqs_response)
    fg_mean = np.mean(fg_time, axis=0)
    fg_interp = np.interp(target_freqs, freqs_response, fg_mean, left=0.0, right=0.0)
    return np.maximum(noise_tdi15_psd(channel, target_freqs) + fg_interp, 1e-60)


# ── WDM noise utilities ───────────────────────────────────────────────────────


def wdm_noise_variance(
    noise_psd: np.ndarray,
    freq_grid: np.ndarray,
    nt: int,
) -> np.ndarray:
    """WDM pixel variance S(f_m) × Δf broadcast over *nt* time bins."""
    delta_f = float(freq_grid[1] - freq_grid[0])
    var_row = np.maximum(
        np.interp(freq_grid, freq_grid, noise_psd) * delta_f,
        1e-60,
    )
    # broadcast_to avoids allocating nt identical rows until .copy() is called
    return np.broadcast_to(var_row[None, :], (nt, len(var_row))).copy()


def trim_frequency_band(
    freqs: np.ndarray,
    f_lo: float,
    f_hi: float,
    pad_bins: int = 2,
) -> slice:
    """Slice of *freqs* grid covering [f_lo, f_hi] with optional padding bins."""
    keep = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    if keep.size == 0:
        raise ValueError(f"No bins in [{f_lo:.3e}, {f_hi:.3e}] Hz.")
    return slice(
        max(int(keep[0]) - pad_bins, 0),
        min(int(keep[-1]) + pad_bins + 1, len(freqs)),
    )


# ── Posterior diagnostics ─────────────────────────────────────────────────────


def print_posterior_summary(
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
) -> None:
    """Print median and 90 % CI for each parameter alongside the true value."""
    med = np.median(samples, axis=0)
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    header = f"{'Parameter':<24} {'True':>12} {'Median':>12} {'5%':>12} {'95%':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, true_v, median, low, high in zip(param_names, truth, med, lo, hi):
        print(
            f"  {name:<22} {true_v:>12.4e} {median:>12.4e} {low:>12.4e} {high:>12.4e}"
        )


def check_posterior_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
    ci: float = 0.9,
) -> np.ndarray:
    """Print CI coverage and return boolean mask of covered parameters."""
    lo = np.percentile(samples, 100.0 * (1 - ci) / 2, axis=0)
    hi = np.percentile(samples, 100.0 * (1 + ci) / 2, axis=0)
    covered = (truth >= lo) & (truth <= hi)
    print(f"\n{ci * 100:.0f}% posterior coverage:")
    for name, true_v, low, high, in_ci in zip(param_names, truth, lo, hi, covered):
        mark = "✓" if in_ci else "✗"
        print(f"  {mark} {name:<22}: true={true_v:+.4e}  [{low:+.4e}, {high:+.4e}]")
    print(f"  {int(covered.sum())}/{len(covered)} parameters covered")
    return covered
