"""Generic spectral and WDM-domain signal-processing helpers."""

from __future__ import annotations

import numpy as np


def rfft_characteristic_strain(
    coeffs: np.ndarray,
    freqs: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Convert one-sided rFFT coefficients into characteristic strain."""
    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)
    freqs_arr = np.asarray(freqs, dtype=float)
    h_c = np.zeros_like(freqs_arr, dtype=float)
    pos = freqs_arr > 0.0
    h_c[pos] = 2.0 * freqs_arr[pos] * np.abs(dt * coeffs_arr[pos])
    return h_c


def noise_characteristic_strain(noise_psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Convert a one-sided PSD into characteristic noise strain."""
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


def wdm_noise_variance(
    noise_psd: np.ndarray,
    *,
    nt: int,
    dt: float,
) -> np.ndarray:
    """Broadcast one-sided PSD samples into diagonal WDM coefficient variance.

    For a stationary process with one-sided physical PSD ``S_n(f_m)``, the
    per-pixel WDM variance is

        E[w[n, m]^2] = S_n(f_m) / (2 * dt),

    which is independent of the WDM frequency-bin spacing. The input PSD must
    already be sampled at the WDM frequency channels of interest.
    """
    if nt <= 0:
        raise ValueError("nt must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    f_nyquist = 1.0 / (2.0 * dt)
    var_row = np.maximum(np.asarray(noise_psd, dtype=float) * f_nyquist, 1e-60)
    return np.broadcast_to(var_row[None, :], (nt, len(var_row))).copy()


__all__ = [
    "matched_filter_snr_rfft",
    "matched_filter_snr_wdm",
    "noise_characteristic_strain",
    "rfft_characteristic_strain",
    "wdm_noise_variance",
]
