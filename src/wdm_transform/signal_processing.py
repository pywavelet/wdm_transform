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
    """Matched-filter SNR for real WDM coefficients with diagonal noise variance.

    NaN entries in ``noise_var`` (used to flag rank-deficient DC/Nyquist
    columns) are excluded from the sum.  Callers may therefore pass the full
    ``(nt, nf+1)`` arrays returned by :func:`wdm_noise_variance` directly;
    the edge channels will be skipped automatically.
    """
    coeffs_arr = np.asarray(coeffs, dtype=float)
    var = np.asarray(noise_var, dtype=float)
    mask = np.isfinite(var) & (var > 0)
    snr2 = np.sum(np.where(mask, coeffs_arr**2 / np.where(mask, var, 1.0), 0.0))
    return float(np.sqrt(max(float(snr2), 0.0)))


def wdm_noise_variance(
    noise_psd: np.ndarray,
    *,
    nt: int,
    nf: int,
    dt: float,
) -> np.ndarray:
    """Diagonal WDM noise variance for interior channels.

    For stationary noise with one-sided PSD ``S(f)``, the per-pixel WDM
    coefficient variance under the discrete-FFT normalisation used here is

        sigma^2_{nm} = N * S(f_m) / (2 * dt)   for  1 <= m <= nf - 1

    where ``N = nt * nf``.

    The DC (m=0) and Nyquist (m=nf) edge channels are returned as ``NaN``.
    Those channels are overcomplete (rank-deficient) and must be excluded from
    any diagonal-Whittle SNR or likelihood sum.  :func:`matched_filter_snr_wdm`
    skips NaN entries automatically, so callers can pass the full arrays
    returned by this function without manual slicing.

    Parameters
    ----------
    noise_psd:
        One-sided physical PSD sampled at the WDM channel frequencies
        ``f_m = m * Delta_F`` for ``m = 0, 1, …, nf``; shape ``(nf + 1,)``.
    nt, nf:
        WDM grid dimensions.
    dt:
        Sampling interval of the underlying time series.

    Returns
    -------
    sigma2 : ndarray, shape ``(nt, nf + 1)``
        Per-pixel WDM variance, with DC (column 0) and Nyquist (column nf)
        set to ``NaN`` as a deliberate guardrail against including the
        rank-deficient edge channels in sums.
    """
    if nt <= 0:
        raise ValueError("nt must be positive.")
    if nf <= 0:
        raise ValueError("nf must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    psd = np.asarray(noise_psd, dtype=float)
    if psd.shape[-1] != nf + 1:
        raise ValueError(
            f"noise_psd must have length nf+1 = {nf + 1}, got {psd.shape[-1]}."
        )
    N = nt * nf
    var_row = N * psd / (2.0 * dt)
    out = np.broadcast_to(var_row[None, :], (nt, nf + 1)).copy()
    out[:, 0] = np.nan
    out[:, nf] = np.nan
    return out


__all__ = [
    "matched_filter_snr_rfft",
    "matched_filter_snr_wdm",
    "noise_characteristic_strain",
    "rfft_characteristic_strain",
    "wdm_noise_variance",
]
