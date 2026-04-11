from __future__ import annotations

import numpy as np

from wdm_transform.transforms import from_time_to_wdm


def _wdm_analysis_matrix(*, nt: int, nf: int, dt: float, a: float = 1.0 / 3.0) -> np.ndarray:
    """Dense forward WDM operator for a small problem size."""
    n_total = nt * nf
    matrix = np.empty((nt * (nf + 1), n_total), dtype=float)
    for idx in range(n_total):
        basis = np.zeros(n_total, dtype=float)
        basis[idx] = 1.0
        matrix[:, idx] = from_time_to_wdm(
            basis,
            nt=nt,
            nf=nf,
            a=a,
            d=1.0,
            dt=dt,
            backend="numpy",
        ).reshape(-1)
    return matrix


def test_wdm_matched_filter_snr_matches_fft_for_white_noise() -> None:
    nt = nf = 16
    dt = 1.0
    sigma = 0.3
    n_total = nt * nf

    times = np.arange(n_total, dtype=float)
    k_bin = 13
    signal = 0.8 * np.sin(2.0 * np.pi * k_bin * times / n_total)

    analysis = _wdm_analysis_matrix(nt=nt, nf=nf, dt=dt)
    signal_wdm = analysis @ signal

    # For white time-domain noise with variance sigma^2, the exact WDM
    # covariance is sigma^2 * A A^T where A is the forward analysis operator.
    noise_cov_wdm = sigma**2 * (analysis @ analysis.T)
    snr2_wdm = float(signal_wdm @ (np.linalg.pinv(noise_cov_wdm, rcond=1e-12) @ signal_wdm))

    signal_fft = np.fft.fft(signal)
    snr2_fft = float(np.sum(np.abs(signal_fft) ** 2) / (n_total * sigma**2))

    np.testing.assert_allclose(snr2_wdm, snr2_fft, rtol=1e-12, atol=1e-12)
