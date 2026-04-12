from __future__ import annotations

import numpy as np
import pytest

from wdm_transform import (
    matched_filter_snr_rfft,
    matched_filter_snr_wdm,
    noise_characteristic_strain,
    rfft_characteristic_strain,
    wdm_noise_variance,
)


def test_matched_filter_snr_rfft_matches_manual_one_sided_formula() -> None:
    dt = 0.125
    n_total = 64
    freqs = np.fft.rfftfreq(n_total, d=dt)
    coeffs = np.zeros_like(freqs, dtype=np.complex128)
    coeffs[3] = 2.0 + 0.5j
    coeffs[5] = -1.2j
    noise_psd = 1.0 + 0.2 * freqs

    snr = matched_filter_snr_rfft(coeffs, noise_psd, freqs, dt=dt)

    pos = freqs > 0.0
    df = freqs[pos][1] - freqs[pos][0]
    expected = np.sqrt(
        4.0 * df * np.sum(np.abs(dt * coeffs[pos]) ** 2 / noise_psd[pos])
    )
    np.testing.assert_allclose(snr, expected)


def test_matched_filter_snr_rfft_returns_zero_when_df_is_undefined() -> None:
    coeffs = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    freqs = np.array([0.0, 1.0], dtype=float)
    noise_psd = np.array([1.0, 1.0], dtype=float)

    assert matched_filter_snr_rfft(coeffs, noise_psd, freqs, dt=0.5) == 0.0


def test_matched_filter_snr_wdm_matches_diagonal_inner_product() -> None:
    coeffs = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=float)
    noise_var = np.array([[4.0, 2.0], [0.5, 9.0]], dtype=float)

    snr = matched_filter_snr_wdm(coeffs, noise_var)
    expected = np.sqrt(np.sum(coeffs**2 / noise_var))

    np.testing.assert_allclose(snr, expected)


def test_wdm_noise_variance_matches_parseval_scaling() -> None:
    noise_psd = np.array([2.0, 4.0, 8.0], dtype=float)
    nt = 4
    dt = 0.25

    variance = wdm_noise_variance(noise_psd, nt=nt, dt=dt)

    expected_row = noise_psd / (2.0 * dt)
    assert variance.shape == (nt, noise_psd.size)
    np.testing.assert_allclose(variance, np.tile(expected_row, (nt, 1)))


@pytest.mark.parametrize(
    ("nt", "dt"),
    [
        (0, 0.25),
        (4, 0.0),
    ],
)
def test_wdm_noise_variance_validates_inputs(nt: int, dt: float) -> None:
    with pytest.raises(ValueError):
        wdm_noise_variance(np.ones(3), nt=nt, dt=dt)


def test_characteristic_strain_helpers_match_manual_definitions() -> None:
    dt = 0.25
    freqs = np.array([0.0, 0.5, 1.0], dtype=float)
    coeffs = np.array([3.0 + 4.0j, 1.0 - 2.0j, -2.0 + 0.5j], dtype=np.complex128)
    noise_psd = np.array([10.0, 8.0, 6.0], dtype=float)

    h_c = rfft_characteristic_strain(coeffs, freqs, dt)
    h_n = noise_characteristic_strain(noise_psd, freqs)

    expected_h_c = np.array(
        [
            0.0,
            2.0 * freqs[1] * abs(dt * coeffs[1]),
            2.0 * freqs[2] * abs(dt * coeffs[2]),
        ]
    )
    expected_h_n = np.array([0.0, np.sqrt(freqs[1] * noise_psd[1]), np.sqrt(freqs[2] * noise_psd[2])])

    np.testing.assert_allclose(h_c, expected_h_c)
    np.testing.assert_allclose(h_n, expected_h_n)
