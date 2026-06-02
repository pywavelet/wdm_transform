from __future__ import annotations

import numpy as np
import pytest

from wdm_transform import (
    TimeSeries,
    WDM,
    matched_filter_snr_rfft,
    matched_filter_snr_wdm,
    noise_characteristic_strain,
    rfft_characteristic_strain,
    rfft_periodogram_characteristic_strain,
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


def test_wdm_noise_variance_returns_N_S_over_2dt_for_interior() -> None:
    """Interior entries return N*S/(2*dt); DC and Nyquist columns are NaN."""
    nt, nf, dt = 4, 4, 0.25
    N = nt * nf
    noise_psd = np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)  # length nf+1

    sigma2 = wdm_noise_variance(noise_psd, nt=nt, nf=nf, dt=dt)

    expected_interior = N * noise_psd / (2.0 * dt)
    assert sigma2.shape == (nt, nf + 1)
    for m in range(1, nf):
        np.testing.assert_allclose(sigma2[:, m], expected_interior[m])
    assert np.all(np.isnan(sigma2[:, 0]))
    assert np.all(np.isnan(sigma2[:, nf]))


@pytest.mark.parametrize(
    ("nt", "nf", "dt"),
    [
        (0, 2, 0.25),
        (4, 0, 0.25),
        (4, 2, 0.0),
    ],
)
def test_wdm_noise_variance_validates_inputs(nt: int, nf: int, dt: float) -> None:
    with pytest.raises(ValueError):
        wdm_noise_variance(np.ones(nf + 1), nt=nt, nf=nf, dt=dt)


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


def test_rfft_periodogram_characteristic_strain_matches_psd_estimator() -> None:
    dt = 0.25
    freqs = np.array([0.0, 0.5, 1.0], dtype=float)
    coeffs = np.array([3.0 + 4.0j, 1.0 - 2.0j, -2.0 + 0.5j], dtype=np.complex128)

    h_c = rfft_periodogram_characteristic_strain(coeffs, freqs, dt)

    df = freqs[2] - freqs[1]
    psd_hat = 2.0 * df * dt**2 * np.abs(coeffs[1:]) ** 2
    expected = np.array([0.0, *(np.sqrt(freqs[1:] * psd_hat))])
    np.testing.assert_allclose(h_c, expected)


def test_wdm_per_pixel_variance_matches_monte_carlo() -> None:
    """Empirical <w_nm^2> from white-noise realisations matches N*S0/(2*dt)."""
    nt, nf, dt = 32, 32, 1.0
    N = nt * nf
    S0 = 1.5
    sigma_t = np.sqrt(S0 / (2.0 * dt))
    rng = np.random.default_rng(0)
    n_real = 5000

    var_acc = np.zeros((nt, nf + 1))
    for _ in range(n_real):
        x = rng.standard_normal(N) * sigma_t
        w = WDM.from_time_series(TimeSeries(x, dt=dt), nt=nt).coeffs[0]
        var_acc += np.asarray(w) ** 2
    emp = var_acc / n_real

    expected_interior = N * S0 / (2.0 * dt)
    expected_edge = N * S0 / (4.0 * dt)
    # 5% tolerance from Monte Carlo noise
    np.testing.assert_allclose(np.mean(emp[:, 1:nf]), expected_interior, rtol=0.05)
    np.testing.assert_allclose(np.mean(emp[:, 0]), expected_edge, rtol=0.05)
    np.testing.assert_allclose(np.mean(emp[:, nf]), expected_edge, rtol=0.05)


def test_diagonal_wdm_snr_matches_fd_snr_for_atom_locally_flat_psd() -> None:
    """WDM diagonal Whittle SNR equals FD SNR for a tone at a channel centre
    with flat PSD, summed over interior channels only."""
    nt, nf, dt = 64, 64, 1.0
    N = nt * nf
    delta_F = 1.0 / (2 * nf * dt)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Tone at channel 24 centre, flat PSD
    t = np.arange(N) * dt
    f0 = 24 * delta_F
    h = np.sin(2.0 * np.pi * f0 * t)
    S_flat = 3.0
    S_rfft = np.full_like(freqs, S_flat)
    S_at_wdm = np.full(nf + 1, S_flat)

    snr_fd = matched_filter_snr_rfft(np.fft.rfft(h), S_rfft, freqs, dt=dt)
    w = WDM.from_time_series(TimeSeries(h, dt=dt), nt=nt).coeffs[0]
    sigma2 = wdm_noise_variance(S_at_wdm, nt=nt, nf=nf, dt=dt)
    snr_wdm = matched_filter_snr_wdm(w[:, 1:nf], sigma2[:, 1:nf])

    np.testing.assert_allclose(snr_wdm, snr_fd, rtol=1e-10)
