from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from data_generation import draw_rfft_from_psd
from demo_wdm_nonstationary_whitening import (
    _compute_fractional_miscalibration,
    _load_injection_with_check,
    _source_time_series_for_truncation,
    _stationary_variance,
)
from lisa_common import build_wdm_nonstationary_variance, noise_tdi15_psd
from wdm_transform.signal_processing import wdm_noise_variance
from wdm_transform.transforms import from_time_to_wdm


def test_rfft_psd_draw_matches_whittle_convention() -> None:
    rng = np.random.default_rng(1234)
    dt = 0.25
    n_total = 512
    df = 1.0 / (n_total * dt)
    psd = np.full(n_total // 2 + 1, 3.5)

    draws = np.stack(
        [draw_rfft_from_psd(psd, rng=rng, df=df, dt=dt) for _ in range(4000)]
    )

    interior = slice(1, -1)
    whitened_power = (
        2.0 * df * dt**2 * np.abs(draws[:, interior]) ** 2 / psd[interior]
    )
    assert whitened_power.mean() == pytest.approx(1.0, rel=0.03)


@dataclass
class _FakeInjection:
    dt: float
    freqs: np.ndarray
    noise_psd_A: np.ndarray
    noise_psd_E: np.ndarray
    noise_psd_T: np.ndarray


def test_stationary_wdm_variance_uses_wdm_coefficient_normalization() -> None:
    dt = 0.25
    n_freqs_full = 99
    n_total = 2 * (n_freqs_full - 1)
    freqs = np.array([0.0, 1.0, 2.0, 3.0])
    injection = _FakeInjection(
        dt=dt,
        freqs=freqs,
        noise_psd_A=np.array([1.0, 2.0, 3.0, 4.0]),
        noise_psd_E=np.array([2.0, 4.0, 6.0, 8.0]),
        noise_psd_T=np.array([3.0, 6.0, 9.0, 12.0]),
    )
    wdm_freq_centers = np.array([1.0, 2.0])

    variance = _stationary_variance(
        injection,
        wdm_freq_centers=wdm_freq_centers,
        nt=3,
        n_freqs_full=n_freqs_full,
    )

    expected_psd = np.array([[2.0, 3.0], [4.0, 6.0], [6.0, 9.0]])
    expected = np.broadcast_to(
        (n_total * expected_psd / (2.0 * dt))[:, None, :],
        (3, 3, 2),
    )
    np.testing.assert_allclose(variance, expected, rtol=1e-12, atol=0.0)


def test_nonstationary_wdm_variance_uses_nearest_time_psd() -> None:
    dt = 0.5
    n_freqs_full = 999
    n_total = 2 * (n_freqs_full - 1)
    wdm_freq_centers = np.array([1.0e-3, 2.0e-3])
    wdm_time_centers = np.array([0.2, 8.8])
    gal_psd_freqs = np.array([1.0e-3, 2.0e-3])
    gal_psd_a = np.array(
        [
            [1.0e-40, 2.0e-40],
            [3.0e-40, 4.0e-40],
        ]
    )
    inj_npz = {
        "gal_psd_A_tf": gal_psd_a,
        "gal_psd_E_tf": np.zeros_like(gal_psd_a),
        "gal_psd_T_tf": np.zeros_like(gal_psd_a),
        "gal_psd_freqs": gal_psd_freqs,
        "gal_psd_times": np.array([0.0, 10.0]),
    }

    variance = build_wdm_nonstationary_variance(
        inj_npz,
        channel=0,
        wdm_freq_centers=wdm_freq_centers,
        wdm_time_centers=wdm_time_centers,
        dt=dt,
        n_freqs_full=n_freqs_full,
    )

    inst = noise_tdi15_psd(0, wdm_freq_centers)
    expected = np.stack(
        [
            n_total * (inst + gal_psd_a[0]) / (2.0 * dt),
            n_total * (inst + gal_psd_a[1]) / (2.0 * dt),
        ]
    )
    np.testing.assert_allclose(variance, expected, rtol=1e-12, atol=0.0)


def test_wdm_variance_matches_psd_normalized_noise_draws() -> None:
    nt = 32
    nf = 64
    n_total = nt * nf
    dt = 0.25
    df = 1.0 / (n_total * dt)
    psd_value = 3.5
    psd = np.full(n_total // 2 + 1, psd_value)
    rng = np.random.default_rng(5678)
    variance = n_total * wdm_noise_variance(
        np.full(nf + 1, psd_value),
        nt=nt,
        dt=dt,
    )
    whitened_means = []
    for _ in range(160):
        spectrum = draw_rfft_from_psd(psd, rng=rng, df=df, dt=dt)
        time_series = np.fft.irfft(spectrum, n=n_total)
        coeffs = np.asarray(
            from_time_to_wdm(
                time_series,
                nt=nt,
                nf=nf,
                a=1.0 / 3.0,
                d=1.0,
                dt=dt,
                backend="numpy",
            )
        )[0]
        whitened_means.append((coeffs[:, 1:-1] ** 2 / variance[:, 1:-1]).mean())

    assert np.mean(whitened_means) == pytest.approx(1.0, rel=0.08)


def test_source_subtraction_reconstructs_before_truncating() -> None:
    full_length = 16
    n_keep = 12
    time_series = np.arange(full_length, dtype=float)
    source_f = np.fft.rfft(time_series)

    reconstructed = _source_time_series_for_truncation(source_f, full_length, n_keep)
    incorrectly_reinterpreted = np.fft.irfft(source_f, n=n_keep)

    np.testing.assert_allclose(reconstructed, time_series[:n_keep])
    assert not np.allclose(reconstructed, incorrectly_reinterpreted)


def test_fractional_miscalibration_zero_when_equal() -> None:
    sigma2 = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    result = _compute_fractional_miscalibration(sigma2, sigma2)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_fractional_miscalibration_sign_and_magnitude() -> None:
    sigma2_stat = np.array([[[1.0, 4.0]]])          # (1, 1, 2)
    sigma2_nonstat = np.array([[[1.2, 3.6]]])
    result = _compute_fractional_miscalibration(sigma2_stat, sigma2_nonstat)
    np.testing.assert_allclose(result[0, 0, 0], 0.2, rtol=1e-12)
    np.testing.assert_allclose(result[0, 0, 1], -0.1, rtol=1e-12)


def test_stale_injection_without_normalization_version_exits(tmp_path) -> None:
    injection_path = tmp_path / "injection.npz"
    np.savez(
        injection_path,
        gal_psd_A_tf=np.zeros((1, 1)),
        gal_psd_E_tf=np.zeros((1, 1)),
        gal_psd_T_tf=np.zeros((1, 1)),
        gal_psd_freqs=np.array([1.0]),
        gal_psd_times=np.array([0.0]),
    )

    with pytest.raises(SystemExit) as excinfo:
        _load_injection_with_check(injection_path)

    assert "normalization_version" in str(excinfo.value)
    assert "Regenerate the injection" in str(excinfo.value)
