"""Tests for input validation and error paths."""

from __future__ import annotations

import numpy as np
import pytest

from wdm_transform import FrequencySeries, TimeSeries, WDM


class TestWDMValidation:
    def test_odd_nt_raises(self) -> None:
        data = np.zeros((3, 4), dtype=complex)
        with pytest.raises(ValueError, match="nt and nf must both be even"):
            WDM(coeffs=data, dt=1.0)

    def test_odd_nf_raises(self) -> None:
        data = np.zeros((4, 3), dtype=complex)
        with pytest.raises(ValueError, match="nt and nf must both be even"):
            WDM(coeffs=data, dt=1.0)

    def test_bad_window_parameter_raises(self) -> None:
        data = np.zeros((4, 4), dtype=complex)
        with pytest.raises(ValueError, match="a must be in"):
            WDM(coeffs=data, dt=1.0, a=0.0)
        with pytest.raises(ValueError, match="a must be in"):
            WDM(coeffs=data, dt=1.0, a=0.5)

    def test_negative_dt_raises(self) -> None:
        data = np.zeros((4, 4), dtype=complex)
        with pytest.raises(ValueError, match="dt must be positive"):
            WDM(coeffs=data, dt=-1.0)

    def test_1d_coeffs_raises(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            WDM(coeffs=np.zeros(16), dt=1.0)

    def test_from_time_series_indivisible_length(self) -> None:
        series = TimeSeries(np.zeros(100), dt=1.0)
        with pytest.raises(ValueError, match="not divisible"):
            WDM.from_time_series(series, nt=7)

    def test_forward_wrong_length_raises(self) -> None:
        series = TimeSeries(np.zeros(128), dt=1.0)
        with pytest.raises(ValueError, match="not divisible"):
            WDM.from_time_series(series, nt=6)


class TestWDMRepr:
    def test_repr_is_compact(self) -> None:
        data = np.zeros((4, 4), dtype=complex)
        w = WDM(coeffs=data, dt=0.5)
        r = repr(w)
        assert "nt=4" in r
        assert "nf=4" in r
        assert "dt=0.5" in r
        assert "array" not in r.lower()


class TestEdgeChannelAccessors:
    def test_dc_and_nyquist_channels(self) -> None:
        nt, nf, dt = 32, 32, 1.1
        n_total = nt * nf
        times = np.arange(n_total) * dt
        signal = np.sin(2.0 * np.pi * times * 0.08)
        series = TimeSeries(signal, dt=dt)
        w = WDM.from_time_series(series, nt=nt)
        np.testing.assert_array_equal(w.dc_channel, np.real(w.coeffs[:, 0]))
        np.testing.assert_array_equal(w.nyquist_channel, np.imag(w.coeffs[:, 0]))


class TestFromFrequencySeries:
    def test_roundtrip_via_frequency_series(self) -> None:
        nt, nf, dt = 32, 32, 1.1
        n_total = nt * nf
        times = np.arange(n_total) * dt
        T = n_total * dt
        signal = np.exp(-((times - T / 2) ** 2) / (T / 4) ** 2) * np.cos(
            2.0 * np.pi * times * (0.1 + times * 0.153 / T) + 0.3
        )
        ts = TimeSeries(signal, dt=dt)
        fs = ts.to_frequency_series()

        w_from_time = WDM.from_time_series(ts, nt=nt)
        w_from_freq = WDM.from_frequency_series(fs, nt=nt)

        np.testing.assert_allclose(
            np.real(w_from_freq.coeffs),
            np.real(w_from_time.coeffs),
            atol=1e-10,
        )
