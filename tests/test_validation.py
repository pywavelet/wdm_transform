"""Tests for input validation and error paths."""

from __future__ import annotations

import numpy as np
import pytest

from wdm_transform import FrequencySeries, TimeSeries, WDM


class TestWDMValidation:
    def test_odd_nt_raises(self) -> None:
        # shape (3, 5) → nt=3 (odd), nf=4 (5-1)
        data = np.zeros((3, 5), dtype=float)
        with pytest.raises(ValueError, match="nt and nf must both be even"):
            WDM(coeffs=data, dt=1.0)

    def test_odd_nf_raises(self) -> None:
        # shape (4, 4) → nt=4, nf=3 (4-1, odd)
        data = np.zeros((4, 4), dtype=float)
        with pytest.raises(ValueError, match="nt and nf must both be even"):
            WDM(coeffs=data, dt=1.0)

    def test_bad_window_parameter_raises(self) -> None:
        # shape (4, 5) → nt=4, nf=4
        data = np.zeros((4, 5), dtype=float)
        with pytest.raises(ValueError, match="a must be in"):
            WDM(coeffs=data, dt=1.0, a=0.0)
        with pytest.raises(ValueError, match="a must be in"):
            WDM(coeffs=data, dt=1.0, a=0.5)

    def test_negative_dt_raises(self) -> None:
        data = np.zeros((4, 5), dtype=float)
        with pytest.raises(ValueError, match="dt must be positive"):
            WDM(coeffs=data, dt=-1.0)

    def test_1d_coeffs_raises(self) -> None:
        with pytest.raises(ValueError, match="two- or three-dimensional"):
            WDM(coeffs=np.zeros(16), dt=1.0)

    def test_4d_coeffs_raises(self) -> None:
        with pytest.raises(ValueError, match="two- or three-dimensional"):
            WDM(coeffs=np.zeros((2, 3, 4, 5)), dt=1.0)

    def test_from_time_series_indivisible_length(self) -> None:
        series = TimeSeries(np.zeros(100), dt=1.0)
        with pytest.raises(ValueError, match="not divisible"):
            WDM.from_time_series(series, nt=7)

    def test_forward_wrong_length_raises(self) -> None:
        series = TimeSeries(np.zeros(128), dt=1.0)
        with pytest.raises(ValueError, match="not divisible"):
            WDM.from_time_series(series, nt=6)


class TestWDMShape:
    def test_shape_is_nt_nf_plus_one(self) -> None:
        nt, nf, dt = 32, 32, 1.1
        n_total = nt * nf
        signal = np.sin(2.0 * np.pi * np.arange(n_total) * dt * 0.08)
        series = TimeSeries(signal, dt=dt)
        w = WDM.from_time_series(series, nt=nt)
        assert w.coeffs.shape == (1, nt, nf + 1)
        assert w.nt == nt
        assert w.nf == nf
        assert w.batch_size == 1
        assert w.shape == (1, nt, nf + 1)

    def test_coeffs_are_real(self) -> None:
        nt, nf, dt = 32, 32, 1.1
        n_total = nt * nf
        signal = np.sin(2.0 * np.pi * np.arange(n_total) * dt * 0.08)
        series = TimeSeries(signal, dt=dt)
        w = WDM.from_time_series(series, nt=nt)
        assert w.coeffs.dtype == np.float64


class TestWDMRepr:
    def test_repr_is_compact(self) -> None:
        data = np.zeros((4, 5), dtype=float)  # nt=4, nf=4
        r = repr(WDM(coeffs=data, dt=0.5))
        assert "batch_size=1" in r
        assert "nt=4" in r
        assert "nf=4" in r
        assert "n=16" in r
        assert "dt=0.5" in r
        assert "df=0.125" in r
        assert "fs=2.0" in r
        assert "nyquist=1.0" in r
        assert "delta_t=2.0" in r
        assert "delta_f=0.25" in r
        assert "duration=8.0" in r
        assert "array" not in r.lower()


class TestSeriesRepr:
    def test_time_series_repr_is_compact(self) -> None:
        r = repr(TimeSeries(np.zeros(8), dt=0.25))
        assert "batch_size=1" in r
        assert "n=8" in r
        assert "dt=0.25" in r
        assert "df=0.5" in r
        assert "fs=4.0" in r
        assert "nyquist=2.0" in r
        assert "duration=2.0" in r
        assert "array" not in r.lower()

    def test_frequency_series_repr_is_compact(self) -> None:
        r = repr(FrequencySeries(np.zeros(8, dtype=complex), df=0.5))
        assert "batch_size=1" in r
        assert "n=8" in r
        assert "df=0.5" in r
        assert "dt=0.25" in r
        assert "fs=4.0" in r
        assert "nyquist=2.0" in r
        assert "duration=2.0" in r
        assert "array" not in r.lower()

    def test_high_rank_series_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            TimeSeries(np.zeros((2, 3, 4)), dt=0.25)
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            FrequencySeries(np.zeros((2, 3, 4), dtype=complex), df=0.5)


class TestEdgeChannelAccessors:
    def test_dc_and_nyquist_channels(self) -> None:
        nt, nf, dt = 32, 32, 1.1
        n_total = nt * nf
        signal = np.sin(2.0 * np.pi * np.arange(n_total) * dt * 0.08)
        series = TimeSeries(signal, dt=dt)
        w = WDM.from_time_series(series, nt=nt)
        np.testing.assert_array_equal(w.dc_channel, w.coeffs[..., 0])
        np.testing.assert_array_equal(w.nyquist_channel, w.coeffs[..., nf])

    def test_batched_edge_channel_accessors(self) -> None:
        coeffs = np.zeros((3, 4, 5), dtype=float)
        w = WDM(coeffs=coeffs, dt=1.0)
        np.testing.assert_array_equal(w.dc_channel, coeffs[..., 0])
        np.testing.assert_array_equal(w.nyquist_channel, coeffs[..., -1])


class TestBatchedBackendValidation:
    def test_batched_numpy_backend_supports_forward_and_inverse(self) -> None:
        batched = np.zeros((3, 128), dtype=float)
        coeffs = WDM.from_time_series(TimeSeries(batched, dt=1.0), nt=32, backend="numpy")
        recovered = coeffs.to_time_series()

        assert coeffs.coeffs.shape == (3, 32, 5)
        np.testing.assert_allclose(recovered.data, batched, atol=1e-12, rtol=1e-12)

        freq_coeffs = WDM.from_frequency_series(
            FrequencySeries(np.zeros((3, 128), dtype=complex), df=1.0 / 128.0),
            nt=32,
            backend="numpy",
        )
        assert freq_coeffs.coeffs.shape == (3, 32, 5)


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
            w_from_freq.coeffs,
            w_from_time.coeffs,
            atol=1e-10,
        )
