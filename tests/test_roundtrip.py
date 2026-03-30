from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from wdm_transform import TimeSeries, WDM
from wdm_transform.backends import Backend, get_backend
from wdm_transform.freqseries import FrequencySeries
from wdm_transform.timeseries import TimeSeries as TimeSeriesModule
from wdm_transform.wdm import WDM as WDMModule


def _chirplet(times: np.ndarray, duration: float) -> np.ndarray:
    envelope = np.exp(-((times - duration / 2.0) ** 2) / (duration / 4.0) ** 2)
    carrier = np.cos(2.0 * np.pi * times * (0.1 + times * 0.153 / duration) + 0.3)
    return envelope * carrier


def _example_series(nt: int = 32, nf: int = 32, dt: float = 1.1) -> tuple[TimeSeries, int, int]:
    n_total = nt * nf
    times = np.arange(n_total) * dt
    duration = n_total * dt
    data = _chirplet(times, duration) + 0.2 * np.sin(2.0 * np.pi * times * 0.08)
    return TimeSeries(data, dt=dt), nt, nf


def _series_for_backend(series: TimeSeries, backend: Backend) -> TimeSeries:
    return TimeSeries(np.asarray(series.data), dt=series.dt, backend=backend)


def test_time_frequency_roundtrip() -> None:
    samples = np.sin(2.0 * np.pi * np.arange(128) * 0.125 * 0.15)
    series = TimeSeries(samples, dt=0.125)

    recovered = series.to_frequency_series().to_time_series(real=True)

    np.testing.assert_allclose(recovered.data, series.data, atol=1e-12, rtol=1e-12)


def test_wdm_time_roundtrip(backend_name: str, backend: Backend) -> None:
    series, nt, nf = _example_series()
    numpy_coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)

    backend_series = _series_for_backend(series, backend)
    coeffs = WDM.from_time_series(backend_series, nt=nt, a=1.0 / 3.0, backend=backend)
    recovered = coeffs.to_time_series()

    assert coeffs.coeffs.shape == (nt, nf + 1)
    np.testing.assert_allclose(
        np.asarray(coeffs.coeffs),
        np.asarray(numpy_coeffs.coeffs),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(recovered.data),
        np.asarray(series.data),
        atol=1e-10,
        rtol=1e-10,
    )

    if backend_name == "jax":
        jax = pytest.importorskip("jax")
        assert isinstance(coeffs.coeffs, jax.Array)


def test_wdm_frequency_reconstruction_matches_fft(backend: Backend) -> None:
    series, nt, _ = _example_series()
    expected = series.to_frequency_series()

    backend_series = _series_for_backend(series, backend)
    coeffs = WDM.from_time_series(backend_series, nt=nt, a=1.0 / 3.0, backend=backend)
    reconstructed = coeffs.to_frequency_series()

    np.testing.assert_allclose(
        np.asarray(reconstructed.data),
        np.asarray(expected.data),
        atol=1e-10,
        rtol=1e-10,
    )


def test_top_level_module_exports_and_env_backend(monkeypatch) -> None:
    assert TimeSeriesModule is TimeSeries
    assert WDMModule is WDM
    assert FrequencySeries.__name__ == "FrequencySeries"

    monkeypatch.setenv("WDM_BACKEND", "numpy")
    assert get_backend().name == "numpy"


def test_wdm_roundtrip_diagnostics_plots(outdir: Path) -> None:
    pytest = __import__("pytest")
    plt = pytest.importorskip("matplotlib.pyplot")

    series, nt, _ = _example_series()
    coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)
    reconstructed = coeffs.to_time_series()

    residual = reconstructed.data - series.data
    test_outdir = outdir / "test_roundtrip"
    test_outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(series.times, series.data)
    ax.set_title("Original time series")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    fig.tight_layout()
    fig.savefig(test_outdir / "1_original.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(np.abs(coeffs.coeffs), aspect="auto", origin="lower")
    ax.set_title("WDM coefficient magnitude")
    ax.set_xlabel("channel")
    ax.set_ylabel("time bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(test_outdir / "2_wdm.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.specgram(series.data, NFFT=64, Fs=1.0 / series.dt, noverlap=48, cmap="magma")
    ax.set_title("Spectrogram")
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(test_outdir / "3_spectrogram.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(reconstructed.times, reconstructed.data)
    ax.set_title("Reconstruction")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    fig.tight_layout()
    fig.savefig(test_outdir / "4_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(np.asarray(residual), bins=40)
    ax.set_title("Residual histogram")
    ax.set_xlabel("reconstruction - original")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(test_outdir / "5_residual_hist.png", dpi=140)
    plt.close(fig)

    np.testing.assert_allclose(reconstructed.data, series.data, atol=1e-10, rtol=1e-10)
