from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from wdm_transform import FrequencySeries, TimeSeries, WDM
from wdm_transform.backends import Backend, get_backend
from wdm_transform.plotting import plot_periodogram, plot_spectrogram


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
    assert FrequencySeries.__name__ == "FrequencySeries"

    monkeypatch.setenv("WDM_BACKEND", "numpy")
    assert get_backend().name == "numpy"


def test_wdm_roundtrip_diagnostics_plots(outdir: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")

    series, nt, _ = _example_series()
    coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)
    freq_series = series.to_frequency_series()
    display_freq_series = FrequencySeries(np.real(freq_series.data), df=freq_series.df)
    reconstructed = coeffs.to_time_series()

    residual = reconstructed.data - series.data
    test_outdir = outdir / "test_roundtrip"
    test_outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    series.plot(ax=ax, color="C0", label="input")
    ax.set_title("Original time series")
    assert ax.get_xlabel() == "Time [min]"
    assert ax.get_ylabel() == "Amplitude"
    fig.savefig(test_outdir / "1_original.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    display_freq_series.plot(
        ax=ax,
        magnitude=False,
        positive_only=False,
        color="C1",
        label="fft",
    )
    ax.set_title("Complex FFT coefficients")
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Value"
    fig.savefig(test_outdir / "2_frequency.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_periodogram(freq_series, ax=ax, color="C2")
    ax.set_title("Periodogram")
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Periodogram"
    fig.savefig(test_outdir / "3_periodogram.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_spectrogram(
        series,
        ax=ax,
        spec_kwargs={"nperseg": 64, "noverlap": 48},
        plot_kwargs={"cmap": "magma"},
    )
    ax.set_title("Spectrogram")
    assert ax.get_xlabel() == "Time [min]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    fig.savefig(test_outdir / "4_spectrogram.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    coeffs.plot(
        ax=ax,
        zscale="log",
        freq_scale="linear",
        freq_range=(coeffs.df, 1.0 / (2.0 * series.dt)),
        detailed_axes=True,
        label="coeffs",
        whiten_by=np.ones((coeffs.nf + 1, coeffs.nt)),
    )
    ax.set_title("WDM coefficient magnitude")
    assert "Time bins" in ax.get_xlabel()
    assert "Frequency bins" in ax.get_ylabel()
    assert any("coeffs" in text.get_text() for text in ax.texts)
    fig.savefig(test_outdir / "5_wdm.png", dpi=140)
    plt.close(fig)

    with pytest.warns(UserWarning, match="Falling back to default linear normalization"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        coeffs.plot(
            ax=ax,
            absolute=False,
            show_colorbar=False,
            show_gridinfo=False,
            whiten_by=np.full((coeffs.nf + 1, coeffs.nt), np.nan),
        )
    assert len(fig.axes) == 1
    fig.savefig(test_outdir / "6_wdm_fallback.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    reconstructed.plot(ax=ax, color="C3")
    ax.set_title("Reconstruction")
    assert ax.get_xlabel() == "Time [min]"
    assert ax.get_ylabel() == "Amplitude"
    fig.savefig(test_outdir / "7_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(np.asarray(residual), bins=40)
    ax.set_title("Residual histogram")
    ax.set_xlabel("reconstruction - original")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(test_outdir / "8_residual_hist.png", dpi=140)
    plt.close(fig)

    np.testing.assert_allclose(reconstructed.data, series.data, atol=1e-10, rtol=1e-10)
