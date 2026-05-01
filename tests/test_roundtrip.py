from __future__ import annotations

import numpy as np
from pathlib import Path
from importlib.util import find_spec
import pytest

from wdm_transform import FrequencySeries, TimeSeries, WDM
from wdm_transform.backends import Backend, get_backend
from wdm_transform.plotting import plot_periodogram, plot_spectrogram
import matplotlib.pyplot as plt


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

    assert coeffs.coeffs.shape == (1, nt, nf + 1)
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


def test_batched_jax_time_roundtrip_matches_stacked_single_series() -> None:
    pytest.importorskip("jax")

    series, nt, nf = _example_series()
    samples = np.asarray(series.data[0])
    batched_samples = np.stack([samples, 0.5 * samples, -samples], axis=0)
    batched_series = TimeSeries(batched_samples, dt=series.dt, backend="jax")

    coeffs = WDM.from_time_series(batched_series, nt=nt, backend="jax")
    recovered = coeffs.to_time_series()

    expected_coeffs = np.stack([
        np.asarray(WDM.from_time_series(TimeSeries(row, dt=series.dt), nt=nt).coeffs[0])
        for row in batched_samples
    ], axis=0)

    assert coeffs.batch_size == 3
    assert coeffs.coeffs.shape == (3, nt, nf + 1)
    np.testing.assert_allclose(np.asarray(coeffs.coeffs), expected_coeffs, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(recovered.data), batched_samples, atol=1e-10, rtol=1e-10)


def test_batched_jax_frequency_roundtrip_matches_stacked_single_series() -> None:
    pytest.importorskip("jax")

    series, nt, _ = _example_series()
    base_spectrum = np.asarray(series.to_frequency_series().data[0])
    spectra = np.stack([
        base_spectrum,
        0.5 * base_spectrum,
        -base_spectrum,
    ], axis=0)
    batched_series = FrequencySeries(spectra, df=series.df, backend="jax")

    coeffs = WDM.from_frequency_series(batched_series, nt=nt, backend="jax")
    reconstructed = coeffs.to_frequency_series()

    expected_coeffs = np.stack([
        np.asarray(WDM.from_frequency_series(FrequencySeries(row, df=series.df), nt=nt).coeffs[0])
        for row in spectra
    ], axis=0)

    np.testing.assert_allclose(np.asarray(coeffs.coeffs), expected_coeffs, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(reconstructed.data), spectra, atol=1e-10, rtol=1e-10)


def test_batched_jax_batch_size_one_roundtrip() -> None:
    pytest.importorskip("jax")

    series, nt, nf = _example_series()
    batched = TimeSeries(np.asarray(series.data[0])[None, :], dt=series.dt, backend="jax")
    coeffs = batched.to_wdm(nt=nt)
    recovered = coeffs.to_time_series()

    assert batched.batch_size == 1
    assert coeffs.batch_size == 1
    assert coeffs.shape == (1, nt, nf + 1)
    np.testing.assert_allclose(np.asarray(recovered.data), np.asarray(batched.data), atol=1e-10, rtol=1e-10)


def test_batched_numpy_time_roundtrip_matches_stacked_single_series() -> None:
    series, nt, nf = _example_series()
    samples = np.asarray(series.data[0])
    batched_samples = np.stack([samples, 0.5 * samples, -samples], axis=0)
    batched_series = TimeSeries(batched_samples, dt=series.dt, backend="numpy")

    coeffs = WDM.from_time_series(batched_series, nt=nt, backend="numpy")
    recovered = coeffs.to_time_series()

    expected_coeffs = np.stack([
        np.asarray(WDM.from_time_series(TimeSeries(row, dt=series.dt), nt=nt, backend="numpy").coeffs[0])
        for row in batched_samples
    ], axis=0)

    assert coeffs.batch_size == 3
    assert coeffs.coeffs.shape == (3, nt, nf + 1)
    np.testing.assert_allclose(np.asarray(coeffs.coeffs), expected_coeffs, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(recovered.data), batched_samples, atol=1e-10, rtol=1e-10)


def test_batched_numpy_frequency_roundtrip_matches_stacked_single_series() -> None:
    series, nt, _ = _example_series()
    base_spectrum = np.asarray(series.to_frequency_series().data[0])
    spectra = np.stack([base_spectrum, 0.5 * base_spectrum, -base_spectrum], axis=0)
    batched_series = FrequencySeries(spectra, df=series.df, backend="numpy")

    coeffs = WDM.from_frequency_series(batched_series, nt=nt, backend="numpy")
    reconstructed = coeffs.to_frequency_series()

    expected_coeffs = np.stack([
        np.asarray(
            WDM.from_frequency_series(FrequencySeries(row, df=series.df), nt=nt, backend="numpy").coeffs[0]
        )
        for row in spectra
    ], axis=0)

    np.testing.assert_allclose(np.asarray(coeffs.coeffs), expected_coeffs, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(reconstructed.data), spectra, atol=1e-10, rtol=1e-10)


def test_batched_numpy_batch_size_one_roundtrip() -> None:
    series, nt, nf = _example_series()
    batched = TimeSeries(np.asarray(series.data[0])[None, :], dt=series.dt, backend="numpy")
    coeffs = batched.to_wdm(nt=nt)
    recovered = coeffs.to_time_series()

    assert batched.batch_size == 1
    assert coeffs.batch_size == 1
    assert coeffs.shape == (1, nt, nf + 1)
    np.testing.assert_allclose(np.asarray(recovered.data), np.asarray(batched.data), atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(find_spec("cupy") is None, reason="cupy is not installed")
def test_batched_cupy_matches_stacked_single_series() -> None:
    series, nt, nf = _example_series()
    samples = np.asarray(series.data[0])
    batched_samples = np.stack([samples, 0.5 * samples, -samples], axis=0)
    batched_series = TimeSeries(batched_samples, dt=series.dt, backend="cupy")

    coeffs = WDM.from_time_series(batched_series, nt=nt, backend="cupy")
    recovered = coeffs.to_time_series()

    expected_coeffs = np.stack([
        np.asarray(WDM.from_time_series(TimeSeries(row, dt=series.dt), nt=nt, backend="cupy").coeffs[0])
        for row in batched_samples
    ], axis=0)

    assert coeffs.coeffs.shape == (3, nt, nf + 1)
    np.testing.assert_allclose(np.asarray(coeffs.coeffs), expected_coeffs, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(recovered.data), batched_samples, atol=1e-10, rtol=1e-10)


def test_batched_numpy_matches_jax_coefficients() -> None:
    pytest.importorskip("jax")

    series, nt, _ = _example_series()
    samples = np.asarray(series.data[0])
    batched_samples = np.stack([samples, 0.5 * samples, -samples], axis=0)

    numpy_coeffs = WDM.from_time_series(
        TimeSeries(batched_samples, dt=series.dt, backend="numpy"),
        nt=nt,
        backend="numpy",
    )
    jax_coeffs = WDM.from_time_series(
        TimeSeries(batched_samples, dt=series.dt, backend="jax"),
        nt=nt,
        backend="jax",
    )

    np.testing.assert_allclose(
        np.asarray(numpy_coeffs.coeffs),
        np.asarray(jax_coeffs.coeffs),
        atol=1e-10,
        rtol=1e-10,
    )


def test_top_level_module_exports_and_env_backend(monkeypatch) -> None:
    assert FrequencySeries.__name__ == "FrequencySeries"

    monkeypatch.setenv("WDM_BACKEND", "numpy")
    assert get_backend().name == "numpy"


def test_batched_plot_methods_create_one_axis_per_batch_element() -> None:
    pytest.importorskip("matplotlib")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")

    time_series = TimeSeries(np.zeros((2, 16)), dt=0.25)
    freq_series = FrequencySeries(np.zeros((2, 16), dtype=complex), df=0.25)
    wdm = WDM(np.zeros((2, 4, 5)), dt=0.25, backend="jax")

    fig, axes = time_series.plot()
    assert len(np.ravel(axes)) == 2
    assert all(ax.get_ylabel() == "Amplitude" for ax in np.ravel(axes))
    plt.close(fig)

    fig, axes = freq_series.plot()
    assert len(np.ravel(axes)) == 2
    assert all(ax.get_ylabel() == "Magnitude" for ax in np.ravel(axes))
    plt.close(fig)

    fig, axes = wdm.plot()
    assert len(np.ravel(axes)) == 2
    assert all(ax.get_ylabel() == "Frequency [Hz]" for ax in np.ravel(axes))
    plt.close(fig)


def test_singleton_plot_methods_return_one_element_axes_arrays() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")

    time_series = TimeSeries(np.zeros(16), dt=0.25)
    freq_series = FrequencySeries(np.zeros(16, dtype=complex), df=0.25)
    wdm = WDM(np.zeros((4, 5)), dt=0.25, backend="numpy")

    fig, axes = time_series.plot()
    assert np.asarray(axes).shape == (1,)
    plt.close(fig)

    fig, axes = freq_series.plot()
    assert np.asarray(axes).shape == (1,)
    plt.close(fig)

    fig, axes = wdm.plot()
    assert np.asarray(axes).shape == (1,)
    plt.close(fig)


def test_wdm_roundtrip_diagnostics_plots(outdir: Path) -> None:

    series, nt, _ = _example_series()
    coeffs = WDM.from_time_series(series, nt=nt, a=1.0 / 3.0)
    freq_series = series.to_frequency_series()
    display_freq_series = FrequencySeries(np.real(freq_series.data), df=freq_series.df)
    reconstructed = coeffs.to_time_series()

    residual = reconstructed.data - series.data

    fig, ax = plt.subplots(figsize=(8, 3.5))
    series.plot(ax=ax, color="C0", label="input")
    ax.set_title("Original time series")
    assert ax.get_xlabel() == "Time [min]"
    assert ax.get_ylabel() == "Amplitude"
    fig.savefig(outdir / "1_original.png", dpi=140)
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
    fig.savefig(outdir / "2_frequency.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_periodogram(freq_series, ax=ax, color="C2")
    ax.set_title("Periodogram")
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Periodogram"
    fig.savefig(outdir / "3_periodogram.png", dpi=140)
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
    fig.savefig(outdir / "4_spectrogram.png", dpi=140)
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
    fig.savefig(outdir / "5_wdm.png", dpi=140)
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
    fig.savefig(outdir / "6_wdm_fallback.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    reconstructed.plot(ax=ax, color="C3")
    ax.set_title("Reconstruction")
    assert ax.get_xlabel() == "Time [min]"
    assert ax.get_ylabel() == "Amplitude"
    fig.savefig(outdir / "7_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(np.asarray(residual), bins=40)
    ax.set_title("Residual histogram")
    ax.set_xlabel("reconstruction - original")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(outdir / "8_residual_hist.png", dpi=140)
    plt.close(fig)

    np.testing.assert_allclose(reconstructed.data, series.data, atol=1e-10, rtol=1e-10)
