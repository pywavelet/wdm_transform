from __future__ import annotations

import numpy as np
import pytest
import time
from importlib.util import find_spec

from wdm_transform import TimeSeries, WDM
from wdm_transform.backends import get_backend
from wdm_transform.transforms import (
    fourier_span_from_wdm_span,
    forward_wdm_subband,
    inverse_wdm_subband,
    wdm_span_from_fourier_span,
)
from wdm_transform.windows import cnm, phi_unit


NT = 32
NF = 32
N_TOTAL = NT * NF
NFOURIER = N_TOTAL // 2 + 1
DT = 0.125
DF = 1.0 / (N_TOTAL * DT)
NUMPY_BACKEND = get_backend("numpy")


def _generate_sinusoid(
    *,
    k0: int,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    times = np.arange(N_TOTAL) * DT
    frequency = k0 * DF
    return amplitude * np.cos(2.0 * np.pi * frequency * times + phase)


def _generate_wd_binary_signal(
    *,
    f0: float,
    fdot: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    times = np.arange(N_TOTAL) * DT
    phase = 2.0 * np.pi * (f0 * times + 0.5 * fdot * times**2)
    return amplitude * np.cos(phase)


def _filter_h_analytic(l0: float, n: np.ndarray, nt: int) -> np.ndarray:
    m0 = int(2.0 * abs(l0) / nt)
    phi_minus = np.asarray(
        phi_unit(NUMPY_BACKEND, (l0 - m0 * nt / 2.0) * 2.0 / nt, 1.0 / 3.0, 1.0)
    ) / np.sqrt(nt / 2.0)
    phi_plus = np.asarray(
        phi_unit(NUMPY_BACKEND, (l0 + m0 * nt / 2.0) * 2.0 / nt, 1.0 / 3.0, 1.0)
    ) / np.sqrt(nt / 2.0)
    cnm_vals = np.asarray(cnm(NUMPY_BACKEND, n, m0))
    return (
        np.conjugate(cnm_vals) * phi_minus
        + cnm_vals * phi_plus
    ) / np.sqrt(2.0)


def _sinusoid_wdm_analytic(
    amplitude: float,
    frequency: float,
    phase: float,
    *,
    dt: float,
    nt: int,
    nf: int,
) -> np.ndarray:
    ntot = nt * nf
    l0 = abs(frequency * ntot * dt)
    coeffs = np.zeros((nt, nf + 1), dtype=np.complex128)
    narr = np.arange(nt)
    m0 = int(2.0 * l0 / nt)
    coeffs[:, m0] = amplitude * ntot / (2j) * (
        np.exp(1j * phase + 2j * np.pi * narr * l0 / nt)
        * _filter_h_analytic(l0, narr, nt)
        - np.exp(-1j * phase - 2j * np.pi * narr * l0 / nt)
        * _filter_h_analytic(-l0, narr, nt)
    )
    return coeffs


def _subband_case_data(kmin: int, lendata: int) -> np.ndarray:
    values = np.linspace(1.0, float(lendata), lendata) + 1j * np.linspace(
        0.25,
        0.25 * lendata,
        lendata,
    )
    data = values.astype(np.complex128)
    if kmin == 0:
        data[0] = complex(data[0].real, 0.0)
    if kmin + lendata - 1 == NFOURIER - 1:
        data[-1] = complex(data[-1].real, 0.0)
    return data


def _one_sided_full_spectrum(kmin: int, data: np.ndarray) -> np.ndarray:
    full = np.zeros(NFOURIER, dtype=np.complex128)
    full[kmin:kmin + len(data)] = data
    return full


def _snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise_norm = np.linalg.norm(reference - estimate)
    if noise_norm == 0.0:
        return float("inf")
    return float(20.0 * np.log10(np.linalg.norm(reference) / noise_norm))


def _format_snr(value: float) -> str:
    if np.isinf(value):
        return "SNR: inf"
    return f"SNR: {value:.2f}"


def _annotate_snr(axis, value: float) -> None:
    axis.text(
        0.03,
        0.97,
        _format_snr(value),
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _annotate_metrics(
    axis,
    *,
    snr: float,
    runtime_ms: float | None = None,
) -> None:
    lines = [_format_snr(snr)]
    if runtime_ms is not None:
        lines.append(f"Runtime: {runtime_ms:.2f} ms")
    axis.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _mean_runtime_ms(fn, repeats: int = 5) -> float:
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(durations))


def test_subband_grid_validation_errors() -> None:
    with pytest.raises(ValueError, match="Inconsistent full-grid sizes"):
        wdm_span_from_fourier_span(
            nfreqs_fourier=NFOURIER,
            nfreqs_wdm=NF,
            ntimes_wdm=NT + 2,
            kmin=0,
            lendata=1,
        )

    with pytest.raises(ValueError, match="lendata must be positive"):
        wdm_span_from_fourier_span(
            nfreqs_fourier=NFOURIER,
            nfreqs_wdm=NF,
            ntimes_wdm=NT,
            kmin=0,
            lendata=0,
        )

    with pytest.raises(ValueError, match="nf_sub_wdm must be positive"):
        fourier_span_from_wdm_span(
            nfreqs_fourier=NFOURIER,
            nfreqs_wdm=NF,
            ntimes_wdm=NT,
            mmin=0,
            nf_sub_wdm=0,
        )

    with pytest.raises(ValueError, match="df must be positive"):
        forward_wdm_subband(
            np.ones(4, dtype=np.complex128),
            df=0.0,
            nfreqs_fourier=NFOURIER,
            kmin=4,
            nfreqs_wdm=NF,
            ntimes_wdm=NT,
        )


@pytest.mark.parametrize(
    ("kmin", "lendata", "expected_wdm", "expected_fourier"),
    [
        (0, 1, (0, 2), (0, 32)),
        (160, 16, (10, 2), (144, 48)),
        (512, 1, (32, 1), (496, 17)),
        (23, 29, (1, 4), (0, 80)),
    ],
)
def test_span_mappings(
    kmin: int,
    lendata: int,
    expected_wdm: tuple[int, int],
    expected_fourier: tuple[int, int],
) -> None:
    mapped_wdm = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=kmin,
        lendata=lendata,
    )
    assert mapped_wdm == expected_wdm

    mapped_fourier = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=expected_wdm[0],
        nf_sub_wdm=expected_wdm[1],
    )
    assert mapped_fourier == expected_fourier
    assert mapped_fourier[0] <= kmin
    assert mapped_fourier[0] + mapped_fourier[1] >= kmin + lendata


@pytest.mark.parametrize(
    ("kmin", "lendata"),
    [
        (0, 1),
        (512, 1),
        (160, 16),
        (23, 29),
    ],
)
def test_forward_subband_matches_full_wdm_slice(kmin: int, lendata: int) -> None:
    subband = _subband_case_data(kmin, lendata)
    one_sided = _one_sided_full_spectrum(kmin, subband)
    signal = np.fft.irfft(one_sided, n=N_TOTAL)

    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    coeffs, mmin = forward_wdm_subband(
        subband,
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )

    expected_mmin, expected_nf_sub = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=kmin,
        lendata=lendata,
    )
    assert mmin == expected_mmin
    assert coeffs.shape == (NT, expected_nf_sub)
    np.testing.assert_allclose(
        coeffs,
        np.asarray(full_wdm.coeffs)[:, mmin:mmin + expected_nf_sub],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.skipif(find_spec("jax") is None, reason="jax is not installed")
def test_forward_subband_matches_full_wdm_slice_jax() -> None:
    jax = pytest.importorskip("jax.numpy")
    kmin = 23
    lendata = 29
    subband = _subband_case_data(kmin, lendata)
    one_sided = _one_sided_full_spectrum(kmin, subband)
    signal = np.fft.irfft(one_sided, n=N_TOTAL)

    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    coeffs, mmin = forward_wdm_subband(
        jax.asarray(subband),
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        backend="jax",
    )

    expected_mmin, expected_nf_sub = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=kmin,
        lendata=lendata,
    )
    assert mmin == expected_mmin
    assert coeffs.shape == (NT, expected_nf_sub)
    np.testing.assert_allclose(
        np.asarray(coeffs),
        np.asarray(full_wdm.coeffs)[:, mmin:mmin + expected_nf_sub],
        atol=1e-10,
        rtol=1e-10,
    )


def test_forward_subband_matches_full_wdm_slice_for_sinusoid() -> None:
    k0 = 37
    signal = _generate_sinusoid(k0=k0, amplitude=1.7, phase=0.31)
    one_sided = np.fft.rfft(signal)

    coeffs, mmin = forward_wdm_subband(
        one_sided[k0:k0 + 1],
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=k0,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)

    expected_mmin, expected_nf_sub = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=k0,
        lendata=1,
    )
    assert mmin == expected_mmin
    assert coeffs.shape == (NT, expected_nf_sub)
    np.testing.assert_allclose(
        coeffs,
        np.asarray(full_wdm.coeffs)[:, mmin:mmin + expected_nf_sub],
        atol=1e-10,
        rtol=1e-10,
    )


def test_full_wdm_matches_analytic_sinusoid_formula_after_normalization() -> None:
    k0 = 37
    amplitude = 1.7
    phase = 0.31
    frequency = k0 * DF

    signal = amplitude * np.sin(
        2.0 * np.pi * frequency * np.arange(N_TOTAL) * DT + phase
    )
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    analytic = _sinusoid_wdm_analytic(
        amplitude,
        frequency,
        phase,
        dt=DT,
        nt=NT,
        nf=NF,
    )

    # This analytic convention differs by a factor nf from the package normalization.
    np.testing.assert_allclose(
        np.asarray(full_wdm.coeffs),
        np.real(analytic) / NF,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("mmin", "nf_sub_wdm"),
    [
        (0, 2),
        (10, 2),
        (30, 3),
        (32, 1),
    ],
)
def test_inverse_subband_matches_full_frequency_slice(
    mmin: int,
    nf_sub_wdm: int,
) -> None:
    times = np.arange(N_TOTAL) * DT
    signal = (
        np.sin(2.0 * np.pi * times * 0.15)
        + 0.2 * np.cos(2.0 * np.pi * times * 0.04)
        + 0.05 * np.sin(2.0 * np.pi * times * 0.31)
    )
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    coeffs = np.zeros_like(np.asarray(full_wdm.coeffs))
    coeffs[:, mmin:mmin + nf_sub_wdm] = np.asarray(full_wdm.coeffs)[:, mmin:mmin + nf_sub_wdm]

    reconstructed, kmin = inverse_wdm_subband(
        coeffs[:, mmin:mmin + nf_sub_wdm],
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )
    expected = full_wdm.__class__(
        coeffs=coeffs,
        dt=DT,
        a=full_wdm.a,
        d=full_wdm.d,
    ).to_frequency_series()

    expected_kmin, lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=mmin,
        nf_sub_wdm=nf_sub_wdm,
    )
    assert kmin == expected_kmin
    assert reconstructed.shape == (lendata,)
    np.testing.assert_allclose(
        reconstructed,
        np.asarray(expected.data)[kmin:kmin + lendata],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("kmin", "lendata"),
    [
        (0, 1),
        (512, 1),
        (160, 16),
        (23, 29),
    ],
)
def test_forward_then_inverse_recovers_zero_padded_fourier_span(
    kmin: int,
    lendata: int,
) -> None:
    subband = _subband_case_data(kmin, lendata)
    full_spectrum = _one_sided_full_spectrum(kmin, subband)

    coeffs, mmin = forward_wdm_subband(
        subband,
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )
    recovered, recovered_kmin = inverse_wdm_subband(
        coeffs,
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )

    expected_kmin, expected_lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=mmin,
        nf_sub_wdm=coeffs.shape[1],
    )
    assert recovered_kmin == expected_kmin
    np.testing.assert_allclose(
        recovered,
        full_spectrum[expected_kmin:expected_kmin + expected_lendata],
        atol=1e-10,
        rtol=1e-10,
    )


def test_chirping_binary_subband_roundtrip() -> None:
    signal = _generate_wd_binary_signal(
        f0=0.15,
        fdot=5.0e-4,
        amplitude=1.0,
    )
    one_sided = np.fft.rfft(signal)
    k_peak = int(np.argmax(np.abs(one_sided)))
    kmin = max(0, k_peak - 3)
    lendata = 8
    subband = one_sided[kmin:kmin + lendata]
    full_spectrum = _one_sided_full_spectrum(kmin, subband)

    coeffs, mmin = forward_wdm_subband(
        subband,
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )
    recovered, recovered_kmin = inverse_wdm_subband(
        coeffs,
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )

    expected_kmin, expected_lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=mmin,
        nf_sub_wdm=coeffs.shape[1],
    )
    assert recovered_kmin == expected_kmin
    np.testing.assert_allclose(
        recovered,
        full_spectrum[expected_kmin:expected_kmin + expected_lendata],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.skipif(find_spec("jax") is None, reason="jax is not installed")
def test_chirping_binary_subband_roundtrip_jax() -> None:
    jax = pytest.importorskip("jax.numpy")
    signal = _generate_wd_binary_signal(
        f0=0.15,
        fdot=5.0e-4,
        amplitude=1.0,
    )
    one_sided = np.fft.rfft(signal)
    k_peak = int(np.argmax(np.abs(one_sided)))
    kmin = max(0, k_peak - 3)
    lendata = 8
    subband = one_sided[kmin:kmin + lendata]
    full_spectrum = _one_sided_full_spectrum(kmin, subband)

    coeffs, mmin = forward_wdm_subband(
        jax.asarray(subband),
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        backend="jax",
    )
    recovered, recovered_kmin = inverse_wdm_subband(
        coeffs,
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        backend="jax",
    )

    expected_kmin, expected_lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=mmin,
        nf_sub_wdm=coeffs.shape[1],
    )
    assert recovered_kmin == expected_kmin
    np.testing.assert_allclose(
        np.asarray(recovered),
        full_spectrum[expected_kmin:expected_kmin + expected_lendata],
        atol=1e-10,
        rtol=1e-10,
    )


def test_chirping_binary_subband_diagnostics_plot(outdir) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    test_outdir = outdir / "test_subband"
    test_outdir.mkdir(parents=True, exist_ok=True)

    signal = _generate_wd_binary_signal(
        f0=0.15,
        fdot=5.0e-4,
        amplitude=1.0,
    )
    full_runtime_ms = _mean_runtime_ms(
        lambda: WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    )
    one_sided = np.fft.rfft(signal)
    k_peak = int(np.argmax(np.abs(one_sided)))
    kmin = max(0, k_peak - 3)
    lendata = 8
    subband = one_sided[kmin:kmin + lendata]
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    full_spectrum = _one_sided_full_spectrum(kmin, subband)
    subband_runtime_ms = _mean_runtime_ms(
        lambda: forward_wdm_subband(
            subband,
            df=DF,
            nfreqs_fourier=NFOURIER,
            kmin=kmin,
            nfreqs_wdm=NF,
            ntimes_wdm=NT,
        )
    )

    coeffs, mmin = forward_wdm_subband(
        subband,
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )
    recovered, recovered_kmin = inverse_wdm_subband(
        coeffs,
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
    )

    expected_kmin, expected_lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=mmin,
        nf_sub_wdm=coeffs.shape[1],
    )
    reference_fourier = full_spectrum[
        expected_kmin:expected_kmin + expected_lendata
    ]
    np.testing.assert_allclose(
        recovered,
        reference_fourier,
        atol=1e-10,
        rtol=1e-10,
    )

    zero_padded_signal = np.fft.irfft(full_spectrum, n=N_TOTAL)
    expected_wdm = WDM.from_time_series(TimeSeries(zero_padded_signal, dt=DT), nt=NT)
    reference_coeffs = np.asarray(expected_wdm.coeffs)[:, mmin:mmin + coeffs.shape[1]]

    fourier_snr = _snr(reference_fourier, recovered)
    wdm_snr = _snr(reference_coeffs, coeffs)

    fourier_freqs = (recovered_kmin + np.arange(recovered.shape[0])) * DF
    full_freqs = np.fft.rfftfreq(N_TOTAL, d=DT)
    time_bin = NF * DT
    freq_bin = 1.0 / (2.0 * NF * DT)
    recovered_full = _one_sided_full_spectrum(recovered_kmin, recovered)
    full_band_coeffs = np.full((NT, NF + 1), np.nan, dtype=float)
    full_band_coeffs[:, mmin:mmin + coeffs.shape[1]] = np.abs(coeffs)
    full_wdm_abs = np.abs(np.asarray(full_wdm.coeffs))
    wdm_vmin = 0.0
    wdm_vmax = float(np.nanmax(full_wdm_abs))
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

    axes[0].plot(full_freqs, np.abs(one_sided), color="0.75", linewidth=1.5, label="full rFFT")
    axes[0].plot(
        full_freqs[kmin:kmin + lendata],
        np.abs(subband),
        color="C1",
        linewidth=2.0,
        label="input sub-band",
    )
    axes[0].axvspan(
        full_freqs[kmin],
        full_freqs[kmin + lendata - 1],
        color="C1",
        alpha=0.12,
    )
    axes[0].set_title("Extracted Fourier sub-band")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend(loc="upper right")
    _annotate_metrics(axes[0], snr=fourier_snr)

    full_image = axes[1].imshow(
        full_wdm_abs.T,
        aspect="auto",
        origin="lower",
        extent=[
            0.0,
            NT * time_bin,
            -0.5 * freq_bin,
            (NF + 0.5) * freq_bin,
        ],
        interpolation="nearest",
        cmap="magma",
        vmin=wdm_vmin,
        vmax=wdm_vmax,
    )
    axes[1].axhspan(
        mmin * freq_bin - 0.5 * freq_bin,
        (mmin + coeffs.shape[1]) * freq_bin - 0.5 * freq_bin,
        color="C1",
        alpha=0.12,
    )
    axes[1].set_title("Full WDM grid with overlapping global channel range highlighted")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    _annotate_metrics(axes[1], snr=wdm_snr, runtime_ms=full_runtime_ms)
    fig.colorbar(full_image, ax=axes[1], label="|WDM coefficient|")

    compact_image = axes[2].imshow(
        full_band_coeffs.T,
        aspect="auto",
        origin="lower",
        extent=[
            0.0,
            NT * time_bin,
            -0.5 * freq_bin,
            (NF + 0.5) * freq_bin,
        ],
        interpolation="nearest",
        cmap="viridis",
        vmin=wdm_vmin,
        vmax=wdm_vmax,
    )
    axes[2].axhspan(
        mmin * freq_bin - 0.5 * freq_bin,
        (mmin + coeffs.shape[1]) * freq_bin - 0.5 * freq_bin,
        color="white",
        alpha=0.08,
    )
    axes[2].set_title("Returned compact WDM block positioned on the full global channel grid")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Frequency [Hz]")
    _annotate_metrics(axes[2], snr=wdm_snr, runtime_ms=subband_runtime_ms)
    fig.colorbar(compact_image, ax=axes[2], label="|WDM coefficient|")

    axes[3].plot(full_freqs, np.abs(one_sided), color="0.82", linewidth=1.4, label="full rFFT")
    axes[3].plot(full_freqs, np.abs(recovered_full), color="C3", linewidth=1.8, label="inverse WDM band")
    axes[3].plot(
        fourier_freqs,
        np.abs(recovered),
        color="C0",
        linestyle="--",
        linewidth=1.6,
        label="recovered touched span",
    )
    axes[3].axvspan(
        full_freqs[recovered_kmin],
        full_freqs[recovered_kmin + recovered.shape[0] - 1],
        color="C3",
        alpha=0.12,
    )
    axes[3].set_title("Inverse reconstruction on touched Fourier support")
    axes[3].set_xlabel("Frequency [Hz]")
    axes[3].set_ylabel("Magnitude")
    axes[3].legend(loc="upper right")
    _annotate_metrics(axes[3], snr=fourier_snr)

    fig.savefig(test_outdir / "subband_diagnostics_vertical.png", dpi=140)
    plt.close(fig)
