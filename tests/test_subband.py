from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import pytest
import time
from importlib.util import find_spec

from wdm_transform import TimeSeries, WDM
from wdm_transform.backends import get_backend
from wdm_transform.transforms import (
    from_freq_to_wdm_band,
    from_time_to_wdm,
    fourier_span_from_wdm_span,
    forward_wdm_subband,
    inverse_wdm_subband,
    wdm_span_from_fourier_span,
)
from wdm_transform.windows import cnm, phi_unit
import matplotlib.pyplot as plt


NT = 32
NF = 32
N_TOTAL = NT * NF
NFOURIER = N_TOTAL // 2 + 1
DT = 0.125
DF = 1.0 / (N_TOTAL * DT)
NUMPY_BACKEND = get_backend("numpy")


@dataclass(frozen=True)
class ForwardSubbandResult:
    coeffs: np.ndarray
    mmin: int
    expected_mmin: int
    expected_nf_sub: int
    full_spectrum: np.ndarray
    full_wdm_slice: np.ndarray


@dataclass(frozen=True)
class InverseSubbandResult:
    forward: ForwardSubbandResult
    recovered: np.ndarray
    recovered_kmin: int
    expected_kmin: int
    expected_lendata: int


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


@lru_cache(maxsize=None)
def _wdm_block_analysis_matrix(
    *,
    nt: int,
    nf: int,
    dt: float,
    mmin: int,
    nf_sub_wdm: int,
    a: float = 1.0 / 3.0,
) -> np.ndarray:
    """Dense forward WDM operator restricted to a compact channel block."""
    n_total = nt * nf
    matrix = np.empty((nt * nf_sub_wdm, n_total), dtype=float)
    for idx in range(n_total):
        basis = np.zeros(n_total, dtype=float)
        basis[idx] = 1.0
        coeffs = from_time_to_wdm(
            basis,
            nt=nt,
            nf=nf,
            a=a,
            d=1.0,
            dt=dt,
            backend="numpy",
        )[0, :, mmin:mmin + nf_sub_wdm]
        matrix[:, idx] = coeffs.reshape(-1)
    return matrix


@lru_cache(maxsize=None)
def _rfft_block_analysis_matrix(
    *,
    nt: int,
    nf: int,
    dt: float,
    kmin: int,
    lendata: int,
) -> np.ndarray:
    """Dense real-valued operator for a compact one-sided rFFT span.

    The output stacks real and imaginary parts so the white-noise covariance
    lives on a standard real vector space.
    """
    n_total = nt * nf
    matrix = np.empty((2 * lendata, n_total), dtype=float)
    for idx in range(n_total):
        basis = np.zeros(n_total, dtype=float)
        basis[idx] = 1.0
        coeffs = np.fft.rfft(basis)[kmin:kmin + lendata]
        matrix[:lendata, idx] = np.real(coeffs)
        matrix[lendata:, idx] = np.imag(coeffs)
    return matrix


def _matched_filter_snr_from_linear_observable(
    signal_observable: np.ndarray,
    analysis_matrix: np.ndarray,
    *,
    sigma: float,
) -> float:
    """Exact matched-filter SNR for white time-domain noise.

    If ``y = A x`` is the observed linear data vector and the underlying
    time-domain noise is white with covariance ``sigma^2 I``, then the induced
    covariance in the observable space is

        C_y = sigma^2 A A^T.

    The optimal matched-filter SNR is the noise-weighted inner product

        rho^2 = (y|y) = y^T C_y^+ y,

    where ``C_y^+`` denotes the pseudoinverse. This is the exact finite-
    dimensional version of the usual GW inner product.
    """
    cov = sigma**2 * (analysis_matrix @ analysis_matrix.T)
    snr2 = float(signal_observable @ (np.linalg.pinv(cov, rcond=1e-12) @ signal_observable))
    return float(np.sqrt(max(snr2, 0.0)))


def _matched_filter_snr_wdm_diagonal(
    coeffs: np.ndarray,
    noise_var: np.ndarray,
) -> float:
    """Diagonal WDM matched-filter SNR.

    This is the WDM-domain approximation used in the study notes:

        rho^2 = sum_{nm} h_nm^2 / S_nm,

    where ``S_nm`` is treated as diagonal coefficient variance.
    """
    coeffs_arr = np.asarray(coeffs, dtype=float)
    noise_var_arr = np.asarray(noise_var, dtype=float)
    snr2 = np.sum(coeffs_arr**2 / np.maximum(noise_var_arr, 1e-60))
    return float(np.sqrt(max(float(snr2), 0.0)))


def _format_snr(value: float) -> str:
    if np.isinf(value):
        return "MF SNR: inf"
    return f"MF SNR: {value:.2f}"


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
    mf_snr: float,
    diagonal_mf_snr: float | None = None,
    runtime_ms: float | None = None,
) -> None:
    lines = [_format_snr(mf_snr)]
    if diagonal_mf_snr is not None:
        lines.append(f"Diag WDM MF SNR: {diagonal_mf_snr:.2f}")
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


def _forward_subband_result(
    subband: np.ndarray,
    *,
    kmin: int,
    backend: str = "numpy",
) -> ForwardSubbandResult:
    lendata = int(len(subband))
    one_sided = _one_sided_full_spectrum(kmin, np.asarray(subband))
    signal = np.fft.irfft(one_sided, n=N_TOTAL)
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    coeffs, mmin = forward_wdm_subband(
        subband,
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        backend=backend,
    )
    expected_mmin, expected_nf_sub = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=kmin,
        lendata=lendata,
    )
    return ForwardSubbandResult(
        coeffs=np.asarray(coeffs),
        mmin=int(mmin),
        expected_mmin=expected_mmin,
        expected_nf_sub=expected_nf_sub,
        full_spectrum=one_sided,
        full_wdm_slice=np.asarray(full_wdm.coeffs)[0, :, expected_mmin:expected_mmin + expected_nf_sub],
    )


def _assert_forward_matches_full_slice(result: ForwardSubbandResult) -> None:
    assert result.mmin == result.expected_mmin
    assert result.coeffs.shape == (NT, result.expected_nf_sub)
    np.testing.assert_allclose(
        result.coeffs,
        result.full_wdm_slice,
        atol=1e-10,
        rtol=1e-10,
    )


def _inverse_subband_result(
    subband: np.ndarray,
    *,
    kmin: int,
    backend: str = "numpy",
) -> InverseSubbandResult:
    forward = _forward_subband_result(subband, kmin=kmin, backend=backend)
    recovered, recovered_kmin = inverse_wdm_subband(
        forward.coeffs,
        df=DF,
        nfreqs_fourier=NFOURIER,
        mmin=forward.mmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        backend=backend,
    )
    expected_kmin, expected_lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=forward.mmin,
        nf_sub_wdm=forward.coeffs.shape[1],
    )
    return InverseSubbandResult(
        forward=forward,
        recovered=np.asarray(recovered),
        recovered_kmin=int(recovered_kmin),
        expected_kmin=expected_kmin,
        expected_lendata=expected_lendata,
    )


def _assert_roundtrip_matches_touched_fourier(result: InverseSubbandResult) -> None:
    assert result.recovered_kmin == result.expected_kmin
    np.testing.assert_allclose(
        result.recovered,
        result.forward.full_spectrum[result.expected_kmin:result.expected_kmin + result.expected_lendata],
        atol=1e-10,
        rtol=1e-10,
    )


def _chirping_binary_subband_case() -> tuple[np.ndarray, np.ndarray, int, int]:
    signal = _generate_wd_binary_signal(
        f0=0.15,
        fdot=5.0e-4,
        amplitude=1.0,
    )
    one_sided = np.fft.rfft(signal)
    k_peak = int(np.argmax(np.abs(one_sided)))
    kmin = max(0, k_peak - 3)
    lendata = 8
    return signal, one_sided, kmin, lendata


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


def test_lisa_style_local_frequency_band_touches_expected_wdm_channels() -> None:
    """Mirror the study's Fourier-support rule and check the touched channel span."""
    band_start = 7
    band_stop = 10
    half = NT // 2
    kmin = max((band_start - 1) * half, 0)
    kmax = min(band_stop * half, NFOURIER)
    lendata = kmax - kmin

    mapped_wdm = wdm_span_from_fourier_span(
        nfreqs_fourier=NFOURIER,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        kmin=kmin,
        lendata=lendata,
    )
    assert mapped_wdm == (band_start - 1, band_stop - band_start + 2)


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
    result = _forward_subband_result(_subband_case_data(kmin, lendata), kmin=kmin)
    _assert_forward_matches_full_slice(result)


def test_lisa_style_local_frequency_band_matches_expected_wdm_slice() -> None:
    rng = np.random.default_rng(1234)
    signal = rng.normal(size=N_TOTAL)
    one_sided = np.fft.rfft(signal)
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)

    band_start = 7
    band_stop = 10
    half = NT // 2
    kmin = max((band_start - 1) * half, 0)
    kmax = min(band_stop * half, NFOURIER)

    coeffs = from_freq_to_wdm_band(
        one_sided[kmin:kmax],
        df=DF,
        nfreqs_fourier=NFOURIER,
        kmin=kmin,
        nfreqs_wdm=NF,
        ntimes_wdm=NT,
        mmin=band_start,
        nf_sub_wdm=band_stop - band_start,
    )
    assert coeffs.shape == (NT, band_stop - band_start)
    np.testing.assert_allclose(
        coeffs,
        np.asarray(full_wdm.coeffs)[0, :, band_start:band_stop],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.skipif(find_spec("jax") is None, reason="jax is not installed")
def test_forward_subband_matches_full_wdm_slice_jax() -> None:
    jax = pytest.importorskip("jax.numpy")
    kmin = 23
    lendata = 29
    result = _forward_subband_result(
        jax.asarray(_subband_case_data(kmin, lendata)),
        kmin=kmin,
        backend="jax",
    )
    _assert_forward_matches_full_slice(result)


def test_forward_subband_matches_full_wdm_slice_for_sinusoid() -> None:
    k0 = 37
    signal = _generate_sinusoid(k0=k0, amplitude=1.7, phase=0.31)
    result = _forward_subband_result(np.fft.rfft(signal)[k0:k0 + 1], kmin=k0)
    _assert_forward_matches_full_slice(result)


def test_full_wdm_matches_analytic_sinusoid_formula() -> None:
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

    np.testing.assert_allclose(
        np.asarray(full_wdm.coeffs)[0],
        np.real(analytic),
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
    coeffs = np.zeros_like(np.asarray(full_wdm.coeffs)[0])
    coeffs[:, mmin:mmin + nf_sub_wdm] = np.asarray(full_wdm.coeffs)[0, :, mmin:mmin + nf_sub_wdm]

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
        np.asarray(expected.data)[0, kmin:kmin + lendata],
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
    result = _inverse_subband_result(_subband_case_data(kmin, lendata), kmin=kmin)
    _assert_roundtrip_matches_touched_fourier(result)


def test_chirping_binary_subband_roundtrip() -> None:
    _, one_sided, kmin, lendata = _chirping_binary_subband_case()
    result = _inverse_subband_result(one_sided[kmin:kmin + lendata], kmin=kmin)
    _assert_roundtrip_matches_touched_fourier(result)


def test_chirping_binary_subband_matched_filter_snr_matches_fourier() -> None:
    sigma = 0.3
    _, one_sided, kmin, lendata = _chirping_binary_subband_case()
    result = _forward_subband_result(one_sided[kmin:kmin + lendata], kmin=kmin)

    fourier_operator = _rfft_block_analysis_matrix(
        nt=NT,
        nf=NF,
        dt=DT,
        kmin=kmin,
        lendata=lendata,
    )
    wdm_operator = _wdm_block_analysis_matrix(
        nt=NT,
        nf=NF,
        dt=DT,
        mmin=result.mmin,
        nf_sub_wdm=result.coeffs.shape[1],
    )
    signal_fourier = np.concatenate([np.real(one_sided[kmin:kmin + lendata]), np.imag(one_sided[kmin:kmin + lendata])])
    signal_wdm = result.coeffs.reshape(-1)

    snr_fourier = _matched_filter_snr_from_linear_observable(
        signal_fourier,
        fourier_operator,
        sigma=sigma,
    )
    snr_wdm = _matched_filter_snr_from_linear_observable(
        signal_wdm,
        wdm_operator,
        sigma=sigma,
    )
    noise_var_diag = sigma**2 * np.sum(wdm_operator**2, axis=1).reshape(result.coeffs.shape)
    snr_wdm_diag = _matched_filter_snr_wdm_diagonal(result.coeffs, noise_var_diag)
    np.testing.assert_allclose(snr_wdm, snr_fourier, rtol=1e-10, atol=1e-10)
    assert np.isfinite(snr_wdm_diag)


@pytest.mark.skipif(find_spec("jax") is None, reason="jax is not installed")
def test_chirping_binary_subband_roundtrip_jax() -> None:
    jax = pytest.importorskip("jax.numpy")
    _, one_sided, kmin, lendata = _chirping_binary_subband_case()
    result = _inverse_subband_result(
        jax.asarray(one_sided[kmin:kmin + lendata]),
        kmin=kmin,
        backend="jax",
    )
    _assert_roundtrip_matches_touched_fourier(result)


def test_chirping_binary_subband_diagnostics_plot(outdir) -> None:

    signal, one_sided, kmin, lendata = _chirping_binary_subband_case()
    full_runtime_ms = _mean_runtime_ms(
        lambda: WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    )
    full_wdm = WDM.from_time_series(TimeSeries(signal, dt=DT), nt=NT)
    subband = one_sided[kmin:kmin + lendata]
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
    result = _inverse_subband_result(subband, kmin=kmin)
    _assert_roundtrip_matches_touched_fourier(result)
    coeffs = result.forward.coeffs
    mmin = result.forward.mmin
    recovered = result.recovered
    recovered_kmin = result.recovered_kmin
    reference_fourier = subband
    reference_coeffs = coeffs
    sigma = 0.3
    fourier_operator = _rfft_block_analysis_matrix(
        nt=NT,
        nf=NF,
        dt=DT,
        kmin=kmin,
        lendata=lendata,
    )
    wdm_operator = _wdm_block_analysis_matrix(
        nt=NT,
        nf=NF,
        dt=DT,
        mmin=mmin,
        nf_sub_wdm=coeffs.shape[1],
    )
    fourier_mf_snr = _matched_filter_snr_from_linear_observable(
        np.concatenate([np.real(reference_fourier), np.imag(reference_fourier)]),
        fourier_operator,
        sigma=sigma,
    )
    wdm_mf_snr = _matched_filter_snr_from_linear_observable(
        reference_coeffs.reshape(-1),
        wdm_operator,
        sigma=sigma,
    )
    wdm_diag_noise_var = sigma**2 * np.sum(wdm_operator**2, axis=1).reshape(reference_coeffs.shape)
    wdm_diag_mf_snr = _matched_filter_snr_wdm_diagonal(
        reference_coeffs,
        wdm_diag_noise_var,
    )
    np.testing.assert_allclose(wdm_mf_snr, fourier_mf_snr, rtol=1e-10, atol=1e-10)

    fourier_freqs = (recovered_kmin + np.arange(recovered.shape[0])) * DF
    full_freqs = np.fft.rfftfreq(N_TOTAL, d=DT)
    time_bin = NF * DT
    freq_bin = 1.0 / (2.0 * NF * DT)
    recovered_full = _one_sided_full_spectrum(recovered_kmin, recovered)
    full_band_coeffs = np.full((NT, NF + 1), np.nan, dtype=float)
    full_band_coeffs[:, mmin:mmin + coeffs.shape[1]] = np.abs(coeffs)
    full_wdm_abs = np.abs(np.asarray(full_wdm.coeffs)[0])
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
    _annotate_metrics(axes[0], mf_snr=fourier_mf_snr)

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
    _annotate_metrics(
        axes[1],
        mf_snr=wdm_mf_snr,
        diagonal_mf_snr=wdm_diag_mf_snr,
        runtime_ms=full_runtime_ms,
    )
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
    _annotate_metrics(
        axes[2],
        mf_snr=wdm_mf_snr,
        diagonal_mf_snr=wdm_diag_mf_snr,
        runtime_ms=subband_runtime_ms,
    )
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
    _annotate_metrics(axes[3], mf_snr=fourier_mf_snr)

    fig.savefig(outdir / "subband_diagnostics_vertical.png", dpi=140)
    plt.close(fig)
