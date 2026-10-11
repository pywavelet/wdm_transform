"""Extended-support Wilson analysis/synthesis for NumPy and CuPy.

Only one length-nt FFT is used per band. Weighted spectral taps are folded
modulo nt for analysis, and unfolded/scattered onto the full periodic FFT for
synthesis. Synthesis is the adjoint divided by N, including redundant edges.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from ..backends import Backend
from ..windows import validate_transform_shape
from ..windows_gaussian import gaussian_window


def _project(spectrum: Any, backend: Backend) -> Any:
    """Keep the Hermitian component, equivalent to FFT(real(IFFT(spectrum)))."""
    xp = backend.xp
    conjugate_indices = (-xp.arange(spectrum.shape[-1])) % spectrum.shape[-1]
    return (spectrum + xp.conjugate(spectrum[..., conjugate_indices])) / 2


def _inputs(data: Any, nt: int, nf: int, backend: Backend) -> Any:
    if (
        isinstance(nt, bool)
        or not isinstance(nt, Integral)
        or isinstance(nf, bool)
        or not isinstance(nf, Integral)
        or nt < 2
        or nf < 2
    ):
        raise ValueError("nt and nf must be positive even integers.")
    validate_transform_shape(nt, nf)
    samples = backend.asarray(data, dtype=backend.xp.complex128)
    if samples.ndim not in (1, 2):
        raise ValueError("Input data must be one- or two-dimensional.")
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[-1] != nt * nf:
        raise ValueError(f"Input length {samples.shape[-1]} must equal nt*nf={nt*nf}.")
    return samples


def _coefficients(coeffs: Any, backend: Backend) -> Any:
    coefficients = backend.asarray(coeffs, dtype=backend.xp.float64)
    if coefficients.ndim not in (2, 3):
        raise ValueError("WDM coefficients must be a two- or three-dimensional array.")
    if coefficients.ndim == 2:
        coefficients = coefficients[None, :, :]
    nt, nf = coefficients.shape[-2], coefficients.shape[-1] - 1
    validate_transform_shape(nt, nf)
    if nt < 2 or nf < 2:
        raise ValueError("nt and nf must be positive even integers.")
    return coefficients


def _analysis(
    spectrum: Any, nt: int, nf: int, nu: float, support: int, backend: Backend
) -> Any:
    xp = backend.xp
    weights = backend.asarray(gaussian_window(nt, nu, support))
    offsets = xp.arange(-support * nt // 2, support * nt // 2)
    times = xp.arange(nt)
    coeffs = xp.empty((spectrum.shape[0], nt, nf + 1), dtype=xp.float64)
    for channel in range(nf + 1):
        spectrum_indices = (channel * (nt // 2) + offsets) % (nt * nf)
        weighted_taps = spectrum[:, spectrum_indices] * weights[None, :]
        # Taps with the same residue share one nt-point inverse FFT.
        folded_spectrum = weighted_taps.reshape(spectrum.shape[0], support, nt).sum(
            axis=1
        )
        folded_spectrum = xp.roll(folded_spectrum, (-support * nt // 2) % nt, axis=-1)
        band_samples = backend.fft.ifft(folded_spectrum, axis=-1) * nt
        if channel in (0, nf):
            # Doubled time phases produce the duplicated half-norm edge atoms.
            coeffs[:, :, channel] = band_samples[:, (2 * times) % nt].real / xp.sqrt(
                2.0
            )
        else:
            wilson_phase = xp.where((times + channel) % 2 == 0, 1.0, -1j)
            center_shift_sign = xp.where((times * channel) % 2 == 0, 1.0, -1.0)
            coeffs[:, :, channel] = (
                xp.sqrt(2.0) * center_shift_sign * (wilson_phase * band_samples).real
            )
    return coeffs


def _synthesis(coefficients: Any, nu: float, support: int, backend: Backend) -> Any:
    xp = backend.xp
    nt, nf = coefficients.shape[-2], coefficients.shape[-1] - 1
    signal_length = nt * nf
    weights = backend.asarray(gaussian_window(nt, nu, support))
    offsets = xp.arange(-support * nt // 2, support * nt // 2)
    times = xp.arange(nt)
    interior_spectrum = xp.zeros(
        (coefficients.shape[0], signal_length), dtype=xp.complex128
    )
    edge_spectrum = xp.zeros_like(interior_spectrum)
    # A band's offsets have length <= N, hence no duplicate scatter indices.
    for channel in range(nf + 1):
        spectrum_indices = (channel * (nt // 2) + offsets) % signal_length
        if channel in (0, nf):
            band_fft = backend.fft.fft(coefficients[:, :, channel], axis=-1)[
                :, (2 * offsets) % nt
            ]
            edge_spectrum[:, spectrum_indices] += band_fft * weights / xp.sqrt(2.0)
        else:
            wilson_phase = xp.where((times + channel) % 2 == 0, 1.0, 1j)
            center_shift_sign = xp.where((times * channel) % 2 == 0, 1.0, -1.0)
            band_fft = backend.fft.fft(
                coefficients[:, :, channel] * wilson_phase * center_shift_sign, axis=-1
            )
            interior_spectrum[:, spectrum_indices] += (
                band_fft[:, offsets % nt] * weights / xp.sqrt(2.0)
            )
    conjugate_indices = (-xp.arange(signal_length)) % signal_length
    # Real Wilson atoms contain both lobes; self-conjugate edges are projected once.
    return (
        _project(edge_spectrum, backend)
        + interior_spectrum
        + xp.conjugate(interior_spectrum[:, conjugate_indices])
    )


def from_time_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
    nu: float,
    support: int,
) -> Any:
    samples = _inputs(data, nt, nf, backend)
    spectrum = backend.fft.fft(samples.real, axis=-1)
    return _analysis(spectrum, nt, nf, nu, support, backend)


def from_freq_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
    nu: float,
    support: int,
) -> Any:
    spectrum = _project(_inputs(data, nt, nf, backend), backend)
    return _analysis(spectrum, nt, nf, nu, support, backend)


def from_wdm_to_freq(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
    nu: float,
    support: int,
) -> Any:
    return _synthesis(_coefficients(coeffs, backend), nu, support, backend)


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
    nu: float,
    support: int,
) -> Any:
    spectrum = from_wdm_to_freq(
        coeffs, a=a, d=d, dt=dt, backend=backend, nu=nu, support=support
    )
    # The normalized inverse FFT supplies the adjoint synthesis factor 1/N.
    return backend.fft.ifft(spectrum, axis=-1).real
