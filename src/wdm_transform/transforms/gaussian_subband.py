"""Compact Gaussian Wilson transforms for NumPy and CuPy.

The supplied one-sided Fourier crop is zero outside its stated global span.
Analysis gathers its implicit Hermitian extension without allocating a full
FFT grid. Synthesis folds conjugate lobes directly into the touched one-sided
output span, with the same adjoint normalization as the full Gaussian path.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any

from .._window_spec import WindowSpec, resolve_window
from ..backends import Backend
from ..windows_gaussian import gaussian_window
from ._subband import (
    fourier_span_from_wdm_span,
    validate_gaussian_subband_grid,
    wdm_span_from_fourier_span,
)


def _validate_parameters(
    *,
    df: float,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float,
    d: float,
    nu: float,
    support: int,
    offset: int,
    offset_name: str,
) -> WindowSpec:
    """Validate static metadata shared by the NumPy/CuPy and JAX wrappers."""
    validate_gaussian_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    if isinstance(offset, bool) or not isinstance(offset, Integral):
        raise ValueError(f"{offset_name} must be an integer.")
    if isinstance(df, bool) or not isinstance(df, Real) or not isfinite(df) or df <= 0:
        raise ValueError("df must be a finite positive real number.")
    return resolve_window("gaussian", nu, support, nfreqs_wdm, a, d)


def _hermitian_gather(
    data: Any,
    spectrum_indices: Any,
    *,
    kmin: int,
    signal_length: int,
    backend: Backend,
) -> Any:
    """Gather a full periodic FFT from a zero-filled one-sided crop."""
    xp = backend.xp
    one_sided_indices = xp.minimum(spectrum_indices, signal_length - spectrum_indices)
    local_indices = one_sided_indices - kmin
    inside_crop = (local_indices >= 0) & (local_indices < data.shape[0])
    safe_indices = xp.clip(local_indices, 0, data.shape[0] - 1)
    values = data[safe_indices]
    values = xp.where(
        spectrum_indices > signal_length // 2, xp.conjugate(values), values
    )
    self_conjugate = (one_sided_indices == 0) | (
        one_sided_indices == signal_length // 2
    )
    values = xp.where(self_conjugate, values.real, values)
    return xp.where(inside_crop, values, 0.0j)


def _accumulate_one_sided(
    target: Any,
    values: Any,
    spectrum_indices: Any,
    *,
    kmin: int,
    signal_length: int,
    backend: Backend,
) -> None:
    """Add each lobe and its conjugate reflection to a compact positive grid."""
    xp = backend.xp
    one_sided_indices = xp.minimum(spectrum_indices, signal_length - spectrum_indices)
    contributions = xp.where(
        spectrum_indices > signal_length // 2, xp.conjugate(values), values
    )
    self_conjugate = (one_sided_indices == 0) | (
        one_sided_indices == signal_length // 2
    )
    # A self-conjugate FFT bin contains the lobe plus its own conjugate.
    contributions = xp.where(self_conjugate, 2.0 * values.real, contributions)
    local_indices = one_sided_indices - kmin
    inside_output = (local_indices >= 0) & (local_indices < target.shape[0])
    # Reflected taps can share a destination, requiring a true scatter sum.
    valid_indices = local_indices[inside_output]
    valid_values = contributions[inside_output]
    # CuPy's atomic scatter supports float64, but not complex128.
    xp.add.at(target.real, valid_indices, valid_values.real)
    xp.add.at(target.imag, valid_indices, valid_values.imag)


def forward_wdm_subband(
    data: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    kmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float,
    d: float,
    backend: Backend,
    nu: float,
    support: int,
) -> tuple[Any, int]:
    """Analyze a compact one-sided Fourier span into touched Gaussian channels."""
    spec = _validate_parameters(
        df=df,
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        a=a,
        d=d,
        nu=nu,
        support=support,
        offset=kmin,
        offset_name="kmin",
    )
    xp = backend.xp
    spectrum = backend.asarray(data, dtype=xp.complex128)
    if spectrum.ndim != 1:
        raise ValueError("Input Fourier sub-band data must be one-dimensional.")
    mmin, channel_count = wdm_span_from_fourier_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        kmin=kmin,
        lendata=int(spectrum.shape[0]),
        window="gaussian",
        nu=spec.nu,
        support=spec.support,
    )
    nt, nf = int(ntimes_wdm), int(nfreqs_wdm)
    signal_length = nt * nf
    weights = backend.asarray(gaussian_window(nt, spec.nu, spec.support))
    offsets = xp.arange(-spec.support * nt // 2, spec.support * nt // 2)
    times = xp.arange(nt)
    coeffs = xp.empty((nt, channel_count), dtype=xp.float64)
    for local_channel, channel in enumerate(range(mmin, mmin + channel_count)):
        spectrum_indices = (channel * (nt // 2) + offsets) % signal_length
        taps = (
            _hermitian_gather(
                spectrum,
                spectrum_indices,
                kmin=int(kmin),
                signal_length=signal_length,
                backend=backend,
            )
            * weights
        )
        # Fold all spectral tails by residue before the single nt-point FFT.
        folded_spectrum = taps.reshape(spec.support, nt).sum(axis=0)
        folded_spectrum = xp.roll(folded_spectrum, (-spec.support * nt // 2) % nt)
        band_samples = backend.fft.ifft(folded_spectrum) * nt
        if channel in (0, nf):
            coeffs[:, local_channel] = band_samples[(2 * times) % nt].real / xp.sqrt(
                2.0
            )
        else:
            wilson_phase = xp.where((times + channel) % 2 == 0, 1.0, -1j)
            center_shift_sign = xp.where((times * channel) % 2 == 0, 1.0, -1.0)
            coeffs[:, local_channel] = (
                xp.sqrt(2.0) * center_shift_sign * (wilson_phase * band_samples).real
            )
    return coeffs, mmin


def inverse_wdm_subband(
    coeffs: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    mmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float,
    d: float,
    backend: Backend,
    nu: float,
    support: int,
) -> tuple[Any, int]:
    """Synthesize a compact Gaussian channel block into its touched Fourier span."""
    spec = _validate_parameters(
        df=df,
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        a=a,
        d=d,
        nu=nu,
        support=support,
        offset=mmin,
        offset_name="mmin",
    )
    xp = backend.xp
    coefficients = backend.asarray(coeffs, dtype=xp.float64)
    if coefficients.ndim != 2:
        raise ValueError("WDM sub-band coefficients must be a two-dimensional array.")
    if int(coefficients.shape[0]) != int(ntimes_wdm):
        raise ValueError(
            f"WDM sub-band coeffs must have ntimes_wdm={ntimes_wdm} rows; "
            f"got {coefficients.shape[0]}."
        )
    channel_count = int(coefficients.shape[1])
    kmin, output_length = fourier_span_from_wdm_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        mmin=mmin,
        nf_sub_wdm=channel_count,
        window="gaussian",
        nu=spec.nu,
        support=spec.support,
    )
    nt, nf = int(ntimes_wdm), int(nfreqs_wdm)
    signal_length = nt * nf
    weights = backend.asarray(gaussian_window(nt, spec.nu, spec.support))
    offsets = xp.arange(-spec.support * nt // 2, spec.support * nt // 2)
    times = xp.arange(nt)
    spectrum = xp.zeros(output_length, dtype=xp.complex128)
    for local_channel, channel in enumerate(
        range(int(mmin), int(mmin) + channel_count)
    ):
        spectrum_indices = (channel * (nt // 2) + offsets) % signal_length
        if channel in (0, nf):
            band_fft = backend.fft.fft(coefficients[:, local_channel])
            values = band_fft[(2 * offsets) % nt] * weights / xp.sqrt(2.0)
            # Full synthesis projects an edge once, rather than doubling it.
            values = values * 0.5
        else:
            wilson_phase = xp.where((times + channel) % 2 == 0, 1.0, 1j)
            center_shift_sign = xp.where((times * channel) % 2 == 0, 1.0, -1.0)
            band_fft = backend.fft.fft(
                coefficients[:, local_channel] * wilson_phase * center_shift_sign
            )
            values = band_fft[offsets % nt] * weights / xp.sqrt(2.0)
        _accumulate_one_sided(
            spectrum,
            values,
            spectrum_indices,
            kmin=kmin,
            signal_length=signal_length,
            backend=backend,
        )
    return spectrum, kmin
