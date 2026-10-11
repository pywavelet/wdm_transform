"""Native JAX Gaussian transforms on compact one-sided Fourier spans.

The signal is never expanded onto the full Fourier or WDM grid. Analysis
gathers the implicit Hermitian spectrum from the supplied crop; synthesis
adds both Wilson lobes directly to the returned one-sided crop. Workspaces
are bounded to 16 channels of ``support * nt`` taps.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ..backends import Backend
from ._subband import (
    fourier_span_from_wdm_span,
    validate_fourier_span,
    validate_wdm_span,
    wdm_span_from_fourier_span,
)
from .gaussian_subband import _validate_parameters
from .jax_gaussian import _BANDS_PER_CHUNK, _device_window


def _hermitian_gather(
    spectrum: jax.Array, full_indices: jax.Array, *, kmin: int, n_total: int
) -> jax.Array:
    """Read a full-FFT index from a compact crop, conjugating negative bins."""
    one_sided_indices = jnp.minimum(full_indices, n_total - full_indices)
    local_indices = one_sided_indices - kmin
    present = (local_indices >= 0) & (local_indices < spectrum.shape[0])
    safe_indices = jnp.clip(local_indices, 0, spectrum.shape[0] - 1)
    values = jnp.where(present, spectrum[safe_indices], 0.0j)
    values = jnp.where(full_indices > n_total // 2, jnp.conj(values), values)
    self_conjugate = (full_indices == 0) | (full_indices == n_total // 2)
    return jnp.where(self_conjugate, values.real, values)


@partial(
    jax.jit,
    static_argnames=("nt", "nf", "support", "kmin", "mmin", "channel_count"),
)
def _analysis(
    spectrum: jax.Array,
    weights: jax.Array,
    *,
    nt: int,
    nf: int,
    support: int,
    kmin: int,
    mmin: int,
    channel_count: int,
) -> jax.Array:
    offsets = jnp.arange(-support * nt // 2, support * nt // 2)
    times = jnp.arange(nt)
    chunk_count = (channel_count + _BANDS_PER_CHUNK - 1) // _BANDS_PER_CHUNK

    def analyze_chunk(_carry: None, chunk: jax.Array) -> tuple[None, jax.Array]:
        channels = mmin + chunk * _BANDS_PER_CHUNK + jnp.arange(_BANDS_PER_CHUNK)
        full_indices = (channels[:, None] * (nt // 2) + offsets[None, :]) % (nt * nf)
        gathered = _hermitian_gather(spectrum, full_indices, kmin=kmin, n_total=nt * nf)
        weighted_taps = gathered * weights[None, :]
        # Equal FFT residues fold together before one nt-point IFFT per channel.
        folded_spectrum = weighted_taps.reshape(_BANDS_PER_CHUNK, support, nt).sum(
            axis=1
        )
        folded_spectrum = jnp.roll(folded_spectrum, (-support * nt // 2) % nt, axis=-1)
        band_samples = jnp.fft.ifft(folded_spectrum, axis=-1) * nt
        wilson_phase = jnp.where(
            (times[None, :] + channels[:, None]) % 2 == 0, 1.0, -1j
        )
        center_shift_sign = jnp.where(
            (times[None, :] * channels[:, None]) % 2 == 0, 1.0, -1.0
        )
        interior_coefficients = (
            jnp.sqrt(2.0) * center_shift_sign * (wilson_phase * band_samples).real
        )
        # DC/Nyquist retain the duplicated, half-norm edge convention.
        edge_coefficients = band_samples[:, (2 * times) % nt].real / jnp.sqrt(2.0)
        is_edge = (channels == 0) | (channels == nf)
        return None, jnp.where(
            is_edge[:, None], edge_coefficients, interior_coefficients
        )

    _, chunk_coefficients = jax.lax.scan(analyze_chunk, None, jnp.arange(chunk_count))
    coefficients = chunk_coefficients.reshape(chunk_count * _BANDS_PER_CHUNK, nt)
    return coefficients[:channel_count].T


@partial(
    jax.jit,
    static_argnames=("nt", "nf", "support", "mmin", "kmin", "fourier_count"),
)
def _synthesis(
    coefficients: jax.Array,
    weights: jax.Array,
    *,
    nt: int,
    nf: int,
    support: int,
    mmin: int,
    kmin: int,
    fourier_count: int,
) -> jax.Array:
    channel_count = coefficients.shape[1]
    chunk_count = (channel_count + _BANDS_PER_CHUNK - 1) // _BANDS_PER_CHUNK
    padded_coefficients = jnp.pad(
        coefficients.T,
        ((0, chunk_count * _BANDS_PER_CHUNK - channel_count), (0, 0)),
    )
    offsets = jnp.arange(-support * nt // 2, support * nt // 2)
    times = jnp.arange(nt)
    n_total = nt * nf

    def synthesize_chunk(chunk: jax.Array, spectrum: jax.Array) -> jax.Array:
        channels = mmin + chunk * _BANDS_PER_CHUNK + jnp.arange(_BANDS_PER_CHUNK)
        chunk_coefficients = jax.lax.dynamic_slice(
            padded_coefficients, (chunk * _BANDS_PER_CHUNK, 0), (_BANDS_PER_CHUNK, nt)
        )
        is_edge = (channels == 0) | (channels == nf)
        wilson_phase = jnp.where((times[None, :] + channels[:, None]) % 2 == 0, 1.0, 1j)
        center_shift_sign = jnp.where(
            (times[None, :] * channels[:, None]) % 2 == 0, 1.0, -1.0
        )
        modulation = jnp.where(is_edge[:, None], 1.0, wilson_phase * center_shift_sign)
        band_fft = jnp.fft.fft(chunk_coefficients * modulation, axis=-1)
        fft_residues = (
            jnp.where(is_edge[:, None], 2 * offsets[None, :], offsets[None, :]) % nt
        )
        tap_contributions = jnp.take_along_axis(band_fft, fft_residues, axis=-1)
        # Edges use half the lobe scale, as in the full-spectrum adjoint.
        lobe_scales = jnp.where(is_edge, 0.5, 1.0) / jnp.sqrt(2.0)
        tap_contributions = tap_contributions * weights[None, :] * lobe_scales[:, None]
        full_indices = (channels[:, None] * (nt // 2) + offsets[None, :]) % n_total

        # Fold both lobes directly into the one-sided output. A self-conjugate
        # bin receives v + conj(v), rather than a second independent bin.
        one_sided_indices = jnp.minimum(full_indices, n_total - full_indices)
        values = jnp.where(
            full_indices > n_total // 2, jnp.conj(tap_contributions), tap_contributions
        )
        self_conjugate = (full_indices == 0) | (full_indices == n_total // 2)
        values = jnp.where(self_conjugate, 2 * tap_contributions.real, values)
        local_indices = one_sided_indices - kmin
        present = (local_indices >= 0) & (local_indices < fourier_count)
        present = present & (channels[:, None] < mmin + channel_count)
        safe_indices = jnp.clip(local_indices, 0, fourier_count - 1)
        # Native scatter-add is necessary: reflection creates repeated indices.
        return spectrum.at[safe_indices].add(jnp.where(present, values, 0.0j))

    return jax.lax.fori_loop(
        0,
        chunk_count,
        synthesize_chunk,
        jnp.zeros(fourier_count, dtype=jnp.complex128),
    )


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
    """Analyze a compact Fourier crop with zeros outside its implicit mirror."""
    specification = _validate_parameters(
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
    spectrum = backend.asarray(data, dtype=jnp.complex128)
    if spectrum.ndim != 1:
        raise ValueError("Input Fourier sub-band data must be one-dimensional.")
    validate_fourier_span(
        nfreqs_fourier=nfreqs_fourier, kmin=kmin, lendata=spectrum.shape[0]
    )
    mmin, channel_count = wdm_span_from_fourier_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        kmin=kmin,
        lendata=spectrum.shape[0],
        window="gaussian",
        nu=specification.nu,
        support=specification.support,
    )
    coefficients = _analysis(
        spectrum,
        _device_window(ntimes_wdm, specification.nu, specification.support),
        nt=ntimes_wdm,
        nf=nfreqs_wdm,
        support=specification.support,
        kmin=kmin,
        mmin=mmin,
        channel_count=channel_count,
    )
    return coefficients, mmin


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
    """Synthesize a compact one-sided spectrum from the supplied channels."""
    specification = _validate_parameters(
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
    coefficients = backend.asarray(coeffs, dtype=jnp.float64)
    if coefficients.ndim != 2:
        raise ValueError("WDM sub-band coefficients must be a two-dimensional array.")
    if coefficients.shape[0] != ntimes_wdm:
        raise ValueError(
            f"WDM sub-band coeffs must have ntimes_wdm={ntimes_wdm} rows; "
            f"got {coefficients.shape[0]}."
        )
    validate_wdm_span(
        nfreqs_wdm=nfreqs_wdm, mmin=mmin, nf_sub_wdm=coefficients.shape[1]
    )
    kmin, fourier_count = fourier_span_from_wdm_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        mmin=mmin,
        nf_sub_wdm=coefficients.shape[1],
        window="gaussian",
        nu=specification.nu,
        support=specification.support,
    )
    spectrum = _synthesis(
        coefficients,
        _device_window(ntimes_wdm, specification.nu, specification.support),
        nt=ntimes_wdm,
        nf=nfreqs_wdm,
        support=specification.support,
        mmin=mmin,
        kmin=kmin,
        fourier_count=fourier_count,
    )
    return spectrum, kmin
