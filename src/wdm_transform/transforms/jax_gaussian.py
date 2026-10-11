"""JAX Gaussian Wilson kernels with bounded frequency-band workspaces.

Chunked gather/fold and scatter use one nt-point FFT per band. The largest
tap workspace has shape (batch, 16, support*nt), independent of total nf.
Window generation is on the host; only the signal remains differentiable.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp

from ..backends import Backend
from ..windows_gaussian import gaussian_window
from ._gaussian import _coefficients, _inputs

_BANDS_PER_CHUNK = 16


@lru_cache(maxsize=8)
def _device_window(nt: int, nu: float, support: int) -> jax.Array:
    # Safe even when a user's first invocation is inside jax.jit: no tracer
    # can escape into the bounded device cache.
    with jax.ensure_compile_time_eval():
        return jnp.asarray(gaussian_window(nt, nu, support))


def _project(spectrum: jax.Array) -> jax.Array:
    """Project full Fourier data onto the Hermitian spectrum of a real signal."""
    conjugate_indices = (-jnp.arange(spectrum.shape[-1])) % spectrum.shape[-1]
    return (spectrum + jnp.conj(spectrum[..., conjugate_indices])) / 2


@partial(jax.jit, static_argnames=("nt", "nf", "support"))
def _analysis(
    spectrum: jax.Array, weights: jax.Array, *, nt: int, nf: int, support: int
) -> jax.Array:
    offsets = jnp.arange(-support * nt // 2, support * nt // 2)
    times = jnp.arange(nt)
    chunk_count = (nf + 1 + _BANDS_PER_CHUNK - 1) // _BANDS_PER_CHUNK

    def analyze_chunk(_carry: None, chunk: jax.Array) -> tuple[None, jax.Array]:
        channels = chunk * _BANDS_PER_CHUNK + jnp.arange(_BANDS_PER_CHUNK)
        spectrum_indices = (channels[:, None] * (nt // 2) + offsets[None, :]) % (
            nt * nf
        )
        weighted_taps = spectrum[:, spectrum_indices] * weights[None, None, :]
        # Sum equal FFT residues before computing one nt-point inverse FFT per band.
        folded_spectrum = weighted_taps.reshape(
            spectrum.shape[0], _BANDS_PER_CHUNK, support, nt
        ).sum(axis=2)
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
        edge_coefficients = band_samples[:, :, (2 * times) % nt].real / jnp.sqrt(2.0)
        is_edge = (channels == 0) | (channels == nf)
        return None, jnp.where(
            is_edge[None, :, None], edge_coefficients, interior_coefficients
        )

    _, chunk_coefficients = jax.lax.scan(analyze_chunk, None, jnp.arange(chunk_count))
    coefficients = chunk_coefficients.transpose(1, 0, 2, 3).reshape(
        spectrum.shape[0], chunk_count * _BANDS_PER_CHUNK, nt
    )
    # Discard padded bands and restore the public (batch, time, channel) layout.
    return coefficients[:, : nf + 1, :].transpose(0, 2, 1)


@partial(jax.jit, static_argnames=("nt", "nf", "support"))
def _synthesis(
    coefficients: jax.Array, weights: jax.Array, *, nt: int, nf: int, support: int
) -> jax.Array:
    offsets = jnp.arange(-support * nt // 2, support * nt // 2)
    times = jnp.arange(nt)
    chunk_count = (nf + 1 + _BANDS_PER_CHUNK - 1) // _BANDS_PER_CHUNK
    padded_coefficients = jnp.pad(
        coefficients.transpose(0, 2, 1),
        ((0, 0), (0, chunk_count * _BANDS_PER_CHUNK - nf - 1), (0, 0)),
    )
    batch_indices = jnp.arange(coefficients.shape[0])[:, None, None]

    def synthesize_chunk(chunk: jax.Array, spectrum: jax.Array) -> jax.Array:
        channels = chunk * _BANDS_PER_CHUNK + jnp.arange(_BANDS_PER_CHUNK)
        chunk_coefficients = jax.lax.dynamic_slice(
            padded_coefficients,
            (0, chunk * _BANDS_PER_CHUNK, 0),
            (coefficients.shape[0], _BANDS_PER_CHUNK, nt),
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
        tap_contributions = jnp.take_along_axis(
            band_fft, fft_residues[None, :, :], axis=-1
        )
        # The final conjugate reflection supplies the other half of edges.
        lobe_scales = jnp.where(is_edge, 0.5, 1.0) / jnp.sqrt(2.0)
        tap_contributions = (
            tap_contributions * weights[None, None, :] * lobe_scales[None, :, None]
        )
        spectrum_indices = (channels[:, None] * (nt // 2) + offsets[None, :]) % (
            nt * nf
        )
        return spectrum.at[batch_indices, spectrum_indices[None, :, :]].add(
            tap_contributions
        )

    spectrum = jax.lax.fori_loop(
        0,
        chunk_count,
        synthesize_chunk,
        jnp.zeros((coefficients.shape[0], nt * nf), jnp.complex128),
    )
    conjugate_indices = (-jnp.arange(nt * nf)) % (nt * nf)
    return spectrum + jnp.conj(spectrum[:, conjugate_indices])


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
    spectrum = jnp.fft.fft(samples.real, axis=-1)
    return _analysis(
        spectrum, _device_window(nt, nu, support), nt=nt, nf=nf, support=support
    )


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
    spectrum = _project(_inputs(data, nt, nf, backend))
    return _analysis(
        spectrum, _device_window(nt, nu, support), nt=nt, nf=nf, support=support
    )


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
    coefficients = _coefficients(coeffs, backend)
    nt, nf = coefficients.shape[-2], coefficients.shape[-1] - 1
    return _synthesis(
        coefficients, _device_window(nt, nu, support), nt=nt, nf=nf, support=support
    )


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
    return jnp.fft.ifft(spectrum, axis=-1).real
