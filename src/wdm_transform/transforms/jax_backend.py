from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit, lax

from ..backends import Backend
from ..datatypes.series import FrequencySeries
from ..windows import phi_window, validate_transform_shape, validate_window_parameter


def _cnm_jax(n: Any, m: Any) -> Any:
    parity = jnp.where((n + m) % 2 == 0, 1.0, -1.0)
    return jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity))


def _phi_unit_jax(f: Any, a: float) -> Any:
    b = 1.0 - 2.0 * a
    abs_f = jnp.abs(f)
    tapered = jnp.cos((jnp.pi / 2.0) * (abs_f - a) / b)
    inner = jnp.where(abs_f > a, tapered, 1.0)
    return jnp.where(abs_f > a + b, 0.0, inner)


@partial(jit, static_argnames=("nt", "nf"))
def _forward_wdm_impl(
    samples: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    n_total = nt * nf
    half = nt // 2
    x_fft = jnp.fft.fft(samples)
    narr = jnp.arange(nt)
    sqrt2 = jnp.sqrt(2.0)

    block = x_fft[1:half] * window[1:half]
    larr = jnp.arange(1, half)
    coeffs_dc = jnp.real(
        jnp.sum(
            jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[0] * window[0] / 2.0
    ) / (nt * nf)

    mid_m = jnp.arange(1, nf)
    upper = mid_m[:, None] * half + jnp.arange(half)[None, :]
    lower = (mid_m[:, None] - 1) * half + jnp.arange(half)[None, :]
    mid_indices = jnp.concatenate([upper, lower], axis=1)
    mid_blocks = x_fft[mid_indices] * window[None, :]
    mid_times = jnp.fft.ifft(mid_blocks, axis=1).T
    mid_phase = jnp.conjugate(_cnm_jax(narr[:, None], mid_m[None, :]))
    coeffs_mid = (sqrt2 / nf) * jnp.real(mid_phase * mid_times)

    block = x_fft[n_total // 2 - half:n_total // 2] * window[-half:]
    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    coeffs_nyq = jnp.real(
        jnp.sum(
            jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[n_total // 2] * window[0] / 2.0
    ) / (nt * nf)

    first_column = (coeffs_dc + 1j * coeffs_nyq)[:, None]
    return jnp.concatenate([first_column, coeffs_mid], axis=1)


@partial(jit, static_argnames=("nt", "nf"))
def _inverse_wdm_impl(
    packed: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    n_total = nt * nf
    half = nt // 2
    coeffs_dc = jnp.real(packed[:, 0])
    coeffs_nyq = jnp.imag(packed[:, 0])

    n_idx = jnp.arange(nt)[:, None]
    m_idx = jnp.arange(nf)[None, :]
    ylm = _cnm_jax(n_idx, m_idx) * jnp.real(packed) * nf / jnp.sqrt(2.0)
    spectrum_blocks = jnp.fft.fft(ylm, axis=0)

    x_recon = jnp.zeros(n_total, dtype=packed.dtype)
    narr = jnp.arange(nt)

    larr = jnp.arange(1, half)
    x_recon = x_recon.at[1:half].add(
        jnp.sum(
            coeffs_dc[:, None]
            * jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * nf
        * window[1:half]
    )
    x_recon = x_recon.at[0].add(jnp.sum(coeffs_dc) * nf * window[0] / 2.0)

    mid_blocks = spectrum_blocks[:, 1:].T * window[None, :]
    shifted_blocks = jnp.concatenate(
        [mid_blocks[:, half:], mid_blocks[:, :half]],
        axis=1,
    )
    mid_indices = (
        (jnp.arange(1, nf)[:, None] - 1) * half + jnp.arange(nt)[None, :]
    ).reshape(-1)
    x_recon = x_recon.at[mid_indices].add(shifted_blocks.reshape(-1))

    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    x_recon = x_recon.at[n_total // 2 - half:n_total // 2].add(
        jnp.sum(
            coeffs_nyq[:, None]
            * jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * nf
        * window[-half:]
    )
    x_recon = x_recon.at[n_total // 2].add(jnp.sum(coeffs_nyq) * nf * window[0] / 2.0)

    return jnp.real(jnp.fft.ifft(x_recon)) / (nf / 2.0)


def _mid_frequency_contribution(
    reconstructed: jnp.ndarray,
    m: int,
    packed: jnp.ndarray,
    freqs: jnp.ndarray,
    dt_block: float,
    a: float,
) -> jnp.ndarray:
    n_idx = jnp.arange(packed.shape[0])
    channel = jnp.real(packed[:, m])
    scaled_freqs = freqs * (2.0 * dt_block)
    phase = jnp.exp(-2j * jnp.pi * n_idx[:, None] * freqs[None, :] * dt_block)
    cnm_nm = _cnm_jax(n_idx, m)
    parity = jnp.where((n_idx * m) % 2 == 0, 1.0, -1.0)
    basis = (
        parity[:, None]
        * (
            jnp.conjugate(cnm_nm)[:, None] * _phi_unit_jax(scaled_freqs + m, a)[None, :]
            + cnm_nm[:, None] * _phi_unit_jax(scaled_freqs - m, a)[None, :]
        )
        * phase
        / jnp.sqrt(2.0)
    )
    return reconstructed + jnp.sum(channel[:, None] * basis, axis=0)


@partial(jit, static_argnames=("nt", "nf"))
def _frequency_wdm_impl(
    packed: jnp.ndarray,
    dt: float,
    a: float,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    freqs = jnp.fft.fftfreq(nt * nf, d=dt)
    dt_block = nf * dt
    scaled_freqs = freqs * (2.0 * dt_block)

    coeffs_dc = jnp.real(packed[:, 0])
    coeffs_nyq = jnp.imag(packed[:, 0])
    n_idx = jnp.arange(nt)
    phase_dc = jnp.exp(-4j * jnp.pi * n_idx[:, None] * freqs[None, :] * dt_block)

    reconstructed = jnp.sum(coeffs_dc[:, None] * phase_dc, axis=0) * _phi_unit_jax(
        scaled_freqs,
        a,
    )
    reconstructed = reconstructed + (
        jnp.sum(coeffs_nyq[:, None] * phase_dc, axis=0)
        * (
            _phi_unit_jax(scaled_freqs + nf, a)
            + _phi_unit_jax(scaled_freqs - nf, a)
        )
    )

    reconstructed = lax.fori_loop(
        1,
        nf,
        lambda m, acc: _mid_frequency_contribution(acc, m, packed, freqs, dt_block, a),
        reconstructed,
    )
    return reconstructed * jnp.sqrt(2.0 * nf)


def forward_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
) -> Any:
    xp = backend.xp
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    n_total = nt * nf
    samples = backend.asarray(data, dtype=xp.complex128)
    if samples.ndim != 1:
        raise ValueError("Input time-domain data must be one-dimensional.")
    if int(samples.shape[0]) != n_total:
        raise ValueError(f"Input length {samples.shape[0]} must equal nt*nf={n_total}.")

    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    return _forward_wdm_impl(samples, window, nt, nf)


def inverse_wdm(coeffs: Any, *, a: float, d: float, dt: float, backend: Backend) -> Any:
    xp = backend.xp
    packed = backend.asarray(coeffs, dtype=xp.complex128)
    if packed.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, nf = (int(dim) for dim in packed.shape)
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    return _inverse_wdm_impl(packed, window, nt, nf)


def frequency_wdm(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> FrequencySeries:
    del d
    xp = backend.xp
    packed = backend.asarray(coeffs, dtype=xp.complex128)
    if packed.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, nf = (int(dim) for dim in packed.shape)
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    reconstructed = _frequency_wdm_impl(packed, dt, a, nt, nf)
    return FrequencySeries(
        reconstructed,
        df=1.0 / (nt * nf * dt),
        backend=backend,
    )
