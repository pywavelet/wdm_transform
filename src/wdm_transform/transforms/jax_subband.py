"""JAX implementation of low-level WDM sub-band transforms."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit

from ..backends import Backend
from ..windows import phi_window, validate_window_parameter
from ._subband import (
    dt_from_df,
    fourier_span_from_wdm_span,
    validate_fourier_span,
    validate_subband_grid,
    validate_wdm_span,
    wdm_span_from_fourier_span,
)


def _cnm_jax(n: Any, m: Any) -> Any:
    parity = jnp.where((n + m) % 2 == 0, 1.0, -1.0)
    return jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity))


def _extract_fourier_slice(
    data: jnp.ndarray,
    *,
    start: int,
    stop: int,
    kmin: int,
) -> jnp.ndarray:
    indices = jnp.arange(start, stop)
    local = indices - int(kmin)
    valid = (local >= 0) & (local < data.shape[0])
    safe = jnp.clip(local, 0, max(data.shape[0] - 1, 0))
    return jnp.where(valid, data[safe], 0.0j)


def _accumulate_fourier_slice(
    target: jnp.ndarray,
    values: jnp.ndarray,
    *,
    start: int,
    kmin: int,
) -> jnp.ndarray:
    start = int(start)
    stop = start + int(values.shape[0])
    span_start = int(kmin)
    span_stop = span_start + int(target.shape[0])

    overlap_start = max(start, span_start)
    overlap_stop = min(stop, span_stop)
    if overlap_start >= overlap_stop:
        return target
    return target.at[
        overlap_start - span_start:overlap_stop - span_start
    ].add(values[overlap_start - start:overlap_stop - start])


@partial(
    jit,
    static_argnames=("nfreqs_fourier", "kmin", "nfreqs_wdm", "ntimes_wdm", "mmin", "nf_sub_wdm"),
)
def _forward_wdm_subband_impl(
    spectrum: jnp.ndarray,
    window: jnp.ndarray,
    *,
    nfreqs_fourier: int,
    kmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    mmin: int,
    nf_sub_wdm: int,
) -> jnp.ndarray:
    del nfreqs_fourier
    half = int(ntimes_wdm) // 2
    n_total = int(ntimes_wdm) * int(nfreqs_wdm)
    narr = jnp.arange(int(ntimes_wdm))
    coeff_columns = []

    for m in range(int(mmin), int(mmin) + int(nf_sub_wdm)):
        if m == 0:
            block = _extract_fourier_slice(
                spectrum,
                start=1,
                stop=half,
                kmin=kmin,
            ) * window[1:half]
            larr = jnp.arange(1, half)
            x0 = _extract_fourier_slice(spectrum, start=0, stop=1, kmin=kmin)[0]
            coeff = jnp.real(
                jnp.sum(
                    jnp.exp(
                        4j * jnp.pi * larr[None, :] * narr[:, None] / int(ntimes_wdm)
                    )
                    * block[None, :],
                    axis=1,
                )
                + x0 * window[0] / 2.0
            ) / (int(ntimes_wdm) * int(nfreqs_wdm))
            coeff_columns.append(coeff)
            continue

        if m == int(nfreqs_wdm):
            start = n_total // 2 - half
            block = _extract_fourier_slice(
                spectrum,
                start=start,
                stop=n_total // 2,
                kmin=kmin,
            ) * window[-half:]
            larr = jnp.arange(start, n_total // 2)
            xnyq = _extract_fourier_slice(
                spectrum,
                start=n_total // 2,
                stop=n_total // 2 + 1,
                kmin=kmin,
            )[0]
            coeff = jnp.real(
                jnp.sum(
                    jnp.exp(
                        4j * jnp.pi * larr[None, :] * narr[:, None] / int(ntimes_wdm)
                    )
                    * block[None, :],
                    axis=1,
                )
                + xnyq * window[0] / 2.0
            ) / (int(ntimes_wdm) * int(nfreqs_wdm))
            coeff_columns.append(coeff)
            continue

        phase = jnp.conjugate(_cnm_jax(narr, m))
        upper = _extract_fourier_slice(
            spectrum,
            start=m * half,
            stop=(m + 1) * half,
            kmin=kmin,
        )
        lower = _extract_fourier_slice(
            spectrum,
            start=(m - 1) * half,
            stop=m * half,
            kmin=kmin,
        )
        xnm_time = jnp.fft.ifft(jnp.concatenate([upper, lower]) * window)
        coeff_columns.append(
            (jnp.sqrt(2.0) / int(nfreqs_wdm)) * jnp.real(phase * xnm_time)
        )

    return jnp.stack(coeff_columns, axis=1)


@partial(
    jit,
    static_argnames=("nfreqs_fourier", "mmin", "nfreqs_wdm", "ntimes_wdm", "nf_sub_wdm", "kmin", "lendata"),
)
def _inverse_wdm_subband_impl(
    coeffs: jnp.ndarray,
    window: jnp.ndarray,
    *,
    nfreqs_fourier: int,
    mmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    nf_sub_wdm: int,
    kmin: int,
    lendata: int,
) -> jnp.ndarray:
    half = int(ntimes_wdm) // 2
    n_total = int(ntimes_wdm) * int(nfreqs_wdm)
    narr = jnp.arange(int(ntimes_wdm))
    reconstructed = jnp.zeros(int(lendata), dtype=jnp.complex128)

    for local_idx, m in enumerate(range(int(mmin), int(mmin) + int(nf_sub_wdm))):
        if m == 0:
            coeffs_dc = coeffs[:, local_idx]
            larr = jnp.arange(1, half)
            dc_block = (
                jnp.sum(
                    coeffs_dc[:, None]
                    * jnp.exp(
                        -4j * jnp.pi * narr[:, None] * larr[None, :] / int(ntimes_wdm)
                    ),
                    axis=0,
                )
                * int(nfreqs_wdm)
                * window[1:half]
            )
            reconstructed = _accumulate_fourier_slice(
                reconstructed,
                dc_block,
                start=1,
                kmin=kmin,
            )
            reconstructed = _accumulate_fourier_slice(
                reconstructed,
                jnp.asarray([jnp.sum(coeffs_dc) * int(nfreqs_wdm) * window[0] / 2.0]),
                start=0,
                kmin=kmin,
            )
            continue

        if m == int(nfreqs_wdm):
            coeffs_nyq = coeffs[:, local_idx]
            start = n_total // 2 - half
            larr = jnp.arange(start, n_total // 2)
            nyq_block = (
                jnp.sum(
                    coeffs_nyq[:, None]
                    * jnp.exp(
                        -4j * jnp.pi * narr[:, None] * larr[None, :] / int(ntimes_wdm)
                    ),
                    axis=0,
                )
                * int(nfreqs_wdm)
                * window[-half:]
            )
            reconstructed = _accumulate_fourier_slice(
                reconstructed,
                nyq_block,
                start=start,
                kmin=kmin,
            )
            reconstructed = _accumulate_fourier_slice(
                reconstructed,
                jnp.asarray(
                    [jnp.sum(coeffs_nyq) * int(nfreqs_wdm) * window[0] / 2.0]
                ),
                start=n_total // 2,
                kmin=kmin,
            )
            continue

        spectrum_block = jnp.fft.fft(
            _cnm_jax(narr, m) * coeffs[:, local_idx] * int(nfreqs_wdm) / jnp.sqrt(2.0)
        )
        block = spectrum_block * window
        reconstructed = _accumulate_fourier_slice(
            reconstructed,
            jnp.concatenate([block[half:], block[:half]]),
            start=(m - 1) * half,
            kmin=kmin,
        )

    reconstructed = reconstructed / int(nfreqs_wdm)
    if kmin == 0:
        reconstructed = reconstructed.at[0].multiply(2.0)

    nyquist = int(nfreqs_fourier) - 1
    if kmin <= nyquist < kmin + reconstructed.shape[0]:
        reconstructed = reconstructed.at[nyquist - kmin].multiply(2.0)
    return reconstructed


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
) -> tuple[Any, int]:
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_window_parameter(a)

    dt = dt_from_df(df=df, nfreqs_fourier=nfreqs_fourier)
    spectrum = backend.asarray(data, dtype=jnp.complex128)
    if spectrum.ndim != 1:
        raise ValueError("Input Fourier sub-band data must be one-dimensional.")

    validate_fourier_span(
        nfreqs_fourier=nfreqs_fourier,
        kmin=kmin,
        lendata=int(spectrum.shape[0]),
    )

    mmin, nf_sub_wdm = wdm_span_from_fourier_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        kmin=kmin,
        lendata=int(spectrum.shape[0]),
    )
    window = jnp.asarray(
        phi_window(backend, int(ntimes_wdm), int(nfreqs_wdm), dt, a, d),
        dtype=jnp.complex128,
    )
    coeffs = _forward_wdm_subband_impl(
        spectrum,
        window,
        nfreqs_fourier=int(nfreqs_fourier),
        kmin=int(kmin),
        nfreqs_wdm=int(nfreqs_wdm),
        ntimes_wdm=int(ntimes_wdm),
        mmin=int(mmin),
        nf_sub_wdm=int(nf_sub_wdm),
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
) -> tuple[Any, int]:
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_window_parameter(a)

    dt = dt_from_df(df=df, nfreqs_fourier=nfreqs_fourier)
    w = backend.asarray(coeffs, dtype=jnp.float64)
    if w.ndim != 2:
        raise ValueError("WDM sub-band coefficients must be a two-dimensional array.")
    if int(w.shape[0]) != int(ntimes_wdm):
        raise ValueError(
            f"WDM sub-band coeffs must have ntimes_wdm={ntimes_wdm} rows; got {w.shape[0]}."
        )

    nf_sub_wdm = int(w.shape[1])
    validate_wdm_span(
        nfreqs_wdm=nfreqs_wdm,
        mmin=mmin,
        nf_sub_wdm=nf_sub_wdm,
    )

    kmin, lendata = fourier_span_from_wdm_span(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        mmin=mmin,
        nf_sub_wdm=nf_sub_wdm,
    )
    window = jnp.asarray(
        phi_window(backend, int(ntimes_wdm), int(nfreqs_wdm), dt, a, d),
        dtype=jnp.complex128,
    )
    reconstructed = _inverse_wdm_subband_impl(
        w,
        window,
        nfreqs_fourier=int(nfreqs_fourier),
        mmin=int(mmin),
        nfreqs_wdm=int(nfreqs_wdm),
        ntimes_wdm=int(ntimes_wdm),
        nf_sub_wdm=int(nf_sub_wdm),
        kmin=int(kmin),
        lendata=int(lendata),
    )
    return reconstructed, kmin
