"""NumPy / CuPy implementation of low-level WDM sub-band transforms.

This module implements the additive "sub-band" API for users who already
have data on a compact slice of the full one-sided Fourier grid and do not
want to materialize the entire spectrum.

Grid conventions
----------------
The input Fourier data lives on the full non-negative FFT grid of length
``nfreqs_fourier = N / 2 + 1`` where ``N = ntimes_wdm * nfreqs_wdm`` is the
length of the corresponding full time series.  A compact input array
``data`` of length ``lendata`` is interpreted as occupying indices
``kmin .. kmin + lendata - 1`` on that global one-sided grid and being zero
everywhere else.

The output WDM coefficients are also compact: only the frequency channels
that can receive non-zero contributions from the supplied Fourier span are
returned.  If ``half = ntimes_wdm / 2``, then:

* the DC channel ``m = 0`` touches Fourier bins ``[0, half)``
* an interior channel ``m`` touches bins ``[(m-1) * half, (m+1) * half)``
* the Nyquist channel ``m = nfreqs_wdm`` touches bins
  ``[nfreqs_wdm * half - half, nfreqs_wdm * half]``

This locality is what makes the sub-band transform practical: a compact
Fourier span only overlaps a compact WDM span.

Forward transform
-----------------
``forward_wdm_subband`` reuses the same per-channel formulas as the full
transform, but replaces full-spectrum indexing with local gathers from the
compact one-sided input plus implicit zero fill outside the supplied span.
The returned coefficient block therefore matches the corresponding slice of
the full WDM transform under the assumption that the omitted Fourier bins are
exactly zero.

Inverse transform
-----------------
``inverse_wdm_subband`` performs the opposite operation.  It reconstructs
only the minimal touched one-sided Fourier span for the supplied compact WDM
block, again using the same DC / interior / Nyquist formulas as the full
inverse.  The result is returned as a compact complex array together with the
starting Fourier index ``kmin``.

Normalization
-------------
The compact inverse returns values on the package's one-sided Fourier
convention.  After accumulating channel contributions, the spectrum is
divided by ``nfreqs_wdm`` and the DC / Nyquist bins receive the usual extra
factor of two when present in the compact output.
"""

from __future__ import annotations

from typing import Any

from ..backends import Backend
from ._subband import (
    dt_from_df,
    fourier_span_from_wdm_span,
    validate_fourier_span,
    validate_subband_grid,
    validate_wdm_span,
    wdm_span_from_fourier_span,
)
from ..windows import cnm, phi_window, validate_window_parameter


def _extract_fourier_slice(
    data: Any,
    *,
    start: int,
    stop: int,
    kmin: int,
    backend: Backend,
) -> Any:
    """Return ``data[start:stop]`` on the full one-sided grid, zero-filling outside the input span."""
    xp = backend.xp
    start = int(start)
    stop = int(stop)
    span_start = int(kmin)
    span_stop = span_start + int(data.shape[0])

    block = xp.zeros(stop - start, dtype=xp.complex128)
    overlap_start = max(start, span_start)
    overlap_stop = min(stop, span_stop)
    if overlap_start < overlap_stop:
        block[overlap_start - start:overlap_stop - start] = data[
            overlap_start - span_start:overlap_stop - span_start
        ]
    return block


def _accumulate_fourier_slice(
    target: Any,
    values: Any,
    *,
    start: int,
    kmin: int,
) -> None:
    """Accumulate a full-grid slice onto the compact output span."""
    start = int(start)
    stop = start + int(values.shape[0])
    span_start = int(kmin)
    span_stop = span_start + int(target.shape[0])

    overlap_start = max(start, span_start)
    overlap_stop = min(stop, span_stop)
    if overlap_start < overlap_stop:
        target[overlap_start - span_start:overlap_stop - span_start] += values[
            overlap_start - start:overlap_stop - start
        ]


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
    """Forward WDM transform from a compact one-sided Fourier sub-band."""
    xp = backend.xp
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_window_parameter(a)

    dt = dt_from_df(df=df, nfreqs_fourier=nfreqs_fourier)
    spectrum = backend.asarray(data, dtype=xp.complex128)
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

    n_total = int(ntimes_wdm) * int(nfreqs_wdm)
    half = int(ntimes_wdm) // 2
    window = backend.asarray(
        phi_window(backend, int(ntimes_wdm), int(nfreqs_wdm), dt, a, d),
        dtype=xp.complex128,
    )

    coeffs = xp.zeros((int(ntimes_wdm), nf_sub_wdm), dtype=xp.float64)
    narr = xp.arange(int(ntimes_wdm))

    for local_idx, m in enumerate(range(mmin, mmin + nf_sub_wdm)):
        if m == 0:
            block = _extract_fourier_slice(
                spectrum,
                start=1,
                stop=half,
                kmin=kmin,
                backend=backend,
            ) * window[1:half]
            larr = xp.arange(1, half)
            x0 = _extract_fourier_slice(
                spectrum,
                start=0,
                stop=1,
                kmin=kmin,
                backend=backend,
            )[0]
            coeffs[:, local_idx] = xp.real(
                xp.sum(
                    xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / int(ntimes_wdm))
                    * block[None, :],
                    axis=1,
                )
                + x0 * window[0] / 2.0
            ) / (int(ntimes_wdm) * int(nfreqs_wdm))
            continue

        if m == int(nfreqs_wdm):
            start = n_total // 2 - half
            block = _extract_fourier_slice(
                spectrum,
                start=start,
                stop=n_total // 2,
                kmin=kmin,
                backend=backend,
            ) * window[-half:]
            larr = xp.arange(start, n_total // 2)
            xnyq = _extract_fourier_slice(
                spectrum,
                start=n_total // 2,
                stop=n_total // 2 + 1,
                kmin=kmin,
                backend=backend,
            )[0]
            coeffs[:, local_idx] = xp.real(
                xp.sum(
                    xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / int(ntimes_wdm))
                    * block[None, :],
                    axis=1,
                )
                + xnyq * window[0] / 2.0
            ) / (int(ntimes_wdm) * int(nfreqs_wdm))
            continue

        phase = xp.conjugate(cnm(backend, narr, m))
        upper = _extract_fourier_slice(
            spectrum,
            start=m * half,
            stop=(m + 1) * half,
            kmin=kmin,
            backend=backend,
        )
        lower = _extract_fourier_slice(
            spectrum,
            start=(m - 1) * half,
            stop=m * half,
            kmin=kmin,
            backend=backend,
        )
        block = xp.concatenate([upper, lower])
        xnm_time = backend.fft.ifft(block * window)
        coeffs[:, local_idx] = (xp.sqrt(2.0) / int(nfreqs_wdm)) * xp.real(
            phase * xnm_time
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
    """Inverse WDM transform onto a compact one-sided Fourier sub-band."""
    xp = backend.xp
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_window_parameter(a)

    dt = dt_from_df(df=df, nfreqs_fourier=nfreqs_fourier)
    w = backend.asarray(coeffs, dtype=xp.float64)
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

    half = int(ntimes_wdm) // 2
    n_total = int(ntimes_wdm) * int(nfreqs_wdm)
    window = backend.asarray(
        phi_window(backend, int(ntimes_wdm), int(nfreqs_wdm), dt, a, d),
        dtype=xp.complex128,
    )
    reconstructed = xp.zeros(lendata, dtype=xp.complex128)
    narr = xp.arange(int(ntimes_wdm))

    for local_idx, m in enumerate(range(int(mmin), int(mmin) + nf_sub_wdm)):
        if m == 0:
            coeffs_dc = w[:, local_idx]
            larr = xp.arange(1, half)
            dc_block = (
                xp.sum(
                    coeffs_dc[:, None]
                    * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / int(ntimes_wdm)),
                    axis=0,
                )
                * int(nfreqs_wdm)
                * window[1:half]
            )
            _accumulate_fourier_slice(reconstructed, dc_block, start=1, kmin=kmin)
            _accumulate_fourier_slice(
                reconstructed,
                xp.asarray([xp.sum(coeffs_dc) * int(nfreqs_wdm) * window[0] / 2.0]),
                start=0,
                kmin=kmin,
            )
            continue

        if m == int(nfreqs_wdm):
            coeffs_nyq = w[:, local_idx]
            start = n_total // 2 - half
            larr = xp.arange(start, n_total // 2)
            nyq_block = (
                xp.sum(
                    coeffs_nyq[:, None]
                    * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / int(ntimes_wdm)),
                    axis=0,
                )
                * int(nfreqs_wdm)
                * window[-half:]
            )
            _accumulate_fourier_slice(reconstructed, nyq_block, start=start, kmin=kmin)
            _accumulate_fourier_slice(
                reconstructed,
                xp.asarray([xp.sum(coeffs_nyq) * int(nfreqs_wdm) * window[0] / 2.0]),
                start=n_total // 2,
                kmin=kmin,
            )
            continue

        spectrum_block = backend.fft.fft(
            cnm(backend, narr, m) * w[:, local_idx] * int(nfreqs_wdm) / xp.sqrt(2.0)
        )
        block = spectrum_block * window
        shifted = xp.concatenate([block[half:], block[:half]])
        _accumulate_fourier_slice(
            reconstructed,
            shifted,
            start=(m - 1) * half,
            kmin=kmin,
        )

    reconstructed = reconstructed / int(nfreqs_wdm)
    if kmin == 0:
        reconstructed[0] *= 2.0

    nyquist = int(nfreqs_fourier) - 1
    if kmin <= nyquist < kmin + lendata:
        reconstructed[nyquist - kmin] *= 2.0

    return reconstructed, kmin
