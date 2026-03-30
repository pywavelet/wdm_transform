"""NumPy / CuPy implementation of the WDM forward, inverse, and frequency-domain transforms.

All three kernels share the same backend-agnostic style: they accept a
``Backend`` object whose ``.xp`` and ``.fft`` attributes supply the array
namespace and FFT routines respectively.  This file is used for both the
``"numpy"`` and ``"cupy"`` backends.

Packing convention
------------------
The WDM coefficient matrix ``W`` has shape ``(nt, nf)`` where:

* ``W[:, 0].real``  — DC edge channel   (m = 0)
* ``W[:, 0].imag``  — Nyquist edge channel (m = nf)
* ``W[:, 1:]``      — interior channels  (m = 1 … nf−1), real-valued

This complex packing halves the storage cost of the two edge channels
which each carry only real information.
"""

from __future__ import annotations

from typing import Any

from ..backends import Backend
from ..datatypes.series import FrequencySeries
from ..windows import cnm, phi_unit, phi_window, validate_transform_shape, validate_window_parameter


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
    """Compute the forward WDM transform of a time-domain signal.

    Parameters
    ----------
    data : array, shape (nt * nf,)
        Real-valued time-domain samples.
    nt : int
        Number of WDM time bins (must be even).
    nf : int
        Number of WDM frequency channels (must be even).
    a : float
        Window roll-off parameter, in (0, 0.5).
    d : float
        Reserved window parameter (unused).
    dt : float
        Sampling interval of the input signal.
    backend : Backend
        Array / FFT backend.

    Returns
    -------
    coeffs : array, shape (nt, nf), complex128
        Packed WDM coefficients (see module docstring for packing convention).
    """
    xp = backend.xp
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    n_total = nt * nf
    samples = backend.asarray(data, dtype=xp.complex128)
    if samples.ndim != 1:
        raise ValueError("Input time-domain data must be one-dimensional.")
    if int(samples.shape[0]) != n_total:
        raise ValueError(f"Input length {samples.shape[0]} must equal nt*nf={n_total}.")

    x_fft = backend.fft.fft(samples)
    window = backend.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=xp.complex128)

    coeffs = xp.zeros((nt, nf), dtype=xp.complex128)
    half = nt // 2
    narr = xp.arange(nt)

    # --- DC edge channel (m = 0) ---
    block = x_fft[1:half] * window[1:half]
    larr = xp.arange(1, half)
    coeffs_dc = xp.real(
        xp.sum(xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :], axis=1)
        + x_fft[0] * window[0] / 2.0
    ) / (nt * nf)

    # --- Interior sub-bands (m = 1 … nf−1) ---
    for m in range(1, nf):
        phase = xp.conjugate(cnm(backend, narr, m))
        block = xp.concatenate(
            [
                x_fft[m * half:(m + 1) * half],
                x_fft[(m - 1) * half:m * half],
            ]
        )
        xnm_time = backend.fft.ifft(block * window)
        coeffs[:, m] = (xp.sqrt(2.0) / nf) * xp.real(phase * xnm_time)

    # --- Nyquist edge channel (m = nf) ---
    block = x_fft[n_total // 2 - half:n_total // 2] * window[-half:]
    larr = xp.arange(n_total // 2 - half, n_total // 2)
    coeffs_nyq = xp.real(
        xp.sum(xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :], axis=1)
        + x_fft[n_total // 2] * window[0] / 2.0
    ) / (nt * nf)

    # Pack DC (real) and Nyquist (imag) into column 0
    coeffs[:, 0] = coeffs_dc + 1j * coeffs_nyq
    return coeffs


def inverse_wdm(coeffs: Any, *, a: float, d: float, dt: float, backend: Backend) -> Any:
    """Compute the inverse WDM transform, recovering a time-domain signal.

    Parameters
    ----------
    coeffs : array, shape (nt, nf), complex128
        Packed WDM coefficients.
    a, d : float
        Window parameters (d is reserved/unused).
    dt : float
        Sampling interval of the original signal.
    backend : Backend
        Array / FFT backend.

    Returns
    -------
    signal : array, shape (nt * nf,), float64
        Reconstructed time-domain signal.
    """
    xp = backend.xp
    packed = backend.asarray(coeffs, dtype=xp.complex128)
    if packed.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, nf = (int(dim) for dim in packed.shape)
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    n_total = nt * nf
    half = nt // 2
    window = backend.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=xp.complex128)

    coeffs_dc = xp.real(packed[:, 0])
    coeffs_nyq = xp.imag(packed[:, 0])

    n_idx = xp.arange(nt)[:, None]
    m_idx = xp.arange(nf)[None, :]
    ylm = cnm(backend, n_idx, m_idx) * xp.real(packed) * nf / xp.sqrt(2.0)
    spectrum_blocks = backend.fft.fft(ylm, axis=0)

    x_recon = xp.zeros(n_total, dtype=xp.complex128)
    narr = xp.arange(nt)

    # --- DC edge ---
    larr = xp.arange(1, half)
    x_recon[1:half] += (
        xp.sum(coeffs_dc[:, None] * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt), axis=0)
        * nf
        * window[1:half]
    )
    x_recon[0] += xp.sum(coeffs_dc) * nf * window[0] / 2.0

    # --- Interior sub-bands ---
    for m in range(1, nf):
        block = spectrum_blocks[:, m] * window
        x_recon[(m - 1) * half:(m + 1) * half] += xp.concatenate([block[half:], block[:half]])

    # --- Nyquist edge ---
    larr = xp.arange(n_total // 2 - half, n_total // 2)
    x_recon[n_total // 2 - half:n_total // 2] += (
        xp.sum(coeffs_nyq[:, None] * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt), axis=0)
        * nf
        * window[-half:]
    )
    x_recon[n_total // 2] += xp.sum(coeffs_nyq) * nf * window[0] / 2.0

    return xp.real(backend.fft.ifft(x_recon)) / (nf / 2.0)


def frequency_wdm(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> FrequencySeries:
    """Reconstruct the frequency-domain representation from WDM coefficients.

    This is equivalent to summing the Gabor atoms g_{n,m}(f) weighted by
    their coefficients.  The implementation precomputes atom tables for
    the DC, Nyquist, and interior channels and contracts them with the
    coefficient matrix via ``einsum``, avoiding slow Python loops.

    Parameters
    ----------
    coeffs : array, shape (nt, nf), complex128
        Packed WDM coefficients.
    dt : float
        Sampling interval of the original signal.
    a, d : float
        Window parameters (d is reserved/unused).
    backend : Backend
        Array / FFT backend.

    Returns
    -------
    FrequencySeries
        Reconstructed frequency-domain signal.
    """
    xp = backend.xp
    packed = backend.asarray(coeffs, dtype=xp.complex128)
    if packed.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, nf = (int(dim) for dim in packed.shape)
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    n_total = nt * nf
    freqs = backend.fft.fftfreq(n_total, d=dt)
    dt_block = nf * dt

    # Precompute atom tables: g0[n, k], gN[n, k], gmid[n, m-1, k]
    g0_tab = xp.stack([gnmf(backend, n, 0, freqs, dt_block, nf, a, d) for n in range(nt)])
    gN_tab = xp.stack([gnmf(backend, n, nf, freqs, dt_block, nf, a, d) for n in range(nt)])
    gmid_tab = xp.stack([
        xp.stack([gnmf(backend, n, m, freqs, dt_block, nf, a, d) for m in range(1, nf)])
        for n in range(nt)
    ])

    # Contract: sum over time-shift index n (and channel index m for interior)
    reconstructed = (
        xp.einsum("n,nk->k", xp.real(packed[:, 0]), g0_tab)
        + xp.einsum("n,nk->k", xp.imag(packed[:, 0]), gN_tab)
        + xp.einsum("nm,nmk->k", xp.real(packed[:, 1:]), gmid_tab)
    )

    return FrequencySeries(
        reconstructed * xp.sqrt(2.0 * nf),
        df=1.0 / (n_total * dt),
        backend=backend,
    )


# Re-import gnmf here so frequency_wdm can use it without a circular dep issue
from ..windows import gnmf  # noqa: E402
