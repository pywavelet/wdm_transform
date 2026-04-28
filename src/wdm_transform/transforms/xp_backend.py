"""NumPy / CuPy implementation of the WDM forward, inverse, and frequency-domain transforms.

All three kernels share the same backend-agnostic style: they accept a
``Backend`` object whose ``.xp`` and ``.fft`` attributes supply the array
namespace and FFT routines respectively.  This file is used for both the
``"numpy"`` and ``"cupy"`` backends.

Mathematical background
-----------------------
The WDM (Wilson-Daubechies-Meyer) transform decomposes a discrete signal
x[k] of length N = nt * nf into a 2-D grid of real coefficients W[n, m]
where n indexes time bins (0 ≤ n < nt) and m indexes frequency channels
(0 ≤ m ≤ nf).

The signal is expressed as a superposition of Gabor-like atoms g_{n,m}(f):

    X(f) = Σ_{n,m} W[n,m] · g_{n,m}(f)

Each atom is a frequency-shifted, time-shifted copy of a cosine-tapered
window Φ (see ``windows.phi_unit``), with a phase factor C_{n,m} that
ensures the coefficients are real-valued.

The forward transform projects the signal onto these atoms; the inverse
reconstructs the signal from the coefficients.  Both exploit the fact
that the phi-window is compactly supported in frequency, so each
sub-band only touches a length-``nt`` slice of the full spectrum.

Coefficient layout
------------------
The coefficient matrix ``W`` is returned canonically with shape
``(batch, nt, nf + 1)`` and all real entries. Public entry points still
accept legacy unbatched inputs and normalize them to a singleton leading
batch axis.

* ``W[..., 0]``       — DC edge channel     (m = 0)
* ``W[..., 1:nf]``    — interior channels   (m = 1 … nf−1)
* ``W[..., nf]``      — Nyquist edge channel (m = nf)

There are nt time bins and nf + 1 frequency channels, giving
nt · (nf + 1) total coefficients for a signal of length N = nt · nf.
"""

from __future__ import annotations

from typing import Any

from ..backends import Backend
from ..windows import (
    cnm,
    phi_window,
    validate_transform_shape,
    validate_window_order,
    validate_window_parameter,
)


def _project_to_real_signal_spectrum(spectrum: Any, backend: Backend) -> Any:
    """Return the discrete Fourier coefficients of ``real(ifft(spectrum))``."""
    xp = backend.xp
    n_total = int(spectrum.shape[-1])
    mirror = (-xp.arange(n_total)) % n_total
    return 0.5 * (spectrum + xp.conjugate(spectrum[..., mirror]))


def _compute_wdm_from_spectrum(
    x_fft: Any,
    *,
    nt: int,
    nf: int,
    window: Any,
    backend: Backend,
) -> Any:
    """Project a full Fourier-domain signal onto the WDM basis."""
    xp = backend.xp
    n_total = nt * nf
    coeffs = xp.zeros((nt, nf + 1), dtype=xp.float64)
    half = nt // 2
    narr = xp.arange(nt)

    # Eq. \ref{eq:wn0}: DC edge channel.
    block = x_fft[1:half] * window[1:half]
    larr = xp.arange(1, half)
    coeffs[:, 0] = xp.real(
        xp.sum(
            xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / nt)
            * block[None, :],
            axis=1,
        )
        + x_fft[0] * window[0] / 2.0
    ) * xp.sqrt(2.0)

    # Eq. \ref{eq:wnm_interior}: interior channels.
    for m in range(1, nf):
        phase = xp.conjugate(cnm(backend, narr, m))
        block = xp.concatenate(
            [
                x_fft[m * half:(m + 1) * half],
                x_fft[(m - 1) * half:m * half],
            ]
        )
        xnm_time = backend.fft.ifft(block * window) * nt
        coeffs[:, m] = ((-1) ** (m * narr)) * xp.sqrt(2.0) * xp.real(phase * xnm_time)

    # Eq. \ref{eq:wnNf}: Nyquist edge channel.
    block = x_fft[n_total // 2 - half:n_total // 2] * window[-half:]
    larr = xp.arange(n_total // 2 - half, n_total // 2)
    coeffs[:, nf] = xp.real(
        xp.sum(
            xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / nt)
            * block[None, :],
            axis=1,
        )
        + xp.conjugate(x_fft[n_total // 2]) * window[0] / 2.0
    ) * xp.sqrt(2.0)

    return coeffs


def _reconstruct_spectrum_from_wdm(
    w: Any,
    *,
    nt: int,
    nf: int,
    window: Any,
    backend: Backend,
) -> Any:
    """Reconstruct the Fourier-domain signal represented by WDM coefficients."""
    xp = backend.xp
    n_total = nt * nf
    half = nt // 2
    coeffs_dc = w[:, 0]
    coeffs_nyq = w[:, nf]

    xfr = xp.zeros(n_total // 2 + 1, dtype=xp.complex128)
    narr = xp.arange(nt)

    # Eq. \ref{eq:inverse_fourier}: synthesize the one-sided spectrum.
    larr = xp.arange(half)
    xfr[:half] += (
        xp.sum(
            coeffs_dc[:, None]
            * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * window[:half]
        / xp.sqrt(2.0)
    )

    for m in range(1, nf):
        # FFT trick (manuscript App. \ref{app:forward_derivation}, inverse subsection):
        # the explicit DFT sum Σ_n C_{nm}·w[n,m]·exp(-2πi·n·l/nt) over
        # l ∈ [(m-1)·half, (m+1)·half) factorises as
        #     exp(-2πi·n·(m-1)·half/nt) · exp(-2πi·n·k/nt)
        #         = (-1)^((m-1)·n) · exp(-2πi·n·k/nt),
        # turning the sum into fft(C·w·(-1)^((m-1)n)).
        sign = (-1) ** ((m - 1) * narr)
        spectrum_block = backend.fft.fft(cnm(backend, narr, m) * w[:, m] * sign)
        xfr[(m - 1) * half:(m + 1) * half] += (
            spectrum_block
            * xp.concatenate([window[-half:], window[:half]])
            / xp.sqrt(2.0)
        )

    larr = xp.arange(n_total // 2 - half, n_total // 2)
    xfr[n_total // 2 - half:n_total // 2] += (
        xp.sum(
            coeffs_nyq[:, None]
            * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * window[-half:]
        / xp.sqrt(2.0)
    )
    xfr[n_total // 2] += xp.sum(coeffs_nyq) * window[0] / xp.sqrt(2.0)

    return xp.concatenate([xfr, xp.conjugate(xfr[1:-1][::-1])])


def _compute_wdm_from_spectrum_batch(
    x_fft: Any,
    *,
    nt: int,
    nf: int,
    window: Any,
    backend: Backend,
) -> Any:
    """Batch wrapper around the single-series forward WDM kernel."""
    xp = backend.xp
    coeffs = [
        _compute_wdm_from_spectrum(
            x_fft[idx],
            nt=nt,
            nf=nf,
            window=window,
            backend=backend,
        )
        for idx in range(int(x_fft.shape[0]))
    ]
    return xp.stack(coeffs, axis=0)


def _reconstruct_spectrum_from_wdm_batch(
    w: Any,
    *,
    nt: int,
    nf: int,
    window: Any,
    backend: Backend,
) -> Any:
    """Batch wrapper around the single-series inverse WDM kernel."""
    xp = backend.xp
    spectra = [
        _reconstruct_spectrum_from_wdm(
            w[idx],
            nt=nt,
            nf=nf,
            window=window,
            backend=backend,
        )
        for idx in range(int(w.shape[0]))
    ]
    return xp.stack(spectra, axis=0)


def from_time_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
) -> Any:
    r"""Compute the forward WDM transform of a time-domain signal.

    Given a signal x of length N = nt * nf, this computes the real-valued
    WDM coefficients W[n, m] by projecting the signal's FFT onto
    frequency-localised sub-bands defined by the phi-window.

    Algorithm (three stages)
    ~~~~~~~~~~~~~~~~~~~~~~~~

    **1. DC edge channel (m = 0)**

    The m=0 atom has support only in the lowest ``half = nt/2`` frequency
    bins.  The projection reduces to a DFT-like sum:

        W[n, 0] = √2 Re{ Σ_l exp(4πi·l·n/nt) · X[l] · Φ[l] }

    where l runs from 1 to half−1, plus a half-weight DC term at l=0.

    **2. Interior channels (m = 1 … nf−1)**

    Each interior atom is centred at normalised frequency m and occupies
    a band of width nt spanning indices [(m−1)·half, (m+1)·half).  We
    extract this length-nt block from X(f), multiply by the phi-window,
    and take an inverse FFT to obtain a time-domain "sub-band signal".
    The coefficient is then:

        W[n, m] = √2 (-1)^(n·m) Re{ conj(C_{n,m}) · x_m[n] }

    where ``x_m`` is the unnormalised inverse DFT of the windowed block.

    where C_{n,m} is the phase factor that makes the result real.

    **3. Nyquist edge channel (m = nf)**

    Analogous to the DC channel but centred at the Nyquist frequency
    N/2.  The sum runs over the top ``half`` bins of the spectrum.

    Parameters
    ----------
    data : array, shape (nt * nf,) or (batch, nt * nf)
        Real-valued time-domain samples.
    nt : int
        Number of WDM time bins (must be even).
    nf : int
        Number of WDM frequency channels (must be even).  The output
        will have nf + 1 columns (channels m = 0 … nf).
    a : float
        Window roll-off parameter, in (0, 0.5).  Controls the trade-off
        between time and frequency resolution.  Common choices are
        1/4 (Necula/Cornish) and 1/3.
    d : float
        Reserved window parameter (unused, kept for API symmetry).
    dt : float
        Sampling interval of the input signal.
    backend : Backend
        Array / FFT backend.

    Returns
    -------
    coeffs : array, shape (batch, nt, nf + 1), float64
        Real-valued WDM coefficients.  Column m corresponds to
        frequency channel m.
    """
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    n_total = nt * nf
    samples = backend.asarray(data, dtype=backend.xp.complex128)
    if samples.ndim not in (1, 2):
        raise ValueError("Input time-domain data must be one- or two-dimensional.")
    if samples.ndim == 1:
        samples = samples[None, :]
    if int(samples.shape[-1]) != n_total:
        raise ValueError(f"Input length {samples.shape[-1]} must equal nt*nf={n_total}.")

    x_fft = backend.asarray(backend.fft.fft(samples, axis=-1), dtype=backend.xp.complex128)
    window = backend.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=backend.xp.complex128,
    )
    return _compute_wdm_from_spectrum_batch(
        x_fft,
        nt=nt,
        nf=nf,
        window=window,
        backend=backend,
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
) -> Any:
    """Compute WDM coefficients from full Fourier-domain samples."""
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    n_total = nt * nf
    spectrum = backend.asarray(data, dtype=backend.xp.complex128)
    if spectrum.ndim not in (1, 2):
        raise ValueError("Input frequency-domain data must be one- or two-dimensional.")
    if spectrum.ndim == 1:
        spectrum = spectrum[None, :]
    if int(spectrum.shape[-1]) != n_total:
        raise ValueError(f"Input length {spectrum.shape[-1]} must equal nt*nf={n_total}.")

    projected = backend.asarray(
        _project_to_real_signal_spectrum(spectrum, backend),
        dtype=backend.xp.complex128,
    )
    window = backend.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=backend.xp.complex128,
    )
    return _compute_wdm_from_spectrum_batch(
        projected,
        nt=nt,
        nf=nf,
        window=window,
        backend=backend,
    )


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
) -> Any:
    r"""Reconstruct a time-domain signal from WDM coefficients.

    This inverts ``from_time_to_wdm``.  The reconstruction operates in the
    frequency domain: each sub-band's contribution to the full spectrum
    X(f) is computed and accumulated, then a single inverse FFT recovers
    x(t).

    Algorithm (three stages)
    ~~~~~~~~~~~~~~~~~~~~~~~~

    **1. Form modulated sub-band spectra (interior channels)**

    For each channel m = 1 … nf−1, define:

        X[l] += Σ_n C_{n,m} · W[n,m] · exp(-2πi·n·l/nt) · Φ[l-m·nt/2] / √2

    The sub-band block for channel m occupies indices
    [(m−1)·half, (m+1)·half) of the one-sided spectrum.

    **2. DC edge channel (m = 0)**

    The DC contribution is reconstructed by a direct DFT sum (inverse of
    the projection in the forward transform):

        X[l] += Σ_n W[n,0] · exp(-4πi·n·l/nt) · Φ[l] / √2

    **3. Nyquist edge channel (m = nf)**

    Analogous to DC but targeting the top half of the spectrum.

    Finally, taking the real part of the inverse FFT of the accumulated
    spectrum gives x(t).

    Parameters
    ----------
    coeffs : array, shape (nt, nf + 1) or (batch, nt, nf + 1), float64
        Real-valued WDM coefficients. Outputs are returned canonically as
        ``(batch, nt, nf + 1)``.
    a, d : float
        Window parameters (d is reserved/unused).
    dt : float
        Sampling interval of the original signal.
    backend : Backend
        Array / FFT backend.

    Returns
    -------
    signal : array, shape (batch, nt * nf), float64
        Reconstructed time-domain signal.
    """
    xp = backend.xp
    w = backend.asarray(coeffs, dtype=xp.float64)
    if w.ndim not in (2, 3):
        raise ValueError("WDM coefficients must be a two- or three-dimensional array.")
    if w.ndim == 2:
        w = w[None, :, :]

    nt, ncols = (int(dim) for dim in w.shape[-2:])
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    window = backend.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=xp.complex128,
    )
    spectrum = _reconstruct_spectrum_from_wdm_batch(
        w,
        nt=nt,
        nf=nf,
        window=window,
        backend=backend,
    )
    return xp.real(backend.fft.ifft(spectrum, axis=-1))


def from_wdm_to_freq(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
) -> Any:
    """Reconstruct the Fourier-domain signal represented by WDM coefficients."""
    xp = backend.xp
    w = backend.asarray(coeffs, dtype=xp.float64)
    if w.ndim not in (2, 3):
        raise ValueError("WDM coefficients must be a two- or three-dimensional array.")
    if w.ndim == 2:
        w = w[None, :, :]

    nt, ncols = (int(dim) for dim in w.shape[-2:])
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    window = backend.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=xp.complex128,
    )
    analytic = _reconstruct_spectrum_from_wdm_batch(
        w,
        nt=nt,
        nf=nf,
        window=window,
        backend=backend,
    )
    return _project_to_real_signal_spectrum(analytic, backend)
