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

Packing convention
------------------
The coefficient matrix ``W`` has shape ``(nt, nf)`` where:

* ``W[:, 0].real``  — DC edge channel   (m = 0)
* ``W[:, 0].imag``  — Nyquist edge channel (m = nf)
* ``W[:, 1:]``      — interior channels  (m = 1 … nf−1), real-valued

The DC and Nyquist channels each carry only real information (their
atoms are purely real in the time domain).  Packing them into the real
and imaginary parts of a single complex column saves one column of
storage without loss.
"""

from __future__ import annotations

from typing import Any

from ..backends import Backend
from ..datatypes.series import FrequencySeries
from ..windows import cnm, gnmf, phi_unit, phi_window, validate_transform_shape, validate_window_parameter


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
    r"""Compute the forward WDM transform of a time-domain signal.

    Given a signal x of length N = nt * nf, this computes the real-valued
    WDM coefficients W[n, m] by projecting the signal's FFT onto
    frequency-localised sub-bands defined by the phi-window.

    Algorithm (three stages)
    ~~~~~~~~~~~~~~~~~~~~~~~~

    **1. DC edge channel (m = 0)**

    The m=0 atom has support only in the lowest ``half = nt/2`` frequency
    bins.  The projection reduces to a DFT-like sum:

        W_dc[n] = Re{ Σ_l  exp(4πi·l·n/nt) · X[l] · Φ[l] } / (nt·nf)

    where l runs from 1 to half−1, plus a half-weight DC term at l=0.

    **2. Interior channels (m = 1 … nf−1)**

    Each interior atom is centred at normalised frequency m and occupies
    a band of width nt spanning indices [(m−1)·half, (m+1)·half).  We
    extract this length-nt block from X(f), multiply by the phi-window,
    and take an inverse FFT to obtain a time-domain "sub-band signal".
    The coefficient is then:

        W[n, m] = (√2 / nf) · Re{ conj(C_{n,m}) · IFFT(block · Φ)[n] }

    where C_{n,m} is the phase factor that makes the result real.

    **3. Nyquist edge channel (m = nf)**

    Analogous to the DC channel but centred at the Nyquist frequency
    N/2.  The sum runs over the top ``half`` bins of the spectrum.

    Finally, both edge channels are packed into column 0:
    ``W[:, 0] = W_dc + i·W_nyq``.

    Parameters
    ----------
    data : array, shape (nt * nf,)
        Real-valued time-domain samples.
    nt : int
        Number of WDM time bins (must be even).
    nf : int
        Number of WDM frequency channels (must be even).
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

    # Full-length FFT of the input signal
    x_fft = backend.fft.fft(samples)

    # Build the length-nt phi-window, scaled by √(2·nf) for unitarity
    window = backend.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=xp.complex128)

    coeffs = xp.zeros((nt, nf), dtype=xp.complex128)
    half = nt // 2
    narr = xp.arange(nt)

    # --- DC edge channel (m = 0) ---
    # Extract the lowest half frequency bins, window them, and project
    # onto the DFT basis exp(4πi·l·n/nt).
    block = x_fft[1:half] * window[1:half]
    larr = xp.arange(1, half)
    coeffs_dc = xp.real(
        xp.sum(xp.exp(4j * xp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :], axis=1)
        + x_fft[0] * window[0] / 2.0
    ) / (nt * nf)

    # --- Interior sub-bands (m = 1 … nf−1) ---
    # For each channel m, the relevant spectrum slice is
    # [m·half, (m+1)·half) ∪ [(m-1)·half, m·half), i.e. a length-nt
    # block centred on frequency index m·half.  We concatenate upper
    # then lower halves, multiply by the phi-window, IFFT, and extract
    # the real part after phase correction by conj(C_{n,m}).
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
    # Same structure as DC but centred at N/2.
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
    r"""Reconstruct a time-domain signal from packed WDM coefficients.

    This inverts ``forward_wdm``.  The reconstruction operates in the
    frequency domain: each sub-band's contribution to the full spectrum
    X(f) is computed and accumulated, then a single inverse FFT recovers
    x(t).

    Algorithm (three stages)
    ~~~~~~~~~~~~~~~~~~~~~~~~

    **1. Form modulated sub-band spectra (interior channels)**

    For each channel m = 1 … nf−1, define:

        y[n, m] = C_{n,m} · W[n, m] · nf / √2

    where C_{n,m} is the phase factor.  Taking the FFT of y[:, m] along
    the time axis n gives Y[:, m], which when multiplied by the
    phi-window and placed at the correct offset in X(f), reconstructs
    that sub-band's contribution.

    The sub-band block for channel m occupies indices
    [(m−1)·half, (m+1)·half) of the full spectrum, with the upper and
    lower halves swapped (matching the forward transform's concatenation
    order).

    **2. DC edge channel (m = 0)**

    The DC contribution is reconstructed by a direct DFT sum (inverse of
    the projection in the forward transform):

        X[l] += Σ_n  W_dc[n] · exp(-4πi·n·l/nt) · nf · Φ[l]

    with a half-weight term at l = 0.

    **3. Nyquist edge channel (m = nf)**

    Analogous to DC but targeting the top half of the spectrum.

    Finally, an inverse FFT of the accumulated X(f) gives x(t), scaled
    by 2/nf for normalisation.

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

    # Unpack the edge channels from column 0
    coeffs_dc = xp.real(packed[:, 0])
    coeffs_nyq = xp.imag(packed[:, 0])

    # Build modulated coefficients y[n, m] = C_{n,m} · Re(W[n, m]) · nf / √2
    # and FFT along the time axis to get sub-band spectra
    n_idx = xp.arange(nt)[:, None]
    m_idx = xp.arange(nf)[None, :]
    ylm = cnm(backend, n_idx, m_idx) * xp.real(packed) * nf / xp.sqrt(2.0)
    spectrum_blocks = backend.fft.fft(ylm, axis=0)

    x_recon = xp.zeros(n_total, dtype=xp.complex128)
    narr = xp.arange(nt)

    # --- DC edge: direct DFT synthesis into bins [1, half) ---
    larr = xp.arange(1, half)
    x_recon[1:half] += (
        xp.sum(coeffs_dc[:, None] * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt), axis=0)
        * nf
        * window[1:half]
    )
    x_recon[0] += xp.sum(coeffs_dc) * nf * window[0] / 2.0

    # --- Interior sub-bands: place each windowed block at its offset ---
    # spectrum_blocks[:, m] · Φ is split into [upper, lower] halves and
    # written to indices [(m-1)·half, (m+1)·half).
    for m in range(1, nf):
        block = spectrum_blocks[:, m] * window
        x_recon[(m - 1) * half:(m + 1) * half] += xp.concatenate([block[half:], block[:half]])

    # --- Nyquist edge: direct DFT synthesis near N/2 ---
    larr = xp.arange(n_total // 2 - half, n_total // 2)
    x_recon[n_total // 2 - half:n_total // 2] += (
        xp.sum(coeffs_nyq[:, None] * xp.exp(-4j * xp.pi * narr[:, None] * larr[None, :] / nt), axis=0)
        * nf
        * window[-half:]
    )
    x_recon[n_total // 2] += xp.sum(coeffs_nyq) * nf * window[0] / 2.0

    # Inverse FFT and normalise: factor 2/nf completes the unitarity relation
    return xp.real(backend.fft.ifft(x_recon)) / (nf / 2.0)


def frequency_wdm(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> FrequencySeries:
    r"""Reconstruct the frequency-domain signal directly from WDM coefficients.

    Rather than going through the time domain, this evaluates the Gabor
    atom expansion in the frequency domain:

        X(f) = √(2·nf) · Σ_{n,m}  W[n, m] · g_{n,m}(f)

    where g_{n,m}(f) are the WDM basis atoms (see ``windows.gnmf``).

    The implementation precomputes three atom tables:

    * ``g0_tab[n, k]``    — DC atoms     (m = 0),   shape (nt, N)
    * ``gN_tab[n, k]``    — Nyquist atoms (m = nf),  shape (nt, N)
    * ``gmid_tab[n, m, k]`` — interior atoms (m = 1…nf−1), shape (nt, nf−1, N)

    and contracts them with the coefficient matrix via ``einsum``:

        X[k] = Σ_n W_dc[n] · g0[n,k]
             + Σ_n W_nyq[n] · gN[n,k]
             + Σ_{n,m} W_mid[n,m] · gmid[n,m,k]

    This is O(nt · nf · N) in memory but avoids Python loops over
    (n, m) pairs, giving a large speed-up for moderate problem sizes.

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
        Reconstructed frequency-domain signal, with ``df = 1 / (N · dt)``.
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
