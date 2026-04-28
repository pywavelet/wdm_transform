"""Window functions and basis atoms for the WDM (Wilson-Daubechies-Meyer) transform.

The WDM transform decomposes a time-domain signal into a time-frequency grid
using a set of Gabor-like atoms defined via a cosine-tapered window (phi).
The window parameter ``a`` controls the roll-off width: the flat passband
spans [-a, a] and the cosine taper extends to [-a-B, a+B] where B = 1-2a.

The ``d`` parameter is reserved for future generalisations of the window but
is currently unused (equivalent to d=1).  It is carried through the API for
forward compatibility.

Key references
--------------
* Necula, Klimenko & Mitselmakher, CQG 29 (2012) 155009
* Cornish, arXiv:2011.09494
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends import Backend


def validate_window_params(a: float, d: float) -> None:
    """Validate both window parameters in one call (convenience for entry points)."""
    validate_window_parameter(a)
    validate_window_order(d)


def validate_transform_shape(nt: int, nf: int) -> None:
    """Raise if ``nt`` or ``nf`` are not both even."""
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError("nt and nf must both be even.")


def validate_window_parameter(a: float) -> None:
    """Raise if ``a`` is not in the open interval (0, 0.5)."""
    if not (0.0 < a < 0.5):
        raise ValueError("a must be in (0, 0.5).")


def validate_window_order(d: float) -> None:
    """Raise if ``d`` is not 1.

    Only the ``d=1`` cosine-tapered Meyer window is implemented (manuscript
    Eq. \\ref{eq:phi_discrete}). The parameter is kept in the API for
    future generalisation but must currently be 1.
    """
    if d != 1:
        raise NotImplementedError(
            f"Only d=1 (cosine-tapered Meyer window) is implemented; got d={d}."
        )


def cnm(backend: Backend, n: Any, m: Any) -> Any:
    r"""Phase factor C_{n,m} used to construct real-valued WDM coefficients.

    .. math::

        C_{n,m} = \exp\!\bigl(\tfrac{i\pi}{4}(1 - (-1)^{n+m})\bigr)

    This equals 1 when n+m is even and exp(iπ/4) when n+m is odd.
    """
    xp = backend.xp
    return xp.exp((1j * xp.pi / 4.0) * (1.0 - (-1) ** (n + m)))


def phi_unit(backend: Backend, f: Any, a: float, d: float) -> Any:
    r"""Unitary phi-window in the frequency domain.

    The window is a cosine-tapered rectangle:

    * |f| ≤ a        → 1
    * a < |f| ≤ a+B  → cos(π/2 · (|f|-a)/B)   where B = 1-2a
    * |f| > a+B      → 0

    Parameters
    ----------
    backend : Backend
        Array backend to use for computation.
    f : array-like
        Normalised frequencies at which to evaluate the window.
    a : float
        Half-width of the flat passband (must be in (0, 0.5)).
    d : float
        Reserved parameter (unused, kept for API symmetry).
    """
    validate_window_order(d)
    xp = backend.xp
    b = 1.0 - 2.0 * a
    frequencies = xp.asarray(f, dtype=float)
    abs_f = xp.abs(frequencies)
    tapered = xp.cos((xp.pi / 2.0) * (abs_f - a) / b)
    inner = xp.where(abs_f > a, tapered, 1.0)
    return xp.where(abs_f > a + b, 0.0, inner)


def phi_window(backend: Backend, nt: int, nf: int, dt: float, a: float, d: float) -> Any:
    r"""Build the length-``nt`` phi-window used by the forward and inverse WDM transforms.

    The window follows manuscript Eq. ``\ref{eq:phi_discrete}``: it is
    evaluated on the length-``nt`` FFT frequency-index grid and scaled by
    ``sqrt(2 / nt)``.

    Parameters
    ----------
    backend : Backend
        Array backend to use for computation.
    nt, nf : int
        Number of time bins and frequency channels.
    dt : float
        Sampling interval of the original signal.
    a : float
        Window roll-off parameter.
    d : float
        Reserved (unused).
    """
    del nf, dt
    xp = backend.xp
    # Use NumPy to compute the frequency grid (small array, always on host),
    # then convert to the target backend.
    l_values = np.fft.fftfreq(nt) * nt
    return phi_unit(backend, 2.0 * l_values / nt, a, d) * xp.sqrt(2.0 / nt)


def gnmf(
    backend: Backend,
    n: int,
    m: int,
    freqs: Any,
    dt_block: float,
    nf: int,
    a: float,
    d: float,
) -> Any:
    r"""Gabor atom g_{n,m}(f) in the frequency domain.

    Each atom is a modulated, shifted copy of the phi-window.  The three
    cases correspond to the DC edge channel (m=0), the Nyquist edge channel
    (m=nf), and the interior channels (0 < m < nf).

    Parameters
    ----------
    backend : Backend
        Array backend.
    n : int
        Time-shift index.
    m : int
        Frequency-channel index (0 ≤ m ≤ nf).
    freqs : array
        Frequency grid of the full signal.
    dt_block : float
        Duration of one WDM time bin (= nf · dt).
    nf : int
        Total number of interior frequency channels.
    a, d : float
        Window parameters (d is reserved/unused).
    """
    xp = backend.xp
    nt = int(xp.asarray(freqs).shape[-1]) // int(nf)
    df = 1.0 / (2.0 * dt_block)
    scale = xp.sqrt(2.0 / nt)

    def _phi(u: Any) -> Any:
        return phi_unit(backend, u, a, d) * scale

    if m == 0:
        return (
            xp.exp(-4j * xp.pi * n * freqs * dt_block)
            * _phi(freqs / df)
            / xp.sqrt(2.0)
        )
    if m == nf:
        return (
            (_phi(freqs / df + m) + _phi(freqs / df - m))
            * xp.exp(-4j * xp.pi * n * freqs * dt_block)
            / xp.sqrt(2.0)
        )
    return (
        (
            xp.conjugate(cnm(backend, n, m)) * _phi(freqs / df + m)
            + cnm(backend, n, m) * _phi(freqs / df - m)
        )
        * xp.exp(-2j * xp.pi * n * freqs * dt_block)
        / xp.sqrt(2.0)
    )
