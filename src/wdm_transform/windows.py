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


def validate_transform_shape(nt: int, nf: int) -> None:
    """Raise if ``nt`` or ``nf`` are not both even."""
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError("nt and nf must both be even.")


def validate_window_parameter(a: float) -> None:
    """Raise if ``a`` is not in the open interval (0, 0.5)."""
    if not (0.0 < a < 0.5):
        raise ValueError("a must be in (0, 0.5).")


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
    # Assume d=1 for simplicity for now; ask Giorgio if/when this should vary.
    del d
    xp = backend.xp
    b = 1.0 - 2.0 * a
    frequencies = xp.asarray(f, dtype=float)
    abs_f = xp.abs(frequencies)
    tapered = xp.cos((xp.pi / 2.0) * (abs_f - a) / b)
    inner = xp.where(abs_f > a, tapered, 1.0)
    return xp.where(abs_f > a + b, 0.0, inner)


def phi_window(backend: Backend, nt: int, nf: int, dt: float, a: float, d: float) -> Any:
    r"""Build the length-``nt`` phi-window used by the forward and inverse WDM transforms.

    The window is evaluated at the ``nt`` frequencies that tile one sub-band
    of width DF = 1/(2·nf·dt), then scaled by √(2·nf) so that the
    forward/inverse pair is unitary.

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
    xp = backend.xp
    n_total = nt * nf
    df_phi = 1.0 / (2.0 * nf * dt)
    # Use NumPy to compute the frequency grid (small array, always on host),
    # then convert to the target backend.
    fs_full = np.fft.fftfreq(n_total, dt)
    half = nt // 2
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])
    return phi_unit(backend, fs_phi / df_phi, a, d) * xp.sqrt(2.0 * nf)


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
    df = 1.0 / (2.0 * dt_block)
    if m == 0:
        return xp.exp(-4j * xp.pi * n * freqs * dt_block) * phi_unit(backend, freqs / df, a, d)
    if m == nf:
        return (
            phi_unit(backend, freqs / df + m, a, d)
            + phi_unit(backend, freqs / df - m, a, d)
        ) * xp.exp(-4j * xp.pi * n * freqs * dt_block)
    return (
        (-1) ** (m * n)
        * (
            xp.conjugate(cnm(backend, n, m)) * phi_unit(backend, freqs / df + m, a, d)
            + cnm(backend, n, m) * phi_unit(backend, freqs / df - m, a, d)
        )
        * xp.exp(-2j * xp.pi * n * freqs * dt_block)
        / xp.sqrt(2.0)
    )
