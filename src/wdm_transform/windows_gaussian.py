"""Orthogonalized modified Gaussian, in frequency-channel units.

This independently implemented parity/Zak normalization evaluates the canonical
tight Gaussian window of Pearson et al., arXiv:2609.05601, Eqs. (1)--(5).
It is checked against a separate implementation of their coefficient recursion.
Neither a raw Gaussian taper nor per-transform energy renormalization is used.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from ._window_spec import validate_nu


def _zak_window(f: np.ndarray, nu: float, period: int = 128) -> np.ndarray:
    """Normalize both parity sequences by the same frame symbol.

    The finite parity lattice is an orthogonalization truncation, independent
    of the transform's spectral support. The evaluator excludes distant
    coordinates before the FFT lattice can wrap onto its central peak.
    """
    frequency_points = np.asarray(f, dtype=float).ravel()
    result = np.zeros_like(frequency_points)
    retained = np.abs(frequency_points) <= period / 2
    coordinates = frequency_points[retained]
    integer_shifts = np.floor(coordinates + 0.5).astype(int)
    fractional_positions, position_indices = np.unique(
        coordinates - integer_shifts, return_inverse=True
    )
    lattice_indices = np.arange(period) - period // 2
    # Evaluate in bounded blocks even for a dense, nonuniform input grid.
    normalized_parities = np.empty((len(fractional_positions), 2, period), dtype=float)
    for start in range(0, len(fractional_positions), 256):
        position_block = fractional_positions[start : start + 256]
        # Each position has two sequences: g(f + 2r) and g(f + 2r + 1).
        gaussian_positions = position_block[:, None, None] + np.arange(2)[None, :, None]
        gaussian_positions = gaussian_positions + 2 * lattice_indices[None, None, :]
        gaussian = (2 * nu) ** 0.25 * np.exp(-np.pi * nu * gaussian_positions**2)
        parity_fft = np.fft.fft(gaussian, axis=-1)
        # A common denominator preserves cancellation between shifted bands.
        frame_normalization = np.sqrt(np.sum(np.abs(parity_fft) ** 2, axis=1))
        normalized_parities[start : start + len(position_block)] = np.fft.ifft(
            parity_fft / frame_normalization[:, None, :], axis=-1
        ).real
    result[retained] = normalized_parities[
        position_indices,
        integer_shifts % 2,
        (integer_shifts // 2 + period // 2) % period,
    ]
    return result.reshape(np.shape(f))


def gaussian_unit(f: np.ndarray, nu: float = 0.5) -> np.ndarray:
    """Evaluate the real, even unit window at frequencies ``f / DeltaF``.

    Currently validated for ``0.3 <= nu <= 0.8``. Values outside 64 channel
    spacings are numerically truncated, far beyond the tested spectral tails.
    """
    nu = validate_nu(nu)
    frequencies = np.asarray(f, dtype=float)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Gaussian window frequencies must be finite.")
    return _zak_window(frequencies, nu)


@lru_cache(maxsize=8)
def _cached_window(nt: int, nu: float, support: int, dtype: str) -> np.ndarray:
    bin_offsets = np.arange(-support * nt // 2, support * nt // 2)
    window_samples = gaussian_unit(2.0 * bin_offsets / nt, nu) * np.sqrt(2.0 / nt)
    # Immutable backing storage also prevents callers from enabling writes.
    return np.frombuffer(window_samples.astype(dtype).tobytes(), dtype=dtype)


def gaussian_window(
    nt: int,
    nu: float = 0.5,
    support: int = 16,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Return cached host samples on offsets ``[-support*nt/2, support*nt/2)``.

    The physical coordinate is ``2*offset/nt`` and the package scaling is
    ``sqrt(2/nt)``. ``support`` therefore measures channel spacings on each
    side, rather than samples. The returned array is immutable. Backend
    kernels transfer this fixed array; dynamic signal data never visit NumPy.
    """
    nu = validate_nu(nu)
    if isinstance(nt, bool) or not isinstance(nt, Integral) or nt < 2 or nt % 2:
        raise ValueError("nt must be a positive even integer.")
    if isinstance(support, bool) or not isinstance(support, Integral) or support < 1:
        raise ValueError("support must be a positive integer.")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Gaussian window dtype must be float32 or float64.")
    return _cached_window(int(nt), nu, int(support), resolved_dtype.str)
