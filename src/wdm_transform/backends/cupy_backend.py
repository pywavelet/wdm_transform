from __future__ import annotations

from .base import Backend


def load_cupy_backend() -> Backend:
    """Import CuPy and return a :class:`Backend` wrapper for it."""
    try:
        import cupy as cp
        import cupy.fft as cpfft
    except ImportError as exc:
        raise ImportError(
            "CuPy backend requested, but cupy is not installed. Install a compatible cupy package first."
        ) from exc

    return Backend(name="cupy", xp=cp, fft=cpfft)
