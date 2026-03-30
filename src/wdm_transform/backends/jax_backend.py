from __future__ import annotations

from .base import Backend


def load_jax_backend() -> Backend:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import jax.numpy.fft as jfft
    except ImportError as exc:
        raise ImportError(
            "JAX backend requested, but jax is not installed. Install a compatible jax package first."
        ) from exc

    return Backend(name="jax", xp=jnp, fft=jfft)
