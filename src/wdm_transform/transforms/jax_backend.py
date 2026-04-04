"""JAX implementation of the WDM forward, inverse, and frequency-domain transforms.

This module mirrors ``xp_backend.py`` but uses JAX-specific features:

* ``@jit`` compilation with static shape arguments for performance
* Fully vectorised interior-channel computation (all m = 1…nf−1 processed
  as a batch rather than in a Python loop)
* Immutable array updates via ``jnp.ndarray.at[...].add()`` (JAX arrays
  are immutable, so in-place ``+=`` is not available)

The mathematical algorithm is identical to the NumPy/CuPy backend — see
``xp_backend.py`` for the detailed derivation.  Below we note only the
JAX-specific implementation choices.

Coefficient layout
------------------
Same as ``xp_backend``: shape ``(nt, nf + 1)`` with all-real entries.

* ``W[:, 0]``       — DC edge channel     (m = 0)
* ``W[:, 1:nf]``    — interior channels   (m = 1 … nf−1)
* ``W[:, nf]``      — Nyquist edge channel (m = nf)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit

from ..backends import Backend
from ..windows import phi_window, validate_transform_shape, validate_window_parameter


def _cnm_jax(n: Any, m: Any) -> Any:
    r"""Phase factor C_{n,m} (JAX-traceable version).

    Uses ``jnp.where`` on the parity of (n+m) instead of ``(-1)**(n+m)``
    to remain compatible with JAX's tracing (integer exponentiation of
    traced values is not supported).

    .. math::

        C_{n,m} = \exp\!\bigl(\tfrac{i\pi}{4}(1 - (-1)^{n+m})\bigr)
    """
    parity = jnp.where((n + m) % 2 == 0, 1.0, -1.0)
    return jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity))


def _project_to_real_signal_spectrum_impl(spectrum: jnp.ndarray) -> jnp.ndarray:
    """Return the discrete Fourier coefficients of ``real(ifft(spectrum))``."""
    n_total = spectrum.shape[0]
    mirror = (-jnp.arange(n_total)) % n_total
    return 0.5 * (spectrum + jnp.conjugate(spectrum[mirror]))


@partial(jit, static_argnames=("nt", "nf"))
def _from_spectrum_to_wdm_impl(
    x_fft: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    r"""JIT-compiled forward WDM kernel.

    Unlike the NumPy backend which loops over m in Python, this
    implementation batches all interior channels into a single
    vectorised operation:

    1. Build an index array of shape (nf−1, nt) that gathers the
       correct spectral slices for all interior channels at once.
    2. Batch-multiply by the phi-window and batch-IFFT.
    3. Apply the phase correction C_{n,m} via a vectorised outer product.

    Returns shape ``(nt, nf + 1)`` with all-real coefficients.

    Parameters
    ----------
    x_fft : jnp.ndarray, shape (N,)
        Complex-valued Fourier-domain signal.
    window : jnp.ndarray, shape (nt,)
        Precomputed phi-window of length nt.
    nt, nf : int
        Static shape parameters (used by JIT for specialisation).
    """
    n_total = nt * nf
    half = nt // 2
    narr = jnp.arange(nt)
    sqrt2 = jnp.sqrt(2.0)

    # --- DC edge channel (m = 0) ---
    block = x_fft[1:half] * window[1:half]
    larr = jnp.arange(1, half)
    coeffs_dc = jnp.real(
        jnp.sum(
            jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[0] * window[0] / 2.0
    ) / (nt * nf)

    # --- Interior channels (m = 1…nf−1), fully vectorised ---
    mid_m = jnp.arange(1, nf)
    upper = mid_m[:, None] * half + jnp.arange(half)[None, :]   # shape (nf-1, half)
    lower = (mid_m[:, None] - 1) * half + jnp.arange(half)[None, :]
    mid_indices = jnp.concatenate([upper, lower], axis=1)        # shape (nf-1, nt)
    mid_blocks = x_fft[mid_indices] * window[None, :]            # broadcast window
    mid_times = jnp.fft.ifft(mid_blocks, axis=1).T               # shape (nt, nf-1)
    mid_phase = jnp.conjugate(_cnm_jax(narr[:, None], mid_m[None, :]))
    coeffs_mid = (sqrt2 / nf) * jnp.real(mid_phase * mid_times)

    # --- Nyquist edge channel (m = nf) ---
    block = x_fft[n_total // 2 - half:n_total // 2] * window[-half:]
    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    coeffs_nyq = jnp.real(
        jnp.sum(
            jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[n_total // 2] * window[0] / 2.0
    ) / (nt * nf)

    # Assemble (nt, nf+1): [DC | interior | Nyquist]
    return jnp.concatenate([
        coeffs_dc[:, None],
        coeffs_mid,
        coeffs_nyq[:, None],
    ], axis=1)


@partial(jit, static_argnames=("nt", "nf"))
def _from_wdm_to_spectrum_impl(
    w: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    r"""JIT-compiled inverse WDM kernel.

    Reconstructs X(f) from the WDM coefficients, then applies IFFT.

    The interior channels are handled in a vectorised scatter-add:
    all sub-band contributions are computed as a batch, then accumulated
    into the output spectrum via ``x_recon.at[indices].add(...)``.

    JAX's ``at[...].add`` is used instead of in-place ``+=`` because
    JAX arrays are immutable.

    Parameters
    ----------
    w : jnp.ndarray, shape (nt, nf + 1)
        Real-valued WDM coefficients.
    window : jnp.ndarray, shape (nt,)
        Precomputed phi-window.
    nt, nf : int
        Static shape parameters.
    """
    n_total = nt * nf
    half = nt // 2
    coeffs_dc = w[:, 0]
    coeffs_nyq = w[:, nf]

    # Modulate and FFT along the time axis: y[n,m] = C_{n,m} · W[n,m] · nf/√2
    n_idx = jnp.arange(nt)[:, None]
    m_idx = jnp.arange(1, nf)[None, :]
    ylm = _cnm_jax(n_idx, m_idx) * w[:, 1:nf] * nf / jnp.sqrt(2.0)
    spectrum_blocks = jnp.fft.fft(ylm, axis=0)

    x_recon = jnp.zeros(n_total, dtype=jnp.complex128)
    narr = jnp.arange(nt)

    # --- DC edge ---
    larr = jnp.arange(1, half)
    x_recon = x_recon.at[1:half].add(
        jnp.sum(
            coeffs_dc[:, None]
            * jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * nf
        * window[1:half]
    )
    x_recon = x_recon.at[0].add(jnp.sum(coeffs_dc) * nf * window[0] / 2.0)

    # --- Interior channels: vectorised scatter-add ---
    mid_blocks = spectrum_blocks.T * window[None, :]
    shifted_blocks = jnp.concatenate(
        [mid_blocks[:, half:], mid_blocks[:, :half]],
        axis=1,
    )
    mid_indices = (
        (jnp.arange(1, nf)[:, None] - 1) * half + jnp.arange(nt)[None, :]
    ).reshape(-1)
    x_recon = x_recon.at[mid_indices].add(shifted_blocks.reshape(-1))

    # --- Nyquist edge ---
    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    x_recon = x_recon.at[n_total // 2 - half:n_total // 2].add(
        jnp.sum(
            coeffs_nyq[:, None]
            * jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt),
            axis=0,
        )
        * nf
        * window[-half:]
    )
    x_recon = x_recon.at[n_total // 2].add(jnp.sum(coeffs_nyq) * nf * window[0] / 2.0)

    return x_recon / (nf / 2.0)


# ---------------------------------------------------------------------------
# Public entry points (match the xp_backend interface)
# ---------------------------------------------------------------------------


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
    """Forward WDM transform (JAX backend).

    See ``xp_backend.from_time_to_wdm`` for the full mathematical description.
    This entry point validates inputs, builds the phi-window on the host,
    converts it to a JAX array, and dispatches to the JIT-compiled kernel.

    Returns shape ``(nt, nf + 1)`` with all-real coefficients.
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

    spectrum = jnp.fft.fft(samples)
    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    return _from_spectrum_to_wdm_impl(spectrum, window, nt, nf)


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
    """Forward WDM transform from Fourier-domain samples (JAX backend)."""
    xp = backend.xp
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    n_total = nt * nf
    spectrum = backend.asarray(data, dtype=xp.complex128)
    if spectrum.ndim != 1:
        raise ValueError("Input frequency-domain data must be one-dimensional.")
    if int(spectrum.shape[0]) != n_total:
        raise ValueError(f"Input length {spectrum.shape[0]} must equal nt*nf={n_total}.")

    projected = _project_to_real_signal_spectrum_impl(spectrum)
    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    return _from_spectrum_to_wdm_impl(projected, window, nt, nf)


def from_wdm_to_time(coeffs: Any, *, a: float, d: float, dt: float, backend: Backend) -> Any:
    """Inverse WDM transform (JAX backend).

    See ``xp_backend.from_wdm_to_time`` for the full mathematical description.
    Accepts shape ``(nt, nf + 1)``.
    """
    xp = backend.xp
    w = backend.asarray(coeffs, dtype=xp.float64)
    if w.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, ncols = (int(dim) for dim in w.shape)
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    spectrum = _from_wdm_to_spectrum_impl(w, window, nt, nf)
    return jnp.real(jnp.fft.ifft(spectrum))


def from_wdm_to_freq(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> Any:
    """Frequency-domain reconstruction from WDM coefficients (JAX backend)."""
    xp = backend.xp
    w = backend.asarray(coeffs, dtype=xp.float64)
    if w.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, ncols = (int(dim) for dim in w.shape)
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    analytic = _from_wdm_to_spectrum_impl(w, window, nt, nf)
    return _project_to_real_signal_spectrum_impl(analytic)
