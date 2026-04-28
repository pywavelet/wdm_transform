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
Same as ``xp_backend``: outputs are returned canonically with shape
``(batch, nt, nf + 1)`` and all-real entries. Legacy unbatched inputs are
accepted by the public entry points and normalized to a singleton batch.

* ``W[..., 0]``       — DC edge channel     (m = 0)
* ``W[..., 1:nf]``    — interior channels   (m = 1 … nf−1)
* ``W[..., nf]``      — Nyquist edge channel (m = nf)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit

from ..backends import Backend
from ..windows import (
    phi_window,
    validate_transform_shape,
    validate_window_order,
    validate_window_parameter,
)


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
    n_total = spectrum.shape[-1]
    mirror = (-jnp.arange(n_total)) % n_total
    return 0.5 * (spectrum + jnp.conjugate(spectrum[..., mirror]))


@partial(jit, static_argnames=("nt", "nf"))
def _from_spectrum_to_wdm_batch_impl(
    x_fft_batch: jnp.ndarray,
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

    Returns shape ``(batch, nt, nf + 1)`` with all-real coefficients.

    Parameters
    ----------
    x_fft_batch : jnp.ndarray, shape (batch, N)
        Complex-valued Fourier-domain signals.
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
    block = x_fft_batch[:, 1:half] * window[None, 1:half]
    larr = jnp.arange(1, half)
    dc_phase = jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt)
    coeffs_dc = jnp.real(
        jnp.einsum("bl,nl->bn", block, dc_phase)
        + x_fft_batch[:, 0][:, None] * window[0] / 2.0
    ) * sqrt2

    # --- Interior channels (m = 1…nf−1), fully vectorised ---
    mid_m = jnp.arange(1, nf)
    upper = mid_m[:, None] * half + jnp.arange(half)[None, :]   # shape (nf-1, half)
    lower = (mid_m[:, None] - 1) * half + jnp.arange(half)[None, :]
    mid_indices = jnp.concatenate([upper, lower], axis=1)        # shape (nf-1, nt)
    mid_blocks = x_fft_batch[:, mid_indices] * window[None, None, :]
    mid_times = jnp.fft.ifft(mid_blocks, axis=-1).transpose(0, 2, 1)
    mid_phase = jnp.conjugate(_cnm_jax(narr[:, None], mid_m[None, :]))
    sign = jnp.where((narr[:, None] * mid_m[None, :]) % 2 == 0, 1.0, -1.0)
    coeffs_mid = (
        sign[None, :, :]
        * sqrt2
        * jnp.real(nt * mid_times * mid_phase[None, :, :])
    )

    # --- Nyquist edge channel (m = nf) ---
    block = x_fft_batch[:, n_total // 2 - half:n_total // 2] * window[None, -half:]
    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    nyq_phase = jnp.exp(4j * jnp.pi * larr[None, :] * narr[:, None] / nt)
    coeffs_nyq = jnp.real(
        jnp.einsum("bl,nl->bn", block, nyq_phase)
        + jnp.conjugate(x_fft_batch[:, n_total // 2])[:, None] * window[0] / 2.0
    ) * sqrt2

    # Assemble (batch, nt, nf+1): [DC | interior | Nyquist]
    return jnp.concatenate([
        coeffs_dc[:, :, None],
        coeffs_mid,
        coeffs_nyq[:, :, None],
    ], axis=2)


@partial(jit, static_argnames=("nt", "nf"))
def _from_wdm_to_spectrum_batch_impl(
    w_batch: jnp.ndarray,
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
    w_batch : jnp.ndarray, shape (batch, nt, nf + 1)
        Real-valued WDM coefficients.
    window : jnp.ndarray, shape (nt,)
        Precomputed phi-window.
    nt, nf : int
        Static shape parameters.
    """
    n_total = nt * nf
    half = nt // 2
    coeffs_dc = w_batch[:, :, 0]
    coeffs_nyq = w_batch[:, :, nf]
    batch_idx = jnp.arange(w_batch.shape[0])[:, None, None]
    narr = jnp.arange(nt)
    xfr = jnp.zeros((w_batch.shape[0], n_total // 2 + 1), dtype=jnp.complex128)

    # Eq. \ref{eq:inverse_fourier}: synthesize the one-sided spectrum.
    larr = jnp.arange(half)
    dc_phase = jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt)
    xfr = xfr.at[:, :half].add(
        jnp.einsum("bn,nl->bl", coeffs_dc, dc_phase)
        * window[None, :half]
        / jnp.sqrt(2.0)
    )

    # FFT trick: for l ∈ [(m-1)·half, (m+1)·half), the explicit DFT sum
    # Σ_n C_{nm}·w[n,m]·exp(-2πi·n·l/nt) factorises into
    # (-1)^((m-1)·n) · exp(-2πi·n·k/nt), so we reduce the inverse to an FFT.
    mid_m = jnp.arange(1, nf)
    mid_larr = (mid_m[:, None] - 1) * half + jnp.arange(nt)[None, :]
    sign = jnp.where((narr[:, None] * (mid_m[None, :] - 1)) % 2 == 0, 1.0, -1.0)
    mid_weighted = _cnm_jax(narr[:, None], mid_m[None, :])[None, :, :]
    mid_weighted = mid_weighted * w_batch[:, :, 1:nf] * sign[None, :, :]
    mid_values = (
        jnp.fft.fft(mid_weighted, axis=1).transpose(0, 2, 1)
        * jnp.concatenate([window[-half:], window[:half]])[None, None, :]
        / jnp.sqrt(2.0)
    )
    xfr = xfr.at[batch_idx, mid_larr[None, :, :]].add(mid_values)

    larr = jnp.arange(n_total // 2 - half, n_total // 2)
    nyq_phase = jnp.exp(-4j * jnp.pi * narr[:, None] * larr[None, :] / nt)
    xfr = xfr.at[:, n_total // 2 - half:n_total // 2].add(
        jnp.einsum("bn,nl->bl", coeffs_nyq, nyq_phase)
        * window[None, -half:]
        / jnp.sqrt(2.0)
    )
    xfr = xfr.at[:, n_total // 2].add(
        jnp.sum(coeffs_nyq, axis=1) * window[0] / jnp.sqrt(2.0)
    )

    return jnp.concatenate([xfr, jnp.conjugate(xfr[:, 1:-1][:, ::-1])], axis=1)


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

    Returns shape ``(batch, nt, nf + 1)``. Rank-1 inputs are normalized to a
    singleton leading batch axis.
    """
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    n_total = nt * nf
    xp = backend.xp
    samples = backend.asarray(data, dtype=xp.complex128)
    if samples.ndim not in (1, 2):
        raise ValueError("Input time-domain data must be one- or two-dimensional.")
    if samples.ndim == 1:
        samples = samples[None, :]
    if int(samples.shape[-1]) != n_total:
        raise ValueError(f"Input length {samples.shape[-1]} must equal nt*nf={n_total}.")

    spectrum = backend.asarray(backend.fft.fft(samples, axis=-1), dtype=xp.complex128)
    window = jnp.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=jnp.complex128,
    )
    return _from_spectrum_to_wdm_batch_impl(spectrum, window, nt, nf)


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
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    n_total = nt * nf
    xp = backend.xp
    spectrum = backend.asarray(data, dtype=xp.complex128)
    if spectrum.ndim not in (1, 2):
        raise ValueError("Input frequency-domain data must be one- or two-dimensional.")
    if spectrum.ndim == 1:
        spectrum = spectrum[None, :]
    if int(spectrum.shape[-1]) != n_total:
        raise ValueError(f"Input length {spectrum.shape[-1]} must equal nt*nf={n_total}.")

    projected = _project_to_real_signal_spectrum_impl(spectrum)
    window = jnp.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=jnp.complex128,
    )
    return _from_spectrum_to_wdm_batch_impl(projected, window, nt, nf)


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: Backend,
) -> Any:
    """Inverse WDM transform (JAX backend).

    See ``xp_backend.from_wdm_to_time`` for the full mathematical description.
    Accepts shape ``(nt, nf + 1)`` or ``(batch, nt, nf + 1)`` and returns
    the canonical batch-first shape ``(batch, nt * nf)``.
    """
    w = backend.asarray(coeffs, dtype=jnp.float64)
    if w.ndim not in (2, 3):
        raise ValueError("WDM coefficients must be a two- or three-dimensional array.")
    if w.ndim == 2:
        w = w[None, :, :]

    nt, ncols = (int(dim) for dim in w.shape[-2:])
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    window = jnp.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=jnp.complex128,
    )
    spectrum = _from_wdm_to_spectrum_batch_impl(w, window, nt, nf)
    return jnp.real(jnp.fft.ifft(spectrum, axis=-1))


def from_wdm_to_freq(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> Any:
    """Frequency-domain reconstruction from WDM coefficients (JAX backend)."""
    w = backend.asarray(coeffs, dtype=jnp.float64)
    if w.ndim not in (2, 3):
        raise ValueError("WDM coefficients must be a two- or three-dimensional array.")
    if w.ndim == 2:
        w = w[None, :, :]

    nt, ncols = (int(dim) for dim in w.shape[-2:])
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)
    validate_window_order(d)

    window = jnp.asarray(
        phi_window(backend, nt, nf, dt, a, d),
        dtype=jnp.complex128,
    )
    analytic = _from_wdm_to_spectrum_batch_impl(w, window, nt, nf)
    return _project_to_real_signal_spectrum_impl(analytic)
