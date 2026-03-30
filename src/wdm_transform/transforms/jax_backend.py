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
from ..datatypes.series import FrequencySeries
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


def _phi_unit_jax(f: Any, a: float) -> Any:
    r"""Cosine-tapered window Φ(f) evaluated at normalised frequencies (JAX version).

    Identical to ``windows.phi_unit`` but uses ``jnp`` operations so it
    can be traced by JAX's JIT compiler.

    * |f| ≤ a        → 1
    * a < |f| ≤ 1−a  → cos(π/2 · (|f|−a) / (1−2a))
    * |f| > 1−a      → 0
    """
    b = 1.0 - 2.0 * a
    abs_f = jnp.abs(f)
    tapered = jnp.cos((jnp.pi / 2.0) * (abs_f - a) / b)
    inner = jnp.where(abs_f > a, tapered, 1.0)
    return jnp.where(abs_f > a + b, 0.0, inner)


@partial(jit, static_argnames=("nt", "nf"))
def _forward_wdm_impl(
    samples: jnp.ndarray,
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
    samples : jnp.ndarray, shape (N,)
        Complex-valued FFT-ready input (already cast to complex128).
    window : jnp.ndarray, shape (nt,)
        Precomputed phi-window of length nt.
    nt, nf : int
        Static shape parameters (used by JIT for specialisation).
    """
    n_total = nt * nf
    half = nt // 2
    x_fft = jnp.fft.fft(samples)
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
def _inverse_wdm_impl(
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

    return jnp.real(jnp.fft.ifft(x_recon)) / (nf / 2.0)


@partial(jit, static_argnames=("nt", "nf"))
def _frequency_wdm_impl(
    w: jnp.ndarray,
    dt: float,
    a: float,
    nt: int,
    nf: int,
) -> jnp.ndarray:
    r"""JIT-compiled frequency-domain reconstruction from WDM coefficients.

    Evaluates the atom expansion:

        X(f) = √(2·nf) · Σ_{n,m} W[n,m] · g_{n,m}(f)

    All three channel types (DC, interior, Nyquist) are computed as
    vectorised outer products and contracted via ``jnp.einsum``.

    Parameters
    ----------
    w : jnp.ndarray, shape (nt, nf + 1)
        Real-valued WDM coefficients.
    dt : float
        Sampling interval.
    a : float
        Window roll-off parameter.
    nt, nf : int
        Static shape parameters.
    """
    n_total = nt * nf
    freqs = jnp.fft.fftfreq(n_total, d=dt)
    dt_block = nf * dt
    scaled_freqs = freqs * (2.0 * dt_block)  # f / DF where DF = 1/(2·DT)

    n_idx = jnp.arange(nt)

    # --- DC channel: g_{n,0}(f) = exp(-4πi·n·f·DT) · Φ(f/DF) ---
    phase_dc = jnp.exp(-4j * jnp.pi * n_idx[:, None] * freqs[None, :] * dt_block)
    phi_dc = _phi_unit_jax(scaled_freqs, a)
    reconstructed = jnp.einsum("n,nk->k", w[:, 0], phase_dc) * phi_dc

    # --- Nyquist channel: g_{n,nf}(f) = [Φ(f/DF+nf) + Φ(f/DF-nf)] · exp(-4πi·n·f·DT) ---
    phi_nyq = _phi_unit_jax(scaled_freqs + nf, a) + _phi_unit_jax(scaled_freqs - nf, a)
    reconstructed += jnp.einsum("n,nk->k", w[:, nf], phase_dc) * phi_nyq

    # --- Interior channels (m = 1…nf−1), fully vectorised ---
    mid_m = jnp.arange(1, nf)
    mid_coeffs = w[:, 1:nf]

    phase_mid = jnp.exp(-2j * jnp.pi * n_idx[:, None] * freqs[None, :] * dt_block)
    parity = jnp.where((n_idx[:, None] * mid_m[None, :]) % 2 == 0, 1.0, -1.0)
    cnm_vals = _cnm_jax(n_idx[:, None], mid_m[None, :])

    phi_plus = _phi_unit_jax(scaled_freqs[None, :] + mid_m[:, None], a)
    phi_minus = _phi_unit_jax(scaled_freqs[None, :] - mid_m[:, None], a)

    basis = (
        parity[:, :, None]
        * (
            jnp.conjugate(cnm_vals)[:, :, None] * phi_plus[None, :, :]
            + cnm_vals[:, :, None] * phi_minus[None, :, :]
        )
        * phase_mid[:, None, :]
        / jnp.sqrt(2.0)
    )

    reconstructed += jnp.einsum("nm,nmk->k", mid_coeffs, basis)

    return reconstructed * jnp.sqrt(2.0 * nf)


# ---------------------------------------------------------------------------
# Public entry points (match the xp_backend interface)
# ---------------------------------------------------------------------------


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
    """Forward WDM transform (JAX backend).

    See ``xp_backend.forward_wdm`` for the full mathematical description.
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

    window = jnp.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=jnp.complex128)
    return _forward_wdm_impl(samples, window, nt, nf)


def inverse_wdm(coeffs: Any, *, a: float, d: float, dt: float, backend: Backend) -> Any:
    """Inverse WDM transform (JAX backend).

    See ``xp_backend.inverse_wdm`` for the full mathematical description.
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
    return _inverse_wdm_impl(w, window, nt, nf)


def frequency_wdm(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: Backend,
) -> FrequencySeries:
    """Frequency-domain reconstruction from WDM coefficients (JAX backend).

    See ``xp_backend.frequency_wdm`` for the full mathematical description.
    Builds the full (nt, nf−1, N) atom tensor and contracts via ``einsum``.
    Accepts shape ``(nt, nf + 1)``.
    """
    del d
    xp = backend.xp
    w = backend.asarray(coeffs, dtype=xp.float64)
    if w.ndim != 2:
        raise ValueError("WDM coefficients must be a two-dimensional array.")

    nt, ncols = (int(dim) for dim in w.shape)
    nf = ncols - 1
    validate_transform_shape(nt, nf)
    validate_window_parameter(a)

    reconstructed = _frequency_wdm_impl(w, dt, a, nt, nf)
    return FrequencySeries(
        reconstructed,
        df=1.0 / (nt * nf * dt),
        backend=backend,
    )
