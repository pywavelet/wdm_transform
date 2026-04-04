"""Backend-dispatched entry points for the WDM transform.

These thin wrappers resolve the active backend (NumPy, JAX, or CuPy),
lazily import the corresponding transform module, and forward all
arguments.  Users should call these functions (or the higher-level
``WDM`` methods) rather than importing a specific backend module.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ..backends import Backend, get_backend

_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms.xp_backend",
    "jax": "wdm_transform.transforms.jax_backend",
    "cupy": "wdm_transform.transforms.xp_backend",
}


def _get_transform_module(backend: str | Backend | None) -> ModuleType:
    resolved_backend = get_backend(backend)
    try:
        module_name = _TRANSFORM_MODULES[resolved_backend.name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRANSFORM_MODULES))
        raise ValueError(
            f"No transform implementation is registered for backend {resolved_backend.name!r}. "
            f"Available transform backends: {available}."
        ) from exc
    return import_module(module_name)


def from_time_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
    dtype: Any | None = None,
) -> Any:
    """Compute WDM coefficients from a one-dimensional time-domain signal.

    Parameters
    ----------
    data : array_like
        Input samples with length ``nt * nf``.
    nt : int
        Number of WDM time bins.
    nf : int
        Number of interior WDM frequency channels.
    a : float
        Window roll-off parameter.
    d : float
        Reserved window parameter.
    dt : float
        Sampling interval of the input signal.
    backend : str, Backend, or None
        Backend name or instance. When omitted, :func:`wdm_transform.get_backend`
        chooses the default backend.

    Returns
    -------
    array_like
        Real-valued coefficient array with shape ``(nt, nf + 1)``.

    Notes
    -----
    This is the backend-dispatched public entry point. It forwards to the
    NumPy/CuPy or JAX implementation after resolving ``backend``.
    """
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    kwargs = {
        "data": data,
        "nt": nt,
        "nf": nf,
        "a": a,
        "d": d,
        "dt": dt,
        "backend": resolved_backend,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return module.from_time_to_wdm(**kwargs)


def from_freq_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
    dtype: Any | None = None,
) -> Any:
    """Compute WDM coefficients from full Fourier-domain samples."""
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    kwargs = {
        "data": data,
        "nt": nt,
        "nf": nf,
        "a": a,
        "d": d,
        "dt": dt,
        "backend": resolved_backend,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return module.from_freq_to_wdm(**kwargs)


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
    dtype: Any | None = None,
) -> Any:
    """Reconstruct a time-domain signal from WDM coefficients.

    Parameters
    ----------
    coeffs : array_like
        Real-valued WDM coefficient array with shape ``(nt, nf + 1)``.
    a : float
        Window roll-off parameter.
    d : float
        Reserved window parameter.
    dt : float
        Sampling interval of the original signal.
    backend : str, Backend, or None
        Backend name or instance. When omitted, :func:`wdm_transform.get_backend`
        chooses the default backend.

    Returns
    -------
    array_like
        Reconstructed one-dimensional time-domain samples.
    """
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    kwargs = {
        "coeffs": coeffs,
        "a": a,
        "d": d,
        "dt": dt,
        "backend": resolved_backend,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return module.from_wdm_to_time(**kwargs)


def from_wdm_to_freq(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: str | Backend | None = None,
    dtype: Any | None = None,
) -> Any:
    """Reconstruct the Fourier-domain signal represented by WDM coefficients.

    Parameters
    ----------
    coeffs : array_like
        Real-valued WDM coefficient array with shape ``(nt, nf + 1)``.
    dt : float
        Sampling interval of the original signal.
    a : float
        Window roll-off parameter.
    d : float
        Reserved window parameter.
    backend : str, Backend, or None
        Backend name or instance. When omitted, :func:`wdm_transform.get_backend`
        chooses the default backend.

    Returns
    -------
    array_like
        Complex-valued Fourier samples on the discrete Fourier grid implied by
        ``dt`` and the coefficient shape.
    """
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    kwargs = {
        "coeffs": coeffs,
        "dt": dt,
        "a": a,
        "d": d,
        "backend": resolved_backend,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return module.from_wdm_to_freq(**kwargs)


__all__ = [
    "from_time_to_wdm",
    "from_freq_to_wdm",
    "from_wdm_to_time",
    "from_wdm_to_freq",
]
