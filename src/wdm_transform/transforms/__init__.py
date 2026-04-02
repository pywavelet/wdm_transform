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
from ._subband import fourier_span_from_wdm_span, wdm_span_from_fourier_span

_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms.xp_backend",
    "jax": "wdm_transform.transforms.jax_backend",
    "cupy": "wdm_transform.transforms.xp_backend",
}

_SUBBAND_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms.xp_subband",
    "jax": "wdm_transform.transforms.jax_subband",
    "cupy": "wdm_transform.transforms.xp_subband",
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


def _get_subband_transform_module(backend: str | Backend | None) -> ModuleType:
    resolved_backend = get_backend(backend)
    try:
        module_name = _SUBBAND_TRANSFORM_MODULES[resolved_backend.name]
    except KeyError as exc:
        available = ", ".join(sorted(_SUBBAND_TRANSFORM_MODULES))
        raise ValueError(
            f"No sub-band transform implementation is registered for backend {resolved_backend.name!r}. "
            f"Available sub-band backends: {available}."
        ) from exc
    return import_module(module_name)


def forward_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
) -> Any:
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.forward_wdm(data, nt=nt, nf=nf, a=a, d=d, dt=dt, backend=resolved_backend)


def inverse_wdm(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
) -> Any:
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.inverse_wdm(coeffs, a=a, d=d, dt=dt, backend=resolved_backend)


def frequency_wdm(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: str | Backend | None = None,
) -> Any:
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.frequency_wdm(coeffs, dt=dt, a=a, d=d, backend=resolved_backend)


def forward_wdm_subband(
    data: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    kmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float = 1.0 / 3.0,
    d: float = 1.0,
    backend: str | Backend | None = None,
) -> tuple[Any, int]:
    resolved_backend = get_backend(backend)
    module = _get_subband_transform_module(resolved_backend)
    return module.forward_wdm_subband(
        data,
        df=df,
        nfreqs_fourier=nfreqs_fourier,
        kmin=kmin,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        a=a,
        d=d,
        backend=resolved_backend,
    )


def inverse_wdm_subband(
    coeffs: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    mmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float = 1.0 / 3.0,
    d: float = 1.0,
    backend: str | Backend | None = None,
) -> tuple[Any, int]:
    resolved_backend = get_backend(backend)
    module = _get_subband_transform_module(resolved_backend)
    return module.inverse_wdm_subband(
        coeffs,
        df=df,
        nfreqs_fourier=nfreqs_fourier,
        mmin=mmin,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        a=a,
        d=d,
        backend=resolved_backend,
    )


__all__ = [
    "forward_wdm",
    "inverse_wdm",
    "frequency_wdm",
    "forward_wdm_subband",
    "inverse_wdm_subband",
    "wdm_span_from_fourier_span",
    "fourier_span_from_wdm_span",
]
