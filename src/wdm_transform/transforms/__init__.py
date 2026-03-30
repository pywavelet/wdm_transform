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


__all__ = ["forward_wdm", "inverse_wdm", "frequency_wdm"]
