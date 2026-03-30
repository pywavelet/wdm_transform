from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .base import Backend
from .numpy_backend import NUMPY_BACKEND

_BACKENDS: dict[str, Backend] = {
    NUMPY_BACKEND.name: NUMPY_BACKEND,
}

_BUILTIN_BACKEND_LOADERS = {
    "jax": ("wdm_transform.backends.jax_backend", "load_jax_backend"),
    "cupy": ("wdm_transform.backends.cupy_backend", "load_cupy_backend"),
}


def register_backend(name: str, xp: Any, fft: Any) -> Backend:
    backend = Backend(name=name, xp=xp, fft=fft)
    _BACKENDS[name] = backend
    return backend


def _load_builtin_backend(name: str) -> Backend:
    try:
        module_name, loader_name = _BUILTIN_BACKEND_LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend {name!r}.") from exc

    module = import_module(module_name)
    loader = getattr(module, loader_name)
    backend = loader()
    _BACKENDS[name] = backend
    return backend


def get_backend(backend: str | Backend | None = None) -> Backend:
    if isinstance(backend, Backend):
        return backend
    if backend is None:
        backend = os.environ.get("WDM_BACKEND")
    if backend is None:
        return NUMPY_BACKEND
    try:
        return _BACKENDS[backend]
    except KeyError as exc:
        try:
            return _load_builtin_backend(backend)
        except ValueError:
            available = ", ".join(sorted(set(_BACKENDS) | set(_BUILTIN_BACKEND_LOADERS)))
            raise ValueError(f"Unknown backend {backend!r}. Available backends: {available}.") from exc


__all__ = ["Backend", "NUMPY_BACKEND", "get_backend", "register_backend"]
