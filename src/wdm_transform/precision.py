from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backends import Backend


@dataclass(frozen=True)
class Precision:
    """Resolved real/complex dtype pair for a backend."""

    name: str
    real_dtype: Any
    complex_dtype: Any


def _normalize_dtype_name(backend: Backend, dtype: Any | None) -> str | None:
    if dtype is None:
        return None
    try:
        return backend.xp.dtype(dtype).name
    except TypeError as exc:
        raise ValueError("dtype must be float32 or float64.") from exc


def resolve_precision(
    backend: Backend,
    dtype: Any | None = None,
    *,
    default: str = "float64",
) -> Precision:
    """Resolve a supported real precision and its paired complex dtype."""
    name = _normalize_dtype_name(backend, dtype) or default
    if name == "float32":
        return Precision(
            name=name,
            real_dtype=backend.xp.float32,
            complex_dtype=backend.xp.complex64,
        )
    if name == "float64":
        return Precision(
            name=name,
            real_dtype=backend.xp.float64,
            complex_dtype=backend.xp.complex128,
        )
    raise ValueError("dtype must be float32 or float64.")


def infer_real_precision(
    backend: Backend,
    value: Any,
    *,
    default: str = "float64",
) -> Precision:
    """Infer precision from a real-valued array or fall back to a default."""
    value_dtype = getattr(value, "dtype", None)
    name = _normalize_dtype_name(backend, value_dtype)
    if name in {"float32", "float64"}:
        return resolve_precision(backend, name)
    return resolve_precision(backend, None, default=default)
