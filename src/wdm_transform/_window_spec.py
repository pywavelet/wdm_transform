"""Small, shared resolver for the two supported Wilson window families."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    """Selected family and the effective Gaussian settings retained for inversion."""

    window: str
    nu: float | None = None
    support: int | None = None


_MEYER_SPEC = WindowSpec("meyer")


def validate_nu(nu: float) -> float:
    """Validate the interval independently checked against the paper recursion."""
    if isinstance(nu, bool) or not isinstance(nu, Real) or not np.isfinite(nu):
        raise ValueError("nu must be a finite positive real number.")
    if nu <= 0:
        raise ValueError("nu must be a finite positive real number.")
    if not 0.3 <= nu <= 0.8:
        raise ValueError("Gaussian nu is currently validated only in [0.3, 0.8].")
    return float(nu)


def resolve_window(
    window: str = "meyer",
    nu: float | None = None,
    support: int | None = None,
    nf: int | None = None,
    a: float = 1.0 / 3.0,
    d: float = 1.0,
) -> WindowSpec:
    """Resolve Gaussian defaults without changing historical Meyer validation."""
    if not isinstance(window, str) or window not in ("meyer", "gaussian"):
        raise ValueError("window must be 'meyer' or 'gaussian' (case-sensitive).")
    if window == "meyer":
        if nu is not None or support is not None:
            raise ValueError(
                "nu and support are only meaningful for window='gaussian'."
            )
        return _MEYER_SPEC
    if a != 1.0 / 3.0 or d != 1.0:
        raise ValueError("Gaussian ignores default a=1/3, d=1; other a/d are invalid.")
    resolved_nu = validate_nu(0.5 if nu is None else nu)
    if isinstance(nf, bool) or not isinstance(nf, Integral) or nf < 2 or nf % 2:
        raise ValueError("Gaussian nf must be a positive even integer.")
    if support is None:
        support = min(16, nf)
    if isinstance(support, bool) or not isinstance(support, Integral):
        raise ValueError("Gaussian support must be an integer with 1 <= support <= nf.")
    if not 1 <= support <= nf:
        raise ValueError("Gaussian support must be an integer with 1 <= support <= nf.")
    return WindowSpec(window, resolved_nu, int(support))
