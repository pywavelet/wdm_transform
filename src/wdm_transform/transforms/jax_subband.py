"""JAX stubs for low-level WDM sub-band transforms."""

from __future__ import annotations

from typing import Any

from ..backends import Backend


def forward_wdm_subband(
    data: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    kmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float,
    d: float,
    backend: Backend,
) -> tuple[Any, int]:
    del data, df, nfreqs_fourier, kmin, nfreqs_wdm, ntimes_wdm, a, d, backend
    raise NotImplementedError(
        "forward_wdm_subband is not implemented for the JAX backend."
    )


def inverse_wdm_subband(
    coeffs: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    mmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    a: float,
    d: float,
    backend: Backend,
) -> tuple[Any, int]:
    del coeffs, df, nfreqs_fourier, mmin, nfreqs_wdm, ntimes_wdm, a, d, backend
    raise NotImplementedError(
        "inverse_wdm_subband is not implemented for the JAX backend."
    )
