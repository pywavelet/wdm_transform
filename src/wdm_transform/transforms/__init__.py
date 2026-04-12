"""Backend-dispatched entry points for the WDM transform.

These thin wrappers resolve the active backend (NumPy, JAX, or CuPy),
lazily import the corresponding transform module, and forward all
arguments. Users should call these functions (or the higher-level
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


def from_time_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
) -> Any:
    """Compute WDM coefficients from a one-dimensional time-domain signal."""
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.from_time_to_wdm(
        data,
        nt=nt,
        nf=nf,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
    )


def from_freq_to_wdm(
    data: Any,
    *,
    nt: int,
    nf: int,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
) -> Any:
    """Compute WDM coefficients from full Fourier-domain samples."""
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.from_freq_to_wdm(
        data,
        nt=nt,
        nf=nf,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
    )


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
) -> Any:
    """Reconstruct a time-domain signal from WDM coefficients."""
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.from_wdm_to_time(
        coeffs,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
    )


def from_wdm_to_freq(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: str | Backend | None = None,
) -> Any:
    """Reconstruct the Fourier-domain signal represented by WDM coefficients."""
    resolved_backend = get_backend(backend)
    module = _get_transform_module(resolved_backend)
    return module.from_wdm_to_freq(
        coeffs,
        dt=dt,
        a=a,
        d=d,
        backend=resolved_backend,
    )


def from_freq_to_wdm_subband(
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
    """Compute the overlapping WDM sub-band from a compact Fourier span."""
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


def from_freq_to_wdm_band(
    data: Any,
    *,
    df: float,
    nfreqs_fourier: int,
    kmin: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    mmin: int,
    nf_sub_wdm: int,
    a: float = 1.0 / 3.0,
    d: float = 1.0,
    backend: str | Backend | None = None,
) -> Any:
    """Compute a requested WDM channel block from a compact Fourier span.

    This is the study-facing variant of the subband API: it first computes the
    minimal touched WDM span, then returns only the requested compact block
    ``mmin : mmin + nf_sub_wdm``. The requested block must lie inside the
    touched span of the supplied Fourier support.
    """
    coeffs, touched_mmin = from_freq_to_wdm_subband(
        data,
        df=df,
        nfreqs_fourier=nfreqs_fourier,
        kmin=kmin,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
        a=a,
        d=d,
        backend=backend,
    )
    local_start = int(mmin) - int(touched_mmin)
    local_stop = local_start + int(nf_sub_wdm)
    if local_start < 0 or local_stop > int(coeffs.shape[1]):
        raise ValueError(
            "Requested WDM band lies outside the Fourier span's touched channels: "
            f"requested m=[{mmin}, {mmin + nf_sub_wdm}), "
            f"touched m=[{touched_mmin}, {touched_mmin + int(coeffs.shape[1])})."
        )
    return coeffs[:, local_start:local_stop]


def from_wdm_to_freq_subband(
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
    """Reconstruct the touched Fourier span from a compact WDM block."""
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


# Backward-compatible aliases for the older branch API.
forward_wdm = from_time_to_wdm
inverse_wdm = from_wdm_to_time
frequency_wdm = from_wdm_to_freq
forward_wdm_subband = from_freq_to_wdm_subband
forward_wdm_band = from_freq_to_wdm_band
inverse_wdm_subband = from_wdm_to_freq_subband


__all__ = [
    "from_time_to_wdm",
    "from_freq_to_wdm",
    "from_wdm_to_time",
    "from_wdm_to_freq",
    "from_freq_to_wdm_subband",
    "from_freq_to_wdm_band",
    "from_wdm_to_freq_subband",
    "forward_wdm",
    "inverse_wdm",
    "frequency_wdm",
    "forward_wdm_subband",
    "forward_wdm_band",
    "inverse_wdm_subband",
    "wdm_span_from_fourier_span",
    "fourier_span_from_wdm_span",
]
