"""Backend-dispatched entry points for the WDM transform.

These thin wrappers resolve the active backend (NumPy, JAX, or CuPy),
lazily import the corresponding transform module, and forward all
arguments. Users should call these functions (or the higher-level
``WDM`` methods) rather than importing a specific backend module.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Literal

from .._window_spec import WindowSpec, resolve_window
from ..backends import Backend, get_backend
from ._subband import (
    _require_integer_span,
    fourier_span_from_wdm_span,
    validate_wdm_span,
    wdm_span_from_fourier_span,
)

_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms.xp_backend",
    "jax": "wdm_transform.transforms.jax_backend",
    "cupy": "wdm_transform.transforms.xp_backend",
}

_GAUSSIAN_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms._gaussian",
    "jax": "wdm_transform.transforms.jax_gaussian",
    "cupy": "wdm_transform.transforms._gaussian",
}

_SUBBAND_TRANSFORM_MODULES = {
    "numpy": "wdm_transform.transforms.xp_subband",
    "jax": "wdm_transform.transforms.jax_subband",
    "cupy": "wdm_transform.transforms.xp_subband",
}

_GAUSSIAN_SUBBAND_MODULES = {
    "numpy": "wdm_transform.transforms.gaussian_subband",
    "jax": "wdm_transform.transforms.jax_gaussian_subband",
    "cupy": "wdm_transform.transforms.gaussian_subband",
}


def _get_transform_module(
    backend: str | Backend | None, window: str = "meyer"
) -> ModuleType:
    resolved_backend = get_backend(backend)
    modules = (
        _GAUSSIAN_TRANSFORM_MODULES if window == "gaussian" else _TRANSFORM_MODULES
    )
    try:
        module_name = modules[resolved_backend.name]
    except KeyError as exc:
        available = ", ".join(sorted(modules))
        raise ValueError(
            "No transform implementation is registered for backend "
            f"{resolved_backend.name!r}. "
            f"Available transform backends: {available}."
        ) from exc
    return import_module(module_name)


def _inverse_window_spec(
    coeffs: Any,
    backend: Backend,
    window: str,
    nu: float | None,
    support: int | None,
    a: float,
    d: float,
) -> WindowSpec:
    # Meyer keeps its existing validation and array conversion in the kernels.
    nf = 0
    if window == "gaussian":
        values = backend.asarray(coeffs)
        if values.ndim not in (2, 3):
            raise ValueError("WDM coeffs must be a two- or three-dimensional array.")
        nf = int(values.shape[-1]) - 1
    return resolve_window(window, nu, support, nf, a, d)


def _get_subband_transform_module(
    backend: str | Backend | None, window: str = "meyer"
) -> ModuleType:
    resolved_backend = get_backend(backend)
    modules = (
        _GAUSSIAN_SUBBAND_MODULES
        if window == "gaussian"
        else _SUBBAND_TRANSFORM_MODULES
    )
    try:
        module_name = modules[resolved_backend.name]
    except KeyError as exc:
        available = ", ".join(sorted(modules))
        raise ValueError(
            "No sub-band transform implementation is registered for backend "
            f"{resolved_backend.name!r}. "
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
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> Any:
    """Compute Wilson coefficients with Meyer or modified-Gaussian windows.

    Gaussian ``nu`` defaults to 0.5 and support to ``min(16, nf)`` half-band
    spacings. Its default ``a=1/3, d=1`` values are ignored. Meyer rejects
    explicit ``nu`` or ``support``.
    """
    resolved_backend = get_backend(backend)
    spec = resolve_window(window, nu, support, nf, a, d)
    module = _get_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
    return module.from_time_to_wdm(
        data,
        nt=nt,
        nf=nf,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
        **options,
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
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> Any:
    """Compute Wilson coefficients from full complex Fourier-domain samples.

    The spectrum is projected to the Fourier transform of its real inverse.
    Window options follow :func:`from_time_to_wdm`; an rFFT half-spectrum is
    not a full Fourier grid.
    """
    resolved_backend = get_backend(backend)
    spec = resolve_window(window, nu, support, nf, a, d)
    module = _get_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
    return module.from_freq_to_wdm(
        data,
        nt=nt,
        nf=nf,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
        **options,
    )


def from_wdm_to_time(
    coeffs: Any,
    *,
    a: float,
    d: float,
    dt: float,
    backend: str | Backend | None = None,
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> Any:
    """Reconstruct a time-domain signal with the originating window settings.

    Bare arrays carry no window metadata: the caller must supply the family
    and Gaussian settings used for analysis. :class:`WDM` does this for you.
    """
    resolved_backend = get_backend(backend)
    spec = _inverse_window_spec(coeffs, resolved_backend, window, nu, support, a, d)
    module = _get_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
    return module.from_wdm_to_time(
        coeffs,
        a=a,
        d=d,
        dt=dt,
        backend=resolved_backend,
        **options,
    )


def from_wdm_to_freq(
    coeffs: Any,
    *,
    dt: float,
    a: float,
    d: float,
    backend: str | Backend | None = None,
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> Any:
    """Reconstruct a full Fourier grid with the originating window settings."""
    resolved_backend = get_backend(backend)
    spec = _inverse_window_spec(coeffs, resolved_backend, window, nu, support, a, d)
    module = _get_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
    return module.from_wdm_to_freq(
        coeffs,
        dt=dt,
        a=a,
        d=d,
        backend=resolved_backend,
        **options,
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
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> tuple[Any, int]:
    """Compute the overlapping WDM sub-band from a compact Fourier span.

    Use this when the input is already cropped in the one-sided Fourier
    domain. The function evaluates only the WDM frequency channels whose
    window support intersects the supplied Fourier span, rather than
    materializing the full WDM grid.

    Args:
        data: One-dimensional one-sided Fourier samples for the compact span
            ``[kmin, kmin + len(data))``.
        df: Frequency spacing of the full one-sided Fourier grid.
        nfreqs_fourier: Number of samples in the full one-sided Fourier grid,
            including DC and Nyquist.
        kmin: Starting Fourier-bin index of ``data`` in the full one-sided
            grid.
        nfreqs_wdm: Number of positive WDM frequency intervals in the full
            transform. A full WDM coefficient grid has ``nfreqs_wdm + 1``
            frequency channels.
        ntimes_wdm: Number of WDM time bins.
        a: Window roll-off parameter.
        d: Window order parameter.
        backend: Backend name or object. If omitted, the default backend is
            used.
        window: Meyer (default) or orthogonalized modified Gaussian.
        nu: Gaussian shape parameter, default 0.5; omitted for Meyer.
        support: Gaussian half-band spacings retained on each side, default
            ``min(16, nfreqs_wdm)``; omitted for Meyer.

    Returns:
        A tuple ``(coeffs, mmin)``. ``coeffs`` has shape
        ``(ntimes_wdm, nf_sub_wdm)`` and stores the touched WDM channels.
        ``mmin`` is the first WDM frequency-channel index represented by
        ``coeffs[:, 0]``.

    Notes:
        Fourier bins outside the supplied crop are zero. Retain ``window``,
        ``nu`` and ``support`` for the inverse; the returned array carries no
        window metadata.
    """
    resolved_backend = get_backend(backend)
    spec = resolve_window(window, nu, support, nfreqs_wdm, a, d)
    module = _get_subband_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
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
        **options,
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
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> Any:
    """Compute a requested WDM channel block from a compact Fourier span.

    This is the study-facing variant of the subband API: it first computes the
    minimal touched WDM span, then returns only the requested compact block
    ``mmin : mmin + nf_sub_wdm``. The requested block must lie inside the
    touched span of the supplied Fourier support.

    Args:
        data: One-dimensional one-sided Fourier samples for the compact span
            ``[kmin, kmin + len(data))``.
        df: Frequency spacing of the full one-sided Fourier grid.
        nfreqs_fourier: Number of samples in the full one-sided Fourier grid,
            including DC and Nyquist.
        kmin: Starting Fourier-bin index of ``data`` in the full one-sided
            grid.
        nfreqs_wdm: Number of positive WDM frequency intervals in the full
            transform. A full WDM coefficient grid has ``nfreqs_wdm + 1``
            frequency channels.
        ntimes_wdm: Number of WDM time bins.
        mmin: First WDM frequency-channel index to return.
        nf_sub_wdm: Number of WDM frequency channels to return.
        a: Window roll-off parameter.
        d: Window order parameter.
        backend: Backend name or object. If omitted, the default backend is
            used.
        window: Meyer (default) or orthogonalized modified Gaussian.
        nu: Gaussian shape parameter, default 0.5; omitted for Meyer.
        support: Gaussian support, default ``min(16, nfreqs_wdm)``.

    Returns:
        The requested coefficient block with shape
        ``(ntimes_wdm, nf_sub_wdm)``.
    """
    if window == "gaussian":
        # Validate the requested block before computing the larger touched span.
        _require_integer_span(mmin=mmin, nf_sub_wdm=nf_sub_wdm)
        validate_wdm_span(nfreqs_wdm=nfreqs_wdm, mmin=mmin, nf_sub_wdm=nf_sub_wdm)
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
        window=window,
        nu=nu,
        support=support,
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
    window: Literal["meyer", "gaussian"] = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> tuple[Any, int]:
    """Reconstruct the touched Fourier span from a compact WDM block.

    This is the inverse of the compact sub-band transform. It expands a block
    of adjacent WDM frequency channels into the one-sided Fourier bins touched
    by those channels' window support.

    Args:
        coeffs: Two-dimensional WDM coefficient block with shape
            ``(ntimes_wdm, nf_sub_wdm)``.
        df: Frequency spacing of the full one-sided Fourier grid.
        nfreqs_fourier: Number of samples in the full one-sided Fourier grid,
            including DC and Nyquist.
        mmin: First WDM frequency-channel index represented by
            ``coeffs[:, 0]``.
        nfreqs_wdm: Number of positive WDM frequency intervals in the full
            transform. A full WDM coefficient grid has ``nfreqs_wdm + 1``
            frequency channels.
        ntimes_wdm: Number of WDM time bins.
        a: Window roll-off parameter.
        d: Window order parameter.
        backend: Backend name or object. If omitted, the default backend is
            used.
        window: Window family used for the supplied coefficients.
        nu: Originating Gaussian shape parameter, default 0.5.
        support: Originating Gaussian support, default ``min(16, nfreqs_wdm)``.

    Returns:
        A tuple ``(spectrum, kmin)``. ``spectrum`` contains the reconstructed
        one-sided Fourier span, and ``kmin`` is the starting Fourier-bin index
        of ``spectrum[0]`` in the full one-sided grid.

    Notes:
        WDM channels outside the supplied block are zero. Supply the same
        ``window``, ``nu`` and ``support`` used to create the coefficients.
    """
    resolved_backend = get_backend(backend)
    spec = resolve_window(window, nu, support, nfreqs_wdm, a, d)
    module = _get_subband_transform_module(resolved_backend, spec.window)
    options = {} if spec.window == "meyer" else {"nu": spec.nu, "support": spec.support}
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
        **options,
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
