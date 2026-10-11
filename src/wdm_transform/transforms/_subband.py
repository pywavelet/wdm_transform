"""Shared helpers for low-level WDM sub-band transforms."""

from __future__ import annotations

from numbers import Integral

from .._window_spec import resolve_window
from ..windows import validate_transform_shape


def validate_subband_grid(
    *,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
) -> int:
    """Validate the full one-sided Fourier grid against the full WDM grid."""
    nfreqs_fourier = int(nfreqs_fourier)
    nfreqs_wdm = int(nfreqs_wdm)
    ntimes_wdm = int(ntimes_wdm)

    if nfreqs_fourier < 2:
        raise ValueError("nfreqs_fourier must be at least 2.")

    validate_transform_shape(ntimes_wdm, nfreqs_wdm)

    n_total = 2 * (nfreqs_fourier - 1)
    if n_total != ntimes_wdm * nfreqs_wdm:
        raise ValueError(
            "Inconsistent full-grid sizes: 2 * (nfreqs_fourier - 1) must equal "
            "ntimes_wdm * nfreqs_wdm."
        )
    return n_total


def validate_gaussian_subband_grid(
    *,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
) -> int:
    """Validate integer Gaussian dimensions without changing Meyer coercions."""
    for name, value in (
        ("nfreqs_fourier", nfreqs_fourier),
        ("nfreqs_wdm", nfreqs_wdm),
        ("ntimes_wdm", ntimes_wdm),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError(f"{name} must be a positive integer of at least 2.")
    return validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )


def _require_integer_span(**coordinates: int) -> None:
    for name, value in coordinates.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")


def _overlap_bounds(
    start: int, stop: int, lower: int, upper: int
) -> tuple[int, int] | None:
    """Intersect inclusive integer intervals, retaining their exact endpoints."""
    start, stop = max(start, lower), min(stop, upper)
    return (start, stop) if start <= stop else None


def _gaussian_wdm_span(
    *, nt: int, nf: int, support: int, kmin: int, lendata: int
) -> tuple[int, int]:
    half_band = nt // 2
    signal_length = nt * nf
    radius = support * half_band
    kmax = kmin + lendata - 1
    touched_channels = []
    # A crop contributes its direct and conjugate-reflected periodic images.
    for start, stop in ((kmin, kmax), (-kmax, -kmin)):
        for wrap in (-1, 0, 1):
            start_image = start + wrap * signal_length
            stop_image = stop + wrap * signal_length
            # Channel taps are [m*half_band-radius, m*half_band+radius-1].
            first_channel = -(-(start_image - radius + 1) // half_band)
            last_channel = (stop_image + radius) // half_band
            overlap = _overlap_bounds(first_channel, last_channel, 0, nf)
            if overlap is not None:
                touched_channels.append(overlap)
    first = min(start for start, _ in touched_channels)
    last = max(stop for _, stop in touched_channels)
    return first, last - first + 1


def _gaussian_fourier_span(
    *, nt: int, nf: int, support: int, mmin: int, nf_sub_wdm: int
) -> tuple[int, int]:
    half_band = nt // 2
    signal_length = nt * nf
    last_channel = mmin + nf_sub_wdm - 1
    first_tap = (mmin - support) * half_band
    last_tap = (last_channel + support) * half_band - 1
    touched_bins = []
    # Folding either lobe onto the one-sided grid includes DC and Nyquist.
    for start, stop in ((first_tap, last_tap), (-last_tap, -first_tap)):
        for wrap in (-1, 0, 1):
            overlap = _overlap_bounds(
                start + wrap * signal_length,
                stop + wrap * signal_length,
                0,
                signal_length // 2,
            )
            if overlap is not None:
                touched_bins.append(overlap)
    first = min(start for start, _ in touched_bins)
    last = max(stop for _, stop in touched_bins)
    return first, last - first + 1


def dt_from_df(*, df: float, nfreqs_fourier: int) -> float:
    """Recover the full-grid sampling cadence from the one-sided spacing."""
    if df <= 0.0:
        raise ValueError("df must be positive.")
    return 1.0 / (2 * (int(nfreqs_fourier) - 1) * df)


def validate_fourier_span(
    *,
    nfreqs_fourier: int,
    kmin: int,
    lendata: int,
) -> None:
    """Validate a compact one-sided Fourier span."""
    nfreqs_fourier = int(nfreqs_fourier)
    kmin = int(kmin)
    lendata = int(lendata)

    if lendata <= 0:
        raise ValueError("lendata must be positive.")
    if kmin < 0 or kmin >= nfreqs_fourier:
        raise ValueError(
            f"kmin must satisfy 0 <= kmin < nfreqs_fourier={nfreqs_fourier}."
        )
    if kmin + lendata > nfreqs_fourier:
        raise ValueError(
            "Fourier span exceeds the full one-sided grid: "
            f"kmin + lendata = {kmin + lendata}, nfreqs_fourier = {nfreqs_fourier}."
        )


def validate_wdm_span(
    *,
    nfreqs_wdm: int,
    mmin: int,
    nf_sub_wdm: int,
) -> None:
    """Validate a compact WDM frequency-channel span."""
    nfreqs_wdm = int(nfreqs_wdm)
    mmin = int(mmin)
    nf_sub_wdm = int(nf_sub_wdm)

    if nf_sub_wdm <= 0:
        raise ValueError("nf_sub_wdm must be positive.")
    if mmin < 0 or mmin > nfreqs_wdm:
        raise ValueError(f"mmin must satisfy 0 <= mmin <= nfreqs_wdm={nfreqs_wdm}.")
    if mmin + nf_sub_wdm > nfreqs_wdm + 1:
        raise ValueError(
            "WDM span exceeds the full grid: "
            f"mmin + nf_sub_wdm = {mmin + nf_sub_wdm}, "
            f"nfreqs_wdm + 1 = {nfreqs_wdm + 1}."
        )


def wdm_span_from_fourier_span(
    *,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    kmin: int,
    lendata: int,
    window: str = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> tuple[int, int]:
    """Map a compact one-sided Fourier span onto the touched WDM channels.

    Args:
        nfreqs_fourier: Number of samples in the full one-sided Fourier grid,
            including DC and Nyquist.
        nfreqs_wdm: Number of positive WDM frequency intervals in the full
            transform. A full WDM coefficient grid has ``nfreqs_wdm + 1``
            frequency channels.
        ntimes_wdm: Number of WDM time bins.
        kmin: Starting Fourier-bin index of the compact Fourier span.
        lendata: Number of Fourier bins in the compact span.
        window: Meyer or modified Gaussian; default Meyer preserves historical spans.
        nu: Gaussian shape parameter; it does not alter the geometric tap span.
        support: Gaussian spectral support in channel spacings on either side.

    Returns:
        A tuple ``(mmin, nf_sub_wdm)``. ``mmin`` is the first touched WDM
        frequency-channel index, and ``nf_sub_wdm`` is the number of adjacent
        WDM channels touched by the Fourier span.
    """
    spec = resolve_window(window, nu, support, nfreqs_wdm)
    if spec.window == "gaussian":
        validate_gaussian_subband_grid(
            nfreqs_fourier=nfreqs_fourier,
            nfreqs_wdm=nfreqs_wdm,
            ntimes_wdm=ntimes_wdm,
        )
        _require_integer_span(kmin=kmin, lendata=lendata)
        validate_fourier_span(nfreqs_fourier=nfreqs_fourier, kmin=kmin, lendata=lendata)
        return _gaussian_wdm_span(
            nt=int(ntimes_wdm),
            nf=int(nfreqs_wdm),
            support=spec.support,
            kmin=int(kmin),
            lendata=int(lendata),
        )
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_fourier_span(
        nfreqs_fourier=nfreqs_fourier,
        kmin=kmin,
        lendata=lendata,
    )

    half = int(ntimes_wdm) // 2
    kmin = int(kmin)
    kmax = kmin + int(lendata) - 1

    mmin = 0 if kmin < half else kmin // half
    upper_start = int(nfreqs_wdm) * half - half
    mmax = int(nfreqs_wdm) if kmax >= upper_start else kmax // half + 1
    return mmin, mmax - mmin + 1


def fourier_span_from_wdm_span(
    *,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    mmin: int,
    nf_sub_wdm: int,
    window: str = "meyer",
    nu: float | None = None,
    support: int | None = None,
) -> tuple[int, int]:
    """Map a compact WDM channel span onto the touched one-sided Fourier bins.

    Args:
        nfreqs_fourier: Number of samples in the full one-sided Fourier grid,
            including DC and Nyquist.
        nfreqs_wdm: Number of positive WDM frequency intervals in the full
            transform. A full WDM coefficient grid has ``nfreqs_wdm + 1``
            frequency channels.
        ntimes_wdm: Number of WDM time bins.
        mmin: First WDM frequency-channel index in the compact WDM span.
        nf_sub_wdm: Number of adjacent WDM frequency channels in the compact
            span.
        window: Meyer or modified Gaussian; default Meyer preserves historical spans.
        nu: Gaussian shape parameter; it does not alter the geometric tap span.
        support: Gaussian spectral support in channel spacings on either side.

    Returns:
        A tuple ``(kmin, lendata)``. ``kmin`` is the first touched one-sided
        Fourier-bin index, and ``lendata`` is the number of touched Fourier
        bins.
    """
    spec = resolve_window(window, nu, support, nfreqs_wdm)
    if spec.window == "gaussian":
        validate_gaussian_subband_grid(
            nfreqs_fourier=nfreqs_fourier,
            nfreqs_wdm=nfreqs_wdm,
            ntimes_wdm=ntimes_wdm,
        )
        _require_integer_span(mmin=mmin, nf_sub_wdm=nf_sub_wdm)
        validate_wdm_span(nfreqs_wdm=nfreqs_wdm, mmin=mmin, nf_sub_wdm=nf_sub_wdm)
        return _gaussian_fourier_span(
            nt=int(ntimes_wdm),
            nf=int(nfreqs_wdm),
            support=spec.support,
            mmin=int(mmin),
            nf_sub_wdm=int(nf_sub_wdm),
        )
    validate_subband_grid(
        nfreqs_fourier=nfreqs_fourier,
        nfreqs_wdm=nfreqs_wdm,
        ntimes_wdm=ntimes_wdm,
    )
    validate_wdm_span(
        nfreqs_wdm=nfreqs_wdm,
        mmin=mmin,
        nf_sub_wdm=nf_sub_wdm,
    )

    half = int(ntimes_wdm) // 2
    mmin = int(mmin)
    mmax = mmin + int(nf_sub_wdm) - 1

    kmin = 0 if mmin == 0 else (mmin - 1) * half
    nyquist = int(nfreqs_wdm) * half
    kmax = nyquist if mmax == int(nfreqs_wdm) else (mmax + 1) * half - 1
    return kmin, kmax - kmin + 1
