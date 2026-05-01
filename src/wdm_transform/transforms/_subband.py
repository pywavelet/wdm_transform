"""Shared helpers for low-level WDM sub-band transforms."""

from __future__ import annotations

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
        raise ValueError(
            f"mmin must satisfy 0 <= mmin <= nfreqs_wdm={nfreqs_wdm}."
        )
    if mmin + nf_sub_wdm > nfreqs_wdm + 1:
        raise ValueError(
            "WDM span exceeds the full grid: "
            f"mmin + nf_sub_wdm = {mmin + nf_sub_wdm}, nfreqs_wdm + 1 = {nfreqs_wdm + 1}."
        )


def wdm_span_from_fourier_span(
    *,
    nfreqs_fourier: int,
    nfreqs_wdm: int,
    ntimes_wdm: int,
    kmin: int,
    lendata: int,
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

    Returns:
        A tuple ``(mmin, nf_sub_wdm)``. ``mmin`` is the first touched WDM
        frequency-channel index, and ``nf_sub_wdm`` is the number of adjacent
        WDM channels touched by the Fourier span.
    """
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

    Returns:
        A tuple ``(kmin, lendata)``. ``kmin`` is the first touched one-sided
        Fourier-bin index, and ``lendata`` is the number of touched Fourier
        bins.
    """
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
