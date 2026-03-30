"""WDM coefficient container with forward/inverse transform integration.

The WDM (Wilson-Daubechies-Meyer) transform maps a time-domain signal of
length ``nt * nf`` into a 2-D grid of real-valued coefficients with ``nt``
time bins and ``nf`` frequency channels.

Edge-channel packing
--------------------
The DC (m=0) and Nyquist (m=nf) channels each carry only real information.
To avoid wasting a full complex column on each, they are packed into a
single complex column:

* ``coeffs[:, 0].real`` — DC channel coefficients
* ``coeffs[:, 0].imag`` — Nyquist channel coefficients
* ``coeffs[:, 1:]``     — interior channels (m = 1 … nf−1), real-valued

Use the :pyattr:`dc_channel` and :pyattr:`nyquist_channel` properties to
access these without knowing the packing layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends import Backend, get_backend
from ..transforms import forward_wdm, frequency_wdm, inverse_wdm
from ..windows import validate_transform_shape, validate_window_parameter
from .series import FrequencySeries, TimeSeries


@dataclass(frozen=True)
class WDM:
    """Packed WDM coefficients together with transform metadata.

    Parameters
    ----------
    coeffs : array, shape (nt, nf)
        Packed coefficient matrix (see module docstring for packing details).
    dt : float
        Sampling interval of the original time-domain signal.
    a : float
        Window roll-off parameter (default 1/3).
    d : float
        Reserved window parameter (default 1, currently unused).
    backend : Backend
        Array backend used for computation.
    """

    coeffs: Any
    dt: float
    a: float = 1.0 / 3.0
    d: float = 1.0
    backend: Backend = field(default_factory=get_backend)

    def __post_init__(self) -> None:
        backend = get_backend(self.backend)
        coeffs = backend.asarray(self.coeffs, dtype=backend.xp.complex128)

        if coeffs.ndim != 2:
            raise ValueError("WDM coeffs must be a two-dimensional array.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        nt, nf = (int(dim) for dim in coeffs.shape)
        validate_transform_shape(nt, nf)
        validate_window_parameter(self.a)

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "coeffs", coeffs)

    def __repr__(self) -> str:
        return f"WDM(nt={self.nt}, nf={self.nf}, dt={self.dt}, a={self.a}, d={self.d})"

    @classmethod
    def from_time_series(
        cls,
        series: TimeSeries,
        *,
        nt: int,
        a: float = 1.0 / 3.0,
        d: float = 1.0,
        backend: str | Backend | None = None,
    ) -> "WDM":
        """Compute the forward WDM transform of a time-domain signal.

        Parameters
        ----------
        series : TimeSeries
            Input signal.  Its length must be divisible by ``nt``.
        nt : int
            Number of WDM time bins (must be even).
        a : float
            Window roll-off parameter.
        d : float
            Reserved (unused).
        backend : str, Backend, or None
            Override backend; defaults to the series' backend.
        """
        resolved_backend = get_backend(backend or series.backend)
        if series.n % nt != 0:
            raise ValueError(f"TimeSeries length {series.n} is not divisible by nt={nt}.")
        nf = series.n // nt
        coeffs = forward_wdm(
            series.data,
            nt=nt,
            nf=nf,
            a=a,
            d=d,
            dt=series.dt,
            backend=resolved_backend,
        )
        return cls(coeffs=coeffs, dt=series.dt, a=a, d=d, backend=resolved_backend)

    @classmethod
    def from_frequency_series(
        cls,
        series: FrequencySeries,
        *,
        nt: int,
        a: float = 1.0 / 3.0,
        d: float = 1.0,
        backend: str | Backend | None = None,
    ) -> "WDM":
        """Compute the forward WDM transform from a frequency-domain signal.

        The signal is first converted to the time domain via inverse FFT,
        then the standard forward WDM transform is applied.

        Parameters
        ----------
        series : FrequencySeries
            Input spectrum.  Its length must be divisible by ``nt``.
        nt : int
            Number of WDM time bins (must be even).
        a : float
            Window roll-off parameter.
        d : float
            Reserved (unused).
        backend : str, Backend, or None
            Override backend; defaults to the series' backend.
        """
        time_series = series.to_time_series(real=True)
        return cls.from_time_series(time_series, nt=nt, a=a, d=d, backend=backend)

    @property
    def nt(self) -> int:
        """Number of WDM time bins."""
        return int(self.coeffs.shape[0])

    @property
    def nf(self) -> int:
        """Number of WDM frequency channels."""
        return int(self.coeffs.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """(nt, nf) shape of the coefficient matrix."""
        return (self.nt, self.nf)

    @property
    def df(self) -> float:
        """Frequency resolution of the original signal."""
        return 1.0 / (self.nt * self.nf * self.dt)

    @property
    def dc_channel(self) -> Any:
        """DC edge-channel coefficients (m = 0), real-valued."""
        xp = self.backend.xp
        return xp.real(self.coeffs[:, 0])

    @property
    def nyquist_channel(self) -> Any:
        """Nyquist edge-channel coefficients (m = nf), real-valued."""
        xp = self.backend.xp
        return xp.imag(self.coeffs[:, 0])

    def to_time_series(self) -> TimeSeries:
        """Reconstruct the time-domain signal via the inverse WDM transform."""
        recovered = inverse_wdm(self.coeffs, a=self.a, d=self.d, dt=self.dt, backend=self.backend)
        return TimeSeries(recovered, dt=self.dt, backend=self.backend)

    def to_frequency_series(self) -> FrequencySeries:
        """Reconstruct the frequency-domain signal from the Gabor atom expansion."""
        return frequency_wdm(self.coeffs, dt=self.dt, a=self.a, d=self.d, backend=self.backend)
