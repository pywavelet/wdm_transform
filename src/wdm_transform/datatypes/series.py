from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..backends import Backend, get_backend

if TYPE_CHECKING:
    from .wdm import WDM


@dataclass(frozen=True)
class TimeSeries:
    """One-dimensional sampled time-domain data.

    Parameters
    ----------
    data : array_like
        Sample values on a uniform time grid.
    dt : float
        Time spacing between adjacent samples.
    backend : Backend
        Array backend used for computation.
    """

    data: Any
    dt: float
    backend: Backend = field(default_factory=get_backend)

    def __post_init__(self) -> None:
        backend = get_backend(self.backend)
        data = backend.asarray(self.data)

        if data.ndim != 1:
            raise ValueError("TimeSeries data must be one-dimensional.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "data", data)

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.data.shape[0])

    @property
    def df(self) -> float:
        """Fourier frequency spacing implied by the sample cadence."""
        return 1.0 / (self.n * self.dt)

    @property
    def duration(self) -> float:
        """Total signal duration ``n * dt``.

        This follows the discrete-Fourier convention used throughout the
        package, not ``times[-1] - times[0]``.
        """
        return self.n * self.dt

    @property
    def times(self) -> Any:
        """Sample-time grid ``arange(n) * dt``."""
        return self.backend.xp.arange(self.n) * self.dt

    def to_frequency_series(self) -> "FrequencySeries":
        transformed = self.backend.fft.fft(self.data)
        return FrequencySeries(transformed, df=self.df, backend=self.backend)

    def to_wdm(
        self,
        *,
        nt: int,
        a: float = 1.0 / 3.0,
        d: float = 1.0,
        backend: str | Backend | None = None,
    ) -> "WDM":
        """Compute the WDM transform of this time series."""
        from .wdm import WDM

        return WDM.from_time_series(self, nt=nt, a=a, d=d, backend=backend)

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Plot the time-domain samples using the shared plotting helper."""
        from ..plotting import plot_time_series

        return plot_time_series(self, **kwargs)


@dataclass(frozen=True)
class FrequencySeries:
    """One-dimensional FFT-domain data with spacing metadata.

    Parameters
    ----------
    data : array_like
        Spectrum samples on the discrete Fourier grid.
    df : float
        Frequency spacing between adjacent Fourier bins.
    backend : Backend
        Array backend used for computation.
    """

    data: Any
    df: float
    backend: Backend = field(default_factory=get_backend)

    def __post_init__(self) -> None:
        backend = get_backend(self.backend)
        data = backend.asarray(self.data)

        if data.ndim != 1:
            raise ValueError("FrequencySeries data must be one-dimensional.")
        if self.df <= 0:
            raise ValueError("df must be positive.")

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "data", data)

    @property
    def n(self) -> int:
        """Number of Fourier bins."""
        return int(self.data.shape[0])

    @property
    def dt(self) -> float:
        """Time spacing implied by the discrete Fourier grid."""
        return 1.0 / (self.n * self.df)

    @property
    def duration(self) -> float:
        """Total signal duration ``n * dt`` represented by the spectrum."""
        return self.n * self.dt

    @property
    def freqs(self) -> Any:
        """Discrete Fourier frequency grid."""
        return self.backend.fft.fftfreq(self.n, d=self.dt)

    def to_time_series(self, *, real: bool = False) -> TimeSeries:
        recovered = self.backend.fft.ifft(self.data)
        if real:
            recovered = self.backend.xp.real(recovered)
        return TimeSeries(recovered, dt=self.dt, backend=self.backend)

    def to_wdm(
        self,
        *,
        nt: int,
        a: float = 1.0 / 3.0,
        d: float = 1.0,
        backend: str | Backend | None = None,
    ) -> "WDM":
        """Compute the WDM transform of this frequency-domain series."""
        from .wdm import WDM

        return WDM.from_frequency_series(self, nt=nt, a=a, d=d, backend=backend)

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Plot the spectrum using the shared plotting helper."""
        from ..plotting import plot_frequency_series

        return plot_frequency_series(self, **kwargs)
