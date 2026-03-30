from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..backends import Backend, get_backend

if TYPE_CHECKING:
    from .wdm import WDM


@dataclass(frozen=True)
class TimeSeries:
    """One-dimensional sampled time-domain data."""

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
        return int(self.data.shape[0])

    @property
    def df(self) -> float:
        return 1.0 / (self.n * self.dt)

    @property
    def times(self) -> Any:
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
    """One-dimensional FFT-domain data with spacing metadata."""

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
        return int(self.data.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / (self.n * self.df)

    @property
    def freqs(self) -> Any:
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
