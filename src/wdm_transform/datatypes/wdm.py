"""WDM coefficient container with forward/inverse transform integration.

The WDM (Wilson-Daubechies-Meyer) transform maps a time-domain signal of
length N = nt * nf into a 2-D grid of real-valued coefficients with ``nt``
time bins and ``nf + 1`` frequency channels (m = 0, 1, …, nf).

Coefficient layout
------------------
The coefficient matrix has shape ``(nt, nf + 1)``:

* ``coeffs[:, 0]``    — DC channel      (m = 0)
* ``coeffs[:, 1:nf]`` — interior channels (m = 1 … nf−1)
* ``coeffs[:, nf]``   — Nyquist channel (m = nf)

All entries are real-valued (float64).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends import Backend, get_backend
from ..transforms import from_freq_to_wdm, from_time_to_wdm, from_wdm_to_freq, from_wdm_to_time
from ..windows import validate_transform_shape, validate_window_parameter
from .series import FrequencySeries, TimeSeries


@dataclass(frozen=True)
class WDM:
    """Real-valued WDM coefficients together with transform metadata.

    Parameters
    ----------
    coeffs : array, shape (nt, nf + 1)
        Real-valued coefficient matrix.  Column m corresponds to
        frequency channel m (0 ≤ m ≤ nf).
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
        coeffs = backend.asarray(self.coeffs, dtype=backend.xp.float64)

        if coeffs.ndim != 2:
            raise ValueError("WDM coeffs must be a two-dimensional array.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        nt, ncols = (int(dim) for dim in coeffs.shape)
        nf = ncols - 1
        validate_transform_shape(nt, nf)
        validate_window_parameter(self.a)

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "coeffs", coeffs)

    def __repr__(self) -> str:
        return (
            "WDM("
            f"nt={self.nt}, nf={self.nf}, n={self.n}, "
            f"dt={self.dt}, df={self.df}, fs={self.fs}, nyquist={self.nyquist}, "
            f"delta_t={self.delta_t}, delta_f={self.delta_f}, "
            f"duration={self.duration}, a={self.a}, d={self.d}"
            ")"
        )

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
            raise ValueError(
                f"TimeSeries length {series.n} is not divisible by nt={nt}."
            )
        nf = series.n // nt
        coeffs = from_time_to_wdm(
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

        Any non-Hermitian component of ``series`` is discarded so the result
        matches applying the WDM transform to ``real(ifft(series.data))``.

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
        resolved_backend = get_backend(backend or series.backend)
        if series.n % nt != 0:
            raise ValueError(
                f"FrequencySeries length {series.n} is not divisible by nt={nt}."
            )
        nf = series.n // nt
        coeffs = from_freq_to_wdm(
            series.data,
            nt=nt,
            nf=nf,
            a=a,
            d=d,
            dt=series.dt,
            backend=resolved_backend,
        )
        return cls(coeffs=coeffs, dt=series.dt, a=a, d=d, backend=resolved_backend)

    @property
    def nt(self) -> int:
        """Number of WDM time bins."""
        return int(self.coeffs.shape[0])

    @property
    def nf(self) -> int:
        """Number of interior frequency channels.

        The total number of frequency channels is ``nf + 1``
        (m = 0, 1, …, nf), so ``coeffs.shape[1] == nf + 1``.
        """
        return int(self.coeffs.shape[1]) - 1

    @property
    def shape(self) -> tuple[int, int]:
        """(nt, nf + 1) shape of the coefficient matrix."""
        return (self.nt, self.nf + 1)

    @property
    def n(self) -> int:
        """Total number of time-domain samples represented by this transform."""
        return self.nt * self.nf

    @property
    def df(self) -> float:
        """Fourier-bin spacing of the underlying original signal.

        This is the discrete-Fourier spacing implied by the original sample
        cadence. For the WDM frequency-grid spacing, use :attr:`delta_f`.
        """
        return 1.0 / (self.nt * self.nf * self.dt)

    @property
    def fs(self) -> float:
        """Sampling frequency of the underlying original time series."""
        return 1.0 / self.dt

    @property
    def nyquist(self) -> float:
        """Nyquist frequency of the underlying original signal.

        This is also the frequency of the highest WDM channel,
        ``freq_grid[-1]``.
        """
        return 0.5 * self.fs

    @property
    def delta_t(self) -> float:
        """Spacing of the WDM time grid.

        Each WDM time bin spans ``nf * dt`` in the original sampling.
        For the underlying original sample spacing, use :attr:`dt`.
        """
        return self.nf * self.dt

    @property
    def delta_f(self) -> float:
        """Spacing of the WDM frequency grid.

        This is the spacing between adjacent WDM channels. For the underlying
        Fourier-bin spacing of the original signal, use :attr:`df`.
        """
        return 1.0 / (2.0 * self.delta_t)

    @property
    def duration(self) -> float:
        """Total signal duration ``nt * delta_t`` represented by this transform.

        Equivalently, this is ``nt * nf * dt``. This convention matches the
        transform construction and is not the same as ``time_grid[-1] - time_grid[0]``.
        """
        return self.nt * self.delta_t

    @property
    def time_grid(self) -> Any:
        """WDM time-grid coordinates ``arange(nt) * delta_t``."""
        return self.backend.xp.arange(self.nt) * self.delta_t

    @property
    def freq_grid(self) -> Any:
        """WDM frequency-grid coordinates ``arange(nf + 1) * delta_f``."""
        return self.backend.xp.arange(self.nf + 1) * self.delta_f

    @property
    def dc_channel(self) -> Any:
        """DC edge-channel coefficients (m = 0)."""
        return self.coeffs[:, 0]

    @property
    def nyquist_channel(self) -> Any:
        """Nyquist edge-channel coefficients (m = nf)."""
        return self.coeffs[:, self.nf]

    def to_time_series(self) -> TimeSeries:
        """Reconstruct the time-domain signal via the inverse WDM transform."""
        recovered = from_wdm_to_time(
            self.coeffs,
            a=self.a,
            d=self.d,
            dt=self.dt,
            backend=self.backend,
        )
        return TimeSeries(recovered, dt=self.dt, backend=self.backend)

    def to_frequency_series(self) -> FrequencySeries:
        """Reconstruct the frequency-domain signal from the Gabor atom expansion."""
        recovered = from_wdm_to_freq(
            self.coeffs,
            dt=self.dt,
            a=self.a,
            d=self.d,
            backend=self.backend,
        )
        return FrequencySeries(recovered, df=self.df, backend=self.backend)

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Plot the WDM coefficient grid using the shared plotting helper."""
        from ..plotting import plot_wdm_grid

        return plot_wdm_grid(self, **kwargs)
