from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .datatypes import WDM, FrequencySeries, TimeSeries

MIN_S = 60.0
HOUR_S = 60.0 * MIN_S
DAY_S = 24.0 * HOUR_S


def _get_pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def _require_scipy() -> Any:
    try:
        from scipy.signal import spectrogram
    except ImportError as exc:
        raise ImportError(
            "This plotting function requires scipy. Reinstall project dependencies "
            "to ensure scipy is available."
        ) from exc
    return spectrogram


def _to_numpy(array: Any) -> np.ndarray:
    return np.asarray(array)


def _batched_axes(
    *,
    ax: Any | None,
    batch_size: int,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Any, np.ndarray]:
    plt = _get_pyplot()
    if ax is None:
        fig, axes = plt.subplots(
            batch_size,
            1,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
        )
        return fig, axes[:, 0]

    axes = np.asarray(ax, dtype=object)
    if axes.ndim == 0:
        if batch_size == 1:
            return axes.item().figure, np.array([axes.item()], dtype=object)
        raise ValueError(
            "Batched plotting requires one axis per batch element when `ax` is provided."
        )

    flat_axes = axes.reshape(-1)
    if flat_axes.size != batch_size:
        raise ValueError(
            f"Batched plotting expected {batch_size} axes, got {flat_axes.size}."
        )
    return flat_axes[0].figure, flat_axes


def _fmt_time_axis(
    t: np.ndarray,
    axis: Any,
    *,
    t0: float | None = None,
    tmax: float | None = None,
) -> None:
    plt = _get_pyplot()
    if t[-1] > DAY_S:
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / DAY_S:.1f}")
        )
        axis.set_xlabel("Time [days]")
    elif t[-1] > HOUR_S:
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / HOUR_S:.1f}")
        )
        axis.set_xlabel("Time [hr]")
    elif t[-1] > MIN_S:
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / MIN_S:.1f}")
        )
        axis.set_xlabel("Time [min]")
    else:
        axis.set_xlabel("Time [s]")

    axis.set_xlim(t[0] if t0 is None else t0, t[-1] if tmax is None else tmax)


def _wdm_time_grid(wdm: WDM) -> np.ndarray:
    return _to_numpy(wdm.time_grid)


def _wdm_freq_grid(wdm: WDM) -> np.ndarray:
    return _to_numpy(wdm.freq_grid)


def _wdm_unpacked_grid(wdm: WDM) -> np.ndarray:
    return _to_numpy(wdm.coeffs)


def plot_time_series(
    series: TimeSeries,
    *,
    ax: Any | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot a time-domain series against its sample times."""
    data = _to_numpy(series.data)
    if data.ndim == 2:
        fig, axes = _batched_axes(ax=ax, batch_size=data.shape[0], sharex=True)
        times = _to_numpy(series.times)
        for idx, axis in enumerate(axes):
            axis.plot(times, data[idx], label=label, **kwargs)
            axis.set_ylabel("Amplitude")
            axis.set_title(f"Batch {idx}")
            _fmt_time_axis(times, axis)
        fig.tight_layout()
        return fig, axes

    plt = _get_pyplot()
    if ax is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = ax.figure, ax
    times = _to_numpy(series.times)
    axis.plot(times, data, label=label, **kwargs)
    axis.set_ylabel("Amplitude")
    _fmt_time_axis(times, axis)
    return fig, axis


def plot_frequency_series(
    series: FrequencySeries,
    *,
    ax: Any | None = None,
    magnitude: bool = True,
    label: str | None = None,
    positive_only: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot a frequency-domain series.

    By default, only non-negative frequencies are shown.
    """
    data = _to_numpy(series.data)
    if data.ndim == 2:
        fig, axes = _batched_axes(ax=ax, batch_size=data.shape[0], sharex=True)
        freqs = _to_numpy(series.freqs)
        if positive_only:
            mask = freqs >= 0.0
            freqs = freqs[mask]
            data = data[:, mask]

        values = np.abs(data) if magnitude else data
        for idx, axis in enumerate(axes):
            axis.plot(freqs, values[idx], label=label, **kwargs)
            axis.set_xlabel("Frequency [Hz]")
            axis.set_ylabel("Magnitude" if magnitude else "Value")
            axis.set_title(f"Batch {idx}")
            if positive_only and len(freqs) > 0:
                axis.set_xlim(0.0, float(freqs[-1]))
        fig.tight_layout()
        return fig, axes

    plt = _get_pyplot()
    if ax is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = ax.figure, ax

    freqs = _to_numpy(series.freqs)
    if positive_only:
        mask = freqs >= 0.0
        freqs = freqs[mask]
        data = data[mask]

    values = np.abs(data) if magnitude else data
    axis.plot(freqs, values, label=label, **kwargs)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Magnitude" if magnitude else "Value")
    if positive_only and len(freqs) > 0:
        axis.set_xlim(0.0, float(freqs[-1]))
    return fig, axis


def plot_periodogram(
    series: FrequencySeries,
    *,
    ax: Any | None = None,
    positive_only: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot the squared spectrum magnitude on log-log axes."""
    data = _to_numpy(series.data)
    if data.ndim == 2:
        fig, axes = _batched_axes(ax=ax, batch_size=data.shape[0], sharex=True, sharey=True)
        freqs = _to_numpy(series.freqs)
        if positive_only:
            mask = freqs > 0.0
            freqs = freqs[mask]
            data = data[:, mask]

        for idx, axis in enumerate(axes):
            axis.loglog(freqs, np.abs(data[idx]) ** 2, **kwargs)
            axis.set_xlabel("Frequency [Hz]")
            axis.set_ylabel("Periodogram")
            axis.set_title(f"Batch {idx}")
        fig.tight_layout()
        return fig, axes

    plt = _get_pyplot()
    if ax is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = ax.figure, ax

    freqs = _to_numpy(series.freqs)
    if positive_only:
        mask = freqs > 0.0
        freqs = freqs[mask]
        data = data[mask]

    axis.loglog(freqs, np.abs(data) ** 2, **kwargs)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Periodogram")
    return fig, axis


def plot_spectrogram(
    series: TimeSeries,
    *,
    ax: Any | None = None,
    spec_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Compute and plot a scipy spectrogram for a time-domain series."""
    spectrogram = _require_scipy()
    data = _to_numpy(series.data)
    if data.ndim == 2:
        fig, axes = _batched_axes(ax=ax, batch_size=data.shape[0], sharex=True, sharey=True)
        spec_kwargs = {} if spec_kwargs is None else dict(spec_kwargs)
        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        plot_kwargs.setdefault("cmap", "Reds")
        fs = 1.0 / series.dt

        for idx, axis in enumerate(axes):
            freqs, times, sxx = spectrogram(data[idx], fs=fs, **spec_kwargs)
            mesh = axis.pcolormesh(times, freqs, sxx, shading="nearest", **plot_kwargs)
            _fmt_time_axis(times, axis)
            axis.set_ylabel("Frequency [Hz]")
            axis.set_ylim(top=fs / 2.0)
            axis.set_title(f"Batch {idx}")
            colorbar = fig.colorbar(mesh, ax=axis)
            colorbar.set_label("Spectrogram Amplitude")
        fig.tight_layout()
        return fig, axes

    plt = _get_pyplot()
    if ax is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = ax.figure, ax

    spec_kwargs = {} if spec_kwargs is None else dict(spec_kwargs)
    plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
    plot_kwargs.setdefault("cmap", "Reds")

    fs = 1.0 / series.dt
    freqs, times, sxx = spectrogram(data, fs=fs, **spec_kwargs)
    mesh = axis.pcolormesh(times, freqs, sxx, shading="nearest", **plot_kwargs)
    _fmt_time_axis(times, axis)
    axis.set_ylabel("Frequency [Hz]")
    axis.set_ylim(top=fs / 2.0)
    colorbar = plt.colorbar(mesh, ax=axis)
    colorbar.set_label("Spectrogram Amplitude")
    return fig, axis


def plot_wdm_grid(
    wdm: WDM,
    *,
    ax: Any | None = None,
    zscale: str = "linear",
    freq_scale: str = "linear",
    absolute: bool = True,
    freq_range: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    cmap: str | Any | None = None,
    norm: Any | None = None,
    cbar_label: str | None = None,
    nan_color: str = "black",
    detailed_axes: bool = False,
    show_gridinfo: bool = True,
    txtbox_kwargs: dict[str, Any] | None = None,
    whiten_by: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Render the WDM coefficient grid as an image over time and frequency."""
    plt = _get_pyplot()
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    coeffs = _wdm_unpacked_grid(wdm)
    if coeffs.ndim == 3:
        fig, axes = _batched_axes(ax=ax, batch_size=coeffs.shape[0], sharex=True, sharey=True)
        time_grid = _wdm_time_grid(wdm)
        freq_grid = _wdm_freq_grid(wdm)

        for idx, axis in enumerate(axes):
            z = coeffs[idx].T
            whiten_slice = None if whiten_by is None else np.asarray(whiten_by)[idx]
            if whiten_slice is not None:
                z = z / whiten_slice
            if absolute:
                z = np.abs(z)
            else:
                z = np.real(z)

            local_norm = norm
            if local_norm is None:
                try:
                    if np.all(np.isnan(z)):
                        raise ValueError("All WDM data is NaN.")
                    if zscale == "log":
                        positive = z[z > 0]
                        local_norm = LogNorm(vmin=np.nanmin(positive), vmax=np.nanmax(positive))
                    elif not absolute:
                        local_norm = TwoSlopeNorm(
                            vmin=float(np.nanmin(z)),
                            vcenter=0.0,
                            vmax=float(np.nanmax(z)),
                        )
                except Exception as exc:
                    warnings.warn(
                        f"Falling back to default linear normalization for WDM plot: {exc}",
                        stacklevel=2,
                    )
                    local_norm = None

            local_cmap = "viridis" if absolute else "bwr" if cmap is None else cmap
            cmap_obj = plt.get_cmap(local_cmap).copy()
            cmap_obj.set_bad(color=nan_color)

            image = axis.imshow(
                z,
                aspect="auto",
                extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
                origin="lower",
                cmap=cmap_obj,
                norm=local_norm,
                interpolation="nearest",
                **kwargs,
            )

            if show_colorbar:
                colorbar = fig.colorbar(image, ax=axis)
                if cbar_label is None:
                    local_cbar_label = "Absolute WDM Amplitude" if absolute else "WDM Amplitude"
                else:
                    local_cbar_label = cbar_label
                colorbar.set_label(local_cbar_label)

            axis.set_yscale(freq_scale)
            axis.set_ylabel("Frequency [Hz]")
            _fmt_time_axis(time_grid, axis)
            axis.set_title(f"Batch {idx}")

            local_freq_range = freq_range or (float(freq_grid[0]), float(freq_grid[-1]))
            axis.set_ylim(local_freq_range)

            if detailed_axes:
                axis.set_xlabel(rf"Time bins [$\Delta T$={wdm.delta_t:.4g}s, Nt={wdm.nt}]")
                axis.set_ylabel(
                    rf"Frequency bins [$\Delta F$={wdm.delta_f:.4g}Hz, Nf={wdm.nf + 1}]"
                )

            label = kwargs.get("label", "")
            info = f"{wdm.nf + 1}x{wdm.nt}" if show_gridinfo else ""
            text = f"{label}\n{info}" if label and info else (label or info)
            if text:
                txtbox_local = {} if txtbox_kwargs is None else dict(txtbox_kwargs)
                txtbox_local.setdefault("boxstyle", "round")
                txtbox_local.setdefault("facecolor", "white")
                txtbox_local.setdefault("alpha", 0.2)
                axis.text(
                    0.05,
                    0.95,
                    text,
                    transform=axis.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=txtbox_local,
                )

        fig.tight_layout()
        return fig, axes

    if ax is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = ax.figure, ax

    time_grid = _wdm_time_grid(wdm)
    freq_grid = _wdm_freq_grid(wdm)
    z = coeffs.T
    if whiten_by is not None:
        z = z / whiten_by
    if absolute:
        z = np.abs(z)
    else:
        z = np.real(z)

    if norm is None:
        try:
            if np.all(np.isnan(z)):
                raise ValueError("All WDM data is NaN.")
            if zscale == "log":
                positive = z[z > 0]
                norm = LogNorm(vmin=np.nanmin(positive), vmax=np.nanmax(positive))
            elif not absolute:
                norm = TwoSlopeNorm(
                    vmin=float(np.nanmin(z)),
                    vcenter=0.0,
                    vmax=float(np.nanmax(z)),
                )
        except Exception as exc:
            warnings.warn(
                f"Falling back to default linear normalization for WDM plot: {exc}",
                stacklevel=2,
            )
            norm = None

    if cmap is None:
        cmap = "viridis" if absolute else "bwr"
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    image = axis.imshow(
        z,
        aspect="auto",
        extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
        origin="lower",
        cmap=cmap_obj,
        norm=norm,
        interpolation="nearest",
        **kwargs,
    )

    if show_colorbar:
        colorbar = fig.colorbar(image, ax=axis)
        if cbar_label is None:
            cbar_label = "Absolute WDM Amplitude" if absolute else "WDM Amplitude"
        colorbar.set_label(cbar_label)

    axis.set_yscale(freq_scale)
    axis.set_ylabel("Frequency [Hz]")
    _fmt_time_axis(time_grid, axis)

    freq_range = freq_range or (float(freq_grid[0]), float(freq_grid[-1]))
    axis.set_ylim(freq_range)

    if detailed_axes:
        axis.set_xlabel(rf"Time bins [$\Delta T$={wdm.delta_t:.4g}s, Nt={wdm.nt}]")
        axis.set_ylabel(
            rf"Frequency bins [$\Delta F$={wdm.delta_f:.4g}Hz, Nf={wdm.nf + 1}]"
        )

    label = kwargs.get("label", "")
    info = f"{wdm.nf + 1}x{wdm.nt}" if show_gridinfo else ""
    text = f"{label}\n{info}" if label and info else (label or info)
    if text:
        txtbox_kwargs = {} if txtbox_kwargs is None else dict(txtbox_kwargs)
        txtbox_kwargs.setdefault("boxstyle", "round")
        txtbox_kwargs.setdefault("facecolor", "white")
        txtbox_kwargs.setdefault("alpha", 0.2)
        axis.text(
            0.05,
            0.95,
            text,
            transform=axis.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=txtbox_kwargs,
        )

    fig.tight_layout()
    return fig, axis
