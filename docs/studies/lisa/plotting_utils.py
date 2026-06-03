"""Plotting helpers for the LISA galactic-binary study."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lisa_common import save_figure


def _rfft_periodogram(coeffs: np.ndarray, freqs: np.ndarray, dt: float) -> np.ndarray:
    psd = np.zeros_like(freqs, dtype=float)
    pos = freqs > 0.0
    if pos.sum() >= 2:
        df = float(freqs[pos][1] - freqs[pos][0])
        psd[pos] = 2.0 * df * dt**2 * np.abs(coeffs[pos]) ** 2
    return psd


def _mark_zoom_region(parent_ax, zoom_ax, xlim, ylim, *, color: str) -> None:
    from matplotlib.patches import ConnectionPatch, Rectangle

    rect = Rectangle(
        (xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
        fill=False, edgecolor=color, linewidth=1.0, alpha=0.8,
    )
    parent_ax.add_patch(rect)
    for y_parent, y_zoom in ((ylim[0], 0.0), (ylim[1], 1.0)):
        connector = ConnectionPatch(
            xyA=(xlim[1], y_parent), coordsA=parent_ax.transData,
            xyB=(0.0, y_zoom), coordsB=zoom_ax.transAxes,
            axesA=parent_ax, axesB=zoom_ax,
            color=color, linewidth=0.8, alpha=0.55,
        )
        connector.set_clip_on(False)
        zoom_ax.add_artist(connector)


def plot_data_overview(
    run_dir: Path,
    *,
    dt: float,
    freqs: np.ndarray,
    psd_A: np.ndarray,
    source_Af: np.ndarray,
    data_At: np.ndarray,
    f0: float,
    nt_spectrogram: int,
    a_wdm: float,
    d_wdm: float,
) -> None:
    """Save the A-channel PSD and WDM overview figure."""
    import matplotlib.pyplot as plt

    from wdm_transform.transforms import from_time_to_wdm

    fig = plt.figure(figsize=(11, 9), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 3,
        left=0.08,
        right=0.96,
        bottom=0.08,
        top=0.94,
        width_ratios=(1.0, 0.27, 0.035),
        height_ratios=(1.0, 1.0),
        wspace=0.08,
        hspace=0.28,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_colorbar_placeholder = fig.add_subplot(grid[0, 2])
    ax_bot = fig.add_subplot(grid[1, 0])
    ax_wdm_zoom = fig.add_subplot(grid[1, 1])
    ax_colorbar = fig.add_subplot(grid[1, 2])
    ax_colorbar_placeholder.axis("off")

    data_Af = np.fft.rfft(data_At)
    data_psd = _rfft_periodogram(data_Af, freqs, dt)
    source_psd = _rfft_periodogram(source_Af, freqs, dt)
    ax_top.loglog(freqs[1:], data_psd[1:],
                  color="0.75", lw=0.7, alpha=0.8, zorder=2, label="A data")
    ax_top.loglog(freqs[1:], psd_A[1:],
                  color="black", lw=1.4, zorder=3, label="Instrument PSD")
    ax_top.set_xlim(1e-4, 3e-3)
    band = (freqs >= 1e-4) & (freqs <= 3e-3)
    main_positive = np.concatenate([
        data_psd[band][data_psd[band] > 0.0],
        source_psd[band][source_psd[band] > 0.0],
        psd_A[band][psd_A[band] > 0.0],
    ])
    if main_positive.size:
        ax_top.set_ylim(np.percentile(main_positive, 0.2) / 3.0,
                        np.percentile(main_positive, 99.9) * 3.0)
    ax_top.set_xlabel("Frequency (Hz)")
    ax_top.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]")
    ax_top.set_title("Channel A - frequency domain")
    ax_top.grid(True, which="both", alpha=0.25)
    ax_top.text(0.01, 0.96, "(a)", transform=ax_top.transAxes, ha="left", va="top")
    ax_top.legend(loc="upper right")

    zoom_half_width = max(2.0e-5, 40.0 / (dt * (len(freqs) - 1) * 2.0))
    zoom = (freqs >= f0 - zoom_half_width) & (freqs <= f0 + zoom_half_width)
    if np.count_nonzero(zoom) >= 3:
        ax_zoom.semilogy(freqs[zoom] * 1e3, source_psd[zoom],
                         color="C1", lw=1.8, alpha=0.9, zorder=1)
        ax_zoom.semilogy(freqs[zoom] * 1e3, data_psd[zoom],
                         color="0.80", lw=0.7, alpha=0.9, zorder=2)
        ax_zoom.semilogy(freqs[zoom] * 1e3, psd_A[zoom],
                         color="black", lw=1.0, zorder=3)
        positive_zoom = np.concatenate([
            data_psd[zoom][data_psd[zoom] > 0.0],
            source_psd[zoom][source_psd[zoom] > 0.0],
            psd_A[zoom][psd_A[zoom] > 0.0],
        ])
        if positive_zoom.size:
            ax_zoom.set_ylim(np.percentile(positive_zoom, 1), np.percentile(positive_zoom, 99.5) * 3.0)
        ax_zoom.set_xlabel("Frequency (mHz)", fontsize=8)
        ax_zoom.set_title("f0 zoom", fontsize=10)
        ax_zoom.tick_params(axis="x", labelsize=8)
        ax_zoom.tick_params(axis="y", labelleft=False)
        ax_zoom.grid(True, which="both", alpha=0.15)
        _mark_zoom_region(
            ax_top, ax_zoom,
            (float(np.min(freqs[zoom])), float(np.max(freqs[zoom]))),
            ax_top.get_ylim(),
            color="black",
        )
    else:
        ax_zoom.axis("off")

    nt = nt_spectrogram
    n_keep = (len(data_At) // (2 * nt)) * (2 * nt)
    nf = n_keep // nt
    wdm_freqs = np.linspace(0.0, 0.5 / dt, nf + 1)
    coeffs = np.asarray(
        from_time_to_wdm(data_At[:n_keep], nt=nt, nf=nf, a=a_wdm, d=d_wdm, dt=dt, backend="numpy")
    )[0]
    power = coeffs[:, 1:-1] ** 2
    t_axis = np.linspace(0.0, n_keep * dt, nt) / 86400.0
    f_axis = wdm_freqs[1:-1] * 1e3
    floor = np.percentile(power[power > 0], 5) if np.any(power > 0) else 1e-60
    vmax = np.percentile(power[np.isfinite(power)], 99.7) if np.any(np.isfinite(power)) else 1.0
    mesh = ax_bot.pcolormesh(
        t_axis, f_axis, np.log10(np.maximum(power, floor)).T, shading="auto",
        cmap="magma", vmin=np.log10(floor), vmax=np.log10(max(vmax, floor * 10.0))
    )
    ax_bot.set_ylim(1e-4 * 1e3, 3e-3 * 1e3)
    ax_bot.set_xlabel("Time (days)")
    ax_bot.set_ylabel("Frequency (mHz)")
    ax_bot.set_title(f"Channel A - WDM time-frequency power (nt={nt})")
    ax_bot.text(0.01, 0.96, "(b)", transform=ax_bot.transAxes, ha="left", va="top", color="white")
    fig.colorbar(mesh, cax=ax_colorbar, label=r"$\log_{10}$ power")

    wdm_zoom = (f_axis >= f0 * 1e3 - 0.06) & (f_axis <= f0 * 1e3 + 0.06)
    if np.count_nonzero(wdm_zoom) >= 3:
        ax_wdm_zoom.pcolormesh(
            t_axis, f_axis[wdm_zoom], np.log10(np.maximum(power[:, wdm_zoom], floor)).T,
            shading="auto", cmap="magma", vmin=np.log10(floor),
            vmax=np.log10(max(vmax, floor * 10.0))
        )
        zoom_ticks = np.linspace(f_axis[wdm_zoom][0], f_axis[wdm_zoom][-1], 3)
        ax_wdm_zoom.set_yticks(zoom_ticks)
        ax_wdm_zoom.set_xlabel("Time (days)", fontsize=8)
        ax_wdm_zoom.set_title("f0 zoom", fontsize=10)
        ax_wdm_zoom.tick_params(axis="x", labelsize=8)
        ax_wdm_zoom.tick_params(axis="y", labelleft=False)
        _mark_zoom_region(
            ax_bot, ax_wdm_zoom,
            (float(t_axis[0]), float(t_axis[-1])),
            (float(f_axis[wdm_zoom][0]), float(f_axis[wdm_zoom][-1])),
            color="black",
        )
    else:
        ax_wdm_zoom.axis("off")
    save_figure(fig, run_dir, "data_overview")
    print(f"[data] saved data overview figure to {run_dir / 'data_overview.png'}")


def _log10_transform_dataset(ds, log10_vars: frozenset[str]):
    """Return a copy of *ds* with log-scale variables replaced by log10 values."""
    ds = ds.copy()
    for name in log10_vars:
        if name in ds:
            ds[name] = np.log10(ds[name])
    return ds


def plot_marginal_comparison(
    freq_ds,
    wdm_ds,
    truth: np.ndarray,
    run_dir: Path,
    *,
    posterior_vars: tuple[str, ...],
    log10_vars: frozenset[str],
) -> None:
    """Overlay frequency- and WDM-domain marginal posteriors with truth lines."""
    import arviz_plots as azp
    import matplotlib.pyplot as plt
    import xarray as xr

    azp.plot_dist(
        {
            "Frequency": xr.DataTree.from_dict({"posterior": _log10_transform_dataset(freq_ds, log10_vars)}),
            "WDM": xr.DataTree.from_dict({"posterior": _log10_transform_dataset(wdm_ds, log10_vars)}),
        },
        var_names=list(posterior_vars),
        backend="matplotlib",
        point_estimate=None,
    )
    fig = plt.gcf()
    for ax, true_val in zip(fig.axes[: len(posterior_vars)], truth, strict=False):
        ax.axvline(true_val, color="black", linestyle="--", linewidth=1.5, label="Truth")
    save_figure(fig, run_dir, "posterior_comparison")
    print(f"[compare] saved marginal comparison to {run_dir / 'posterior_comparison.png'}")
