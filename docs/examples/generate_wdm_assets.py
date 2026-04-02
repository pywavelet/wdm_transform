"""Generate offline WDM guide assets for the docs.

This script is intentionally not executed during the docs build. It generates
static assets once and writes them into ``docs/_static``.

Outputs:
- wdm_freq_packetization.gif
- wdm_basis_atom_shift.gif
- wdm_channel_shift.gif
- wdm_phi_parameter_comparison.png
- wdm_shifted_windows.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LogNorm
from scipy.signal import chirp

from wdm_transform import TimeSeries, get_backend
from wdm_transform.windows import cnm, gnmf, phi_unit, phi_window


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"


def _ensure_outdir() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)


def _save_gif(anim: animation.FuncAnimation, name: str, fps: int = 8) -> None:
    _ensure_outdir()
    out = STATIC / name
    anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=110)
    print(f"Wrote {out}")


def _save_figure(fig: plt.Figure, name: str) -> None:
    _ensure_outdir()
    out = STATIC / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def _chirp_signal(n_total: int, dt: float) -> np.ndarray:
    times = np.arange(n_total) * dt
    return chirp(times, f0=6.0, f1=110.0, t1=times[-1], method="quadratic")


def _positive_spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.fftfreq(len(values), d=dt)
    positive = freqs >= 0.0
    return freqs[positive], np.fft.fft(values)[positive]


def _window_response(
    freqs: np.ndarray,
    *,
    m: int,
    nf: int,
    dt: float,
    a: float,
    d: float,
) -> np.ndarray:
    backend = get_backend()
    df_band = 1.0 / (2.0 * nf * dt)
    scaled = freqs / df_band
    if m == 0:
        return np.asarray(phi_unit(backend, scaled, a, d))
    if m == nf:
        return np.asarray(
            phi_unit(backend, scaled + m, a, d)
            + phi_unit(backend, scaled - m, a, d)
        )
    return np.asarray(
        (
            phi_unit(backend, scaled + m, a, d)
            + phi_unit(backend, scaled - m, a, d)
        )
        / np.sqrt(2.0)
    )


def _forward_columns(
    data: np.ndarray,
    *,
    nt: int,
    nf: int,
    dt: float,
    a: float,
    d: float,
) -> np.ndarray:
    backend = get_backend()
    xp = backend.xp
    narr = xp.arange(nt)
    half = nt // 2
    n_total = nt * nf
    x_fft = np.fft.fft(data.astype(np.complex128))
    window = np.asarray(phi_window(backend, nt, nf, dt, a, d), dtype=np.complex128)
    coeffs = np.zeros((nt, nf + 1), dtype=float)

    block = x_fft[1:half] * window[1:half]
    larr = np.arange(1, half)
    coeffs[:, 0] = np.real(
        np.sum(
            np.exp(4j * np.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[0] * window[0] / 2.0
    ) / (nt * nf)

    for m in range(1, nf):
        phase = np.conjugate(np.asarray(cnm(backend, narr, m)))
        block = np.concatenate(
            [x_fft[m * half:(m + 1) * half], x_fft[(m - 1) * half:m * half]]
        )
        packet = np.fft.ifft(block * window)
        coeffs[:, m] = (np.sqrt(2.0) / nf) * np.real(phase * packet)

    block = x_fft[n_total // 2 - half:n_total // 2] * window[-half:]
    larr = np.arange(n_total // 2 - half, n_total // 2)
    coeffs[:, nf] = np.real(
        np.sum(
            np.exp(4j * np.pi * larr[None, :] * narr[:, None] / nt) * block[None, :],
            axis=1,
        )
        + x_fft[n_total // 2] * window[0] / 2.0
    ) / (nt * nf)

    return coeffs


def make_phi_parameter_comparison_png() -> None:
    d = 1.0
    a_values = [0.20, 0.25, 1.0 / 3.0, 0.40]
    colors = ["#cf222e", "#fb8c00", "#0969da", "#8250df"]
    backend = get_backend()

    scaled_freqs = np.linspace(-1.2, 1.2, 1200)
    # Use a dense normalized-frequency grid to build a smooth inverse-Fourier
    # approximation of phi itself, rather than the short sampled transform
    # window. This makes the time-domain tradeoff with `a` visually cleaner.
    n_dense = 16384
    dx = 8.0 / n_dense
    dense_freqs = (np.arange(n_dense) - n_dense // 2) * dx
    tau = np.fft.fftshift(np.fft.fftfreq(n_dense, d=dx))
    time_mask = np.abs(tau) <= 24.0

    fig, (ax_freq, ax_time) = plt.subplots(1, 2, figsize=(10.2, 4.3))

    for a, color in zip(a_values, colors):
        phi_f = np.asarray(phi_unit(backend, scaled_freqs, a, d))
        dense_phi = np.asarray(phi_unit(backend, dense_freqs, a, d))
        kernel = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(dense_phi)))
        kernel_abs = np.abs(kernel)
        kernel_abs /= np.max(kernel_abs)

        ax_freq.plot(
            scaled_freqs,
            phi_f,
            color=color,
            linewidth=2,
            label=fr"$a={a:.2f}$",
        )
        ax_time.plot(
            tau[time_mask],
            kernel_abs[time_mask],
            color=color,
            linewidth=2,
            label=fr"$a={a:.2f}$",
        )

    ax_freq.set_title(r"Frequency-domain window $\phi(f / \Delta F)$")
    ax_freq.set_xlabel(r"Normalized frequency $f / \Delta F$")
    ax_freq.set_ylabel("Window amplitude")
    ax_freq.set_xlim(-1.1, 1.1)
    ax_freq.set_ylim(-0.02, 1.05)
    ax_freq.grid(True, alpha=0.2)

    ax_time.set_title("Time-domain shape implied by the same window")
    ax_time.set_xlabel("Normalized time coordinate")
    ax_time.set_ylabel("Normalized amplitude")
    ax_time.set_yscale("log")
    ax_time.set_xlim(-24.0, 24.0)
    ax_time.grid(True, alpha=0.2)

    handles, labels = ax_freq.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, "wdm_phi_parameter_comparison.png")
    plt.close(fig)


def make_shifted_windows_png() -> None:
    nt = 32
    nf = 32
    dt = 1.0 / 256.0
    a = 1.0 / 3.0
    d = 1.0

    signal = _chirp_signal(nt * nf, dt)
    freqs, spectrum = _positive_spectrum(signal, dt)
    channels = [0, 4, 8, 16, nf]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(freqs, np.abs(spectrum), color="#6e7781", alpha=0.35, label="|FFT(data)|")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Spectrum magnitude")
    ax1.set_title("Shifted WDM windows over the positive-frequency spectrum")

    ax2 = ax1.twinx()
    colors = ["#cf222e", "#fb8c00", "#1a7f37", "#0969da", "#8250df"]
    for m, color in zip(channels, colors):
        ax2.plot(
            freqs,
            _window_response(freqs, m=m, nf=nf, dt=dt, a=a, d=d),
            color=color,
            linewidth=2,
            label=f"channel m={m}",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_ylabel("Window amplitude")
    ax2.set_ylim(0.0, 1.05)
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, "wdm_shifted_windows.png")
    plt.close(fig)


def make_freq_packetization_gif() -> None:
    nt = 32
    nf = 32
    dt = 1.0 / 256.0
    a = 1.0 / 3.0
    d = 1.0

    n_total = nt * nf
    signal = _chirp_signal(n_total, dt)
    series = TimeSeries(signal, dt=dt)
    coeffs_full = np.asarray(series.to_wdm(nt=nt, a=a, d=d).coeffs)
    coeffs_manual = _forward_columns(signal, nt=nt, nf=nf, dt=dt, a=a, d=d)

    if not np.allclose(coeffs_manual, coeffs_full, rtol=1e-10, atol=1e-12):
        max_err = float(np.max(np.abs(coeffs_manual - coeffs_full)))
        raise RuntimeError(f"manual forward columns do not match package output ({max_err:.3e})")

    freqs, spectrum = _positive_spectrum(signal, dt)
    coeffs_partial = np.zeros_like(coeffs_full)
    z = np.abs(coeffs_full).ravel()
    z_pos = z[z > 0]
    norm = LogNorm(
        vmin=max(float(np.percentile(z_pos, 5)), 1e-12),
        vmax=max(float(np.percentile(z_pos, 99.5)), 1e-11),
    )

    fig = plt.figure(figsize=(10.0, 5.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_col = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, :])

    ax_spec.plot(freqs, np.abs(spectrum), color="#6e7781", alpha=0.55)
    ax_spec.set_title("FFT spectrum with active WDM channel window")
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("Magnitude")
    ax_spec.set_xlim(0.0, freqs[-1])
    ax_spec.set_ylim(0.0, np.max(np.abs(spectrum)) * 1.1)

    window_line, = ax_spec.plot(freqs, np.zeros_like(freqs), color="#fb8c00", linewidth=2)
    band_span = ax_spec.axvspan(0.0, 0.0, color="#fb8c00", alpha=0.12)

    n_idx = np.arange(nt)
    column_line, = ax_col.plot(n_idx, coeffs_full[:, 0], color="#0969da")
    ax_col.set_title("One coefficient column W[:, m]")
    ax_col.set_xlabel("time-bin index n")
    ax_col.set_ylabel("coefficient value")
    col_lim = np.max(np.abs(coeffs_full)) * 1.1
    ax_col.set_ylim(-col_lim, col_lim)

    grid_im = ax_grid.imshow(
        np.abs(coeffs_partial).T + 1e-12,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Reds",
        norm=norm,
    )
    ax_grid.set_title("Packed WDM grid filling one frequency channel at a time")
    ax_grid.set_xlabel("time bin n")
    ax_grid.set_ylabel("channel m")
    fig.colorbar(grid_im, ax=ax_grid, pad=0.01, label=r"$|W[n,m]|$")
    fig.tight_layout()

    def _update(frame: int):
        nonlocal band_span
        m = frame
        coeffs_partial[:, m] = coeffs_full[:, m]

        response = _window_response(freqs, m=m, nf=nf, dt=dt, a=a, d=d)
        response_scaled = response / max(np.max(response), 1e-12) * np.max(np.abs(spectrum))
        active = response > 1e-6

        window_line.set_ydata(response_scaled)
        band_span.remove()
        if np.any(active):
            active_freqs = freqs[active]
            band_span = ax_spec.axvspan(
                active_freqs[0], active_freqs[-1], color="#fb8c00", alpha=0.12
            )
        else:
            band_span = ax_spec.axvspan(0.0, 0.0, color="#fb8c00", alpha=0.12)

        column_line.set_ydata(coeffs_full[:, m])
        ax_col.set_title(f"Coefficient column for channel m={m}")
        grid_im.set_data(np.abs(coeffs_partial).T + 1e-12)
        ax_grid.set_title(f"Packed WDM grid after filling channels 0..{m}")
        return window_line, column_line, grid_im

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=nf + 1,
        interval=150,
        blit=False,
        repeat=True,
    )
    _save_gif(anim, "wdm_freq_packetization.gif", fps=8)
    plt.close(fig)


def make_basis_atom_shift_gif() -> None:
    nt = 32
    nf = 32
    dt = 1.0 / 256.0
    a = 1.0 / 3.0
    d = 1.0
    m_fixed = 7

    n_total = nt * nf
    dt_block = nf * dt
    backend = get_backend()
    freqs = np.fft.fftfreq(n_total, d=dt)
    positive = freqs >= 0.0
    positive_freqs = freqs[positive]

    atom_freqs = []
    atom_times = []
    for n in range(nt):
        atom_f = np.asarray(gnmf(backend, n, m_fixed, freqs, dt_block, nf, a, d))
        atom_t = np.fft.ifft(atom_f)
        atom_freqs.append(np.abs(atom_f[positive]))
        atom_times.append(atom_t)

    atom_freqs = np.asarray(atom_freqs)
    atom_times = np.asarray(atom_times)
    time_axis = np.arange(n_total) * dt
    time_envelope_max = np.max(np.abs(atom_times))
    freq_max = np.max(atom_freqs)

    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_freq = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, :])

    freq_line, = ax_freq.plot(positive_freqs, atom_freqs[0], color="#fb8c00", linewidth=2)
    ax_freq.set_title(f"Frequency support of atom g[n, m] with fixed m={m_fixed}")
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel(r"$|g_{n,m}(f)|$")
    ax_freq.set_xlim(0.0, positive_freqs[-1])
    ax_freq.set_ylim(0.0, freq_max * 1.1)

    real_line, = ax_time.plot(time_axis, np.real(atom_times[0]), color="#0969da", label="Re")
    env_line, = ax_time.plot(time_axis, np.abs(atom_times[0]), color="#8250df", label="|atom|")
    ax_time.set_title("Time-domain atom shape")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlim(time_axis[0], time_axis[-1])
    ax_time.set_ylim(-time_envelope_max * 1.1, time_envelope_max * 1.1)
    ax_time.legend(frameon=False, loc="upper right")

    highlight = np.zeros((nt, nf + 1), dtype=float)
    highlight[0, m_fixed] = 1.0
    grid_im = ax_grid.imshow(
        highlight.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_grid.set_title(f"Which WDM atom is moving? fixed channel m={m_fixed}, shifting n")
    ax_grid.set_xlabel("time bin n")
    ax_grid.set_ylabel("channel m")
    fig.colorbar(grid_im, ax=ax_grid, pad=0.01, label="current atom")
    fig.tight_layout()

    def _update(frame: int):
        n = frame
        real_line.set_ydata(np.real(atom_times[n]))
        env_line.set_ydata(np.abs(atom_times[n]))
        freq_line.set_ydata(atom_freqs[n])
        highlight.fill(0.0)
        highlight[n, m_fixed] = 1.0
        grid_im.set_data(highlight.T)
        ax_time.set_title(f"Time-domain atom shape for n={n}, m={m_fixed}")
        ax_grid.set_title(f"Current atom location in the WDM grid: (n={n}, m={m_fixed})")
        return real_line, env_line, freq_line, grid_im

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=nt,
        interval=160,
        blit=False,
        repeat=True,
    )
    _save_gif(anim, "wdm_basis_atom_shift.gif", fps=8)
    plt.close(fig)


def make_channel_shift_gif() -> None:
    nt = 32
    nf = 32
    dt = 1.0 / 256.0
    a = 1.0 / 3.0
    d = 1.0
    n_fixed = 7

    n_total = nt * nf
    dt_block = nf * dt
    backend = get_backend()
    freqs = np.fft.fftfreq(n_total, d=dt)
    positive = freqs >= 0.0
    positive_freqs = freqs[positive]

    atom_freqs = []
    atom_times = []
    for m in range(nf + 1):
        atom_f = np.asarray(gnmf(backend, n_fixed, m, freqs, dt_block, nf, a, d))
        atom_t = np.fft.ifft(atom_f)
        atom_freqs.append(np.abs(atom_f[positive]))
        atom_times.append(atom_t)

    atom_freqs = np.asarray(atom_freqs)
    atom_times = np.asarray(atom_times)
    time_axis = np.arange(n_total) * dt
    time_envelope_max = np.max(np.abs(atom_times))
    freq_max = np.max(atom_freqs)

    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_freq = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, :])

    freq_line, = ax_freq.plot(positive_freqs, atom_freqs[0], color="#fb8c00", linewidth=2)
    ax_freq.set_title(f"Frequency support of atom g[n, m] with fixed n={n_fixed}")
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel(r"$|g_{n,m}(f)|$")
    ax_freq.set_xlim(0.0, positive_freqs[-1])
    ax_freq.set_ylim(0.0, freq_max * 1.1)

    real_line, = ax_time.plot(time_axis, np.real(atom_times[0]), color="#0969da", label="Re")
    env_line, = ax_time.plot(time_axis, np.abs(atom_times[0]), color="#8250df", label="|atom|")
    ax_time.set_title(f"Time-domain atom shape for n={n_fixed}, m=0")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlim(time_axis[0], time_axis[-1])
    ax_time.set_ylim(-time_envelope_max * 1.1, time_envelope_max * 1.1)
    ax_time.legend(frameon=False, loc="upper right")

    highlight = np.zeros((nt, nf + 1), dtype=float)
    highlight[n_fixed, 0] = 1.0
    grid_im = ax_grid.imshow(
        highlight.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_grid.set_title(
        f"Current atom location in the WDM grid: fixed time bin n={n_fixed}, shifting m"
    )
    ax_grid.set_xlabel("time bin n")
    ax_grid.set_ylabel("channel m")
    fig.colorbar(grid_im, ax=ax_grid, pad=0.01, label="current atom")
    fig.tight_layout()

    def _update(frame: int):
        m = frame
        real_line.set_ydata(np.real(atom_times[m]))
        env_line.set_ydata(np.abs(atom_times[m]))
        freq_line.set_ydata(atom_freqs[m])
        highlight.fill(0.0)
        highlight[n_fixed, m] = 1.0
        grid_im.set_data(highlight.T)
        ax_time.set_title(f"Time-domain atom shape for n={n_fixed}, m={m}")
        ax_grid.set_title(
            f"Current atom location in the WDM grid: (n={n_fixed}, m={m})"
        )
        return real_line, env_line, freq_line, grid_im

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=nf + 1,
        interval=160,
        blit=False,
        repeat=True,
    )
    _save_gif(anim, "wdm_channel_shift.gif", fps=8)
    plt.close(fig)


def main() -> None:
    make_phi_parameter_comparison_png()
    make_shifted_windows_png()
    make_freq_packetization_gif()
    make_basis_atom_shift_gif()
    make_channel_shift_gif()


if __name__ == "__main__":
    main()
