"""Publication-style infographic for one LISA GB injection.

Shows:
- A-channel frequency-domain power with the injected GB over the stochastic background
- Characteristic strain of the injected source against the A-channel sensitivity
- Whitened WDM map of the injected data with a zoomed source inset

Usage:
    cd docs/studies/lisa
    python3 infographic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lisa_common import (
    INJECTION_PATH,
    RUN_DIR,
    ensure_output_dir,
    load_injection,
    noise_tdi15_psd,
    place_local_tdi,
    save_figure,
    setup_jax_and_matplotlib,
)

A_WDM = 1.0 / 3.0
D_WDM = 1.0
DEFAULT_NT_WDM = 48


def setup_plotting():
    """Configure matplotlib for publication-style plots."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.6,
    })
    return matplotlib, plt


def compute_characteristic_strains(freqs, psd, data_fft, dt):
    """Compute characteristic strains for noise and signal."""
    from wdm_transform.signal_processing import (
        noise_characteristic_strain,
        rfft_characteristic_strain,
    )

    h_c_noise = noise_characteristic_strain(psd, freqs)
    h_c_signal = rfft_characteristic_strain(data_fft, freqs, dt)
    return h_c_noise, h_c_signal


def one_sided_periodogram_density(rfft_coeffs: np.ndarray, dt: float, n_samples: int) -> np.ndarray:
    """Return a simple one-sided PSD estimate from NumPy rFFT coefficients."""
    psd = (2.0 * dt / n_samples) * np.abs(np.asarray(rfft_coeffs)) ** 2
    if psd.size:
        psd[0] *= 0.5
    if n_samples % 2 == 0 and psd.size > 1:
        psd[-1] *= 0.5
    return np.maximum(psd, 1e-60)


def regenerate_source_rfft(injection) -> np.ndarray:
    """Rebuild the exact injected A-channel source template from the saved parameters."""
    import jax
    import jax.numpy as jnp
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    jax.config.update("jax_enable_x64", True)

    params = np.asarray(injection.source_params[0], dtype=float)
    orbit = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit, t_obs=float(injection.t_obs), t0=0.0, n=256)
    params_j = jnp.asarray(params, dtype=jnp.float64)
    a_loc, _, _ = jgb.get_tdi(params_j, tdi_generation=1.5, tdi_combination="AET")
    kmin = int(jnp.asarray(jgb.get_kmin(params_j[None, 0:1])).reshape(-1)[0])
    return place_local_tdi(np.asarray(a_loc), kmin, len(injection.freqs))


def build_wdm_products(
    data_t: np.ndarray,
    source_t: np.ndarray,
    *,
    dt: float,
    noise_psd: np.ndarray,
    nt: int = DEFAULT_NT_WDM,
):
    """Return total/source WDM maps and the whitening array."""
    from wdm_transform import TimeSeries
    from wdm_transform.signal_processing import wdm_noise_variance

    total_wdm = TimeSeries(np.asarray(data_t, dtype=float), dt=dt).to_wdm(nt=nt, a=A_WDM, d=D_WDM)
    source_wdm = TimeSeries(np.asarray(source_t, dtype=float), dt=dt).to_wdm(nt=nt, a=A_WDM, d=D_WDM)
    noise_row = np.interp(
        np.asarray(total_wdm.freq_grid, dtype=float),
        np.linspace(0.0, total_wdm.nyquist, len(noise_psd)),
        np.asarray(noise_psd, dtype=float),
        left=float(noise_psd[0]),
        right=float(noise_psd[-1]),
    )
    whitening = np.sqrt(wdm_noise_variance(noise_row, nt=total_wdm.nt, dt=dt))
    return total_wdm, source_wdm, whitening


def whitened_wdm_image(wdm, whitening: np.ndarray) -> np.ndarray:
    """Return the absolute whitened WDM coefficient image in display orientation."""
    coeffs = np.asarray(wdm.coeffs[0], dtype=float)
    return np.abs(coeffs / whitening).T


def add_wdm_panel(
    fig,
    ax,
    *,
    matplotlib,
    plt,
    total_wdm,
    source_wdm,
    whitening: np.ndarray,
    source_params: np.ndarray,
):
    """Draw the total-data WDM map plus a source-only zoom inset."""
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    total_img = whitened_wdm_image(total_wdm, whitening)
    source_img = whitened_wdm_image(source_wdm, whitening)
    time_grid = np.asarray(total_wdm.time_grid, dtype=float)
    freq_grid = np.asarray(total_wdm.freq_grid, dtype=float)
    extent = [time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]]

    positive = total_img[total_img > 0.0]
    vmin = max(float(np.nanpercentile(positive, 35)), 0.3)
    vmax = max(float(np.nanpercentile(total_img, 99.7)), vmin * 4.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        total_img,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    chirp_track = source_params[0] + source_params[1] * time_grid
    ax.plot(time_grid, chirp_track, color="#7ce2ff", lw=1.6, alpha=0.9, label="Injected source track")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3e-3)
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_title("(c) A-Channel WDM Map: Galactic Background + Injected GB", fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.12, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.92)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Per-pixel WDM amplitude / noise std")

    delta_f = max(8.0 * total_wdm.delta_f, 1.5e-5)
    freq_lo = max(1e-4, source_params[0] - delta_f)
    freq_hi = min(3e-3, source_params[0] + delta_f)

    inset = inset_axes(ax, width="42%", height="42%", loc="lower left", borderpad=1.2)
    inset.imshow(
        source_img,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    inset.plot(time_grid, chirp_track, color="white", lw=1.0, alpha=0.85)
    inset.set_yscale("log")
    inset.set_xlim(time_grid[0], time_grid[-1])
    inset.set_ylim(freq_lo, freq_hi)
    inset.set_xticks([])
    inset.set_yticks([source_params[0]])
    inset.set_yticklabels([f"{source_params[0]:.4e}"])
    inset.set_title("Injected source only", fontsize=8, pad=4)
    for spine in inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.1)

    ax.axhspan(freq_lo, freq_hi, facecolor="none", edgecolor="white", linewidth=0.9, alpha=0.35)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="white", alpha=0.4)


def create_infographic(
    injection_path: Path = INJECTION_PATH,
    output_dir: Path | None = None,
    seed: int = 10,
):
    """Generate an infographic for one GB case with a WDM inset."""

    matplotlib, plt = setup_plotting()

    if output_dir is None:
        output_dir = RUN_DIR / "infographic"

    ensure_output_dir(output_dir)

    injection = load_injection(injection_path)
    freqs = injection.freqs
    dt = injection.dt
    t_obs = injection.t_obs
    source_params = np.asarray(injection.source_params[0], dtype=float)
    source_fft = regenerate_source_rfft(injection)
    data_fft = np.fft.rfft(np.asarray(injection.data_At, dtype=float))
    background_fft = data_fft - source_fft
    source_t = np.fft.irfft(source_fft, n=len(injection.data_At))

    psd_data = one_sided_periodogram_density(data_fft, dt, len(injection.data_At))
    psd_background = one_sided_periodogram_density(background_fft, dt, len(injection.data_At))
    psd_source = one_sided_periodogram_density(source_fft, dt, len(injection.data_At))
    inst_psd = np.maximum(noise_tdi15_psd(0, freqs), 1e-60)
    total_psd = np.maximum(np.asarray(injection.noise_psd_A, dtype=float), 1e-60)

    h_c_total, h_c_source = compute_characteristic_strains(freqs, total_psd, source_fft, dt)
    h_c_inst, _ = compute_characteristic_strains(freqs, inst_psd, source_fft, dt)
    h_c_data = compute_characteristic_strains(freqs, total_psd, data_fft, dt)[1]

    total_wdm, source_wdm, whitening = build_wdm_products(
        injection.data_At,
        source_t,
        dt=dt,
        noise_psd=total_psd,
    )

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.34, height_ratios=[1.0, 0.9, 1.25])

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(freqs[1:], psd_data[1:], color="#1f77b4", alpha=0.45, lw=1.6, label="A data: background + injection")
    ax1.loglog(freqs[1:], psd_background[1:], color="#d6cf66", alpha=0.72, lw=1.3, label="A background realization")
    ax1.loglog(freqs[1:], total_psd[1:], color="#2ca02c", lw=2.0, linestyle=":", label="A total PSD model")
    ax1.loglog(freqs[1:], inst_psd[1:], color="#4d4d4d", lw=1.4, linestyle="--", label="Instrument PSD")
    ax1.loglog(freqs[1:], psd_source[1:], color="#ff7f0e", alpha=0.95, lw=1.8, label="Injected GB")
    ax1.axvline(source_params[0], color="#b22222", lw=1.8, linestyle="--", alpha=0.8)
    ax1.set_xlim(1e-4, 3e-3)
    ax1.set_ylim(1e-44, max(float(np.nanpercentile(psd_data[1:], 99.9)) * 2.0, 1e-36))
    ax1.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]", fontweight="bold")
    ax1.set_title("(a) A-Channel Power: Galactic Background, Noise, and Injected Source", fontweight="bold", pad=10)
    ax1.grid(True, which="both", alpha=0.18, linestyle=":")
    ax1.legend(loc="upper right", framealpha=0.95)

    ax2 = fig.add_subplot(gs[1])
    ax2.loglog(freqs[1:], h_c_inst[1:], color="#4d4d4d", lw=1.4, linestyle="--", label="Instrument noise")
    ax2.loglog(freqs[1:], h_c_total[1:], color="#2ca02c", lw=2.0, linestyle=":", label="Total A-channel sensitivity")
    ax2.loglog(freqs[1:], h_c_data[1:], color="#1f77b4", alpha=0.5, lw=1.2, label="Observed A data")
    ax2.loglog(freqs[1:], h_c_source[1:], color="#ff7f0e", lw=2.3, label="Injected GB")
    ax2.axvline(source_params[0], color="#b22222", lw=1.8, linestyle="--", alpha=0.8)
    ax2.set_xlim(1e-4, 3e-3)
    ax2.set_ylabel(r"$h_c(f)$", fontweight="bold")
    ax2.set_title("(b) Characteristic Strain: Resolved Signal Above the LISA Foreground", fontweight="bold", pad=10)
    ax2.grid(True, which="both", alpha=0.18, linestyle=":")
    ax2.legend(loc="upper right", framealpha=0.95)

    ax3 = fig.add_subplot(gs[2])
    add_wdm_panel(
        fig,
        ax3,
        matplotlib=matplotlib,
        plt=plt,
        total_wdm=total_wdm,
        source_wdm=source_wdm,
        whitening=whitening,
        source_params=source_params,
    )

    info_text = (
        "Seed {seed}  |  f₀ = {f0:.4e} Hz  |  ḟ = {fdot:.3e} Hz/s  |  A = {amp:.3e}\n"
        "Observation = {years:.1f} year  |  dt = {dt:.1f} s  |  WDM grid = {nt}×{nf}"
    ).format(
        seed=seed,
        f0=source_params[0],
        fdot=source_params[1],
        amp=source_params[2],
        years=t_obs / (365.25 * 24 * 3600),
        dt=dt,
        nt=total_wdm.nt,
        nf=total_wdm.nf + 1,
    )
    fig.text(
        0.5,
        0.035,
        info_text,
        fontsize=9,
        family="monospace",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#f4f4f4",
            "edgecolor": "#333333",
            "linewidth": 0.8,
        },
    )

    fig.subplots_adjust(top=0.96, bottom=0.12, left=0.08, right=0.96)

    path = save_figure(fig, output_dir, "gb_infographic", dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    setup_jax_and_matplotlib()

    for seed in [21, 0]:
        try:
            run_dir = Path(__file__).parent / f"outdir_lisa/galactic_background/seed_{seed}"
            injection_path = run_dir / "injection.npz"
            if not injection_path.exists():
                print(f"Skipping seed {seed}: no injection.npz")
                continue

            print(f"Generating infographic for seed {seed}...")
            path = create_infographic(
                injection_path=injection_path,
                output_dir=run_dir / "infographic",
                seed=seed,
            )
            print(f"  Saved to {path}")
        except Exception as exc:
            print(f"  Error for seed {seed}: {exc}")
            import traceback

            traceback.print_exc()
