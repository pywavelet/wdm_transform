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
DEFAULT_NT_WDM = 432

FINSET_SPINE_ = "#434242"
DATA_COL = "#D8D8D8"
INJECTION_COL = "#ff7f0e"
NOISE_PSD = "#172919"
FULL_PSD = "#172919"
SECONDS_PER_YEAR = 365.25 * 24 * 3600


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


def wdm_image(wdm, whitening: np.ndarray | None = None) -> np.ndarray:
    """Return the absolute WDM coefficient image in display orientation."""
    coeffs = np.asarray(wdm.coeffs[0], dtype=float)
    if whitening is not None:
        coeffs = coeffs / whitening
    return np.abs(coeffs).T


def add_frequency_inset(
    ax,
    *,
    freqs: np.ndarray,
    psd_data: np.ndarray,
    psd_background: np.ndarray,
    psd_source: np.ndarray,
    source_params: np.ndarray,
):
    """Add a compact zoom around the injected carrier in the PSD panel."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    f0 = float(source_params[0])
    half_width = max(18.0 * (freqs[1] - freqs[0]), 1.8e-5)
    freq_lo = max(1e-4, f0 - half_width)
    freq_hi = min(3e-3, f0 + half_width)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if np.count_nonzero(mask) < 8:
        return

    local_data = np.maximum(psd_data[mask], 1e-60)
    local_bg = np.maximum(psd_background[mask], 1e-60)
    local_src = np.maximum(psd_source[mask], 1e-60)
    local_freqs = freqs[mask]

    inset = inset_axes(
        ax,
        width="31%",
        height="48%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.09, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.set_facecolor("white")
    inset.loglog(
        local_freqs,
        local_src,
        color=INJECTION_COL,
        lw=5.0,
        alpha=0.28,
        zorder=-10,
        solid_capstyle="round",
    )
    inset.loglog(local_freqs, local_data, color=DATA_COL, lw=1.1, alpha=1, zorder=10)
    inset.loglog(local_freqs, local_src, color=INJECTION_COL, lw=1.8, alpha=0.95, zorder=4)
    inset.set_xlim(freq_lo, freq_hi)
    inset.set_ylim(
        max(float(np.nanmin(local_bg)) * 0.5, 1e-44),
        max(float(np.nanmax(np.maximum(local_data, local_src))) * 2.0, 1e-38),
    )
    xticks = np.linspace(freq_lo, freq_hi, 5)
    inset.xaxis.set_major_locator(FixedLocator(xticks))
    inset.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e3:.4f}"))
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    inset.yaxis.set_minor_formatter(NullFormatter())
    inset.tick_params(axis="both", labelsize=7, pad=1, length=3)
    inset.set_xlabel("Frequency (mHz)", fontsize=7, labelpad=1)
    inset.set_ylabel(r"$S_h(f)$", fontsize=7, labelpad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(FINSET_SPINE_)

    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec=FINSET_SPINE_, alpha=0.55)


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

    total_img = wdm_image(total_wdm, whitening)
    source_img = wdm_image(source_wdm, whitening)
    time_grid = np.asarray(total_wdm.time_grid, dtype=float)
    time_years = time_grid / SECONDS_PER_YEAR
    freq_grid = np.asarray(total_wdm.freq_grid, dtype=float)
    extent = [time_years[0], time_years[-1], freq_grid[0], freq_grid[-1]]

    positive = total_img[total_img > 0.0]
    vmin = max(float(np.nanpercentile(positive, 18)), 0.25)
    vmax = max(float(np.nanpercentile(total_img, 99.5)), vmin * 6.0)
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
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3e-3)
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax.set_xlabel("Time (yr)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.12, linestyle=":")
    ax.text(
        0.01,
        0.98,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color="white",
    )
    ax.set_xticks(np.linspace(time_years[0], time_years[-1], 5))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("WDM amplitude / noise std")

    delta_f = max(8.0 * total_wdm.delta_f, 1.5e-5)
    freq_lo = max(1e-4, source_params[0] - delta_f)
    freq_hi = min(3e-3, source_params[0] + delta_f)

    source_positive = source_img[source_img > 0.0]
    source_vmin = max(float(np.nanpercentile(source_positive, 20)), 0.35)
    source_vmax = max(float(np.nanpercentile(source_img, 99.7)), source_vmin * 5.0)
    source_norm = LogNorm(vmin=source_vmin, vmax=source_vmax)

    inset = inset_axes(
        ax,
        width="31%",
        height="48%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.09, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.imshow(
        source_img,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=source_norm,
        interpolation="nearest",
        rasterized=True,
    )
    inset.set_yscale("log")
    inset.set_xlim(time_years[0], time_years[-1])
    inset.set_ylim(freq_lo, freq_hi)
    inset.tick_params(axis="both", colors="white", labelsize=7, pad=1)
    year_ticks = np.linspace(time_years[0], time_years[-1], 4)
    inset.set_xticks(year_ticks)
    inset.set_xticklabels([f"{tick:.2f}" for tick in year_ticks], color="white")
    inset.set_xlabel("Time (yr)", color="white", fontsize=7, labelpad=1)
    inset.set_yticks([freq_lo, source_params[0], freq_hi])
    inset.set_yticklabels(
        [f"{freq_lo * 1e3:.3f}", f"{source_params[0] * 1e3:.3f}", f"{freq_hi * 1e3:.3f}"],
        color="white",
    )
    inset.set_ylabel("mHz", color="white", fontsize=7, labelpad=1)
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
    source_params = np.asarray(injection.source_params[0], dtype=float)
    source_fft = regenerate_source_rfft(injection)
    data_fft = np.fft.rfft(np.asarray(injection.data_At, dtype=float))
    background_fft = data_fft - source_fft
    source_t = np.fft.irfft(source_fft, n=len(injection.data_At))

    psd_data = one_sided_periodogram_density(data_fft, dt, len(injection.data_At))
    psd_background = one_sided_periodogram_density(background_fft, dt, len(injection.data_At))
    psd_source = one_sided_periodogram_density(source_fft, dt, len(injection.data_At))
    total_psd = np.maximum(np.asarray(injection.noise_psd_A, dtype=float), 1e-60)
    inst_psd = np.maximum(noise_tdi15_psd(0, freqs), 1e-60)

    total_wdm, source_wdm, whitening = build_wdm_products(
        injection.data_At,
        source_t,
        dt=dt,
        noise_psd=total_psd,
    )

    fig = plt.figure(figsize=(10, 9.2))
    gs = fig.add_gridspec(2, 1, hspace=0.28, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(freqs[1:], psd_data[1:], color=DATA_COL, alpha=0.9, lw=1.2, label="A data")
    ax1.loglog(freqs[1:], total_psd[1:], color=NOISE_PSD, lw=2.0, label="Total PSD model")
    ax1.loglog(freqs[1:], inst_psd[1:], color=FULL_PSD, lw=1.4, linestyle="--", label="Instrument PSD")
    ax1.loglog([], [], color=INJECTION_COL, lw=1.8, alpha=0.95, label="Injected source")
    ax1.set_xlim(1e-4, 3e-3)
    ax1.set_ylim(1e-44, 10**-36)
    ax1.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.18, linestyle=":")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, ha="left", va="top", fontweight="bold")
    add_frequency_inset(
        ax1,
        freqs=freqs,
        psd_data=psd_data,
        psd_background=psd_background,
        psd_source=psd_source,
        source_params=source_params,
    )

    ax2 = fig.add_subplot(gs[1])
    add_wdm_panel(
        fig,
        ax2,
        matplotlib=matplotlib,
        plt=plt,
        total_wdm=total_wdm,
        source_wdm=source_wdm,
        whitening=whitening,
        source_params=source_params,
    )

    # save fig as PDF 
    path = output_dir / f"lisa_gb_infographic_seed_{seed}.pdf"
    plt.savefig(path, format="pdf" )
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
