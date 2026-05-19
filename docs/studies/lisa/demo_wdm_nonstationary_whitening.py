"""Diagnostic demo: stationary vs non-stationary WDM whitening of LISA residuals.

Loads an injection.npz that must contain the time-dependent galactic foreground
PSD arrays saved by the updated data_generation.py:

    gal_psd_A_tf, gal_psd_E_tf, gal_psd_T_tf  -- shape (n_gal_t, n_gal_f)
    gal_psd_freqs                               -- (n_gal_f,)
    gal_psd_times                               -- (n_gal_t,)

If these keys are absent the script exits with a clear error message.

Outputs (written to the same run directory as injection.npz):
    wdm_nonstationary_whitening_demo.png   -- 2x3 whitened-power maps
    wdm_nonstationary_whitening_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.fft import irfft, rfft

# ── LISA study imports (same directory) ──────────────────────────────────────
from lisa_common import (
    INJECTION_PATH,
    RUN_DIR,
    build_wdm_nonstationary_variance,
    ensure_output_dir,
    load_injection,
    noise_tdi15_psd,
    save_figure,
    setup_jax_and_matplotlib,
    trim_frequency_band,
)
from lisa_mcmc import A_WDM, D_WDM, NT
from data_generation import INJECTION_NORMALIZATION_VERSION

from wdm_transform.signal_processing import wdm_noise_variance
from wdm_transform.transforms import forward_wdm_band

setup_jax_and_matplotlib()

# Galactic foreground frequency band of interest.
F_GAL_MIN = 1e-4  # Hz
F_GAL_MAX = 3e-3  # Hz

CH_LABELS = ("A", "E", "T")


# ── 1. Load injection ─────────────────────────────────────────────────────────

def _load_injection_with_check(path: Path) -> tuple[dict, object]:
    required = (
        "gal_psd_A_tf",
        "gal_psd_E_tf",
        "gal_psd_T_tf",
        "gal_psd_freqs",
        "gal_psd_times",
        "normalization_version",
    )
    with np.load(path, allow_pickle=False) as raw:
        missing = [k for k in required if k not in raw]
        if missing:
            sys.exit(
                f"ERROR: {path} is missing required fields: {missing}.\n"
                "Regenerate the injection with the updated data_generation.py."
            )
        normalization_version = str(np.asarray(raw["normalization_version"]).item())
        if normalization_version != INJECTION_NORMALIZATION_VERSION:
            sys.exit(
                f"ERROR: {path} was generated with normalization_version="
                f"{normalization_version!r}, expected {INJECTION_NORMALIZATION_VERSION!r}.\n"
                "Regenerate the injection with the updated data_generation.py."
            )
        inj_npz = {k: np.asarray(v) for k, v in raw.items()}
    injection = load_injection(path)
    return inj_npz, injection


# ── 2. WDM grid helpers ───────────────────────────────────────────────────────

def _wdm_grid(injection, nt: int) -> tuple[int, int, float, int, np.ndarray, np.ndarray, slice]:
    """Return WDM grid parameters for the full galactic foreground band."""
    n_keep = (len(injection.data_At) // (2 * nt)) * (2 * nt)
    nf = n_keep // nt
    t_obs = n_keep * injection.dt
    n_freqs_full = n_keep // 2 + 1
    df_rfft = 1.0 / t_obs
    freq_grid = np.linspace(0.0, 0.5 / injection.dt, nf + 1)
    band_slice = trim_frequency_band(freq_grid, F_GAL_MIN, F_GAL_MAX, pad_bins=2)
    wdm_freq_centers = freq_grid[band_slice]
    wdm_time_centers = (np.arange(nt) + 0.5) * t_obs / nt
    return n_keep, nf, t_obs, n_freqs_full, wdm_freq_centers, wdm_time_centers, band_slice


# ── 3. WDM transform of residuals ────────────────────────────────────────────

def _source_time_series_for_truncation(source_f: np.ndarray, full_length: int, n_keep: int) -> np.ndarray:
    """Reconstruct a full-length rFFT source before truncating the time series."""
    return irfft(source_f, n=full_length, axis=-1)[..., :n_keep]


def _compute_wdm_residuals(
    injection,
    n_keep: int,
    nf: int,
    n_freqs_full: int,
    df_rfft: float,
    band_slice: slice,
    nt: int,
) -> np.ndarray:
    """Return WDM coefficients of source-subtracted residuals, shape (3, NT, n_band)."""
    data = np.stack([
        injection.data_At[:n_keep],
        injection.data_Et[:n_keep],
        injection.data_Tt[:n_keep],
    ])
    # Subtract injected source (transform source freq-series to time domain first)
    source_f = np.stack([
        np.asarray(injection.source_Af, dtype=np.complex128),
        np.asarray(injection.source_Ef, dtype=np.complex128),
        np.asarray(injection.source_Tf, dtype=np.complex128),
    ])
    source_t = _source_time_series_for_truncation(source_f, len(injection.data_At), n_keep)
    residual = data - source_t  # (3, n_keep)

    data_rfft = rfft(residual, axis=1)  # (3, n_freqs_full)

    # Band indices for the rFFT: each WDM channel m uses rFFT bins [m*nt/2, (m+1)*nt/2].
    half = nt // 2
    kmin_rfft = max((band_slice.start - 1) * half, 0)
    kmax_rfft = min(band_slice.stop * half, n_freqs_full)
    nf_sub = band_slice.stop - band_slice.start

    band_kwargs = {
        "df": df_rfft,
        "nfreqs_fourier": n_freqs_full,
        "kmin": kmin_rfft,
        "nfreqs_wdm": nf,
        "ntimes_wdm": nt,
        "mmin": band_slice.start,
        "nf_sub_wdm": nf_sub,
        "a": A_WDM,
        "d": D_WDM,
        "backend": "numpy",
    }
    coeffs = np.stack([
        np.asarray(
            forward_wdm_band(data_rfft[ch, kmin_rfft:kmax_rfft], **band_kwargs)
        )
        for ch in range(3)
    ])  # (3, NT, n_band)
    return coeffs


# ── 4. Variance models ────────────────────────────────────────────────────────

def _stationary_variance(
    injection,
    wdm_freq_centers: np.ndarray,
    nt: int,
    n_freqs_full: int,
) -> np.ndarray:
    """Stationary WDM variance (3, NT, n_band), broadcast from freq-only PSD."""
    from lisa_common import interp_psd_channels
    psd_channels = np.stack([
        injection.noise_psd_A,
        injection.noise_psd_E,
        injection.noise_psd_T,
    ])
    noise_psd = interp_psd_channels(wdm_freq_centers, injection.freqs, psd_channels)
    sigma2_stat = (2 * (n_freqs_full - 1)) * np.stack([
        wdm_noise_variance(psd, nt=nt, dt=injection.dt) for psd in noise_psd
    ])
    return sigma2_stat  # (3, NT, n_band)


def _nonstationary_variance(
    inj_npz: dict,
    wdm_freq_centers: np.ndarray,
    wdm_time_centers: np.ndarray,
    dt: float,
    n_freqs_full: int,
) -> np.ndarray:
    """Non-stationary WDM variance (3, NT, n_band) using time-varying galactic PSD."""
    return np.stack([
        build_wdm_nonstationary_variance(
            inj_npz,
            channel=ch,
            wdm_freq_centers=wdm_freq_centers,
            wdm_time_centers=wdm_time_centers,
            dt=dt,
            n_freqs_full=n_freqs_full,
        )
        for ch in range(3)
    ])  # (3, NT, n_band)


# ── 5. Diagnostics ────────────────────────────────────────────────────────────

def _compute_fractional_miscalibration(
    sigma2_stat: np.ndarray,
    sigma2_nonstat: np.ndarray,
) -> np.ndarray:
    """Fractional sigma2 error from using stationary model, shape (3, NT, n_band)."""
    return (sigma2_nonstat - sigma2_stat) / np.maximum(sigma2_stat, 1e-300)


def _compute_metrics(
    coeffs: np.ndarray,
    sigma2_stat: np.ndarray,
    sigma2_nonstat: np.ndarray,
) -> dict:
    """Return scalar whitening-quality metrics for each channel and combined."""
    metrics: dict = {}
    all_stat_pwr: list[np.ndarray] = []
    all_ns_pwr: list[np.ndarray] = []

    for ch, label in enumerate(CH_LABELS):
        w = coeffs[ch]                          # (NT, n_band)
        var_s = sigma2_stat[ch]
        var_n = sigma2_nonstat[ch]

        wp_s = w**2 / var_s                     # whitened power, stationary
        wp_n = w**2 / var_n                     # whitened power, non-stationary

        # Per-time-bin mean whitened power (should be ~1.0 for well-whitened data)
        tb_mean_s = wp_s.mean(axis=1)           # (NT,)
        tb_mean_n = wp_n.mean(axis=1)           # (NT,)

        n_pix = float(w.size)
        ch_metrics = {
            "mean_whitened_stat": float(wp_s.mean()),
            "std_whitened_stat": float(tb_mean_s.std()),
            "mean_whitened_nonstat": float(wp_n.mean()),
            "std_whitened_nonstat": float(tb_mean_n.std()),
            "reduced_chi2_stat": float(wp_s.sum() / n_pix),
            "reduced_chi2_nonstat": float(wp_n.sum() / n_pix),
        }
        std_s = ch_metrics["std_whitened_stat"]
        std_n = ch_metrics["std_whitened_nonstat"]
        ch_metrics["time_flat_ratio"] = float(std_s / std_n) if std_n > 0 else float("inf")

        frac_miscal = (var_n - var_s) / np.maximum(var_s, 1e-300)
        ch_metrics["max_abs_frac_miscalibration"] = float(np.abs(frac_miscal).max())
        ch_metrics["rms_frac_miscalibration"] = float(np.sqrt((frac_miscal**2).mean()))

        metrics[label] = ch_metrics

        all_stat_pwr.append(wp_s.ravel())
        all_ns_pwr.append(wp_n.ravel())

    combined_s = np.concatenate(all_stat_pwr)
    combined_n = np.concatenate(all_ns_pwr)
    metrics["combined"] = {
        "mean_whitened_stat": float(combined_s.mean()),
        "std_whitened_stat": float(combined_s.std()),
        "mean_whitened_nonstat": float(combined_n.mean()),
        "std_whitened_nonstat": float(combined_n.std()),
        "reduced_chi2_stat": float(combined_s.mean()),
        "reduced_chi2_nonstat": float(combined_n.mean()),
    }
    return metrics


def _print_summary(metrics: dict) -> None:
    print("\n── WDM Non-Stationary Whitening Summary ─────────────────────────────")
    header = (
        f"{'Channel':>8}  {'χ²_stat':>10}  {'χ²_nonstat':>12}"
        f"  {'time-flat':>10}  {'max|δ|':>8}  {'rms δ':>8}"
    )
    print(header)
    print("─" * len(header))
    for ch in (*CH_LABELS, "combined"):
        m = metrics[ch]
        time_flat = m.get("time_flat_ratio", float("nan"))
        max_miscal = m.get("max_abs_frac_miscalibration", float("nan"))
        rms_miscal = m.get("rms_frac_miscalibration", float("nan"))
        print(
            f"{ch:>8}  {m['reduced_chi2_stat']:>10.4f}  "
            f"{m['reduced_chi2_nonstat']:>12.4f}  {time_flat:>10.3f}"
            f"  {max_miscal:>8.3f}  {rms_miscal:>8.3f}"
        )
    print("─" * len(header))
    print(
        "(χ² ~ 1.0 = well-whitened; time-flat > 1 = nonstat. flatter; "
        "δ = (σ²_ns−σ²_s)/σ²_s)\n"
    )


# ── 6. Plotting ───────────────────────────────────────────────────────────────

def _plot_whitened_maps(
    coeffs: np.ndarray,
    sigma2_stat: np.ndarray,
    sigma2_nonstat: np.ndarray,
    frac_miscal: np.ndarray,
    wdm_freq_centers: np.ndarray,
    wdm_time_centers: np.ndarray,
    run_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)

    freqs_mhz = wdm_freq_centers * 1e3
    times_days = wdm_time_centers / 86400.0

    # ── Row 0: PSD ratio S_ns(t,f) / S_s(f) ─────────────────────────────────
    ratio_vmin, ratio_vmax = 0.5, 3.0
    for ch, label in enumerate(CH_LABELS):
        ax = axes[0, ch]
        ratio = sigma2_nonstat[ch] / np.maximum(sigma2_stat[ch], 1e-300)
        norm = TwoSlopeNorm(vcenter=1.0, vmin=ratio_vmin, vmax=ratio_vmax)
        im = ax.pcolormesh(
            times_days,
            freqs_mhz,
            ratio.T,
            norm=norm,
            cmap="RdBu_r",
            shading="nearest",
            rasterized=True,
        )
        ax.set_title(f"Channel {label} — PSD ratio", fontsize=10)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Frequency (mHz)")
        ax.annotate(
            f"min={ratio.min():.2f}  max={ratio.max():.2f}",
            xy=(0.02, 0.96),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
        )
        plt.colorbar(im, ax=ax, label=r"$S_\mathrm{ns}(t,f)\,/\,S_\mathrm{s}(f)$")

    # ── Rows 1–2: whitened power maps ────────────────────────────────────────
    vmin, vmax = 1e-2, 1e2
    row_specs = [
        (1, sigma2_stat,    "Stationary $\\sigma^2$"),
        (2, sigma2_nonstat, "Non-stationary $\\sigma^2$"),
    ]
    for row, sigma2, model_label in row_specs:
        for ch, label in enumerate(CH_LABELS):
            ax = axes[row, ch]
            w = coeffs[ch]
            wp = np.maximum(w**2 / sigma2[ch], 1e-30)
            im = ax.pcolormesh(
                times_days,
                freqs_mhz,
                wp.T,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap="RdBu_r",
                shading="nearest",
                rasterized=True,
            )
            ax.set_title(f"Channel {label} — {model_label}", fontsize=10)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Frequency (mHz)")
            ax.annotate(
                f"mean={wp.mean():.3f}  std={wp.mean(axis=1).std():.3f}",
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                va="top",
                fontsize=7,
                color="white",
            )
            plt.colorbar(im, ax=ax, label=r"$w^2 / \sigma^2$")

    # ── Row 3: fractional miscalibration (σ²_ns − σ²_s) / σ²_s ─────────────
    miscal_lim = 0.5
    for ch, label in enumerate(CH_LABELS):
        ax = axes[3, ch]
        fm = frac_miscal[ch]
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-miscal_lim, vmax=miscal_lim)
        im = ax.pcolormesh(
            times_days,
            freqs_mhz,
            fm.T,
            norm=norm,
            cmap="RdBu_r",
            shading="nearest",
            rasterized=True,
        )
        ax.set_title(f"Channel {label} — fractional miscalibration", fontsize=10)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Frequency (mHz)")
        ax.annotate(
            f"max|δ|={np.abs(fm).max():.2f}  rms={np.sqrt((fm**2).mean()):.3f}",
            xy=(0.02, 0.96),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
        )
        plt.colorbar(im, ax=ax, label=r"$(\sigma^2_\mathrm{ns}-\sigma^2_\mathrm{s})\,/\,\sigma^2_\mathrm{s}$")

    fig.suptitle(
        r"Row 0: PSD ratio $S_\mathrm{ns}/S_\mathrm{s}$  |  "
        r"Row 1: stationary whitened power  |  Row 2: non-stationary whitened power  |  "
        r"Row 3: fractional $\sigma^2$ miscalibration $\delta$",
        fontsize=9,
    )
    save_figure(fig, run_dir, "wdm_nonstationary_whitening_demo")
    print(f"Saved plot to {run_dir / 'wdm_nonstationary_whitening_demo.png'}")


def _plot_noise_budget(
    sigma2_stat: np.ndarray,
    sigma2_nonstat: np.ndarray,
    wdm_freq_centers: np.ndarray,
    dt: float,
    n_freqs_full: int,
    run_dir: Path,
) -> None:
    """1D noise budget: stationary PSD vs min/max envelope of non-stationary PSD."""
    import matplotlib.pyplot as plt

    n_total = 2 * (n_freqs_full - 1)
    # Convert WDM variance back to one-sided PSD: S = sigma2 * 2*dt / N
    psd_stat = sigma2_stat * (2.0 * dt) / n_total        # (3, NT, n_band)
    psd_ns = sigma2_nonstat * (2.0 * dt) / n_total       # (3, NT, n_band)

    freqs_mhz = wdm_freq_centers * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)

    for ch, label in enumerate(CH_LABELS):
        # ── Top row: absolute PSD ──────────────────────────────────────────
        ax = axes[0, ch]
        s_stat = psd_stat[ch, 0]          # same for all time bins
        s_ns_min = psd_ns[ch].min(axis=0)
        s_ns_max = psd_ns[ch].max(axis=0)

        ax.semilogy(freqs_mhz, s_stat, "k-", lw=1.5, label=r"$S_\mathrm{stat}$ (mean)")
        ax.fill_between(freqs_mhz, s_ns_min, s_ns_max, alpha=0.4, label="nonstat. range")
        ax.set_title(f"Channel {label} — noise PSD", fontsize=10)
        ax.set_xlabel("Frequency (mHz)")
        ax.set_ylabel(r"$S(f)$  [Hz$^{-1}$]")
        ax.legend(fontsize=7)

        # ── Bottom row: fractional temporal variation ──────────────────────
        ax2 = axes[1, ch]
        frac_std = psd_ns[ch].std(axis=0) / np.maximum(s_stat, 1e-300)
        frac_max = (s_ns_max - s_stat) / np.maximum(s_stat, 1e-300)
        frac_min = (s_ns_min - s_stat) / np.maximum(s_stat, 1e-300)

        ax2.plot(freqs_mhz, frac_max, "r-", lw=1, label="max δ")
        ax2.plot(freqs_mhz, frac_min, "b-", lw=1, label="min δ")
        ax2.plot(freqs_mhz, frac_std, "k--", lw=1, label=r"std$_t$ δ")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_title(f"Channel {label} — fractional temporal variation", fontsize=10)
        ax2.set_xlabel("Frequency (mHz)")
        ax2.set_ylabel(r"$(S_\mathrm{ns}(t,f) - S_\mathrm{s}(f))\,/\,S_\mathrm{s}(f)$")
        ax2.legend(fontsize=7)

    fig.suptitle(
        "Noise budget: stationary (mean) vs non-stationary PSD envelope\n"
        "Bottom row shows fractional miscalibration — explains why MCMC posteriors "
        "are unbiased at typical SNR",
        fontsize=9,
    )
    save_figure(fig, run_dir, "wdm_nonstationary_noise_budget")
    print(f"Saved plot to {run_dir / 'wdm_nonstationary_noise_budget.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(injection_path: Path = INJECTION_PATH, run_dir: Path = RUN_DIR) -> None:
    print(f"Loading injection from {injection_path}")
    inj_npz, injection = _load_injection_with_check(injection_path)

    n_keep, nf, t_obs, n_freqs_full, wdm_freq_centers, wdm_time_centers, band_slice = (
        _wdm_grid(injection, NT)
    )
    df_rfft = 1.0 / t_obs
    print(
        f"WDM grid: NT={NT}, nf={nf}, n_band={band_slice.stop - band_slice.start} channels, "
        f"t_obs={t_obs/86400:.1f} days"
    )

    print("Computing WDM transform of source-subtracted residuals …")
    coeffs = _compute_wdm_residuals(
        injection, n_keep, nf, n_freqs_full, df_rfft, band_slice, NT
    )

    print("Building stationary variance …")
    sigma2_stat = _stationary_variance(injection, wdm_freq_centers, NT, n_freqs_full)

    print("Building non-stationary variance …")
    sigma2_nonstat = _nonstationary_variance(
        inj_npz, wdm_freq_centers, wdm_time_centers, injection.dt, n_freqs_full
    )

    frac_miscal = _compute_fractional_miscalibration(sigma2_stat, sigma2_nonstat)

    metrics = _compute_metrics(coeffs, sigma2_stat, sigma2_nonstat)
    _print_summary(metrics)

    ensure_output_dir(run_dir)
    metrics_path = run_dir / "wdm_nonstationary_whitening_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    _plot_whitened_maps(
        coeffs, sigma2_stat, sigma2_nonstat, frac_miscal,
        wdm_freq_centers, wdm_time_centers, run_dir,
    )
    _plot_noise_budget(
        sigma2_stat, sigma2_nonstat,
        wdm_freq_centers, injection.dt, n_freqs_full, run_dir,
    )


if __name__ == "__main__":
    main()
