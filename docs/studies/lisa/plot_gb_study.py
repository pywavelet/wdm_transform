"""Figures for the LISA galactic-binary freq-vs-WDM study.

Produces, for a chosen seed and the full outdir_gb/ population:

  1. data overview  -> outdir_gb/fig_data_seed<N>.png
       channel-A frequency-domain overview, with frequency-domain and WDM
       source-band zooms below it.
  2. comparison corner (one seed) -> outdir_gb/fig_corner_seed<N>.png
       freq (blue) vs WDM (orange) posteriors with truth lines.
  3. PP plot (all params one axes) -> outdir_gb/fig_pp.png
       WDM solid, freq dashed.
  4. JSD table -> printed (median, q05, q95 per parameter).

Usage:  python plot_gb_study.py --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from lisa_gb_study import (
    A_WDM,
    D_WDM,
    POSTERIOR_LABELS,
    _make_grid,
    configure_production_env,
    injection_for_seed,
    make_jgb,
)

OUTDIR = Path(__file__).resolve().parent / "outdir_gb"
FREQ_COL, WDM_COL, TRUTH_COL = "tab:blue", "tab:orange", "black"
DATA_COL, SRC_COL, PSD_COL = "#B8B8B8", "#ff7f0e", "#172919"
LATEX = {"log10(f0 / Hz)": r"$\log_{10} f_0$", "fdot [Hz/s]": r"$\dot f$ [Hz/s]",
         "log10(A)": r"$\log_{10} A$", "phi0 [rad]": r"$\phi_0$ [rad]"}


def _one_sided_density(rfft, dt, n):
    psd = (2.0 * dt / n) * np.abs(rfft) ** 2
    if psd.size:
        psd[0] *= 0.5
    if n % 2 == 0 and psd.size > 1:
        psd[-1] *= 0.5
    return np.maximum(psd, 1e-60)


# ── 1. data overview ────────────────────────────────────────────────────────────


def plot_data(seed: int, grid: dict, jgb, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from wdm_transform import TimeSeries
    from wdm_transform.signal_processing import wdm_noise_variance

    inj = injection_for_seed(seed, grid=grid, jgb=jgb)
    n = grid["n_total"]; dt = inj["dt"]; freqs = inj["freqs"]; f0 = float(inj["source_params"][0])
    data_A = np.fft.irfft(inj["data_rfft"][0], n=n)
    psd_data = _one_sided_density(inj["data_rfft"][0], dt, n)
    psd_src = _one_sided_density(inj["signal_rfft"][0], dt, n)
    psd_inst = np.maximum(inj["psd_full"][0], 1e-60)
    k0, k1 = inj["band"]["kmin_rfft"], min(inj["band"]["kmax_rfft"], len(freqs) - 1)
    lo, hi = freqs[k0], freqs[k1]
    source_bins = np.flatnonzero(np.abs(inj["signal_rfft"][0]) > 0)
    source_lo, source_hi = freqs[source_bins[[0, -1]]]
    source_pad = 0.05 * (source_hi - source_lo)
    zoom_lo = max(lo, source_lo - source_pad)
    zoom_hi = min(hi, source_hi + source_pad)

    fig = plt.figure(figsize=(11, 8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.12)
    bottom_gs = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.22)

    # (a) frequency domain: data periodogram, instrument PSD, source; source band shaded.
    ax = fig.add_subplot(gs[0])
    ax.axvspan(lo, hi, color=SRC_COL, alpha=0.12, lw=0, zorder=0)
    ax.loglog(freqs[1:], np.where(psd_src[1:] > 1e-50, psd_src[1:], np.nan),
              color=SRC_COL, lw=1.6, label="Injected source", zorder=1)
    ax.loglog(freqs[1:], psd_data[1:], color=DATA_COL, lw=1.0, label="A data", zorder=2)
    ax.loglog(freqs[1:], psd_inst[1:], color=PSD_COL, lw=2.0, label="Instrument PSD", zorder=3)
    ax.set_xlim(1e-4, 3e-3); ax.set_ylim(1e-44, 1e-36)
    ax.set_xticks([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3])
    ax.set_xticklabels([r"$10^{-4}$", r"$2\times10^{-4}$", r"$5\times10^{-4}$",
                        r"$10^{-3}$", r"$2\times10^{-3}$", r"$3\times10^{-3}$"])
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]")
    ax.grid(True, which="both", alpha=0.18, ls=":")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (1, 2, 0)], [labels[i] for i in (1, 2, 0)],
              loc="upper right", frameon=False)
    ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    # (b) frequency-domain source-band zoom.
    ax2 = fig.add_subplot(bottom_gs[0])
    m = (freqs >= zoom_lo) & (freqs <= zoom_hi)
    ax2.semilogy(freqs[m] * 1e3, np.where(psd_src[m] > 1e-50, psd_src[m], np.nan),
                 color=SRC_COL, lw=1.8, zorder=1)
    ax2.semilogy(freqs[m] * 1e3, psd_data[m], color=DATA_COL, lw=1.0, zorder=2)
    ax2.semilogy(freqs[m] * 1e3, psd_inst[m], color=PSD_COL, lw=1.4, zorder=3)
    ax2.set_xlim(zoom_lo * 1e3, zoom_hi * 1e3)
    ax2.set_xlabel("Frequency (mHz)"); ax2.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]")
    ax2.grid(True, which="both", alpha=0.18, ls=":")
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, va="top", fontweight="bold")

    # (c) WDM time-frequency source-band zoom, whitened to ~unit noise.
    ax3 = fig.add_subplot(bottom_gs[1])
    cax = fig.add_subplot(bottom_gs[2])
    nt = 64
    wdm = TimeSeries(np.asarray(data_A, float), dt=dt).to_wdm(nt=nt, a=A_WDM, d=D_WDM)
    fg = np.asarray(wdm.freq_grid, float)
    nf = fg.size - 1
    noise_on_grid = np.interp(fg, np.linspace(0.0, float(wdm.nyquist), psd_inst.size), psd_inst)
    var = wdm_noise_variance(noise_on_grid, nt=nt, nf=nf, dt=dt)
    whiten = np.sqrt(np.where(np.isfinite(var) & (var > 0), var, np.inf))
    img = (np.abs(np.asarray(wdm.coeffs[0], float)) / whiten[None, :]).T  # (freq, time), ~N(0,1) noise
    tg = np.asarray(wdm.time_grid, float) / 86400.0
    norm = LogNorm(vmin=0.5, vmax=max(np.nanpercentile(img, 99.8), 3.0))
    im = ax3.imshow(np.nan_to_num(img), aspect="auto", extent=[tg[0], tg[-1], fg[0] * 1e3, fg[-1] * 1e3],
                    origin="lower", cmap="magma", norm=norm, interpolation="nearest", rasterized=True)
    df = max(10.0 * (fg[1] - fg[0]), 1.5e-5)
    ax3.set_ylim(max(1e-4, f0 - df) * 1e3, min(3e-3, f0 + df) * 1e3); ax3.set_xlim(tg[0], tg[-1])
    ax3.set_xlabel("Time (days)"); ax3.set_ylabel("Frequency (mHz)")
    ax3.text(0.02, 0.96, "(c)", transform=ax3.transAxes, va="top",
             color="white", fontweight="bold")
    cbar = fig.colorbar(im, cax=cax, label="|WDM coeff| / noise std")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    fig.canvas.draw()
    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0, cax.get_position().x1 - ax_pos.x0, ax_pos.height])
    fig.set_layout_engine(None)
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"[data] {out}")


# ── 2. comparison corner ─────────────────────────────────────────────────────────


def _recenter_circular(label, fs_col, ws_col, truth):
    """Unwrap a circular (phi) marginal around the combined posterior mean so the
    cluster is contiguous and centered instead of split across the +-pi cut."""
    if "phi" not in label.lower():
        return fs_col, ws_col, truth
    both = np.concatenate([fs_col, ws_col])
    center = float(np.arctan2(np.mean(np.sin(both)), np.mean(np.cos(both))))
    wrap = lambda x: center + ((x - center + np.pi) % (2 * np.pi) - np.pi)
    return wrap(fs_col), wrap(ws_col), float(wrap(np.array([truth]))[0])


def plot_corner(seed: int, out: Path) -> None:
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r = json.load(open(OUTDIR / f"seed_{seed}.json"))
    if not r.get("samples"):
        print(f"[corner] seed {seed} has no stored samples — skipping"); return
    labels = POSTERIOR_LABELS
    fs_cols, ws_cols, truth = [], [], []
    for lab in labels:
        f = np.asarray(r["samples"]["freq"][lab]); w = np.asarray(r["samples"]["wdm"][lab])
        t = next(x["truth"] for x in r["freq"] if x["label"] == lab)
        f, w, t = _recenter_circular(lab, f, w, t)
        fs_cols.append(f); ws_cols.append(w); truth.append(t)
    fs, ws = np.column_stack(fs_cols), np.column_stack(ws_cols)
    # Shared ranges with a little padding so both posteriors + truth are framed.
    rng = []
    for i, t in enumerate(truth):
        lo = min(fs[:, i].min(), ws[:, i].min(), t)
        hi = max(fs[:, i].max(), ws[:, i].max(), t)
        pad = 0.08 * (hi - lo + 1e-30)
        rng.append((lo - pad, hi + pad))
    latex = [LATEX[l] for l in labels]
    ckw = dict(labels=latex, range=rng, plot_datapoints=False, smooth=1.0, bins=30,
               levels=(0.5, 0.9))
    fig = corner.corner(
        fs,
        truths=truth,
        truth_color=TRUTH_COL,
        color=FREQ_COL,
        hist_kwargs={"density": True, "lw": 1.4, "color": FREQ_COL},
        **ckw,
    )
    corner.corner(
        ws,
        fig=fig,
        color=WDM_COL,
        hist_kwargs={"density": True, "lw": 1.4, "color": WDM_COL},
        **ckw,
    )
    axes = np.asarray(fig.axes).reshape(len(labels), len(labels))

    # Overlay the prior on each 1-D panel so it is visually obvious the posteriors
    # are far narrower than (i.e. dominated by data rather than) the prior.
    from lisa_gb_study import _make_grid, _prior_scales
    sc = _prior_scales(_make_grid()["t_obs"])
    f0t = next(x["truth"] for x in r["freq"] if x["label"] == "log10(f0 / Hz)")
    f0_lin = 10.0 ** f0t
    prior = {  # (center, std) in display units; None center => ~flat over the panel
        "log10(f0 / Hz)": (f0t, sc["delta_f0_sigma"] / (f0_lin * np.log(10))),
        "fdot [Hz/s]": (truth[labels.index("fdot [Hz/s]")], sc["delta_fdot_sigma"]),
        # amplitude prior is Rayleigh(3*A) -> log10(A) std ~0.31 (shape-invariant);
        # phi0 prior is exactly uniform (isotropic Gaussian on g_c,g_s) -> flat.
        "log10(A)": (next(x["truth"] for x in r["freq"] if x["label"] == "log10(A)"), 0.31),
        "phi0 [rad]": (None, 1.0),
    }
    for i, lab in enumerate(labels):
        ax = axes[i, i]; c, s = prior[lab]
        x0, x1 = ax.get_xlim(); xg = np.linspace(x0, x1, 200)
        pdf = np.ones_like(xg) if c is None else np.exp(-0.5 * ((xg - c) / s) ** 2)
        ax.plot(xg, pdf / pdf.max() * ax.get_ylim()[1] * 0.95, color="0.45", ls=":", lw=1.4, zorder=0)

    axes[0, -1].legend(handles=[Patch(color=FREQ_COL, label=f"Frequency (SNR={r['snr']:.1f})"),
                                Patch(color=WDM_COL, label="WDM"),
                                plt.Line2D([0], [0], color="0.45", ls=":", label="Prior"),
                                plt.Line2D([0], [0], color=TRUTH_COL, ls="-", label="Truth")],
                       loc="upper right", frameon=False, fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[corner] {out}")


# ── 3. PP plot (all params, WDM solid / freq dashed) ─────────────────────────────


def _load_all() -> list[dict]:
    files = sorted(glob.glob(str(OUTDIR / "seed_*.json")), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return [json.load(open(f)) for f in files]


def plot_pp(res: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(0, 1, 200)
    colors = {lab: f"C{i}" for i, lab in enumerate(POSTERIOR_LABELS)}
    n = len(res)
    band = np.sqrt(x * (1 - x) / n) * 1.96  # ~95% pointwise band for uniform
    ax.fill_between(x, np.clip(x - band, 0, 1), np.clip(x + band, 0, 1), color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.9, ls=":", zorder=1)
    for lab in POSTERIOR_LABELS:
        for dom, ls in (("wdm", "-"), ("freq", "--")):
            ranks = np.sort([next(x2["rank"] for x2 in r[dom] if x2["label"] == lab) for r in res])
            cdf = np.searchsorted(ranks, x, side="right") / n
            ax.plot(x, cdf, ls=ls, color=colors[lab], lw=1.6,
                    label=(LATEX[lab] if dom == "wdm" else None))
    ax.set_xlabel("credible level"); ax.set_ylabel("fraction of truths below")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    leg1 = ax.legend(loc="upper left", frameon=False, title="parameter")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", frameon=False, handles=[
        plt.Line2D([0], [0], color="0.3", ls="-", label="WDM"),
        plt.Line2D([0], [0], color="0.3", ls="--", label="Frequency")])
    ax.set_title(f"PP plot ({n} seeds)")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[pp] {out}")


# ── 4. JSD table ─────────────────────────────────────────────────────────────────


def _sample_jsd_bits(a, b):
    """JSD (bits) between two 1-D sample sets via KDE densities on a shared grid.

    KDE avoids the finite-sample noise floor of fixed-bin histograms, so two
    samplings of the same posterior read ~0 instead of a spurious ~0.03 bits."""
    from scipy.stats import gaussian_kde

    a, b = np.asarray(a), np.asarray(b)
    if a.std() == 0 and b.std() == 0:
        return 0.0
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.1 * (hi - lo + 1e-30)
    x = np.linspace(lo - pad, hi + pad, 1000)
    try:
        p = gaussian_kde(a)(x); q = gaussian_kde(b)(x)
    except np.linalg.LinAlgError:
        return 0.0
    p /= np.trapezoid(p, x); q /= np.trapezoid(q, x); m = 0.5 * (p + q)
    kl = lambda u, v: np.trapezoid(np.where(u > 0, u * np.log2(np.where(u > 0, u, 1) / np.where(v > 0, v, 1)), 0.0), x)
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0))


def jsd_table(res: list[dict]) -> None:
    have = [r for r in res if r.get("samples")]
    print(f"\nfreq-vs-WDM JSD (bits), KDE estimate, over {len(have)} seeds  (0=identical, 1=disjoint):")
    print(f"{'parameter':<16}{'median':>10}{'q05':>10}{'q95':>10}")
    for lab in POSTERIOR_LABELS:
        J = np.array([_sample_jsd_bits(r["samples"]["freq"][lab], r["samples"]["wdm"][lab]) for r in have])
        print(f"{lab:<16}{np.median(J):>10.4g}{np.quantile(J, 0.05):>10.4g}{np.quantile(J, 0.95):>10.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="seed for the data + corner figures")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    configure_production_env()
    grid = _make_grid()
    jgb = make_jgb(grid)

    plot_data(args.seed, grid, jgb, OUTDIR / f"fig_data_seed{args.seed}.png")
    plot_corner(args.seed, OUTDIR / f"fig_corner_seed{args.seed}.png")
    res = _load_all()
    plot_pp(res, OUTDIR / "fig_pp.png")
    jsd_table(res)


if __name__ == "__main__":
    main()
