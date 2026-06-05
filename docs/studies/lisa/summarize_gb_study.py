"""Summarize a finished GB study: calibration + freq-vs-WDM agreement + PP plot.

Reads outdir_gb/seed_*.json (written by lisa_gb_study.py) and prints, per
parameter and domain, the mean truth-rank (~0.5 = unbiased), the worst |z|
(truth inside the posterior if small), the freq-vs-WDM posterior-mean offset in
sigma, and the freq-vs-WDM JSD (exact histogram JSD from stored samples when
present, else a Gaussian-marginal approximation).  Regenerates outdir_gb/pp_plot.png.

Usage:  python summarize_gb_study.py
"""
import glob
import json
from pathlib import Path

import numpy as np

from lisa_gb_study import POSTERIOR_LABELS, make_pp_plot

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _gauss_jsd_bits(m1, s1, m2, s2):
    """Jensen-Shannon divergence (bits) between two Gaussian marginals.

    Gaussian approximation from the stored (mean, std); 0 = identical, 1 =
    disjoint.  Overstates the divergence for sharp/overconfident posteriors
    whose true shape is non-Gaussian (exact JSD needs the samples)."""
    if s1 <= 0 or s2 <= 0:
        return 0.0
    lo, hi = min(m1 - 6 * s1, m2 - 6 * s2), max(m1 + 6 * s1, m2 + 6 * s2)
    x = np.linspace(lo, hi, 2000)
    p = np.exp(-0.5 * ((x - m1) / s1) ** 2); p /= _TRAPZ(p, x)
    q = np.exp(-0.5 * ((x - m2) / s2) ** 2); q /= _TRAPZ(q, x)
    mix = 0.5 * (p + q)
    kl = lambda a, b: _TRAPZ(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0.0), x)
    return 0.5 * kl(p, mix) + 0.5 * kl(q, mix)


def _hist_jsd_bits(a, b, nbins=40):
    """Exact JSD (bits) between two sample sets via shared-bin histograms."""
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi - lo <= 0:
        return 0.0
    edges = np.linspace(lo, hi, nbins + 1)
    p = np.histogram(a, edges)[0] / a.size
    q = np.histogram(b, edges)[0] / b.size
    m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(y > 0, y, 1)), 0.0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _jsd_for(r, lab):
    """KDE JSD from stored samples if present, else Gaussian-marginal approx."""
    if r.get("samples"):
        from plot_gb_study import _sample_jsd_bits  # KDE estimator (no histogram floor)
        return _sample_jsd_bits(r["samples"]["freq"][lab], r["samples"]["wdm"][lab])
    fr = next(x for x in r["freq"] if x["label"] == lab)
    wd = next(x for x in r["wdm"] if x["label"] == lab)
    return _gauss_jsd_bits(fr["mean"], fr["std"], wd["mean"], wd["std"])


def _convergence_report(res):
    """Per-seed R-hat / divergences if recorded; flag any non-converged seed."""
    have = [r for r in res if r.get("diagnostics")]
    if not have:
        print("\n(no convergence diagnostics recorded in these runs)")
        return
    worst, bad = [], []
    for r in have:
        rh = max(max(r["diagnostics"][d]["rhat"].values()) for d in ("freq", "wdm"))
        div = sum(r["diagnostics"][d]["divergences"] for d in ("freq", "wdm"))
        worst.append(rh)
        if rh >= 1.05 or div > 0:
            bad.append((r["seed"], rh, div))
    worst = np.array(worst)
    print(f"\nconvergence ({len(have)} seeds): converged (R-hat<1.05, 0 div) = "
          f"{len(have) - len(bad)}/{len(have)}  |  median R-hat={np.median(worst):.3f}  worst={worst.max():.3f}")
    for seed, rh, div in bad:
        print(f"   NOT CONVERGED: seed {seed}  max R-hat={rh:.3f}  div={div}")


def _config_check(res):
    """Warn if the aggregated seeds were not all produced with the same config."""
    cfgs = {json.dumps(r.get("config"), sort_keys=True) for r in res if r.get("config")}
    if len(cfgs) > 1:
        print(f"\n!! WARNING: {len(cfgs)} different run configs mixed in outdir_gb -- "
              "results may be incomparable. Re-run a clean batch.")


OUTDIR = Path(__file__).resolve().parent / "outdir_gb"
files = sorted(glob.glob(str(OUTDIR / "seed_*.json")),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
res = [json.load(open(f)) for f in files]
snrs = [r["snr"] for r in res]
print(f"{len(res)} seeds  |  SNR {min(snrs):.0f}-{max(snrs):.0f}")
print(f"{'param':<16} {'freq rank':>10} {'wdm rank':>9} {'max|z| f/w':>12} {'freq-wdm Δ':>14}")
for lab in POSTERIOR_LABELS:
    def col(dom, key):
        return np.array([next(x[key] for x in r[dom] if x["label"] == lab) for r in res])
    fr_rank, wd_rank = col("freq", "rank").mean(), col("wdm", "rank").mean()
    zmax_f, zmax_w = np.abs(col("freq", "z")).max(), np.abs(col("wdm", "z")).max()
    deltas = []
    for r in res:
        fr = next(x for x in r["freq"] if x["label"] == lab)
        wd = next(x for x in r["wdm"] if x["label"] == lab)
        s = 0.5 * (fr["std"] + wd["std"])
        if s > 0:
            deltas.append(abs(fr["mean"] - wd["mean"]) / s)
    print(f"{lab:<16} {fr_rank:>10.3f} {wd_rank:>9.3f} {zmax_f:>5.1f}/{zmax_w:<5.1f} "
          f"{np.mean(deltas):>6.2f}σ (max {np.max(deltas):.2f})")

# freq-vs-WDM divergence.  Exact histogram JSD when samples are stored, else
# a Gaussian-marginal approximation of the stored (mean, std).
exact = bool(res and res[0].get("samples"))
print(f"\nfreq vs WDM JSD bits ({'exact, from samples' if exact else 'Gaussian approx'}; "
      f"0=identical, 1=disjoint)")
print(f"{'param':<16} {'median':>9} {'mean':>9} {'max':>9}")
jsd_by_lab = {lab: np.array([_jsd_for(r, lab) for r in res]) for lab in POSTERIOR_LABELS}
for lab, J in jsd_by_lab.items():
    print(f"{lab:<16} {np.median(J):>9.4f} {np.mean(J):>9.4f} {np.max(J):>9.4f}")
max_per_seed = np.max(np.stack(list(jsd_by_lab.values())), axis=0)
print(f"\nper-seed max-over-params JSD:  <0.02 (near-identical): {(max_per_seed < 0.02).sum()}/{len(res)}"
      f"  |  >=0.1 (visibly diff): {(max_per_seed >= 0.1).sum()}/{len(res)}")

_convergence_report(res)
_config_check(res)
make_pp_plot(res, OUTDIR / "pp_plot.png")
print(f"PP plot -> {OUTDIR / 'pp_plot.png'}")
