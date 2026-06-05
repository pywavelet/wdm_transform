"""Run a range of seeds in one process and report convergence + freq/WDM JSD.

Usage:  python eval_seeds.py [N] [START]   (default N=20, START=0)

Prints a per-seed table (max R-hat, min ESS, divergences, max JSD over params)
and a summary: how many seeds are converged (R-hat<1.05, no divergences) and
how many have small freq-vs-WDM JSD (<0.05).
"""
import sys
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import lisa_gb_study as L  # noqa: E402
import plot_gb_study as P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0

L.configure_production_env()
grid = L._make_grid()
jgb = L.make_jgb(grid)
print(f"channels={L.CHANNELS} chains={L.NUM_CHAINS} warmup={L.N_WARMUP} draws={L.N_DRAWS} "
      f"fdot_drift_bins={L.FDOT_DRIFT_BINS}")
print(f"{'seed':>4} {'SNR':>5} {'fdotbin':>8} {'maxRhat':>8} {'minESS':>7} {'div':>4} {'maxJSD':>7}  flag")

rows = []
for seed in range(START, START + N):
    r = L.run_one_seed(seed, grid=grid, jgb=jgb)
    rhat = max(max(r["diagnostics"][d]["rhat"].values()) for d in ("freq", "wdm"))
    ess = min(min(r["diagnostics"][d]["ess"].values()) for d in ("freq", "wdm"))
    div = sum(r["diagnostics"][d]["divergences"] for d in ("freq", "wdm"))
    jsd = max(P._sample_jsd_bits(r["samples"]["freq"][lab], r["samples"]["wdm"][lab])
              for lab in L.POSTERIOR_LABELS)
    fbin = r["truth"]["fdot"] * grid["t_obs"] ** 2
    conv = rhat < 1.05 and div == 0
    flag = "ok" if conv and jsd < 0.05 else ("JSD" if conv else "RHAT")
    rows.append((seed, r["snr"], fbin, rhat, ess, div, jsd, conv))
    print(f"{seed:>4} {r['snr']:>5.1f} {fbin:>+8.1f} {rhat:>8.3f} {ess:>7.0f} {div:>4} {jsd:>7.3f}  {flag}")
    jax.clear_caches()

rh = np.array([x[3] for x in rows]); jsd = np.array([x[6] for x in rows]); conv = np.array([x[7] for x in rows])
print(f"\n=== {N} seeds ===")
print(f"converged (R-hat<1.05, 0 div): {conv.sum()}/{N}")
print(f"small JSD (<0.05): {(jsd < 0.05).sum()}/{N}   median maxJSD={np.median(jsd):.4f}  worst={jsd.max():.3f}")
print(f"R-hat: median={np.median(rh):.3f}  worst={rh.max():.3f}")
