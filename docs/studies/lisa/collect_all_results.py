"""Aggregate per-seed LISA study results into PP plots and a JSD summary.

Each ``lisa_study.py`` run writes ``results.json`` into its seed directory.
This script loads every available seed and produces:

* ``pp_plot.png`` — probability-probability calibration plot of the posterior
  rank of the injected truth, drawn separately for the frequency- and
  WDM-domain samplers,
* ``jsd_histogram.png`` — distribution of the per-parameter frequency-vs-WDM
  Jensen-Shannon divergence across seeds,
* ``summary.csv`` / ``summary.json`` — the flattened per-seed, per-parameter
  table plus aggregate statistics.

Run from the study directory::

    python collect_all_results.py --start-seed 0 --end-seed 99
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lisa_common import OUTDIR_ROOT

PARAM_LABELS = ["log10(f0 / Hz)", "log10(fdot / Hz/s)", "log10(A)", "phi0 [rad]"]


def load_results(start_seed: int, end_seed: int) -> list[dict]:
    """Load every available ``results.json`` within the seed range."""
    rows: list[dict] = []
    for seed in range(start_seed, end_seed + 1):
        path = OUTDIR_ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def _ranks_by_param(results: list[dict], key: str) -> dict[str, np.ndarray]:
    """Collect the per-parameter truth-rank arrays across seeds for *key*."""
    out: dict[str, list[float]] = {label: [] for label in PARAM_LABELS}
    for res in results:
        for param in res["parameters"]:
            if param["label"] in out and param.get(key) is not None:
                out[param["label"]].append(float(param[key]))
    return {label: np.asarray(values, dtype=float) for label, values in out.items()}


def _pp_axis(ax, ranks_by_param: dict[str, np.ndarray], title: str) -> None:
    """Draw one PP panel: empirical CDF of truth-ranks vs the uniform diagonal."""
    n = max((len(v) for v in ranks_by_param.values()), default=0)
    grid = np.linspace(0.0, 1.0, 200)
    if n > 0:
        # Pointwise Gaussian confidence envelopes for a uniform sample of size n.
        sigma = np.sqrt(np.clip(grid * (1.0 - grid), 0.0, None) / n)
        for k, alpha in zip((1, 2, 3), (0.3, 0.2, 0.1), strict=True):
            ax.fill_between(grid, np.clip(grid - k * sigma, 0, 1),
                            np.clip(grid + k * sigma, 0, 1), color="0.6", alpha=alpha, lw=0)
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--")
    for label, ranks in ranks_by_param.items():
        if ranks.size == 0:
            continue
        sorted_ranks = np.sort(ranks)
        empirical = np.arange(1, sorted_ranks.size + 1) / sorted_ranks.size
        ax.plot(sorted_ranks, empirical, marker=".", ms=4, lw=1.0, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Posterior rank of truth")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{title}  (n={n})")
    ax.legend(loc="upper left", fontsize=8)


def make_pp_plot(results: list[dict], output_dir: Path) -> Path:
    freq_ranks = _ranks_by_param(results, "freq_rank")
    wdm_ranks = _ranks_by_param(results, "wdm_rank")
    fig, (ax_freq, ax_wdm) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    _pp_axis(ax_freq, freq_ranks, "Frequency-domain")
    _pp_axis(ax_wdm, wdm_ranks, "WDM-domain")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pp_plot.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_jsd_histogram(results: list[dict], output_dir: Path) -> Path:
    jsd_by_param: dict[str, list[float]] = {label: [] for label in PARAM_LABELS}
    for res in results:
        for param in res["parameters"]:
            if param["label"] in jsd_by_param:
                jsd_by_param[param["label"]].append(float(param["jsd_bits"]))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, values in jsd_by_param.items():
        if values:
            ax.hist(values, bins=20, histtype="step", lw=1.5, label=label)
    ax.set_xlabel("Frequency-vs-WDM Jensen-Shannon divergence (bits)")
    ax.set_ylabel("Seed count")
    ax.set_title("Marginal posterior agreement across seeds")
    ax.legend(fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "jsd_histogram.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "snr_aet", "parameter", "jsd_bits",
                         "truth", "freq_mean", "wdm_mean", "freq_rank", "wdm_rank"])
        for res in results:
            for param in res["parameters"]:
                writer.writerow([
                    res["seed"], res.get("snr_aet"), param["label"], param["jsd_bits"],
                    param["truth"], param["freq_mean"], param["wdm_mean"],
                    param["freq_rank"], param["wdm_rank"],
                ])

    aggregate: dict[str, dict[str, float]] = {}
    for label in PARAM_LABELS:
        jsd = np.array([p["jsd_bits"] for r in results for p in r["parameters"] if p["label"] == label])
        if jsd.size:
            aggregate[label] = {
                "n_seeds": int(jsd.size),
                "jsd_mean": float(np.mean(jsd)),
                "jsd_median": float(np.median(jsd)),
                "jsd_max": float(np.max(jsd)),
            }
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps({"n_seeds": len(results), "per_parameter": aggregate}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=99)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (OUTDIR_ROOT / "_summary")
    results = load_results(args.start_seed, args.end_seed)
    if not results:
        print(f"No results.json found under {OUTDIR_ROOT} for seeds {args.start_seed}..{args.end_seed}.")
        return
    print(f"Loaded {len(results)} seed result(s).")
    pp_path = make_pp_plot(results, output_dir)
    jsd_path = make_jsd_histogram(results, output_dir)
    csv_path, json_path = write_summary(results, output_dir)
    for path in (pp_path, jsd_path, csv_path, json_path):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
