"""Collect WDM-vs-frequency JSD diagnostics across seeded LISA runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lisa_common import OUTDIR_ROOT, lisa_mode_dirname


def _mode_dir(mode: str) -> Path:
    if mode not in {"galactic_background", "stationary_noise"}:
        raise ValueError(f"Unknown mode {mode!r}.")
    return OUTDIR_ROOT / mode


def load_jsd_rows(mode: str, start_seed: int, end_seed: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    mode_dir = _mode_dir(mode)
    for seed in range(start_seed, end_seed + 1):
        diagnostics_path = mode_dir / f"seed_{seed}" / "posterior_diagnostics.json"
        if not diagnostics_path.exists():
            print(f"Skipping seed {seed}: missing {diagnostics_path}")
            continue

        with diagnostics_path.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        comparison = diagnostics.get("comparison", {})
        for item in comparison.get("marginal_jsd", []):
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "label": str(item["label"]),
                    "jsd_bits": float(item["jsd_bits"]),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "seed", "label", "jsd_bits"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    labels = sorted({str(row["label"]) for row in rows})
    summary: dict[str, object] = {"n_rows": len(rows), "parameters": {}}
    for label in labels:
        values = np.array([float(row["jsd_bits"]) for row in rows if row["label"] == label])
        summary["parameters"][label] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "q05": float(np.percentile(values, 5.0)),
            "q95": float(np.percentile(values, 95.0)),
            "max": float(np.max(values)),
        }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_histograms(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    labels = sorted({str(row["label"]) for row in rows})
    if not labels:
        return

    ncols = min(3, len(labels))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, label in zip(axes.ravel(), labels):
        ax.set_visible(True)
        values = np.array([float(row["jsd_bits"]) for row in rows if row["label"] == label])
        ax.hist(values, bins="auto", color="tab:blue", alpha=0.75, edgecolor="white")
        ax.axvline(np.median(values), color="black", lw=1.5, label="median")
        ax.set_title(label)
        ax.set_xlabel("JSD [bits]")
        ax.set_ylabel("count")
        ax.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=lisa_mode_dirname(), choices=["galactic_background", "stationary_noise"])
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=99)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (_mode_dir(args.mode) / "summary")
    rows = load_jsd_rows(args.mode, args.start_seed, args.end_seed)
    if not rows:
        raise SystemExit("No JSD diagnostics found.")

    csv_path = output_dir / "jsd_marginals.csv"
    summary_path = output_dir / "jsd_marginals_summary.json"
    hist_path = output_dir / "jsd_marginals_hist.png"
    write_csv(rows, csv_path)
    write_summary(rows, summary_path)
    plot_histograms(rows, hist_path)

    print(f"Collected {len(rows)} JSD rows from {args.mode} seeds {args.start_seed}..{args.end_seed}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {hist_path}")


if __name__ == "__main__":
    main()
