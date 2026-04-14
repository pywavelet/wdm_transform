"""Build multi-seed PP plots for the LISA frequency and WDM posterior runs."""

from __future__ import annotations

import argparse
import atexit
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D

from lisa_common import OUTDIR_ROOT, lisa_mode_dirname, save_figure

FREQ_FILENAME = "freq_posteriors.npz"
WDM_FILENAME = "wdm_posteriors.npz"
DERIVED_LABELS = {"snr"}
CONFIDENCE_INTERVALS = (0.68, 0.95, 0.997)
SCRIPT_START = time.perf_counter()


def _print_runtime() -> None:
    elapsed = time.perf_counter() - SCRIPT_START
    print(f"\n[pp_plot.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)


@dataclass(frozen=True)
class SeedPosterior:
    seed: int
    path: Path
    labels: list[str]
    samples: np.ndarray
    truth: np.ndarray


def _parse_seed(seed_dir: Path) -> int:
    return int(seed_dir.name.split("_", maxsplit=1)[1])


def _normalize_phi(values: np.ndarray, label: str) -> np.ndarray:
    if "phi" not in label.lower() and "phase" not in label.lower():
        return np.asarray(values, dtype=float)
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _is_sampled_parameter(label: str) -> bool:
    lowered = label.strip().lower()
    return lowered not in DERIVED_LABELS and "snr" not in lowered


def _short_label(label: str) -> str:
    return (
        label.replace(" [Hz]", "")
        .replace(" [Hz/s]", "")
        .replace(" [1/s]", "")
        .replace(" [rad]", "")
        .strip()
    )


def load_seed_posterior(path: Path) -> SeedPosterior:
    with np.load(path) as data:
        labels = [str(item) for item in np.asarray(data["labels"]).tolist()]
        samples = np.asarray(data["samples_source"], dtype=float)
        truth = np.asarray(data["truth"], dtype=float).reshape(-1)

    if samples.ndim != 2:
        raise ValueError(f"{path} has invalid samples_source shape {samples.shape}.")
    if samples.shape[1] != len(labels):
        raise ValueError(f"{path} has {samples.shape[1]} sample columns for {len(labels)} labels.")
    if truth.size != len(labels):
        raise ValueError(f"{path} has {truth.size} truth values for {len(labels)} labels.")

    keep_idx = [idx for idx, label in enumerate(labels) if _is_sampled_parameter(label)]
    if not keep_idx:
        raise ValueError(f"{path} has no sampled parameters after filtering derived labels.")

    kept_labels = [labels[idx] for idx in keep_idx]
    kept_samples = samples[:, keep_idx].copy()
    kept_truth = truth[keep_idx].copy()
    for idx, label in enumerate(kept_labels):
        kept_samples[:, idx] = _normalize_phi(kept_samples[:, idx], label)
        kept_truth[idx] = float(_normalize_phi(np.array([kept_truth[idx]]), label)[0])

    return SeedPosterior(
        seed=_parse_seed(path.parent),
        path=path,
        labels=kept_labels,
        samples=kept_samples,
        truth=kept_truth,
    )


def discover_seed_dirs(mode_dir: Path) -> list[Path]:
    seed_dirs = sorted((path for path in mode_dir.glob("seed_*") if path.is_dir()), key=_parse_seed)
    return seed_dirs


def collect_runs(mode_dir: Path, seed_dirs: list[Path]) -> tuple[list[SeedPosterior], list[SeedPosterior], list[int]]:
    freq_runs: list[SeedPosterior] = []
    wdm_runs: list[SeedPosterior] = []
    used_seeds: list[int] = []

    for seed_dir in seed_dirs:
        freq_path = seed_dir / FREQ_FILENAME
        wdm_path = seed_dir / WDM_FILENAME
        seed = _parse_seed(seed_dir)
        if not freq_path.exists() or not wdm_path.exists():
            missing = []
            if not freq_path.exists():
                missing.append(FREQ_FILENAME)
            if not wdm_path.exists():
                missing.append(WDM_FILENAME)
            print(f"Skipping seed {seed}: missing {', '.join(missing)}")
            continue

        freq_runs.append(load_seed_posterior(freq_path))
        wdm_runs.append(load_seed_posterior(wdm_path))
        used_seeds.append(seed)

    if not used_seeds:
        raise FileNotFoundError(f"No seed directories in {mode_dir} contained both posterior archives.")

    return freq_runs, wdm_runs, used_seeds


def common_labels(runs: list[SeedPosterior]) -> list[str]:
    labels = list(runs[0].labels)
    for run in runs[1:]:
        labels = [label for label in labels if label in run.labels]
    if not labels:
        raise ValueError("No shared sampled labels across the selected runs.")
    return labels


def align_run(run: SeedPosterior, labels: list[str]) -> SeedPosterior:
    idx = [run.labels.index(label) for label in labels]
    return SeedPosterior(
        seed=run.seed,
        path=run.path,
        labels=labels,
        samples=run.samples[:, idx],
        truth=run.truth[idx],
    )


def calculate_credible_levels(posterior_samples: list[np.ndarray], true_values: list[float]) -> np.ndarray:
    credible_levels = []
    for samples, true_value in zip(posterior_samples, true_values, strict=True):
        credible_levels.append(float(np.mean(np.asarray(samples, dtype=float) < float(true_value))))
    return np.asarray(credible_levels, dtype=float)


def empirical_pp(credible_levels: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(credible_levels <= x) for x in x_values], dtype=float)


def build_results(runs: list[SeedPosterior], labels: list[str]) -> dict[str, dict[str, list[np.ndarray] | list[float]]]:
    results: dict[str, dict[str, list[np.ndarray] | list[float]]] = {
        label: {"posteriors": [], "trues": []}
        for label in labels
    }
    for run in runs:
        aligned = align_run(run, labels)
        for idx, label in enumerate(labels):
            results[label]["posteriors"].append(aligned.samples[:, idx])
            results[label]["trues"].append(float(aligned.truth[idx]))
    return results


def print_summary(results: dict[str, dict[str, list[np.ndarray] | list[float]]], *, name: str) -> dict[str, np.ndarray]:
    print(f"\n{name} PP summary")
    print(f"{'Parameter':<20} {'mean(CL)':>10} {'std(CL)':>10} {'KS p':>12}")
    print("-" * 56)
    credible_map: dict[str, np.ndarray] = {}
    p_values = []
    for label, data in results.items():
        credible_levels = calculate_credible_levels(
            data["posteriors"],  # type: ignore[arg-type]
            data["trues"],       # type: ignore[arg-type]
        )
        credible_map[label] = credible_levels
        p_value = float(st.kstest(credible_levels, "uniform").pvalue)
        p_values.append(p_value)
        print(
            f"{_short_label(label):<20} "
            f"{credible_levels.mean():>10.3f} "
            f"{credible_levels.std(ddof=0):>10.3f} "
            f"{p_value:>12.3g}"
        )
    combined_p = float(st.combine_pvalues(p_values)[1])
    print(f"{'combined':<20} {'':>10} {'':>10} {combined_p:>12.3g}")
    return credible_map


def plot_pp_compare(
    *,
    freq_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    wdm_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    labels: list[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> Path:
    x_values = np.linspace(0.0, 1.0, 201)
    n_sims = len(next(iter(freq_results.values()))["trues"])
    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)

    for ci in CONFIDENCE_INTERVALS:
        edge = 0.5 * (1.0 - ci)
        lower = st.binom.ppf(edge, n_sims, x_values) / n_sims
        upper = st.binom.ppf(1.0 - edge, n_sims, x_values) / n_sims
        ax.fill_between(
            x_values,
            lower,
            upper,
            color="0.7",
            alpha=0.12,
            label=f"{int(ci * 100)}% interval" if ci == 0.95 else None,
        )

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(labels), 2)))
    freq_pvalues = []
    wdm_pvalues = []

    for color, label in zip(colors, labels, strict=True):
        freq_cls = calculate_credible_levels(
            freq_results[label]["posteriors"],  # type: ignore[arg-type]
            freq_results[label]["trues"],       # type: ignore[arg-type]
        )
        wdm_cls = calculate_credible_levels(
            wdm_results[label]["posteriors"],   # type: ignore[arg-type]
            wdm_results[label]["trues"],        # type: ignore[arg-type]
        )
        freq_pvalues.append(float(st.kstest(freq_cls, "uniform").pvalue))
        wdm_pvalues.append(float(st.kstest(wdm_cls, "uniform").pvalue))
        ax.plot(x_values, empirical_pp(wdm_cls, x_values), color=color, lw=2.0, ls="-")
        ax.plot(x_values, empirical_pp(freq_cls, x_values), color=color, lw=2.0, ls="--")

    ax.plot([0.0, 1.0], [0.0, 1.0], color="k", ls=":", lw=1.5, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted Credible Level")
    ax.set_ylabel("Observed Fraction")
    ax.set_title(
        f"{title}\n"
        f"WDM combined p={st.combine_pvalues(wdm_pvalues)[1]:.3g}, "
        f"Frequency combined p={st.combine_pvalues(freq_pvalues)[1]:.3g}"
    )

    parameter_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=_short_label(label))
        for color, label in zip(colors, labels, strict=True)
    ]
    method_handles = [
        Line2D([0], [0], color="k", lw=2.0, ls="-", label="WDM"),
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="Frequency"),
        Line2D([0], [0], color="k", lw=1.5, ls=":", label="Ideal"),
    ]
    legend_params = ax.legend(handles=parameter_handles, loc="lower right", title="Parameters", fontsize=9)
    ax.add_artist(legend_params)
    ax.legend(handles=method_handles, loc="upper left", title="Curves", fontsize=9)

    return save_figure(fig, output_dir, stem, dpi=200)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        default=lisa_mode_dirname(),
        help="Study mode directory under outdir_lisa (default: current LISA_INCLUDE_GALACTIC mode).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the PP plot (default: mode directory).")
    parser.add_argument("--stem", type=str, default="posterior_pp_compare", help="Output figure stem.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode_dir = OUTDIR_ROOT / args.mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode directory not found: {mode_dir}")

    output_dir = args.output_dir if args.output_dir is not None else mode_dir
    seed_dirs = discover_seed_dirs(mode_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No matching seed directories found in {mode_dir}")

    freq_runs, wdm_runs, used_seeds = collect_runs(mode_dir, seed_dirs)
    labels = common_labels(freq_runs + wdm_runs)
    freq_results = build_results(freq_runs, labels)
    wdm_results = build_results(wdm_runs, labels)

    print(f"Using mode: {args.mode}")
    print(f"Seeds: {used_seeds}")
    print_summary(wdm_results, name="WDM")
    print_summary(freq_results, name="Frequency")

    out_path = plot_pp_compare(
        freq_results=freq_results,
        wdm_results=wdm_results,
        labels=labels,
        output_dir=output_dir,
        stem=args.stem,
        title=f"LISA Posterior PP Plot ({args.mode}, N={len(used_seeds)})",
    )
    print(f"\nSaved PP plot to {out_path}")


if __name__ == "__main__":
    main()
