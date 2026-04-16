"""Build multi-seed PP plots for the LISA frequency and WDM posterior runs."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D

from lisa_common import (
    OUTDIR_ROOT,
    circular_credible_level,
    credible_level,
    is_phase_parameter,
    lisa_mode_dirname,
    save_figure,
)

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


@dataclass(frozen=True)
class SkippedSeed:
    seed: int
    missing: list[str]


def _parse_seed(seed_dir: Path) -> int:
    return int(seed_dir.name.split("_", maxsplit=1)[1])


def _normalize_phi(values: np.ndarray, label: str) -> np.ndarray:
    if not is_phase_parameter(label):
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


def collect_runs(
    mode_dir: Path,
    seed_dirs: list[Path],
) -> tuple[list[SeedPosterior], list[SeedPosterior], list[int], list[SkippedSeed]]:
    freq_runs: list[SeedPosterior] = []
    wdm_runs: list[SeedPosterior] = []
    used_seeds: list[int] = []
    skipped: list[SkippedSeed] = []

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
            skipped.append(SkippedSeed(seed=seed, missing=missing))
            continue

        freq_runs.append(load_seed_posterior(freq_path))
        wdm_runs.append(load_seed_posterior(wdm_path))
        used_seeds.append(seed)

    if not used_seeds:
        raise FileNotFoundError(f"No seed directories in {mode_dir} contained both posterior archives.")

    return freq_runs, wdm_runs, used_seeds, skipped


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


def calculate_credible_levels(
    posterior_samples: list[np.ndarray],
    true_values: list[float],
    *,
    circular: bool = False,
) -> np.ndarray:
    credible_levels = []
    for samples, true_value in zip(posterior_samples, true_values, strict=True):
        if circular:
            credible_levels.append(circular_credible_level(samples, true_value))
        else:
            credible_levels.append(credible_level(samples, true_value))
    return np.asarray(credible_levels, dtype=float)


def empirical_pp(credible_levels: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(credible_levels <= x) for x in x_values], dtype=float)


def build_results(runs: list[SeedPosterior], labels: list[str]) -> dict[str, dict[str, list[np.ndarray] | list[float]]]:
    results: dict[str, dict[str, list[np.ndarray] | list[float]]] = {
        label: {"posteriors": [], "trues": [], "seeds": []}
        for label in labels
    }
    for run in runs:
        aligned = align_run(run, labels)
        for idx, label in enumerate(labels):
            results[label]["posteriors"].append(aligned.samples[:, idx])
            results[label]["trues"].append(float(aligned.truth[idx]))
            results[label]["seeds"].append(aligned.seed)
    return results


def summarize_credible_levels(credible_levels: np.ndarray) -> dict[str, float | int]:
    return {
        "mean_cl": float(credible_levels.mean()),
        "std_cl": float(credible_levels.std(ddof=0)),
        "ks_p": float(st.kstest(credible_levels, "uniform").pvalue),
        "inside_68": int(np.sum((credible_levels >= 0.16) & (credible_levels <= 0.84))),
        "inside_90": int(np.sum((credible_levels >= 0.05) & (credible_levels <= 0.95))),
        "inside_95": int(np.sum((credible_levels >= 0.025) & (credible_levels <= 0.975))),
    }


def print_summary(
    results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    *,
    name: str,
    printer=print,
) -> dict[str, np.ndarray]:
    printer(f"\n{name} PP summary")
    printer(f"{'Parameter':<20} {'mean(CL)':>10} {'std(CL)':>10} {'KS p':>12}")
    printer("-" * 56)
    credible_map: dict[str, np.ndarray] = {}
    p_values = []
    for label, data in results.items():
        credible_levels = calculate_credible_levels(
            data["posteriors"],  # type: ignore[arg-type]
            data["trues"],       # type: ignore[arg-type]
            circular=is_phase_parameter(label),
        )
        credible_map[label] = credible_levels
        p_value = float(summarize_credible_levels(credible_levels)["ks_p"])
        p_values.append(p_value)
        printer(
            f"{_short_label(label):<20} "
            f"{credible_levels.mean():>10.3f} "
            f"{credible_levels.std(ddof=0):>10.3f} "
            f"{p_value:>12.3g}"
        )
    combined_p = float(st.combine_pvalues(p_values)[1])
    printer(f"{'combined':<20} {'':>10} {'':>10} {combined_p:>12.3g}")
    return credible_map


def print_phase_audit(
    results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    *,
    name: str,
    printer=print,
) -> dict[str, dict[str, np.ndarray]]:
    audit: dict[str, dict[str, np.ndarray]] = {}
    phase_labels = [label for label in results if is_phase_parameter(label)]
    if not phase_labels:
        return audit

    printer(f"\n{name} phase audit")
    printer(
        f"{'Parameter':<20} {'linear std':>12} {'circular std':>14} "
        f"{'linear KS p':>12} {'circular KS p':>14}"
    )
    printer("-" * 76)
    for label in phase_labels:
        linear_levels = calculate_credible_levels(
            results[label]["posteriors"],  # type: ignore[arg-type]
            results[label]["trues"],       # type: ignore[arg-type]
        )
        circular_levels = calculate_credible_levels(
            results[label]["posteriors"],  # type: ignore[arg-type]
            results[label]["trues"],       # type: ignore[arg-type]
            circular=True,
        )
        linear_summary = summarize_credible_levels(linear_levels)
        circular_summary = summarize_credible_levels(circular_levels)
        audit[label] = {
            "linear": linear_levels,
            "circular": circular_levels,
        }
        printer(
            f"{_short_label(label):<20} "
            f"{float(linear_summary['std_cl']):>12.3f} "
            f"{float(circular_summary['std_cl']):>14.3f} "
            f"{float(linear_summary['ks_p']):>12.3g} "
            f"{float(circular_summary['ks_p']):>14.3g}"
        )
    return audit


def write_credible_level_table(
    output_path: Path,
    *,
    freq_results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    wdm_results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    labels: list[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                "parameter",
                "credible_level",
                "circular_credible_level",
            ],
        )
        writer.writeheader()
        for method_name, results in (("WDM", wdm_results), ("Frequency", freq_results)):
            for label in labels:
                linear_levels = calculate_credible_levels(
                    results[label]["posteriors"],  # type: ignore[arg-type]
                    results[label]["trues"],       # type: ignore[arg-type]
                )
                circular_levels = (
                    calculate_credible_levels(
                        results[label]["posteriors"],  # type: ignore[arg-type]
                        results[label]["trues"],       # type: ignore[arg-type]
                        circular=True,
                    )
                    if is_phase_parameter(label)
                    else np.full_like(linear_levels, np.nan)
                )
                seeds = results[label]["seeds"]  # type: ignore[assignment]
                for seed, linear_level, circular_level in zip(
                    seeds,
                    linear_levels,
                    circular_levels,
                    strict=True,
                ):
                    writer.writerow(
                        {
                            "method": method_name,
                            "seed": int(seed),
                            "parameter": label,
                            "credible_level": f"{float(linear_level):.12g}",
                            "circular_credible_level": (
                                "" if np.isnan(circular_level) else f"{float(circular_level):.12g}"
                            ),
                        }
                    )


def build_summary_payload(
    *,
    args: argparse.Namespace,
    used_seeds: list[int],
    skipped: list[SkippedSeed],
    freq_results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    wdm_results: dict[str, dict[str, list[np.ndarray] | list[float] | list[int]]],
    labels: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": args.mode,
        "output_dir": str(args.output_dir if args.output_dir is not None else OUTDIR_ROOT / args.mode),
        "used_seeds": used_seeds,
        "skipped_seeds": [{"seed": item.seed, "missing": item.missing} for item in skipped],
        "methods": {},
    }

    methods = {
        "WDM": wdm_results,
        "Frequency": freq_results,
    }
    for method_name, results in methods.items():
        method_payload: dict[str, object] = {"parameters": {}}
        p_values = []
        for label in labels:
            _is_circular = is_phase_parameter(label)
            primary_levels = calculate_credible_levels(
                results[label]["posteriors"],  # type: ignore[arg-type]
                results[label]["trues"],       # type: ignore[arg-type]
                circular=_is_circular,
            )
            primary_summary = summarize_credible_levels(primary_levels)
            p_values.append(float(primary_summary["ks_p"]))
            entry: dict[str, object] = {
                **primary_summary,
                "credible_levels": primary_levels.tolist(),
                "seeds": [int(seed) for seed in results[label]["seeds"]],  # type: ignore[arg-type]
            }
            if _is_circular:
                linear_levels = calculate_credible_levels(
                    results[label]["posteriors"],  # type: ignore[arg-type]
                    results[label]["trues"],       # type: ignore[arg-type]
                    circular=False,
                )
                entry["linear"] = {
                    **summarize_credible_levels(linear_levels),
                    "credible_levels": linear_levels.tolist(),
                }
            method_payload["parameters"][label] = entry
        method_payload["combined_p"] = float(st.combine_pvalues(p_values)[1])
        payload["methods"][method_name] = method_payload
    return payload


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
        is_circular = is_phase_parameter(label)
        freq_cls = calculate_credible_levels(
            freq_results[label]["posteriors"],  # type: ignore[arg-type]
            freq_results[label]["trues"],       # type: ignore[arg-type]
            circular=is_circular,
        )
        wdm_cls = calculate_credible_levels(
            wdm_results[label]["posteriors"],   # type: ignore[arg-type]
            wdm_results[label]["trues"],        # type: ignore[arg-type]
            circular=is_circular,
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


def plot_pp_per_parameter(
    *,
    freq_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    wdm_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    labels: list[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> Path:
    """One PP subplot per parameter so per-parameter calibration is easy to read."""
    x_values = np.linspace(0.0, 1.0, 201)
    n_sims = len(next(iter(freq_results.values()))["trues"])
    ncols = min(len(labels), 3)
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)

    for ax, label in zip(axes_flat, labels, strict=False):
        for ci in CONFIDENCE_INTERVALS:
            edge = 0.5 * (1.0 - ci)
            lower = st.binom.ppf(edge, n_sims, x_values) / n_sims
            upper = st.binom.ppf(1.0 - edge, n_sims, x_values) / n_sims
            ax.fill_between(x_values, lower, upper, color="0.7", alpha=0.12)

        is_circular = is_phase_parameter(label)
        freq_cls = calculate_credible_levels(
            freq_results[label]["posteriors"],  # type: ignore[arg-type]
            freq_results[label]["trues"],       # type: ignore[arg-type]
            circular=is_circular,
        )
        wdm_cls = calculate_credible_levels(
            wdm_results[label]["posteriors"],  # type: ignore[arg-type]
            wdm_results[label]["trues"],       # type: ignore[arg-type]
            circular=is_circular,
        )
        ax.plot(x_values, empirical_pp(wdm_cls, x_values), color="C0", lw=2.0, label="WDM")
        ax.plot(x_values, empirical_pp(freq_cls, x_values), color="C1", lw=2.0, ls="--", label="Frequency")
        ax.plot([0.0, 1.0], [0.0, 1.0], color="k", ls=":", lw=1.0)
        freq_p = float(st.kstest(freq_cls, "uniform").pvalue)
        wdm_p = float(st.kstest(wdm_cls, "uniform").pvalue)
        ax.set_title(f"{_short_label(label)}\nWDM p={wdm_p:.2g}, Freq p={freq_p:.2g}", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted CL", fontsize=8)
        ax.set_ylabel("Observed fraction", fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(labels):]:
        ax.set_visible(False)

    fig.suptitle(title)
    return save_figure(fig, output_dir, f"{stem}_per_param", dpi=200)


def plot_bias_vs_truth(
    *,
    freq_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    wdm_results: dict[str, dict[str, list[np.ndarray] | list[float]]],
    labels: list[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> Path:
    """Scatter of standardized bias (posterior_mean - truth) / posterior_std vs truth.

    A horizontal scatter near zero means no systematic offset.  A trend (positive
    slope or negative slope) indicates the model over/under-shoots depending on where
    the true value falls — the root cause of an S-curve PP plot.
    """
    ncols = min(len(labels), 3)
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)

    for ax, label in zip(axes_flat, labels, strict=False):
        for method_name, results, color, marker in (
            ("WDM", wdm_results, "C0", "o"),
            ("Frequency", freq_results, "C1", "s"),
        ):
            posteriors: list[np.ndarray] = results[label]["posteriors"]  # type: ignore[assignment]
            trues: list[float] = results[label]["trues"]  # type: ignore[assignment]
            biases, truths = [], []
            for samples, truth in zip(posteriors, trues, strict=True):
                arr = np.asarray(samples, dtype=float)
                std = float(arr.std(ddof=0))
                if std == 0.0:
                    continue
                mean = float(arr.mean())
                biases.append((mean - truth) / std)
                truths.append(truth)
            if truths:
                ax.scatter(truths, biases, s=12, alpha=0.6, color=color, marker=marker, label=method_name)

        ax.axhline(0.0, color="k", ls=":", lw=1.0)
        ax.set_title(_short_label(label), fontsize=9)
        ax.set_xlabel("Truth", fontsize=8)
        ax.set_ylabel("(mean − truth) / std", fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(labels):]:
        ax.set_visible(False)

    fig.suptitle(title)
    return save_figure(fig, output_dir, f"{stem}_bias_vs_truth", dpi=200)


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

    freq_runs, wdm_runs, used_seeds, skipped = collect_runs(mode_dir, seed_dirs)
    labels = common_labels(freq_runs + wdm_runs)
    freq_results = build_results(freq_runs, labels)
    wdm_results = build_results(wdm_runs, labels)

    log_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log_lines.append(line)

    emit(f"Using mode: {args.mode}")
    emit(f"Complete seeds ({len(used_seeds)}): {used_seeds}")
    if skipped:
        emit("Skipped seeds:")
        for item in skipped:
            emit(f"  seed {item.seed}: missing {', '.join(item.missing)}")
    else:
        emit("Skipped seeds: none")

    print_summary(wdm_results, name="WDM", printer=emit)
    print_summary(freq_results, name="Frequency", printer=emit)
    print_phase_audit(wdm_results, name="WDM", printer=emit)
    print_phase_audit(freq_results, name="Frequency", printer=emit)

    _plot_title = f"LISA Posterior PP Plot ({args.mode}, N={len(used_seeds)})"

    out_path = plot_pp_compare(
        freq_results=freq_results,
        wdm_results=wdm_results,
        labels=labels,
        output_dir=output_dir,
        stem=args.stem,
        title=_plot_title,
    )
    emit(f"\nSaved PP plot to {out_path}")

    per_param_path = plot_pp_per_parameter(
        freq_results=freq_results,
        wdm_results=wdm_results,
        labels=labels,
        output_dir=output_dir,
        stem=args.stem,
        title=_plot_title,
    )
    emit(f"Saved per-parameter PP plot to {per_param_path}")

    bias_path = plot_bias_vs_truth(
        freq_results=freq_results,
        wdm_results=wdm_results,
        labels=labels,
        output_dir=output_dir,
        stem=args.stem,
        title=_plot_title,
    )
    emit(f"Saved bias-vs-truth plot to {bias_path}")

    summary_txt = output_dir / f"{args.stem}_summary.txt"
    summary_json = output_dir / f"{args.stem}_summary.json"
    credible_csv = output_dir / f"{args.stem}_credible_levels.csv"
    write_credible_level_table(
        credible_csv,
        freq_results=freq_results,
        wdm_results=wdm_results,
        labels=labels,
    )
    summary_json.write_text(
        json.dumps(
            build_summary_payload(
                args=args,
                used_seeds=used_seeds,
                skipped=skipped,
                freq_results=freq_results,
                wdm_results=wdm_results,
                labels=labels,
            ),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    emit(f"Saved PP sidecars to {summary_txt}, {summary_json}, and {credible_csv}")
    summary_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
