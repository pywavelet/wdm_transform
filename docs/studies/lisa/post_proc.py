"""Post-process one-source LISA posterior files from the frequency and WDM runs."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:
    corner = None

from lisa_common import (
    FREQ_POSTERIOR_PATH,
    RUN_DIR,
    WDM_POSTERIOR_PATH,
    build_parameter_diagnostics,
    circular_credible_level,
    is_phase_parameter,
    print_posterior_report,
    save_figure,
    wrap_phase,
)

DEFAULT_OUTPUT_DIR = RUN_DIR
DEFAULT_WDM_PATH = WDM_POSTERIOR_PATH
DEFAULT_FREQ_PATH = FREQ_POSTERIOR_PATH
_SCRIPT_START = time.perf_counter()


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[post_proc.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)


@dataclass(frozen=True)
class RunPosterior:
    name: str
    path: Path
    samples: np.ndarray
    labels: list[str]
    truth: np.ndarray
    snr: float | None


def _normalize_phi(samples: np.ndarray, labels: list[str]) -> np.ndarray:
    wrapped = np.asarray(samples, dtype=float).copy()
    for idx, label in enumerate(labels):
        if "phi" in label.lower() or "phase" in label.lower():
            wrapped[:, idx] = (wrapped[:, idx] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def plot_single_corner(run: RunPosterior, output_dir: Path) -> None:
    """Plot a single corner plot when only one posterior is available."""
    if corner is None:
        print("corner package not installed; skipping corner plot.")
        return
    if run.samples.ndim != 2 or run.samples.shape[0] <= run.samples.shape[1]:
        print("Skipping corner plot: need more samples than plotted dimensions.")
        return

    fig = corner.corner(
        run.samples,
        labels=[label.replace("source 1 ", "") for label in run.labels],
        truths=run.truth,
        color="tab:blue",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )

    from matplotlib.patches import Patch

    axes = np.asarray(fig.axes).reshape((len(run.labels), len(run.labels)))
    legend_ax = axes[0, -1]
    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=run.name),
            plt.Line2D([0], [0], color="tab:red", ls="--", label="truth"),
        ],
        loc="upper left",
        fontsize=8,
    )
    save_figure(fig, output_dir, "corner_source_1")


def load_run_if_exists(path: Path, name: str) -> RunPosterior | None:
    """Load a run if the file exists, otherwise return None."""
    if not path.exists():
        return None
    return load_run(path, name)


def load_run(path: Path, name: str) -> RunPosterior:
    if not path.exists():
        raise FileNotFoundError(f"Posterior file not found: {path}")

    with np.load(path) as data:
        samples = np.asarray(data["samples_source"], dtype=float)
        labels = [str(item) for item in np.asarray(data["labels"]).tolist()]
        truth = np.asarray(data["truth"], dtype=float).reshape(-1)
        snr_arr = np.asarray(data["snr_optimal"], dtype=float).reshape(-1)

    if samples.ndim != 2:
        raise ValueError(f"{path} has invalid samples_source shape {samples.shape}.")
    if truth.size != len(labels):
        raise ValueError(f"{path} has {truth.size} truth values for {len(labels)} labels.")
    if samples.shape[1] != len(labels):
        raise ValueError(f"{path} has {samples.shape[1]} sample columns for {len(labels)} labels.")

    truth_2d = _normalize_phi(truth[None, :], labels)
    return RunPosterior(
        name=name,
        path=path,
        samples=_normalize_phi(samples, labels),
        labels=labels,
        truth=truth_2d.reshape(-1),
        snr=float(snr_arr[0]) if snr_arr.size else None,
    )


def align_common_labels(run_a: RunPosterior, run_b: RunPosterior) -> tuple[RunPosterior, RunPosterior, list[str]]:
    common = [label for label in run_a.labels if label in run_b.labels]
    if not common:
        raise ValueError("No shared parameter labels across posterior files.")

    idx_a = [run_a.labels.index(label) for label in common]
    idx_b = [run_b.labels.index(label) for label in common]

    aligned_a = RunPosterior(
        name=run_a.name,
        path=run_a.path,
        samples=run_a.samples[:, idx_a],
        labels=common,
        truth=run_a.truth[idx_a],
        snr=run_a.snr,
    )
    aligned_b = RunPosterior(
        name=run_b.name,
        path=run_b.path,
        samples=run_b.samples[:, idx_b],
        labels=common,
        truth=run_b.truth[idx_b],
        snr=run_b.snr,
    )
    return aligned_a, aligned_b, common


def report_run(run: RunPosterior) -> None:
    print(f"\n{run.name} ({run.path.name})")
    print_posterior_report(run.name, run.samples, run.truth, run.labels)


def compare_summary(run_a: RunPosterior, run_b: RunPosterior, labels: list[str]) -> None:
    print("\nRun-to-run median deltas (A - B)")
    med_a = np.median(run_a.samples, axis=0)
    med_b = np.median(run_b.samples, axis=0)
    for idx, label in enumerate(labels):
        delta = float(med_a[idx] - med_b[idx])
        if is_phase_parameter(label):
            delta = float(wrap_phase(delta))
        print(f"  {label:24s} delta={delta:+.6e}")


def build_run_diagnostics(run: RunPosterior) -> dict[str, object]:
    return {
        "name": run.name,
        "path": str(run.path),
        "snr": run.snr,
        "parameters": build_parameter_diagnostics(run.samples, run.truth, run.labels),
    }


def build_comparison_diagnostics(
    run_a: RunPosterior,
    run_b: RunPosterior,
    labels: list[str],
) -> dict[str, object]:
    med_a = np.median(run_a.samples, axis=0)
    med_b = np.median(run_b.samples, axis=0)
    deltas = []
    for idx, label in enumerate(labels):
        delta = float(med_a[idx] - med_b[idx])
        if is_phase_parameter(label):
            delta = float(wrap_phase(delta))
        deltas.append(
            {
                "label": label,
                "median_delta": delta,
                "credible_level_delta": (
                    float(circular_credible_level(run_a.samples[:, idx], run_a.truth[idx]))
                    - float(circular_credible_level(run_b.samples[:, idx], run_b.truth[idx]))
                    if is_phase_parameter(label)
                    else None
                ),
            }
        )
    return {"median_deltas": deltas}


def plot_corner(run_a: RunPosterior, run_b: RunPosterior, output_dir: Path) -> None:
    if corner is None:
        print("corner package not installed; skipping corner plot.")
        return
    if (
        run_a.samples.ndim != 2
        or run_b.samples.ndim != 2
        or run_a.samples.shape[0] <= run_a.samples.shape[1]
        or run_b.samples.shape[0] <= run_b.samples.shape[1]
    ):
        print("Skipping corner plot: need more samples than plotted dimensions.")
        return

    fig = corner.corner(
        run_a.samples,
        labels=[label.replace("source 1 ", "") for label in run_a.labels],
        truths=run_a.truth,
        color="tab:blue",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )
    corner.corner(
        run_b.samples,
        fig=fig,
        labels=[label.replace("source 1 ", "") for label in run_b.labels],
        truths=None,
        color="tab:orange",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )

    from matplotlib.patches import Patch

    axes = np.asarray(fig.axes).reshape((len(run_a.labels), len(run_a.labels)))
    legend_ax = axes[0, -1]
    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=run_a.name),
            Patch(facecolor="tab:orange", alpha=0.5, label=run_b.name),
            plt.Line2D([0], [0], color="tab:red", ls="--", label="truth"),
        ],
        loc="upper left",
        fontsize=8,
    )
    save_figure(fig, output_dir, "corner_source_1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, default=DEFAULT_WDM_PATH, help="Path to first posterior NPZ (default: WDM posterior).")
    parser.add_argument("--run-b", type=Path, default=DEFAULT_FREQ_PATH, help="Path to second posterior NPZ (default: frequency posterior).")
    parser.add_argument("--name-a", type=str, default="WDM")
    parser.add_argument("--name-b", type=str, default="Frequency")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_path = args.output_dir / "posterior_diagnostics.txt"
    json_path = args.output_dir / "posterior_diagnostics.json"

    run_a = load_run_if_exists(args.run_a, args.name_a)
    run_b = load_run_if_exists(args.run_b, args.name_b)

    payload: dict[str, object] = {
        "output_dir": str(args.output_dir),
        "run_a_path": str(args.run_a),
        "run_b_path": str(args.run_b),
    }
    report_buffer = io.StringIO()

    with contextlib.redirect_stdout(report_buffer):
        if run_a is None and run_b is None:
            payload["status"] = "missing"
            print(f"Error: No posterior files found.")
            print(f"  Expected WDM at: {args.run_a}")
            print(f"  Expected Freq at: {args.run_b}")
        elif run_a is None or run_b is None:
            run = run_a if run_a is not None else run_b
            assert run is not None
            payload["status"] = "single_run"
            payload["runs"] = {run.name: build_run_diagnostics(run)}
            report_run(run)
            plot_single_corner(run, args.output_dir)
            print(f"\nSaved corner plot to {args.output_dir / 'corner_source_1.png'}")
        else:
            run_a, run_b, labels = align_common_labels(run_a, run_b)
            payload["status"] = "paired_runs"
            payload["runs"] = {
                run_a.name: build_run_diagnostics(run_a),
                run_b.name: build_run_diagnostics(run_b),
            }
            payload["comparison"] = build_comparison_diagnostics(run_a, run_b, labels)
            report_run(run_a)
            report_run(run_b)
            compare_summary(run_a, run_b, labels)
            plot_corner(run_a, run_b, args.output_dir)

            print("\nSaved comparison figures:")
            print(f"  {args.output_dir / 'posterior_marginals_compare.png'}")
            print(f"  {args.output_dir / 'posterior_interval_compare.png'}")
            if (args.output_dir / "snr_compare.png").exists():
                print(f"  {args.output_dir / 'snr_compare.png'}")
            if (args.output_dir / "corner_source_1.png").exists():
                print(f"  {args.output_dir / 'corner_source_1.png'}")

    report_text = report_buffer.getvalue().rstrip()
    if report_text:
        report_text = (
            f"{report_text}\n\nSaved diagnostics sidecars:\n"
            f"  {text_path}\n"
            f"  {json_path}\n"
        )
    else:
        report_text = (
            "Saved diagnostics sidecars:\n"
            f"  {text_path}\n"
            f"  {json_path}\n"
        )

    print(report_text)
    text_path.write_text(report_text + ("\n" if not report_text.endswith("\n") else ""), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
