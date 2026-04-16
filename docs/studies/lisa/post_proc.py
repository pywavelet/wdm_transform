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
from matplotlib.ticker import EngFormatter, FuncFormatter, ScalarFormatter

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


def _setup_matplotlib_style() -> None:
    """Configure matplotlib for publication-style corner plots."""
    plt.rcParams.update({
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def _format_axis_label(label: str) -> str:
    """Convert parameter label to formatted version with math symbols."""
    # Replace common parameter names with clean symbols (no units - corner already adds them)
    label_map = {
        "f0": r"$f_0$",
        "fdot": r"$\dot{f}$",
        "A": r"$A$",
        "phi0": r"$\phi_0$",
        "psi": r"$\psi$",
        "iota": r"$\iota$",
        "SNR": r"SNR",
    }
    # Try exact match first
    if label in label_map:
        return label_map[label]
    # Try partial match for complex names
    for key, value in label_map.items():
        if key in label.lower():
            return label.replace(key, value)
    return label


def _inject_unit_prefix(ax, label: str, truth_value: float | None = None) -> tuple[str, float, bool]:
    """
    Inject unit prefix (mHz, µHz, etc.) into axis label based on data magnitude.
    Returns modified label, scaling factor for tick conversion, and whether delta formatting was applied.
    """
    # Extract the base unit from the label (e.g., "Hz" from "f0 [Hz]")
    import re
    match = re.search(r'\[([^\]]+)\]', label)
    if not match:
        return label, 1.0, False

    base_unit = match.group(1)

    # Skip unit prefixing for certain parameter types that should stay in their base units
    if any(param in label.lower() for param in ['phi', 'psi', 'iota', 'phase']):
        # Phase parameters should stay in radians
        return label, 1.0, False

    if base_unit.lower() == 'rad':
        # All radian measurements should stay in radians
        return label, 1.0, False

    if 'snr' in label.lower():
        # SNR is dimensionless, don't scale
        return label, 1.0, False

    # Get axis limits to determine scale
    try:
        lim = ax.get_xlim() if hasattr(ax, 'get_xlim') else ax.get_ylim()
        if not lim or lim[0] == lim[1]:
            return label, 1.0, False
    except (AttributeError, ValueError):
        return label, 1.0, False

    data_range = np.abs(lim[1] - lim[0])
    data_mag = np.abs(np.mean(lim))

    if data_mag == 0:
        return label, 1.0, False

    # Special handling for f0: if range is very small compared to magnitude, use delta formatting
    if 'f0' in label.lower() and data_range / data_mag < 1e-4 and truth_value is not None:
        # Use delta formatting: show f0_sample - f0_truth
        param_name = label.split()[0]  # Extract "f0" from "f0 [Hz]"
        new_label = label.replace(param_name, f'Δ{param_name}')
        return new_label, 1.0, True

    mag = np.log10(data_mag) if data_mag > 0 else 0

    # Define prefixes: (exponent, symbol, name)
    prefixes = [
        (12, "T", "tera"),
        (9, "G", "giga"),
        (6, "M", "mega"),
        (3, "k", "kilo"),
        (0, "", ""),
        (-3, "m", "milli"),
        (-6, "µ", "micro"),
        (-9, "n", "nano"),
        (-12, "p", "pico"),
        (-15, "f", "femto"),
    ]

    # Find best prefix (closest to data magnitude)
    best_prefix = ""
    best_exp = 0
    for exp, symbol, _ in prefixes:
        if abs(mag - exp) < abs(mag - best_exp):
            best_exp = exp
            best_prefix = symbol

    if best_exp == 0:
        return label, 1.0, False

    # Replace unit with prefixed unit
    scale_factor = 10.0 ** best_exp
    new_unit = f"{best_prefix}{base_unit}"
    new_label = label.replace(f"[{base_unit}]", f"[{new_unit}]")

    return new_label, scale_factor, False


def _apply_offset_tick_scaling(ax, reference: float, scale_factor: float, axis: str = "both") -> None:
    """Apply offset formatting to tick values (showing difference from reference).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    reference : float
        The reference value to subtract from tick values
    scale_factor : float
        The scale factor to apply after offset
    axis : str
        Which axis to format: "x", "y", or "both"
    """
    def offset_formatter(val, pos):
        # Calculate offset from reference
        offset = val - reference
        # Apply scaling
        scaled_offset = offset * scale_factor

        if abs(scaled_offset) < 1e-15:
            return "0"
        elif abs(scaled_offset) >= 1e3 or abs(scaled_offset) <= 1e-6:
            return f"{scaled_offset:.2e}"
        else:
            return f"{scaled_offset:.6g}"

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(offset_formatter))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(offset_formatter))


def _apply_tick_scaling(ax, scale_factor: float, axis: str = "both") -> None:
    """Apply scaling to tick values via FuncFormatter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    scale_factor : float
        The scale factor to apply
    axis : str
        Which axis to format: "x", "y", or "both"
    """
    def smart_formatter(x, pos):
        # More aggressive about showing small values
        if abs(x) < 1e-15:
            return "0"
        elif abs(x) >= 1e4 or abs(x) <= 1e-6:
            return f"{x:.2e}"
        else:
            return f"{x:.5g}"

    if scale_factor == 1.0:
        # No scaling needed; use smart formatter
        if axis in ("x", "both"):
            ax.xaxis.set_major_formatter(FuncFormatter(smart_formatter))
        if axis in ("y", "both"):
            ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))
        return

    # Create formatters that scale the tick values
    def x_formatter(x, pos):
        val = x * scale_factor
        if abs(val) < 1e-15:
            return "0"
        elif abs(val) >= 1e4 or abs(val) <= 1e-6:
            return f"{val:.2e}"
        else:
            return f"{val:.5g}"

    def y_formatter(y, pos):
        val = y * scale_factor
        if abs(val) < 1e-15:
            return "0"
        elif abs(val) >= 1e4 or abs(val) <= 1e-6:
            return f"{val:.2e}"
        else:
            return f"{val:.5g}"

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))


def _configure_axes_formatting(fig, ndim: int) -> None:
    """Apply consistent formatting and positioning to all axes."""
    for i, ax in enumerate(fig.axes):
        row = i // ndim
        col = i % ndim

        # Try to disable offset text display (only works with ScalarFormatter)
        try:
            ax.ticklabel_format(style="plain", axis="both")
        except (AttributeError, ValueError):
            # Some axes might not support this (e.g., with NullFormatter)
            pass

        # Handle y-axis labels and formatting
        if col == 0:  # Leftmost column - show y-labels
            y_label = ax.get_ylabel()
            if y_label:
                new_label, scale = _inject_unit_prefix(ax, y_label)
                # Check if offset formatting was applied
                if " - " in new_label:
                    # Extract reference value and apply offset formatting
                    import re
                    match = re.search(r' - ([0-9.e-]+)', new_label)
                    if match:
                        reference = float(match.group(1))
                        ax.set_ylabel(new_label)
                        _apply_offset_tick_scaling(ax, reference, scale, axis="y")
                    else:
                        ax.set_ylabel(new_label)
                        _apply_tick_scaling(ax, scale, axis="y")
                else:
                    ax.set_ylabel(new_label)
                    _apply_tick_scaling(ax, scale, axis="y")
        else:  # Not leftmost column - hide y-labels
            ax.set_yticklabels([])

        # Handle x-axis labels and formatting
        if row == ndim - 1:  # Bottom row - show x-labels
            x_label = ax.get_xlabel()
            if x_label:
                new_label, scale = _inject_unit_prefix(ax, x_label)
                # Check if offset formatting was applied
                if " - " in new_label:
                    # Extract reference value and apply offset formatting
                    import re
                    match = re.search(r' - ([0-9.e-]+)', new_label)
                    if match:
                        reference = float(match.group(1))
                        ax.set_xlabel(new_label)
                        _apply_offset_tick_scaling(ax, reference, scale, axis="x")
                    else:
                        ax.set_xlabel(new_label)
                        _apply_tick_scaling(ax, scale, axis="x")
                else:
                    ax.set_xlabel(new_label)
                    _apply_tick_scaling(ax, scale, axis="x")
        else:  # Not bottom row - hide x-labels
            ax.set_xticklabels([])

        ax.tick_params(axis="both", labelsize=8, pad=3)


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

    _setup_matplotlib_style()

    clean_labels = [label.replace("source 1 ", "") for label in run.labels]
    latex_labels = [_format_axis_label(lbl) for lbl in clean_labels]

    fig = corner.corner(
        run.samples,
        labels=latex_labels,
        truths=run.truth,
        truth_color="black",
        color="tab:blue",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )

    from matplotlib.patches import Patch

    axes = np.asarray(fig.axes).reshape((len(run.labels), len(run.labels)))
    legend_ax = axes[0, -1]

    # Configure all axis formatting consistently
    _configure_axes_formatting(fig, len(run.labels))

    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=run.name),
            plt.Line2D([0], [0], color="black", ls="-", lw=1.5, label="Truth"),
        ],
        loc="upper left",
        fontsize=14,
        frameon=False,
        fancybox=False,
        # edgecolor="F",
        framealpha=0.0,
        # make big
        handlelength=1.5,
        handleheight=1.5,
        # increase spacing
        labelspacing=0.3,
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

    _setup_matplotlib_style()

    clean_labels = [label.replace("source 1 ", "") for label in run_a.labels]
    latex_labels = [_format_axis_label(lbl) for lbl in clean_labels]

    fig = corner.corner(
        run_a.samples,
        labels=latex_labels,
        truths=run_a.truth,
        truth_color="black",
        color="tab:blue",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )
    corner.corner(
        run_b.samples,
        fig=fig,
        labels=latex_labels,
        truths=None,
        color="tab:orange",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )

    from matplotlib.patches import Patch

    axes = np.asarray(fig.axes).reshape((len(run_a.labels), len(run_a.labels)))
    legend_ax = axes[0, -1]

    # Configure all axis formatting consistently
    _configure_axes_formatting(fig, len(run_a.labels))

    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=run_a.name),
            Patch(facecolor="tab:orange", alpha=0.5, label=run_b.name),
            plt.Line2D([0], [0], color="black", ls="-", lw=1.5, label="Truth"),
        ],
        loc="upper left",
        fontsize=14,
        frameon=False,
        fancybox=False,
        # edgecolor="black",
        framealpha=0.0,
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
