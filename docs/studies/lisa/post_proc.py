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
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

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
    compact_label = re.sub(r"[\s$\\{}]", "", label)
    if compact_label == "A":
        return r"$A\ [10^{-23}]$", 1.0e23, False

    # Extract the base unit from the label (e.g., "Hz" from "f0 [Hz]")
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

    if "fdot" in label.lower() or r"\dot{f}" in label:
        # Put the scientific scale in the axis label instead of using Matplotlib's
        # offset text, which tends to collide with the left-column spines.
        return r"$\dot{f}\ [10^{-18}\ \mathrm{Hz/s}]$", 1.0e18, False

    # Special handling for f0: always use delta formatting when truth value is available
    if ('f0' in label.lower() or 'f_0' in label.lower()) and truth_value is not None:
        try:
            lim = ax.get_xlim() if hasattr(ax, 'get_xlim') else ax.get_ylim()
            delta_scale = max(abs(lim[0] - truth_value), abs(lim[1] - truth_value))
        except (AttributeError, ValueError, TypeError):
            delta_scale = 0.0

        if delta_scale > 0.0:
            delta_exp = int(np.floor(np.log10(delta_scale)))
            # Snap to engineering-style powers for cleaner labels.
            delta_exp = int(3 * np.floor(delta_exp / 3))
            scale_factor = 10.0 ** (-delta_exp)
            unit_label = rf"10^{{{delta_exp}}}"
        else:
            scale_factor = 1.0
            unit_label = "Hz"

        return rf"$\Delta f_0\ [{unit_label}]$ Hz", scale_factor, True

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


def _apply_delta_tick_scaling(ax, truth_value: float, scale_factor: float, axis: str = "both") -> None:
    """Apply delta formatting to tick values (showing sample - truth).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    truth_value : float
        The truth value to subtract from tick values
    scale_factor : float
        The scale factor to apply after delta calculation
    axis : str
        Which axis to format: "x", "y", or "both"
    """
    def delta_formatter(val, pos):
        # Calculate delta from truth
        delta = val - truth_value
        # Apply scaling
        scaled_delta = delta * scale_factor

        if abs(scaled_delta) < 1e-25:  # Even smaller threshold
            return "0"
        elif abs(scaled_delta) <= 1e-1:  # Much more aggressive scientific notation
            return f"{scaled_delta:.4e}"  # Even higher precision
        else:
            return f"{scaled_delta:.6g}"

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(delta_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(delta_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _make_scalar_formatter() -> ScalarFormatter:
    """Create a formatter that shows one shared exponent instead of repeating it."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(True)
    return formatter


def _apply_scalar_tick_format(ax, axis: str = "both") -> None:
    """Use Matplotlib's scientific formatter with shared exponent/offset text."""
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(_make_scalar_formatter())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(_make_scalar_formatter())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_offset_position("left")


def _reposition_offset_text(ax, row: int, col: int, ndim: int) -> None:
    """Move scientific offset text away from spines so it does not overlap ticks."""
    x_offset = ax.xaxis.get_offset_text()
    y_offset = ax.yaxis.get_offset_text()

    x_offset.set_size(8)
    y_offset.set_size(8)

    if row == ndim - 1:
        x_offset.set_horizontalalignment("right")
        x_offset.set_verticalalignment("top")
        x_offset.set_x(1.0)
        x_offset.set_y(-0.08)

    if col == 0:
        y_offset.set_horizontalalignment("left")
        y_offset.set_verticalalignment("bottom")
        y_offset.set_x(0.0)
        y_offset.set_y(1.02)


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
    if scale_factor == 1.0:
        _apply_scalar_tick_format(ax, axis=axis)
        return

    # Create formatters that scale the tick values
    def x_formatter(x, pos):
        val = x * scale_factor
        if abs(val) < 1e-30:
            return "0"
        elif abs(val) <= 1e-2:
            return f"{val:.3e}"
        else:
            return f"{val:.5g}"

    def y_formatter(y, pos):
        val = y * scale_factor
        if abs(val) < 1e-30:
            return "0"
        elif abs(val) <= 1e-2:
            return f"{val:.3e}"
        else:
            return f"{val:.5g}"

    if axis in ("x", "both"):
        ax.xaxis.offsetText.set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _configure_axes_formatting(fig, ndim: int, truth_values: np.ndarray | None = None) -> dict[str, bool]:
    """Apply consistent formatting and positioning to all axes.

    Returns
    -------
    dict[str, bool]
        Dictionary indicating which parameters used delta formatting
    """
    delta_used = {}

    for i, ax in enumerate(fig.axes):
        row = i // ndim
        col = i % ndim

        # Try to disable offset text display (only works with ScalarFormatter)
        try:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2), useMathText=True)
        except (AttributeError, ValueError):
            # Some axes might not support this (e.g., with NullFormatter)
            pass

        # Handle y-axis labels and formatting
        if col == 0:  # Leftmost column - show y-labels
            y_label = ax.get_ylabel()
            if y_label:
                truth_val = truth_values[row] if truth_values is not None else None
                new_label, scale, is_delta = _inject_unit_prefix(ax, y_label, truth_val)
                ax.set_ylabel(new_label)

                if is_delta and truth_val is not None:
                    # Extract parameter name from LaTeX or regular format
                    if '$f_0$' in y_label:
                        param_name = 'f0'
                    else:
                        param_name = y_label.split()[0].strip('$')
                    delta_used[param_name] = True
                    _apply_delta_tick_scaling(ax, truth_val, scale, axis="y")
                else:
                    _apply_tick_scaling(ax, scale, axis="y")
        else:  # Not leftmost column - hide y-labels
            ax.set_yticklabels([])

        # Handle x-axis labels and formatting
        if row == ndim - 1:  # Bottom row - show x-labels
            x_label = ax.get_xlabel()
            if x_label:
                truth_val = truth_values[col] if truth_values is not None else None
                new_label, scale, is_delta = _inject_unit_prefix(ax, x_label, truth_val)
                ax.set_xlabel(new_label)

                if is_delta and truth_val is not None:
                    # Extract parameter name from LaTeX or regular format
                    if '$f_0$' in x_label:
                        param_name = 'f0'
                    else:
                        param_name = x_label.split()[0].strip('$')
                    delta_used[param_name] = True
                    _apply_delta_tick_scaling(ax, truth_val, scale, axis="x")
                else:
                    _apply_tick_scaling(ax, scale, axis="x")
        else:  # Not bottom row - hide x-labels
            ax.set_xticklabels([])

        ax.tick_params(axis="both", labelsize=8, pad=3)
        _reposition_offset_text(ax, row, col, ndim)

    return delta_used


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


def _corner_run(run: RunPosterior) -> RunPosterior:
    """Return a plotting-only view of the run without SNR parameters."""
    keep = [idx for idx, label in enumerate(run.labels) if "snr" not in label.lower()]
    return RunPosterior(
        name=run.name,
        path=run.path,
        samples=run.samples[:, keep],
        labels=[run.labels[idx] for idx in keep],
        truth=run.truth[keep],
        snr=run.snr,
    )


def _legend_run_label(run: RunPosterior) -> str:
    if run.snr is None:
        return run.name
    return f"{run.name} ($\\rho={run.snr:.2f}$)"


def plot_single_corner(run: RunPosterior, output_dir: Path) -> None:
    """Plot a single corner plot when only one posterior is available."""
    run = _corner_run(run)
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

    # Configure all axis formatting consistently and get delta info
    delta_used = _configure_axes_formatting(fig, len(run.labels), run.truth)

    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=_legend_run_label(run)),
            plt.Line2D([0], [0], color="black", ls="-", lw=1.5, label="Truth"),
        ],
        loc="upper left",
        fontsize=14,
        frameon=False,
        fancybox=False,
        framealpha=0.0,
        handlelength=1.5,
        handleheight=1.5,
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
    run_a = _corner_run(run_a)
    run_b = _corner_run(run_b)
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

    # Configure all axis formatting consistently and get delta info
    delta_used = _configure_axes_formatting(fig, len(run_a.labels), run_a.truth)

    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=_legend_run_label(run_a)),
            Patch(facecolor="tab:orange", alpha=0.5, label=_legend_run_label(run_b)),
            plt.Line2D([0], [0], color="black", ls="-", lw=1.5, label="Truth"),
        ],
        loc="upper left",
        fontsize=14,
        frameon=False,
        fancybox=False,
        framealpha=0.0,
    )

    fig.savefig(output_dir / "corner.pdf", dpi=300, bbox_inches="tight")


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
