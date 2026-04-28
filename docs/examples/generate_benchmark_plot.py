"""Generate benchmark artifacts used by the documentation.

This script is intentionally not executed as part of the docs build because
the benchmark can be slow, especially when optional backends are installed.
Run it manually when you want to refresh the checked-in benchmark plot and
JSON summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdm_transform.benchmarking import (  # noqa: E402
    DEFAULT_BACKENDS,
    plot_results,
    print_summary,
    resolve_n_values,
    run_benchmarks,
    save_results,
)

DEFAULT_N_VALUES = resolve_n_values()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark artifacts for the documentation.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["numpy", "jax", "cupy"],
        help="Backends to benchmark (default: numpy jax)",
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=DEFAULT_N_VALUES,
        help="Input sizes to test (default: 2048 through 33554432)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Number of timed runs per benchmark (default: 7)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "benchmark_results.json",
        help="Path for the JSON benchmark summary.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=ROOT / "docs" / "_static" / "benchmark_runtime.png",
        help="Path for the benchmark plot image.",
    )
    parser.add_argument(
        "--plot-title",
        default="WDM Transform Runtime by Backend",
        help="Title to use for the generated benchmark plot.",
    )

    args = parser.parse_args()

    results = run_benchmarks(
        backends_to_test=args.backends,
        n_values=args.n,
        num_runs=args.runs,
    )
    print_summary(results)
    json_path = save_results(results, args.output_json)
    plot_path = plot_results(results, args.output_plot, title=args.plot_title)
    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
