"""Benchmarking CLI for WDM transforms across different backends and input sizes.

This module is excluded from pytest coverage and should not be imported
when loading the main wdm_transform package.

Run with: python -m wdm_transform.benchmarking [options]
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from . import backends, transforms

DEFAULT_BACKENDS = ["numpy", "jax"]
DEFAULT_BATCH_SIZE = 3
DEFAULT_N_VALUES = [
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
]
FIXED_PARAMS = {"a": 0.4, "d": 0.8, "dt": 1.0}


def validate_backend_available(backend_name: str) -> bool:  # pragma: no cover
    """Check if a backend is available."""
    try:
        backends.get_backend(backend_name)
        return True
    except (ImportError, ValueError):
        return False


def generate_benchmark_signal(
    n: int,
    *,
    batch_size: int | None = None,
) -> np.ndarray:  # pragma: no cover
    """Generate a deterministic real benchmark signal."""
    rng = np.random.default_rng(42)
    if batch_size is None:
        return rng.standard_normal((1, n))
    return rng.standard_normal((batch_size, n))


def generate_benchmark_coeffs(nt: int, nf: int) -> np.ndarray:  # pragma: no cover
    """Generate deterministic real WDM coefficient grids."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, nt, nf + 1))


def find_factorization(n: int) -> tuple[int, int] | None:  # pragma: no cover
    """Find a factorization n = nt * nf where both nt and nf are even.

    Prioritizes balanced factorizations (nt close to sqrt(n)).
    Returns (nt, nf) or None if no valid factorization exists.
    """
    if n < 4:
        return None

    sqrt_n = int(np.sqrt(n))
    for nt in range(sqrt_n, 1, -1):
        if n % nt != 0:
            continue
        nf = n // nt
        if nt % 2 == 0 and nf % 2 == 0:
            return (nt, nf)

    return None


def _synchronize_result(value: Any) -> None:
    """Block on asynchronous backend work before reading the timer."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            _synchronize_result(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _synchronize_result(item)


def _measure_runtime(
    fn: Callable[[], Any],
    num_runs: int,
) -> tuple[float, float]:
    """Warm up once, then return mean and std runtime in seconds."""
    warmup_result = fn()
    _synchronize_result(warmup_result)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn()
        _synchronize_result(result)
        times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times))


def benchmark_forward(
    signal: np.ndarray,
    backend_name: str,
    nt: int,
    nf: int,
    num_runs: int = 7,
) -> tuple[float, float]:  # pragma: no cover
    """Benchmark the forward WDM transform.

    Returns ``(mean_seconds, std_seconds)``.
    """
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}
    backend_data = backend.asarray(signal)

    return _measure_runtime(
        lambda: transforms.from_time_to_wdm(backend_data, nt=nt, nf=nf, **fixed_params),
        num_runs,
    )


def benchmark_inverse(
    coeffs: np.ndarray,
    backend_name: str,
    num_runs: int = 7,
) -> tuple[float, float]:  # pragma: no cover
    """Benchmark the inverse WDM transform.

    Returns ``(mean_seconds, std_seconds)``.
    """
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}
    backend_coeffs = backend.asarray(coeffs)

    return _measure_runtime(
        lambda: transforms.from_wdm_to_time(backend_coeffs, **fixed_params),
        num_runs,
    )


def benchmark_forward_serial_batch(
    signal_batch: np.ndarray,
    backend_name: str,
    nt: int,
    nf: int,
    num_runs: int = 7,
) -> tuple[float, float]:  # pragma: no cover
    """Benchmark serial forward transforms over a leading batch axis."""
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}
    backend_batch = backend.asarray(signal_batch)

    return _measure_runtime(
        lambda: [
            transforms.from_time_to_wdm(backend_batch[idx], nt=nt, nf=nf, **fixed_params)
            for idx in range(int(backend_batch.shape[0]))
        ],
        num_runs,
    )


def benchmark_forward_batched(
    signal_batch: np.ndarray,
    backend_name: str,
    nt: int,
    nf: int,
    num_runs: int = 7,
) -> tuple[float, float]:  # pragma: no cover
    """Benchmark a single batched forward transform call."""
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}
    backend_batch = backend.asarray(signal_batch)

    return _measure_runtime(
        lambda: transforms.from_time_to_wdm(backend_batch, nt=nt, nf=nf, **fixed_params),
        num_runs,
    )


def benchmark_roundtrip_error(
    signal: np.ndarray,
    backend_name: str,
    nt: int,
    nf: int,
) -> dict[str, float | int]:  # pragma: no cover
    """Compute reconstruction error after a forward and inverse transform."""
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}
    backend_signal = backend.asarray(signal)

    coeffs = transforms.from_time_to_wdm(backend_signal, nt=nt, nf=nf, **fixed_params)
    recovered = transforms.from_wdm_to_time(coeffs, **fixed_params)
    _synchronize_result(recovered)

    recovered_np = np.asarray(recovered)
    signal_np = np.asarray(signal)
    diff = recovered_np - signal_np
    signal_norm = float(np.linalg.norm(signal_np))
    relative_l2_error = (
        float(np.linalg.norm(diff) / signal_norm) if signal_norm else 0.0
    )

    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_l2_error": relative_l2_error,
        "nt": nt,
        "nf": nf,
    }


def _result_record(
    mean_seconds: float,
    std_seconds: float,
    nt: int,
    nf: int,
) -> dict[str, float | int]:
    return {
        "mean_seconds": mean_seconds,
        "std_seconds": std_seconds,
        "nt": nt,
        "nf": nf,
    }


def _batch_forward_result_record(
    *,
    single_mean_seconds: float,
    single_std_seconds: float,
    serial_mean_seconds: float,
    serial_std_seconds: float,
    batched_mean_seconds: float,
    batched_std_seconds: float,
    batch_size: int,
    nt: int,
    nf: int,
) -> dict[str, float | int]:
    return {
        "single_mean_seconds": single_mean_seconds,
        "single_std_seconds": single_std_seconds,
        "serial_mean_seconds": serial_mean_seconds,
        "serial_std_seconds": serial_std_seconds,
        "batched_mean_seconds": batched_mean_seconds,
        "batched_std_seconds": batched_std_seconds,
        "batched_vs_serial_speedup": (
            serial_mean_seconds / batched_mean_seconds if batched_mean_seconds else float("inf")
        ),
        "batch_size": batch_size,
        "nt": nt,
        "nf": nf,
    }


def run_benchmarks(  # pragma: no cover
    backends_to_test: list[str],
    n_values: list[int],
    num_runs: int = 7,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run benchmark suite across backends and input sizes."""
    available_backends = [b for b in backends_to_test if validate_backend_available(b)]
    results: dict[str, Any] = {
        "metadata": {
            "requested_backends": backends_to_test,
            "available_backends": available_backends,
            "n_values": n_values,
            "num_runs": num_runs,
            "batch_size": batch_size,
            "parameters": FIXED_PARAMS,
        },
        "forward": {},
        "inverse": {},
        "batch_forward": {},
        "error": {},
    }

    if not available_backends:
        print(f"ERROR: No backends available from: {backends_to_test}")
        return results

    if len(available_backends) < len(backends_to_test):
        unavailable = sorted(set(backends_to_test) - set(available_backends))
        print(f"WARNING: Skipping unavailable backends: {', '.join(unavailable)}")

    print("\n" + "=" * 70)
    print("FORWARD WDM TRANSFORM BENCHMARKS")
    print("=" * 70)

    for backend_name in available_backends:
        results["forward"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)
        for n in n_values:
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue

            nt, nf = factorization
            signal = generate_benchmark_signal(nt * nf)
            try:
                mean_seconds, std_seconds = benchmark_forward(
                    signal,
                    backend_name,
                    nt,
                    nf,
                    num_runs,
                )
                results["forward"][backend_name][n] = _result_record(
                    mean_seconds,
                    std_seconds,
                    nt,
                    nf,
                )
                print(
                    f"  N={n:>8}: {mean_seconds*1e3:>10.4f} ms "
                    f"+/- {std_seconds*1e3:>8.4f} ms (nt={nt}, nf={nf})"
                )
            except Exception as exc:
                print(f"  N={n:>8}: FAILED ({type(exc).__name__}: {exc})")

    print("\n" + "=" * 70)
    print("INVERSE WDM TRANSFORM BENCHMARKS")
    print("=" * 70)

    for backend_name in available_backends:
        results["inverse"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)
        for n in n_values:
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue

            nt, nf = factorization
            coeffs = generate_benchmark_coeffs(nt, nf)
            try:
                mean_seconds, std_seconds = benchmark_inverse(
                    coeffs,
                    backend_name,
                    num_runs,
                )
                results["inverse"][backend_name][n] = _result_record(
                    mean_seconds,
                    std_seconds,
                    nt,
                    nf,
                )
                print(
                    f"  N={n:>8}: {mean_seconds*1e3:>10.4f} ms "
                    f"+/- {std_seconds*1e3:>8.4f} ms (nt={nt}, nf={nf})"
                )
            except Exception as exc:
                print(f"  N={n:>8}: FAILED ({type(exc).__name__}: {exc})")

    print("\n" + "=" * 70)
    print(f"{batch_size}-CHANNEL JAX FORWARD BATCH COMPARISON")
    print("=" * 70)

    batch_backend_names = [backend_name for backend_name in available_backends if backend_name == "jax"]
    for backend_name in batch_backend_names:
        results["batch_forward"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)
        for n in n_values:
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue

            nt, nf = factorization
            signal_batch = generate_benchmark_signal(nt * nf, batch_size=batch_size)
            single_record = results["forward"].get(backend_name, {}).get(n)
            try:
                serial_mean_seconds, serial_std_seconds = benchmark_forward_serial_batch(
                    signal_batch,
                    backend_name,
                    nt,
                    nf,
                    num_runs,
                )
                batched_mean_seconds, batched_std_seconds = benchmark_forward_batched(
                    signal_batch,
                    backend_name,
                    nt,
                    nf,
                    num_runs,
                )
                if single_record is None:
                    single_mean_seconds, single_std_seconds = benchmark_forward(
                        signal_batch[0],
                        backend_name,
                        nt,
                        nf,
                        num_runs,
                    )
                else:
                    single_mean_seconds = float(single_record["mean_seconds"])
                    single_std_seconds = float(single_record["std_seconds"])

                results["batch_forward"][backend_name][n] = _batch_forward_result_record(
                    single_mean_seconds=single_mean_seconds,
                    single_std_seconds=single_std_seconds,
                    serial_mean_seconds=serial_mean_seconds,
                    serial_std_seconds=serial_std_seconds,
                    batched_mean_seconds=batched_mean_seconds,
                    batched_std_seconds=batched_std_seconds,
                    batch_size=batch_size,
                    nt=nt,
                    nf=nf,
                )
                print(
                    f"  N={n:>8}: single={single_mean_seconds*1e3:>10.4f} ms  "
                    f"serial={serial_mean_seconds*1e3:>10.4f} ms  "
                    f"batched={batched_mean_seconds*1e3:>10.4f} ms  "
                    f"speedup={serial_mean_seconds / batched_mean_seconds:>6.3f}x"
                )
            except Exception as exc:
                print(f"  N={n:>8}: FAILED ({type(exc).__name__}: {exc})")

    print("\n" + "=" * 70)
    print("ROUNDTRIP RECONSTRUCTION ERROR")
    print("=" * 70)

    for backend_name in available_backends:
        results["error"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)
        for n in n_values:
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue

            nt, nf = factorization
            signal = generate_benchmark_signal(nt * nf)
            try:
                error_record = benchmark_roundtrip_error(
                    signal,
                    backend_name,
                    nt,
                    nf,
                )
                results["error"][backend_name][n] = error_record
                print(
                    f"  N={n:>8}: max abs={error_record['max_abs_error']:.3e}, "
                    f"rel L2={error_record['relative_l2_error']:.3e} "
                    f"(nt={nt}, nf={nf})"
                )
            except Exception as exc:
                print(f"  N={n:>8}: FAILED ({type(exc).__name__}: {exc})")

    return results


def print_summary(results: dict[str, Any]) -> None:  # pragma: no cover
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    for transform_type in ("forward", "inverse", "batch_forward", "error"):
        backends_data = results.get(transform_type, {})
        if not backends_data:
            continue

        if transform_type == "error":
            print("\nROUNDTRIP ERROR:")
        elif transform_type == "batch_forward":
            print("\nBATCHED JAX FORWARD:")
        else:
            print(f"\n{transform_type.upper()} TRANSFORM:")
        print("-" * 40)

        labels = list(backends_data)
        all_n = sorted({n for records in backends_data.values() for n in records})
        header = f"{'N':>10} | " + " | ".join(f"{backend:>12}" for backend in labels)
        print(header)
        print("-" * len(header))

        for n in all_n:
            row = f"{n:>10} | "
            values = []
            for backend_name in labels:
                record = backends_data.get(backend_name, {}).get(n)
                if record is None:
                    values.append(f"{'SKIPPED':>12}")
                    continue
                if transform_type == "error":
                    values.append(f"{record['max_abs_error']:.3e}".rjust(12))
                elif transform_type == "batch_forward":
                    values.append(f"{record['batched_vs_serial_speedup']:>10.3f}x")
                else:
                    values.append(f"{record['mean_seconds']*1e3:>10.4f} ms")
            print(row + " | ".join(values))


def save_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> Path:  # pragma: no cover
    """Write benchmark results to JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def plot_results(  # pragma: no cover
    results: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "WDM Transform Runtime by Backend",
) -> Path:
    """Render a benchmark plot for forward and inverse transforms."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    transform_titles = {
        "forward": "Forward Transform",
        "inverse": "Inverse Transform",
        "batch_forward": "3x Forward Comparison (JAX)",
        "error": "Round-Trip Error",
    }
    backend_styles = {
        "numpy": {
            "color": "tab:blue",
            "zorder": 3,
        },
        "jax": {
            "color": "tab:orange",
            "zorder": 4,
        },
    }
    for ax, transform_type in zip(axes, ("forward", "inverse", "batch_forward", "error"), strict=True):
        backends_data = results.get(transform_type, {})
        for backend_name, backend_results in backends_data.items():
            ns = sorted(backend_results)
            if not ns:
                continue
            style = {
                "linewidth": 2.2,
                "markersize": 6.5,
                "label": backend_name,
                "linestyle": "-",
                "marker": "o",
                **backend_styles.get(backend_name, {}),
            }
            if transform_type == "error":
                error_values = [backend_results[n]["max_abs_error"] for n in ns]
                ax.plot(ns, error_values, **style)
            elif transform_type == "batch_forward":
                single_ms = [backend_results[n]["single_mean_seconds"] * 1e3 for n in ns]
                serial_ms = [backend_results[n]["serial_mean_seconds"] * 1e3 for n in ns]
                batched_ms = [backend_results[n]["batched_mean_seconds"] * 1e3 for n in ns]
                ax.plot(
                    ns,
                    single_ms,
                    linewidth=2.2,
                    markersize=6.5,
                    linestyle="-",
                    marker="o",
                    label="jax single",
                    color="tab:orange",
                    zorder=4,
                )
                ax.plot(
                    ns,
                    serial_ms,
                    linewidth=2.2,
                    markersize=6.5,
                    linestyle="-",
                    marker="s",
                    label="jax serial x3",
                    color="tab:red",
                    zorder=3,
                )
                ax.plot(
                    ns,
                    batched_ms,
                    linewidth=2.2,
                    markersize=6.5,
                    linestyle="-",
                    marker="^",
                    label="jax batched x3",
                    color="tab:green",
                    zorder=5,
                )
            else:
                mean_ms = [backend_results[n]["mean_seconds"] * 1e3 for n in ns]
                std_ms = [backend_results[n]["std_seconds"] * 1e3 for n in ns]
                line = ax.plot(ns, mean_ms, **style)[0]
                lower = np.maximum(np.array(mean_ms) - np.array(std_ms), 1e-9)
                upper = np.array(mean_ms) + np.array(std_ms)
                ax.fill_between(
                    ns,
                    lower,
                    upper,
                    color=line.get_color(),
                    alpha=0.12,
                    zorder=1,
                )

        ax.set_title(transform_titles[transform_type])
        ax.set_xlabel("Input size N")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Mean runtime (ms)")
    axes[3].set_ylabel("Max abs reconstruction error")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncols=max(1, len(labels)),
            frameon=True,
        )
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.80, wspace=0.28)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:  # pragma: no cover
    """CLI entry point for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark WDM transforms across backends and input sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wdm_transform.benchmarking
  python -m wdm_transform.benchmarking --backends numpy jax
  python -m wdm_transform.benchmarking --n 2048 4096 8192
  python -m wdm_transform.benchmarking --runs 5
  python -m wdm_transform.benchmarking --output-json benchmark.json
  python -m wdm_transform.benchmarking --output-plot benchmark.png
        """,
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
        help="Number of runs per benchmark (default: 7)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write benchmark results as JSON.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional path to write a benchmark plot image.",
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

    if args.output_json is not None:
        json_path = save_results(results, args.output_json)
        print(f"\nWrote benchmark results to {json_path}")

    if args.output_plot is not None:
        plot_path = plot_results(results, args.output_plot, title=args.plot_title)
        print(f"Wrote benchmark plot to {plot_path}")

    print("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
