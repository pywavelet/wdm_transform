"""Benchmarking CLI for WDM transforms across different backends and input sizes.

This module is excluded from pytest coverage and should not be imported
when loading the main wdm_transform package.

Run with: python -m wdm_transform.benchmarking [options]
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from . import backends, transforms


def validate_backend_available(backend_name: str) -> bool:  # pragma: no cover
    """Check if a backend is available."""
    try:
        backends.get_backend(backend_name)
        return True
    except ValueError:
        return False


def generate_benchmark_data(n: int) -> np.ndarray:  # pragma: no cover
    """Generate random complex benchmark data."""
    np.random.seed(42)
    return np.random.randn(n) + 1j * np.random.randn(n)


def find_factorization(n: int) -> tuple[int, int] | None:  # pragma: no cover
    """Find a factorization n = nt * nf where both nt and nf are even.
    
    Prioritizes balanced factorizations (nt close to sqrt(n)).
    Returns (nt, nf) or None if no valid factorization exists.
    """
    if n < 4:
        return None
    
    # Try factorizations starting from sqrt(n) and working outward
    sqrt_n = int(np.sqrt(n))
    
    for nt in range(sqrt_n, 1, -1):
        if n % nt != 0:
            continue
        nf = n // nt
        # Both must be even
        if nt % 2 == 0 and nf % 2 == 0:
            return (nt, nf)
    
    return None


def benchmark_forward(
    data: np.ndarray,
    backend_name: str,
    nt: int,
    nf: int,
    num_runs: int = 3,
) -> float:  # pragma: no cover
    """Benchmark the forward WDM transform.

    Returns the mean runtime in seconds.
    """
    backend = backends.get_backend(backend_name)
    fixed_params = {"a": 0.4, "d": 0.8, "dt": 1.0, "backend": backend}

    # Convert data to backend array
    backend_data = backend.asarray(data)

    # Warmup run
    transforms.forward_wdm(backend_data, nt=nt, nf=nf, **fixed_params)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        transforms.forward_wdm(backend_data, nt=nt, nf=nf, **fixed_params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def benchmark_inverse(
    coeffs: np.ndarray,
    backend_name: str,
    num_runs: int = 3,
) -> float:  # pragma: no cover
    """Benchmark the inverse WDM transform.

    Returns the mean runtime in seconds.
    """
    backend = backends.get_backend(backend_name)
    fixed_params = {"a": 0.4, "d": 0.8, "dt": 1.0, "backend": backend}

    # Convert coeffs to backend array
    backend_coeffs = backend.asarray(coeffs)

    # Warmup run
    transforms.inverse_wdm(backend_coeffs, **fixed_params)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        transforms.inverse_wdm(backend_coeffs, **fixed_params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def run_benchmarks(  # pragma: no cover
    backends_to_test: list[str],
    n_values: list[int],
    num_runs: int = 3,
) -> dict[str, Any]:
    """Run benchmark suite across backends and input sizes.

    Returns results dictionary with structure:
    {
        'forward': {backend_name: {n: time_in_seconds}},
        'inverse': {backend_name: {n: time_in_seconds}},
    }
    """
    results = {"forward": {}, "inverse": {}}

    # Validate backends
    available_backends = [b for b in backends_to_test if validate_backend_available(b)]
    if not available_backends:
        print(f"ERROR: No backends available from: {backends_to_test}")
        return results

    if len(available_backends) < len(backends_to_test):
        unavailable = set(backends_to_test) - set(available_backends)
        print(f"WARNING: Skipping unavailable backends: {unavailable}")

    # Forward transform benchmarks
    print("\n" + "=" * 70)
    print("FORWARD WDM TRANSFORM BENCHMARKS")
    print("=" * 70)

    for backend_name in available_backends:
        results["forward"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)

        for n in n_values:
            # Find a valid factorization for this N
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue
            
            nt, nf = factorization
            data = generate_benchmark_data(nt * nf)
            try:
                mean_time = benchmark_forward(data, backend_name, nt, nf, num_runs)
                results["forward"][backend_name][n] = mean_time
                print(f"  N={n:>8}: {mean_time*1e3:>10.4f} ms (nt={nt}, nf={nf})")
            except Exception as e:
                print(f"  N={n:>8}: FAILED ({type(e).__name__}: {e})")

    # Inverse transform benchmarks
    print("\n" + "=" * 70)
    print("INVERSE WDM TRANSFORM BENCHMARKS")
    print("=" * 70)

    for backend_name in available_backends:
        results["inverse"][backend_name] = {}
        print(f"\nBackend: {backend_name}")
        print("-" * 40)

        for n in n_values:
            # Find a valid factorization for this N
            factorization = find_factorization(n)
            if factorization is None:
                print(f"  N={n:>8}: SKIPPED (no valid factorization)")
                continue
            
            nt, nf = factorization
            coeffs = generate_benchmark_data(nt * nf).reshape(nt, nf)
            try:
                mean_time = benchmark_inverse(coeffs, backend_name, num_runs)
                results["inverse"][backend_name][n] = mean_time
                print(f"  N={n:>8}: {mean_time*1e3:>10.4f} ms (nt={nt}, nf={nf})")
            except Exception as e:
                print(f"  N={n:>8}: FAILED ({type(e).__name__}: {e})")

    return results


def print_summary(results: dict[str, Any]) -> None:  # pragma: no cover
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    for transform_type, backends_data in results.items():
        if not backends_data:
            continue

        print(f"\n{transform_type.upper()} TRANSFORM:")
        print("-" * 40)

        # Get all N values
        all_n = set()
        for n_dict in backends_data.values():
            all_n.update(n_dict.keys())
        all_n = sorted(all_n)

        # Header
        header = f"{'N':>10} | " + " | ".join(f"{b:>12}" for b in backends_data.keys())
        print(header)
        print("-" * len(header))

        # Rows
        for n in all_n:
            row = f"{n:>10} | "
            times = [backends_data.get(b, {}).get(n) for b in backends_data.keys()]
            time_strs = [
                f"{t*1e3:>10.4f} ms" if t is not None else f"{'SKIPPED':>12}" for t in times
            ]
            row += " | ".join(time_strs)
            print(row)


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
        """,
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        default=["numpy"],
        choices=["numpy", "jax", "cupy"],
        help="Backends to benchmark (default: numpy)",
    )

    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192, 16384],
        help="Input sizes to test (default: 2048 4096 8192 16384)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per benchmark (default: 3)",
    )

    args = parser.parse_args()

    results = run_benchmarks(
        backends_to_test=args.backends,
        n_values=args.n,
        num_runs=args.runs,
    )

    print_summary(results)
    print("\n")


if __name__ == "__main__":  # pragma: no cover
    main()

