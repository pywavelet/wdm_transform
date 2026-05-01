"""Benchmarking CLI for the distinct low-level WDM kernels.

This module benchmarks:

* ``from_freq_to_wdm``
* ``from_wdm_to_freq``

For each kernel it compares:

* scalar input
* batched input
* serial application over the same batch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from . import backends, transforms

DEFAULT_BACKENDS = ["numpy", "jax"]
DEFAULT_POW2_RANGE = (11, 20)
MIN_POW2 = 8
MAX_POW2 = 25
FIT_MIN_POW2 = 16
DEFAULT_BATCH_SIZE = 3
DEFAULT_OUTDIR = Path("benchmark_artifacts")
DEFAULT_JSON_NAME = "benchmark_results.json"
DEFAULT_PLOT_NAME = "benchmark_runtime.png"
FIXED_PARAMS = {"a": 0.4, "d": 1, "dt": 1.0}

def _resolve_device(requested: str) -> str:
    """Resolve 'cpu'/'gpu'/'auto' and configure JAX platform env vars.

    Must be called before any backend is imported.
    """
    if requested == "auto":
        device = "gpu" if shutil.which("nvidia-smi") is not None else "cpu"
    else:
        device = requested

    jax_platform = "cuda" if device == "gpu" else "cpu"
    os.environ.setdefault("JAX_PLATFORMS", jax_platform)
    os.environ.setdefault("JAX_PLATFORM_NAME", jax_platform)
    return device


OPERATION_TITLES = {
    "from_freq": "Forward Kernel: from_freq_to_wdm",
    "to_freq": "Inverse Kernel: from_wdm_to_freq",
}

OPERATION_DESCRIPTIONS = {
    "from_freq": "WDM projection kernel from Fourier-domain samples.",
    "to_freq": "WDM reconstruction kernel to Fourier-domain samples.",
}


def resolve_n_values(
    *,
    pow2_range: tuple[int, int] | None = None,
) -> list[int]:
    """Resolve benchmark sizes from an inclusive power-of-two range."""
    start, end = pow2_range if pow2_range is not None else DEFAULT_POW2_RANGE
    if start < MIN_POW2 or end > MAX_POW2:
        raise ValueError(f"pow2 range must stay within [{MIN_POW2}, {MAX_POW2}].")
    if start > end:
        raise ValueError("pow2 start must be <= end.")
    return [2**power for power in range(start, end + 1)]


def validate_backend_available(backend_name: str) -> bool:  # pragma: no cover
    """Check if a backend is importable and registered."""
    try:
        backends.get_backend(backend_name)
        return True
    except (ImportError, ValueError):
        return False


def generate_benchmark_signal(
    n: int,
    *,
    batch_size: int = 1,
) -> np.ndarray:  # pragma: no cover
    """Generate deterministic real benchmark signals."""
    rng = np.random.default_rng(42)
    samples = rng.standard_normal((batch_size, n))
    return samples[0] if batch_size == 1 else samples


def generate_benchmark_spectrum(
    n: int,
    *,
    batch_size: int = 1,
) -> np.ndarray:  # pragma: no cover
    """Generate deterministic Fourier-domain samples from real signals."""
    signals = generate_benchmark_signal(n, batch_size=batch_size)
    return np.fft.fft(signals, axis=-1)


def find_factorization(n: int) -> tuple[int, int] | None:  # pragma: no cover
    """Find an even-even factorization ``n = nt * nf``."""
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
) -> np.ndarray:
    """Warm up once, then return per-run runtimes in seconds."""
    warmup_result = fn()
    _synchronize_result(warmup_result)

    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn()
        _synchronize_result(result)
        times.append(time.perf_counter() - start)

    return np.asarray(times, dtype=float)


def _prepare_inputs(
    backend_name: str,
    *,
    n: int,
    nt: int,
    nf: int,
    batch_size: int,
) -> dict[str, Any]:
    backend = backends.get_backend(backend_name)
    fixed_params = {**FIXED_PARAMS, "backend": backend}

    freq_scalar = backend.asarray(generate_benchmark_spectrum(n))
    freq_batch = backend.asarray(generate_benchmark_spectrum(n, batch_size=batch_size))

    coeff_scalar = transforms.from_freq_to_wdm(
        freq_scalar,
        nt=nt,
        nf=nf,
        **fixed_params,
    )
    coeff_batch = transforms.from_freq_to_wdm(
        freq_batch,
        nt=nt,
        nf=nf,
        **fixed_params,
    )
    _synchronize_result((coeff_scalar, coeff_batch))

    return {
        "freq_scalar": freq_scalar,
        "freq_batch": freq_batch,
        "coeff_scalar": coeff_scalar,
        "coeff_batch": coeff_batch,
    }


def _build_scalar_call(
    operation: str,
    backend_name: str,
    payload: dict[str, Any],
    *,
    nt: int,
    nf: int,
) -> Callable[[], Any]:
    fixed_params = {**FIXED_PARAMS, "backend": backends.get_backend(backend_name)}
    if operation == "from_freq":
        return lambda: transforms.from_freq_to_wdm(
            payload["freq_scalar"],
            nt=nt,
            nf=nf,
            **fixed_params,
        )
    if operation == "to_freq":
        return lambda: transforms.from_wdm_to_freq(payload["coeff_scalar"], **fixed_params)
    raise ValueError(f"Unknown operation {operation!r}.")


def _build_batch_call(
    operation: str,
    backend_name: str,
    payload: dict[str, Any],
    *,
    nt: int,
    nf: int,
) -> Callable[[], Any]:
    fixed_params = {**FIXED_PARAMS, "backend": backends.get_backend(backend_name)}
    if operation == "from_freq":
        return lambda: transforms.from_freq_to_wdm(
            payload["freq_batch"],
            nt=nt,
            nf=nf,
            **fixed_params,
        )
    if operation == "to_freq":
        return lambda: transforms.from_wdm_to_freq(payload["coeff_batch"], **fixed_params)
    raise ValueError(f"Unknown operation {operation!r}.")


def _build_serial_call(
    operation: str,
    backend_name: str,
    payload: dict[str, Any],
    *,
    nt: int,
    nf: int,
    batch_size: int,
) -> Callable[[], list[Any]]:
    fixed_params = {**FIXED_PARAMS, "backend": backends.get_backend(backend_name)}
    if operation == "from_freq":
        return lambda: [
            transforms.from_freq_to_wdm(payload["freq_batch"][index], nt=nt, nf=nf, **fixed_params)
            for index in range(batch_size)
        ]
    if operation == "to_freq":
        return lambda: [
            transforms.from_wdm_to_freq(payload["coeff_batch"][index], **fixed_params)
            for index in range(batch_size)
        ]
    raise ValueError(f"Unknown operation {operation!r}.")


def _timing_record(
    samples_seconds: np.ndarray,
    *,
    nt: int,
    nf: int,
    shape: tuple[int, ...],
) -> dict[str, float | int | list[int]]:
    lower_seconds, upper_seconds = np.percentile(samples_seconds, [16.0, 84.0])
    return {
        "mean_seconds": float(np.mean(samples_seconds)),
        "median_seconds": float(np.median(samples_seconds)),
        "std_seconds": float(np.std(samples_seconds)),
        "p16_seconds": float(lower_seconds),
        "p84_seconds": float(upper_seconds),
        "nt": nt,
        "nf": nf,
        "shape": list(shape),
        "samples_seconds": samples_seconds.tolist(),
    }


def _fit_runtime_curve(ns: list[int], medians_seconds: list[float]) -> dict[str, float] | None:
    """Fit ``a N log2(N) + b N + c`` on the large-N regime."""
    fit_points = [
        (float(n), float(runtime))
        for n, runtime in zip(ns, medians_seconds, strict=True)
        if n >= 2**FIT_MIN_POW2
    ]
    if len(fit_points) < 3:
        return None

    fit_ns = np.asarray([point[0] for point in fit_points], dtype=float)
    fit_ys = np.asarray([point[1] for point in fit_points], dtype=float)
    design = np.column_stack(
        [
            fit_ns * np.log2(fit_ns),
            fit_ns,
            np.ones_like(fit_ns),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, fit_ys, rcond=None)
    return {
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
    }


def _evaluate_runtime_curve(ns: list[int], coeffs: dict[str, float]) -> np.ndarray:
    ns_array = np.asarray(ns, dtype=float)
    return (
        coeffs["a"] * ns_array * np.log2(ns_array)
        + coeffs["b"] * ns_array
        + coeffs["c"]
    )


def run_benchmarks(  # pragma: no cover
    backends_to_test: list[str],
    n_values: list[int],
    *,
    num_runs: int = 7,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the benchmark suite across backends and input sizes."""
    available_backends = [name for name in backends_to_test if validate_backend_available(name)]
    results: dict[str, Any] = {
        "metadata": {
            "requested_backends": backends_to_test,
            "available_backends": available_backends,
            "n_values": n_values,
            "num_runs": num_runs,
            "batch_size": batch_size,
            "parameters": FIXED_PARAMS,
        },
        "operations": {
            operation: {
                "description": OPERATION_DESCRIPTIONS[operation],
                "fits": {},
                "results": {},
            }
            for operation in OPERATION_TITLES
        },
    }

    if not available_backends:
        print(f"ERROR: No backends available from: {backends_to_test}")
        return results

    if len(available_backends) < len(backends_to_test):
        unavailable = sorted(set(backends_to_test) - set(available_backends))
        print(f"WARNING: Skipping unavailable backends: {', '.join(unavailable)}")

    for operation in OPERATION_TITLES:
        print("\n" + "=" * 72)
        print(OPERATION_TITLES[operation])
        print("=" * 72)

        for backend_name in available_backends:
            backend_results = results["operations"][operation]["results"]
            backend_results[backend_name] = {}
            print(f"\nBackend: {backend_name}")
            print("-" * 44)

            for n in n_values:
                factorization = find_factorization(n)
                if factorization is None:
                    print(f"  N={n:>8}: SKIPPED (no valid even-even factorization)")
                    continue

                nt, nf = factorization
                try:
                    payload = _prepare_inputs(
                        backend_name,
                        n=n,
                        nt=nt,
                        nf=nf,
                        batch_size=batch_size,
                    )

                    scalar_samples = _measure_runtime(
                        _build_scalar_call(operation, backend_name, payload, nt=nt, nf=nf),
                        num_runs,
                    )
                    batch_samples = _measure_runtime(
                        _build_batch_call(operation, backend_name, payload, nt=nt, nf=nf),
                        num_runs,
                    )
                    serial_samples = _measure_runtime(
                        _build_serial_call(
                            operation,
                            backend_name,
                            payload,
                            nt=nt,
                            nf=nf,
                            batch_size=batch_size,
                        ),
                        num_runs,
                    )
                    speedup_samples = serial_samples / batch_samples

                    scalar_shape = (n,) if operation == "from_freq" else (nt, nf + 1)
                    batch_shape = (
                        (batch_size, n)
                        if operation == "from_freq"
                        else (batch_size, nt, nf + 1)
                    )
                    speedup_p16, speedup_p84 = np.percentile(speedup_samples, [16.0, 84.0])
                    backend_results[backend_name][n] = {
                        "scalar": _timing_record(
                            scalar_samples,
                            nt=nt,
                            nf=nf,
                            shape=scalar_shape,
                        ),
                        "batch": _timing_record(
                            batch_samples,
                            nt=nt,
                            nf=nf,
                            shape=batch_shape,
                        ),
                        "serial": _timing_record(
                            serial_samples,
                            nt=nt,
                            nf=nf,
                            shape=batch_shape,
                        ),
                        "speedup_serial_over_batch": {
                            "mean": float(np.mean(speedup_samples)),
                            "median": float(np.median(speedup_samples)),
                            "std": float(np.std(speedup_samples)),
                            "p16": float(speedup_p16),
                            "p84": float(speedup_p84),
                            "samples": speedup_samples.tolist(),
                        },
                    }
                    print(
                        f"  N={n:>8}: scalar={np.median(scalar_samples)*1e3:>9.3f} ms | "
                        f"batch={np.median(batch_samples)*1e3:>9.3f} ms | "
                        f"serial {batch_size}x={np.median(serial_samples)*1e3:>9.3f} ms | "
                        f"speedup={np.median(speedup_samples):>6.2f}x"
                    )
                except Exception as exc:
                    print(f"  N={n:>8}: FAILED ({type(exc).__name__}: {exc})")

            fitted_ns = sorted(backend_results[backend_name])
            fitted_medians = [
                backend_results[backend_name][n]["scalar"]["median_seconds"]
                for n in fitted_ns
            ]
            fit = _fit_runtime_curve(fitted_ns, fitted_medians)
            if fit is not None:
                results["operations"][operation]["fits"][backend_name] = {
                    **fit,
                    "model": "a*N*log2(N) + b*N + c",
                    "fit_min_pow2": FIT_MIN_POW2,
                }

    return results


def print_summary(results: dict[str, Any]) -> None:  # pragma: no cover
    """Print a compact summary table."""
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    operations = results.get("operations", {})
    for operation, operation_data in operations.items():
        print(f"\n{OPERATION_TITLES[operation]}")
        print("-" * 72)
        for backend_name, backend_results in operation_data.get("results", {}).items():
            ns = sorted(backend_results)
            if not ns:
                continue
            print(f"  {backend_name}")
            for n in ns:
                record = backend_results[n]
                print(
                    f"    N={n:>8} | scalar={record['scalar']['median_seconds']*1e3:>9.3f} ms | "
                    f"batch={record['batch']['median_seconds']*1e3:>9.3f} ms | "
                    f"serial={record['serial']['median_seconds']*1e3:>9.3f} ms | "
                    f"speedup={record['speedup_serial_over_batch']['median']:>6.2f}x"
                )


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
    title: str = "WDM Kernel Benchmark Runtime Comparison",
) -> Path:
    """Render a benchmark plot with runtimes and batch speedups."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    operations = ["from_freq", "to_freq"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    backend_styles = {
        "numpy": {"color": "tab:blue"},
        "jax": {"color": "tab:orange"},
        "cupy": {"color": "tab:green"},
    }
    for column, operation in enumerate(operations):
        runtime_ax = axes[0, column]
        speedup_ax = axes[1, column]
        runtime_ax.set_title(OPERATION_TITLES[operation])

        operation_results = results.get("operations", {}).get(operation, {}).get("results", {})
        for backend_name, backend_results in operation_results.items():
            ns = sorted(backend_results)
            if not ns:
                continue

            backend_style = backend_styles.get(backend_name, {})
            median_ms = [backend_results[n]["scalar"]["median_seconds"] * 1e3 for n in ns]
            lower_ms = [backend_results[n]["scalar"]["p16_seconds"] * 1e3 for n in ns]
            upper_ms = [backend_results[n]["scalar"]["p84_seconds"] * 1e3 for n in ns]
            runtime_ax.fill_between(
                ns,
                np.maximum(np.asarray(lower_ms), 1e-9),
                np.asarray(upper_ms),
                color=backend_style.get("color"),
                alpha=0.18,
                zorder=1,
                label=backend_name,
            )
            fit = results.get("operations", {}).get(operation, {}).get("fits", {}).get(backend_name)
            if fit is not None:
                fit_ns = [n for n in ns if n >= 2**FIT_MIN_POW2]
                fit_ms = _evaluate_runtime_curve(fit_ns, fit) * 1e3
                runtime_ax.plot(
                    fit_ns,
                    fit_ms,
                    linewidth=2.2,
                    linestyle="--",
                    color=backend_style.get("color"),
                    alpha=0.95,
                )

            speedup_median = [
                backend_results[n]["speedup_serial_over_batch"]["median"] for n in ns
            ]
            speedup_p16 = [
                backend_results[n]["speedup_serial_over_batch"]["p16"] for n in ns
            ]
            speedup_p84 = [
                backend_results[n]["speedup_serial_over_batch"]["p84"] for n in ns
            ]
            speedup_ax.fill_between(
                ns,
                np.asarray(speedup_p16),
                np.asarray(speedup_p84),
                color=backend_style.get("color"),
                alpha=0.18,
                zorder=1,
                label=backend_name,
            )

        runtime_ax.set_xscale("log", base=2)
        runtime_ax.set_yscale("log")
        runtime_ax.grid(True, which="both", alpha=0.3)
        runtime_ax.set_xlabel("Input size N")
        if column == 0:
            runtime_ax.set_ylabel("Runtime (ms)")

        speedup_ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
        speedup_ax.set_xscale("log", base=2)
        speedup_ax.grid(True, which="both", alpha=0.3)
        speedup_ax.set_xlabel("Input size N")
        if column == 0:
            speedup_ax.set_ylabel("Serial / batch speedup")

    runtime_handles, runtime_labels = axes[0, 0].get_legend_handles_labels()
    if runtime_handles:
        fig.legend(
            runtime_handles,
            runtime_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncols=max(1, min(6, len(runtime_labels))),
            frameon=True,
        )

    speedup_handles, speedup_labels = axes[1, 0].get_legend_handles_labels()
    if speedup_handles:
        fig.legend(
            speedup_handles,
            speedup_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=max(1, len(speedup_labels)),
            frameon=True,
        )

    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.28, hspace=0.26)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark low-level WDM kernels across backends and input sizes.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help=(
            "Target device for JAX (and CuPy). "
            "'auto' detects GPU via nvidia-smi. "
            "Forces JAX platform."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["numpy", "jax", "cupy"],
        help=(
            "Backends to benchmark. "
            "Default: numpy+jax on CPU, numpy+jax+cupy on GPU."
        ),
    )
    parser.add_argument(
        "--pow2",
        nargs=2,
        metavar=("START", "END"),
        type=int,
        help="Inclusive exponent range for N=2**p (min 8, max 25). Default: 11 20.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Number of timed runs per benchmark (default: 7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for batched-vs-serial comparisons (default: 3)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where benchmark JSON and plot will be written.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional explicit path for the JSON benchmark summary.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional explicit path for the benchmark plot.",
    )
    parser.add_argument(
        "--plot-title",
        default="WDM Kernel Benchmark Runtime Comparison",
        help="Title to use for the generated benchmark plot.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI entry point for benchmarking."""
    args = _build_parser().parse_args(argv)

    # Resolve device and configure JAX platform BEFORE any backend is imported.
    device = _resolve_device(args.device)
    print(f"Device: {device.upper()}")

    backends_to_test = args.backends or (
        ["numpy", "jax", "cupy"] if device == "gpu" else DEFAULT_BACKENDS
    )

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    pow2_range = tuple(args.pow2) if args.pow2 is not None else None
    try:
        n_values = resolve_n_values(pow2_range=pow2_range)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Log which JAX devices are in use before running (confirms CPU vs GPU).
    _jax = None
    try:
        import jax as _jax  # noqa: PLC0415

        print(f"JAX devices: {_jax.devices()}")
    except Exception:
        pass

    results = run_benchmarks(
        backends_to_test=backends_to_test,
        n_values=n_values,
        num_runs=args.runs,
        batch_size=args.batch_size,
    )

    if _jax is not None:
        results["metadata"]["jax_devices"] = [str(d) for d in _jax.devices()]

    print_summary(results)

    output_json = args.output_json or (args.outdir / DEFAULT_JSON_NAME)
    output_plot = args.output_plot or (args.outdir / DEFAULT_PLOT_NAME)
    plot_title = args.plot_title + f" ({device.upper()})"
    json_path = save_results(results, output_json)
    plot_path = plot_results(results, output_plot, title=plot_title)

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
