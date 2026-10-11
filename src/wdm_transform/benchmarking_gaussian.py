"""Reproducible window-family benchmarks; timings are evidence, not CI tests.

Workers run in fresh processes so optional archived Meyer code and current code
use the same environment without mixing imported implementations. Array work
is synchronized with the existing benchmark helper. Host allocation probes run
separately from timing and do not measure JAX device allocations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import resource
except ImportError:  # Windows still supports timings and tracemalloc.
    resource = None


def _measure(fn: Callable[[], Any], runs: int) -> dict[str, Any]:
    from wdm_transform.benchmarking import _synchronize_result

    def call() -> Any:
        result = fn()
        if hasattr(result, "coeffs"):
            _synchronize_result(result.coeffs)
        elif hasattr(result, "data"):
            _synchronize_result(result.data)
        else:
            _synchronize_result(result)
        return result

    started = time.perf_counter()
    call()
    first = time.perf_counter() - started
    for _ in range(3):
        call()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    # Probe once separately: tracing overhead must not enter warm timings.
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    if rss is not None and sys.platform != "darwin":
        rss *= 1024
    return {
        "first_seconds": first,
        "median_seconds": float(np.median(samples)),
        "mean_seconds": float(np.mean(samples)),
        "std_seconds": float(np.std(samples)),
        "samples_seconds": samples,
        "host_tracemalloc_peak_bytes": peak,
        "process_maxrss_bytes": rss,
    }


def _window_measure(nt: int, family: str, support: int | None, runs: int) -> dict:
    from wdm_transform.backends import get_backend
    from wdm_transform.windows import phi_window

    if family == "gaussian":
        from wdm_transform.windows_gaussian import _cached_window, gaussian_window

        _cached_window.cache_clear()
        fn = lambda: gaussian_window(nt, 0.5, support)  # noqa: E731
    else:
        fn = lambda: phi_window(  # noqa: E731
            get_backend("numpy"), nt, 32, 1.0, 1.0 / 3.0, 1.0
        )
    started = time.perf_counter()
    samples = fn()
    cold = time.perf_counter() - started
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return {
        "nt": nt,
        "window": family,
        "support": support,
        "cold_seconds": cold,
        "repeat_median_seconds": float(np.median(times)),
        "repeat_mean_seconds": float(np.mean(times)),
        "cache": "bounded host cache" if family == "gaussian" else "uncached",
        "samples_bytes": samples.nbytes,
    }


def _worker(args: argparse.Namespace) -> None:
    # Direct-file workers select the source before any package import.
    if args.source is not None:
        sys.path.insert(0, str(Path(args.source).resolve()))
    from wdm_transform import TimeSeries, transforms
    from wdm_transform.backends import get_backend

    backend = get_backend(args.worker_backend)
    options = {}
    if args.worker_window == "gaussian":
        options = dict(window="gaussian", nu=0.5, support=args.worker_support)
    # Archived code has no window keyword; omitted options reproduce defaults.
    rows = []
    windows = []
    for nt in sorted({64 if n == 4096 else 256 for n in args.n}):
        row = _window_measure(nt, args.worker_window, args.worker_support, args.runs)
        row["implementation"] = args.implementation
        row["backend"] = args.worker_backend
        windows.append(row)
    for n in args.n:
        nt = 64 if n == 4096 else 256
        nf = n // nt
        for batch in args.batch:
            data = np.random.default_rng(321).normal(size=(batch, n))
            series = TimeSeries(data, dt=1.0, backend=backend)
            freq = series.to_frequency_series()
            params = dict(a=1.0 / 3.0, d=1.0, dt=1.0, backend=backend, **options)
            common = dict(
                implementation=args.implementation,
                backend=args.worker_backend,
                window=args.worker_window,
                nu=0.5 if options else None,
                support=args.worker_support,
                n=n,
                nt=nt,
                nf=nf,
                batch=batch,
                dtype="float64",
            )
            calls = {
                "time_to_wdm": partial(
                    transforms.from_time_to_wdm, series.data, nt=nt, nf=nf, **params
                ),
                "freq_to_wdm": partial(
                    transforms.from_freq_to_wdm, freq.data, nt=nt, nf=nf, **params
                ),
                "series_to_wdm": partial(series.to_wdm, nt=nt, **options),
            }
            for operation, call in calls.items():
                rows.append(
                    dict(common, operation=operation, **_measure(call, args.runs))
                )
            wdm = series.to_wdm(nt=nt, **options)
            inverse_calls = {
                "wdm_to_time": partial(
                    transforms.from_wdm_to_time, wdm.coeffs, **params
                ),
                "wdm_to_freq": partial(
                    transforms.from_wdm_to_freq, wdm.coeffs, **params
                ),
                "wdm_to_series": wdm.to_time_series,
            }
            for operation, call in inverse_calls.items():
                rows.append(
                    dict(common, operation=operation, **_measure(call, args.runs))
                )
            restored = np.asarray(wdm.to_time_series().data)
            residual = float(np.linalg.norm(restored - data) / np.linalg.norm(data))
            for row in rows[-6:]:
                row["relative_l2"] = residual
            print(
                f"{args.implementation} {args.worker_backend} {args.worker_window} "
                f"q={args.worker_support} N={n} batch={batch} error={residual:.3e}",
                flush=True,
            )
    Path(args.worker_output).write_text(
        json.dumps({"transforms": rows, "windows": windows}, indent=2) + "\n"
    )


def _environment() -> dict:
    versions = {}
    for name in ("wdm-transform", "numpy", "scipy", "jax", "jaxlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_model = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_model=cpu_model,
        head_revision=revision,
        revision_note="HEAD is the starting revision; working source may be modified.",
        thread_environment={
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "XLA_FLAGS",
            )
        },
        versions=versions,
        jax_platforms=os.environ.get("JAX_PLATFORMS", "default"),
        timer="time.perf_counter; synchronized results",
        first_calls=(
            "Chronological first calls; later operations may reuse compiled kernels."
        ),
        memory=(
            "Host tracemalloc probes run separately; process RSS is cumulative. "
            "JAX device allocator peaks are not measured."
        ),
    )


def _vectorized_comparison(args: argparse.Namespace) -> None:
    """Compare an all-band experiment against production chunks on a modest grid."""
    from wdm_transform.backends import get_backend

    get_backend("jax")  # Enables float64 before constructing experimental arrays.
    import jax
    import jax.numpy as jnp

    from wdm_transform.transforms.jax_gaussian import _analysis, _device_window

    nt, nf, support = 256, 256, 16
    offsets = jnp.arange(-support * nt // 2, support * nt // 2)
    bands = jnp.arange(nf + 1)
    times = jnp.arange(nt)
    weights = _device_window(nt, 0.5, support)

    @jax.jit
    def vectorized(spectrum: jax.Array) -> jax.Array:
        indices = (bands[:, None] * (nt // 2) + offsets[None, :]) % (nt * nf)
        taps = spectrum[:, indices] * weights[None, None, :]
        folded = taps.reshape(spectrum.shape[0], nf + 1, support, nt).sum(axis=2)
        folded = jnp.roll(folded, (-support * nt // 2) % nt, axis=-1)
        local = jnp.fft.ifft(folded, axis=-1) * nt
        phase = jnp.where((times[None, :] + bands[:, None]) % 2 == 0, 1.0, -1j)
        sign = jnp.where((times[None, :] * bands[:, None]) % 2 == 0, 1.0, -1.0)
        interior = jnp.sqrt(2.0) * sign * (phase * local).real
        edge = local[:, :, (2 * times) % nt].real / jnp.sqrt(2.0)
        is_edge = (bands == 0) | (bands == nf)
        return jnp.where(is_edge[None, :, None], edge, interior).transpose(0, 2, 1)

    results = []
    for batch in (1, 3):
        signal = np.random.default_rng(321).normal(size=(batch, nt * nf))
        spectrum = jnp.asarray(np.fft.fft(signal, axis=-1))
        chunked = partial(_analysis, spectrum, weights, nt=nt, nf=nf, support=support)
        all_bands = partial(vectorized, spectrum)
        # Each shape's first timings below include compilation, before comparisons.
        chunk_times = _measure(chunked, args.runs)
        vector_times = _measure(all_bands, args.runs)
        delta = float(np.max(np.abs(np.asarray(chunked()) - np.asarray(all_bands()))))
        results.append(
            dict(
                n=nt * nf,
                nt=nt,
                nf=nf,
                support=support,
                batch=batch,
                maximum_coefficient_delta=delta,
                chunked=chunk_times,
                vectorized=vector_times,
                nominal_chunk_tap_bytes=batch * 16 * support * nt * 16,
                nominal_vector_tap_bytes=batch * (nf + 1) * support * nt * 16,
                nominal_vector_tap_bytes_at_n1048576=batch * 4097 * support * nt * 16,
            )
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "vectorized_comparison.json").write_text(
        json.dumps(
            {
                "environment": _environment(),
                "results": results,
                "memory_note": "Nominal tensor sizes; XLA may fuse allocations.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


def _report(payload: dict, destination: Path) -> None:
    rows = payload["transforms"]
    keyed = {
        (
            r["implementation"],
            r["backend"],
            r["window"],
            r["support"],
            r["n"],
            r["batch"],
            r["operation"],
        ): r
        for r in rows
    }
    lines = [
        "# Window benchmark report",
        "",
        f"Environment: `{payload['environment']['platform']}`, "
        f"Python {payload['environment']['python']}.",
        "",
        f"Measured {payload['runs']} warm repetitions after three warmups. "
        "The first call is retained separately and synchronized, including any "
        "compilation incurred by that call. Workers start fresh per family/backend; "
        "later operations may reuse earlier compiled kernels.",
        "",
        "Host tracemalloc peaks come from separate untimed probes. RSS is the "
        "cumulative process maximum, including input/output and compilation. "
        "JAX device allocations are not measured.",
        "",
        "All transforms use float64, dt=1, and seeded noise. N=4096 uses nt=64; "
        "larger shapes use nt=256. Gaussian nu=0.5.",
        "",
        "| Backend | N | Batch | Window | q | Forward ms | Inverse ms | "
        "Ratio to current Meyer fwd/inv | Relative L2 | RSS MiB |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        if row["implementation"] != "current" or row["operation"] != "time_to_wdm":
            continue
        prefix = (
            "current",
            row["backend"],
            row["window"],
            row["support"],
            row["n"],
            row["batch"],
        )
        inverse = keyed[prefix + ("wdm_to_time",)]
        meyer = ("current", row["backend"], "meyer", None, row["n"], row["batch"])
        fr = row["median_seconds"] / keyed[meyer + ("time_to_wdm",)]["median_seconds"]
        ir = (
            inverse["median_seconds"]
            / keyed[meyer + ("wdm_to_time",)]["median_seconds"]
        )
        rss = inverse["process_maxrss_bytes"]
        rss_label = f"{rss/2**20:.1f}" if rss is not None else "unavailable"
        lines.append(
            f"| {row['backend']} | {row['n']} | {row['batch']} | {row['window']} "
            f"| {row['support'] or '—'} | {1e3*row['median_seconds']:.3f} "
            f"| {1e3*inverse['median_seconds']:.3f} | {fr:.2f}/{ir:.2f} "
            f"| {row['relative_l2']:.3e} "
            f"| {rss_label} |"
        )
    comparisons = []
    for row in rows:
        if row["implementation"] != "current" or row["window"] != "meyer":
            continue
        key = (
            "baseline",
            row["backend"],
            "meyer",
            None,
            row["n"],
            row["batch"],
            row["operation"],
        )
        if key in keyed:
            comparisons.append(
                dict(
                    backend=row["backend"],
                    n=row["n"],
                    batch=row["batch"],
                    operation=row["operation"],
                    after_before_ratio=row["median_seconds"]
                    / keyed[key]["median_seconds"],
                )
            )
    payload["meyer_comparison"] = comparisons
    if comparisons:
        ratios = np.array([r["after_before_ratio"] for r in comparisons])
        lines += [
            "",
            "## Meyer before/after",
            "",
            "Archived source and current source ran in adjacent fresh workers with "
            "identical shapes, inputs, dependencies and timing protocol. "
            "Separate-process variability and machine load still affect these ratios; "
            "the 5% objective is a performance investigation target, "
            "not a CI assertion.",
            "",
            f"Across {len(ratios)} operation/shape/backend pairs, median after/before "
            f"ratio is {np.median(ratios):.3f}; "
            f"range {ratios.min():.3f}–{ratios.max():.3f}. "
            f"{int(np.sum(ratios > 1.05))} pairs exceed 1.05.",
        ]
    lines += [
        "",
        "Full means, medians, samples, first calls, memory probes, support errors "
        "and host window cold/cache measurements are in `results.json` "
        "and `transforms.csv`.",
    ]
    destination.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=["numpy", "jax"], default=["numpy", "jax"]
    )
    parser.add_argument(
        "--n", nargs="+", type=int, default=[4096, 65536, 262144, 1048576]
    )
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--support", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument(
        "--meyer-only",
        action="store_true",
        help="Compare archived/current Meyer only, omitting Gaussian workers.",
    )
    parser.add_argument(
        "--vectorized-comparison",
        action="store_true",
        help="Compare experimental all-band JAX analysis against chunks at N=65536.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("test_output/wdg_benchmarks")
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Archived starting package src directory; no git mutation.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-window", help=argparse.SUPPRESS)
    parser.add_argument("--worker-support", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", default="current", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runs < 1 or any(batch < 1 for batch in args.batch):
        parser.error("runs and batch sizes must be positive")
    for n in args.n:
        nt = 64 if n == 4096 else 256
        if n < nt * 2 or n % (2 * nt):
            parser.error("N must be an even multiple of nt (64 for N=4096, else 256)")
        if (
            not args.worker
            and not args.meyer_only
            and any(q < 1 or q > n // nt for q in args.support)
        ):
            parser.error("supports must satisfy 1 <= support <= nf")
    if args.worker:
        _worker(args)
        return
    if args.vectorized_comparison:
        _vectorized_comparison(args)
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        environment=_environment(), runs=args.runs, transforms=[], windows=[]
    )
    current_source = Path(__file__).resolve().parents[1]
    workers = []
    for backend in args.backends:
        if args.baseline_source is not None:
            workers.append(("baseline", args.baseline_source, backend, "meyer", None))
        workers.append(("current", current_source, backend, "meyer", None))
        if not args.meyer_only:
            workers.extend(
                ("current", current_source, backend, "gaussian", q)
                for q in args.support
            )
    for implementation, source, backend, family, support in workers:
        output = args.outdir / f"{implementation}_{backend}_{family}_{support}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(source.resolve()),
            "--implementation",
            implementation,
            "--worker-backend",
            backend,
            "--worker-window",
            family,
            "--worker-output",
            str(output.resolve()),
            "--runs",
            str(args.runs),
            "--n",
            *map(str, args.n),
            "--batch",
            *map(str, args.batch),
        ]
        if support is not None:
            command += ["--worker-support", str(support)]
        subprocess.run(command, check=True)
        result = json.loads(output.read_text())
        payload["transforms"].extend(result["transforms"])
        payload["windows"].extend(result["windows"])
        # Persist after every successful worker so interrupted runs retain evidence.
        (args.outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _report(payload, args.outdir / "report.md")
    (args.outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = [key for key in payload["transforms"][0] if key != "samples_seconds"]
    with (args.outdir / "transforms.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields}
            for row in payload["transforms"]
        )
    print(f"Saved benchmark evidence to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
