"""
Colab-friendly benchmark and accuracy script for ``wdm_transform``.

This file is written as a plain Python script with ``# %%`` cell markers so it
can be opened in lightweight Jupyter editors, executed with ``python3``, or run
from a Google Colab notebook via ``%run``.

Recommended pairing:

  docs/studies/benchmarks/colab_benchmarks.ipynb

That notebook clones the repository into the Colab runtime, installs the local
package, and then executes this script.

Outputs:

  colab_benchmark_results.json
      Aggregate runtime and accuracy report for NumPy, JAX CPU, JAX GPU,
      and CuPy when available.

  colab_benchmark_results_<backend>.json
      Raw per-backend benchmark dumps produced by isolated worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(
    globals().get("__file__", "docs/studies/benchmarks/colab_benchmarks.py")
).resolve()
RESULTS_PATH = Path("colab_benchmark_results.json")

NUM_RUNS = 5
DT = 1.0
A_PARAM = 1.0 / 3.0
D_PARAM = 1.0
BATCH_SIZE = 3
N_VALUES = [2**power for power in range(11, 21)]
BATCH_N_VALUES = [2**power for power in range(11, 19)]
SEED_SINGLE = 42
SEED_BATCH = 314


# %%
def _gpu_name() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "CPU-only"
    return output.strip().splitlines()[0]


def _cpu_name() -> str:
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _hardware_fingerprint() -> dict[str, Any]:
    return {
        "cpu": _cpu_name(),
        "gpu": _gpu_name(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _nt(n: int) -> int:
    nt = int(n**0.5)
    if nt % 2 != 0:
        nt -= 1
    while nt >= 2:
        if n % nt == 0:
            nf = n // nt
            if nf % 2 == 0:
                return nt
        nt -= 2
    raise ValueError(f"Could not find an even-even factorization for n={n}.")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _synchronize(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if hasattr(value, "coeffs"):
        _synchronize(value.coeffs)
        return
    if hasattr(value, "data"):
        _synchronize(value.data)
        return
    if hasattr(value, "deviceSynchronize"):
        value.deviceSynchronize()
        return
    if isinstance(value, dict):
        for item in value.values():
            _synchronize(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _synchronize(item)
        return
    if "cupy" in sys.modules:
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def _measure(fn: Any, runs: int) -> np.ndarray:
    warmup = fn()
    _synchronize(warmup)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        _synchronize(result)
        samples.append(time.perf_counter() - start)
    return np.asarray(samples, dtype=float)


def _rng_signal(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + n)
    return rng.standard_normal(n)


def _rng_batch(n: int, *, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + n)
    return rng.standard_normal((batch_size, n))


def _timing_stats(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean_seconds": float(np.mean(samples)),
        "std_seconds": float(np.std(samples)),
        "median_seconds": float(np.median(samples)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def _benchmark_single_backend(
    backend_name: str,
    *,
    runs: int,
    n_values: list[int],
    jax_platform: str | None = None,
) -> dict[str, Any]:
    from wdm_transform import TimeSeries, WDM

    runtime_metadata: dict[str, Any] = {
        "backend": backend_name,
        "jax_platform": jax_platform,
        "hardware": _hardware_fingerprint(),
        "n_values": n_values,
        "batch_n_values": BATCH_N_VALUES,
        "num_runs": runs,
        "batch_size": BATCH_SIZE,
        "parameters": {
            "dt": DT,
            "a": A_PARAM,
            "d": D_PARAM,
        },
    }

    if backend_name == "jax":
        import jax

        runtime_metadata["jax_default_backend"] = jax.default_backend()
        runtime_metadata["jax_devices"] = [str(device) for device in jax.devices()]
        requested_platform = jax_platform or jax.default_backend()
        if not any(device.platform == requested_platform for device in jax.devices()):
            raise RuntimeError(
                f"Requested JAX platform {requested_platform!r} is not available. "
                f"Visible devices: {runtime_metadata['jax_devices']}"
            )
    elif backend_name == "cupy":
        import cupy as cp

        runtime_metadata["cupy_device_count"] = int(cp.cuda.runtime.getDeviceCount())
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        runtime_metadata["cupy_device_name"] = (
            device_name.decode() if isinstance(device_name, bytes) else str(device_name)
        )

    result: dict[str, Any] = {
        "metadata": runtime_metadata,
        "single_series": {},
        "batch3_serial_vs_vectorized": {},
    }

    for n in n_values:
        nt = _nt(n)
        x_np = _rng_signal(n, seed=SEED_SINGLE)

        ts = TimeSeries(x_np, dt=DT, backend=backend_name)
        wdm = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM, backend=backend_name)
        recovered = wdm.to_time_series()
        _synchronize((wdm.coeffs, recovered.data))

        forward_samples = _measure(
            lambda: WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM, backend=backend_name),
            runs,
        )

        wdm_for_inverse = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM, backend=backend_name)
        _synchronize(wdm_for_inverse.coeffs)
        inverse_samples = _measure(lambda: wdm_for_inverse.to_time_series(), runs)

        recovered_np = _to_numpy(recovered.data).squeeze(0)
        abs_error = np.abs(x_np - recovered_np)

        result["single_series"][str(n)] = {
            "nt": nt,
            "nf": n // nt,
            "forward": _timing_stats(forward_samples),
            "inverse": _timing_stats(inverse_samples),
            "accuracy": {
                "max_abs_error": float(np.max(abs_error)),
                "mean_abs_error": float(np.mean(abs_error)),
                "relative_l2_error": float(np.linalg.norm(abs_error) / np.linalg.norm(x_np)),
            },
        }

    for n in BATCH_N_VALUES:
        nt = _nt(n)
        batch_np = _rng_batch(n, batch_size=BATCH_SIZE, seed=SEED_BATCH)

        batch_series = TimeSeries(batch_np, dt=DT, backend=backend_name)

        vectorized_forward_samples = _measure(
            lambda: WDM.from_time_series(
                batch_series,
                nt=nt,
                a=A_PARAM,
                d=D_PARAM,
                backend=backend_name,
            ),
            runs,
        )

        vectorized_wdm = WDM.from_time_series(
            batch_series,
            nt=nt,
            a=A_PARAM,
            d=D_PARAM,
            backend=backend_name,
        )
        _synchronize(vectorized_wdm.coeffs)
        vectorized_inverse_samples = _measure(lambda: vectorized_wdm.to_time_series(), runs)

        def _serial_forward() -> list[Any]:
            return [
                WDM.from_time_series(
                    TimeSeries(batch_np[index], dt=DT, backend=backend_name),
                    nt=nt,
                    a=A_PARAM,
                    d=D_PARAM,
                    backend=backend_name,
                )
                for index in range(BATCH_SIZE)
            ]

        serial_wdms = _serial_forward()
        _synchronize([item.coeffs for item in serial_wdms])

        serial_forward_samples = _measure(_serial_forward, runs)
        serial_inverse_samples = _measure(
            lambda: [item.to_time_series() for item in serial_wdms],
            runs,
        )

        vectorized_coeffs = _to_numpy(vectorized_wdm.coeffs)
        serial_coeffs = np.stack([_to_numpy(item.coeffs[0]) for item in serial_wdms], axis=0)
        vectorized_recovered = _to_numpy(vectorized_wdm.to_time_series().data)
        serial_recovered = np.stack(
            [_to_numpy(item.to_time_series().data[0]) for item in serial_wdms],
            axis=0,
        )

        coeff_diff = np.abs(vectorized_coeffs - serial_coeffs)
        recovered_diff = np.abs(vectorized_recovered - serial_recovered)

        result["batch3_serial_vs_vectorized"][str(n)] = {
            "nt": nt,
            "nf": n // nt,
            "vectorized_forward": _timing_stats(vectorized_forward_samples),
            "serial_forward": _timing_stats(serial_forward_samples),
            "vectorized_inverse": _timing_stats(vectorized_inverse_samples),
            "serial_inverse": _timing_stats(serial_inverse_samples),
            "speedup_forward_serial_over_vectorized": float(
                np.mean(serial_forward_samples) / np.mean(vectorized_forward_samples)
            ),
            "speedup_inverse_serial_over_vectorized": float(
                np.mean(serial_inverse_samples) / np.mean(vectorized_inverse_samples)
            ),
            "accuracy": {
                "max_abs_coeff_diff": float(np.max(coeff_diff)),
                "mean_abs_coeff_diff": float(np.mean(coeff_diff)),
                "max_abs_reconstruction_diff": float(np.max(recovered_diff)),
                "mean_abs_reconstruction_diff": float(np.mean(recovered_diff)),
            },
        }

    return result


def _run_worker(
    backend_name: str,
    *,
    runs: int,
    jax_platform: str | None = None,
) -> None:
    result = _benchmark_single_backend(
        backend_name,
        runs=runs,
        n_values=N_VALUES,
        jax_platform=jax_platform,
    )
    print(json.dumps(result, indent=2))


def _write_backend_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_isolated_backend(
    backend_name: str,
    *,
    runs: int,
    jax_platform: str | None = None,
) -> dict[str, Any]:
    label = backend_name if jax_platform is None else f"{backend_name}_{jax_platform}"
    output_path = Path(f"colab_benchmark_results_{label}.json")

    env = os.environ.copy()
    if backend_name == "jax" and jax_platform is not None:
        env["JAX_PLATFORMS"] = jax_platform
        env["JAX_PLATFORM_NAME"] = jax_platform

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--backend",
        backend_name,
        "--runs",
        str(runs),
    ]
    if jax_platform is not None:
        command.extend(["--jax-platform", jax_platform])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        return {
            "available": False,
            "label": label,
            "error": completed.stderr.strip() or completed.stdout.strip() or "worker failed",
        }

    payload = json.loads(completed.stdout)
    _write_backend_result(output_path, payload)
    return {
        "available": True,
        "label": label,
        "output_file": str(output_path),
        "result": payload,
    }


def _summarize_backend(label: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    metadata = payload["metadata"]
    if label.startswith("jax"):
        print(f"backend: {metadata.get('jax_default_backend')}  devices: {metadata.get('jax_devices')}")
    if label == "cupy":
        print(f"device: {metadata.get('cupy_device_name')}")

    ref_n = str(N_VALUES[-1])
    if ref_n in payload["single_series"]:
        record = payload["single_series"][ref_n]
        print(
            "single-series "
            f"N={int(ref_n):,}: "
            f"fwd={record['forward']['mean_seconds'] * 1e3:.2f} ms, "
            f"inv={record['inverse']['mean_seconds'] * 1e3:.2f} ms, "
            f"err={record['accuracy']['max_abs_error']:.2e}"
        )

    batch_ref = str(BATCH_N_VALUES[-1])
    if batch_ref in payload["batch3_serial_vs_vectorized"]:
        record = payload["batch3_serial_vs_vectorized"][batch_ref]
        print(
            "batch-3 "
            f"N={int(batch_ref):,}: "
            f"serial/vectorized forward speedup={record['speedup_forward_serial_over_vectorized']:.2f}x, "
            f"inverse speedup={record['speedup_inverse_serial_over_vectorized']:.2f}x, "
            f"coeff diff={record['accuracy']['max_abs_coeff_diff']:.2e}"
        )


# %%
def run_notebook_mode(runs: int = NUM_RUNS) -> dict[str, Any]:
    print("Hardware:")
    for key, value in _hardware_fingerprint().items():
        print(f"  {key}: {value}")

    aggregate: dict[str, Any] = {
        "metadata": {
            "script": str(SCRIPT_PATH),
            "generated_at_unix": time.time(),
            "hardware": _hardware_fingerprint(),
            "runs": runs,
            "n_values": N_VALUES,
            "batch_n_values": BATCH_N_VALUES,
            "batch_size": BATCH_SIZE,
        },
        "results": {},
        "unavailable": {},
    }

    requested_jobs = [
        ("numpy", None),
        ("jax", "cpu"),
        ("jax", "gpu"),
        ("cupy", None),
    ]

    for backend_name, jax_platform in requested_jobs:
        label = backend_name if jax_platform is None else f"{backend_name}_{jax_platform}"
        print(f"\nRunning {label} benchmarks...")
        outcome = _run_isolated_backend(backend_name, runs=runs, jax_platform=jax_platform)
        if not outcome["available"]:
            aggregate["unavailable"][label] = {
                "error": outcome["error"],
            }
            print(f"  skipped: {outcome['error']}")
            continue

        payload = outcome["result"]
        aggregate["results"][label] = payload
        _summarize_backend(label, payload)
        print(f"  saved raw backend result to {outcome['output_file']}")

    RESULTS_PATH.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate report to {RESULTS_PATH.resolve()}")

    if shutil.which("python") is None:
        print("Note: current environment does not expose a `python` shim; raw files are still saved.")

    print("\nDownload from Colab with:")
    print("  from google.colab import files")
    print("  files.download('colab_benchmark_results.json')")

    return aggregate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run a single isolated backend worker.")
    parser.add_argument("--backend", choices=["numpy", "jax", "cupy"], help="Worker backend.")
    parser.add_argument("--jax-platform", choices=["cpu", "gpu"], help="Worker JAX platform.")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Timed runs per measurement.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        if args.backend is None:
            raise SystemExit("--backend is required in worker mode.")
        _run_worker(args.backend, runs=args.runs, jax_platform=args.jax_platform)
        return 0

    run_notebook_mode(runs=args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
elif "__IPYTHON__" in globals():
    run_notebook_mode()
