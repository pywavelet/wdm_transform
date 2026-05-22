"""Benchmark a single 1D FFT across libraries/devices at the WDM sizes.

Produces rows mirroring ``benchmark_data.csv`` for direct overlay on the
runtime figure. Run once per machine (CPU then GPU) and merge the
resulting ``benchmark_fft_data.csv`` files.

Usage:
    python docs/_static/bench_fft.py --device cpu
    python docs/_static/bench_fft.py --device gpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np

N_VALUES = [2**p for p in range(11, 21)]  # 2048 .. 1048576
DEFAULT_FIXED_NT = 1024
DEFAULT_NF_POW2_RANGE = (2, 10)
BATCH_SIZE = 3
NUM_RUNS = 7


def _resolve_n_values(
    *,
    fixed_nt: int | None,
    nf_pow2_range: tuple[int, int] | None,
) -> list[int]:
    if fixed_nt is None:
        return N_VALUES
    if fixed_nt < 2 or fixed_nt % 2:
        raise ValueError("--fixed-nt must be an even integer >= 2.")
    start, end = (
        nf_pow2_range if nf_pow2_range is not None else DEFAULT_NF_POW2_RANGE
    )
    if start < 1 or start > end:
        raise ValueError("--nf-pow2 must satisfy 1 <= START <= END.")
    return [fixed_nt * 2**power for power in range(start, end + 1)]


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        device = "gpu" if shutil.which("nvidia-smi") is not None else "cpu"
    else:
        device = requested
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if device == "gpu" else "cpu")
    return device


def _sync(value) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time(fn, num_runs: int) -> float:
    _sync(fn())  # warmup
    samples = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = fn()
        _sync(result)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def bench_numpy(n: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    scalar = rng.standard_normal(n)
    batch = rng.standard_normal((BATCH_SIZE, n))
    scalar_ms = _time(lambda: np.fft.fft(scalar), NUM_RUNS) * 1e3
    batch_ms = _time(lambda: np.fft.fft(batch, axis=-1), NUM_RUNS) * 1e3
    serial_ms = _time(
        lambda: [np.fft.fft(batch[i]) for i in range(BATCH_SIZE)], NUM_RUNS
    ) * 1e3
    return scalar_ms, batch_ms, serial_ms


def bench_jax(n: int) -> tuple[float, float, float]:
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    scalar = jnp.asarray(rng.standard_normal(n))
    batch = jnp.asarray(rng.standard_normal((BATCH_SIZE, n)))
    fft_scalar = jax.jit(jnp.fft.fft)
    fft_batch = jax.jit(lambda x: jnp.fft.fft(x, axis=-1))
    scalar_ms = _time(lambda: fft_scalar(scalar), NUM_RUNS) * 1e3
    batch_ms = _time(lambda: fft_batch(batch), NUM_RUNS) * 1e3
    serial_ms = _time(
        lambda: [fft_scalar(batch[i]) for i in range(BATCH_SIZE)], NUM_RUNS
    ) * 1e3
    return scalar_ms, batch_ms, serial_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument(
        "--square-tiling",
        action="store_true",
        help="Use the legacy N=2**p grid instead of the fixed-nt WDM grid.",
    )
    parser.add_argument(
        "--fixed-nt",
        type=int,
        default=DEFAULT_FIXED_NT,
        help=(
            "Use the fixed-nt WDM sweep grid N=nt*nf for matching FFT "
            f"overlays. Default: {DEFAULT_FIXED_NT}."
        ),
    )
    parser.add_argument(
        "--nf-pow2",
        nargs=2,
        metavar=("START", "END"),
        type=int,
        help="Inclusive exponent range for nf=2**p when --fixed-nt is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("benchmark_fft_data.csv"),
    )
    args = parser.parse_args()

    device = _resolve_device(args.device).upper()
    n_values = _resolve_n_values(
        fixed_nt=None if args.square_tiling else args.fixed_nt,
        nf_pow2_range=tuple(args.nf_pow2) if args.nf_pow2 is not None else None,
    )

    libraries = ["numpy", "jax"] if device == "CPU" else ["jax"]

    rows = []
    if args.output.exists():
        existing = args.output.read_text().strip().splitlines()
        header = existing[0]
        # keep rows for other devices
        for line in existing[1:]:
            if not line.startswith(f"{device},"):
                rows.append(line)
    else:
        header = "device,library,N,scalar_ms,batch_ms,serial_ms,speedup"

    print(f"Device: {device}")
    for library in libraries:
        print(f"\nLibrary: {library}")
        bench = bench_numpy if library == "numpy" else bench_jax
        for n in n_values:
            scalar_ms, batch_ms, serial_ms = bench(n)
            speedup = serial_ms / batch_ms if batch_ms else float("nan")
            print(
                f"  N={n:>8}: scalar={scalar_ms:>9.3f} ms | "
                f"batch={batch_ms:>9.3f} ms | serial={serial_ms:>9.3f} ms | "
                f"speedup={speedup:>5.2f}x"
            )
            rows.append(
                f"{device},{library},{n},{scalar_ms:.3f},{batch_ms:.3f},"
                f"{serial_ms:.3f},{speedup:.2f}"
            )

    args.output.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nSaved FFT benchmark to {args.output}")


if __name__ == "__main__":
    main()
