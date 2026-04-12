from __future__ import annotations

import numpy as np

from wdm_transform import benchmarking


def test_generate_benchmark_signal_supports_batch_shape() -> None:
    signal = benchmarking.generate_benchmark_signal(16)
    batch = benchmarking.generate_benchmark_signal(16, batch_size=3)

    assert signal.shape == (1, 16)
    assert batch.shape == (3, 16)


def test_run_benchmarks_records_batch_forward_comparison(monkeypatch) -> None:
    monkeypatch.setattr(benchmarking, "validate_backend_available", lambda backend_name: backend_name == "jax")
    monkeypatch.setattr(benchmarking, "benchmark_forward", lambda *args, **kwargs: (1.0, 0.1))
    monkeypatch.setattr(benchmarking, "benchmark_inverse", lambda *args, **kwargs: (2.0, 0.2))
    monkeypatch.setattr(
        benchmarking,
        "benchmark_roundtrip_error",
        lambda signal, backend_name, nt, nf: {
            "max_abs_error": 1e-12,
            "relative_l2_error": 1e-13,
            "nt": nt,
            "nf": nf,
        },
    )
    monkeypatch.setattr(benchmarking, "benchmark_forward_serial_batch", lambda *args, **kwargs: (3.0, 0.3))
    monkeypatch.setattr(benchmarking, "benchmark_forward_batched", lambda *args, **kwargs: (2.0, 0.2))

    results = benchmarking.run_benchmarks(["jax"], [1024], num_runs=2, batch_size=3)

    assert results["metadata"]["batch_size"] == 3
    assert results["forward"]["jax"][1024]["mean_seconds"] == 1.0
    record = results["batch_forward"]["jax"][1024]
    assert record["single_mean_seconds"] == 1.0
    assert record["serial_mean_seconds"] == 3.0
    assert record["batched_mean_seconds"] == 2.0
    assert record["batched_vs_serial_speedup"] == 1.5
    assert record["batch_size"] == 3
