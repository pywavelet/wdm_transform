from __future__ import annotations

import json
from pathlib import Path

import pytest

from wdm_transform.benchmarking import (
    generate_benchmark_signal,
    generate_benchmark_spectrum,
    main,
    resolve_n_values,
    run_benchmarks,
)


def test_generate_benchmark_helpers_support_scalar_and_batch_shapes() -> None:
    signal = generate_benchmark_signal(16)
    signal_batch = generate_benchmark_signal(16, batch_size=3)
    spectrum = generate_benchmark_spectrum(16)
    spectrum_batch = generate_benchmark_spectrum(16, batch_size=3)

    assert signal.shape == (16,)
    assert signal_batch.shape == (3, 16)
    assert spectrum.shape == (16,)
    assert spectrum_batch.shape == (3, 16)


def test_run_benchmarks_records_only_core_kernel_operations() -> None:
    results = run_benchmarks(
        backends_to_test=["numpy"],
        n_values=[256],
        num_runs=1,
        batch_size=3,
    )

    assert results["metadata"]["batch_size"] == 3
    assert sorted(results["operations"]) == ["from_freq", "to_freq"]

    from_freq_record = results["operations"]["from_freq"]["results"]["numpy"][256]
    to_freq_record = results["operations"]["to_freq"]["results"]["numpy"][256]

    assert from_freq_record["scalar"]["shape"] == [256]
    assert from_freq_record["batch"]["shape"] == [3, 256]
    assert from_freq_record["serial"]["shape"] == [3, 256]
    assert "median_seconds" in from_freq_record["scalar"]
    assert "p16_seconds" in from_freq_record["scalar"]
    assert "p84_seconds" in from_freq_record["scalar"]
    assert "speedup_serial_over_batch" in from_freq_record
    assert "median" in from_freq_record["speedup_serial_over_batch"]
    assert "p16" in from_freq_record["speedup_serial_over_batch"]
    assert "p84" in from_freq_record["speedup_serial_over_batch"]

    assert to_freq_record["scalar"]["shape"][-1] > 0
    assert to_freq_record["batch"]["shape"][0] == 3
    assert to_freq_record["serial"]["shape"][0] == 3


def test_resolve_n_values_supports_pow2_ranges() -> None:
    assert resolve_n_values(pow2_range=(8, 9)) == [256, 512]
    assert resolve_n_values() == [2**power for power in range(11, 21)]


def test_resolve_n_values_validates_pow2_range() -> None:
    with pytest.raises(ValueError, match="within \\[8, 25\\]"):
        resolve_n_values(pow2_range=(7, 9))
    with pytest.raises(ValueError, match="within \\[8, 25\\]"):
        resolve_n_values(pow2_range=(8, 26))
    with pytest.raises(ValueError, match="start must be <= end"):
        resolve_n_values(pow2_range=(10, 8))


def test_benchmark_cli_writes_outdir_artifacts(tmp_path: Path) -> None:
    main(
        [
            "--backends",
            "numpy",
            "--pow2",
            "8",
            "8",
            "--runs",
            "1",
            "--outdir",
            str(tmp_path),
        ]
    )

    json_path = tmp_path / "benchmark_results.json"
    plot_path = tmp_path / "benchmark_runtime.png"
    assert json_path.exists()
    assert plot_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["metadata"]["available_backends"] == ["numpy"]
    assert data["metadata"]["n_values"] == [256]
    assert sorted(data["operations"]) == ["from_freq", "to_freq"]
