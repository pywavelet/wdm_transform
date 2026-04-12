# Benchmarks

This page shows a checked-in benchmark snapshot for the `numpy` and `jax` backends.
The artifacts are generated manually so routine docs builds stay fast and deterministic.
The figure includes forward runtime, inverse runtime, a 3-channel JAX batch-vs-serial
comparison, and round-trip reconstruction error against the original real-valued input signal.

![WDM benchmark runtimes](_static/benchmark_runtime.png)

The raw numbers used for the plot are available in
[`benchmark_results.json`](_static/benchmark_results.json).

## Refreshing The Benchmark Snapshot

Regenerate the plot and JSON artifact manually from the repository root:

```bash
uv run python docs/examples/generate_benchmark_plot.py --backends numpy jax
```

If `jax` is not installed in the active environment, the script will warn and only emit the
available backends.

## Notes

- The default benchmark range covers `N = 2048` through `33554432` and uses 7 timed runs per point.
- Refreshing the full default snapshot is now substantially more expensive than before.
- Each measurement uses one warmup call before timed runs.
- JAX timings are synchronized before the timer stops, so they include the actual device work.
- The timing panels show mean runtimes in milliseconds, with a shaded band for one standard deviation.
- The batched JAX panel compares one `3 x N` transform against three independent single-series transforms.
- Batched JAX execution is mainly a throughput optimization. It often wins for small and medium inputs because dispatch and kernel-launch overheads are amortized across the batch.
- At very large `N`, FFT cost and memory traffic dominate. In that regime, the larger batched working set can increase memory-bandwidth pressure and temporary-allocation costs, so the batched call may show much smaller speedups or even become slower than running the channels serially.
- The error panel shows the maximum absolute difference after `from_wdm_to_time(from_time_to_wdm(x))`.
