# wdm-transform

`wdm-transform` provides a small object model for moving between sampled time-domain data,
frequency-domain data, and WDM coefficients.

![wdm-transform demo](_static/demo.gif)

## Core Objects

- `TimeSeries`: sampled one-dimensional time-domain data
- `FrequencySeries`: FFT-domain data with frequency spacing metadata
- `WDM`: real-valued WDM coefficients plus inverse transforms

## Docs Structure

### Learn

Start here if you want the transform intuition before looking at code:

- [What Is WDM?](guide/what-is-wdm.md)
- [Windows And Atoms](guide/windows-and-atoms.md)
- [Reconstruction And Inference](guide/reconstruction-and-inference.md)

### Tutorial

Use the executed walkthrough to see the package API and plots in one place:

- [WDM Walkthrough](examples/wdm_walkthrough.py)

### Reference

Use these when you want implementation-oriented details:

- [API Overview](guide/api-overview.md)
- [Package Layout](guide/package-layout.md)
- [Benchmarks](benchmarks.md)
