# wdm-transform

`wdm-transform` is a small object-oriented package for Wilson-Daubechies-Meyer transforms on
sampled one-dimensional signals.

![wdm-transform demo](docs/_static/demo.gif)

The public API centers on three objects:

- `TimeSeries`
- `FrequencySeries`
- `WDM`

## Installation

```bash
pip install wdm-transform
```

Optional JAX support:

```bash
pip install "wdm-transform[jax]"
```

Documentation lives in `docs/`, including the API reference and the walkthrough example.
