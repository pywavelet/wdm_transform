# wdm-transform

`wdm-transform` is a small object-oriented package for Wilson-Daubechies-Meyer transforms on
sampled one-dimensional signals.

![wdm-transform demo](https://raw.githubusercontent.com/pywavelet/wdm_transform/main/docs/_static/demo.gif)

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

If you prefer `uv`:

```bash
uv add wdm-transform
```

Useful repository commands from the project root:

```bash
# install local dev dependencies
uv sync --extra dev

# include docs dependencies too
uv sync --extra dev --extra docs

# run the walkthrough example
uv run python docs/examples/wdm_walkthrough.py

# run the test suite
uv run pytest

# build the docs
uv run mkdocs build

# serve the docs locally
uv run mkdocs serve

# run the benchmark CLI
uv run python -m wdm_transform.benchmarking --backends numpy jax

# refresh the checked-in benchmark snapshot used in the docs
uv run python docs/examples/generate_benchmark_plot.py --backends numpy jax
```

Documentation and source live at:

- <https://github.com/pywavelet/wdm_transform>
- <https://github.com/pywavelet/wdm_transform/tree/main/docs>
