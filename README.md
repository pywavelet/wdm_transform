# wdm-transform


[![PyPI](https://img.shields.io/pypi/v/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Python](https://img.shields.io/pypi/pyversions/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Tests](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml/badge.svg)](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml)
[![Docs](https://github.com/pywavelet/wdm_transform/actions/workflows/docs.yml/badge.svg)](https://pywavelet.github.io/wdm_transform/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.20269-b31b1b.svg)](https://arxiv.org/abs/2606.20269)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/examples/wdm_walkthrough.ipynb)

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
# make a new venv for the repo
uv venv

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
uv run wdm_transform_benchmarking --backends numpy jax --runs 3 --outdir /tmp/wdm-bench --pow2 12 22

# refresh the checked-in benchmark snapshot used in the docs
uv run python docs/examples/generate_benchmark_plot.py --backends numpy jax
```

Documentation and source live at:

- <https://github.com/pywavelet/wdm_transform>
- <https://github.com/pywavelet/wdm_transform/tree/main/docs>
