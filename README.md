# wdm-transform


[![PyPI](https://img.shields.io/pypi/v/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Python](https://img.shields.io/pypi/pyversions/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Tests](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml/badge.svg)](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml)
[![Docs](https://github.com/pywavelet/wdm_transform/actions/workflows/docs.yml/badge.svg)](https://pywavelet.github.io/wdm_transform/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.20269-b31b1b.svg)](https://arxiv.org/abs/2606.20269)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/examples/wdm_walkthrough.ipynb)

`wdm-transform` provides Wilson-Daubechies-Meyer (WDM) transforms and an opt-in
orthogonalized modified-Gaussian Wilson (WDG) window for sampled one-dimensional
signals. Both use the same package, objects, and coefficient layout.

![wdm-transform demo](https://raw.githubusercontent.com/pywavelet/wdm_transform/main/docs/_static/demo.gif)

The public API centers on three objects:

- `TimeSeries`
- `FrequencySeries`
- `WDM`

## Choose a window

Use Meyer for compact frequency support and the lowest transform cost; try
Gaussian when tighter time localization matters for transients or data gaps.
Gaussian trades that localization for wider frequency mixing. See
[Choosing a window](https://pywavelet.github.io/wdm_transform/guide/choosing-a-window/)
for the comparison figure, parameter guidance and Noah Pearson et al.'s paper.

```python
import numpy as np
from wdm_transform import TimeSeries

data = np.random.default_rng(42).normal(size=8192)
ts = TimeSeries(data, dt=5.0)
meyer = ts.to_wdm(nt=256, window="meyer")
gaussian = ts.to_wdm(nt=256, window="gaussian")
recovered = gaussian.to_time_series()  # retains the originating window
```

Omitting `window` keeps the original Meyer defaults, `a=1/3, d=1`. Gaussian
uses `nu=0.5` and `support=min(16,nf)` by default; support counts half-band
spacings retained on each side of a frequency-window center. It controls
finite-tail reconstruction error, rather than the number of signal samples.
The currently validated shape range is `0.3 <= nu <= 0.8`; small `nf` can
limit accuracy. Set `nu` and `support` explicitly when needed.

Both results have real coefficients shaped `(batch, nt, nf+1)`. `WDM` records
`window`, `nu`, and effective `support` as keyword-only metadata and passes
them to every inverse. Changing basis requires a new transform; changing
metadata does not convert coefficients. Gaussian rejects nondefault `a`/`d`,
and Meyer rejects explicit `nu`/`support`.

Gaussian full transforms support NumPy, JAX, and the shared CuPy implementation;
GPU correctness requires a CuPy-capable machine. Compact subband APIs accept
the same window options; Gaussian uses wider frequency margins. The existing
`wdm_noise_variance` channel-PSD approximation is not a calibrated WDG
colored-noise covariance model. The runnable
`docs/examples/gaussian_comparison.py` example generates the guide's figure.

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
uv run wdm_transform_benchmarking --backends numpy jax --pow2 12 22

# compare window families with synchronized timings and reconstruction errors
uv run wdm_transform_window_benchmark --backends numpy jax --runs 7 --outdir test_output/wdg_benchmarks

# refresh the checked-in benchmark snapshot used in the docs
uv run python docs/examples/generate_benchmark_plot.py --backends numpy jax
```

Documentation and source live at:

- <https://github.com/pywavelet/wdm_transform>
- <https://github.com/pywavelet/wdm_transform/tree/main/docs>
