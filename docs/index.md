# wdm-transform

[![PyPI](https://img.shields.io/pypi/v/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Python](https://img.shields.io/pypi/pyversions/wdm-transform.svg)](https://pypi.org/project/wdm-transform/)
[![Tests](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml/badge.svg)](https://github.com/pywavelet/wdm_transform/actions/workflows/tests.yml)
[![Docs](https://github.com/pywavelet/wdm_transform/actions/workflows/docs.yml/badge.svg)](https://pywavelet.github.io/wdm_transform/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.20269-b31b1b.svg)](https://arxiv.org/abs/2606.20269)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/examples/wdm_walkthrough.ipynb)

**A NumPy / JAX implementation of the Wilson–Daubechies–Meyer (WDM)
wavelet-packet time-frequency transform, with exact closed-form forward
and inverse, GPU-accelerated execution, and a Whittle-style likelihood
for locally stationary inference.**

![wdm-transform demo](_static/demo.gif)

---

## Install

```bash
pip install wdm-transform           # core (NumPy)
pip install wdm-transform[jax]      # CPU/GPU JAX backend
```

## Quickstart

```python
import numpy as np
from wdm_transform import TimeSeries

# Any 1D real signal with N = nt * nf samples (nt, nf both even)
N, dt = 2**14, 1.0 / 4096
t = np.arange(N) * dt
x = np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(N)

series = TimeSeries(x, dt=dt)
wdm    = series.to_wdm(nt=128)           # wdm.coeffs has shape (1, 128, 129)
recon  = wdm.to_time_series()            # round-trips to machine precision

assert np.allclose(recon.data, x, atol=1e-10)
```

The coefficient array `wdm.coeffs` has shape $(\text{batch},\, N_t,\, N_f + 1)$ — a real
time-frequency map that you can plot, slice, or feed straight into a
WDM-Whittle likelihood.

## What Makes WDM Useful

- **Exactly orthonormal interior atoms** — coefficients can be read
  independently, with no leakage from neighbours.
- **Closed-form inverse** to floating-point precision across NumPy and
  JAX backends.
- **JAX/GPU backend** — ~35× faster than NumPy at $N = 10^6$, suitable
  for differentiable inference pipelines.
- **WDM-Whittle likelihood** — a pixel-wise Gaussian likelihood that
  naturally accommodates locally stationary noise without off-diagonal
  covariance terms.

## Where To Go Next

<div class="grid cards" markdown>

- :material-school: **Learn the transform**

    Start with the conceptual pages before reading code.

    [What Is WDM?](guide/what-is-wdm.md) ·
    [Windows And Atoms](guide/windows-and-atoms.md) ·
    [Reconstruction And Inference](guide/reconstruction-and-inference.md)

- :material-notebook-outline: **Run the tutorial**

    Executed walkthrough with plots and timing numbers.
    Open in Colab in one click.

    [WDM Walkthrough](examples/wdm_walkthrough.py)

- :material-api: **Reference**

    Public API, package layout, and benchmark results.

    [API Overview](guide/api-overview.md) ·
    [Package Layout](guide/package-layout.md) ·
    [Benchmarks](benchmarks.ipynb)

- :material-github: **Source code**

    Issue tracker, contributing, and release notes.

    [github.com/pywavelet/wdm_transform](https://github.com/pywavelet/wdm_transform)

</div>

## Citation

If you use `wdm-transform` in academic work, please cite the companion
paper (Baghi et al., in preparation) and the package itself. A
machine-readable citation file is provided in the repository.
