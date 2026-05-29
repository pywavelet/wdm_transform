# Reconstruction And Inference

WDM is useful only if it preserves the underlying signal content cleanly
enough to reconstruct and analyse it. This package supports both, with
exact closed-form inverses and a Whittle-style log-likelihood that lives
naturally on the WDM grid.

## Reconstruction Paths

Starting from a `TimeSeries`, you can move through the WDM domain and back:

```python
import numpy as np
from wdm_transform import TimeSeries

dt   = 1.0 / 4096
data = np.random.default_rng(0).standard_normal(2**12)

series = TimeSeries(data, dt=dt)
coeffs = series.to_wdm(nt=32)

recovered_time = coeffs.to_time_series()
recovered_freq = coeffs.to_frequency_series()
```

These two inverse paths answer slightly different questions:

- `to_time_series()`: do the coefficients preserve the original waveform?
- `to_frequency_series()`: do they preserve the original spectrum?

Both reconstruction errors are at the level of floating-point roundoff
($\epsilon_{\rm rt} \lesssim 10^{-16}$ for double precision, across all
backends and sizes).

## The Explicit Inverse

The package implements the inverse WDM transform in closed form. In the
frequency domain,

$$
\tilde x[\ell] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f} \tilde g_{nm}[\ell]\, w_{nm}
$$

with channel-wise expressions for the three cases — DC ($m=0$), interior
($0 < m < N_f$) and Nyquist ($m = N_f$) — derived from the discrete
orthogonality relation

$$
\sum_{n=0}^{N_t - 1}\sum_{m=0}^{N_f}
   \tilde g_{nm}^{*}[\ell]\, \tilde g_{nm}[\ell'] = \delta_{\ell\ell'} .
$$

The interior atoms are exactly orthonormal; the DC and Nyquist channels
are overcomplete by $N_t$ real numbers each, but enter the inverse with
the same closed form. This is the first explicit statement of the inverse
WDM transform we are aware of in the literature.

## Why Work In WDM Space At All?

If your signal is localised in both time and frequency, WDM can be a more
natural analysis space than either raw time samples or a global FFT.
Typical motivations are:

- identifying localised narrow-band features
- separating signal-rich and noise-rich regions of the grid
- building likelihoods that naturally accommodate **non-stationary**
  noise, which is awkward to express in the frequency domain alone

## The WDM-Whittle Log-Likelihood

For a Gaussian time-domain noise process with covariance $\mathbf{C}$ the
log-likelihood of data $\mathbf{d}$ given a signal model $\mathbf{h}$ is

$$
\ln p(\mathbf{d}\,|\,\mathbf{h}) =
-\tfrac{1}{2}(\mathbf{d}-\mathbf{h})^{\!\top}\mathbf{C}^{-1}(\mathbf{d}-\mathbf{h})
-\tfrac{1}{2}\ln \det(2\pi\mathbf{C}) .
$$

For stationary noise this reduces in the FFT domain to the familiar
Whittle likelihood — a single sum over frequency bins weighted by the
power spectral density (PSD) $S(f)$. The diagonal form follows because
$\mathbf{C}$ is approximately circulant and the FFT diagonalises
circulant matrices.

### Locally Stationary Noise

The WDM transform plays the same role for **locally stationary** noise.
If the noise PSD $S_n(t, f)$ varies slowly compared to a WDM cell of size
$\Delta T \times \Delta F$, the WDM noise covariance is approximately
diagonal:

$$
\langle w^{\rm noise}_{nm}\, w^{\rm noise}_{pq} \rangle
   \approx S_{nm}\, \delta_{np}\, \delta_{mq},
\qquad
S_{nm} \approx S(t_n, f_m)\, \Delta F
$$

and the log-likelihood reduces to a pixel-wise sum over the WDM grid:

$$
\ln p(\mathbf{d}\,|\,\boldsymbol{\theta}) =
-\tfrac{1}{2} \sum_{n, m}
\left[\ln(2\pi S_{nm}) + \frac{(d_{nm} - h_{nm}(\boldsymbol\theta))^{2}}{S_{nm}}\right] .
$$

This is structurally identical to the frequency-domain Whittle likelihood,
but the one-dimensional sum over frequency bins is replaced by a
two-dimensional sum over time-frequency pixels. The key advantage is that
$S_{nm}$ can vary with time — non-stationarity enters as a smoothly
varying surface, not as off-diagonal covariance terms.

### When Does The Diagonal Approximation Hold?

The diagonal covariance is exact for white noise, an excellent
approximation for coloured *stationary* noise (off-diagonal correlations
typically below a few percent), and a controlled approximation for
locally stationary noise as long as:

- the PSD varies slowly compared to the cell width $\Delta T$ in time
- spectral features are wider than $\Delta F$ in frequency
- the noise is approximately Gaussian

If those conditions are violated — sharp spectral lines narrower than
$\Delta F$, or rapid temporal variation — the off-diagonal entries become
non-negligible and the diagonal Whittle form is no longer a safe
approximation. A practical diagnostic is to whiten the data and check
that the empirical WDM coefficient correlation matrix is approximately
diagonal.

## Worked Example

A worked end-to-end example reproducing a LISA-like galactic-binary
posterior using the WDM-Whittle likelihood lives in the executed
walkthrough:

- [WDM Walkthrough](../examples/wdm_walkthrough.py)

Under a shared stationary noise model the WDM and frequency-domain
posteriors agree to within sub-percent Jensen–Shannon divergence on the
1D marginals, which is the controlled-comparison check that the WDM basis
change is not biasing inference.
