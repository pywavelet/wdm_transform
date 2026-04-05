# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---
# ruff: noqa: E402, I001

# %% [markdown]
# # Time-Varying WDM PSD
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pywavelet/wdm_transform/blob/main/docs/studies/wdm_time_varying_psd.py)
#
# This study adapts the "moving PSD + spline surface" idea to the WDM domain.
#
# Notation used throughout:
#
# - `n` indexes WDM time bins
# - `m` indexes WDM frequency channels
# - `w_{nm}` is an observed WDM coefficient
# - `S_{nm}` is the latent local power to be inferred
# - `u_n \in [0,1]` is the rescaled WDM time coordinate
# - `\nu_m` is the WDM frequency coordinate
#
# The key substitution is simple:
#
# - in the Fourier workflow, the noisy local power observation is a moving periodogram
# - in the WDM workflow, the noisy local power observation is a local average of
#   squared WDM coefficients
#
# Under the approximate diagonal WDM likelihood from the manuscript,
# each WDM pixel behaves like
#
# $$
# w_{nm} \sim \mathcal{N}(0, S_{nm}),
# $$
#
# where `S_nm` is the local evolutionary power on the WDM grid. That means a
# locally averaged version of `w_nm**2` plays the same role here that a smoothed
# local periodogram does in the short-time Fourier picture.
#
# In this notebook we:
#
# - simulate one locally stationary time series
# - transform it to WDM coefficients with the package API
# - fit a smooth log-power surface with tensor-product B-splines
# - use a lightweight NumPyro MCMC to quantify posterior uncertainty
# - compare the inferred surface to a Monte Carlo reference built from many draws
#
# The point is not to claim that this is the final word on WDM PSD estimation.
# The point is to make the Bayesian Whittle construction explicit, inspect where
# it works, and identify which ingredients matter most: the WDM tiling, the
# spline prior, and the roughness penalty.

# %% [markdown]
# ## References
#
# - Piepho, H.-P., Boer, M. P., & Williams, E. R. (2022).
#   [Two-dimensional P-spline smoothing for spatial analysis of plant breeding trials](https://doi.org/10.1002/bimj.202100212).
#   *Biometrical Journal*, 64, 835–857. *(tensor-product spline surfaces on rectangular grids)*
# - Tang, Y., Kirch, C., Lee, J. E., & Meyer, R. (2026).
#   [Bayesian nonparametric spectral analysis of locally stationary processes](https://doi.org/10.1080/01621459.2025.2594191).
#   *JASA*. *(same broad target; STFT framework — useful benchmark for the chi-squared DOF comparison)*
# - Bach, P., & Klein, N. (2025).
#   [Anisotropic multidimensional smoothing using Bayesian tensor product P-splines](https://doi.org/10.1007/s11222-025-10569-y).
#   *Statistics and Computing*, 35, 43. *(Bayesian anisotropic penalties, pseudo-determinant terms)*
# - Lim, S., Pyeon, S., & Jeong, S. (2025).
#   [Penalty-Induced Basis Exploration for Bayesian Splines](https://arxiv.org/abs/2311.13481).
#   *(changing the roughness operator matters more than adding knots)*

# %%
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if "google.colab" in sys.modules:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "jax[cpu]>=0.4.30",
            "numpyro>=0.15",
            "ipywidgets>=8.1",
            "git+https://github.com/pywavelet/wdm_transform.git",
        ],
        check=True,
    )

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy import interpolate, stats
from scipy.ndimage import gaussian_filter1d, uniform_filter

from wdm_transform import TimeSeries


if "__file__" in globals():
    NOTEBOOK_DIR = Path(__file__).resolve().parent
else:
    cwd = Path.cwd()
    docs_studies_dir = cwd / "docs" / "studies"
    NOTEBOOK_DIR = docs_studies_dir if docs_studies_dir.exists() else cwd

FIGURE_OUTPUT_DIR = NOTEBOOK_DIR / "wdm_time_varying_psd_assets"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 160) -> Path:
    """Save a notebook figure to the docs static directory and close it."""
    path = FIGURE_OUTPUT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


@dataclass
class PSplineConfig:
    """Configuration for the smooth WDM log-power surface."""

    n_interior_knots_time: int = 8
    n_interior_knots_freq: int = 10
    degree_time: int = 3
    degree_freq: int = 3
    diff_order_time: int = 2
    diff_order_freq: int = 2
    alpha_phi: float = 2.0
    beta_phi: float = 1.0
    ridge_eps: float = 1e-6
    init_penalty_time: float = 5e-2
    init_penalty_freq: float = 5e-2
    weak_weight_scale: float = 5.0
    trim_time_bins: int = 1
    trim_low_freq_channels: int = 1
    trim_high_freq_channels: int = 1
    adaptive_time_knots: bool = True
    adaptive_time_knot_smoothing: float = 1.0
    adaptive_time_knot_floor: float = 0.25
    periodogram_freq_half_width: int = 0
    periodogram_time_half_width: int = 0


def simulate_tv_arma(
    n: int,
    *,
    dgp: str = "LS2",
    innovation: str = "a",
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the locally stationary ARMA examples used in the PSD demo."""
    if innovation == "a":
        w = rng.normal(0.0, 1.0, n + 2)
    elif innovation == "b":
        w = rng.standard_t(3.0, n + 2) / np.sqrt(3.0)
    elif innovation == "c":
        w = (rng.pareto(4.0, n + 2) - 4.0 / 3.0) / np.sqrt(2.0 / 9.0)
    else:
        raise ValueError(f"Unknown innovation model: {innovation}")

    data = np.zeros(n)

    if dgp == "LS1":
        for t in range(n):
            u = t / n
            b1 = 1.122 * (1.0 - 1.718 * np.sin(np.pi * u / 2.0))
            b2 = -0.81
            data[t] = w[t + 2] + b1 * w[t + 1] + b2 * w[t]
    elif dgp == "LS2":
        for t in range(n):
            u = t / n
            b1 = 1.1 * np.cos(1.5 - np.cos(4.0 * np.pi * u))
            data[t] = w[t + 1] + b1 * w[t]
    elif dgp == "LS3":
        data[0] = rng.normal()
        for t in range(1, n):
            u = t / n
            a1 = 1.2 * u - 0.6
            data[t] = a1 * data[t - 1] + w[t]
    else:
        raise ValueError(f"Unknown DGP: {dgp}")

    return data


def compute_true_tv_psd(
    dgp: str,
    time_grid: np.ndarray,
    angular_freq_grid: np.ndarray,
) -> np.ndarray:
    """Compute the analytical pointwise TV-PSD for the simulated DGPs.

    Parameters
    ----------
    dgp
        Data generating process ('LS1', 'LS2', or 'LS3').
    time_grid
        Rescaled time grid in [0, 1].
    angular_freq_grid
        Discrete-time angular frequencies in [0, pi].

    Notes
    -----
    Returns the DTFT PSD in the Oppenheim-Schafer convention:

        S(e^{jω}) = |MA(e^{jω})|² / |AR(e^{jω})|²  (with unit innovation variance)

    so that  σ²_x = (1/2π) ∫_{-π}^{π} S(e^{jω}) dω.

    This is the same scale as the WDM squared-coefficient variance:
    E[w_{nm}²] ≈ S(e^{jω_m})  for a locally stationary process with slowly
    varying PSD.  In particular, for unit-variance white noise both sides equal 1.

    The angular frequencies should be digital (ω = 2π f dt ∈ [0, π]).  Pass
    ``2 * np.pi * dt * freq_grid_hz`` from the WDM output.
    """
    n_time = len(time_grid)
    n_freq = len(angular_freq_grid)
    tv_psd = np.zeros((n_time, n_freq))

    for i, u in enumerate(time_grid):
        if dgp == "LS1":
            a1 = 0.0
            b1 = 1.122 * (1.0 - 1.718 * np.sin(np.pi * u / 2.0))
            b2 = -0.81
        elif dgp == "LS2":
            a1 = 0.0
            b1 = 1.1 * np.cos(1.5 - np.cos(4.0 * np.pi * u))
            b2 = 0.0
        elif dgp == "LS3":
            a1 = 1.2 * u - 0.6
            b1 = 0.0
            b2 = 0.0
        else:
            raise ValueError(f"Unknown DGP: {dgp}")

        for j, omega in enumerate(angular_freq_grid):
            ar_part = 1.0 + a1**2 - 2.0 * a1 * np.cos(omega)
            ma_part = (
                1.0
                + b1**2
                + b2**2
                + 2.0 * b1 * (b2 + 1.0) * np.cos(omega)
                + 2.0 * b2 * np.cos(2.0 * omega)
            )
            tv_psd[i, j] = ma_part / ar_part

    return tv_psd


def chisq1_log_noise_envelope(
    true_psd: np.ndarray,
    *,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """90% envelope for log(w²) given the true E[w²] = S.

    Under the Whittle model w²/S ~ χ²₁.  On the log scale the noise is
    log-χ²₁, which has:

        5th  percentile ≈ -5.54 (i.e. w² can be 0.004 × S)
        95th percentile ≈ +1.35 (i.e. w² can be 3.84 × S)

    This asymmetric, heavy-left-tailed noise is the core reason that single-
    realization spectral estimates are hard: each squared coefficient is almost
    useless on its own — it spans a ~7 log-unit range at the 90% level.
    """
    from scipy.stats import chi2

    lo = chi2.ppf(lower_pct / 100.0, df=1)
    hi = chi2.ppf(upper_pct / 100.0, df=1)
    return true_psd * lo, true_psd * hi


# %% [markdown]
# ## Spline Surface And Roughness Prior
#
# We model the log local power with a tensor-product spline surface:
#
# $$
# \log S_{nm}
# =
# \sum_{r=1}^{R_t}\sum_{s=1}^{R_f}
# B_r^{(t)}(u_n)\,W_{rs}\,B_s^{(f)}(\nu_m),
# $$
#
# where `B_r^{(t)}` and `B_s^{(f)}` are B-spline basis functions in time and
# frequency, and `W_{rs}` are the unknown spline coefficients.
#
# The prior is built from derivative-based roughness matrices rather than simple
# coefficient differences. For example, in time we form
#
# $$
# R_t[i,j] = \int B_i^{(q_t)}(u)\,B_j^{(q_t)}(u)\,du,
# $$
#
# and similarly in frequency. The resulting anisotropic prior is
#
# $$
# p(W \mid \phi_t,\phi_f)
# \propto
# \exp\left[
# -\frac{\phi_t}{2}\operatorname{vec}(W)^\top(R_t \otimes I_f)\operatorname{vec}(W)
# -\frac{\phi_f}{2}\operatorname{vec}(W)^\top(I_t \otimes R_f)\operatorname{vec}(W)
# \right].
# $$
#
# This is closer to penalizing actual curvature of the latent surface in
# physical coordinates than penalizing nearest-neighbor coefficient differences.
#
# Relative to the references above:
#
# - the tensor-product construction follows the same general spirit as
#   Piepho et al. (2022) and Bach & Klein (2025)
# - the Kronecker-structured penalties line up with the computational viewpoint
#   highlighted in the pybaselines 2D Whittaker examples
# - the present notebook is simpler than Bach & Klein (2025): we use a direct
#   NumPyro implementation rather than their more fully developed Bayesian
#   anisotropic P-spline framework
#
# %%
def create_bspline_basis(
    x: np.ndarray,
    n_interior_knots: int,
    *,
    degree: int = 3,
    interior_knots: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a B-spline basis on the supplied grid."""
    x = np.asarray(x, dtype=float)
    if interior_knots is None:
        interior = np.linspace(x.min(), x.max(), n_interior_knots + 2)[1:-1]
    else:
        interior = np.asarray(interior_knots, dtype=float)
        interior = interior[(interior > x.min()) & (interior < x.max())]
        if len(interior) != n_interior_knots:
            raise ValueError("interior_knots must match n_interior_knots.")
    knots = np.concatenate(
        [
            np.repeat(x.min(), degree + 1),
            interior,
            np.repeat(x.max(), degree + 1),
        ]
    )
    n_basis = len(knots) - degree - 1
    basis = np.zeros((len(x), n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = interpolate.BSpline(knots, coeffs, degree, extrapolate=False)
        basis[:, i] = spline(x)

    basis = np.nan_to_num(basis)
    row_sums = basis.sum(axis=1, keepdims=True)
    basis /= np.maximum(row_sums, 1e-12)
    return basis, knots


def evaluate_bspline_basis(
    x: np.ndarray,
    knots: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Evaluate a B-spline basis defined by a full knot vector on a new grid."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    n_basis = len(knots) - degree - 1
    basis = np.zeros((len(x), n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = interpolate.BSpline(knots, coeffs, degree, extrapolate=False)
        basis[:, i] = spline(x)

    basis = np.nan_to_num(basis)
    row_sums = basis.sum(axis=1, keepdims=True)
    basis /= np.maximum(row_sums, 1e-12)
    return basis


def create_adaptive_time_knots(
    x: np.ndarray,
    pilot_profile: np.ndarray,
    *,
    n_interior_knots: int,
    smoothing_sigma: float = 1.0,
    variation_floor: float = 0.25,
) -> np.ndarray:
    """Place more time knots where a pilot time profile changes fastest."""
    x = np.asarray(x, dtype=float)
    pilot_profile = np.asarray(pilot_profile, dtype=float)

    if x.ndim != 1 or pilot_profile.ndim != 1 or len(x) != len(pilot_profile):
        raise ValueError("x and pilot_profile must be one-dimensional with matching length.")

    if n_interior_knots <= 0:
        return np.array([], dtype=float)

    smooth_profile = gaussian_filter1d(pilot_profile, sigma=smoothing_sigma, mode="nearest")
    local_variation = np.abs(np.gradient(smooth_profile, x))
    density = variation_floor + local_variation
    density = np.maximum(density, 1e-10)
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / np.maximum(cdf[-1] - cdf[0], 1e-12)
    targets = np.linspace(0.0, 1.0, n_interior_knots + 2)[1:-1]
    interior = np.interp(targets, cdf, x)

    # Guard against duplicate knots in flat regions.
    interior = np.maximum.accumulate(interior)
    eps = np.finfo(float).eps * max(1.0, x.max() - x.min()) * 32.0
    for i in range(1, len(interior)):
        if interior[i] <= interior[i - 1]:
            interior[i] = interior[i - 1] + eps

    interior = np.clip(interior, x.min() + eps, x.max() - eps)
    return interior


def create_bspline_roughness_penalty(
    knots: np.ndarray,
    *,
    degree: int,
    derivative_order: int = 2,
    quad_order: int = 8,
) -> np.ndarray:
    r"""Derivative-based B-spline roughness matrix.

    The matrix entries are

    .. math::

        R_{ij} = \int B_i^{(q)}(x) B_j^{(q)}(x)\,dx

    where ``q = derivative_order``. We evaluate the integral by Gauss-Legendre
    quadrature on each non-degenerate knot span.
    """
    if derivative_order > degree:
        raise ValueError("derivative_order must be <= degree.")

    n_basis = len(knots) - degree - 1
    coeffs = np.eye(n_basis)
    deriv_splines = [
        interpolate.BSpline(knots, coeffs[i], degree, extrapolate=False).derivative(
            derivative_order
        )
        for i in range(n_basis)
    ]

    penalty = np.zeros((n_basis, n_basis))
    abscissa, weights = np.polynomial.legendre.leggauss(quad_order)

    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        x_eval = midpoint + half_width * abscissa
        values = np.stack([spline(x_eval) for spline in deriv_splines], axis=0)
        values = np.nan_to_num(values)
        penalty += (values * weights[None, :]) @ values.T * half_width

    penalty = 0.5 * (penalty + penalty.T)
    return penalty / np.maximum(np.trace(penalty), 1e-12)


def create_difference_penalty_matrix(
    n_basis: int,
    *,
    diff_order: int = 2,
) -> np.ndarray:
    """Return the normalized finite-difference penalty matrix D^T D."""
    if n_basis <= diff_order:
        raise ValueError("Need more basis functions than the penalty order.")

    D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    penalty = D.T @ D
    return penalty / np.maximum(np.trace(penalty), 1e-12)


def compute_smoothed_periodogram(
    power: np.ndarray,
    *,
    freq_half_width: int = 1,
    time_half_width: int = 0,
) -> tuple[np.ndarray, int]:
    """Local average of squared WDM coefficients to build a periodogram with more DOF.

    Under the Whittle model each w_{nm}^2 / S_{nm} ~ chi-squared(1). Averaging K
    approximately independent such terms gives a Gamma(K/2, K/(2S)) observation
    with variance reduced by a factor of K.

    Averaging in frequency is preferred because the PSD is typically smoother in
    frequency than in time for locally stationary processes.
    """
    size = (2 * time_half_width + 1, 2 * freq_half_width + 1)
    smoothed = uniform_filter(power, size=size, mode="nearest")
    dof = (2 * time_half_width + 1) * (2 * freq_half_width + 1)
    return smoothed, dof


def create_kronecker_penalties(
    P_time: np.ndarray,
    P_freq: np.ndarray,
    *,
    ridge_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate time/frequency penalty terms plus a ridge-stabilized sum."""
    n_basis_time = P_time.shape[0]
    n_basis_freq = P_freq.shape[0]
    penalty_time = np.kron(np.eye(n_basis_freq), P_time)
    penalty_freq = np.kron(P_freq, np.eye(n_basis_time))
    penalty_sum = penalty_time + penalty_freq + ridge_eps * np.eye(
        penalty_time.shape[0]
    )
    return penalty_time, penalty_freq, penalty_sum


def initialize_with_penalized_least_squares(
    observed_power: np.ndarray,
    B_time: np.ndarray,
    B_freq: np.ndarray,
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    config: PSplineConfig,
) -> dict[str, np.ndarray | float]:
    """Build a stable MCMC start from a penalized least-squares fit."""
    floor = max(1e-8, 0.05 * np.percentile(observed_power, 10.0))
    target = np.log(observed_power + floor).reshape(-1, order="F")
    design = np.kron(B_freq, B_time)
    system = (
        design.T @ design
        + config.init_penalty_time * penalty_time
        + config.init_penalty_freq * penalty_freq
        + config.ridge_eps * np.eye(design.shape[1])
    )
    rhs = design.T @ target
    weights = np.linalg.solve(system, rhs)

    fitted = (design @ weights).reshape(observed_power.shape, order="F")
    penalty_time_energy = float(weights @ penalty_time @ weights)
    penalty_freq_energy = float(weights @ penalty_freq @ weights)
    phi_time_init = max(1e-2, fitted.size / (penalty_time_energy + 1e-6))
    phi_freq_init = max(1e-2, fitted.size / (penalty_freq_energy + 1e-6))

    return {
        "weights": weights,
        "phi_time": phi_time_init,
        "phi_freq": phi_freq_init,
        "log_psd": fitted,
    }


def pspline_wdm_model(
    coeffs: jnp.ndarray,
    B_time: jnp.ndarray,
    B_freq: jnp.ndarray,
    penalty_time: jnp.ndarray,
    penalty_freq: jnp.ndarray,
    penalty_time_rank: int,
    penalty_freq_rank: int,
    config: PSplineConfig,
) -> None:
    """Bayesian tensor-product spline model with a WDM Whittle likelihood."""
    n_basis_time = B_time.shape[1]
    n_basis_freq = B_freq.shape[1]
    n_weights = n_basis_time * n_basis_freq

    phi_time = numpyro.sample(
        "phi_time",
        dist.Gamma(config.alpha_phi, config.beta_phi),
    )
    phi_freq = numpyro.sample(
        "phi_freq",
        dist.Gamma(config.alpha_phi, config.beta_phi),
    )

    with numpyro.plate("weights_plate", n_weights):
        weights = numpyro.sample(
            "weights",
            dist.Normal(0.0, config.weak_weight_scale),
        )

    penalty_term = -0.5 * phi_time * jnp.dot(weights, penalty_time @ weights)
    penalty_term += 0.5 * penalty_time_rank * jnp.log(phi_time)
    penalty_term += -0.5 * phi_freq * jnp.dot(weights, penalty_freq @ weights)
    penalty_term += 0.5 * penalty_freq_rank * jnp.log(phi_freq)
    numpyro.factor("spline_penalty", penalty_term)

    W = weights.reshape((n_basis_time, n_basis_freq), order="F")
    log_psd = B_time @ W @ B_freq.T

    log_like = -0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi) + log_psd + coeffs**2 * jnp.exp(-log_psd)
    )
    numpyro.factor("wdm_whittle", log_like)
    numpyro.deterministic("log_psd", log_psd)


def pspline_wdm_gamma_model(
    periodogram: jnp.ndarray,
    dof: int,
    B_time: jnp.ndarray,
    B_freq: jnp.ndarray,
    penalty_time: jnp.ndarray,
    penalty_freq: jnp.ndarray,
    penalty_time_rank: int,
    penalty_freq_rank: int,
    config: PSplineConfig,
) -> None:
    """Gamma likelihood on a smoothed WDM periodogram.

    If the squared WDM coefficients are averaged over K ~ independent pixels,
    the resulting periodogram has an approximate Gamma(K/2, K/(2*S)) distribution
    instead of the chi-squared(1) that applies to a single w_{nm}^2.  The
    per-pixel log-likelihood curvature at the mode is -K/2 rather than -1/2,
    giving K times sharper identification of the latent surface.
    """
    n_basis_time = B_time.shape[1]
    n_basis_freq = B_freq.shape[1]
    n_weights = n_basis_time * n_basis_freq

    phi_time = numpyro.sample(
        "phi_time",
        dist.Gamma(config.alpha_phi, config.beta_phi),
    )
    phi_freq = numpyro.sample(
        "phi_freq",
        dist.Gamma(config.alpha_phi, config.beta_phi),
    )

    with numpyro.plate("weights_plate", n_weights):
        weights = numpyro.sample(
            "weights",
            dist.Normal(0.0, config.weak_weight_scale),
        )

    penalty_term = -0.5 * phi_time * jnp.dot(weights, penalty_time @ weights)
    penalty_term += 0.5 * penalty_time_rank * jnp.log(phi_time)
    penalty_term += -0.5 * phi_freq * jnp.dot(weights, penalty_freq @ weights)
    penalty_term += 0.5 * penalty_freq_rank * jnp.log(phi_freq)
    numpyro.factor("spline_penalty", penalty_term)

    W = weights.reshape((n_basis_time, n_basis_freq), order="F")
    log_psd = B_time @ W @ B_freq.T
    S = jnp.exp(log_psd)

    nu = dof
    concentration = nu / 2.0
    rate = nu / (2.0 * S)
    safe_periodogram = jnp.maximum(periodogram, 1e-30)
    log_like = jnp.sum(dist.Gamma(concentration, rate).log_prob(safe_periodogram))
    numpyro.factor("gamma_whittle", log_like)
    numpyro.deterministic("log_psd", log_psd)


def run_wdm_psd_mcmc(
    data: np.ndarray,
    *,
    dt: float,
    nt: int,
    config: PSplineConfig,
    n_warmup: int = 250,
    n_samples: int = 300,
    num_chains: int = 1,
    random_seed: int = 7,
) -> dict[str, np.ndarray]:
    """Fit a smooth WDM-domain evolutionary PSD surface to one realization."""
    series = TimeSeries(data, dt=dt)
    wdm = series.to_wdm(nt=nt)
    coeffs = np.asarray(wdm.coeffs)
    raw_power = coeffs**2

    keep_time = np.arange(config.trim_time_bins, wdm.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels,
        wdm.nf + 1 - config.trim_high_freq_channels,
    )
    coeffs_fit = coeffs[np.ix_(keep_time, keep_freq)]
    power = raw_power[np.ix_(keep_time, keep_freq)]

    time_grid = np.asarray(wdm.time_grid)[keep_time] / wdm.duration
    freq_grid = np.asarray(wdm.freq_grid)[keep_freq]
    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)

    time_interior_knots = None
    if config.adaptive_time_knots:
        pilot_time_profile = np.mean(np.log(power + 1e-8), axis=1)
        time_interior_knots = create_adaptive_time_knots(
            time_grid,
            pilot_time_profile,
            n_interior_knots=config.n_interior_knots_time,
            smoothing_sigma=config.adaptive_time_knot_smoothing,
            variation_floor=config.adaptive_time_knot_floor,
        )

    B_time, knots_time = create_bspline_basis(
        time_grid,
        config.n_interior_knots_time,
        degree=config.degree_time,
        interior_knots=time_interior_knots,
    )
    B_freq, knots_freq = create_bspline_basis(
        freq_unit,
        config.n_interior_knots_freq,
        degree=config.degree_freq,
    )

    P_time = create_bspline_roughness_penalty(
        knots_time,
        degree=config.degree_time,
        derivative_order=config.diff_order_time,
    )
    P_freq = create_bspline_roughness_penalty(
        knots_freq,
        degree=config.degree_freq,
        derivative_order=config.diff_order_freq,
    )
    penalty_time, penalty_freq, penalty_sum = create_kronecker_penalties(
        P_time,
        P_freq,
        ridge_eps=config.ridge_eps,
    )
    penalty_time_rank = int(np.linalg.matrix_rank(P_time, tol=1e-10)) * B_freq.shape[1]
    penalty_freq_rank = int(np.linalg.matrix_rank(P_freq, tol=1e-10)) * B_time.shape[1]

    use_gamma = (
        config.periodogram_freq_half_width > 0
        or config.periodogram_time_half_width > 0
    )
    if use_gamma:
        periodogram, dof = compute_smoothed_periodogram(
            power,
            freq_half_width=config.periodogram_freq_half_width,
            time_half_width=config.periodogram_time_half_width,
        )
    else:
        periodogram = None
        dof = 1

    init_vals = initialize_with_penalized_least_squares(
        periodogram if use_gamma else power,
        B_time,
        B_freq,
        penalty_time,
        penalty_freq,
        config,
    )

    if use_gamma:
        model = pspline_wdm_gamma_model
        model_args = (
            jnp.asarray(periodogram),
            dof,
            jnp.asarray(B_time),
            jnp.asarray(B_freq),
            jnp.asarray(penalty_time),
            jnp.asarray(penalty_freq),
            penalty_time_rank,
            penalty_freq_rank,
            config,
        )
    else:
        model = pspline_wdm_model
        model_args = (
            jnp.asarray(coeffs_fit),
            jnp.asarray(B_time),
            jnp.asarray(B_freq),
            jnp.asarray(penalty_time),
            jnp.asarray(penalty_freq),
            penalty_time_rank,
            penalty_freq_rank,
            config,
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            values={
                "weights": init_vals["weights"],
                "phi_time": init_vals["phi_time"],
                "phi_freq": init_vals["phi_freq"],
            }
        ),
        max_tree_depth=10,
        target_accept_prob=0.85,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(random_seed), *model_args)

    samples = mcmc.get_samples()
    log_psd_samples = np.asarray(samples["log_psd"])

    return {
        "wdm": wdm,
        "mcmc": mcmc,
        "coeffs": coeffs,
        "keep_time": keep_time,
        "keep_freq": keep_freq,
        "raw_power": raw_power,
        "coeffs_fit": coeffs_fit,
        "power": power,
        "periodogram": periodogram,
        "periodogram_dof": dof,
        "time_grid": time_grid,
        "freq_grid": freq_grid,
        "B_time": B_time,
        "B_freq": B_freq,
        "knots_time": knots_time,
        "knots_freq": knots_freq,
        "penalty_time": penalty_time,
        "penalty_freq": penalty_freq,
        "penalty_sum": penalty_sum,
        "init_vals": init_vals,
        "samples": {name: np.asarray(value) for name, value in samples.items()},
        "log_psd_mean": np.mean(log_psd_samples, axis=0),
        "log_psd_lower": np.percentile(log_psd_samples, 5.0, axis=0),
        "log_psd_upper": np.percentile(log_psd_samples, 95.0, axis=0),
        "psd_mean": np.exp(np.mean(log_psd_samples, axis=0)),
        "psd_lower": np.exp(np.percentile(log_psd_samples, 5.0, axis=0)),
        "psd_upper": np.exp(np.percentile(log_psd_samples, 95.0, axis=0)),
    }


def evaluate_dense_posterior_mean(
    results: dict[str, np.ndarray],
    config: PSplineConfig,
    *,
    n_time_dense: int = 200,
    n_freq_dense: int = 200,
) -> dict[str, np.ndarray]:
    """Evaluate the posterior mean spline surface on a dense plotting grid."""
    dense_time_grid = np.linspace(results["time_grid"][0], results["time_grid"][-1], n_time_dense)
    dense_freq_grid = np.linspace(results["freq_grid"][0], results["freq_grid"][-1], n_freq_dense)
    dense_freq_unit = dense_freq_grid / np.maximum(results["freq_grid"][-1], 1e-12)

    B_time_dense = evaluate_bspline_basis(
        dense_time_grid,
        results["knots_time"],
        degree=config.degree_time,
    )
    B_freq_dense = evaluate_bspline_basis(
        dense_freq_unit,
        results["knots_freq"],
        degree=config.degree_freq,
    )

    mean_weights = np.mean(results["samples"]["weights"], axis=0)
    W_mean = mean_weights.reshape(
        (results["B_time"].shape[1], results["B_freq"].shape[1]),
        order="F",
    )
    dense_log_psd = B_time_dense @ W_mean @ B_freq_dense.T

    return {
        "time_grid": dense_time_grid,
        "freq_grid": dense_freq_grid,
        "log_psd_mean": dense_log_psd,
        "psd_mean": np.exp(dense_log_psd),
    }


def summarize_mcmc_diagnostics(results: dict[str, np.ndarray]) -> dict[str, object]:
    """Compute compact convergence diagnostics for the Whittle fit."""
    mcmc = results["mcmc"]
    samples = mcmc.get_samples(group_by_chain=True)
    diag = summary(samples, group_by_chain=True)
    divergences = int(
        np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()
    )

    n_time, n_freq = results["psd_mean"].shape
    probe_points = [
        ("center", n_time // 2, n_freq // 2),
        ("low_freq", n_time // 2, max(1, n_freq // 5)),
        ("high_freq", n_time // 2, min(n_freq - 2, (4 * n_freq) // 5)),
    ]

    latent = {}
    for label, i, j in probe_points:
        site_diag = diag["log_psd"]
        latent[label] = {
            "index": (i, j),
            "mean": float(site_diag["mean"][i, j]),
            "n_eff": float(site_diag["n_eff"][i, j]),
            "r_hat": float(site_diag["r_hat"][i, j]),
        }

    return {
        "num_chains": int(samples["phi_time"].shape[0]),
        "divergences": divergences,
        "phi_time": {
            "mean": float(diag["phi_time"]["mean"]),
            "n_eff": float(diag["phi_time"]["n_eff"]),
            "r_hat": float(diag["phi_time"]["r_hat"]),
        },
        "phi_freq": {
            "mean": float(diag["phi_freq"]["mean"]),
            "n_eff": float(diag["phi_freq"]["n_eff"]),
            "r_hat": float(diag["phi_freq"]["r_hat"]),
        },
        "latent_log_psd": latent,
    }


def run_dof_sweep(
    data: np.ndarray,
    *,
    dt: float,
    nt: int,
    reference_psd: np.ndarray,
    freq_hw_values: list[int],
    base_config: PSplineConfig,
    n_warmup: int = 200,
    n_samples: int = 200,
    random_seed: int = 42,
) -> list[dict]:
    """Fit the model for different amounts of periodogram averaging (DOF sweep).

    This directly tests the core thesis: aggregating K squared WDM coefficients
    reduces the per-pixel CV from sqrt(2) toward 1, approaching the efficiency
    of a complex STFT periodogram.  The STFT Whittle literature (Dahlhaus 1997,
    Tang et al. 2026) achieves Exp(1) noise per pixel, equivalent to K=2 here.

    Args:
        data: One realization of the locally stationary process.
        dt: Sampling interval.
        nt: Number of WDM time bins.
        reference_psd: Monte Carlo estimate of E[w²] used as ground truth.
        freq_hw_values: List of frequency half-widths to sweep (K = 2*hw+1).
        base_config: Base PSplineConfig; periodogram_freq_half_width is overridden.
        n_warmup: NUTS warmup steps (lighter than the main run).
        n_samples: NUTS sample steps.
        random_seed: PRNG seed.
    """
    rows = []
    for freq_hw in freq_hw_values:
        cfg = PSplineConfig(
            n_interior_knots_time=base_config.n_interior_knots_time,
            n_interior_knots_freq=base_config.n_interior_knots_freq,
            degree_time=base_config.degree_time,
            degree_freq=base_config.degree_freq,
            diff_order_time=base_config.diff_order_time,
            diff_order_freq=base_config.diff_order_freq,
            alpha_phi=base_config.alpha_phi,
            beta_phi=base_config.beta_phi,
            ridge_eps=base_config.ridge_eps,
            init_penalty_time=base_config.init_penalty_time,
            init_penalty_freq=base_config.init_penalty_freq,
            weak_weight_scale=base_config.weak_weight_scale,
            trim_time_bins=base_config.trim_time_bins,
            trim_low_freq_channels=base_config.trim_low_freq_channels,
            trim_high_freq_channels=base_config.trim_high_freq_channels,
            adaptive_time_knots=base_config.adaptive_time_knots,
            periodogram_freq_half_width=freq_hw,
            periodogram_time_half_width=0,
        )
        res = run_wdm_psd_mcmc(
            data,
            dt=dt,
            nt=nt,
            config=cfg,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=1,
            random_seed=random_seed,
        )
        dof = res["periodogram_dof"]
        rows.append(
            {
                "freq_hw": freq_hw,
                "dof": dof,
                "post_error": relative_surface_error(reference_psd, res["psd_mean"]),
                "raw_error": relative_surface_error(reference_psd, res["power"]),
                "psd_mean": res["psd_mean"],
            }
        )
    return rows


def run_nt_sweep(
    *,
    n_total: int,
    dt: float,
    dgp: str,
    nt_values: list[int],
    base_config: PSplineConfig,
    random_seed: int = 42,
) -> list[dict[str, float | int]]:
    """Compare several WDM tilings under the same Bayesian Whittle model."""
    rows: list[dict[str, float | int]] = []

    for nt in nt_values:
        if n_total % nt != 0:
            continue
        nf = n_total // nt
        if nt % 2 != 0 or nf % 2 != 0:
            continue

        config = PSplineConfig(
            n_interior_knots_time=max(6, nt // 4),
            n_interior_knots_freq=max(8, nf // 2 - 2),
            degree_time=base_config.degree_time,
            degree_freq=base_config.degree_freq,
            diff_order_time=base_config.diff_order_time,
            diff_order_freq=base_config.diff_order_freq,
            alpha_phi=base_config.alpha_phi,
            beta_phi=base_config.beta_phi,
            ridge_eps=base_config.ridge_eps,
            init_penalty_time=base_config.init_penalty_time,
            init_penalty_freq=base_config.init_penalty_freq,
            weak_weight_scale=base_config.weak_weight_scale,
            trim_time_bins=base_config.trim_time_bins,
            trim_low_freq_channels=base_config.trim_low_freq_channels,
            trim_high_freq_channels=base_config.trim_high_freq_channels,
            adaptive_time_knots=base_config.adaptive_time_knots,
            adaptive_time_knot_smoothing=base_config.adaptive_time_knot_smoothing,
            adaptive_time_knot_floor=base_config.adaptive_time_knot_floor,
        )

        rng = np.random.default_rng(random_seed)
        data = simulate_tv_arma(n_total, dgp=dgp, rng=rng)
        results = run_wdm_psd_mcmc(
            data,
            dt=dt,
            nt=nt,
            config=config,
            n_warmup=120,
            n_samples=120,
            num_chains=1,
            random_seed=random_seed,
        )
        reference_psd = monte_carlo_reference_wdm_psd(
            n_draws=40,
            n_total=n_total,
            dt=dt,
            nt=nt,
            dgp=dgp,
            seed=123,
            config=config,
        )

        rows.append(
            {
                "nt": nt,
                "nf": nf,
                "delta_t": float(results["wdm"].delta_t),
                "delta_f": float(results["wdm"].delta_f),
                "raw_error": relative_surface_error(reference_psd, results["power"]),
                "post_error": relative_surface_error(
                    reference_psd, results["psd_mean"]
                ),
                "n_knots_time": config.n_interior_knots_time,
                "n_knots_freq": config.n_interior_knots_freq,
            }
        )

    return rows


def monte_carlo_reference_wdm_psd(
    *,
    n_draws: int,
    n_total: int,
    dt: float,
    nt: int,
    dgp: str,
    seed: int,
    config: PSplineConfig,
) -> np.ndarray:
    """Empirical reference for the trimmed WDM local-power surface."""
    probe_wdm = TimeSeries(np.zeros(n_total), dt=dt).to_wdm(nt=nt)
    keep_time = np.arange(config.trim_time_bins, probe_wdm.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels,
        probe_wdm.nf + 1 - config.trim_high_freq_channels,
    )

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        sample = simulate_tv_arma(n_total, dgp=dgp, rng=rng)
        coeffs = np.asarray(TimeSeries(sample, dt=dt).to_wdm(nt=nt).coeffs)
        draws.append(coeffs[np.ix_(keep_time, keep_freq)] ** 2)
    return np.mean(draws, axis=0)


def estimate_vectorized_wdm_covariance(
    *,
    n_draws: int,
    n_total: int,
    dt: float,
    nt: int,
    dgp: str,
    seed: int,
    config: PSplineConfig,
) -> dict[str, np.ndarray]:
    """Estimate covariance and correlation of vec(w) across many realizations."""
    probe_wdm = TimeSeries(np.zeros(n_total), dt=dt).to_wdm(nt=nt)
    keep_time = np.arange(config.trim_time_bins, probe_wdm.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels,
        probe_wdm.nf + 1 - config.trim_high_freq_channels,
    )

    rng = np.random.default_rng(seed)
    coeff_vectors = []
    for _ in range(n_draws):
        sample = simulate_tv_arma(n_total, dgp=dgp, rng=rng)
        coeffs = np.asarray(TimeSeries(sample, dt=dt).to_wdm(nt=nt).coeffs)
        coeff_vectors.append(coeffs[np.ix_(keep_time, keep_freq)].reshape(-1, order="C"))

    coeff_vectors = np.stack(coeff_vectors, axis=0)
    covariance = np.cov(coeff_vectors, rowvar=False, ddof=1)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(std, std)
    correlation = np.clip(correlation, -1.0, 1.0)

    return {
        "covariance": covariance,
        "correlation": correlation,
        "keep_time": keep_time,
        "keep_freq": keep_freq,
    }


def relative_surface_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / np.linalg.norm(reference))


def plot_wdm_psd_results(
    results: dict[str, np.ndarray],
    *,
    data: np.ndarray,
    dt: float,
    reference_psd: np.ndarray,
    figure_stem: str,
) -> None:
    """Visual comparison of trimmed WDM power, posterior summary, and reference."""
    time_axis = np.arange(len(data)) * dt
    time_grid = results["time_grid"]
    freq_grid = results["freq_grid"]

    raw_log_power = np.log(results["power"] + 1e-8)
    post_mean = np.log(results["psd_mean"] + 1e-8)
    post_high = np.log(results["psd_upper"] + 1e-8)
    post_low = np.log(results["psd_lower"] + 1e-8)
    ref_log = np.log(reference_psd + 1e-8)
    ci_width = post_high - post_low

    vmin = min(raw_log_power.min(), post_mean.min(), ref_log.min())
    vmax = max(raw_log_power.max(), post_mean.max(), ref_log.max())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axes[0, 0].plot(time_axis, data, color="tab:blue", lw=1.0)
    axes[0, 0].set_title("Locally Stationary Time Series")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")

    mesh = axes[0, 1].pcolormesh(
        time_grid,
        freq_grid,
        raw_log_power.T,
        shading="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title("Raw WDM Log Power")
    axes[0, 1].set_xlabel("Rescaled WDM Time")
    axes[0, 1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes[0, 1], label="log local power")

    mesh = axes[0, 2].pcolormesh(
        time_grid,
        freq_grid,
        post_low.T,
        shading="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 2].set_title("Posterior 90% Lower Bound")
    axes[0, 2].set_xlabel("Rescaled WDM Time")
    axes[0, 2].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes[0, 2], label="log local power")

    mesh = axes[1, 0].pcolormesh(
        time_grid,
        freq_grid,
        post_mean.T,
        shading="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title("Posterior Mean Log PSD")
    axes[1, 0].set_xlabel("Rescaled WDM Time")
    axes[1, 0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes[1, 0], label="log local power")

    mesh = axes[1, 1].pcolormesh(
        time_grid,
        freq_grid,
        ref_log.T,
        shading="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title("Monte Carlo Reference")
    axes[1, 1].set_xlabel("Rescaled WDM Time")
    axes[1, 1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes[1, 1], label="log local power")

    mesh = axes[1, 2].pcolormesh(
        time_grid,
        freq_grid,
        ci_width.T,
        shading="nearest",
        cmap="magma",
    )
    axes[1, 2].set_title("Posterior 90% Width")
    axes[1, 2].set_xlabel("Rescaled WDM Time")
    axes[1, 2].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes[1, 2], label="log-width")
    save_figure(fig, figure_stem)


# %% [markdown]
# ## Experiment

# %%
RNG = np.random.default_rng(42)
dt = 0.1
nt = 24
n_total = 8192
dgp = "LS2"

data = simulate_tv_arma(n_total, dgp=dgp, rng=RNG)
config = PSplineConfig()

# %% [markdown]
# ## Is A Diagonal WDM Likelihood Plausible?
#
# Before fitting any Bayesian smoother, it is worth checking the core Whittle
# assumption in the most direct way: vectorize the trimmed WDM array and look at
# its empirical covariance over many simulated realizations.
#
# Define
#
# $$
# y = \operatorname{vec}(w),
# $$
#
# where each entry of `y` is one trimmed WDM pixel `(n,m)`. If the diagonal WDM
# likelihood were exact, then `Cov(y)` would be diagonal. In practice we only
# expect it to be approximately diagonal.
#
# The two heatmaps below show:
#
# - left: empirical covariance of `vec(w)`
# - right: empirical correlation of `vec(w)`
#
# The correlation view is usually easier to interpret, because the marginal
# variances vary across the WDM plane.

# %%
vec_cov = estimate_vectorized_wdm_covariance(
    n_draws=120,
    n_total=n_total,
    dt=dt,
    nt=nt,
    dgp=dgp,
    seed=1234,
    config=config,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)

cov_lim = float(np.max(np.abs(vec_cov["covariance"])))
corr_matrix = vec_cov["correlation"]
offdiag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
corr_lim = float(np.max(np.abs(corr_matrix[offdiag_mask])))

mesh = axes[0].imshow(
    vec_cov["covariance"],
    origin="lower",
    cmap="coolwarm",
    vmin=-cov_lim,
    vmax=cov_lim,
    aspect="auto",
)
axes[0].set_title("Empirical Covariance of vec(w)")
axes[0].set_xlabel("vectorized pixel index")
axes[0].set_ylabel("vectorized pixel index")
fig.colorbar(mesh, ax=axes[0], label="covariance")

mesh = axes[1].imshow(
    corr_matrix,
    origin="lower",
    cmap="coolwarm",
    vmin=-corr_lim,
    vmax=corr_lim,
    aspect="auto",
)
axes[1].set_title("Empirical Correlation of vec(w)")
axes[1].set_xlabel("vectorized pixel index")
fig.colorbar(mesh, ax=axes[1], label="correlation")

_ = save_figure(fig, "vectorized_wdm_covariance")

print(
    "Largest off-diagonal absolute correlation:",
    round(float(np.max(np.abs(corr_matrix[offdiag_mask]))), 3),
)
print(
    "Median off-diagonal absolute correlation:",
    round(float(np.median(np.abs(corr_matrix[offdiag_mask]))), 3),
)

# %% [markdown]
# ![Covariance and correlation of vec(w)](../wdm_time_varying_psd_assets/vectorized_wdm_covariance.png)

# %%
results = run_wdm_psd_mcmc(
    data,
    dt=dt,
    nt=nt,
    config=config,
    n_warmup=250,
    n_samples=250,
    num_chains=2,
    random_seed=12,
)
reference_psd = monte_carlo_reference_wdm_psd(
    n_draws=80,
    n_total=n_total,
    dt=dt,
    nt=nt,
    dgp=dgp,
    seed=123,
    config=config,
)
true_pointwise_psd = compute_true_tv_psd(
    dgp,
    results["time_grid"],
    2.0 * np.pi * dt * results["freq_grid"],
)

raw_error = relative_surface_error(reference_psd, results["power"])
smooth_error = relative_surface_error(reference_psd, results["psd_mean"])
true_psd_error = relative_surface_error(true_pointwise_psd, results["psd_mean"])
diagnostics = summarize_mcmc_diagnostics(results)

print(f"Original WDM grid shape:         {results['wdm'].shape}")
print(f"Fitted interior grid shape:      {results['power'].shape}")
print(f"Raw trimmed-power relative err.: {raw_error:.3f}")
print(f"Posterior mean relative error:   {smooth_error:.3f}")
print(f"Posterior vs true PSD error:     {true_psd_error:.3f}")
print(f"MCMC chains:                     {diagnostics['num_chains']}")
print(f"Divergences:                     {diagnostics['divergences']}")
print(
    "phi_time: "
    f"mean={diagnostics['phi_time']['mean']:.2f}, "
    f"n_eff={diagnostics['phi_time']['n_eff']:.1f}, "
    f"r_hat={diagnostics['phi_time']['r_hat']:.3f}"
)
print(
    "phi_freq: "
    f"mean={diagnostics['phi_freq']['mean']:.2f}, "
    f"n_eff={diagnostics['phi_freq']['n_eff']:.1f}, "
    f"r_hat={diagnostics['phi_freq']['r_hat']:.3f}"
)
for label, site in diagnostics["latent_log_psd"].items():
    i, j = site["index"]
    print(
        f"log_psd[{label}] at ({i}, {j}): "
        f"mean={site['mean']:.2f}, "
        f"n_eff={site['n_eff']:.1f}, "
        f"r_hat={site['r_hat']:.3f}"
    )

# %% [markdown]
# This version uses the Bayesian WDM Whittle likelihood directly on the trimmed
# coefficients. The posterior is still regularized by the spline prior, but the
# data term is the same Whittle form discussed in the manuscript. We also run
# two sequential chains and report divergences, `n_eff`, and `r_hat` for the
# smoothing hyperparameters and a few representative latent pixels.
#
# We report two comparison targets:
#
# - the Monte Carlo WDM reference, which estimates E[w_nm^2] directly
# - the analytical pointwise PSD of the underlying DGP, sampled on the same
#   WDM grid
#
# The first is the more like-for-like benchmark for the current WDM model. The
# second is still useful, but it mixes the WDM estimation problem with the extra
# question of how atom-averaged local power relates to the pointwise Fourier PSD.
#
# Interpretation:
#
# - `divergences = 0` is a necessary basic check for NUTS
# - `r_hat \approx 1` suggests different chains are mixing to the same region
# - larger `n_eff` means more stable posterior summaries
# - checking only `\phi_t` and `\phi_f` is not enough, so we also report a few
#   representative latent `log_psd` pixels

# %% [markdown]
# ## What Is The Posterior Estimating?
#
# The easiest way to get confused in this notebook is to compare two surfaces
# that look similar but are not actually the same mathematical object.
#
# There are four distinct surfaces in play:
#
# 1. **Analytical pointwise PSD**
#
#    $$
#    S_{\mathrm{true}}(u,\omega).
#    $$
#
#    This is the exact Fourier-domain time-varying PSD of the simulated ARMA
#    process. It is the smooth "ground-truth" function defined on continuous
#    time-frequency coordinates.
#
# 2. **Analytical PSD sampled on the WDM grid**
#
#    $$
#    S_{\mathrm{true}}(u_n,\omega_m).
#    $$
#
#    This is just the same function evaluated at the WDM bin centers. It is
#    still the pointwise Fourier PSD, but shown on the coarse grid used by the
#    WDM fit.
#
# 3. **Expected WDM local power**
#
#    $$
#    S_{nm}^{\mathrm{wdm}}
#    :=
#    \mathbb{E}[w_{nm}^2]
#    \approx
#    \iint |g_{nm}(t,f)|^2 S_{\mathrm{true}}(t,f)\,dt\,df.
#    $$
#
#    This is the quantity naturally linked to the WDM coefficient variance.
#    It is an atom-averaged version of the true PSD, not the pointwise PSD
#    itself. If the true PSD varies slowly across one WDM atom, then
#    `S_nm^wdm` and `S_true(u_n, omega_m)` are close. If not, they can differ.
#
# 4. **Posterior mean**
#
#    $$
#    \widehat{S}_{nm}^{\mathrm{post}}
#    =
#    \mathbb{E}[S_{nm}^{\mathrm{wdm}} \mid w].
#    $$
#
#    This is the Bayesian estimate produced by the spline model from one noisy
#    realization. It targets the WDM-domain local power surface, not the
#    continuous analytical PSD directly.
#
# So the key comparison is:
#
# - **analytical PSD**: the physical pointwise spectrum of the DGP
# - **WDM local power**: what the WDM coefficients actually measure
# - **posterior mean**: the estimate of that WDM local power from noisy data
#
# That is why the posterior can look smoother and slightly different from the
# analytical PSD even when the model is working correctly:
#
# - the posterior targets an atom-averaged quantity
# - each pixel is observed with heavy `\chi^2_1` noise
# - the spline prior shrinks peaks downward and troughs upward
#
# The dense posterior plot below should therefore be read as:
#
# - "what smooth WDM local-power surface did the Bayesian model infer?"
#
# not as:
#
# - "the exact analytical Fourier PSD recovered without approximation."

# %%
plot_wdm_psd_results(
    results,
    data=data,
    dt=dt,
    reference_psd=reference_psd,
    figure_stem="whittle_overview_surface",
)

# %% [markdown]
# ![Posterior surface overview](../wdm_time_varying_psd_assets/whittle_overview_surface.png)

# %% [markdown]
# The posterior surface above is shown on the native WDM grid. Since the latent
# model is a smooth tensor-product spline, we can also evaluate its posterior
# mean on a much denser plotting grid to visualize the fitted trend without the
# coarse WDM pixelation.

# %%
dense_posterior = evaluate_dense_posterior_mean(results, config)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True, sharey=True)

coarse_mesh = axes[0].pcolormesh(
    results["time_grid"],
    results["freq_grid"],
    np.log(results["psd_mean"] + 1e-8).T,
    shading="nearest",
    cmap="viridis",
)
axes[0].set_title("Posterior mean on WDM grid")
axes[0].set_xlabel("Rescaled WDM Time")
axes[0].set_ylabel("Frequency [Hz]")
fig.colorbar(coarse_mesh, ax=axes[0], label="log local power")

dense_mesh = axes[1].pcolormesh(
    dense_posterior["time_grid"],
    dense_posterior["freq_grid"],
    np.log(dense_posterior["psd_mean"] + 1e-8).T,
    shading="auto",
    cmap="viridis",
)
axes[1].set_title("Dense posterior mean from spline evaluation")
axes[1].set_xlabel("Rescaled Time")
fig.colorbar(dense_mesh, ax=axes[1], label="log local power")

_ = save_figure(fig, "posterior_mean_dense_grid")

# %% [markdown]
# ![Posterior mean on the WDM grid and on a dense spline grid](../wdm_time_varying_psd_assets/posterior_mean_dense_grid.png)

# %% [markdown]
# ## Whitening Check
#
# A downstream use of `S[n,m]` is to whiten WDM coefficients via
#
# $$
# z_{nm} = \frac{w_{nm}}{\sqrt{\widehat{S}_{nm}}}.
# $$
#
# If the fitted surface is a reasonable WDM noise model, then these whitened
# coefficients should look approximately standard normal: centered near zero and
# with variance close to one.

# %%
whitened = results["coeffs_fit"] / np.sqrt(np.maximum(results["psd_mean"], 1e-10))
whitened_flat = whitened.reshape(-1)
whitened_mean = float(np.mean(whitened_flat))
whitened_var = float(np.var(whitened_flat))

quantile_grid = np.linspace(0.01, 0.99, 150)
theoretical_q = stats.norm.ppf(quantile_grid)
empirical_q = np.quantile(whitened_flat, quantile_grid)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
hist_range = np.linspace(-4.0, 4.0, 50)
ax.hist(whitened_flat, bins=hist_range, density=True, alpha=0.7, color="tab:blue")
xgrid = np.linspace(-4.0, 4.0, 400)
ax.plot(xgrid, stats.norm.pdf(xgrid), color="black", lw=2.0, label="N(0,1)")
ax.set_title("Histogram of whitened coefficients")
ax.set_xlabel("w / sqrt(Shat)")
ax.set_ylabel("density")
ax.legend(loc="upper right")

ax = axes[1]
ax.scatter(theoretical_q, empirical_q, s=12, alpha=0.6, color="tab:blue")
qmin = min(theoretical_q.min(), empirical_q.min())
qmax = max(theoretical_q.max(), empirical_q.max())
ax.plot([qmin, qmax], [qmin, qmax], color="black", lw=2.0, ls="--")
ax.set_title("QQ plot of whitened coefficients")
ax.set_xlabel("Theoretical N(0,1) quantiles")
ax.set_ylabel("Empirical quantiles")

_ = save_figure(fig, "whitening_check")

print(f"Whitened coefficient mean:       {whitened_mean:.3f}")
print(f"Whitened coefficient variance:   {whitened_var:.3f}")

# %% [markdown]
# ![Whitening check](../wdm_time_varying_psd_assets/whitening_check.png)

# %% [markdown]
# ## Posterior Predictive Check
#
# Surface error alone can make a fit look worse than it is. A more relevant
# question for later inference is:
#
# - can the fitted Bayesian model generate raw WDM local powers that look like
#   the observed ones?
#
# To check this, we sample replicated coefficients from the posterior predictive
# distribution
#
# $$
# w^{\mathrm{rep}}_{nm} \mid S_{nm} \sim \mathcal{N}(0, S_{nm}),
# $$
#
# transform them to local power, and compare the observed `w[n,m]^2` to the
# resulting posterior predictive intervals.

# %%
posterior_log_psd_samples = np.asarray(results["samples"]["log_psd"])
rng_ppc = np.random.default_rng(123)
ppc_idx = rng_ppc.choice(len(posterior_log_psd_samples), size=min(200, len(posterior_log_psd_samples)), replace=True)
ppc_log_psd = posterior_log_psd_samples[ppc_idx]
ppc_std = np.exp(0.5 * ppc_log_psd)
ppc_coeffs = rng_ppc.normal(size=ppc_std.shape) * ppc_std
ppc_power = ppc_coeffs**2

ppc_power_median = np.median(ppc_power, axis=0)
ppc_power_lower = np.percentile(ppc_power, 5.0, axis=0)
ppc_power_upper = np.percentile(ppc_power, 95.0, axis=0)
ppc_coverage = float(
    np.mean(
        (results["power"] >= ppc_power_lower)
        & (results["power"] <= ppc_power_upper)
    )
)

channel_ppc = int(np.argmax(reference_psd.mean(axis=0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
ax.plot(
    results["time_grid"],
    results["power"][:, channel_ppc],
    color="tab:orange",
    lw=1.5,
    label="Observed w^2",
)
ax.plot(
    results["time_grid"],
    ppc_power_median[:, channel_ppc],
    color="tab:blue",
    lw=2.0,
    label="Posterior predictive median",
)
ax.fill_between(
    results["time_grid"],
    ppc_power_lower[:, channel_ppc],
    ppc_power_upper[:, channel_ppc],
    color="tab:blue",
    alpha=0.2,
    label="90% predictive interval",
)
ax.set_title(f"Posterior predictive check  —  channel m={channel_ppc}")
ax.set_xlabel("Rescaled WDM Time")
ax.set_ylabel("Local power")
ax.legend(loc="upper right")

ax = axes[1]
obs_log_power = np.log(results["power"].reshape(-1) + 1e-8)
rep_log_power = np.log(ppc_power.reshape(-1) + 1e-8)
bins = np.linspace(
    min(obs_log_power.min(), rep_log_power.min()),
    max(obs_log_power.max(), rep_log_power.max()),
    50,
)
ax.hist(obs_log_power, bins=bins, density=True, alpha=0.55, color="tab:orange", label="Observed log w^2")
ax.hist(rep_log_power, bins=bins, density=True, alpha=0.45, color="tab:blue", label="Replicated log w^2")
ax.set_title("Distribution of observed vs replicated log power")
ax.set_xlabel("log local power")
ax.set_ylabel("density")
ax.legend(loc="upper right")

_ = save_figure(fig, "posterior_predictive_check")

print(f"Posterior predictive 90% coverage of observed w^2: {ppc_coverage:.3f}")

# %% [markdown]
# ![Posterior predictive check](../wdm_time_varying_psd_assets/posterior_predictive_check.png)
#
# ## Pointwise True PSD Reference
#
# For the simulation DGPs we also know the analytical pointwise time-varying
# PSD. This is a different target from the Monte Carlo WDM reference:
#
# - Monte Carlo reference: estimates E[w_nm^2] on the WDM grid
# - true PSD reference: evaluates the Fourier-domain PSD S(u, omega) at the WDM
#   channel center frequencies
#
# These are not identical in principle, but comparing against both helps
# separate "WDM local power estimation" from "recovery of the underlying
# pointwise PSD".
#
# Two plots are useful here:
#
# - a dense analytical PSD plot, which shows the smooth underlying surface
# - the same analytical PSD sampled on the coarse WDM grid used by the fit
#
# If the sampled version looks blocky, that is mostly a grid-resolution effect,
# not a statement that the underlying analytical PSD itself is rough.

# %%
dense_time_grid = np.linspace(0.0, 1.0, 200)
dense_freq_grid = np.linspace(results["freq_grid"][0], results["freq_grid"][-1], 200)
dense_true_pointwise_psd = compute_true_tv_psd(
    dgp,
    dense_time_grid,
    2.0 * np.pi * dt * dense_freq_grid,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True, sharey=True)

dense_mesh = axes[0].pcolormesh(
    dense_time_grid,
    dense_freq_grid,
    np.log(dense_true_pointwise_psd + 1e-8).T,
    shading="auto",
    cmap="viridis",
)
axes[0].set_title("Analytical Pointwise True PSD (dense grid)")
axes[0].set_xlabel("Rescaled Time")
axes[0].set_ylabel("Frequency [Hz]")
fig.colorbar(dense_mesh, ax=axes[0], label="log local power")

sampled_mesh = axes[1].pcolormesh(
    results["time_grid"],
    results["freq_grid"],
    np.log(true_pointwise_psd + 1e-8).T,
    shading="nearest",
    cmap="viridis",
)
axes[1].set_title("Analytical Pointwise True PSD (sampled on WDM grid)")
axes[1].set_xlabel("Rescaled WDM Time")
fig.colorbar(sampled_mesh, ax=axes[1], label="log local power")
_ = save_figure(fig, "true_psd_dense_and_sampled")

# %% [markdown]
# ![Analytical PSD on dense and WDM grids](../wdm_time_varying_psd_assets/true_psd_dense_and_sampled.png)

# %% [markdown]
# ## One channel slice
#
# A single WDM frequency channel is easier to read than the full surface. The
# line plot below shows how the posterior mean tracks the Monte Carlo reference
# through time for the channel with the largest average reference power.
#
# This is often the easiest place to spot the qualitative failure mode:
#
# - peaks too low and troughs too high imply oversmoothing
# - large errors near the ends suggest boundary effects
# - very jagged posterior means suggest under-regularization

# %%
channel = int(np.argmax(reference_psd.mean(axis=0)))

fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
ax.plot(
    results["time_grid"],
    reference_psd[:, channel],
    color="black",
    lw=2.0,
    label="Monte Carlo WDM reference",
)
ax.plot(
    results["time_grid"],
    true_pointwise_psd[:, channel],
    color="tab:green",
    lw=1.8,
    ls="--",
    label="Pointwise true PSD",
)
ax.plot(
    results["time_grid"],
    results["power"][:, channel],
    color="tab:orange",
    alpha=0.55,
    lw=1.0,
    label="Raw squared coeffs",
)
ax.plot(
    results["time_grid"],
    results["psd_mean"][:, channel],
    color="tab:blue",
    lw=2.0,
    label="Posterior mean",
)
ax.fill_between(
    results["time_grid"],
    results["psd_lower"][:, channel],
    results["psd_upper"][:, channel],
    color="tab:blue",
    alpha=0.2,
    label="Posterior 90% interval",
)
ax.set_title(f"WDM channel m={channel} local power")
ax.set_xlabel("Rescaled WDM Time")
ax.set_ylabel("Local power")
ax.legend(loc="upper right")
_ = save_figure(fig, "channel_slice_whittle")

# %% [markdown]
# ![Single-channel comparison](../wdm_time_varying_psd_assets/channel_slice_whittle.png)
#
# ## Gamma likelihood: motivation and DOF sweep
#
# Each WDM coefficient is a single real Gaussian, so `w²/S ~ χ²₁` — one degree
# of freedom per pixel. Locally averaging K adjacent squared coefficients in
# frequency gives a Gamma(K/2, K/(2S)) observation with K× sharper curvature.
# K=2 matches single-taper STFT efficiency; K=3 (`freq_half_width=1`) already
# exceeds it. The sweep below confirms the improvement empirically.

# %%
dof_sweep_rows = run_dof_sweep(
    data,
    dt=dt,
    nt=nt,
    reference_psd=reference_psd,
    freq_hw_values=[0, 1, 2, 3, 5],
    base_config=config,
    n_warmup=200,
    n_samples=200,
    random_seed=42,
)

print(f"{'hw':>4}  {'K=DOF':>6}  {'raw_err':>9}  {'post_err':>10}")
for row in dof_sweep_rows:
    print(
        f"{row['freq_hw']:4d}  {row['dof']:6d}  "
        f"{row['raw_error']:9.3f}  {row['post_error']:10.3f}"
    )

# %%
dofs = [row["dof"] for row in dof_sweep_rows]
post_errors = [row["post_error"] for row in dof_sweep_rows]
raw_errors = [row["raw_error"] for row in dof_sweep_rows]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
ax.plot(dofs, raw_errors, marker="s", color="tab:orange", lw=1.5, label="Raw periodogram")
ax.plot(dofs, post_errors, marker="o", color="tab:blue", lw=2.0, label="Posterior mean")
ax.axvline(2, color="tab:green", ls="--", lw=1.2, label="K=2 ≈ STFT Exp(1) parity")
ax.set_xlabel("Effective DOF per pixel  K")
ax.set_ylabel("Relative error vs MC reference")
ax.set_title("Fit quality vs. periodogram DOF")
ax.legend()

# Channel slice comparing K=1 vs best K
best_row = min(dof_sweep_rows, key=lambda r: float(r["post_error"]))
k1_row = dof_sweep_rows[0]

ax = axes[1]
ax.plot(
    results["time_grid"],
    true_pointwise_psd[:, channel],
    color="tab:green",
    lw=2.0,
    label=f"True PSD  S(e^{{jω}})  ch={channel}",
)
ax.plot(
    results["time_grid"],
    reference_psd[:, channel],
    color="black",
    lw=1.5,
    ls="--",
    label="MC reference  E[w²]",
)
ax.plot(
    results["time_grid"],
    k1_row["psd_mean"][:, channel],
    color="tab:orange",
    lw=1.8,
    label=f"Posterior mean  K=1 (Whittle)",
)
ax.plot(
    results["time_grid"],
    best_row["psd_mean"][:, channel],
    color="tab:blue",
    lw=2.0,
    label=f"Posterior mean  K={best_row['dof']} (Gamma)",
)
ax.set_title(f"Channel m={channel}  —  K=1 vs K={best_row['dof']}")
ax.set_xlabel("Rescaled WDM Time")
ax.set_ylabel("Local power")
ax.legend(fontsize=8)
_ = save_figure(fig, "dof_sweep")

# %% [markdown]
# ![DOF sweep and channel comparison](../wdm_time_varying_psd_assets/dof_sweep.png)
#
# K=1 is the plain Whittle baseline; K=2 (green dashed) matches single-taper
# STFT efficiency; K≥3 exceeds it. Past K≈5 the gain flattens — the bottleneck
# shifts to spline resolution, not pixel noise.
#
# ## Comparison: Whittle vs Gamma Likelihood
#
# We now run the same data through both the original Whittle model
# (`periodogram_freq_half_width=0`) and the Gamma model with frequency-smoothed
# periodogram (`periodogram_freq_half_width=2`, giving K=5 DOF).

# %%
config_whittle = PSplineConfig()
config_gamma = PSplineConfig(periodogram_freq_half_width=2)

RNG2 = np.random.default_rng(42)
data2 = simulate_tv_arma(n_total, dgp=dgp, rng=RNG2)

results_whittle = run_wdm_psd_mcmc(
    data2,
    dt=dt,
    nt=nt,
    config=config_whittle,
    n_warmup=250,
    n_samples=250,
    num_chains=2,
    random_seed=12,
)
results_gamma = run_wdm_psd_mcmc(
    data2,
    dt=dt,
    nt=nt,
    config=config_gamma,
    n_warmup=250,
    n_samples=250,
    num_chains=2,
    random_seed=12,
)

reference_psd2 = monte_carlo_reference_wdm_psd(
    n_draws=80,
    n_total=n_total,
    dt=dt,
    nt=nt,
    dgp=dgp,
    seed=123,
    config=config_whittle,
)

err_whittle = relative_surface_error(reference_psd2, results_whittle["psd_mean"])
err_gamma = relative_surface_error(reference_psd2, results_gamma["psd_mean"])
diag_whittle = summarize_mcmc_diagnostics(results_whittle)
diag_gamma = summarize_mcmc_diagnostics(results_gamma)

print("Whittle (chi-squared 1 per pixel)")
print(f"  relative error:  {err_whittle:.3f}")
print(f"  divergences:     {diag_whittle['divergences']}")
print(
    f"  phi_time: mean={diag_whittle['phi_time']['mean']:.2f}, "
    f"n_eff={diag_whittle['phi_time']['n_eff']:.1f}, "
    f"r_hat={diag_whittle['phi_time']['r_hat']:.3f}"
)
print(
    f"  phi_freq: mean={diag_whittle['phi_freq']['mean']:.2f}, "
    f"n_eff={diag_whittle['phi_freq']['n_eff']:.1f}, "
    f"r_hat={diag_whittle['phi_freq']['r_hat']:.3f}"
)

print(f"\nGamma (K={results_gamma['periodogram_dof']} DOF, freq_half_width=2)")
print(f"  relative error:  {err_gamma:.3f}")
print(f"  divergences:     {diag_gamma['divergences']}")
print(
    f"  phi_time: mean={diag_gamma['phi_time']['mean']:.2f}, "
    f"n_eff={diag_gamma['phi_time']['n_eff']:.1f}, "
    f"r_hat={diag_gamma['phi_time']['r_hat']:.3f}"
)
print(
    f"  phi_freq: mean={diag_gamma['phi_freq']['mean']:.2f}, "
    f"n_eff={diag_gamma['phi_freq']['n_eff']:.1f}, "
    f"r_hat={diag_gamma['phi_freq']['r_hat']:.3f}"
)

# %% [markdown]
# The line plot below is useful for a single channel, but it can hide where the
# Gamma likelihood helps most. The next figure compares the full posterior mean
# surfaces directly against the same Monte Carlo reference.

# %%
surface_ref_log = np.log(reference_psd2 + 1e-8)
surface_whittle_log = np.log(results_whittle["psd_mean"] + 1e-8)
surface_gamma_log = np.log(results_gamma["psd_mean"] + 1e-8)
width_whittle_log = (
    np.log(results_whittle["psd_upper"] + 1e-8)
    - np.log(results_whittle["psd_lower"] + 1e-8)
)
width_gamma_log = (
    np.log(results_gamma["psd_upper"] + 1e-8)
    - np.log(results_gamma["psd_lower"] + 1e-8)
)

vmin = min(
    surface_ref_log.min(),
    surface_whittle_log.min(),
    surface_gamma_log.min(),
)
vmax = max(
    surface_ref_log.max(),
    surface_whittle_log.max(),
    surface_gamma_log.max(),
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharex=True, sharey=True)

panels = [
    (axes[0, 0], surface_ref_log, "Monte Carlo reference", "viridis", vmin, vmax, "log local power"),
    (axes[0, 1], surface_whittle_log, "Whittle posterior mean", "viridis", vmin, vmax, "log local power"),
    (axes[0, 2], surface_gamma_log, f"Gamma posterior mean (K={results_gamma['periodogram_dof']})", "viridis", vmin, vmax, "log local power"),
    (axes[1, 0], surface_whittle_log - surface_ref_log, "Whittle minus reference", "coolwarm", None, None, "log error"),
    (axes[1, 1], surface_gamma_log - surface_ref_log, "Gamma minus reference", "coolwarm", None, None, "log error"),
    (axes[1, 2], width_gamma_log - width_whittle_log, "Gamma minus Whittle width", "magma", None, None, "delta log-width"),
]

for ax, field, title, cmap, panel_vmin, panel_vmax, cbar_label in panels:
    mesh = ax.pcolormesh(
        results_whittle["time_grid"],
        results_whittle["freq_grid"],
        field.T,
        shading="nearest",
        cmap=cmap,
        vmin=panel_vmin,
        vmax=panel_vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Rescaled WDM Time")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=ax, label=cbar_label)
_ = save_figure(fig, "whittle_vs_gamma_surface")

# %% [markdown]
# ![Whittle versus Gamma posterior surfaces](../wdm_time_varying_psd_assets/whittle_vs_gamma_surface.png)

# %%
# Dense-grid three-way comparison: Whittle | Gamma | True PSD
_dense_t = np.linspace(results_whittle["time_grid"][0], results_whittle["time_grid"][-1], 300)
_dense_f = np.linspace(results_whittle["freq_grid"][0], results_whittle["freq_grid"][-1], 300)

dense_whittle = evaluate_dense_posterior_mean(results_whittle, config_whittle,
                                               n_time_dense=300, n_freq_dense=300)
dense_gamma = evaluate_dense_posterior_mean(results_gamma, config_gamma,
                                             n_time_dense=300, n_freq_dense=300)
dense_true = compute_true_tv_psd(
    dgp, _dense_t, 2.0 * np.pi * dt * _dense_f
)

_vmin = np.log(dense_true + 1e-8).min()
_vmax = np.log(dense_true + 1e-8).max()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True, sharey=True)
for ax, surface, title in [
    (axes[0], np.log(dense_whittle["psd_mean"] + 1e-8), "Whittle posterior mean"),
    (axes[1], np.log(dense_gamma["psd_mean"]   + 1e-8), f"Gamma posterior mean (K={results_gamma['periodogram_dof']})"),
    (axes[2], np.log(dense_true + 1e-8),                "True PSD  S(e^{jω})"),
]:
    mesh = ax.pcolormesh(
        _dense_t, _dense_f, surface.T,
        shading="auto", cmap="viridis", vmin=_vmin, vmax=_vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Rescaled Time")
    fig.colorbar(mesh, ax=ax, label="log local power")
axes[0].set_ylabel("Frequency [Hz]")
_ = save_figure(fig, "dense_three_way_comparison")

# %% [markdown]
# ![Dense three-way comparison](../wdm_time_varying_psd_assets/dense_three_way_comparison.png)

# %%
channel2 = int(np.argmax(reference_psd2.mean(axis=0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)

for ax, res, label in [
    (axes[0], results_whittle, "Whittle (K=1)"),
    (axes[1], results_gamma, f"Gamma (K={results_gamma['periodogram_dof']})"),
]:
    ax.plot(
        res["time_grid"],
        reference_psd2[:, channel2],
        color="black",
        lw=2.0,
        label="MC reference",
    )
    ax.plot(
        res["time_grid"],
        res["psd_mean"][:, channel2],
        color="tab:blue",
        lw=2.0,
        label="Posterior mean",
    )
    ax.fill_between(
        res["time_grid"],
        res["psd_lower"][:, channel2],
        res["psd_upper"][:, channel2],
        color="tab:blue",
        alpha=0.2,
        label="90% interval",
    )
    ax.set_title(f"{label}  —  channel m={channel2}")
    ax.set_xlabel("Rescaled WDM Time")
    ax.legend(loc="upper right")

axes[0].set_ylabel("Local power")
_ = save_figure(fig, "whittle_vs_gamma_channel")

# %% [markdown]
# ![Whittle versus Gamma channel slice](../wdm_time_varying_psd_assets/whittle_vs_gamma_channel.png)
#
# ## Takeaway
#
# The Whittle likelihood on WDM coefficients is conceptually correct: it is
# the natural diagonal approximation for a locally stationary process observed
# through a near-orthogonal time-frequency transform. But from a statistical
# standpoint, each pixel contributes a single chi-squared(1) observation,
# which has 141% relative noise. The spline prior regularizes this, but the
# bias-variance tradeoff is severe for one realization.
#
# The smoothed-periodogram + Gamma likelihood reformulation addresses this
# directly: by locally averaging K squared coefficients before fitting, the
# per-pixel noise variance drops by a factor of K while the bias remains
# small (the PSD varies smoothly relative to the averaging window). The
# Gamma(K/2, K/(2S)) likelihood correctly accounts for the reduced noise,
# giving K times sharper curvature and better-identified posterior surfaces.
#
# Summary of findings:
#
# - **The atom-averaging concern is real but misidentified**: the MC reference
#   and the Bayesian estimate both target the same atom-averaged quantity, so
#   the residual error is not from this source.
# - **The real bottleneck is chi-squared(1) noise**: 141% CV per pixel forces
#   aggressive smoothing under any reasonable prior.
# - **Derivative-based roughness penalties** help more than coefficient
#   differences, and the **WDM tiling** remains a major lever.
# - **The Gamma likelihood on a smoothed WDM periodogram** is the most direct
#   improvement to the observation model, reducing noise without changing the
#   spline or penalty machinery.
# - For recovering the pointwise PSD rather than atom-averaged power, a
#   deconvolution step accounting for the atom footprint would be needed, but
#   that is a separate problem from the noise-reduction issue addressed here.
