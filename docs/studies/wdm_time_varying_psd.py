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
# \[
# w_{nm} \sim \mathcal{N}(0, S_{nm}),
# \]
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

# %%
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

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
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

from wdm_transform import TimeSeries


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


# %% [markdown]
# ## Spline Surface And Roughness Prior
#
# We model the log local power with a tensor-product spline surface:
#
# \[
# \log S_{nm}
# =
# \sum_{r=1}^{R_t}\sum_{s=1}^{R_f}
# B_r^{(t)}(u_n)\,W_{rs}\,B_s^{(f)}(\nu_m),
# \]
#
# where `B_r^{(t)}` and `B_s^{(f)}` are B-spline basis functions in time and
# frequency, and `W_{rs}` are the unknown spline coefficients.
#
# The prior is built from derivative-based roughness matrices rather than simple
# coefficient differences. For example, in time we form
#
# \[
# R_t[i,j] = \int B_i^{(q_t)}(u)\,B_j^{(q_t)}(u)\,du,
# \]
#
# and similarly in frequency. The resulting anisotropic prior is
#
# \[
# p(W \mid \phi_t,\phi_f)
# \propto
# \exp\left[
# -\frac{\phi_t}{2}\operatorname{vec}(W)^\top(R_t \otimes I_f)\operatorname{vec}(W)
# -\frac{\phi_f}{2}\operatorname{vec}(W)^\top(I_t \otimes R_f)\operatorname{vec}(W)
# \right].
# \]
#
# This is closer to penalizing actual curvature of the latent surface in
# physical coordinates than penalizing nearest-neighbor coefficient differences.
#
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

    init_vals = initialize_with_penalized_least_squares(
        power,
        B_time,
        B_freq,
        penalty_time,
        penalty_freq,
        config,
    )

    kernel = NUTS(
        pspline_wdm_model,
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
    mcmc.run(
        random.PRNGKey(random_seed),
        jnp.asarray(coeffs_fit),
        jnp.asarray(B_time),
        jnp.asarray(B_freq),
        jnp.asarray(penalty_time),
        jnp.asarray(penalty_freq),
        penalty_time_rank,
        penalty_freq_rank,
        config,
    )

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
        "time_grid": time_grid,
        "freq_grid": freq_grid,
        "B_time": B_time,
        "B_freq": B_freq,
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


def relative_surface_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / np.linalg.norm(reference))


def plot_wdm_psd_results(
    results: dict[str, np.ndarray],
    *,
    data: np.ndarray,
    dt: float,
    reference_psd: np.ndarray,
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


# %% [markdown]
# ## Observation model
#
# We want a Bayesian fit, but we still want the WDM Whittle likelihood.
# In this setting the observation is the coefficient itself, not an auxiliary
# Gamma-distributed power summary.
#
# The working approximation is:
#
# \[
# w_{nm} \mid S_{nm} \approx \mathcal{N}(0, S_{nm}),
# \]
#
# where `S_nm` is the local evolutionary power on the WDM grid. The resulting
# log-likelihood for one realization is
#
# \[
# \log p(w \mid S)
# =
# -\frac{1}{2}\sum_{n,m}
# \left[
# \log(2\pi) + \log S_{nm} + \frac{w_{nm}^2}{S_{nm}}
# \right].
# \]
#
# This is exactly the WDM Whittle likelihood from the manuscript with `h_nm=0`.
# It is already a Bayesian model once we place priors on the spline surface and
# its smoothing hyperparameters.
#
# We then place a smooth tensor-product spline model on the log local power:
#
# \[
# \log S_{nm} = \sum_{r=1}^{R_t}\sum_{s=1}^{R_f}
# B_r^{(t)}(u_n)\, W_{rs}\, B_s^{(f)}(\nu_m).
# \]
#
# The anisotropic smoothness prior penalizes roughness separately in time and
# frequency:
#
# \[
# p(W \mid \phi_t, \phi_f)
# \propto
# \exp\left[
# -\frac{\phi_t}{2}\int \left(\partial_u^{q_t}\log S(u,\nu)\right)^2\,du\,d\nu
# -\frac{\phi_f}{2}\int \left(\partial_\nu^{q_f}\log S(u,\nu)\right)^2\,du\,d\nu
# \right].
# \]
# In the code, these integrals are approximated by derivative-based B-spline
# roughness matrices instead of coefficient-index differences.
#
# We also drop the edge time bins and the DC/Nyquist channels from the fit,
# since those are where the WDM boundary effects and edge-wavelet behavior are
# least reliable.
#
# A useful caveat is that the exact expectation of the squared coefficient is
# closer to an atom-averaged quantity:
#
# \[
# \mathbb{E}[w_{nm}^2]
# \approx
# \iint |g_{nm}(t,f)|^2\,S(t,f)\,dt\,df.
# \]
#
# So even with a good prior and a converged sampler, some residual mismatch can
# remain if the true PSD changes appreciably across one WDM atom footprint.
#
# ## Synthetic locally stationary data
#
# We use the same `LS2` time-varying MA process as in the original moving-PSD
# demo, but we estimate the local power directly on the WDM grid.

# %%
RNG = np.random.default_rng(42)
dt = 0.1
nt = 24
n_total = 576
dgp = "LS2"

data = simulate_tv_arma(n_total, dgp=dgp, rng=RNG)
config = PSplineConfig()

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

raw_error = relative_surface_error(reference_psd, results["power"])
smooth_error = relative_surface_error(reference_psd, results["psd_mean"])
diagnostics = summarize_mcmc_diagnostics(results)

print(f"Original WDM grid shape:         {results['wdm'].shape}")
print(f"Fitted interior grid shape:      {results['power'].shape}")
print(f"Raw trimmed-power relative err.: {raw_error:.3f}")
print(f"Posterior mean relative error:   {smooth_error:.3f}")
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
# Interpretation:
#
# - `divergences = 0` is a necessary basic check for NUTS
# - `r_hat \approx 1` suggests different chains are mixing to the same region
# - larger `n_eff` means more stable posterior summaries
# - checking only `\phi_t` and `\phi_f` is not enough, so we also report a few
#   representative latent `log_psd` pixels

# %%
plot_wdm_psd_results(
    results,
    data=data,
    dt=dt,
    reference_psd=reference_psd,
)

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
    label="Monte Carlo reference",
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
    label="90% interval",
)
ax.set_title(f"WDM channel m={channel} local power")
ax.set_xlabel("Rescaled WDM Time")
ax.set_ylabel("Local power")
ax.legend(loc="upper right")

# %% [markdown]
# ## Adaptive Time Knots
#
# The WDM time grid used here is uniform, so quantile-based knot placement
# would be identical to uniform knots. To make the time basis more adaptive, we
# build a pilot time profile
#
# \[
# a_n = \frac{1}{M}\sum_m \log(w_{nm}^2 + \epsilon),
# \]
#
# smooth it slightly, and place more knots where its derivative is larger. In
# effect, the knot density is made roughly proportional to
#
# \[
# \rho(u) \propto c + \left|\frac{d}{du}a(u)\right|,
# \]
#
# where `c > 0` prevents all knots from collapsing into a tiny region.
#
# In this notebook that adjustment is a secondary refinement, not the main
# driver of fit quality, but it is a useful way to allocate spline flexibility
# to the parts of the time axis that appear most variable.
#
# %% [markdown]
# ## Sweep over `nt`
#
# The WDM Whittle approximation depends on the tiling itself. Changing `nt`
# changes both
#
# - the WDM time resolution `\Delta T = n_f dt`
# - the WDM frequency resolution `\Delta F = 1/(2 \Delta T)`
#
# so this is not just a computational setting. It changes the statistical
# approximation.
#
# The quick sweep below keeps `n_total` fixed and compares several even values
# of `nt`. For each choice we run a lighter one-chain version of the Bayesian
# spline model and compare the posterior mean surface to a Monte Carlo
# reference.
#
# Interpreting the sweep:
#
# - smaller `nt` means coarser time resolution and finer frequency resolution
# - larger `nt` means finer time resolution and coarser frequency resolution
# - the best value depends on how rapidly the true PSD changes in time versus
#   how sharply it varies in frequency

# %%
sweep_rows = run_nt_sweep(
    n_total=1152,
    dt=dt,
    dgp=dgp,
    nt_values=[16, 24, 32, 48, 72],
    base_config=config,
)

print("nt sweep (lighter one-chain runs)")
print(
    "  nt   nf   delta_t   delta_f   knots_t   knots_f   raw_err   post_err"
)
for row in sweep_rows:
    print(
        f"{row['nt']:4d} "
        f"{row['nf']:4d} "
        f"{row['delta_t']:8.3f} "
        f"{row['delta_f']:8.3f} "
        f"{row['n_knots_time']:9d} "
        f"{row['n_knots_freq']:9d} "
        f"{row['raw_error']:9.3f} "
        f"{row['post_error']:9.3f}"
    )

best_row = min(sweep_rows, key=lambda row: float(row["post_error"]))
print(
    "\nBest posterior surface in this sweep: "
    f"nt={best_row['nt']}, nf={best_row['nf']}, "
    f"post_err={best_row['post_error']:.3f}"
)

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.plot(
    [row["nt"] for row in sweep_rows],
    [row["raw_error"] for row in sweep_rows],
    marker="o",
    color="tab:orange",
    label="Raw trimmed power",
)
ax.plot(
    [row["nt"] for row in sweep_rows],
    [row["post_error"] for row in sweep_rows],
    marker="o",
    color="tab:blue",
    label="Posterior mean",
)
ax.set_xlabel("nt")
ax.set_ylabel("Relative error vs Monte Carlo reference")
ax.set_title("Effect of WDM Tiling Choice")
ax.legend(loc="upper right")

# %% [markdown]
# ## Takeaway
#
# The right Bayesian formulation here is not "Gamma smoothing first, Bayes
# later." It is a spline prior on `log S_nm` together with the WDM Whittle
# likelihood on the coefficients themselves. The main practical stabilizers are
# the prior, the anisotropic penalties, and trimming the least reliable edge
# bins.
#
# In the experiments in this note:
#
# - derivative-based roughness penalties help more than simple coefficient
#   difference penalties
# - the WDM tiling choice `nt` is a major lever
# - adaptive time knots are a smaller effect than the penalty and tiling choices
# - the remaining mismatch is likely tied to the atom-averaged nature of
#   `\mathbb{E}[w_{nm}^2]`, not just lack of MCMC convergence
#
# So a reasonable next step beyond this notebook would be to improve the
# forward model for the expected WDM local power, not just to keep increasing
# spline flexibility.
