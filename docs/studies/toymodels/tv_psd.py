# ruff: noqa: E402, I001

from __future__ import annotations

import atexit
import builtins
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
from scipy.ndimage import gaussian_filter1d

from wdm_transform import TimeSeries

if "__file__" in globals():
    NOTEBOOK_DIR = Path(__file__).resolve().parent
else:
    cwd = Path.cwd()
    docs_studies_dir = cwd / "docs" / "studies"
    NOTEBOOK_DIR = docs_studies_dir if docs_studies_dir.exists() else cwd

outdir_tv_psd = NOTEBOOK_DIR / "outdir_tv_psd"
outdir_tv_psd.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = outdir_tv_psd / "run_log.txt"
MARKDOWN_PATH = NOTEBOOK_DIR / "tv_psd.md"
RUN_LOG_START = "<!-- BEGIN GENERATED RUN LOG -->"
RUN_LOG_END = "<!-- END GENERATED RUN LOG -->"
_run_log_chunks: list[str] = []


def print(*args, sep=" ", end="\n", file=None, flush=False):
    target = sys.stdout if file is None else file
    builtins.print(*args, sep=sep, end=end, file=target, flush=flush)
    if target in (sys.stdout, sys.__stdout__):
        text = sep.join(str(arg) for arg in args) + end
        _run_log_chunks.append(text)
        RUN_LOG_PATH.write_text("".join(_run_log_chunks))


def _update_markdown_run_log() -> None:
    if not MARKDOWN_PATH.exists():
        return
    body = "".join(_run_log_chunks).rstrip()
    block = (
        f"{RUN_LOG_START}\n```text\n{body}\n```\n{RUN_LOG_END}"
        if body
        else f"{RUN_LOG_START}\n_No run output captured yet._\n{RUN_LOG_END}"
    )
    markdown = MARKDOWN_PATH.read_text()
    if RUN_LOG_START in markdown and RUN_LOG_END in markdown:
        start = markdown.index(RUN_LOG_START)
        end = markdown.index(RUN_LOG_END) + len(RUN_LOG_END)
        markdown = markdown[:start] + block + markdown[end:]
    else:
        markdown = markdown.rstrip() + (
            "\n\n## Run log\n\n"
            "This section is generated from the script's `print()` output.\n\n"
            f"{block}\n"
        )
    MARKDOWN_PATH.write_text(markdown)


atexit.register(_update_markdown_run_log)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 160) -> Path:
    """Save a notebook figure to the docs static directory and close it."""
    path = outdir_tv_psd / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def choose_valid_wdm_shape(n_total: int, desired_nt: int) -> tuple[int, int]:
    """Return the nearest valid even WDM tiling for a signal length."""
    valid_nt = [
        candidate_nt
        for candidate_nt in range(2, n_total + 1, 2)
        if n_total % candidate_nt == 0 and (n_total // candidate_nt) % 2 == 0
    ]
    if not valid_nt:
        raise ValueError(
            f"No valid even WDM tiling exists for n_total={n_total}."
        )

    nt = min(
        valid_nt,
        key=lambda candidate_nt: (abs(candidate_nt - desired_nt), -candidate_nt),
    )
    nf = n_total // nt
    return nt, nf


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
        raise ValueError(
            "x and pilot_profile must be one-dimensional with matching length."
        )

    if n_interior_knots <= 0:
        return np.array([], dtype=float)

    smooth_profile = gaussian_filter1d(
        pilot_profile, sigma=smoothing_sigma, mode="nearest"
    )
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
    penalty_sum = (
        penalty_time + penalty_freq + ridge_eps * np.eye(penalty_time.shape[0])
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
    nt, _ = choose_valid_wdm_shape(len(data), nt)
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
    dense_time_grid = np.linspace(
        results["time_grid"][0], results["time_grid"][-1], n_time_dense
    )
    dense_freq_grid = np.linspace(
        results["freq_grid"][0], results["freq_grid"][-1], n_freq_dense
    )
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
    nt, _ = choose_valid_wdm_shape(n_total, nt)
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
    nt, _ = choose_valid_wdm_shape(n_total, nt)
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
        coeff_vectors.append(
            coeffs[np.ix_(keep_time, keep_freq)].reshape(-1, order="C")
        )

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


# Script experiment starts here. Downstream studies load only the helper
# definitions above this marker.
RNG = np.random.default_rng(42)
dt = 0.1
DESIRED_NT = 24
n_total = 2048
dgp = "LS2"

nt, nf = choose_valid_wdm_shape(n_total, DESIRED_NT)

data = simulate_tv_arma(n_total, dgp=dgp, rng=RNG)
config = PSplineConfig()

if nt != DESIRED_NT:
    print(
        f"Adjusted WDM tiling from nt={DESIRED_NT} to nt={nt} so that "
        f"n_total={n_total} factors into an even (nt, nf)=({nt}, {nf}) grid."
    )

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

plot_wdm_psd_results(
    results,
    data=data,
    dt=dt,
    reference_psd=reference_psd,
    figure_stem="whittle_overview_surface",
)

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

posterior_log_psd_samples = np.asarray(results["samples"]["log_psd"])
rng_ppc = np.random.default_rng(123)
ppc_idx = rng_ppc.choice(
    len(posterior_log_psd_samples),
    size=min(200, len(posterior_log_psd_samples)),
    replace=True,
)
ppc_log_psd = posterior_log_psd_samples[ppc_idx]
ppc_std = np.exp(0.5 * ppc_log_psd)
ppc_coeffs = rng_ppc.normal(size=ppc_std.shape) * ppc_std
ppc_power = ppc_coeffs**2

ppc_power_median = np.median(ppc_power, axis=0)
ppc_power_lower = np.percentile(ppc_power, 5.0, axis=0)
ppc_power_upper = np.percentile(ppc_power, 95.0, axis=0)
ppc_coverage = float(
    np.mean(
        (results["power"] >= ppc_power_lower) & (results["power"] <= ppc_power_upper)
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
ax.hist(
    obs_log_power,
    bins=bins,
    density=True,
    alpha=0.55,
    color="tab:orange",
    label="Observed log w^2",
)
ax.hist(
    rep_log_power,
    bins=bins,
    density=True,
    alpha=0.45,
    color="tab:blue",
    label="Replicated log w^2",
)
ax.set_title("Distribution of observed vs replicated log power")
ax.set_xlabel("log local power")
ax.set_ylabel("density")
ax.legend(loc="upper right")

_ = save_figure(fig, "posterior_predictive_check")

print(f"Posterior predictive 90% coverage of observed w^2: {ppc_coverage:.3f}")

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
