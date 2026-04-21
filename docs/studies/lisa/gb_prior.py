"""Shared Galactic-binary prior definitions for the LISA study."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


SOURCE_CATALOG = np.array(
    [
        [1.35962e-3, 8.94581279e-19, 1.07345e-22, 2.40, 0.31, 3.56, 0.52, 3.06],
        [1.41220e-3, 2.30000000e-18, 8.20000000e-23, 2.15, 0.18, 1.20, 0.93, 1.40],
    ],
    dtype=float,
)

F0_GLOBAL_BOUNDS = (
    float(SOURCE_CATALOG[:, 0].min() - 1.5e-5),
    float(SOURCE_CATALOG[:, 0].max() + 1.5e-5),
)
FDOT_GLOBAL_BOUNDS = (5.0e-19, 4.0e-18)
FIXED_FDOT_PRIOR_BOUNDS = FDOT_GLOBAL_BOUNDS
FIXED_A_PRIOR_BOUNDS = (6.0e-24, 1.7e-23)
F0_REF = float(np.mean(SOURCE_CATALOG[:, 0]))


def lisa_delta_f0_prior_half_width() -> float:
    return float(os.getenv("LISA_DELTA_F0_PRIOR_HALF_WIDTH", "3e-8"))


def lisa_delta_f0_prior_sigma() -> float:
    return float(os.getenv("LISA_DELTA_F0_PRIOR_SIGMA", "1e-8"))


def lisa_f0_jitter_width() -> float:
    delta_f0_half_width = lisa_delta_f0_prior_half_width()
    if not 0.0 < delta_f0_half_width < F0_REF:
        raise ValueError(
            "Expected 0 < LISA_DELTA_F0_PRIOR_HALF_WIDTH < F0_REF; "
            f"got {delta_f0_half_width:.6e} with F0_REF={F0_REF:.6e}"
        )
    return float(np.log1p(delta_f0_half_width / F0_REF))


@dataclass(frozen=True)
class LocalPriorInfo:
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


def _draw_truncated_normal(
    rng: np.random.Generator,
    *,
    loc: float,
    scale: float,
    low: float,
    high: float,
) -> float:
    for _ in range(10_000):
        value = float(rng.normal(loc=loc, scale=scale))
        if low <= value <= high:
            return value
    raise RuntimeError(
        f"Failed to draw truncated normal after many attempts: "
        f"loc={loc}, scale={scale}, low={low}, high={high}"
    )


def draw_positive_parameter_from_bounds(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> float:
    log_low = float(np.log(bounds[0]))
    log_high = float(np.log(bounds[1]))
    log_value = _draw_truncated_normal(
        rng,
        loc=0.5 * (log_low + log_high),
        scale=0.25 * (log_high - log_low),
        low=log_low,
        high=log_high,
    )
    return float(np.exp(log_value))


def build_local_prior_info(
    *,
    prior_f0: tuple[float, float],
    prior_fdot: tuple[float, float],
    prior_A: tuple[float, float],
) -> LocalPriorInfo:
    logf0_bounds = (float(np.log(prior_f0[0])), float(np.log(prior_f0[1])))
    logfdot_bounds = (float(np.log(prior_fdot[0])), float(np.log(prior_fdot[1])))
    logA_bounds = (float(np.log(prior_A[0])), float(np.log(prior_A[1])))

    return LocalPriorInfo(
        prior_center=np.array([
            0.5 * (logf0_bounds[0] + logf0_bounds[1]),
            0.5 * (logfdot_bounds[0] + logfdot_bounds[1]),
            0.5 * (logA_bounds[0] + logA_bounds[1]),
        ]),
        prior_scale=np.array([
            0.25 * (logf0_bounds[1] - logf0_bounds[0]),
            0.25 * (logfdot_bounds[1] - logfdot_bounds[0]),
            0.25 * (logA_bounds[1] - logA_bounds[0]),
        ]),
        logf0_bounds=logf0_bounds,
        logfdot_bounds=logfdot_bounds,
        logA_bounds=logA_bounds,
    )


def draw_source_prior_and_params(
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, tuple[float, float], tuple[float, float], tuple[float, float]]:
    f0_ref = F0_REF
    delta_f0_half_width = lisa_delta_f0_prior_half_width()
    delta_f0_sigma = lisa_delta_f0_prior_sigma()
    prior_f0 = (
        float(f0_ref - delta_f0_half_width),
        float(f0_ref + delta_f0_half_width),
    )
    prior_fdot = tuple(float(x) for x in FIXED_FDOT_PRIOR_BOUNDS)
    prior_A = tuple(float(x) for x in FIXED_A_PRIOR_BOUNDS)

    delta_f0_true = _draw_truncated_normal(
        rng,
        loc=0.0,
        scale=delta_f0_sigma,
        low=-delta_f0_half_width,
        high=delta_f0_half_width,
    )
    f0 = float(f0_ref + delta_f0_true)
    delta_logf0_true = float(np.log(f0) - np.log(f0_ref))
    fdot = draw_positive_parameter_from_bounds(rng, prior_fdot)
    A = draw_positive_parameter_from_bounds(rng, prior_A)
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    iota = float(np.arccos(rng.uniform(-1.0, 1.0)))
    phi0 = float(rng.uniform(-np.pi, np.pi))
    source = np.array([f0, fdot, A, ra, dec, psi, iota, phi0], dtype=float)
    return source, f0_ref, delta_logf0_true, prior_f0, prior_fdot, prior_A
