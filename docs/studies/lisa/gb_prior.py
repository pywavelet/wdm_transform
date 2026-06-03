"""Shared Galactic-binary prior definitions for the LISA study."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


SOURCE_CATALOG = np.array(
    [
        # [1.35962e-3, 8.94581279e-19, 1.07345e-22, 2.40, 0.31, 3.56, 0.52, 3.06],
        # [1.41220e-3, 2.30000000e-18, 8.20000000e-23, 2.15, 0.18, 1.20, 0.93, 1.40],
        [1.35962e-3, 8.94581279e-15, 1.07345e-20, 2.40, 0.31, 3.56, 0.52, 3.06],
        [1.41220e-3, 2.30000000e-12, 8.20000000e-21, 2.15, 0.18, 1.20, 0.93, 1.40],
    ],
    dtype=float,
)

F0_GLOBAL_BOUNDS = (
    float(SOURCE_CATALOG[:, 0].min() - 1.5e-7),
    float(SOURCE_CATALOG[:, 0].max() + 1.5e-7),
)
FIXED_A_PRIOR_BOUNDS = (6.0e-25, 1.7e-22)
F0_REF = float(np.mean(SOURCE_CATALOG[:, 0]))
DELTA_F0_PRIOR_HALF_WIDTH = float(os.getenv("LISA_DELTA_F0_PRIOR_HALF_WIDTH", "3e-8"))
DELTA_F0_PRIOR_SIGMA = float(os.getenv("LISA_DELTA_F0_PRIOR_SIGMA", "1e-8"))
F0_JITTER_WIDTH = float(np.log1p(DELTA_F0_PRIOR_HALF_WIDTH / F0_REF))

# fdot is a chirp parameter: over a one-year baseline its likelihood is razor-
# sharp on the frequency-derivative resolution scale 1/T_obs^2 (~1e-15 Hz/s),
# far narrower than any plausible astrophysical prior.  A broad (log) prior makes
# the posterior an un-samplable needle, so — exactly like f0 — we treat fdot as a
# localized follow-up parameter: a reference value FDOT_REF plus a small offset
# sampled on the resolution scale.
FDOT_REF = float(os.getenv("LISA_FDOT_REF", "5e-14"))
DELTA_FDOT_PRIOR_HALF_WIDTH = float(os.getenv("LISA_DELTA_FDOT_PRIOR_HALF_WIDTH", "4e-14"))
DELTA_FDOT_PRIOR_SIGMA = float(os.getenv("LISA_DELTA_FDOT_PRIOR_SIGMA", "1e-14"))


@dataclass(frozen=True)
class LocalPriorInfo:
    logA_center: float
    logA_scale: float
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


def build_local_prior_info(*, prior_A: tuple[float, float]) -> LocalPriorInfo:
    logA_bounds = (float(np.log(prior_A[0])), float(np.log(prior_A[1])))
    return LocalPriorInfo(
        logA_center=0.5 * (logA_bounds[0] + logA_bounds[1]),
        logA_scale=0.25 * (logA_bounds[1] - logA_bounds[0]),
        logA_bounds=logA_bounds,
    )


def draw_source_prior_and_params(rng: np.random.Generator) -> dict:
    """Draw one source and the localized prior boxes around it.

    Returns a dict with the source parameter row plus the f0/fdot reference
    values and prior bounds used to set up the follow-up inference.
    """
    f0_ref = F0_REF
    prior_f0 = (
        float(f0_ref - DELTA_F0_PRIOR_HALF_WIDTH),
        float(f0_ref + DELTA_F0_PRIOR_HALF_WIDTH),
    )
    fdot_ref = FDOT_REF
    prior_fdot = (
        float(fdot_ref - DELTA_FDOT_PRIOR_HALF_WIDTH),
        float(fdot_ref + DELTA_FDOT_PRIOR_HALF_WIDTH),
    )
    prior_A = tuple(float(x) for x in FIXED_A_PRIOR_BOUNDS)

    delta_f0_true = _draw_truncated_normal(
        rng, loc=0.0, scale=DELTA_F0_PRIOR_SIGMA,
        low=-DELTA_F0_PRIOR_HALF_WIDTH, high=DELTA_F0_PRIOR_HALF_WIDTH,
    )
    f0 = float(f0_ref + delta_f0_true)
    delta_logf0_true = float(np.log(f0) - np.log(f0_ref))

    delta_fdot_true = _draw_truncated_normal(
        rng, loc=0.0, scale=DELTA_FDOT_PRIOR_SIGMA,
        low=-DELTA_FDOT_PRIOR_HALF_WIDTH, high=DELTA_FDOT_PRIOR_HALF_WIDTH,
    )
    fdot = float(fdot_ref + delta_fdot_true)

    A = draw_positive_parameter_from_bounds(rng, prior_A)
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    iota = float(np.arccos(rng.uniform(-1.0, 1.0)))
    phi0 = float(rng.uniform(-np.pi, np.pi))
    source = np.array([f0, fdot, A, ra, dec, psi, iota, phi0], dtype=float)
    return {
        "source": source,
        "f0_ref": f0_ref,
        "delta_logf0_true": delta_logf0_true,
        "fdot_ref": fdot_ref,
        "delta_fdot_true": delta_fdot_true,
        "prior_f0": prior_f0,
        "prior_fdot": prior_fdot,
        "prior_A": prior_A,
    }
