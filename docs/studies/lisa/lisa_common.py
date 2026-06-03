"""Shared helpers for the stationary-noise LISA galactic-binary study.

This module holds the pieces that are reused across the runner
(`lisa_study.py`) and the aggregation script (`collect_all_results.py`):

* run-directory / output-path conventions,
* the stationary TDI-1.5 instrument-noise PSD model,
* a physically-normalised rFFT noise draw,
* injection (de)serialisation, and
* small posterior-rank diagnostics used to build PP plots.

The galactic confusion foreground has been removed; the only stochastic
component is stationary instrument noise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
OUTDIR_ROOT = STUDY_DIR / "outdir_lisa"
CACHE_DIR = OUTDIR_ROOT / "_cache"

c = 299792458.0
L_LISA = 2.5e9

# ── Filesystem / run-directory conventions ────────────────────────────────────


def lisa_run_dir(*, seed: int | None = None) -> Path:
    if seed is None:
        seed = int(os.getenv("LISA_SEED", "0"))
    return OUTDIR_ROOT / f"seed_{seed}"


RUN_DIR = lisa_run_dir()
INJECTION_PATH = RUN_DIR / "injection.npz"


# ── JAX and matplotlib setup ──────────────────────────────────────────────────


def setup_jax_and_matplotlib() -> None:
    """Configure JAX (float64) and a writable matplotlib config dir."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", True)


def save_figure(fig, output_dir: Path, stem: str, *, dpi: int = 160) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return path


# ── Phase / posterior-rank utilities (used for PP plots) ──────────────────────


def wrap_phase(phi):
    """Wrap angle to (-π, π].  Works on scalars, NumPy, and JAX arrays."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def is_phase_parameter(label: str) -> bool:
    """Return whether *label* represents a circular phase-like parameter."""
    lowered = label.lower()
    return "phi" in lowered or "phase" in lowered


def credible_level(samples: np.ndarray, truth: float) -> float:
    """Posterior CDF rank evaluated at the injected truth (for PP plots)."""
    return float(np.mean(np.asarray(samples, dtype=float) < float(truth)))


def circular_credible_level(samples: np.ndarray, truth: float) -> float:
    """Boundary-robust posterior rank for a wrapped angular parameter.

    The posterior is unwrapped on a cut opposite its circular mean so values
    concentrated near ±π are ordered consistently before evaluating the rank.
    """
    samples_arr = np.asarray(samples, dtype=float).reshape(-1)
    if samples_arr.size == 0:
        return float("nan")
    sin_mean = float(np.mean(np.sin(samples_arr)))
    cos_mean = float(np.mean(np.cos(samples_arr)))
    mean_angle = float(np.arctan2(sin_mean, cos_mean))
    cut = float(wrap_phase(mean_angle + np.pi))
    samples_unwrapped = np.mod(samples_arr - cut, 2.0 * np.pi)
    truth_unwrapped = float(np.mod(float(truth) - cut, 2.0 * np.pi))
    return float(np.mean(samples_unwrapped < truth_unwrapped))


def posterior_rank(samples: np.ndarray, truth: float, label: str) -> float:
    """Posterior rank of *truth*, choosing the circular variant for phases."""
    if is_phase_parameter(label):
        return circular_credible_level(samples, truth)
    return credible_level(samples, truth)


# ── Frequency-axis models ─────────────────────────────────────────────────────


def freqs_analysis(
    nfreqs: int = 500,
    fmin: float = 1e-4,
    fmax: float = 3e-3,
) -> np.ndarray:
    """LISA analysis frequency band used to set the sampling grid."""
    return np.linspace(fmin, fmax, nfreqs)


def _ntilda_e(f, A: float = 3.0, P: float = 15.0, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0, f, 1.0)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return (
        0.5
        * (2.0 + np.cos(f_safe / fstar))
        * (P / L) ** 2
        * 1e-24
        * (1.0 + (0.002 / f_safe) ** 4)
        + 2.0
        * (1.0 + np.cos(f_safe / fstar) + np.cos(f_safe / fstar) ** 2)
        * (A / L) ** 2
        * 1e-30
        * (1.0 + (0.0004 / f_safe) ** 2)
        * (1.0 + (f_safe / 0.008) ** 4)
        * (1.0 / (2.0 * np.pi * f_safe)) ** 4
    )


def _ntilda_t(f, A: float = 3.0, P: float = 15.0, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0, f, 1.0)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return (1.0 - np.cos(f_safe / fstar)) * (P / L) ** 2 * 1e-24 * (
        1.0 + (0.002 / f_safe) ** 4
    ) + 2.0 * (1.0 - np.cos(f_safe / fstar)) ** 2 * (A / L) ** 2 * 1e-30 * (
        1.0 + (0.0004 / f_safe) ** 2
    ) * (1.0 + (f_safe / 0.008) ** 4) * (1.0 / (2.0 * np.pi * f_safe)) ** 4


def tdi15_factor(f, L: float = L_LISA):
    f = np.asarray(f, dtype=float)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return 4.0 * np.sin(f / fstar) * f / fstar


def noise_tdi15_psd(channel: int, f, L: float = L_LISA):
    """Stationary TDI-1.5 instrument-noise PSD for channel 0=A, 1=E, 2=T."""
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros_like(f_arr, dtype=float)
    pos = f_arr > 0.0
    if np.any(pos):
        base = _ntilda_t if channel == 2 else _ntilda_e
        out[pos] = base(f_arr[pos], L=L) * tdi15_factor(f_arr[pos], L=L)
    return float(out) if np.isscalar(f) else out


def draw_rfft_from_psd(
    psd: np.ndarray,
    *,
    rng: np.random.Generator,
    df: float,
    dt: float,
) -> np.ndarray:
    """Draw one-sided rFFT coefficients from a physical one-sided PSD.

    Uses the convention ``E[2 df dt² |X_k|²] = S_k`` for interior bins so that
    ``irfft`` produces a time series with the requested PSD.
    """
    if df <= 0.0:
        raise ValueError("df must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    psd = np.asarray(psd, dtype=float)
    white = rng.normal(size=psd.shape) + 1j * rng.normal(size=psd.shape)
    coeffs = np.sqrt(np.maximum(psd, 0.0) / (4.0 * df * dt**2)) * white
    if coeffs.size:
        coeffs[0] = np.sqrt(2) * coeffs[0].real
        coeffs[-1] = np.sqrt(2) * coeffs[-1].real
    return coeffs


def interp_psd_channels(
    target_freqs: np.ndarray,
    source_freqs: np.ndarray,
    source_psd_channels: np.ndarray,
) -> np.ndarray:
    """Interpolate a ``(n_channels, n_freqs)`` PSD onto *target_freqs*."""
    return np.maximum(
        np.stack(
            [
                np.interp(
                    target_freqs,
                    source_freqs,
                    psd,
                    left=psd[0],
                    right=psd[-1],
                )
                for psd in source_psd_channels
            ],
            axis=0,
        ),
        1e-60,
    )


def trim_frequency_band(
    freqs: np.ndarray,
    f_lo: float,
    f_hi: float,
    pad_bins: int = 2,
) -> slice:
    """Slice of *freqs* grid covering [f_lo, f_hi] with optional padding bins."""
    keep = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    if keep.size == 0:
        raise ValueError(f"No bins in [{f_lo:.3e}, {f_hi:.3e}] Hz.")
    return slice(
        max(int(keep[0]) - pad_bins, 0),
        min(int(keep[-1]) + pad_bins + 1, len(freqs)),
    )


def place_local_tdi(segment, kmin: int, n_freqs: int) -> np.ndarray:
    """Embed a band-limited TDI segment into a zero-padded full-length array."""
    full = np.zeros(n_freqs, dtype=np.complex128)
    seg = np.asarray(segment, dtype=np.complex128).reshape(-1)
    end = min(kmin + seg.size, n_freqs)
    if end > kmin:
        full[kmin:end] = seg[: end - kmin]
    return full


# ── Injection container ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class InjectionData:
    dt: float
    t_obs: float
    seed: int
    data_At: np.ndarray
    data_Et: np.ndarray
    data_Tt: np.ndarray
    noise_psd_A: np.ndarray
    noise_psd_E: np.ndarray
    noise_psd_T: np.ndarray
    freqs: np.ndarray
    source_params: np.ndarray
    f0_ref: float
    f0_jitter_width: float
    delta_logf0_true: float
    fdot_ref: float
    delta_fdot_true: float
    prior_f0: tuple[float, float]
    prior_fdot: tuple[float, float]
    prior_A: tuple[float, float]
    source_Af: np.ndarray | None = None
    source_Ef: np.ndarray | None = None
    source_Tf: np.ndarray | None = None


def load_injection(path: Path = INJECTION_PATH) -> InjectionData:
    """Load one seeded LISA injection archive into a typed container."""
    with np.load(path) as inj:
        return InjectionData(
            dt=float(inj["dt"]),
            t_obs=float(inj["t_obs"]),
            seed=int(np.asarray(inj["seed"]).reshape(-1)[0]) if "seed" in inj else 0,
            data_At=np.asarray(inj["data_At"], dtype=float),
            data_Et=np.asarray(inj["data_Et"], dtype=float),
            data_Tt=np.asarray(inj["data_Tt"], dtype=float),
            noise_psd_A=np.asarray(inj["noise_psd_A"], dtype=float),
            noise_psd_E=np.asarray(inj["noise_psd_E"], dtype=float),
            noise_psd_T=np.asarray(inj["noise_psd_T"], dtype=float),
            freqs=np.asarray(inj["freqs"], dtype=float),
            source_params=np.atleast_2d(np.asarray(inj["source_params"], dtype=float)),
            f0_ref=float(np.asarray(inj["f0_ref"]).reshape(-1)[0]),
            f0_jitter_width=float(np.asarray(inj["f0_jitter_width"]).reshape(-1)[0]),
            delta_logf0_true=float(np.asarray(inj["delta_logf0_true"]).reshape(-1)[0]),
            fdot_ref=float(np.asarray(inj["fdot_ref"]).reshape(-1)[0]),
            delta_fdot_true=float(np.asarray(inj["delta_fdot_true"]).reshape(-1)[0]),
            prior_f0=tuple(np.asarray(inj["prior_f0"], dtype=float).reshape(2)),
            prior_fdot=tuple(np.asarray(inj["prior_fdot"], dtype=float).reshape(2)),
            prior_A=tuple(np.asarray(inj["prior_A"], dtype=float).reshape(2)),
            source_Af=np.asarray(inj["source_Af"], dtype=np.complex128) if "source_Af" in inj else None,
            source_Ef=np.asarray(inj["source_Ef"], dtype=np.complex128) if "source_Ef" in inj else None,
            source_Tf=np.asarray(inj["source_Tf"], dtype=np.complex128) if "source_Tf" in inj else None,
        )
