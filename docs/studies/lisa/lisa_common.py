"""Shared constants, paths, PSD models, and utilities for the LISA GB study."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import corner
import jax
import numpy as np
from gb_prior import (
    F0_GLOBAL_BOUNDS,
    F0_REF,
    FDOT_GLOBAL_BOUNDS,
    FIXED_A_PRIOR_BOUNDS,
    FIXED_FDOT_PRIOR_BOUNDS,
    SOURCE_CATALOG,
    build_local_prior_info,
    draw_source_prior_and_params,
    lisa_f0_jitter_width,
)
from wdm_transform.signal_processing import (
    matched_filter_snr_rfft,
    matched_filter_snr_wdm,
    noise_characteristic_strain,
    rfft_characteristic_strain,
)

STUDY_DIR = Path(__file__).resolve().parent
OUTDIR_ROOT = STUDY_DIR / "outdir_lisa"
CACHE_DIR = OUTDIR_ROOT / "_cache"

c = 299792458.0
L_LISA = 2.5e9

# ── Filesystem helpers ────────────────────────────────────────────────────────


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def lisa_include_galactic() -> bool:
    return os.getenv("LISA_INCLUDE_GALACTIC", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def lisa_seed() -> int:
    return int(os.getenv("LISA_SEED", "0"))


def lisa_mode_dirname(*, include_galactic: bool | None = None) -> str:
    if include_galactic is None:
        include_galactic = lisa_include_galactic()
    return "galactic_background" if include_galactic else "stationary_noise"


def lisa_run_dir(
    *,
    seed: int | None = None,
    include_galactic: bool | None = None,
) -> Path:
    if seed is None:
        seed = lisa_seed()
    return OUTDIR_ROOT / lisa_mode_dirname(include_galactic=include_galactic) / f"seed_{seed}"


RUN_DIR = lisa_run_dir()
RESPONSE_TENSOR_PATH = CACHE_DIR / "Rtildeop_tf.npz"
INJECTION_PATH = RUN_DIR / "injection.npz"
FREQ_POSTERIOR_PATH = RUN_DIR / "freq_posteriors.npz"
WDM_POSTERIOR_PATH = RUN_DIR / "wdm_posteriors.npz"


# ── JAX and matplotlib setup ──────────────────────────────────────────────────


def setup_jax_and_matplotlib() -> None:
    """Configure JAX and matplotlib for the study scripts."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", True)


def save_figure(fig, output_dir: Path, stem: str, *, dpi: int = 160) -> Path:
    ensure_output_dir(output_dir)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return path


# ── Signal utilities ─────────────────────────────────────────────────────────


def wrap_phase(phi):
    """Wrap angle to (-π, π].  Works on scalars, NumPy, and JAX arrays."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def is_phase_parameter(label: str) -> bool:
    """Return whether *label* represents a circular phase-like parameter."""
    lowered = label.lower()
    return "phi" in lowered or "phase" in lowered


def credible_level(samples: np.ndarray, truth: float) -> float:
    """Return the posterior CDF rank evaluated at the injected truth."""
    return float(np.mean(np.asarray(samples, dtype=float) < float(truth)))


def circular_credible_level(samples: np.ndarray, truth: float) -> float:
    """Return a boundary-robust rank for wrapped angular samples.

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


def build_parameter_diagnostics(
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
    *,
    ci: float = 0.9,
) -> list[dict[str, float | bool | None | list[float] | str]]:
    """Return structured posterior diagnostics for each parameter."""
    samples_arr = np.asarray(samples, dtype=float)
    truth_arr = np.asarray(truth, dtype=float).reshape(-1)
    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 + ci) / 2.0

    med = np.median(samples_arr, axis=0)
    mean = np.mean(samples_arr, axis=0)
    std = np.std(samples_arr, axis=0, ddof=0)
    lo = np.percentile(samples_arr, lo_q, axis=0)
    hi = np.percentile(samples_arr, hi_q, axis=0)
    q05 = np.percentile(samples_arr, 5.0, axis=0)
    q95 = np.percentile(samples_arr, 95.0, axis=0)

    diagnostics: list[dict[str, float | bool | None | list[float] | str]] = []
    for idx, label in enumerate(param_names):
        truth_value = float(truth_arr[idx])
        samples_col = samples_arr[:, idx]
        std_value = float(std[idx])
        is_phase = is_phase_parameter(label)
        circular_rank = circular_credible_level(samples_col, truth_value) if is_phase else None
        diagnostics.append(
            {
                "label": label,
                "truth": truth_value,
                "mean": float(mean[idx]),
                "std": std_value,
                "median": float(med[idx]),
                "q05": float(q05[idx]),
                "q95": float(q95[idx]),
                "ci": ci,
                "ci_low": float(lo[idx]),
                "ci_high": float(hi[idx]),
                "covered": bool(lo[idx] <= truth_value <= hi[idx]),
                "credible_level": credible_level(samples_col, truth_value),
                "circular_credible_level": circular_rank,
                "standardized_truth_offset": (
                    float((truth_value - mean[idx]) / std_value) if std_value > 0.0 else None
                ),
            }
        )
    return diagnostics


def require_positive_fdot(source_params: np.ndarray, *, context: str) -> np.ndarray:
    """Return *source_params* after validating strictly positive chirps."""
    params = np.asarray(source_params, dtype=float)
    params_2d = np.atleast_2d(params)
    if np.any(params_2d[:, 1] <= 0.0):
        raise ValueError(
            f"{context} contains non-positive fdot values. "
            "Regenerate the corresponding docs/studies/lisa/outdir_lisa/.../injection.npz "
            "after updating the injection configuration."
        )
    return params


def draw_random_source_params(rng: np.random.Generator) -> np.ndarray:
    """Draw one resolved GB parameter vector from broad study-scale ranges."""
    catalog = np.asarray(SOURCE_CATALOG, dtype=float)
    f0 = float(rng.uniform(catalog[:, 0].min() - 1.5e-5, catalog[:, 0].max() + 1.5e-5))
    fdot = float(np.exp(rng.uniform(np.log(5.0e-19), np.log(4.0e-18))))
    A = float(np.exp(rng.uniform(np.log(6.0e-23), np.log(1.6e-22))))
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    iota = float(np.arccos(rng.uniform(-1.0, 1.0)))
    phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.array([f0, fdot, A, ra, dec, psi, iota, phi0], dtype=float)


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
    prior_f0: tuple[float, float]
    prior_fdot: tuple[float, float]
    prior_A: tuple[float, float]
    source_Af: np.ndarray | None = None
    source_Ef: np.ndarray | None = None
    source_Tf: np.ndarray | None = None


def load_injection(path: Path = INJECTION_PATH) -> InjectionData:
    """Load one seeded LISA injection archive into a typed container."""
    with np.load(path) as inj:
        missing_prior_keys = [
            key
            for key in ("prior_f0", "prior_fdot", "prior_A", "f0_ref", "f0_jitter_width", "delta_logf0_true")
            if key not in inj
        ]
        if missing_prior_keys:
            raise ValueError(
                f"{path} is missing shared injection metadata {missing_prior_keys}. "
                "Regenerate this injection with docs/studies/lisa/data_generation.py."
            )
        source_params = require_positive_fdot(
            np.asarray(inj["source_params"], dtype=float),
            context=str(path),
        )
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
            source_params=np.atleast_2d(np.asarray(source_params, dtype=float)),
            f0_ref=float(np.asarray(inj["f0_ref"]).reshape(-1)[0]),
            f0_jitter_width=float(np.asarray(inj["f0_jitter_width"]).reshape(-1)[0]),
            delta_logf0_true=float(np.asarray(inj["delta_logf0_true"]).reshape(-1)[0]),
            prior_f0=tuple(np.asarray(inj["prior_f0"], dtype=float).reshape(2)),
            prior_fdot=tuple(np.asarray(inj["prior_fdot"], dtype=float).reshape(2)),
            prior_A=tuple(np.asarray(inj["prior_A"], dtype=float).reshape(2)),
            source_Af=np.asarray(inj["source_Af"], dtype=np.complex128) if "source_Af" in inj else None,
            source_Ef=np.asarray(inj["source_Ef"], dtype=np.complex128) if "source_Ef" in inj else None,
            source_Tf=np.asarray(inj["source_Tf"], dtype=np.complex128) if "source_Tf" in inj else None,
        )


def estimate_frequency_peak(
    freqs: np.ndarray,
    data_Af: np.ndarray,
    data_Ef: np.ndarray,
    data_Tf: np.ndarray,
    noise_psd_A: np.ndarray,
    noise_psd_E: np.ndarray,
    noise_psd_T: np.ndarray,
    *,
    prior_f0: tuple[float, float],
) -> float:
    """Estimate the carrier frequency from the whitened A+E+T FFT peak."""
    keep = (freqs >= prior_f0[0]) & (freqs <= prior_f0[1])
    if not np.any(keep):
        return 0.5 * (prior_f0[0] + prior_f0[1])
    score = (
        np.abs(np.asarray(data_Af)[keep]) ** 2 / np.maximum(np.asarray(noise_psd_A)[keep], 1e-60)
        + np.abs(np.asarray(data_Ef)[keep]) ** 2 / np.maximum(np.asarray(noise_psd_E)[keep], 1e-60)
        + np.abs(np.asarray(data_Tf)[keep]) ** 2 / np.maximum(np.asarray(noise_psd_T)[keep], 1e-60)
    )
    return float(np.asarray(freqs)[keep][int(np.argmax(score))])


def build_sampled_source_params(fixed_params: np.ndarray, samples_i: np.ndarray) -> np.ndarray:
    """Expand sampled [f0, fdot, A, phi0, ...] rows into full 8-parameter vectors."""
    samples_full = np.tile(np.asarray(fixed_params, dtype=float), (samples_i.shape[0], 1))
    samples_full[:, 0] = samples_i[:, 0]
    samples_full[:, 1] = samples_i[:, 1]
    samples_full[:, 2] = samples_i[:, 2]
    samples_full[:, 7] = samples_i[:, 3]
    return samples_full


def source_truth_vector(fixed_params: np.ndarray, *, snr: float | None = None) -> np.ndarray:
    """Truth vector matching the local posterior column convention."""
    truth = [
        float(fixed_params[0]),
        float(fixed_params[1]),
        float(fixed_params[2]),
        float(wrap_phase(fixed_params[7])),
    ]
    if snr is not None:
        truth.append(float(snr))
    return np.asarray(truth, dtype=float)


def save_posterior_archive(
    output_path: Path,
    *,
    source_params: np.ndarray,
    all_samples: list[np.ndarray],
    snr_optimal: list[float],
    labels: list[str] | None = None,
    truth: np.ndarray | None = None,
) -> Path:
    """Write the shared posterior NPZ layout used by the study scripts."""
    ensure_output_dir(output_path.parent)
    source_params = np.atleast_2d(np.asarray(source_params, dtype=float))
    archive_data: dict[str, np.ndarray] = {
        "source_params": source_params,
        "snr_optimal": np.asarray(snr_optimal, dtype=float),
    }
    if labels is not None:
        archive_data["labels"] = np.asarray(labels, dtype=str)
    if truth is not None:
        archive_data["truth"] = np.asarray(truth, dtype=float)
    if all_samples:
        archive_data["samples_source"] = np.asarray(all_samples[0], dtype=float)
    np.savez(
        output_path,
        **archive_data,
    )
    return output_path


def load_posterior_samples_source(path: Path) -> np.ndarray:
    """Load the primary sampled source array from a study posterior archive."""
    with np.load(path) as archive:
        if "samples_source" not in archive:
            raise KeyError(f"{path} does not contain 'samples_source'.")
        return np.asarray(archive["samples_source"], dtype=float)


def print_cross_domain_diagnostics(
    *,
    labels: list[str],
    truth: np.ndarray,
    wdm_mean: np.ndarray,
    freq_mean: np.ndarray | None = None,
    truth_snr_frequency: float | None = None,
    truth_snr_wdm: float | None = None,
    wdm_loglike_truth: float | None = None,
    wdm_loglike_posterior_mean: float | None = None,
) -> None:
    """Print a compact WDM vs frequency comparison report for one source."""
    print("\nCross-domain diagnostics:")

    if truth_snr_frequency is not None and truth_snr_wdm is not None:
        delta = float(truth_snr_wdm - truth_snr_frequency)
        print(
            "  Truth SNR: "
            f"Frequency={truth_snr_frequency:.6f}, "
            f"WDM={truth_snr_wdm:.6f}, "
            f"delta={delta:+.3e}"
        )

    if wdm_loglike_truth is not None and wdm_loglike_posterior_mean is not None:
        delta = float(wdm_loglike_truth - wdm_loglike_posterior_mean)
        print(
            "  WDM log-likelihood: "
            f"truth={wdm_loglike_truth:.6f}, "
            f"posterior_mean={wdm_loglike_posterior_mean:.6f}, "
            f"truth_minus_mean={delta:+.6f}"
        )

    if freq_mean is not None:
        print("  Posterior-mean deltas (WDM - Frequency):")
        for label, wdm_value, freq_value in zip(labels, wdm_mean, freq_mean):
            delta = float(wdm_value - freq_value)
            if "phi0" in label:
                delta = float(wrap_phase(delta))
            print(
                f"    {label}: "
                f"WDM={float(wdm_value):.6e}, "
                f"Frequency={float(freq_value):.6e}, "
                f"delta={delta:+.6e}"
            )

    truth_compare = truth[: len(wdm_mean)]
    print("  Posterior-mean deltas (WDM - Truth):")
    for label, wdm_value, truth_value in zip(labels, wdm_mean, truth_compare):
        delta = float(wdm_value - truth_value)
        if "phi0" in label:
            delta = float(wrap_phase(delta))
        print(
            f"    {label}: "
            f"WDM={float(wdm_value):.6e}, "
            f"Truth={float(truth_value):.6e}, "
            f"delta={delta:+.6e}"
        )




def floor_pow2(n: int) -> int:
    """Largest power of 2 that is ≤ n."""
    return 1 << (int(n).bit_length() - 1)


def place_local_tdi(segment, kmin: int, n_freqs: int) -> np.ndarray:
    """Place a band-limited TDI segment into a zero-padded full-length frequency array."""
    full = np.zeros(n_freqs, dtype=np.complex128)
    seg = np.asarray(segment, dtype=np.complex128).reshape(-1)
    end = min(kmin + seg.size, n_freqs)
    if end > kmin:
        full[kmin:end] = seg[: end - kmin]
    return full


# ── Frequency-axis models ─────────────────────────────────────────────────────


def interp_psd_channels(
    target_freqs: np.ndarray,
    source_freqs: np.ndarray,
    source_psd_channels: np.ndarray,
) -> np.ndarray:
    """Interpolate PSD from source to target frequency grid on (n_channels, n_freqs).

    Args:
        target_freqs: Target frequency grid (1D).
        source_freqs: Source frequency grid (1D).
        source_psd_channels: PSD array of shape (n_channels, len(source_freqs)).

    Returns:
        Interpolated PSD of shape (n_channels, len(target_freqs)), floored at 1e-60.
    """
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


def freqs_gal(
    nfreqs: int = 500,
    fmin_gal: float = 1e-4,
    fmax_gal: float = 3e-3,
) -> np.ndarray:
    return np.linspace(fmin_gal, fmax_gal, nfreqs)


def galactic_psd(
    f,
    Tobsyr: float = 2.0,
    A_gal: float = 10**-43.9,
    alp_gal: float = 1.8,
    a1_gal: float = -0.25,
    b1_gal: float = -2.7,
    ak_gal: float = -0.27,
    bk_gal: float = -2.47,
    f2_gal: float = 10**-3.5,
):
    f = np.asarray(f, dtype=float)
    f_safe = np.where(f > 0.0, f, 1.0)
    f1_gal = 10 ** (a1_gal * np.log10(Tobsyr) + b1_gal)
    fknee_gal = 10 ** (ak_gal * np.log10(Tobsyr) + bk_gal)
    return (
        A_gal
        * f_safe ** (-7.0 / 3.0)
        * np.exp(-((f_safe / f1_gal) ** alp_gal))
        * (1.0 + np.tanh((fknee_gal - f_safe) / f2_gal))
    )


def omega_gw(f, Sh):
    H0 = 2.2e-18
    return (4 * np.pi**2) / (3 * H0**2) * np.asarray(Sh) * np.asarray(f) ** 3


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
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros_like(f_arr, dtype=float)
    pos = f_arr > 0.0
    if np.any(pos):
        base = _ntilda_t if channel == 2 else _ntilda_e
        out[pos] = base(f_arr[pos], L=L) * tdi15_factor(f_arr[pos], L=L)
    return float(out) if np.isscalar(f) else out


def noise_tdi15_a_psd(f, L: float = L_LISA):
    return noise_tdi15_psd(0, f, L=L)


def build_total_noise_psd(
    Rtildeop_tf: np.ndarray,
    freqs_response: np.ndarray,
    target_freqs: np.ndarray,
    channel: int = 0,
) -> np.ndarray:
    """Instrument noise + time-averaged galactic foreground PSD on *target_freqs*."""
    fg_time = np.abs(Rtildeop_tf[channel, channel]) * tdi15_factor(freqs_response)
    fg_mean = np.mean(fg_time, axis=0)
    fg_interp = np.interp(target_freqs, freqs_response, fg_mean, left=0.0, right=0.0)
    return np.maximum(noise_tdi15_psd(channel, target_freqs) + fg_interp, 1e-60)


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


# ── Posterior diagnostics ─────────────────────────────────────────────────────


def print_posterior_summary(
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
) -> None:
    """Print median and 90 % CI for each parameter alongside the true value."""
    med = np.median(samples, axis=0)
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    header = f"{'Parameter':<24} {'True':>12} {'Median':>12} {'5%':>12} {'95%':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, true_v, median, low, high in zip(param_names, truth, med, lo, hi):
        print(
            f"  {name:<22} {true_v:>12.4e} {median:>12.4e} {low:>12.4e} {high:>12.4e}"
        )


def check_posterior_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
    ci: float = 0.9,
) -> np.ndarray:
    """Print CI coverage and return boolean mask of covered parameters."""
    lo = np.percentile(samples, 100.0 * (1 - ci) / 2, axis=0)
    hi = np.percentile(samples, 100.0 * (1 + ci) / 2, axis=0)
    covered = (truth >= lo) & (truth <= hi)
    print(f"\n{ci * 100:.0f}% posterior coverage:")
    for name, true_v, low, high, in_ci in zip(param_names, truth, lo, hi, covered):
        mark = "✓" if in_ci else "✗"
        print(f"  {mark} {name:<22}: true={true_v:+.4e}  [{low:+.4e}, {high:+.4e}]")
    print(f"  {int(covered.sum())}/{len(covered)} parameters covered")
    return covered


def compute_template_injection_overlap(
    template_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    injection_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    noise_psd_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    freqs: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    """Compute A/E/T template-injection overlap at true parameters.

    Returns the overlap for each channel and the combined SNR-weighted overlap.
    The overlap should be ≈ 1 to noise level for a valid template-injection pair.

    Args:
        template_aet: Template A/E/T frequency series (3-tuple)
        injection_aet: Injection A/E/T frequency series (3-tuple)
        noise_psd_aet: Noise PSD for A/E/T channels (3-tuple)
        freqs: Frequency grid
        dt: Time spacing

    Returns:
        overlaps_per_channel: Shape (3,) overlap for A/E/T
        combined_overlap: SNR-weighted combined overlap
    """
    overlaps_per_channel = np.zeros(3)
    snrs_template = np.zeros(3)
    snrs_injection = np.zeros(3)

    for channel in range(3):
        template = np.asarray(template_aet[channel], dtype=complex)
        injection = np.asarray(injection_aet[channel], dtype=complex)
        psd = np.asarray(noise_psd_aet[channel], dtype=float)
        pos = np.asarray(freqs, dtype=float) > 0.0

        # Compute matched filter SNRs and cross-correlation
        snr_template = matched_filter_snr_rfft(template, psd, freqs, dt=dt)
        snr_injection = matched_filter_snr_rfft(injection, psd, freqs, dt=dt)

        # Inner product <template|injection>
        if pos.sum() >= 2:
            df = float(np.asarray(freqs, dtype=float)[pos][1] - np.asarray(freqs, dtype=float)[pos][0])
            h_template = dt * template[pos]
            h_injection = dt * injection[pos]
            inner_product = 4.0 * df * np.real(
                np.sum(np.conj(h_template) * h_injection / np.maximum(psd[pos], 1e-60))
            )
        else:
            inner_product = 0.0

        # Overlap = <h1|h2> / sqrt(<h1|h1> * <h2|h2>)
        if snr_template > 0.0 and snr_injection > 0.0:
            overlaps_per_channel[channel] = inner_product / (snr_template * snr_injection)
        else:
            overlaps_per_channel[channel] = 0.0

        snrs_template[channel] = snr_template
        snrs_injection[channel] = snr_injection

    # Combined overlap weighted by SNR^2
    total_snr2_template = np.sum(snrs_template**2)
    total_snr2_injection = np.sum(snrs_injection**2)

    if total_snr2_template > 0.0 and total_snr2_injection > 0.0:
        # Weight by SNR^2 contribution from each channel
        weights = (snrs_template * snrs_injection) / np.sqrt(total_snr2_template * total_snr2_injection)
        combined_overlap = np.sum(weights * overlaps_per_channel)
    else:
        combined_overlap = 0.0

    return overlaps_per_channel, float(combined_overlap)


def check_template_injection_sanity(
    template_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    injection_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    noise_psd_aet: tuple[np.ndarray, np.ndarray, np.ndarray],
    freqs: np.ndarray,
    dt: float,
    *,
    overlap_threshold: float = 0.95,
    context: str = "Template-injection",
) -> bool:
    """Sanity check: template-injection overlap should be ≈ 1.

    Args:
        template_aet: Template A/E/T frequency series (3-tuple)
        injection_aet: Injection A/E/T frequency series (3-tuple)
        noise_psd_aet: Noise PSD for A/E/T channels (3-tuple)
        freqs: Frequency grid
        dt: Time spacing
        overlap_threshold: Minimum acceptable overlap (default 0.95)
        context: Description for logging

    Returns:
        True if overlap check passes, False otherwise
    """
    overlaps_per_channel, combined_overlap = compute_template_injection_overlap(
        template_aet, injection_aet, noise_psd_aet, freqs, dt
    )

    print(f"\n{context} overlap sanity check:")
    channel_names = ["A", "E", "T"]
    for i, (name, overlap) in enumerate(zip(channel_names, overlaps_per_channel)):
        status = "✓" if overlap >= overlap_threshold else "✗"
        print(f"  {status} Channel {name}: overlap = {overlap:.6f}")

    status = "✓" if combined_overlap >= overlap_threshold else "✗"
    print(f"  {status} Combined:   overlap = {combined_overlap:.6f}")

    all_pass = np.all(overlaps_per_channel >= overlap_threshold) and combined_overlap >= overlap_threshold

    if not all_pass:
        print(f"  WARNING: Template-injection overlap below threshold {overlap_threshold:.3f}")
        print(f"  This suggests potential issues with:")
        print(f"    - Template generation at true parameters")
        print(f"    - Injection data consistency")
        print(f"    - Numerical precision in frequency domain transforms")
        return False
    else:
        print(f"  Template-injection overlap check PASSED (≥ {overlap_threshold:.3f})")
        return True


def print_posterior_report(
    title: str,
    samples: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
    *,
    ci: float = 0.9,
) -> np.ndarray:
    """Print a compact posterior summary plus CI coverage block."""
    print(f"\n{'═' * 56}  {title}")
    print_posterior_summary(samples, truth, param_names)
    return check_posterior_coverage(samples, truth, param_names, ci=ci)
