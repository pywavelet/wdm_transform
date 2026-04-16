"""Shared constants, paths, PSD models, and utilities for the LISA GB study."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import corner
import numpy as np
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

# Reference GB source parameters: [f0, fdot, A, ra, dec, psi, iota, phi0]
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
FIXED_F0_PRIOR_BOUNDS = F0_GLOBAL_BOUNDS
FIXED_FDOT_PRIOR_BOUNDS = FDOT_GLOBAL_BOUNDS
FIXED_A_PRIOR_BOUNDS = (6.0e-24, 1.7e-23)


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
class LocalPriorInfo:
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logf0_bounds: tuple[float, float]
    logfdot_bounds: tuple[float, float]
    logA_bounds: tuple[float, float]


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
    prior_f0: tuple[float, float]
    prior_fdot: tuple[float, float]
    prior_A: tuple[float, float]


def load_injection(path: Path = INJECTION_PATH) -> InjectionData:
    """Load one seeded LISA injection archive into a typed container."""
    with np.load(path) as inj:
        missing_prior_keys = [key for key in ("prior_f0", "prior_fdot", "prior_A") if key not in inj]
        if missing_prior_keys:
            raise ValueError(
                f"{path} is missing shared prior metadata {missing_prior_keys}. "
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
            prior_f0=tuple(np.asarray(inj["prior_f0"], dtype=float).reshape(2)),
            prior_fdot=tuple(np.asarray(inj["prior_fdot"], dtype=float).reshape(2)),
            prior_A=tuple(np.asarray(inj["prior_A"], dtype=float).reshape(2)),
        )


def _draw_truncated_normal(
    rng: np.random.Generator,
    *,
    loc: float,
    scale: float,
    low: float,
    high: float,
) -> float:
    """Sample a scalar truncated normal by rejection."""
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
    """Draw a positive parameter from the shared truncated-normal log prior."""
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
    """Shared log-parameter prior metadata for the local frequency/WDM fits."""
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
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Draw one source from the fixed generation/analysis priors."""
    prior_f0 = tuple(float(x) for x in FIXED_F0_PRIOR_BOUNDS)
    prior_fdot = tuple(float(x) for x in FIXED_FDOT_PRIOR_BOUNDS)
    prior_A = tuple(float(x) for x in FIXED_A_PRIOR_BOUNDS)

    f0 = draw_positive_parameter_from_bounds(rng, prior_f0)
    fdot = draw_positive_parameter_from_bounds(rng, prior_fdot)
    A = draw_positive_parameter_from_bounds(rng, prior_A)
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    iota = float(np.arccos(rng.uniform(-1.0, 1.0)))
    phi0 = float(rng.uniform(-np.pi, np.pi))
    source = np.array(
        [f0, fdot, A, ra, dec, psi, iota, phi0],
        dtype=float,
    )
    return source, prior_f0, prior_fdot, prior_A


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


def save_corner_plot(
    samples: np.ndarray,
    *,
    truth: np.ndarray,
    output_dir: Path,
    stem: str,
    labels: list[str],
) -> Path:
    """Render and save one corner plot using the study defaults."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] <= samples.shape[1]:
        return output_dir / f"{stem}.png"
    fig = corner.corner(
        samples,
        labels=labels,
        truths=np.asarray(truth, dtype=float),
        truth_color="black",
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    return save_figure(fig, output_dir, stem)


def save_corner_plot_dual(
    samples_primary: np.ndarray,
    samples_secondary: np.ndarray | None,
    *,
    truth: np.ndarray,
    output_dir: Path,
    primary_name: str,
    secondary_name: str,
    labels: list[str],
) -> Path:
    """Render corner plot with unified colors: freq (blue), WDM (orange).

    Args:
        samples_primary: Primary posterior samples (freq if freq is running, else WDM)
        samples_secondary: Secondary posterior samples (WDM if freq is running, else freq)
        truth: Injected truth vector
        output_dir: Output directory
        primary_name: Name of primary domain ("freq" or "wdm")
        secondary_name: Name of secondary domain ("wdm" or "freq")
        labels: Parameter labels

    Returns:
        Path to saved corner.png
    """
    samples_primary = np.asarray(samples_primary, dtype=float)
    if samples_primary.ndim != 2 or samples_primary.shape[0] <= samples_primary.shape[1]:
        return output_dir / "corner.png"

    # Color scheme: freq=blue (C0), wdm=orange (C1)
    primary_color = "C0" if primary_name == "freq" else "C1"

    # Create corner plot with primary posterior
    fig = corner.corner(
        samples_primary,
        labels=labels,
        truths=np.asarray(truth, dtype=float),
        truth_color="black",
        truth_linewidth=2,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color=primary_color,
        fill_contours=False,
        plot_density=False,
    )

    # Overlay secondary if available
    if samples_secondary is not None:
        samples_secondary = np.asarray(samples_secondary, dtype=float)
        if samples_secondary.ndim == 2 and samples_secondary.shape[0] > samples_secondary.shape[1]:
            secondary_color = "C0" if secondary_name == "freq" else "C1"
            corner.overplot_lines(fig, samples_secondary, color=secondary_color, alpha=0.5, linewidth=1.5)

    return save_figure(fig, output_dir, "corner")


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
