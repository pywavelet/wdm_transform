from __future__ import annotations

from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
BACKGROUND_DIR = STUDY_DIR / "outdir_gb_background"
FREQ_ASSET_DIR = STUDY_DIR / "lisa_freq_mcmc_assets"
WDM_ASSET_DIR = STUDY_DIR / "lisa_wdm_mcmc_assets"

BACKGROUND_REALIZATION_PATH = BACKGROUND_DIR / "tdi15_background_realization.npz"
RESPONSE_TENSOR_PATH = BACKGROUND_DIR / "Rtildeop_tf.npz"

c = 299792458.0
L_LISA = 2.5e9


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def wrap_phase(phi: float) -> float:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


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
    conv_fact = (4 * np.pi**2) / (3 * H0**2)
    return conv_fact * np.asarray(Sh, dtype=float) * np.asarray(f, dtype=float) ** 3


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
    if np.isscalar(f):
        return float(out)
    return out


def noise_tdi15_a_psd(f, L: float = L_LISA):
    return noise_tdi15_psd(0, f, L=L)
