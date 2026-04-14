"""Galactic-binary foreground simulation for LISA/Taiji-like channels.

Pipeline:
1. Build a Galactic foreground PSD model and sky morphology map.
2. Compute the sky-averaged detector-response tensor (cached to disk).
3. Draw a full-year A/E/T realization of instrument noise + stochastic foreground.
4. Inject resolved Galactic binaries (JaxGB) into the realization.
5. Compute and print per-source matched-filter SNR (A+E+T channels).
6. Save ``injection.npz`` and diagnostic plots.
"""
import atexit
import os
import shutil
import time
from multiprocessing import cpu_count, get_context

import healpy as hp
import lisaorbits
import numpy as np
from lisa_common import (
    CACHE_DIR,
    INJECTION_PATH,
    L_LISA,
    RESPONSE_TENSOR_PATH,
    RUN_DIR,
    build_total_noise_psd,
    c,
    draw_positive_parameter_from_bounds,
    draw_source_prior_and_params,
    ensure_output_dir,
    freqs_gal,
    galactic_psd,
    noise_tdi15_psd,
    omega_gw,
    place_local_tdi,
    save_figure,
    tdi15_factor,
)
from matplotlib import pyplot as plt
from numpy.fft import irfft, rfft, rfftfreq
from scipy.integrate import simpson
from tqdm.auto import tqdm
from wdm_transform.signal_processing import (
    matched_filter_snr_rfft,
    noise_characteristic_strain,
    rfft_characteristic_strain,
)

OUTDIR = RUN_DIR
_SCRIPT_START = time.perf_counter()
INCLUDE_GALACTIC = os.getenv("LISA_INCLUDE_GALACTIC", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RNG_SEED = int(os.getenv("LISA_SEED", "0"))
TARGET_SNR_MIN = 10.0
TARGET_SNR_MAX = 50.0


def _save_plot_once_to_cache(stem: str, draw_plot) -> None:
    """Generate a static diagnostic plot once and mirror it into the run directory."""
    cache_path = CACHE_DIR / f"{stem}.png"
    run_path = OUTDIR / f"{stem}.png"

    if cache_path.exists():
        if not run_path.exists():
            ensure_output_dir(run_path.parent)
            shutil.copy2(cache_path, run_path)
        print(f"Reusing cached plot: {cache_path}")
        return

    draw_plot()
    save_figure(plt.gcf(), CACHE_DIR, stem)
    ensure_output_dir(run_path.parent)
    shutil.copy2(cache_path, run_path)
    print(f"Saved cached plot: {cache_path}")


def _print_runtime() -> None:
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"\n[data_generation.py] runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


atexit.register(_print_runtime)

# Number of orbital time steps for the response-tensor computation.
# Workers read this constant at import time — it must not change.
NTIMES_RESPONSE = 90

# ── Module-level globals populated inside main() ──────────────────────────────
# These exist at module scope so that fork-based multiprocessing workers inherit
# them without pickling large arrays.  Never write to these outside main().
frequencies = None       # frequency grid for response-tensor integration
S_gal = None             # galactic strain PSD on *frequencies*
map_Gal = None           # HEALPix sky map weights
npix = None              # number of HEALPix pixels
kappas_import = None     # unit sky-direction vectors, shape (3, npix)
x_all = None             # spacecraft positions, shape (3, ntimes, 3) metres
l_hat_ij_all = None      # unit arm vectors (3, 3, ntimes, 3)
Rtildeop_tf_times_H = None  # A/E/T response tensor (3, 3, ntimes, nfreqs)
freqs_cut = None         # chunk rFFT frequencies inside analysis band
freqs_chunk = None       # full chunk rFFT frequencies
idx = None               # index where freqs_chunk enters analysis band
e_ab_arrL = None         # left-circular polarisation tensor projected on sky
e_ab_arrR = None         # right-circular polarisation tensor projected on sky
seed_base = 0            # base RNG seed for deterministic draws


# ── Galactic morphology ───────────────────────────────────────────────────────

nside = 16
x_s = 8.0


def rho_gal(x, y, z, rho_0=1.0, A=0.25, R_b=0.5, R_d=2.5, Z_d=0.2):
    r = np.sqrt(x**2 + y**2 + z**2)
    u = np.sqrt(x**2 + y**2)
    z_scaled = np.clip(z / Z_d, -50.0, 50.0)
    sech2 = 1.0 / np.cosh(z_scaled) ** 2
    return rho_0 * (A * np.exp(-(r**2) / R_b**2) + (1.0 - A) * np.exp(-u / R_d) * sech2)


def position_gal(distance, theta, phi):
    x = x_s + distance * np.sin(theta) * np.cos(phi)
    y = distance * np.sin(theta) * np.sin(phi)
    z = distance * np.cos(theta)
    return x, y, z


# ── LISA geometry helpers ─────────────────────────────────────────────────────

TS = 365 * 24 * 3600


def nhat(theta, phi):
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def lisaorbits_positions(times):
    """Spacecraft positions from the ``lisaorbits`` equal-arm model, in metres."""
    orbit = lisaorbits.EqualArmlengthOrbits()
    if hasattr(orbit, "compute_position"):
        pos = np.asarray(
            orbit.compute_position(times, sc=np.array([1, 2, 3])), dtype=float
        )
        return np.transpose(pos, (1, 0, 2))
    positions = np.zeros((3, len(times), 3), dtype=float)
    for sc in range(3):
        try:
            pos = np.asarray(orbit.compute_spacecraft_position(sc, times), dtype=float)
        except Exception:
            pos = np.asarray(
                orbit.compute_spacecraft_position(sc + 1, times), dtype=float
            )
        positions[sc] = pos.T if pos.shape[0] == 3 else pos
    return positions


def get_theta_phi(vector):
    return np.arccos(vector[2]), np.arctan2(vector[1], vector[0])


def phat(theta, phi):
    return np.stack([np.sin(phi), -np.cos(phi), 0 * phi])


def qhat(theta, phi):
    return np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )


def tensor_ab(a, b):
    return np.einsum("ik,jk->ijk", a, b)


def contractor_ab_a_b(ab, a, b):
    return np.einsum("ijk,ik,jk->k", ab, a, b)


def scal_prod(a, b):
    return np.einsum("ij,ij->j", a, b)


def e_plus(theta, phi):
    return (
        tensor_ab(phat(theta, phi), phat(theta, phi))
        - tensor_ab(qhat(theta, phi), qhat(theta, phi))
    ) / np.sqrt(2)


def e_cross(theta, phi):
    return (
        tensor_ab(phat(theta, phi), qhat(theta, phi))
        + tensor_ab(qhat(theta, phi), phat(theta, phi))
    ) / np.sqrt(2)


def e_ab(kap, lam):
    return (e_plus(*get_theta_phi(kap)) + 1j * lam * e_cross(*get_theta_phi(kap))) / np.sqrt(2)


def M_one(t, f, kap, i, j):
    phas = L_LISA * f * (1 + scal_prod(kap, l_hat_ij_all[i, j, t][:, None])) / c
    return np.exp(np.pi * 1j * phas) * np.sinc(phas)


def T_one(t, f, kap, i, j):
    fstar = 1 / (2 * np.pi * L_LISA / c)
    return np.exp(-1j * f / fstar) * M_one(t, f, kap, j, i) + np.exp(
        -1j * f * scal_prod(kap, l_hat_ij_all[i, j, t][:, None]) / fstar
    ) * M_one(t, f, kap, i, j)


def _response_leg_diff(t, f, kap, i, j, k, g_arr):
    arm_ij = l_hat_ij_all[i, j, t][:, None]
    arm_ik = l_hat_ij_all[i, k, t][:, None]
    return (
        contractor_ab_a_b(g_arr, arm_ij, arm_ij) / 2 * T_one(t, f, kap, i, j)
        - contractor_ab_a_b(g_arr, arm_ik, arm_ik) / 2 * T_one(t, f, kap, i, k)
    )


def R_oneL(t, f, kap, i, j, k):
    return _response_leg_diff(t, f, kap, i, j, k, e_ab_arrL)


def R_oneR(t, f, kap, i, j, k):
    return _response_leg_diff(t, f, kap, i, j, k, e_ab_arrR)


def R_tilde_ij_integrand(t, f, i, j, kap):
    phase = np.exp(
        -2j * np.pi * f * scal_prod(kap, x_all[i, t][:, None] - x_all[j, t][:, None]) / c
    )
    return (
        (1 / (8 * np.pi))
        * phase
        * (
            R_oneL(t, f, kap, i, (i + 1) % 3, (i + 2) % 3)
            * np.conjugate(R_oneL(t, f, kap, j % 3, (j + 1) % 3, (j + 2) % 3))
            + R_oneR(t, f, kap, i, (i + 1) % 3, (i + 2) % 3)
            * np.conjugate(R_oneR(t, f, kap, j % 3, (j + 1) % 3, (j + 2) % 3))
        )
    )


# ── Multiprocessing workers ───────────────────────────────────────────────────
# Workers read module globals set before pool creation.  Fork-based pools
# (forced via get_process_pool) inherit those globals from the parent.
# JAX must NOT be imported before these pools run — see step 4 in main().


def compute_frequency(fi):
    Rtildeijt = np.zeros((3, 3, NTIMES_RESPONSE), dtype=np.complex128)
    for ti in range(NTIMES_RESPONSE):
        for i in range(3):
            for j in range(3):
                integ = (
                    np.sum(
                        R_tilde_ij_integrand(
                            int(ti * 365 / NTIMES_RESPONSE),
                            frequencies[fi],
                            i,
                            j,
                            kappas_import,
                        )
                        * map_Gal
                    )
                    * (4 * np.pi / npix)
                )
                Rtildeijt[i, j, ti] = S_gal[fi] * integ
    return fi, Rtildeijt


def _stabilize_covariance_batch(cov_batch, *, rtol: float = 1e-12, atol: float = 1e-60):
    """Hermitianise and add the minimum diagonal jitter needed for Cholesky."""
    cov_batch = 0.5 * (cov_batch + np.swapaxes(np.conjugate(cov_batch), -1, -2))
    eigvals = np.linalg.eigvalsh(cov_batch)
    scale = np.maximum(np.max(np.abs(eigvals), axis=1), atol)
    floor = np.maximum(atol, rtol * scale)
    jitter = np.clip(floor - eigvals[:, 0], a_min=0.0, a_max=None)
    if np.any(jitter > 0.0):
        cov_batch = cov_batch + jitter[:, None, None] * np.eye(cov_batch.shape[-1], dtype=cov_batch.dtype)
    return cov_batch


def compute_time(ti):
    n_f = len(freqs_chunk)
    cov_full = np.zeros((3, 3, n_f))
    for o in range(3):
        for op in range(3):
            cov_full[o, op, idx:] = np.interp(
                freqs_cut,
                frequencies,
                np.abs(Rtildeop_tf_times_H[o, op, ti]) * tdi15_factor(frequencies),
            )

    # Vectorised Cholesky draw — avoids per-bin Python loop
    n_draw = n_f - idx
    cov_band = _stabilize_covariance_batch(np.moveaxis(cov_full[:, :, idx:], -1, 0))
    L = np.linalg.cholesky(cov_band)  # (n_draw, 3, 3)
    rng = np.random.default_rng(seed_base + 1_000 + ti)
    z_r = rng.standard_normal((n_draw, 3))
    z_i = rng.standard_normal((n_draw, 3))
    draws = (np.einsum("nij,nj->ni", L, z_r) + 1j * np.einsum("nij,nj->ni", L, z_i)) / np.sqrt(2)
    out = np.zeros((3, n_f), dtype=complex)
    out[:, idx:] = draws.T
    return ti, out


def get_process_pool():
    """Fork-based pool — avoids JAX+spawn interaction."""
    try:
        return get_context("fork").Pool(cpu_count())
    except ValueError:
        return get_context().Pool(cpu_count())


def stitch_chunked_foreground_spectrum(samples_tf, nfreq_full):
    """Stitch ntimes non-overlapping time chunks into a full-year rFFT spectrum.

    Each chunk is brought back to time domain so the chunks can be concatenated
    end-to-end before the final rFFT reconstructs the full-year spectrum.
    """
    leaked = irfft(samples_tf, axis=1).flatten()
    padded = np.concatenate([leaked, np.zeros(2 * nfreq_full - 2 - len(leaked))])
    return rfft(padded)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    global frequencies, S_gal, map_Gal, npix, kappas_import
    global x_all, l_hat_ij_all, Rtildeop_tf_times_H
    global freqs_cut, freqs_chunk, idx
    global e_ab_arrL, e_ab_arrR, seed_base

    ensure_output_dir(OUTDIR)
    seed_base = RNG_SEED
    rng = np.random.default_rng(RNG_SEED)
    mode_label = (
        "stationary instrument noise + galactic background"
        if INCLUDE_GALACTIC
        else "stationary instrument noise"
    )
    print(f"Generating injection mode: {mode_label}")
    print(f"Using LISA_SEED = {RNG_SEED}")

    # ── 1. Frequency grid and galactic PSD ────────────────────────────────────
    frequencies = freqs_gal()
    if INCLUDE_GALACTIC:
        S_gal = galactic_psd(frequencies)

        # ── 2. Galactic sky map ───────────────────────────────────────────────
        npix = hp.nside2npix(nside)
        thetas, phis = hp.pix2ang(nside, range(npix))
        kappas_import = nhat(thetas, phis)
        e_ab_arrL = e_ab(kappas_import, -1)
        e_ab_arrR = e_ab(kappas_import, 1)

        map_Gal = np.zeros(npix)
        lvec = np.linspace(0, 100, 1000)
        for pi in range(npix):
            x, y, z = position_gal(lvec, thetas[pi], phis[pi])
            map_Gal[pi] = simpson(rho_gal(x, y, z), lvec)

        map_Gal = hp.Rotator(coord=["G", "C"]).rotate_map_pixel(
            hp.Rotator(rot=[180, 0, 0], deg=True).rotate_map_pixel(map_Gal)
        )

        def _draw_galaxy_mollview() -> None:
            hp.mollview(map_Gal, title="Galaxy", unit="arbitrary units", norm="log")

        _save_plot_once_to_cache("galaxy_mollview", _draw_galaxy_mollview)

        def _draw_galaxy_frequency_psd() -> None:
            plt.loglog(frequencies, omega_gw(frequencies, S_gal))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Omega (arbitrary units)")
            plt.title("Frequency PSD of the Galaxy")
            plt.grid()

        _save_plot_once_to_cache("galaxy_frequency_psd", _draw_galaxy_frequency_psd)
    else:
        S_gal = np.zeros_like(frequencies)
        npix = 0
        kappas_import = None
        e_ab_arrL = None
        e_ab_arrR = None
        map_Gal = None

    def _draw_lisa_noise_psd() -> None:
        for ch, label, ls in [(0, "A", "-"), (1, "E", ":"), (2, "T", "--")]:
            plt.loglog(frequencies, noise_tdi15_psd(ch, frequencies), label=label, linestyle=ls)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Noise PSD")
        plt.legend()
        plt.grid()
        plt.title("LISA Noise PSD")

    _save_plot_once_to_cache("lisa_noise_psd", _draw_lisa_noise_psd)

    if INCLUDE_GALACTIC:
        # ── 3. Spacecraft orbits ──────────────────────────────────────────────
        times = np.linspace(0, TS, 365)
        print("Using lisaorbits.EqualArmlengthOrbits() for constellation geometry")
        x_all = lisaorbits_positions(times)

        # Vectorised arm unit vectors: shape (3, 3, ntimes, 3)
        diff = x_all[np.newaxis, :, :, :] - x_all[:, np.newaxis, :, :]
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        l_hat_ij_all = diff / norms

        # ── 4. Response tensor (slow; fork-safe pool; JAX deferred until after) ──
        rtildeop_path = RESPONSE_TENSOR_PATH
        ensure_output_dir(rtildeop_path.parent)
        if rtildeop_path.exists():
            print(f"Loading cached response tensor from {rtildeop_path}")
            Rtildeop_tf_times_H = np.load(rtildeop_path)["Rtildeop_tf"]
            assert Rtildeop_tf_times_H.shape[2] == NTIMES_RESPONSE, (
                f"Cached tensor has ntimes={Rtildeop_tf_times_H.shape[2]}, "
                f"expected {NTIMES_RESPONSE}.  Delete the cache and rerun."
            )
        else:
            print("Cache not found — computing response tensor (slow)…")
            Rtildeijtf = np.zeros((3, 3, NTIMES_RESPONSE, len(frequencies)), dtype=np.complex128)
            with get_process_pool() as pool:
                for fi, res in tqdm(
                    pool.imap_unordered(compute_frequency, range(len(frequencies))),
                    total=len(frequencies),
                    desc="Response tensor",
                ):
                    Rtildeijtf[:, :, :, fi] = res

            cmat = np.array(
                [
                    [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
                    [1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)],
                    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
                ]
            )
            Rtildeop_tf_times_H = np.einsum("oi,pj,ijtf->optf", cmat, cmat, Rtildeijtf)
            np.savez(rtildeop_path, Rtildeop_tf=Rtildeop_tf_times_H)
            print(f"Saved response tensor to {rtildeop_path}")

        ntimes = Rtildeop_tf_times_H.shape[2]

        # Channel A total PSD diagnostic
        for ti in range(0, ntimes, ntimes // 9):
            psd_ti = (
                np.abs(Rtildeop_tf_times_H[0, 0, ti]) * tdi15_factor(frequencies)
                + noise_tdi15_psd(0, frequencies)
            )
            plt.loglog(frequencies, psd_ti, label=f"t={int(ti * 365 / ntimes)} day", alpha=0.5)
        plt.loglog(frequencies, noise_tdi15_psd(0, frequencies), label="Noise only", linestyle="dotted")
        plt.legend()
        plt.title("Channel A")
        save_figure(plt.gcf(), OUTDIR, "channel_a_total_psd")
    else:
        Rtildeop_tf_times_H = None
        ntimes = NTIMES_RESPONSE

    # ── 5. Background realization (pool must close before JAX below) ──────────
    yr = TS
    fmax_comp = np.max(frequencies)
    fmin_comp = np.min(frequencies)
    dt = 1 / (2 * fmax_comp)
    Nsamp_tot = int(yr / dt)
    freqs_all = rfftfreq(Nsamp_tot, dt)

    def _noise_fd(channel):
        psd = noise_tdi15_psd(channel, freqs_all)
        channel_rng = np.random.default_rng(RNG_SEED + 10_000 + channel)
        white = channel_rng.normal(size=len(freqs_all)) + 1j * channel_rng.normal(size=len(freqs_all))
        return np.sqrt(psd) * white / np.sqrt(2)

    nAf, nEf, nTf = _noise_fd(0), _noise_fd(1), _noise_fd(2)

    n_freqs = len(freqs_all)
    if INCLUDE_GALACTIC:
        DT_chunk = yr / ntimes
        Nsamp_chunk = int(DT_chunk / dt)
        freqs_chunk = rfftfreq(Nsamp_chunk, dt)
        freqs_cut = freqs_chunk[freqs_chunk >= fmin_comp]
        idx = len(freqs_chunk) - len(freqs_cut)

        gAtf = np.zeros((ntimes, len(freqs_chunk)), dtype=complex)
        gEtf = np.zeros((ntimes, len(freqs_chunk)), dtype=complex)
        gTtf = np.zeros((ntimes, len(freqs_chunk)), dtype=complex)

        with get_process_pool() as pool:
            for ti, AETf in pool.imap_unordered(compute_time, range(ntimes)):
                gAtf[ti] = AETf[0]
                gEtf[ti] = AETf[1]
                gTtf[ti] = AETf[2]

        gAf = stitch_chunked_foreground_spectrum(gAtf, n_freqs)
        gEf = stitch_chunked_foreground_spectrum(gEtf, n_freqs)
        gTf = stitch_chunked_foreground_spectrum(gTtf, n_freqs)
    else:
        freqs_chunk = freqs_all
        gAf = np.zeros(n_freqs, dtype=complex)
        gEf = np.zeros(n_freqs, dtype=complex)
        gTf = np.zeros(n_freqs, dtype=complex)

    a_bg_f = nAf + gAf
    e_bg_f = nEf + gEf
    t_bg_f = nTf + gTf

    plt.loglog(freqs_all, np.abs(nAf) ** 2, label="A Noise", alpha=0.5)
    if INCLUDE_GALACTIC:
        plt.loglog(freqs_chunk, np.abs(gAtf[ntimes // 2]) ** 2, label=f"A Galaxy t={ntimes//2}", alpha=0.5)
    plt.loglog(freqs_all, noise_tdi15_psd(0, freqs_all), label="Noise PSD", linestyle="dotted")
    if INCLUDE_GALACTIC:
        plt.loglog(freqs_all, np.abs(gAf) ** 2, label="A Galaxy full", alpha=0.2, c="y")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(1e-4, 3e-3)
    plt.ylim(1e-44, 1e-35)
    plt.legend()
    save_figure(plt.gcf(), OUTDIR, "channel_a_noise_vs_galaxy")

    # ── 6. Inject resolved GB sources ─────────────────────────────────────────
    # JAX is imported here — after both multiprocessing pools have closed —
    # to prevent XLA initialisation conflicting with forked workers.
    import jax
    import jax.numpy as jnp
    from jaxgb.jaxgb import JaxGB

    jax.config.update("jax_enable_x64", True)

    if INCLUDE_GALACTIC:
        psd_A = build_total_noise_psd(Rtildeop_tf_times_H, frequencies, freqs_all, channel=0)
        psd_E = build_total_noise_psd(Rtildeop_tf_times_H, frequencies, freqs_all, channel=1)
        psd_T = build_total_noise_psd(Rtildeop_tf_times_H, frequencies, freqs_all, channel=2)
    else:
        psd_A = np.maximum(noise_tdi15_psd(0, freqs_all), 1e-60)
        psd_E = np.maximum(noise_tdi15_psd(1, freqs_all), 1e-60)
        psd_T = np.maximum(noise_tdi15_psd(2, freqs_all), 1e-60)

    gb_orbit = lisaorbits.EqualArmlengthOrbits()
    jgb_full = JaxGB(gb_orbit, t_obs=yr, t0=0.0, n=256)

    source_params_row, prior_f0, prior_fdot = draw_source_prior_and_params(rng)
    source_params = source_params_row.reshape(1, -1)
    src = source_params[0]
    target_snr_aet = float(rng.uniform(TARGET_SNR_MIN, TARGET_SNR_MAX))

    def _source_frequency_series(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params_j = jnp.asarray(params, dtype=jnp.float64)
        a_loc, e_loc, t_loc = jgb_full.get_tdi(params_j, tdi_generation=1.5, tdi_combination="AET")
        kmin = int(jnp.asarray(jgb_full.get_kmin(params_j[None, 0:1])).reshape(-1)[0])
        return (
            place_local_tdi(np.asarray(a_loc), kmin, n_freqs),
            place_local_tdi(np.asarray(e_loc), kmin, n_freqs),
            place_local_tdi(np.asarray(t_loc), kmin, n_freqs),
        )

    source_Af, source_Ef, source_Tf = _source_frequency_series(src)
    snr_A = matched_filter_snr_rfft(source_Af, psd_A, freqs_all, dt=dt)
    snr_E = matched_filter_snr_rfft(source_Ef, psd_E, freqs_all, dt=dt)
    snr_T = matched_filter_snr_rfft(source_Tf, psd_T, freqs_all, dt=dt)
    snr_ae = float(np.hypot(snr_A, snr_E))
    snr_aet = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))
    if snr_aet <= 0.0:
        raise ValueError("Generated GB template has non-positive SNR before amplitude rescaling.")

    amplitude_center = float(src[2] * target_snr_aet / snr_aet)
    prior_A = (0.5 * amplitude_center, 2.0 * amplitude_center)
    src[2] = draw_positive_parameter_from_bounds(rng, prior_A)
    source_Af, source_Ef, source_Tf = _source_frequency_series(src)
    snr_A = matched_filter_snr_rfft(source_Af, psd_A, freqs_all, dt=dt)
    snr_E = matched_filter_snr_rfft(source_Ef, psd_E, freqs_all, dt=dt)
    snr_T = matched_filter_snr_rfft(source_Tf, psd_T, freqs_all, dt=dt)
    snr_ae = float(np.hypot(snr_A, snr_E))
    snr_aet = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))

    print("\nInjecting 1 resolved GB source with JaxGB:")
    print(f"  seed = {RNG_SEED}")
    print(f"  f0 = {src[0]:.5e} Hz")
    print(f"  fdot = {src[1]:.5e} Hz/s")
    print(f"  A = {src[2]:.5e}")
    print(f"  target SNR (A+E+T) = {target_snr_aet:.1f}")
    print(f"  SNR (A+E+T) = {snr_aet:.1f}")

    data_At = irfft(a_bg_f + source_Af, n=Nsamp_tot)
    data_Et = irfft(e_bg_f + source_Ef, n=Nsamp_tot)
    data_Tt = irfft(t_bg_f + source_Tf, n=Nsamp_tot)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    plot_specs = [
        ("A", psd_A, source_Af),
        ("E", psd_E, source_Ef),
        ("T", psd_T, source_Tf),
    ]
    for ax, (label, noise_psd, signal) in zip(axes, plot_specs):
        ax.loglog(
            freqs_all[1:],
            noise_characteristic_strain(noise_psd, freqs_all)[1:],
            label="Noise + stochastic foreground" if INCLUDE_GALACTIC else "Instrument noise",
            color="black",
        )
        ax.loglog(
            freqs_all[1:],
            rfft_characteristic_strain(signal, freqs_all, dt)[1:],
            label="Resolved GB signal",
            color="C1",
        )
        ax.axvline(src[0], color="C2", alpha=0.35, linewidth=1.0, label="Injected f0")
        ax.set_ylabel(r"$h_c(f)$")
        ax.set_title(f"Channel {label}")
        ax.set_xlim(1e-4, 3e-3)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Frequency (Hz)")
    save_figure(plt.gcf(), OUTDIR, "resolved_gb_vs_noise_characteristic_strain")

    # ── 7. Save injection data ────────────────────────────────────────────────
    np.savez(
        INJECTION_PATH,
        dt=dt,
        t_obs=yr,
        seed=np.array(RNG_SEED, dtype=int),
        freqs=freqs_all,
        noise_psd_A=psd_A,
        noise_psd_E=psd_E,
        noise_psd_T=psd_T,
        data_At=data_At,
        data_Et=data_Et,
        data_Tt=data_Tt,
        source_params=source_params,
        prior_f0=np.asarray(prior_f0, dtype=float),
        prior_fdot=np.asarray(prior_fdot, dtype=float),
        prior_A=np.asarray(prior_A, dtype=float),
        source_target_snrs=np.array([target_snr_aet], dtype=float),
        source_snrs=np.array([snr_aet], dtype=float),
        source_snrs_ae=np.array([snr_ae], dtype=float),
        source_snrs_aet=np.array([snr_aet], dtype=float),
        include_galactic=np.array(int(INCLUDE_GALACTIC)),
    )
    print(f"\nSaved injection to {INJECTION_PATH}")
    print(f"  T_obs = {yr / 86400:.1f} days,  dt = {dt:.2f} s,  N = {Nsamp_tot}")
    print(f"  mode = {mode_label}")


if __name__ == "__main__":
    main()
