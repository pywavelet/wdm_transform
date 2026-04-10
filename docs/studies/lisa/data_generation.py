from multiprocessing import cpu_count, get_context

import healpy as hp
import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    BACKGROUND_DIR,
    L_LISA,
    c,
    ensure_output_dir,
    freqs_gal,
    galactic_psd,
    noise_tdi15_psd,
    omega_gw,
    tdi15_factor,
)
from matplotlib import pyplot as plt
from numpy.fft import irfft, rfft, rfftfreq
from scipy.integrate import simpson
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

"""Galactic-binary foreground simulation for LISA/Taiji-like channels.

Goal:
- Build a toy model for the anisotropic Galactic foreground in the A/E/T basis.
- Combine foreground and instrumental-noise PSDs.
- Generate diagnostic plots and save them to OUTDIR.

Pipeline summary:
1. Define a frequency-domain Galactic PSD model.
2. Build a sky morphology map with a simple Galactic density model.
3. Compute detector response and project to the A/E/T basis.
4. Cache expensive response computation in OUTDIR/Rtildeop_tf.npz.
5. Draw random foreground/noise realizations and save diagnostics.
"""

OUTDIR = BACKGROUND_DIR

# Runtime state populated inside main().
#
# Why these are module-level (instead of local inside main): worker functions used
# by multiprocessing (compute_frequency/compute_time) are module functions and read
# shared arrays from module globals. Keeping them here avoids repeatedly pickling
# very large arrays for each task.
frequencies = None  # Frequency grid for all PSD evaluations.
S_gal = None  # Galactic strain PSD evaluated on frequencies.
map_Gal = None  # HEALPix sky map of Galactic morphology weights.
npix = None  # Number of HEALPix pixels in map_Gal.
kappas_import = None  # Unit sky-direction vectors for all HEALPix pixels.
x_all = None  # Spacecraft positions, shape (3, nt, 3) in meters.
l_hat_ij_all = None  # Unit arm vectors between spacecraft pairs over time.
Rtildeop_tf_times_H = None  # Cached A/E/T response tensor in time-frequency domain.
freqs_cut = None  # Chunk frequencies restricted to analysis band.
freqs_chunk = None  # Full chunk rFFT frequencies before low-f cut.
idx = None  # Start index where freqs_chunk enters analysis band.
e_ab_arrL = None  # Left-circular polarization tensor projected on sky.
e_ab_arrR = None  # Right-circular polarization tensor projected on sky.

########### Angular morphology of the galaxy ###########

rg = 4.0
x_s = 8.0


def rho_gal(x, y, z, rho_0=1.0, A=0.25, R_b=0.5, R_d=2.5, Z_d=0.2):  # units of kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    u = np.sqrt(x**2 + y**2)
    z_scaled = np.clip(z / Z_d, -50.0, 50.0)
    sech2 = 1.0 / (np.cosh(z_scaled) ** 2)
    return rho_0 * (A * np.exp(-(r**2) / R_b**2) + (1.0 - A) * np.exp(-u / R_d) * sech2)


def position_gal(distance, theta, phi):
    x = x_s + distance * np.sin(theta) * np.cos(phi)
    y = distance * np.sin(theta) * np.sin(phi)
    z = distance * np.cos(theta)
    return x, y, z


nside = 16


def save_plot(filename):
    """Save the current matplotlib figure into OUTDIR and close it."""
    plt.savefig(OUTDIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def get_process_pool():
    """Create a process pool, preferring fork on macOS script execution."""
    try:
        return get_context("fork").Pool(cpu_count())
    except ValueError:
        return get_context().Pool(cpu_count())


#### Observation duration and GB sources

# Standard observation duration for all analyses
N_SAMPLES = 30976
DT = 166.7  # seconds
T_OBS = N_SAMPLES * DT  # ≈ 59.8 days

# Injected GB parameters: [f0, fdot, amplitude, ra, dec, psi, iota, phi0]
SOURCE_PARAMS = np.array(
    [
        [
            1.35962e-3,
            8.94581279e-19,
            1.07345e-22,
            2.40,
            0.31,
            3.56,
            0.52,
            3.06,
        ],
        [
            1.41220e-3,
            -2.3e-18,
            8.2e-23,
            2.15,
            0.18,
            1.20,
            0.93,
            1.40,
        ],
    ],
    dtype=float,
)

#### Instrumental noise for LISA

#### LISA response

TS = 365 * 24 * 3600


def nhat(theta, phi):
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def lisaorbits_positions(times):
    """Spacecraft positions from the `lisaorbits` equal-arm model, in meters."""
    orbit = lisaorbits.EqualArmlengthOrbits()
    if hasattr(orbit, "compute_position"):
        pos = np.asarray(
            orbit.compute_position(times, sc=np.array([1, 2, 3])),
            dtype=float,
        )
        return np.transpose(pos, (1, 0, 2))

    positions = np.zeros((3, len(times), 3), dtype=float)
    for sc in range(3):
        try:
            pos = np.asarray(orbit.compute_spacecraft_position(sc, times), dtype=float)
        except Exception:
            pos = np.asarray(
                orbit.compute_spacecraft_position(sc + 1, times),
                dtype=float,
            )
        positions[sc] = pos.T if pos.shape[0] == 3 else pos
    return positions


def get_theta_phi(vector):
    theta = np.arccos(vector[2])
    phi = np.arctan2(vector[1], vector[0])
    return theta, phi


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
    return (
        e_plus(*get_theta_phi(kap)) + 1j * lam * e_cross(*get_theta_phi(kap))
    ) / np.sqrt(2)


def M_one(t, f, kap, i, j):
    phas = L_LISA * f * (1 + scal_prod(kap, l_hat_ij_all[i, j, t][:, None])) / c
    return np.exp(np.pi * 1j * phas) * np.sinc(phas)


def G_oneL(t, kap, i, j):
    """Left-circular polarization contraction on arm (i,j) at time index t."""
    arm = l_hat_ij_all[i, j, t][:, None]
    return contractor_ab_a_b(e_ab_arrL, arm, arm) / 2


def G_oneR(t, kap, i, j):
    """Right-circular polarization contraction on arm (i,j) at time index t."""
    arm = l_hat_ij_all[i, j, t][:, None]
    return contractor_ab_a_b(e_ab_arrR, arm, arm) / 2


def T_one(t, f, kap, i, j):
    fstar = 1 / (2 * np.pi * L_LISA / c)
    return np.exp(-1j * f / fstar) * M_one(t, f, kap, j, i) + np.exp(
        -1j * f * scal_prod(kap, l_hat_ij_all[i, j, t][:, None]) / fstar
    ) * M_one(t, f, kap, i, j)


def _response_leg_difference(t, f, kap, i, j, k, g_func):
    """Difference of the two Michelson legs for one polarization model."""
    return g_func(t, kap, i, j) * T_one(t, f, kap, i, j) - g_func(t, kap, i, k) * T_one(
        t, f, kap, i, k
    )


def R_oneL(t, f, kap, i, j, k):
    """Single-link response for left-circular polarization (legacy name)."""
    return _response_leg_difference(t, f, kap, i, j, k, G_oneL)


def R_oneR(t, f, kap, i, j, k):
    """Single-link response for right-circular polarization (legacy name)."""
    return _response_leg_difference(t, f, kap, i, j, k, G_oneR)


def R_tilde_ij_integrand(t, f, i, j, kap):
    """Sky-integrand for the response cross-power between channels i and j."""

    phase = np.exp(
        -2
        * np.pi
        * 1j
        * f
        * scal_prod(kap, x_all[i, t][:, None] - x_all[j, t][:, None])
        / c
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


######## !!! SLOW !!! ########
# Pre-compute the sky integral for each frequency and time to speed up later
# foreground and PSD calculations.

ntimes = 90


def compute_frequency(fi):
    Rtildeijt_times_H = np.zeros((3, 3, ntimes), dtype=np.complex128)
    for ti in range(ntimes):
        for i in range(3):
            for j in range(3):
                integ = np.sum(
                    R_tilde_ij_integrand(
                        int(ti * 365 / ntimes), frequencies[fi], i, j, kappas_import
                    )
                    * map_Gal
                ) * (4 * np.pi / npix)
                Rtildeijt_times_H[i, j, ti] = S_gal[fi] * integ
    return fi, Rtildeijt_times_H


def tot_PSD_tf(o, ti):
    return np.abs(Rtildeop_tf_times_H[o, o, ti]) * tdi15_factor(
        frequencies
    ) + noise_tdi15_psd(o, frequencies)


def compute_time(ti):
    cov_temp_all = np.zeros((3, 3, len(freqs_chunk)))
    for o in range(3):
        for op in range(3):
            cov_temp_all[o, op, idx:] = np.interp(
                freqs_cut,
                frequencies,
                np.abs(Rtildeop_tf_times_H[o, op, ti]) * tdi15_factor(frequencies),
            )
    samplesAETf = np.zeros((3, len(freqs_chunk))) * 1j
    for fi in range(idx, len(freqs_chunk)):
        real_draw = np.random.multivariate_normal(np.zeros(3), cov_temp_all[:, :, fi])
        imag_draw = np.random.multivariate_normal(np.zeros(3), cov_temp_all[:, :, fi])
        samplesAETf[:, fi] += (real_draw + 1j * imag_draw) / np.sqrt(2)
    return ti, samplesAETf


def stitch_chunked_foreground_spectrum(samples_tf, nfreq_full):
    """Turn chunked rFFT draws into one full-year rFFT realization."""
    leaked = irfft(samples_tf, axis=1).flatten()
    padded = np.concatenate([leaked, np.zeros(2 * nfreq_full - 2 - len(leaked))])
    return rfft(padded)


def main():
    global frequencies, S_gal, map_Gal, npix, kappas_import
    global x_all, l_hat_ij_all, Rtildeop_tf_times_H, freqs_cut, freqs_chunk, idx, ntimes
    global e_ab_arrL, e_ab_arrR

    ensure_output_dir(OUTDIR)

    frequencies = freqs_gal()
    S_gal = galactic_psd(frequencies)

    npix = hp.nside2npix(nside)
    thetas, phis = hp.pix2ang(nside, range(npix))
    kappas_import = nhat(thetas, phis)
    e_ab_arrL = e_ab(kappas_import, -1)
    e_ab_arrR = e_ab(kappas_import, 1)

    map_Gal = np.zeros(npix)
    lvec = np.linspace(0, 100, 1000)
    for pi in range(npix):
        x, y, z = position_gal(lvec, thetas[pi], phis[pi])
        rho = rho_gal(x, y, z)
        map_Gal[pi] = simpson(rho, lvec)

    rot = hp.Rotator(rot=[180, 0, 0], deg=True)
    rot2 = hp.Rotator(coord=["G", "C"])
    newmap_Gal = rot.rotate_map_pixel(map_Gal)
    map_Gal = rot2.rotate_map_pixel(newmap_Gal)

    hp.mollview(map_Gal, title="Galaxy", unit="arbitrary units", norm="log")
    save_plot("galaxy_mollview.png")

    plt.loglog(frequencies, omega_gw(frequencies, S_gal))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Omega (arbitrary units)")
    plt.title("Frequency PSD of the Galaxy")
    plt.grid()
    save_plot("galaxy_frequency_psd.png")

    plt.loglog(frequencies, noise_tdi15_psd(0, frequencies), label="A", alpha=0.5)
    plt.loglog(frequencies, noise_tdi15_psd(1, frequencies), label="E", linestyle="dotted")
    plt.loglog(frequencies, noise_tdi15_psd(2, frequencies), label="T")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Noise PSD")
    plt.legend()
    plt.grid()
    plt.title("LISA Noise PSD")
    save_plot("lisa_noise_psd.png")

    times = np.linspace(0, TS, 365)
    print("Using lisaorbits.EqualArmlengthOrbits() for constellation geometry")
    x_all = lisaorbits_positions(times)

    l_hat_ij_all = np.zeros((3, 3, len(times), 3))
    for i in range(3):
        for j in range(3):
            for t in range(len(times)):
                vec = x_all[j, t] - x_all[i, t]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    l_hat_ij_all[i, j, t] = vec / norm
                else:
                    l_hat_ij_all[i, j, t] = 0.0

    rtildeop_path = OUTDIR / "Rtildeop_tf.npz"
    if rtildeop_path.exists():
        print(f"Loading cached response tensor from {rtildeop_path}")
        Rtildeop_tf_times_H = np.load(rtildeop_path)["Rtildeop_tf"]
    else:
        print("Cache not found. Computing response tensor (slow)...")
        Rtildeijtf_times_H = np.zeros(
            (3, 3, ntimes, len(frequencies)), dtype=np.complex128
        )

        with get_process_pool() as pool:
            for fi, res1 in tqdm(
                pool.imap_unordered(compute_frequency, range(len(frequencies))),
                total=len(frequencies),
                desc="Computing response tensor",
            ):
                Rtildeijtf_times_H[:, :, :, fi] = res1

        cmat = np.array(
            [
                [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
                [1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )

        Rtildeop_tf_times_H = np.einsum(
            "oi,pj,ijtf->optf", cmat, cmat, Rtildeijtf_times_H
        )
        np.savez(rtildeop_path, Rtildeop_tf=Rtildeop_tf_times_H)
        print(f"Saved response tensor cache to {rtildeop_path}")

    ntimes = len(Rtildeop_tf_times_H[0, 0, :, 0])

    for ti in range(0, 90, 10):
        plt.loglog(
            frequencies,
            tot_PSD_tf(0, ti),
            label=f"t={int(ti * 365 / 90)} day",
            alpha=0.5,
        )
    plt.loglog(
        frequencies,
        noise_tdi15_psd(0, frequencies),
        label="Noise only",
        linestyle="dotted",
    )
    plt.legend()
    plt.title("Channel A")
    save_plot("channel_a_total_psd.png")

    # Full-year analysis
    yr = 365 * 24 * 3600
    fmax_comp = np.max(frequencies)
    fmin_comp = np.min(frequencies)
    dt_full_year = 1 / (2 * fmax_comp)
    Nsamp_full_year = int(yr / dt_full_year)
    freqs_full_year = rfftfreq(Nsamp_full_year, dt_full_year)

    nAf = (
        np.sqrt(noise_tdi15_psd(0, freqs_full_year))
        * (
            np.random.normal(size=len(freqs_full_year))
            + 1j * np.random.normal(size=len(freqs_full_year))
        )
        / np.sqrt(2)
    )
    nEf = (
        np.sqrt(noise_tdi15_psd(1, freqs_full_year))
        * (
            np.random.normal(size=len(freqs_full_year))
            + 1j * np.random.normal(size=len(freqs_full_year))
        )
        / np.sqrt(2)
    )
    nTf = (
        np.sqrt(noise_tdi15_psd(2, freqs_full_year))
        * (
            np.random.normal(size=len(freqs_full_year))
            + 1j * np.random.normal(size=len(freqs_full_year))
        )
        / np.sqrt(2)
    )

    DT = yr / ntimes
    Nsamp_chunk = int(DT / dt_full_year)
    freqs_chunk = rfftfreq(Nsamp_chunk, dt_full_year)
    freqs_cut = freqs_chunk[freqs_chunk >= fmin_comp]
    idx = len(freqs_chunk) - len(freqs_cut)

    gAtf = np.zeros((ntimes, len(freqs_chunk))) * 1j
    gEtf = np.zeros((ntimes, len(freqs_chunk))) * 1j
    gTtf = np.zeros((ntimes, len(freqs_chunk))) * 1j

    with get_process_pool() as pool:
        for ti, AETf in pool.imap_unordered(compute_time, range(ntimes)):
            gAtf[ti] = AETf[0]
            gEtf[ti] = AETf[1]
            gTtf[ti] = AETf[2]

    gAf = stitch_chunked_foreground_spectrum(gAtf, len(nAf))
    gEf = stitch_chunked_foreground_spectrum(gEtf, len(nEf))
    gTf = stitch_chunked_foreground_spectrum(gTtf, len(nTf))

    a_total_f = nAf + gAf
    e_total_f = nEf + gEf
    t_total_f = nTf + gTf

    a_total_t = irfft(a_total_f, n=Nsamp_full_year)
    e_total_t = irfft(e_total_f, n=Nsamp_full_year)
    t_total_t = irfft(t_total_f, n=Nsamp_full_year)

    # Build JaxGB generator for our observation window
    orbit_model = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit_model, t_obs=T_OBS, t0=0.0, n=256)

    def generate_a_channel(params: np.ndarray) -> np.ndarray:
        """Generate A-channel TDI1.5 time-domain signal for given GB parameters."""
        A, _, _ = jgb.get_tdi(
            jnp.asarray(params, dtype=jnp.float64),
            tdi_generation=1.5,
            tdi_combination="AET",
        )
        return np.asarray(A, dtype=np.float64)

    # Generate injected GB signals (A-channel time domain, at observation sample rate)
    signal_A_t_obs = np.zeros(N_SAMPLES)
    for params in SOURCE_PARAMS:
        signal_A_t_obs += generate_a_channel(params)

    # Resample full-year background to observation sample rate and truncate
    indices_full = np.arange(Nsamp_full_year) * dt_full_year
    indices_obs = np.arange(N_SAMPLES) * DT
    a_total_t_obs = np.interp(indices_obs, indices_full, a_total_t)

    # Injected observation: background + signals (time domain, at observation sample rate)
    data_A_t_obs = a_total_t_obs + signal_A_t_obs

    # Also compute frequency-domain version for lisa_freq_mcmc.py
    data_A_f_obs = rfft(data_A_t_obs)

    # Generate frequency grids for observation window
    dt_obs = DT
    freqs_all = rfftfreq(N_SAMPLES, dt_obs)
    freqs_psd = freqs_gal()  # For PSD calculations

    plt.loglog(freqs_all, np.abs(nAf) ** 2, label="A Noise", alpha=0.5)
    plt.loglog(freqs_chunk, np.abs(gAtf[50]) ** 2, label="A Galaxy t=0", alpha=0.5)
    plt.loglog(
        freqs_all, noise_tdi15_psd(0, freqs_all), label="Noise PSD", linestyle="dotted"
    )
    plt.loglog(freqs_all, np.abs(gAf) ** 2, label="A Galaxy FULL", alpha=0.2, c="y")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(1e-4, 3e-3)
    plt.ylim(1e-44, 1e-35)
    plt.legend()
    save_plot("channel_a_noise_vs_galaxy.png")

    background_path = OUTDIR / "tdi15_background_realization.npz"
    np.savez(
        background_path,
        dt=dt_obs,
        year_seconds=yr,
        T_obs=T_OBS,
        N=N_SAMPLES,
        freqs_all=freqs_all,
        noise_Af=nAf,
        noise_Ef=nEf,
        noise_Tf=nTf,
        gal_Af=gAf,
        gal_Ef=gEf,
        gal_Tf=gTf,
        # Full-year fields for reference
        total_Af=a_total_f,
        total_Ef=e_total_f,
        total_Tf=t_total_f,
        total_At=a_total_t,
        total_Et=e_total_t,
        total_Tt=t_total_t,
        # Observation window with injected signals
        data_Af=data_A_f_obs,
        data_At=data_A_t_obs,
        signal_At=signal_A_t_obs,
        source_params=SOURCE_PARAMS,
        noise_psd_A=noise_tdi15_psd(0, freqs_all),
    )
    print(f"Saved TDI1.5 background realization to {background_path}")
    print(f"  Observation window: T_obs = {T_OBS / 86400:.2f} days, dt = {dt_obs:.1f} s, N = {N_SAMPLES}")
    print(f"  Injected {len(SOURCE_PARAMS)} GB sources with parameters shape {SOURCE_PARAMS.shape}")


if __name__ == "__main__":
    main()
