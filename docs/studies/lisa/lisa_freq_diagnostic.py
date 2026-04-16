"""Diagnostic plots: data, true signal, prior samples for LISA GB inference."""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import jax
import jax.numpy as jnp
from jaxgb.jaxgb import JaxGB
import lisaorbits

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
jax.config.update("jax_enable_x64", True)

from lisa_common import (
    INJECTION_PATH,
    load_injection,
    build_local_prior_info,
)

# ── Load data ─────────────────────────────────────────────────────────────
inj = load_injection(INJECTION_PATH)
dt = inj.dt
t_obs = inj.t_obs
data_aet_full = np.stack([inj.data_At, inj.data_Et, inj.data_Tt], axis=0)
SOURCE_PARAM = inj.source_params[0].copy()

data_aet_f = rfft(data_aet_full, axis=1)
freqs = rfftfreq(data_aet_full.shape[1], dt)
df = 1.0 / t_obs

# ── Prior info ────────────────────────────────────────────────────────────
prior_info = build_local_prior_info(
    prior_f0=inj.prior_f0,
    prior_fdot=inj.prior_fdot,
    prior_A=inj.prior_A,
)

print(f"Data: N={data_aet_full.shape[1]}, dt={dt:.2e}, T_obs={t_obs/86400:.1f} days")
print(f"Prior f0: [{inj.prior_f0[0]:.6e}, {inj.prior_f0[1]:.6e}]")
print(f"Prior logf0 center: {prior_info.prior_center[0]:.3f}, scale: {prior_info.prior_scale[0]:.3f}")
print(f"Prior logfdot center: {prior_info.prior_center[1]:.3f}, scale: {prior_info.prior_scale[1]:.3f}")
print(f"Prior logA center: {prior_info.prior_center[2]:.3f}, scale: {prior_info.prior_scale[2]:.3f}")
print(f"True f0: {SOURCE_PARAM[0]:.6e}, fdot: {SOURCE_PARAM[1]:.6e}, A: {SOURCE_PARAM[2]:.3e}")

# ── JaxGB setup ───────────────────────────────────────────────────────────
orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)

kmin = max(int(np.floor(inj.prior_f0[0] * t_obs)) - jgb.n, 0)
kmax = min(int(np.ceil(inj.prior_f0[1] * t_obs)) + jgb.n + 1, len(freqs))
band_freqs = freqs[kmin:kmax]

print(f"\nBand: kmin={kmin}, kmax={kmax}, n_freq={kmax-kmin}")

def source_aet_band(params, kmin_in, kmax_in):
    """Generate A, E, T in frequency band."""
    A, E, T = jgb.sum_tdi(
        jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
        kmin=int(kmin_in),
        kmax=int(kmax_in),
        tdi_combination="AET",
    )
    return jnp.stack(
        [
            jnp.asarray(A, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(E, dtype=jnp.complex128).reshape(-1),
            jnp.asarray(T, dtype=jnp.complex128).reshape(-1),
        ],
        axis=0,
    )

# ── True signal ───────────────────────────────────────────────────────────
h_true_aet = np.array(source_aet_band(SOURCE_PARAM, kmin, kmax))
print(f"True signal shape: {h_true_aet.shape}")
print(f"True signal norm (A): {np.linalg.norm(h_true_aet[0]):.3e}")

# ── Prior samples (log-space) ─────────────────────────────────────────────
np.random.seed(0)
n_prior_samples = 50

logf0_samples = np.random.normal(
    prior_info.prior_center[0],
    prior_info.prior_scale[0],
    n_prior_samples,
)
logfdot_samples = np.random.normal(
    prior_info.prior_center[1],
    prior_info.prior_scale[1],
    n_prior_samples,
)
logA_samples = np.random.normal(
    prior_info.prior_center[2],
    prior_info.prior_scale[2],
    n_prior_samples,
)
phi0_samples = np.random.uniform(-np.pi, np.pi, n_prior_samples)

# ── Likelihood at various points ──────────────────────────────────────────
def whittle_nll(logf0, logfdot, logA, phi0, channel_idx=0):
    """Evaluate negative log-likelihood for one channel."""
    params = SOURCE_PARAM.copy()
    params[0] = np.exp(logf0)
    params[1] = np.exp(logfdot)
    params[2] = np.exp(logA)
    params[7] = phi0

    h_aet = np.array(source_aet_band(params, kmin, kmax))
    noise_psd_aet = np.stack(
        [np.interp(band_freqs, freqs, psd) for psd in
         [inj.noise_psd_A, inj.noise_psd_E, inj.noise_psd_T]],
        axis=0,
    )

    data_band = data_aet_f[:, kmin:kmax]
    residual_phys = dt * (data_band - h_aet)

    ch = channel_idx
    nll = np.sum(np.log(noise_psd_aet[ch]) + 2.0 * df * np.abs(residual_phys[ch])**2 / noise_psd_aet[ch])
    return nll

# Evaluate at truth
logf0_true = np.log(SOURCE_PARAM[0])
logfdot_true = np.log(SOURCE_PARAM[1])
logA_true = np.log(SOURCE_PARAM[2])
phi0_true = SOURCE_PARAM[7]

nll_true = whittle_nll(logf0_true, logfdot_true, logA_true, phi0_true)
print(f"\nNLL at truth: {nll_true:.1f}")

# Evaluate at prior samples
nlls_prior = []
for i in range(min(5, n_prior_samples)):
    nll = whittle_nll(logf0_samples[i], logfdot_samples[i], logA_samples[i], phi0_samples[i])
    nlls_prior.append(nll)
    print(f"NLL at prior sample {i}: {nll:.1f}")

print(f"NLL at truth vs prior: truth={nll_true:.1f}, prior_mean={np.mean(nlls_prior):.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

channels = [0, 1, 2]
ch_names = ["A", "E", "T"]

for ch in channels:
    # Data + true signal
    ax = axes[ch, 0]
    power_data = np.abs(data_aet_f[ch, kmin:kmax])**2
    power_signal = np.abs(h_true_aet[ch])**2
    ax.semilogy(band_freqs, power_data, "k.", markersize=2, alpha=1, label="Data")
    ax.semilogy(band_freqs, power_signal, "r-", linewidth=1.5, label="True signal", zorder=-10, alpha=0.7)
    ax.set_ylabel(f"{ch_names[ch]} power")
    ax.set_xlim(band_freqs[0], band_freqs[-1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Prior samples
    ax = axes[ch, 1]
    for i in range(n_prior_samples):
        params = SOURCE_PARAM.copy()
        params[0] = np.exp(logf0_samples[i])
        params[1] = np.exp(logfdot_samples[i])
        params[2] = np.exp(logA_samples[i])
        params[7] = phi0_samples[i]

        h_sample = np.array(source_aet_band(params, kmin, kmax))
        power_sample = np.abs(h_sample[ch])**2
        ax.semilogy(band_freqs, power_sample, "b-", alpha=0.1)

    ax.semilogy(band_freqs, power_signal, "r-", linewidth=2, label="True signal", zorder=10)
    ax.set_ylabel(f"{ch_names[ch]} power")
    ax.set_xlim(band_freqs[0], band_freqs[-1])
    ax.set_title(f"{ch_names[ch]}: {n_prior_samples} prior samples")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/lisa_freq_diagnostic.png", dpi=100, bbox_inches="tight")
print(f"\nSaved diagnostic plot to /tmp/lisa_freq_diagnostic.png")

# Summary
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"✓ Data loaded: {data_aet_f.shape} complex FFT bins")
print(f"✓ Band selected: f ∈ [{band_freqs[0]:.3e}, {band_freqs[-1]:.3e}] Hz")
print(f"✓ True signal computed and visible in plots")
print(f"✓ Prior samples generated and checked")
print(f"✓ NLL at truth: {nll_true:.1f} (should be low relative to prior)")
print(f"✓ If prior samples are way outside data, prior is too wide")
print(f"✓ If true signal is not visible in data, SNR issue or data loading problem")
