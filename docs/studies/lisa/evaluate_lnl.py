import jax
import jax.numpy as jnp
import numpy as np
import lisaorbits
from jaxgb.jaxgb import JaxGB

jax.config.update("jax_enable_x64", True)

_inj = np.load("outdir_gb_background/injection.npz")
dt = float(_inj["dt"])
t_obs = float(_inj["t_obs"])
data_At_full = np.asarray(_inj["data_At"], dtype=float)
noise_psd_saved = np.asarray(_inj["noise_psd_A"], dtype=float)
freqs_saved = np.asarray(_inj["freqs"], dtype=float)
SOURCE_PARAMS = _inj["source_params"]

from numpy.fft import rfft, rfftfreq
freqs = rfftfreq(len(data_At_full), dt)
data_f = rfft(data_At_full)
df = 1.0 / t_obs

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)

params = SOURCE_PARAMS[0].copy()
k_center = int(np.rint(params[0] * t_obs))
kmin = max(k_center - jgb.n, 0)
kmax = min(k_center + jgb.n, len(freqs))

band_freqs = freqs[kmin:kmax]
noise_psd_full = np.maximum(np.interp(freqs, freqs_saved, noise_psd_saved), 1e-60)
noise_band = noise_psd_full[kmin:kmax]

data_j = jnp.asarray(data_f[kmin:kmax], dtype=jnp.complex128)
psd_j = jnp.asarray(noise_band, dtype=jnp.float64)

@jax.jit
def get_likelihood(p):
    A, _, _ = jgb.sum_tdi(p.reshape(1, -1), kmin=int(kmin), kmax=int(kmax), tdi_combination="AET")
    h = jnp.asarray(A, dtype=jnp.complex128).reshape(-1)
    res = dt * (data_j - h)
    whittle = -jnp.sum(2.0 * df * jnp.abs(res)**2 / psd_j)
    return whittle

f0_true = params[0]
df_grid = np.linspace(-1e-8, 1e-8, 100)
lnLs = []
for delta in df_grid:
    p2 = params.copy()
    p2[0] = f0_true + delta
    phi0_ref = p2[7]
    t_c = t_obs / 2.0
    phi_c = phi0_ref + 2 * np.pi * f0_true * t_c + np.pi * p2[1] * t_c**2
    p2[7] = phi_c - 2 * np.pi * p2[0] * t_c - np.pi * p2[1] * t_c**2
    lnLs.append(float(get_likelihood(jnp.array(p2))))

import json
print(json.dumps({
    "max_lnL": max(lnLs),
    "min_lnL": min(lnLs)
}))
