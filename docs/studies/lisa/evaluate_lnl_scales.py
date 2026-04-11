import jax
import jax.numpy as jnp
import numpy as np
import lisaorbits
from jaxgb.jaxgb import JaxGB

jax.config.update("jax_enable_x64", True)
_inj = np.load("outdir_gb_background/injection.npz")
dt = float(_inj["dt"])
t_obs = float(_inj["t_obs"])
data_f = np.fft.rfft(np.asarray(_inj["data_At"], dtype=float))
df = 1.0 / t_obs
jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
params = _inj["source_params"][0].copy()

k_center = int(np.rint(params[0] * t_obs))
kmin = max(k_center - jgb.n, 0)
kmax = min(k_center + jgb.n, len(data_f))
data_j = jnp.asarray(data_f[kmin:kmax], dtype=jnp.complex128)
psd_j = jnp.asarray(np.maximum(np.interp(np.fft.rfftfreq(len(_inj["data_At"]), dt)[kmin:kmax], _inj["freqs"], _inj["noise_psd_A"]), 1e-60), dtype=jnp.float64)

@jax.jit
def lnL_full(p):
    A, _, _ = jgb.sum_tdi(p.reshape(1, -1), kmin=int(kmin), kmax=int(kmax), tdi_combination="AET")
    res = dt * (data_j - jnp.asarray(A, dtype=jnp.complex128).reshape(-1))
    return -jnp.sum(2.0 * df * jnp.abs(res)**2 / psd_j)

# Hessian
from jax import hessian
hess = hessian(lnL_full)(jnp.array(params))

# We care about indices: 0 (f0), 1 (fdot), 2 (A), 7 (phi0)
subset_idx = np.array([0, 1, 2, 7])
sub_hess = hess[np.ix_(subset_idx, subset_idx)]

print("Hessian wrt f0, fdot, A, phi0:")
print(sub_hess)

# Invert to get covariance
cov = np.linalg.inv(-sub_hess)
print("Covariance:")
print(cov)
print("Std devs:")
print(np.sqrt(np.diag(cov)))
