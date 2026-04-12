import numpy as np
from wdm_transform import TimeSeries
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

N = 128 * 128
dt = 1.0
t = np.arange(N) * dt
np.random.seed(42)
x = np.random.normal(0, 1.5, size=N)

energy_t = np.sum(x**2)

probe = TimeSeries(x, dt=dt).to_wdm(nt=128)
w = probe.coeffs
energy_w = np.sum(w**2)

X_f = np.fft.rfft(x)
energy_f = (1.0/N) * (np.abs(X_f[0])**2 + np.abs(X_f[-1])**2 + 2*np.sum(np.abs(X_f[1:-1])**2))

print(f"Energy Time: {energy_t}")
print(f"Energy WDM:  {energy_w} (ratio w/t = {energy_w/energy_t})")
print(f"Energy Freq: {energy_f} (ratio f/t = {energy_f/energy_t})")

import sys
sys.path.append("docs/studies/lisa")
from lisa_wdm_mcmc import _wdm_band_from_local_rfft, _window_j

try:
    w_jax = _wdm_band_from_local_rfft(
        jnp.array(X_f, dtype=jnp.complex128),
        _window_j,
        128,
        128,
        0,
        1,
        127
    )
    energy_wjax = np.sum(np.array(w_jax)**2)
    energy_wref = np.sum(w[:, 1:127]**2)
    print(f"Energy JAX WDM (channels 1-127): {energy_wjax}")
    print(f"Energy Ref WDM (channels 1-127): {energy_wref}")
    print(f"Ratio JAX/Ref: {energy_wjax/energy_wref}")
except Exception as e:
    print(e)
