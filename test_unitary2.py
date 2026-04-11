import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from wdm_transform.windows import phi_window
from wdm_transform.backends import get_backend

N = 128 * 128
dt = 1.0
t = np.arange(N) * dt
np.random.seed(42)
x = np.random.normal(0, 1.5, size=N)

from wdm_transform import TimeSeries
w_ref = TimeSeries(x, dt=dt).to_wdm(nt=128).coeffs

def _wdm_band_from_local_rfft(
    x_local: jnp.ndarray,
    window: jnp.ndarray,
    nt: int,
    nf: int,
    kmin_rfft: int,
    band_start: int,
    band_stop: int,
):
    _half = nt // 2
    narr = jnp.arange(nt)
    band_m = jnp.arange(band_start, band_stop)

    upper = band_m[:, None] * _half + jnp.arange(_half)[None, :]
    lower = (band_m[:, None] - 1) * _half + jnp.arange(_half)[None, :]
    mid_local = jnp.concatenate([upper, lower], axis=1) - kmin_rfft

    mid_blocks = x_local[mid_local] * window[None, :]
    mid_times = jnp.fft.ifft(mid_blocks, axis=1).T

    parity = jnp.where((narr[:, None] + band_m[None, :]) % 2 == 0, 1.0, -1.0)
    mid_phase = jnp.conj(jnp.exp((1j * jnp.pi / 4.0) * (1.0 - parity)))

    return (jnp.sqrt(2.0) / nf) * jnp.real(mid_phase * mid_times)

X_f = np.fft.rfft(x)
window = phi_window(get_backend("jax"), 128, 128, dt, 1/3, 1.0)

w_jax = _wdm_band_from_local_rfft(
    jnp.array(X_f, dtype=jnp.complex128),
    jnp.array(window, dtype=jnp.complex128),
    128,
    128,
    0,
    1,
    127
)

eval_jax = np.sum(np.array(w_jax)**2)
eval_ref = np.sum(w_ref[:, 1:127]**2)

print(f"JAX WDM Energy: {eval_jax}")
print(f"Ref WDM Energy: {eval_ref}")
print(f"Ratio: {eval_jax / eval_ref}")
print(f"Max abs diff: {np.max(np.abs(np.array(w_jax) - w_ref[:, 1:127]))}")
