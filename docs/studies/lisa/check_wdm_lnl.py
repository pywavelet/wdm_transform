import jax
import jax.numpy as jnp
import numpy as np
import lisaorbits
from jaxgb.jaxgb import JaxGB
import sys
sys.path.append(".")

jax.config.update("jax_enable_x64", True)

_inj = np.load("outdir_gb_background/injection.npz")
dt = float(_inj["dt"])
data_At_full = np.asarray(_inj["data_At"], dtype=float)
SOURCE_PARAMS = _inj["source_params"]

from wdm_transform import TimeSeries
nt_param = 128
n_keep = (len(data_At_full) // (2*nt_param)) * (2*nt_param)
data_At = data_At_full[:n_keep]
t_obs = n_keep * dt
NF = n_keep // nt_param
probe = TimeSeries(data_At, dt=dt).to_wdm(nt=nt_param)
data_wdm = probe.coeffs
freq_grid = probe.freq_grid

noise_psd_saved = np.asarray(_inj["noise_psd_A"], dtype=float)
freqs_saved = np.asarray(_inj["freqs"], dtype=float)
noise_psd = np.maximum(np.interp(freq_grid, freqs_saved, noise_psd_saved), 1e-60)
from lisa_common import wdm_noise_variance, trim_frequency_band
noise_var = wdm_noise_variance(noise_psd, freq_grid, nt_param)

band = trim_frequency_band(freq_grid, SOURCE_PARAMS[:, 0].min() - 1.5e-4, SOURCE_PARAMS[:, 0].max() + 1.5e-4, pad_bins=2)
data_band_jax = jnp.asarray(data_wdm[:, band], dtype=jnp.float64)
noise_var_band_jax = jnp.asarray(noise_var[:, band], dtype=jnp.float64)

orbit_model = lisaorbits.EqualArmlengthOrbits()
jgb = JaxGB(orbit_model, t_obs=t_obs, t0=0.0, n=256)

n_freqs = n_keep // 2 + 1
half = nt_param // 2
kmin_rfft = max((band.start - 1) * half, 0)
kmax_rfft = min(band.stop * half, n_freqs)
band_rfft_size = kmax_rfft - kmin_rfft

from wdm_transform.backends import get_backend
from wdm_transform.windows import phi_window
_window_j = jnp.asarray(phi_window(get_backend("jax"), nt_param, NF, dt, 1/3, 1), dtype=jnp.complex128)

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

def get_h(params, src_idx):
    kmin_i = max(int(np.rint(SOURCE_PARAMS[src_idx][0] * t_obs)) - jgb.n, 0)
    kmax_i = min(kmin_i + 2 * jgb.n, n_freqs)
    a_loc, _, _ = jgb.sum_tdi(params.reshape(1, -1), kmin=kmin_i, kmax=kmax_i, tdi_combination="AET")
    local_start = kmin_i - kmin_rfft
    local_end = kmax_i - kmin_rfft
    x_local = jnp.zeros(band_rfft_size, dtype=jnp.complex128).at[local_start:local_end].set(a_loc)
    return _wdm_band_from_local_rfft(x_local, _window_j, nt_param, NF, kmin_rfft, band.start, band.stop)

@jax.jit
def lnL_wdm(p):
    h = get_h(p, 0) + get_h(jnp.asarray(SOURCE_PARAMS[1], dtype=jnp.float64), 1)
    diff = data_band_jax - h
    return -0.5 * jnp.sum(diff**2 / noise_var_band_jax)

p_eval = SOURCE_PARAMS[0].copy()
hess = jax.hessian(lnL_wdm)(jnp.array(p_eval))

subset_idx = np.array([0, 1, 2, 7])
sub_hess = hess[np.ix_(subset_idx, subset_idx)]
cov = np.linalg.inv(-sub_hess)
print("WDM Std devs:")
print(np.sqrt(np.diag(cov)))
