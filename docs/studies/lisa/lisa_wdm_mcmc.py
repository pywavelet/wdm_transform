# # WDM-Domain LISA GB Inference With NumPyro
#
# This study builds a compact LISA-like example around the cached
# `data_generation.py` background realization:
#
# - start from a TDI 1.5 A-channel background realization
# - inject two Galactic binaries with `JaxGB`
# - transform the data and model into the WDM domain
# - sample the source parameters with a diagonal WDM Whittle likelihood
#
# To keep the example docs-friendly, the inference problem is intentionally
# narrow:
#
# - it uses only the A channel
# - it keeps sky location and orientation fixed at the injected values
# - it infers only `(f0, amplitude, phase0)` for two sources
#
# The WDM noise surface is built from a stationary one-sided PSD via the simple
# approximation
#
# $$
# S[n,m] \approx S(f_m)\,\Delta F,
# $$
#
# tiled across all time bins, with the likelihood evaluated as
#
# $$
# \log p(d \mid \theta)
# = -\frac12 \sum_{n,m}
# \left[
# \frac{(d[n,m]-h[n,m;\theta])^2}{S[n,m]}
# + \log(2\pi S[n,m])
# \right].
# $$

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import corner
import jax
import jax.numpy as jnp
import lisaorbits
import numpyro
import numpyro.distributions as dist
from jaxgb.jaxgb import JaxGB
from numpyro.infer import MCMC, NUTS, init_to_value


from lisa_common import (
    BACKGROUND_REALIZATION_PATH,
    WDM_ASSET_DIR,
    galactic_psd,
    noise_tdi15_a_psd,
    save_figure,
    wrap_phase,
)

from wdm_transform import TimeSeries
from wdm_transform.transforms import from_time_to_wdm

jax.config.update("jax_enable_x64", True)


FIGURE_OUTPUT_DIR = WDM_ASSET_DIR
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POSTERIOR_OUTPUT_PATH = FIGURE_OUTPUT_DIR / "wdm_posterior_samples.npz"

RNG = np.random.default_rng(12)
A_WDM = 1.0 / 3.0
D_WDM = 1.0
NT = 128


def stationary_a_background_psd(f):
    """Fallback stationary A-channel PSD proxy."""
    return noise_tdi15_a_psd(f) + galactic_psd(f)


def draw_real_noise_from_psd(psd_func, n, dt, rng):
    """Draw one real colored-noise realization from a one-sided PSD shape."""
    freqs = np.fft.rfftfreq(n, d=dt)
    white = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    white[0] = rng.normal()
    if n % 2 == 0:
        white[-1] = rng.normal()
    shaped = np.sqrt(np.maximum(psd_func(freqs), 0.0)) * white / np.sqrt(2.0)
    return np.fft.irfft(shaped, n=n)


def load_cached_background(path: Path):
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "path": path,
        "dt": float(data["dt"]),
        "T_obs": float(data["T_obs"]),
        "N": int(data["N"]),
        "freqs": np.asarray(data["freqs_all"], dtype=float),
        "a_time": np.asarray(data["data_At"], dtype=float),
        "source_params": np.asarray(data["source_params"], dtype=float),
        "noise_psd_A": np.asarray(data["noise_psd_A"], dtype=float),
    }


def place_local_tdi_jax(segment: jnp.ndarray, kmin, n_freqs: int) -> jnp.ndarray:
    segment = jnp.asarray(segment, dtype=jnp.complex128).reshape(-1)
    full = jnp.zeros((n_freqs,), dtype=jnp.complex128)
    max_start = max(n_freqs - segment.shape[0], 0)
    start = jnp.clip(jnp.asarray(kmin, dtype=jnp.int32), 0, max_start)
    return jax.lax.dynamic_update_slice(full, segment, (start,))


def build_jaxgb_a_time_generator(delta_t: float, T: float):
    orbit_model = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit_model, t_obs=T, t0=0.0, n=256)
    n_samples = int(round(T / delta_t))
    n_freqs = n_samples // 2 + 1

    def generate(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.asarray(params, dtype=jnp.float64)
        a_loc, _, _ = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
        kmin = jnp.asarray(jgb.get_kmin(params[None, 0:1])).reshape(-1)[0]
        a_full = place_local_tdi_jax(a_loc, kmin, n_freqs)
        return jnp.fft.irfft(a_full, n=n_samples)

    return generate


def stationary_wdm_variance_from_psd(
    freq_grid: np.ndarray,
    freqs_psd: np.ndarray,
    psd_values: np.ndarray,
    nt: int,
    delta_f: float,
) -> np.ndarray:
    psd_on_wdm = np.interp(
        np.asarray(freq_grid, dtype=float),
        np.asarray(freqs_psd, dtype=float),
        np.asarray(psd_values, dtype=float),
        left=float(np.asarray(psd_values, dtype=float)[0]),
        right=float(np.asarray(psd_values, dtype=float)[-1]),
    )
    channel_var = np.maximum(psd_on_wdm * delta_f, 1e-30)
    return np.tile(channel_var[None, :], (nt, 1))


def trim_frequency_band(coeffs, freq_grid, f_lo, f_hi, pad_bins=2):
    freq_grid = np.asarray(freq_grid)
    keep = np.where((freq_grid >= f_lo) & (freq_grid <= f_hi))[0]
    if keep.size == 0:
        raise ValueError("Requested frequency band does not overlap the WDM grid.")
    start = max(int(keep[0]) - pad_bins, 0)
    stop = min(int(keep[-1]) + pad_bins + 1, coeffs.shape[1])
    return slice(start, stop)


def save_posterior_artifacts(
    path: Path,
    posterior: dict[str, np.ndarray],
    theta_samples: np.ndarray,
    labels: list[str],
    truth: np.ndarray,
    setup: InferenceSetup,
    source_params: np.ndarray,
) -> None:
    """Save posterior products to a compact NPZ for cross-script comparisons."""
    med = np.median(theta_samples, axis=0)
    lo = np.percentile(theta_samples, 5, axis=0)
    hi = np.percentile(theta_samples, 95, axis=0)
    np.savez(
        path,
        theta_samples=np.asarray(theta_samples, dtype=float),
        labels=np.asarray(labels, dtype=str),
        truth=np.asarray(truth, dtype=float),
        theta_median=np.asarray(med, dtype=float),
        theta_ci05=np.asarray(lo, dtype=float),
        theta_ci95=np.asarray(hi, dtype=float),
        source_params=np.asarray(source_params, dtype=float),
        f1=np.asarray(posterior["f1"], dtype=float),
        A1=np.asarray(posterior["A1"], dtype=float),
        phi1=np.asarray(posterior["phi1"], dtype=float),
        f2=np.asarray(posterior["f2"], dtype=float),
        A2=np.asarray(posterior["A2"], dtype=float),
        phi2=np.asarray(posterior["phi2"], dtype=float),
        dt=float(setup.dt),
        nt=int(setup.nt),
        nf=int(setup.nf),
        band_start=int(setup.band.start),
        band_stop=int(setup.band.stop),
    )


@dataclass(frozen=True)
class InferenceSetup:
    fixed_params: np.ndarray
    data_wdm: np.ndarray
    noise_var: np.ndarray
    nt: int
    nf: int
    dt: float
    band: slice
    generator: object
    prior_center: np.ndarray
    prior_scale: np.ndarray
    logA_bounds: tuple[tuple[float, float], tuple[float, float]]
    freq_bounds: tuple[tuple[float, float], tuple[float, float]]


def wrap_phase_jax(phi: jnp.ndarray) -> jnp.ndarray:
    return (phi + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def template_from_theta_jax(theta: jnp.ndarray, setup: InferenceSetup) -> jnp.ndarray:
    params = jnp.asarray(setup.fixed_params, dtype=jnp.float64)
    params = params.at[0, 0].set(theta[0])
    params = params.at[0, 2].set(theta[1])
    params = params.at[0, 7].set(wrap_phase_jax(theta[2]))
    params = params.at[1, 0].set(theta[3])
    params = params.at[1, 2].set(theta[4])
    params = params.at[1, 7].set(wrap_phase_jax(theta[5]))

    signal = setup.generator(params[0]) + setup.generator(params[1])
    coeffs = from_time_to_wdm(
        signal,
        nt=setup.nt,
        nf=setup.nf,
        a=A_WDM,
        d=D_WDM,
        dt=setup.dt,
        backend="jax",
    )
    return coeffs[:, setup.band]


def numpyro_wdm_model(setup: InferenceSetup) -> None:
    delta_f1 = numpyro.sample("delta_f1", dist.Normal(0.0, 1.0))
    delta_logA1 = numpyro.sample("delta_logA1", dist.Normal(0.0, 1.0))
    delta_phi1 = numpyro.sample("delta_phi1", dist.Normal(0.0, 1.0))
    delta_f2 = numpyro.sample("delta_f2", dist.Normal(0.0, 1.0))
    delta_logA2 = numpyro.sample("delta_logA2", dist.Normal(0.0, 1.0))
    delta_phi2 = numpyro.sample("delta_phi2", dist.Normal(0.0, 1.0))

    f1 = numpyro.deterministic(
        "f1",
        setup.prior_center[0] + setup.prior_scale[0] * delta_f1,
    )
    logA1 = numpyro.deterministic(
        "logA1",
        np.log(setup.fixed_params[0, 2]) + setup.prior_scale[1] * delta_logA1,
    )
    phi1 = numpyro.deterministic(
        "phi1",
        setup.prior_center[2] + setup.prior_scale[2] * delta_phi1,
    )
    f2 = numpyro.deterministic(
        "f2",
        setup.prior_center[3] + setup.prior_scale[3] * delta_f2,
    )
    logA2 = numpyro.deterministic(
        "logA2",
        np.log(setup.fixed_params[1, 2]) + setup.prior_scale[4] * delta_logA2,
    )
    phi2 = numpyro.deterministic(
        "phi2",
        setup.prior_center[5] + setup.prior_scale[5] * delta_phi2,
    )

    A1 = numpyro.deterministic("A1", jnp.exp(logA1))
    A2 = numpyro.deterministic("A2", jnp.exp(logA2))
    theta = jnp.array(
        [
            f1,
            A1,
            wrap_phase_jax(phi1),
            f2,
            A2,
            wrap_phase_jax(phi2),
        ],
        dtype=jnp.float64,
    )

    numpyro.factor(
        "f1_support",
        jnp.where(
            (theta[0] >= setup.freq_bounds[0][0])
            & (theta[0] <= setup.freq_bounds[0][1]),
            0.0,
            -jnp.inf,
        ),
    )
    numpyro.factor(
        "f2_support",
        jnp.where(
            (theta[3] >= setup.freq_bounds[1][0])
            & (theta[3] <= setup.freq_bounds[1][1]),
            0.0,
            -jnp.inf,
        ),
    )
    numpyro.factor(
        "a1_support",
        jnp.where(
            (logA1 >= setup.logA_bounds[0][0]) & (logA1 <= setup.logA_bounds[0][1]),
            0.0,
            -jnp.inf,
        ),
    )
    numpyro.factor(
        "a2_support",
        jnp.where(
            (logA2 >= setup.logA_bounds[1][0]) & (logA2 <= setup.logA_bounds[1][1]),
            0.0,
            -jnp.inf,
        ),
    )

    model = template_from_theta_jax(theta, setup)
    data = jnp.asarray(setup.data_wdm[:, setup.band], dtype=jnp.float64)
    var = jnp.maximum(
        jnp.asarray(setup.noise_var[:, setup.band], dtype=jnp.float64),
        1e-30,
    )
    diff = data - model
    log_like = -0.5 * jnp.sum(diff**2 / var + jnp.log(2.0 * jnp.pi * var))
    numpyro.factor("wdm_whittle", log_like)


cached_background = load_cached_background(BACKGROUND_REALIZATION_PATH)

if cached_background is None:
    raise FileNotFoundError(
        f"Expected injected data at {BACKGROUND_REALIZATION_PATH}. "
        "Run data_generation.py first."
    )

dt = cached_background["dt"]
T_OBS = cached_background["T_obs"]
n_keep = int(cached_background["N"])
SOURCE_PARAMS = cached_background["source_params"]
background = cached_background["a_time"]
background_mode = f"injected data from {cached_background['path'].name}"

required_block = 2 * NT
if n_keep % required_block != 0:
    raise ValueError(
        f"N = {n_keep} is not divisible by 2*NT = {required_block}."
    )
NF = n_keep // NT

print(f"Background source: {background_mode}")
print(f"Using T_obs = {T_OBS / 86400:.2f} days, dt = {dt:.1f} s, N = {n_keep}, nt = {NT}, nf = {NF}")

# Data is already injected, so just use it directly
data = background

data_wdm = np.asarray(TimeSeries(data, dt=dt).to_wdm(nt=NT).coeffs)
signal_wdm = None  # Not computing this for injected data
noise_wdm = None  # Not computing this for injected data
probe = TimeSeries(data, dt=dt).to_wdm(nt=NT)

# Use the saved noise PSD
freqs_psd = cached_background["freqs"]
psd_values = cached_background["noise_psd_A"]

delta_f = float(np.asarray(probe.freq_grid)[1] - np.asarray(probe.freq_grid)[0])
noise_var = stationary_wdm_variance_from_psd(
    probe.freq_grid,
    freqs_psd,
    psd_values,
    NT,
    delta_f,
)
band = trim_frequency_band(
    data_wdm,
    probe.freq_grid,
    SOURCE_PARAMS[:, 0].min() - 1.5e-4,
    SOURCE_PARAMS[:, 0].max() + 1.5e-4,
    pad_bins=2,
)

# Create a JaxGB generator for template generation during inference
generate_a = build_jaxgb_a_time_generator(delta_t=dt, T=T_OBS)

fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 4),
    constrained_layout=True,
)
mesh = ax.pcolormesh(
    np.asarray(probe.time_grid),
    np.asarray(probe.freq_grid),
    np.log(data_wdm**2 + 1e-30).T,
    shading="nearest",
    cmap="viridis",
)
ax.axhspan(
    float(np.asarray(probe.freq_grid)[band.start]),
    float(np.asarray(probe.freq_grid)[band.stop - 1]),
    color="white",
    alpha=0.08,
)
ax.set_title("Injected WDM data")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
fig.colorbar(mesh, ax=ax, label="log local power")
_ = save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_overview")

setup = InferenceSetup(
    fixed_params=SOURCE_PARAMS,
    data_wdm=data_wdm,
    noise_var=noise_var,
    nt=NT,
    nf=NF,
    dt=dt,
    band=band,
    generator=generate_a,
    prior_center=np.array(
        [
            SOURCE_PARAMS[0, 0],
            np.log(SOURCE_PARAMS[0, 2]),
            SOURCE_PARAMS[0, 7],
            SOURCE_PARAMS[1, 0],
            np.log(SOURCE_PARAMS[1, 2]),
            SOURCE_PARAMS[1, 7],
        ],
        dtype=float,
    ),
    prior_scale=np.array(
        [
            2.0e-6,
            0.35,
            np.pi / 2.0,
            2.0e-6,
            0.35,
            np.pi / 2.0,
        ],
        dtype=float,
    ),
    logA_bounds=(
        (
            float(np.log(0.3 * SOURCE_PARAMS[0, 2])),
            float(np.log(3.0 * SOURCE_PARAMS[0, 2])),
        ),
        (
            float(np.log(0.3 * SOURCE_PARAMS[1, 2])),
            float(np.log(3.0 * SOURCE_PARAMS[1, 2])),
        ),
    ),
    freq_bounds=(
        (SOURCE_PARAMS[0, 0] - 8.0e-6, SOURCE_PARAMS[0, 0] + 8.0e-6),
        (SOURCE_PARAMS[1, 0] - 8.0e-6, SOURCE_PARAMS[1, 0] + 8.0e-6),
    ),
)

init_values = {
    "delta_f1": 0.0,
    "delta_logA1": 0.0,
    "delta_phi1": 0.0,
    "delta_f2": 0.0,
    "delta_logA2": 0.0,
    "delta_phi2": 0.0,
}
kernel = NUTS(
    lambda: numpyro_wdm_model(setup),
    init_strategy=init_to_value(values=init_values),
    target_accept_prob=0.9,
)
mcmc = MCMC(
    kernel,
    num_warmup=200,
    num_samples=400,
    num_chains=1,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(21))

posterior = mcmc.get_samples()
theta_samples = np.column_stack(
    [
        np.asarray(posterior["f1"]),
        np.asarray(posterior["A1"]),
        np.asarray(posterior["phi1"]),
        np.asarray(posterior["f2"]),
        np.asarray(posterior["A2"]),
        np.asarray(posterior["phi2"]),
    ]
)
theta_med = np.median(theta_samples, axis=0)
map_wdm = np.asarray(template_from_theta_jax(jnp.asarray(theta_med), setup))

labels = [
    "source 1 frequency [Hz]",
    "source 1 amplitude",
    "source 1 phase [rad]",
    "source 2 frequency [Hz]",
    "source 2 amplitude",
    "source 2 phase [rad]",
]
truth = np.array(
    [
        SOURCE_PARAMS[0, 0],
        SOURCE_PARAMS[0, 2],
        wrap_phase(SOURCE_PARAMS[0, 7]),
        SOURCE_PARAMS[1, 0],
        SOURCE_PARAMS[1, 2],
        wrap_phase(SOURCE_PARAMS[1, 7]),
    ]
)
save_posterior_artifacts(
    path=POSTERIOR_OUTPUT_PATH,
    posterior=posterior,
    theta_samples=theta_samples,
    labels=labels,
    truth=truth,
    setup=setup,
    source_params=SOURCE_PARAMS,
)
print(f"Saved WDM posterior samples to: {POSTERIOR_OUTPUT_PATH}")

print("\nWDM NumPyro posterior summary")
med = np.median(theta_samples, axis=0)
lo = np.percentile(theta_samples, 5, axis=0)
hi = np.percentile(theta_samples, 95, axis=0)
for label, true_value, median, low, high in zip(
    labels, truth, med, lo, hi, strict=True
):
    print(
        f"{label:24s} true={true_value:.6e}  "
        f"median={median:.6e}  90% CI=[{low:.6e}, {high:.6e}]"
    )

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    constrained_layout=True,
    sharey=True,
)
meshes = []
for ax, coeffs, title in [
    (axes[0], data_wdm[:, band], "Data in fitted WDM band"),
    (axes[1], map_wdm, "Posterior median template in WDM band"),
]:
    mesh = ax.pcolormesh(
        np.asarray(probe.time_grid),
        np.asarray(probe.freq_grid)[band],
        np.log(coeffs**2 + 1e-30).T,
        shading="nearest",
        cmap="magma",
    )
    meshes.append(mesh)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
axes[0].set_ylabel("Frequency [Hz]")
for ax, mesh in zip(axes, meshes, strict=True):
    fig.colorbar(mesh, ax=ax, label="log local power")
_ = save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_map_fit")

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
x = np.arange(len(labels))
ax.scatter(x - 0.12, truth, label="Injection", s=40, color="tab:blue")
ax.scatter(
    x + 0.12,
    theta_med,
    label="WDM posterior median",
    s=40,
    color="tab:orange",
)
ax.set_xticks(x)
ax.set_xticklabels(
    ["f1", "A1", "phi1", "f2", "A2", "phi2"],
    rotation=30,
    ha="right",
)
ax.set_title("Injected parameters versus WDM posterior median")
ax.legend()
_ = save_figure(fig, FIGURE_OUTPUT_DIR, "parameter_recovery")

if corner is not None:
    fig = corner.corner(
        theta_samples,
        labels=[r"$f_1$", r"$A_1$", r"$\phi_1$", r"$f_2$", r"$A_2$", r"$\phi_2$"],
        truths=truth,
        truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
    )
    _ = save_figure(fig, FIGURE_OUTPUT_DIR, "wdm_corner")

# %% [markdown]
# ![WDM overview](../lisa_wdm_mcmc_assets/wdm_overview.png)
# ![Band-limited WDM fit](../lisa_wdm_mcmc_assets/wdm_map_fit.png)
# ![Parameter recovery](../lisa_wdm_mcmc_assets/parameter_recovery.png)
# ![WDM posterior corner](../lisa_wdm_mcmc_assets/wdm_corner.png)
