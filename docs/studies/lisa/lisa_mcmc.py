"""Frequency- and WDM-domain LISA GB inference with NumPyro."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import arviz_base as azb
import arviz_plots as azp
import jax
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from gb_prior import build_local_prior_info, lisa_delta_f0_prior_sigma
from jaxgb.jaxgb import JaxGB
from lisa_common import (
    interp_psd_channels,
    lisa_run_dir,
    load_injection,
    setup_jax_and_matplotlib,
    trim_frequency_band,
)
from numpy.fft import rfft, rfftfreq
from numpyro.infer import MCMC, NUTS

from wdm_transform.signal_processing import wdm_noise_variance
from wdm_transform.transforms import forward_wdm_band

setup_jax_and_matplotlib()
jax.config.update("jax_enable_x64", True)

N_WARMUP = int(os.getenv("LISA_N_WARMUP", "1500"))
N_DRAWS = int(os.getenv("LISA_N_DRAWS", "1000"))
NUM_CHAINS = 2
NT = int(os.getenv("LISA_NT", "32"))
NUTS_KWARGS = {
    "dense_mass": True,
    "target_accept_prob": 0.95,
    "max_tree_depth": 12,
}
INIT_JITTER_SCALE = 0.15
A_WDM = 1.0 / 3.0
D_WDM = 1.0
POSTERIOR_VARS = ("f0", "fdot", "A", "phi0")
POSTERIOR_LABELS = ["f0 [Hz]", "fdot [Hz/s]", "A", "phi0 [rad]"]

numpyro.set_host_device_count(NUM_CHAINS)


def run_paths(seed: int) -> dict[str, Path]:
    run_dir = lisa_run_dir(seed=seed)
    return {
        "run_dir": run_dir,
        "injection": run_dir / "injection.npz",
        "freq": run_dir / "freq_posterior.nc",
        "wdm": run_dir / "wdm_posterior.nc",
        "diagnostics": run_dir / "posterior_diagnostics",
    }


def ensure_injection(seed: int, paths: dict[str, Path]) -> None:
    if paths["injection"].exists():
        return

    env = os.environ.copy()
    env["LISA_SEED"] = str(seed)
    script = Path(__file__).with_name("data_generation.py")
    print(f"Missing {paths['injection']}; generating seed {seed} data.")
    subprocess.run([sys.executable, str(script)], check=True, env=env)


def aet_rfft(jgb: JaxGB, params: jnp.ndarray, kmin: int, kmax: int) -> jnp.ndarray:
    """Return local A/E/T Fourier modes as ``(3, nfreq)``."""
    return jnp.stack(
        [
            jnp.asarray(mode, dtype=jnp.complex128).reshape(-1)
            for mode in jgb.sum_tdi(
                jnp.asarray(params, dtype=jnp.float64).reshape(1, -1),
                kmin=kmin,
                kmax=kmax,
                tdi_combination="AET",
                tdi_generation=1.5,
            )
        ]
    )


def local_rfft_to_wdm(
    modes: jnp.ndarray,
    *,
    src_kmin: int,
    src_kmax: int,
    kmin_rfft: int,
    band_rfft_size: int,
    band_start: int,
    band_stop: int,
    df_rfft: float,
    n_freqs: int,
    nf: int,
    nt: int,
) -> jnp.ndarray:
    local_start = src_kmin - kmin_rfft
    local_end = src_kmax - kmin_rfft

    def one_channel(mode: jnp.ndarray) -> jnp.ndarray:
        embedded = (
            jnp.zeros(band_rfft_size, dtype=jnp.complex128)
            .at[local_start:local_end]
            .set(mode)
        )
        return forward_wdm_band(
            embedded,
            df=df_rfft,
            nfreqs_fourier=n_freqs,
            kmin=kmin_rfft,
            nfreqs_wdm=nf,
            ntimes_wdm=nt,
            mmin=band_start,
            nf_sub_wdm=band_stop - band_start,
            a=A_WDM,
            d=D_WDM,
            backend="jax",
        )

    return jnp.stack([one_channel(mode) for mode in modes])


def prior_metadata(injection) -> dict[str, np.ndarray | float | tuple[float, float]]:
    prior = build_local_prior_info(
        prior_f0=injection.prior_f0,
        prior_fdot=injection.prior_fdot,
        prior_A=injection.prior_A,
    )
    return {
        "f0_ref": float(injection.f0_ref),
        "delta_f0_bounds": (
            float(injection.prior_f0[0] - injection.f0_ref),
            float(injection.prior_f0[1] - injection.f0_ref),
        ),
        "delta_f0_sigma": float(lisa_delta_f0_prior_sigma()),
        "prior_center": prior.prior_center,
        "prior_scale": prior.prior_scale,
        "logfdot_bounds": prior.logfdot_bounds,
        "logA_bounds": prior.logA_bounds,
    }


def build_frequency_band(injection, source_param: np.ndarray, jgb: JaxGB) -> dict:
    data = np.stack([injection.data_At, injection.data_Et, injection.data_Tt])
    freqs = rfftfreq(data.shape[1], injection.dt)
    kmin = max(int(np.floor(injection.prior_f0[0] * injection.t_obs)) - jgb.n, 0)
    kmax = min(
        int(np.ceil(injection.prior_f0[1] * injection.t_obs)) + jgb.n + 1, len(freqs)
    )
    noise_psd = interp_psd_channels(
        freqs,
        injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    return {
        **prior_metadata(injection),
        "domain": "freq",
        "fixed_params": source_param.copy(),
        "data": rfft(data, axis=1)[:, kmin:kmax],
        "noise_psd": noise_psd[:, kmin:kmax],
        "whittle_weight": 2.0 * (1.0 / injection.t_obs) * injection.dt**2,
        "band_kmin": kmin,
        "band_kmax": kmax,
        "dt": injection.dt,
        "t_obs": injection.t_obs,
    }


def build_wdm_band(injection, source_param: np.ndarray, jgb: JaxGB) -> dict:
    n_keep = (len(injection.data_At) // (2 * NT)) * (2 * NT)
    data = np.stack(
        [
            injection.data_At[:n_keep],
            injection.data_Et[:n_keep],
            injection.data_Tt[:n_keep],
        ]
    )
    t_obs = n_keep * injection.dt
    nf = n_keep // NT
    n_freqs = n_keep // 2 + 1
    df_rfft = 1.0 / t_obs
    freq_grid = np.linspace(0.0, 0.5 / injection.dt, nf + 1)
    margin = jgb.n / t_obs
    band_slice = trim_frequency_band(
        freq_grid,
        injection.prior_f0[0] - margin,
        injection.prior_f0[1] + margin,
        pad_bins=2,
    )
    half = NT // 2
    kmin_rfft = max((band_slice.start - 1) * half, 0)
    kmax_rfft = min(band_slice.stop * half, n_freqs)
    band_kwargs = {
        "df": df_rfft,
        "nfreqs_fourier": n_freqs,
        "kmin": kmin_rfft,
        "nfreqs_wdm": nf,
        "ntimes_wdm": NT,
        "mmin": band_slice.start,
        "nf_sub_wdm": band_slice.stop - band_slice.start,
        "a": A_WDM,
        "d": D_WDM,
        "backend": "jax",
    }
    data_rfft = np.fft.rfft(data, axis=1)
    band_freqs = freq_grid[band_slice]
    noise_psd = interp_psd_channels(
        band_freqs,
        injection.freqs,
        np.stack([injection.noise_psd_A, injection.noise_psd_E, injection.noise_psd_T]),
    )
    wdm_psd = (2 * (n_freqs - 1)) * np.stack(
        [wdm_noise_variance(psd, nt=NT, dt=injection.dt) for psd in noise_psd]
    )
    return {
        **prior_metadata(injection),
        "domain": "wdm",
        "fixed_params": source_param.copy(),
        "data": np.stack(
            [
                np.asarray(
                    forward_wdm_band(
                        jnp.asarray(channel[kmin_rfft:kmax_rfft]), **band_kwargs
                    )
                )
                for channel in data_rfft
            ]
        ),
        "noise_psd": wdm_psd,
        "src_kmin": max(int(np.floor(injection.prior_f0[0] * t_obs)) - jgb.n, 0),
        "src_kmax": min(
            int(np.ceil(injection.prior_f0[1] * t_obs)) + jgb.n + 1, n_freqs
        ),
        "kmin_rfft": kmin_rfft,
        "band_rfft_size": kmax_rfft - kmin_rfft,
        "band_start": band_slice.start,
        "band_stop": band_slice.stop,
        "df_rfft": df_rfft,
        "n_freqs": n_freqs,
        "nf": nf,
        "nt": NT,
        "t_obs": t_obs,
    }


def build_init_values(band: dict, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fixed_params = np.asarray(band["fixed_params"], dtype=float)
    delta_f0_bounds = np.asarray(band["delta_f0_bounds"], dtype=float)
    delta_f0_sigma = float(band["delta_f0_sigma"])
    z_bounds = delta_f0_bounds / max(delta_f0_sigma, 1e-30)
    prior_center = np.asarray(band["prior_center"], dtype=float)
    logfdot_bounds = np.asarray(band["logfdot_bounds"], dtype=float)
    logA_bounds = np.asarray(band["logA_bounds"], dtype=float)
    return {
        "z_f0": float(
            np.clip(
                (fixed_params[0] - float(band["f0_ref"])) / max(delta_f0_sigma, 1e-30)
                + INIT_JITTER_SCALE * rng.standard_normal(),
                z_bounds[0] + 1e-6,
                z_bounds[1] - 1e-6,
            )
        ),
        "logfdot": float(
            np.clip(
                prior_center[1] + INIT_JITTER_SCALE * rng.standard_normal(),
                logfdot_bounds[0] + 1e-6,
                logfdot_bounds[1] - 1e-6,
            )
        ),
        "logA": float(
            np.clip(
                prior_center[2] + INIT_JITTER_SCALE * rng.standard_normal(),
                logA_bounds[0] + 1e-6,
                logA_bounds[1] - 1e-6,
            )
        ),
        "phi0": float(
            np.clip(
                fixed_params[7] + INIT_JITTER_SCALE * rng.standard_normal(),
                -np.pi + 1e-6,
                np.pi - 1e-6,
            )
        ),
    }


def frequency_whittle_lnl(
    residual: jnp.ndarray,
    noise_psd: jnp.ndarray,
    whittle_weight: float,
) -> jnp.ndarray:
    residual_power = jnp.real(jnp.conj(residual) * residual)
    return -jnp.sum(jnp.log(noise_psd) + whittle_weight * residual_power / noise_psd)


def wdm_gaussian_lnl(residual: jnp.ndarray, noise_psd: jnp.ndarray) -> jnp.ndarray:
    residual_power = jnp.real(jnp.conj(residual) * residual)
    return -0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * noise_psd) + residual_power / noise_psd
    )


def run_sampler(jgb: JaxGB, band: dict, seed: int) -> MCMC:
    data = jnp.asarray(
        band["data"], dtype=jnp.complex128 if band["domain"] == "freq" else jnp.float64
    )
    noise_psd = jnp.asarray(band["noise_psd"], dtype=jnp.float64)
    fixed_params = jnp.asarray(band["fixed_params"], dtype=jnp.float64)
    prior_center = jnp.asarray(band["prior_center"], dtype=jnp.float64)
    prior_scale = jnp.asarray(band["prior_scale"], dtype=jnp.float64)
    delta_f0_sigma = float(band["delta_f0_sigma"])
    z_low, z_high = np.asarray(band["delta_f0_bounds"], dtype=float) / max(
        delta_f0_sigma, 1e-30
    )
    logfdot_low, logfdot_high = np.asarray(band["logfdot_bounds"], dtype=float)
    logA_low, logA_high = np.asarray(band["logA_bounds"], dtype=float)

    def model() -> None:
        z_f0 = numpyro.sample(
            "z_f0", dist.TruncatedNormal(0.0, 1.0, low=float(z_low), high=float(z_high))
        )
        delta_f0 = numpyro.deterministic("delta_f0", delta_f0_sigma * z_f0)
        logfdot = numpyro.sample(
            "logfdot",
            dist.TruncatedNormal(
                prior_center[1],
                prior_scale[1],
                low=float(logfdot_low),
                high=float(logfdot_high),
            ),
        )
        logA = numpyro.sample(
            "logA",
            dist.TruncatedNormal(
                prior_center[2],
                prior_scale[2],
                low=float(logA_low),
                high=float(logA_high),
            ),
        )
        f0 = float(band["f0_ref"]) + delta_f0
        fdot = jnp.exp(logfdot)
        amplitude = jnp.exp(logA)
        phi0 = numpyro.sample("phi0", dist.Uniform(-jnp.pi, jnp.pi))
        params = (
            fixed_params.at[0]
            .set(f0)
            .at[1]
            .set(fdot)
            .at[2]
            .set(amplitude)
            .at[7]
            .set(phi0)
        )
        if band["domain"] == "freq":
            template = aet_rfft(
                jgb, params, int(band["band_kmin"]), int(band["band_kmax"])
            )
            loglike = frequency_whittle_lnl(
                data - template,
                noise_psd,
                float(band["whittle_weight"]),
            )
        else:
            wdm_kwargs = {
                key: int(band[key])
                for key in (
                    "src_kmin",
                    "src_kmax",
                    "kmin_rfft",
                    "band_rfft_size",
                    "band_start",
                    "band_stop",
                    "n_freqs",
                    "nf",
                    "nt",
                )
            }
            wdm_kwargs["df_rfft"] = float(band["df_rfft"])
            template = local_rfft_to_wdm(
                aet_rfft(jgb, params, wdm_kwargs["src_kmin"], wdm_kwargs["src_kmax"]),
                **wdm_kwargs,
            )
            loglike = wdm_gaussian_lnl(data - template, noise_psd)
        numpyro.factor(f"{band['domain']}_loglike", loglike)
        numpyro.deterministic("f0", f0)
        numpyro.deterministic("fdot", fdot)
        numpyro.deterministic("A", amplitude)

    mcmc = MCMC(
        NUTS(model, **NUTS_KWARGS),
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=NUM_CHAINS,
        progress_bar=True,
    )
    inits = [build_init_values(band, seed + 7 * i) for i in range(NUM_CHAINS)]
    mcmc.run(
        jax.random.PRNGKey(seed),
        init_params={key: jnp.array([init[key] for init in inits]) for key in inits[0]},
        extra_fields=("diverging",),
    )
    return mcmc


def save_arviz_outputs(mcmc: MCMC, domain: str, output_path: Path) -> Path:
    idata = azb.from_numpyro(mcmc)
    posterior = azb.convert_to_dataset(idata, group="posterior")[list(POSTERIOR_VARS)]
    posterior.attrs.clear()
    for variable in posterior.variables.values():
        variable.attrs.clear()
        variable.encoding.clear()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    posterior.to_netcdf(output_path)

    azp.plot_trace(
        idata,
        var_names=list(POSTERIOR_VARS),
        backend="matplotlib",
    )
    fig = plt.gcf()
    fig.savefig(
        output_path.with_name(f"{domain}_trace.png"), dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    return output_path


def wrap_phase(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def normalize_phase_columns(samples: np.ndarray, labels: list[str]) -> np.ndarray:
    out = np.asarray(samples, dtype=float).copy()
    for idx, label in enumerate(labels):
        if "phi" in label.lower() or "phase" in label.lower():
            out[:, idx] = wrap_phase(out[:, idx])
    return out


def load_posterior_dataset(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as posterior:
        return posterior.load()


def posterior_samples(posterior: xr.Dataset) -> np.ndarray:
    samples = np.column_stack(
        [np.asarray(posterior[name]).reshape(-1) for name in POSTERIOR_VARS]
    )
    return normalize_phase_columns(samples, POSTERIOR_LABELS)


def jensen_shannon_bits(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    lo = float(min(np.min(samples_a), np.min(samples_b)))
    hi = float(max(np.max(samples_a), np.max(samples_b)))
    if not np.isfinite(lo + hi) or lo == hi:
        return 0.0
    counts_a, edges = np.histogram(samples_a, bins=64, range=(lo, hi), density=False)
    counts_b, _ = np.histogram(samples_b, bins=edges, density=False)
    p = counts_a / max(float(np.sum(counts_a)), 1.0)
    q = counts_b / max(float(np.sum(counts_b)), 1.0)
    m = 0.5 * (p + q)

    def kl_bits(lhs: np.ndarray, rhs: np.ndarray) -> float:
        keep = lhs > 0.0
        return float(np.sum(lhs[keep] * np.log2(lhs[keep] / rhs[keep])))

    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def marginal_jsd(
    samples_a: np.ndarray, samples_b: np.ndarray, labels: list[str]
) -> list[dict[str, float | str]]:
    return [
        {
            "label": label,
            "jsd_bits": jensen_shannon_bits(samples_a[:, idx], samples_b[:, idx]),
        }
        for idx, label in enumerate(labels)
    ]


def save_distribution_plot(
    wdm: xr.Dataset,
    freq: xr.Dataset,
    output_dir: Path,
) -> None:
    azp.plot_dist(
        {
            "Frequency": xr.DataTree.from_dict({"posterior": freq}),
            "WDM": xr.DataTree.from_dict({"posterior": wdm}),
        },
        var_names=list(POSTERIOR_VARS),
        backend="matplotlib",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(output_dir / "posterior_dist.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_compare(paths: dict[str, Path]) -> None:
    wdm_path = paths["wdm"]
    freq_path = paths["freq"]
    output_dir = paths["diagnostics"]
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [path for path in (wdm_path, freq_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing posterior archive(s): {missing}")

    wdm = load_posterior_dataset(wdm_path)
    freq = load_posterior_dataset(freq_path)
    samples_wdm = posterior_samples(wdm)
    samples_freq = posterior_samples(freq)
    jsd = marginal_jsd(samples_wdm, samples_freq, POSTERIOR_LABELS)
    lines = ["Marginal Jensen-Shannon divergence:"]
    for row in jsd:
        lines.append(f"  {row['label']}: JSD={row['jsd_bits']:.6e} bits")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    (output_dir / "posterior_diagnostics.txt").write_text(report, encoding="utf-8")
    (output_dir / "posterior_diagnostics.json").write_text(
        json.dumps({"comparison": {"marginal_jsd": jsd}}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    save_distribution_plot(wdm, freq, output_dir)


def run_domain(domain: str, paths: dict[str, Path]) -> None:
    start = time.perf_counter()
    injection_path = paths["injection"]
    if not injection_path.exists():
        raise FileNotFoundError(
            f"Expected cached injection at {injection_path}. "
            "Run data_generation.py first."
        )
    injection = load_injection(injection_path)
    source_param = injection.source_params[0].copy()
    t_obs = (
        injection.t_obs
        if domain == "freq"
        else (len(injection.data_At) // (2 * NT)) * (2 * NT) * injection.dt
    )
    jgb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=t_obs, t0=0.0, n=256)
    band = (
        build_frequency_band(injection, source_param, jgb)
        if domain == "freq"
        else build_wdm_band(injection, source_param, jgb)
    )
    mcmc_seed = injection.seed + 10

    print(
        f"{domain}: T_obs={float(band['t_obs']) / 86400.0:.1f}d, "
        f"seed={injection.seed}, MCMC_seed={mcmc_seed}, chains={NUM_CHAINS}"
    )
    mcmc = run_sampler(jgb, band, mcmc_seed)
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    if divergences:
        print(f"WARNING: {divergences} divergences")

    output_path = save_arviz_outputs(mcmc, domain, paths[domain])
    print(f"Saved {domain} posterior to {output_path}")
    print(f"[lisa_mcmc.py {domain}] runtime: {time.perf_counter() - start:.2f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", type=int, default=0)
    args = parser.parse_args(argv)
    paths = run_paths(args.seed)
    ensure_injection(args.seed, paths)
    run_domain("freq", paths)
    run_domain("wdm", paths)
    run_compare(paths)


if __name__ == "__main__":
    main()
