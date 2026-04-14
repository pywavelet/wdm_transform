"""Empirical diagnostics for the fixed LISA injection/analysis prior.

Draws sources from the fixed prior, evaluates their matched-filter SNR against
the saved PSD grid from an existing injection, and reports how often the source
population falls inside a requested SNR window.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
from jaxgb.jaxgb import JaxGB
from lisa_common import INJECTION_PATH, draw_source_prior_and_params, load_injection
from wdm_transform.signal_processing import matched_filter_snr_rfft

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of prior draws to test.")
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for prior draws.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not INJECTION_PATH.exists():
        raise FileNotFoundError(
            f"Expected reference PSD grid at {INJECTION_PATH}. Run data_generation.py first."
        )

    inj = load_injection(INJECTION_PATH)
    rng = np.random.default_rng(args.seed)
    orbit_model = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit_model, t_obs=inj.t_obs, t0=0.0, n=256)

    freqs = inj.freqs
    n_freqs = len(freqs)
    psd_A = np.maximum(inj.noise_psd_A, 1e-60)
    psd_E = np.maximum(inj.noise_psd_E, 1e-60)
    psd_T = np.maximum(inj.noise_psd_T, 1e-60)

    def source_frequency_series(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params_j = jnp.asarray(params, dtype=jnp.float64)
        a_loc, e_loc, t_loc = jgb.get_tdi(params_j, tdi_generation=1.5, tdi_combination="AET")
        kmin = int(jnp.asarray(jgb.get_kmin(params_j[None, 0:1])).reshape(-1)[0])
        a = np.zeros(n_freqs, dtype=np.complex128)
        e = np.zeros(n_freqs, dtype=np.complex128)
        t = np.zeros(n_freqs, dtype=np.complex128)
        a[kmin:kmin + len(a_loc)] = np.asarray(a_loc)
        e[kmin:kmin + len(e_loc)] = np.asarray(e_loc)
        t[kmin:kmin + len(t_loc)] = np.asarray(t_loc)
        return a, e, t

    snrs = []
    samples = []
    for _ in range(args.n):
        source, prior_f0, prior_fdot, prior_A = draw_source_prior_and_params(rng)
        source_Af, source_Ef, source_Tf = source_frequency_series(source)
        snr_A = matched_filter_snr_rfft(source_Af, psd_A, freqs, dt=inj.dt)
        snr_E = matched_filter_snr_rfft(source_Ef, psd_E, freqs, dt=inj.dt)
        snr_T = matched_filter_snr_rfft(source_Tf, psd_T, freqs, dt=inj.dt)
        snr = float(np.sqrt(snr_A**2 + snr_E**2 + snr_T**2))
        snrs.append(snr)
        samples.append(source[:3])

    snrs = np.asarray(snrs)
    samples = np.asarray(samples)
    in_window = (snrs >= args.snr_min) & (snrs <= args.snr_max)

    print(f"Reference PSD source: {INJECTION_PATH}")
    print(f"Draws: {args.n}")
    print(f"SNR window: [{args.snr_min:.1f}, {args.snr_max:.1f}]")
    print(f"In-window fraction: {np.mean(in_window):.3f}")
    print(
        "SNR quantiles:"
        f" min={np.min(snrs):.2f}  p05={np.quantile(snrs, 0.05):.2f}"
        f"  p50={np.quantile(snrs, 0.50):.2f}  p95={np.quantile(snrs, 0.95):.2f}"
        f"  max={np.max(snrs):.2f}"
    )
    print(
        "Parameter ranges from draws:"
        f" f0=[{samples[:, 0].min():.6e}, {samples[:, 0].max():.6e}]"
        f" fdot=[{samples[:, 1].min():.6e}, {samples[:, 1].max():.6e}]"
        f" A=[{samples[:, 2].min():.6e}, {samples[:, 2].max():.6e}]"
    )
    print("Saved prior bounds used for every draw:")
    print(f"  f0   = [{prior_f0[0]:.6e}, {prior_f0[1]:.6e}] Hz")
    print(f"  fdot = [{prior_fdot[0]:.6e}, {prior_fdot[1]:.6e}] Hz/s")
    print(f"  A    = [{prior_A[0]:.6e}, {prior_A[1]:.6e}]")


if __name__ == "__main__":
    main()
