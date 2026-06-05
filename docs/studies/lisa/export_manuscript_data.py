"""Snapshot the LISA GB study outputs into the manuscript's ``src/data/``.

The manuscript figure scripts must build under showyourwork without the heavy
galactic-binary waveform stack (jax/jaxgb/lisaorbits).  This script runs the
expensive bits once here (using the study venv) and writes two lightweight,
committed ``.npz`` snapshots that the figure scripts read with plain NumPy:

  lisa_gb_demo.npz   one demo seed: channel-A injection (rFFT data + signal +
                     instrument PSD), the analysis band, the freq/WDM posterior
                     samples, the injected truth, and the prior scales.
  lisa_gb_pp.npz     all seeds: per-parameter truth ranks in both domains.

Usage:  lisa_venv/bin/python export_manuscript_data.py [--seed 8]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outdir_gb"
MS_DATA = HERE.parents[1] / "manuscript" / "src" / "data"


def _all_seed_files() -> list[Path]:
    files = glob.glob(str(OUTDIR / "seed_*.json"))
    return sorted((Path(f) for f in files), key=lambda p: int(p.stem.split("_")[1]))


def export_demo(seed: int) -> Path:
    """Snapshot one seed's injection + posteriors for the data and corner figures."""
    import jax

    jax.config.update("jax_enable_x64", True)

    from lisa_gb_study import (
        POSTERIOR_LABELS,
        _make_grid,
        _prior_scales,
        injection_for_seed,
        make_jgb,
    )

    grid = _make_grid()
    jgb = make_jgb(grid)
    inj = injection_for_seed(seed, grid=grid, jgb=jgb)
    scales = _prior_scales(grid["t_obs"])

    record = json.load(open(OUTDIR / f"seed_{seed}.json"))
    truth_display = np.array([next(e["truth"] for e in record["freq"] if e["label"] == lab)
                              for lab in POSTERIOR_LABELS], dtype=float)
    samples_freq = np.column_stack([np.asarray(record["samples"]["freq"][lab], float)
                                    for lab in POSTERIOR_LABELS])
    samples_wdm = np.column_stack([np.asarray(record["samples"]["wdm"][lab], float)
                                   for lab in POSTERIOR_LABELS])

    MS_DATA.mkdir(parents=True, exist_ok=True)
    out = MS_DATA / "lisa_gb_demo.npz"
    np.savez_compressed(
        out,
        seed=seed,
        labels=np.asarray(POSTERIOR_LABELS),
        dt=float(inj["dt"]),
        n_total=int(grid["n_total"]),
        t_obs=float(inj["t_obs"]),
        f0=float(inj["truth"]["f0"]),
        snr=float(record["snr"]),
        freqs=np.asarray(inj["freqs"], dtype=float),
        data_rfft=np.asarray(inj["data_rfft"][0], dtype=np.complex128),
        signal_rfft=np.asarray(inj["signal_rfft"][0], dtype=np.complex128),
        psd_inst=np.asarray(inj["psd_full"][0], dtype=float),
        band_kmin=int(inj["band"]["kmin_rfft"]),
        band_kmax=int(inj["band"]["kmax_rfft"]),
        truth_display=truth_display,
        samples_freq=samples_freq,
        samples_wdm=samples_wdm,
        prior_f0_sigma=float(scales["delta_f0_sigma"]),
        prior_fdot_sigma=float(scales["delta_fdot_sigma"]),
    )
    print(f"[demo] seed {seed} (SNR {record['snr']:.1f}) -> {out}")
    return out


def export_pp() -> Path:
    """Snapshot per-parameter truth ranks (both domains) over all seeds."""
    from lisa_gb_study import POSTERIOR_LABELS

    files = _all_seed_files()
    records = [json.load(open(f)) for f in files]
    ranks = {dom: np.array([[next(e["rank"] for e in r[dom] if e["label"] == lab)
                             for lab in POSTERIOR_LABELS] for r in records], dtype=float)
             for dom in ("freq", "wdm")}

    MS_DATA.mkdir(parents=True, exist_ok=True)
    out = MS_DATA / "lisa_gb_pp.npz"
    np.savez_compressed(
        out,
        labels=np.asarray(POSTERIOR_LABELS),
        n_seeds=len(records),
        ranks_freq=ranks["freq"],
        ranks_wdm=ranks["wdm"],
    )
    print(f"[pp] {len(records)} seeds -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8, help="demo seed for the data + corner figures")
    args = ap.parse_args()
    export_demo(args.seed)
    export_pp()


if __name__ == "__main__":
    main()
