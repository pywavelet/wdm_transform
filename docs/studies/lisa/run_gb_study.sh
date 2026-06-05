#!/usr/bin/env bash
# Run the full LISA galactic-binary freq-vs-WDM study in one command.
#
# Per seed: draw a source (f0, fdot, A, random sky, phi0) + a target SNR in
# [SNR_MIN, SNR_MAX], inject it via JaxGB on A/E/T with LISA colored noise, and
# fit (f0, fdot, A, phi0) with NUTS in both the frequency and the WDM domain.
# Writes per-seed results to outdir_gb/ and a PP plot comparing the two domains.
#
# Each seed runs in its OWN process: a different injected f0 changes the static
# band indices, so JAX recompiles per seed; isolating seeds keeps memory bounded
# and makes the batch crash-resistant and resumable (existing seeds are skipped).
#
# Usage:  ./run_gb_study.sh [N_SEEDS]   (default 100)
set -uo pipefail
cd "$(dirname "$0")"

N=${1:-100}
PY=lisa_venv/bin/python

for ((s = 0; s < N; s++)); do
    if [[ -f "outdir_gb/seed_${s}.json" ]]; then
        echo "[run] seed ${s} already done, skipping"
        continue
    fi
    echo "[run] seed ${s} ($((s + 1))/${N})"
    "$PY" lisa_gb_study.py --seed "$s" || echo "[run] seed ${s} FAILED, continuing"
done

echo "[run] aggregating + PP plot"
"$PY" summarize_gb_study.py
