#!/usr/bin/env bash
# Run data_generation.py for all seeds on the head node.
#
# Each seed takes ~1 min (stochastic foreground realization dominates;
# the response tensor Rtildeop_tf.npz is built once and reused).
# 100 seeds ~ 100 min total.
#
# Usage:
#   bash generate_all.sh
#   START_SEED=0 END_SEED=9 bash generate_all.sh

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${STUDY_DIR}"

module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/wdm_transform/.venv/bin/activate

export PYTHONUNBUFFERED=1

START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-99}"

echo "Generating data for seeds ${START_SEED}..${END_SEED}"
echo "Started at: $(date)"

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] data_generation seed=${seed}"
    LISA_SEED="${seed}" LISA_INCLUDE_GALACTIC="1" python data_generation.py
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All data generation complete."
