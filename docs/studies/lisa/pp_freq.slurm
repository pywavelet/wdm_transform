#!/usr/bin/env bash
# Frequency-domain MCMC for 100 seeds in parallel.
# Runtime per seed: ~5 min.  Walltime set to 10 min (2x headroom).
#
# Submit (from docs/studies/lisa/ or from repo root):
#   sbatch docs/studies/lisa/pp_freq.slurm

#SBATCH --job-name=lisa_freq
#SBATCH --output=outdir_lisa/slurm_logs/freq_%A_%a.out
#SBATCH --error=outdir_lisa/slurm_logs/freq_%A_%a.err
#SBATCH --array=0-99%100
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outdir_lisa/slurm_logs

module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/wdm_transform/.venv/bin/activate

export PYTHONUNBUFFERED=1

START_SEED="${START_SEED:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((START_SEED + TASK_ID))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START lisa_freq_mcmc seed=${SEED}"
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" python lisa_freq_mcmc.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END   lisa_freq_mcmc seed=${SEED}"
