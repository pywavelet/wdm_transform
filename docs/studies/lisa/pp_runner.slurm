#!/usr/bin/env bash
# Slurm array runner for LISA PP tests.
#
# Default behavior:
# - Runs seeds 0..99 (100 total)
# - Runs up to 100 array tasks concurrently
# - Sets 20-minute walltime per seed
# - Always runs galactic_background mode
# - Executes data_generation.py -> lisa_freq_mcmc.py -> lisa_wdm_mcmc.py -> post_proc.py
#
# Submit examples:
#   sbatch docs/studies/lisa/pp_runner.slurm

#SBATCH --job-name=lisa_pp
#SBATCH --output=docs/studies/lisa/outdir_lisa/slurm_logs/pp_%A_%a.out
#SBATCH --error=docs/studies/lisa/outdir_lisa/slurm_logs/pp_%A_%a.err
#SBATCH --array=0-99%100
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/wdm_transform/.venv/bin/activate

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
STUDY_DIR="${STUDY_DIR:-${PROJECT_ROOT}/docs/studies/lisa}"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-99}"

if (( END_SEED < START_SEED )); then
	echo "END_SEED must be >= START_SEED (got START_SEED=${START_SEED}, END_SEED=${END_SEED})" >&2
	exit 1
fi

TOTAL_SEEDS=$((END_SEED - START_SEED + 1))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL_SEEDS )); then
	echo "No work for array task ${TASK_ID}; TOTAL_SEEDS=${TOTAL_SEEDS}. Exiting."
	exit 0
fi

SEED=$((START_SEED + TASK_ID))

mkdir -p "${STUDY_DIR}/outdir_lisa/slurm_logs"
cd "${STUDY_DIR}"

echo "Job ${SLURM_JOB_ID:-manual} task ${TASK_ID}: seed=${SEED}, mode=galactic_background"

echo "=== galactic_background seed ${SEED} ==="
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" data_generation.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" lisa_freq_mcmc.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" lisa_wdm_mcmc.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" post_proc.py

echo "Completed seed ${SEED} for mode=galactic_background"
echo "Tip: run pp_plot.py once after all array tasks finish."
