#SBATCH --job-name=lisa_pp
#SBATCH --output=outdir_lisa/slurm_logs/pp_%A_%a.out
#SBATCH --error=outdir_lisa/slurm_logs/pp_%A_%a.err
#SBATCH --array=0-99%100
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/wdm_transform/.venv/bin/activate


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


echo "Job ${SLURM_JOB_ID:-manual} task ${TASK_ID}: seed=${SEED}, mode=galactic_background"

echo "=== galactic_background seed ${SEED} ==="
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" data_generation.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" lisa_freq_mcmc.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" lisa_wdm_mcmc.py
LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" "${PYTHON_BIN}" post_proc.py

echo "Completed seed ${SEED} for mode=galactic_background"
