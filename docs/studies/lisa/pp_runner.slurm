#!/usr/bin/env bash
#SBATCH --job-name=lisa_pp
#SBATCH --output=outdir_lisa/slurm_logs/pp_%A_%a.out
#SBATCH --error=outdir_lisa/slurm_logs/pp_%A_%a.err
#SBATCH --array=0-99%100
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

set -euo pipefail

module load gcc/13.3.0 python/3.12.3
source /fred/oz303/avajpeyi/codes/wdm_transform/.venv/bin/activate

export PYTHONUNBUFFERED=1
export LISA_PROGRESS_BAR="${LISA_PROGRESS_BAR:-1}"


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


run_stage() {
	local stage_name="$1"
	shift
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${stage_name} seed=${SEED}"
	"$@"
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] END   ${stage_name} seed=${SEED}"
}


echo "Job ${SLURM_JOB_ID:-manual} task ${TASK_ID}: seed=${SEED}, mode=galactic_background"

echo "=== galactic_background seed ${SEED} ==="
run_stage data_generation env LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" python data_generation.py
run_stage lisa_freq_mcmc env LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" python lisa_freq_mcmc.py
run_stage lisa_wdm_mcmc env LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" python lisa_wdm_mcmc.py
run_stage post_proc env LISA_SEED="${SEED}" LISA_INCLUDE_GALACTIC="1" python post_proc.py

echo "Completed seed ${SEED} for mode=galactic_background"
