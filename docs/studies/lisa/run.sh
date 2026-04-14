#!/usr/bin/env bash
set -euo pipefail

# Run the seeded LISA study pipeline.
#
# Examples:
#   ./run.sh
#   MODE=stationary_noise ./run.sh
#   MODE=galactic_background START_SEED=0 END_SEED=9 ./run.sh
#   MODE=both START_SEED=0 END_SEED=9 LISA_N_WARMUP=400 LISA_N_DRAWS=600 ./run.sh

MODE="${MODE:-both}"
START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-9}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"

run_mode() {
  local include_galactic="$1"
  local mode_name="$2"
  local seed

  for seed in $(seq "$START_SEED" "$END_SEED"); do
    echo
    echo "=== ${mode_name} seed ${seed} ==="
    LISA_SEED="$seed" LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN data_generation.py
    LISA_SEED="$seed" LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN lisa_freq_mcmc.py
    LISA_SEED="$seed" LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN lisa_wdm_mcmc.py
    LISA_SEED="$seed" LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN post_proc.py
  done

  echo
  echo "=== ${mode_name} PP plot ==="
  LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN pp_plot.py
}

case "$MODE" in
  stationary_noise)
    run_mode 0 "stationary_noise"
    ;;
  galactic_background)
    run_mode 1 "galactic_background"
    ;;
  both)
    run_mode 0 "stationary_noise"
    run_mode 1 "galactic_background"
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Expected one of: stationary_noise, galactic_background, both" >&2
    exit 1
    ;;
esac
