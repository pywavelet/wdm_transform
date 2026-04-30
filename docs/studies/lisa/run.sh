#!/usr/bin/env bash
set -euo pipefail

# Run the seeded LISA study pipeline.
#
# Examples:
#   ./run.sh
#   MODE=stationary_noise ./run.sh
#   MODE=galactic_background START_SEED=0 END_SEED=9 ./run.sh
#   MODE=galactic_background START_SEED=0 END_SEED=99 ./run.sh
#   MODE=both START_SEED=0 END_SEED=9 LISA_N_WARMUP=400 LISA_N_DRAWS=600 ./run.sh

MODE="${MODE:-both}"
START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-9}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
LISA_NUM_CHAINS="${LISA_NUM_CHAINS:-2}"
export LISA_NUM_CHAINS

# JAX needs this before Python starts to expose multiple CPU devices for
# parallel NumPyro chains. Users can still override XLA_FLAGS explicitly.
if [[ -z "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="--xla_force_host_platform_device_count=${LISA_NUM_CHAINS}"
fi

run_mode() {
  local include_galactic="$1"
  local mode_name="$2"
  local seed

  for seed in $(seq "$START_SEED" "$END_SEED"); do
    echo
    echo "=== ${mode_name} seed ${seed} ==="
    LISA_SEED="$seed" LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN data_generation.py
    LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN lisa_mcmc.py "$seed"
  done

  echo
  echo "=== ${mode_name} JSD summary ==="
  LISA_INCLUDE_GALACTIC="$include_galactic" $PYTHON_BIN collect_jsd.py \
    --mode "$mode_name" \
    --start-seed "$START_SEED" \
    --end-seed "$END_SEED"
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
