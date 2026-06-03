#!/bin/zsh
cd /Users/avi/Documents/projects/wdm_transform/docs/studies/lisa
export LISA_N_WARMUP=2000 LISA_N_DRAWS=1000 LISA_NUM_CHAINS=2
PROG=outdir_lisa/_run_logs/progress.log
: > "$PROG"
echo "RUN START $(date)  seeds 0-99  warmup=$LISA_N_WARMUP draws=$LISA_N_DRAWS" >> "$PROG"
for s in $(seq 0 99); do
  t0=$SECONDS
  if lisa_venv/bin/python lisa_study.py --seed $s > outdir_lisa/_run_logs/seed_$s.log 2>&1; then
    echo "seed $s OK ($((SECONDS-t0))s) $(date +%H:%M:%S)" >> "$PROG"
  else
    echo "seed $s FAILED ($((SECONDS-t0))s) $(date +%H:%M:%S)" >> "$PROG"
  fi
done
echo "RUN DONE $(date)" >> "$PROG"
