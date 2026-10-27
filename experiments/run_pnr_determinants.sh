#!/bin/bash
# All PNR-determinant runs (CPU; ~1-3 min each). From repo root: bash experiments/run_pnr_determinants.sh [seed ...]  (default 13 9 17 21)
SEEDS=${@:-13 9 17 21}
export JAX_PLATFORMS=cpu
for k in $SEEDS; do for c in intact ctx_off snc_off snc_sat; do
  python experiments/pnr_determinants.py $k $c > experiments/pnr_determinants/log_seed${k}_$c.txt 2>&1
  echo "$(date +%T) seed=$k cond=$c exit=$?" >> experiments/pnr_determinants/progress.log
done; done
echo "$(date +%T) DONE" >> experiments/pnr_determinants/progress.log
