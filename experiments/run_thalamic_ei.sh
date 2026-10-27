#!/bin/bash
# Thalamic E/I analysis on every network with PNR-determinant data (CPU). From repo root: bash experiments/run_thalamic_ei.sh [seed ...]
SEEDS=${@:-13 17 9 21 29 22 7 23 1 3 4 24 6}
export JAX_PLATFORMS=cpu
mkdir -p experiments/thalamic_ei
for k in $SEEDS; do
  python experiments/thalamic_ei.py $k > experiments/thalamic_ei/log_seed$k.txt 2>&1
  echo "$(date +%T) seed=$k exit=$?" >> experiments/thalamic_ei/progress.log
done
echo "$(date +%T) DONE" >> experiments/thalamic_ei/progress.log
