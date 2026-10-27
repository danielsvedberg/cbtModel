#!/bin/bash
# usage: bash seeds/run_all.sh <seed> [<seed> ...]   (from repo root)
# Run only one instance at a time: concurrent GPU jobs broke bit-reproducibility (see seeds/README.md).
for s in "$@"; do
  mkdir -p seeds/seed_$s
  for st in train test opto pnr; do
    XLA_PYTHON_CLIENT_PREALLOCATE=false python seeds/run_seed.py $s $st > seeds/seed_$s/log_$st.txt 2>&1
    echo "seed=$s stage=$st exit=$?" >> seeds/progress.log
  done
done
