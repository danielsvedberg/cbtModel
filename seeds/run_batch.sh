#!/bin/bash
# Seed batch at ITERS iterations into seeds/$OUT (from repo root): ITERS=2000 OUT=2k bash seeds/run_batch.sh <deadline epoch s> <seed> [<seed> ...]
# GPU lane trains one seed at a time (concurrent GPU training broke reproducibility);
# GPU lane also runs pnr after each training; CPU lane runs test/opto on each trained seed with JAX_PLATFORMS=cpu so it never shares the GPU.
# No new training starts after the deadline. Output: seeds/$OUT/seed_<k>/, progress in seeds/$OUT/progress.log. Copy of run_3k.sh with ITERS/OUT parameterized.
DEADLINE=$1; shift; : ${ITERS:?}; : ${OUT:?}
P=seeds/$OUT/progress.log; mkdir -p seeds/$OUT
TAG="run $$ $(date +%F_%T)"; echo "--- $TAG ---" >> $P
new() { sed -n "/--- $TAG ---/,\$p" $P; }  # only this run's lines
gpu_lane() {
  for s in "$@"; do
    [ $(date +%s) -ge $DEADLINE ] && { echo "$(date +%T) deadline reached, stop training" >> $P; break; }
    mkdir -p seeds/$OUT/seed_$s
    XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=$ITERS python seeds/run_seed.py $s train $OUT/seed_$s > seeds/$OUT/seed_$s/log_train.txt 2>&1
    echo "$(date +%T) seed=$s stage=train exit=$?" >> $P
    # PNR needs the GPU (XLA's CPU compile runs out of memory); run it here so the GPU still has one job at a time
    XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=$ITERS python seeds/run_seed.py $s pnr $OUT/seed_$s > seeds/$OUT/seed_$s/log_pnr.txt 2>&1
    echo "$(date +%T) seed=$s stage=pnr exit=$? json=$([ -f seeds/$OUT/seed_$s/metrics_pnr.json ] && echo yes || echo no)" >> $P
  done
  echo "$(date +%T) GPU_LANE_DONE" >> $P
}
cpu_lane() {
  for s in "$@"; do
    until new | grep -q "seed=$s stage=train" || new | grep -q GPU_LANE_DONE; do sleep 30; done
    new | grep -q "seed=$s stage=train" || break
    for st in test opto; do
      JAX_PLATFORMS=cpu ITERS=$ITERS python seeds/run_seed.py $s $st $OUT/seed_$s > seeds/$OUT/seed_$s/log_$st.txt 2>&1
      echo "$(date +%T) seed=$s stage=$st exit=$?" >> $P
    done
  done
  echo "$(date +%T) ALL_DONE" >> $P
}
gpu_lane "$@" &
cpu_lane "$@" &
wait
