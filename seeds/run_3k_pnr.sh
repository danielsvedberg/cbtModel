#!/bin/bash
# PNR stage for the 3k batch on the GPU (on CPU, XLA's LLVM compile runs out of memory).
# Waits for run_3k.sh's GPU lane to finish so it never shares the GPU with training.
P=seeds/3k/progress.log
until grep -q GPU_LANE_DONE $P; do sleep 60; done
for d in seeds/3k/seed_*; do
  s=${d##*_}
  [ -f $d/metrics_train.json ] && [ ! -f $d/metrics_pnr.json ] || continue
  XLA_PYTHON_CLIENT_PREALLOCATE=false python seeds/run_seed.py $s pnr 3k/seed_$s > $d/log_pnr.txt 2>&1
  echo "$(date +%T) seed=$s stage=pnr_gpu exit=$? json=$([ -f $d/metrics_pnr.json ] && echo yes || echo no)" >> $P
done
echo "$(date +%T) PNR_DONE" >> $P
