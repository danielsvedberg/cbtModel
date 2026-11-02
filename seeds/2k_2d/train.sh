#!/bin/bash
# seeds/2k_2d training: init seeds 1-10, 2000 iters, cbtModel rank-2 DA gain (s_d1 -> dSPN, s_d2 -> iSPN), nm_bias(c), per-step best selection, one GPU job at a time.
export XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=2000
D=seeds/2k_2d; P=$D/progress.log; echo "--- train $(date +%F_%T) ---" >> $P
for s in 1 2 3 4 5 6 7 8 9 10; do
  mkdir -p $D/seed_$s
  python seeds/run_seed.py $s train 2k_2d/seed_$s > $D/seed_$s/log_train.txt 2>&1
  echo "$(date +%T) seed=$s stage=train exit=$?" >> $P
done
echo "$(date +%T) TRAIN_DONE" >> $P
