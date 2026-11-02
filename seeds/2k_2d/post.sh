#!/bin/bash
# seeds/2k_2d after training: test, striatal opto, SNc opto per seed (GPU, one job at a time), then scaling screen and summary.
# Same stages as cbtModel1D seeds/run_seeds.sh (the seeds/2k_tau pipeline).
export XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=2000
D=seeds/2k_2d; P=$D/progress.log
until grep -q TRAIN_DONE $P; do sleep 30; done
echo "--- post $(date +%F_%T) ---" >> $P
for s in 1 2 3 4 5 6 7 8 9 10; do
  for st in test opto nmopto; do
    python seeds/run_seed.py $s $st 2k_2d/seed_$s > $D/seed_$s/log_$st.txt 2>&1
    echo "$(date +%T) seed=$s stage=$st exit=$?" >> $P
  done
done
python seeds/scaling_screen.py 2k_2d > $D/log_scaling.txt 2>&1
echo "$(date +%T) scaling exit=$?" >> $P
python seeds/summarize_1k.py 2k_2d > $D/log_summary.txt 2>&1
echo "$(date +%T) summary exit=$?" >> $P
echo "$(date +%T) ALL_DONE" >> $P
