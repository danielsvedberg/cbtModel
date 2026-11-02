#!/bin/bash
# seeds/2k_2d_m05: as seeds/2k_2d (rank-2 DA, nm_bias, per-step best selection, 2000 iters, init seeds 1-10) but with the
# SNc->DA readout m initialised at 0.5x (M_INIT_SCALE=0.5), so DA doesn't dominate the step-0 loss and SNc isn't silenced.
# One GPU job at a time. Train all seeds, then test/opto/nmopto per seed, scaling screen, summary.
export XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=2000 M_INIT_SCALE=0.5
D=seeds/2k_2d_m05; P=$D/progress.log; echo "--- train $(date +%F_%T) ---" >> $P
for s in 1 2 3 4 5 6 7 8 9 10; do
  mkdir -p $D/seed_$s
  python seeds/run_seed.py $s train 2k_2d_m05/seed_$s > $D/seed_$s/log_train.txt 2>&1
  echo "$(date +%T) seed=$s stage=train exit=$?" >> $P
done
echo "$(date +%T) TRAIN_DONE" >> $P
for s in 1 2 3 4 5 6 7 8 9 10; do
  for st in test opto nmopto; do
    python seeds/run_seed.py $s $st 2k_2d_m05/seed_$s > $D/seed_$s/log_$st.txt 2>&1
    echo "$(date +%T) seed=$s stage=$st exit=$?" >> $P
  done
done
python seeds/scaling_screen.py 2k_2d_m05 > $D/log_scaling.txt 2>&1
echo "$(date +%T) scaling exit=$?" >> $P
python seeds/summarize_1k.py 2k_2d_m05 > $D/log_summary.txt 2>&1
echo "$(date +%T) summary exit=$?" >> $P
echo "$(date +%T) ALL_DONE" >> $P
