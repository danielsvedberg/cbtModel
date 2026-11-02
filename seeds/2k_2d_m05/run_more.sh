#!/bin/bash
# seeds/2k_2d_m05 extension: init seeds 11-30, same pipeline as run.sh; waits for run.sh's ALL_DONE (one GPU job at a time),
# then re-runs the scaling screen and summary over all seeds in the dir.
export XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=2000 M_INIT_SCALE=0.5
D=seeds/2k_2d_m05; P=$D/progress.log
until grep -q ALL_DONE $P; do sleep 30; done
SEEDS=$(seq 11 30)
echo "--- train seeds 11-30 $(date +%F_%T) ---" >> $P
for s in $SEEDS; do
  mkdir -p $D/seed_$s
  python seeds/run_seed.py $s train 2k_2d_m05/seed_$s > $D/seed_$s/log_train.txt 2>&1
  echo "$(date +%T) seed=$s stage=train exit=$?" >> $P
done
echo "$(date +%T) TRAIN2_DONE" >> $P
for s in $SEEDS; do
  for st in test opto nmopto; do
    python seeds/run_seed.py $s $st 2k_2d_m05/seed_$s > $D/seed_$s/log_$st.txt 2>&1
    echo "$(date +%T) seed=$s stage=$st exit=$?" >> $P
  done
done
python seeds/scaling_screen.py 2k_2d_m05 > $D/log_scaling.txt 2>&1
echo "$(date +%T) scaling exit=$?" >> $P
python seeds/summarize_1k.py 2k_2d_m05 > $D/log_summary.txt 2>&1
echo "$(date +%T) summary exit=$?" >> $P
echo "$(date +%T) ALL_DONE_30" >> $P
