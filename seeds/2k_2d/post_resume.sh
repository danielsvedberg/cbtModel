#!/bin/bash
# Resume post.sh after the 19:11 halt: seeds 8-10 (seed 8 test was interrupted), then scaling screen and summary.
export XLA_PYTHON_CLIENT_PREALLOCATE=false ITERS=2000
D=seeds/2k_2d; P=$D/progress.log
echo "--- post resume $(date +%F_%T) ---" >> $P
for s in 8 9 10; do
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
