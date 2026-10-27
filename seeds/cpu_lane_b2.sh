#!/bin/bash
# CPU lane for 3k batch 2 (the original lane saw batch 1's GPU_LANE_DONE and exited)
P=seeds/3k/progress.log
new() { sed -n '/batch 2/,$p' $P; }
for s in "$@"; do
  until new | grep -q "seed=$s stage=train" || new | grep -q GPU_LANE_DONE; do sleep 30; done
  new | grep -q "seed=$s stage=train" || break
  for st in test opto; do
    ITERS=3000 JAX_PLATFORMS=cpu python seeds/run_seed.py $s $st 3k/seed_$s > seeds/3k/seed_$s/log_$st.txt 2>&1
    echo "$(date +%T) seed=$s stage=$st exit=$?" >> $P
  done
done
echo "$(date +%T) BATCH2_DONE" >> $P
