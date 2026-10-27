#!/bin/bash
# Cortex silenced from 100 ms after cue onset (cue offset) to trial end (stages ctx_inh_test / ctx_inh_pnr / ctx_inh_dspn_pnr) on the intact
# baseline and seeds 9, 17, 21. From repo root: bash experiments/run_ctx.sh [13|9|17|21 ...]  (13 = baseline params)
# Output: experiments/baseline/ctx_inh_*/ and experiments/seeds/seed_<k>/baseline/ctx_inh_*/. One GPU job at a time.
SEEDS=${@:-13 9 17 21}
mkdir -p experiments/seeds
P=experiments/seeds/progress.log
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "--- $(date +%F_%T) run_ctx $SEEDS ---" >> $P
for k in $SEEDS; do
  for st in ctx_inh_test ctx_inh_pnr ctx_inh_dspn_pnr; do
    if [ "$k" = 13 ]; then
      python experiments/run_experiment.py baseline $st > experiments/seeds/log_seed13_baseline_$st.txt 2>&1
    else
      CBT_SEED=$k python experiments/run_experiment.py baseline $st > experiments/seeds/log_seed${k}_baseline_$st.txt 2>&1
    fi
    echo "$(date +%T) seed=$k net=baseline stage=$st exit=$?" >> $P
  done
done
echo "$(date +%T) RUN_CTX_DONE" >> $P
