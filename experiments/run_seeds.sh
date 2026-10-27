#!/bin/bash
# Lesion, SNc inhibition/stimulation and PNR tests on the seeds closest to the baseline (3k scorecard 4/5: 9, 17, 21).
# From repo root: bash experiments/run_seeds.sh [seed ...]. One GPU job at a time. Output: experiments/seeds/seed_<k>/.
# Progress: experiments/seeds/progress.log. Each seed's intact test/opto results are already in seeds/3k/seed_<k>/.
SEEDS=${@:-9 17 21}
mkdir -p experiments/seeds
P=experiments/seeds/progress.log
export XLA_PYTHON_CLIENT_PREALLOCATE=false
go() {  # seed net stage
  CBT_SEED=$1 python experiments/run_experiment.py $2 $3 > experiments/seeds/log_seed$1_$2_$3.txt 2>&1
  echo "$(date +%T) seed=$1 net=$2 stage=$3 exit=$?" >> $P
}
echo "--- $(date +%F_%T) run_seeds $SEEDS ---" >> $P
for k in $SEEDS; do
  for st in pnr dspn_pnr snc_inhibition snc_stimulation; do go $k baseline $st; done
  go $k pd train
  go $k hd train
  for n in pd_acute pd hd_acute hd; do
    for st in test opto pnr dspn_pnr snc_inhibition snc_stimulation; do go $k $n $st; done
  done
done
echo "$(date +%T) RUN_SEEDS_DONE" >> $P
