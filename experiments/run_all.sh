#!/bin/bash
# Runs every lesion stage sequentially on the GPU (one job at a time). From repo root: bash experiments/run_all.sh
# Progress: experiments/progress.log. The baseline's new tests (dspn_pnr, snc_*) are run separately.
P=experiments/progress.log
export XLA_PYTHON_CLIENT_PREALLOCATE=false
go() {  # net stage
  python experiments/run_experiment.py $1 $2 > experiments/log_$1_$2.txt 2>&1
  echo "$(date +%T) net=$1 stage=$2 exit=$?" >> $P
}
echo "--- $(date +%F_%T) run_all ---" >> $P
go pd train
go hd train
for n in pd_acute pd hd_acute hd; do
  for st in test opto pnr dspn_pnr snc_inhibition snc_stimulation; do go $n $st; done
done
echo "$(date +%T) RUN_ALL_DONE" >> $P
