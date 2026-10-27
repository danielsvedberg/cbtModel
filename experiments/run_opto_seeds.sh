#!/bin/bash
# Full test series on the 3k seeds with opto similarity > 0.75 (seeds/similarity.py), excluding 13/17 (done).
# GPU stages one seed at a time (run_seeds.sh then run_ctx.sh); CPU PNR determinants after each seed's GPU stages.
# From repo root: bash experiments/run_opto_seeds.sh [seed ...]. Progress: experiments/seeds/progress.log, experiments/pnr_determinants/progress.log
SEEDS=${@:-29 22 7 23 1 3 4 24 6}
for k in $SEEDS; do
  bash experiments/run_seeds.sh $k
  bash experiments/run_ctx.sh $k
  bash experiments/run_pnr_determinants.sh $k &
done
wait
echo "$(date +%T) RUN_OPTO_SEEDS_DONE" >> experiments/seeds/progress.log
