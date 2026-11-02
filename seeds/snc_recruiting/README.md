# SNc-recruiting seeds with a cortical ramp (2k_2d and 2k_2d_m05)

Symlinks to the original seed folders (seeds/<batch>/seed_k), so plots and metrics stay in one place. Built 2026-11-02.

Criteria, on a noise-free cue trial (cue onset at step 300) with the trained params:

1. **SNc recruited:** SNc moves the DA signals by >= 0.15 from their bias-only value after cue offset, max_t |s(t) - tanh(nm_bias(c))| >= 0.15 (the "active" group of the seeds/2k_2d_m05/summary.md SNc-survival table).
2. **Cortex ramps:** mean cortical rate nln(x_c) rises >= 0.05 above its pre-cue mean (0.5 s before the cue), taking the max between 1 s after cue onset (skips the cue transient) and t80 (output at 80% of its rise), i.e. before movement. Seeds with no output ramp (rise < 0.1, or t80 < 1 s) fail.

18 seeds pass criterion 1; 1 of them fail criterion 2. **17 seeds pass both**: 4 of 10 from 2k_2d, 13 of 30 from 2k_2d_m05.

Alive = SNc units with rate > 0.15 from 0.5 s after the cue. Responding / on time / RT from the test stage (2k_2d seed 10: test stage not finished).

| folder | SNc alive | DA depth | cortex pre-cue | cortex ramp | responding | on time 2.5–3.5 s | median RT (s) | scaling (SNc input) | fine-sweep "yes" (from cue) | grid ridge (t80 ratio) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k_2d_seed_2 | 3/5 | 0.80 | 0.00 | +0.19 | 93% | 77% | 3.24 | partial | diag/tau | same-sign diag 0.73–1.02 |
| 2k_2d_seed_7 | 1/5 | 0.35 | 0.28 | +0.20 | 97% | 97% | 3.06 | partial | diag/tau, da/tau | iSPN only 0.88–1.02 |
| 2k_2d_seed_8 | 2/5 | 0.39 | 0.28 | +0.23 | 86% | 86% | 3.02 | partial | – | same-sign diag 0.95–1.03 |
| 2k_2d_seed_10 | 3/5 | 0.71 | 0.12 | +0.14 | – | – | – | partial | diag/tau, da/tau | iSPN only 0.88–1.09 |
| 2k_2d_m05_seed_1 | 1/5 | 0.19 | 0.28 | +0.09 | 80% | 74% | 3.16 | partial | – | iSPN only 0.99–1.03 |
| 2k_2d_m05_seed_2 | 4/5 | 0.56 | 0.15 | +0.15 | 99% | 98% | 2.96 | no | – | DA direction 0.97–1.46 |
| 2k_2d_m05_seed_4 | 3/5 | 0.21 | 0.06 | +0.28 | 88% | 83% | 3.07 | no | diag/tau | same-sign diag 0.92–1.08 |
| 2k_2d_m05_seed_7 | 3/5 | 0.58 | 0.31 | +0.39 | 100% | 100% | 3.03 | partial | – | same-sign diag 0.87–1.06 |
| 2k_2d_m05_seed_10 | 4/5 | 0.41 | 0.11 | +0.12 | 100% | 86% | 3.14 | partial | – | same 1:2 0.95–1.09 |
| 2k_2d_m05_seed_11 | 5/5 | 0.63 | 0.31 | +0.19 | 100% | 100% | 2.92 | partial | da/tau | iSPN only 0.77–1.05 |
| 2k_2d_m05_seed_13 | 3/5 | 0.28 | 0.00 | +0.25 | 64% | 60% | 3.05 | no | – | same 1:2 0.98–1.02 |
| 2k_2d_m05_seed_14 | 3/5 | 0.27 | 0.31 | +0.11 | 99% | 98% | 3.22 | partial | diag/tau | same-sign diag 0.92–1.03 |
| 2k_2d_m05_seed_16 | 2/5 | 0.35 | 0.08 | +0.17 | 69% | 60% | 3.05 | yes | – | iSPN only 0.97–1.04 |
| 2k_2d_m05_seed_23 | 1/5 | 0.29 | 0.42 | +0.08 | 100% | 68% | 3.41 | partial | – | iSPN only 0.70–1.07 |
| 2k_2d_m05_seed_24 | 4/5 | 0.45 | 0.10 | +0.24 | 84% | 75% | 3.17 | partial | diag/tau | same-sign diag 0.96–1.04 |
| 2k_2d_m05_seed_26 | 3/5 | 0.55 | 0.00 | +0.42 | 95% | 83% | 3.09 | partial | – | DA 1:2 0.70–1.04 |
| 2k_2d_m05_seed_27 | 2/5 | 0.20 | 0.04 | +0.27 | 98% | 85% | 3.31 | no | – | DA direction 1.00–1.11 |
