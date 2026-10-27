# Init-seed comparison

Weight-init seed (`keyind`) varied; everything else is the baseline config (T_start draw PRNGKey(13), training noise PRNGKey(0), 2000 iterations). Produced by `seeds/run_seed.py`; per-seed plots in `seeds/seed_<k>/plots/`. "13 (retrained)" uses the baseline seed but trained in this batch; it did not reproduce the baseline (see below). The baseline row evaluates the frozen params from `baselines/2026-10-08_12a8ade/`.

## Training and timing (testing_script, 100 noise seeds × 6 cue times)

| seed | best logged loss | best step | final loss | respond | on time (RT 2.5–3.5 s) | RT median | RT IQR |
|---|---|---|---|---|---|---|---|
| **baseline** (13, frozen params) | 0.0068 | 800 | 0.0134 | 0.90 | 0.81 | 2.96 s | 2.77–3.15 s |
| 1 | 0.0085 | 1600 | 0.0127 | 0.44 | 0.42 | 2.87 s | 2.71–3.06 s |
| 2 | 0.0087 | 400 | 0.0132 | 0.41 | 0.34 | 3.29 s | 3.15–3.44 s |
| 3 | 0.0105 | 800 | 0.0139 | 0.02 | 0.02 | 3.09 s | 3.00–3.15 s |
| 4 | 0.0089 | 600 | 0.0148 | 0.14 | 0.14 | 3.02 s | 2.91–3.13 s |
| 5 | 0.0098 | 600 | 0.0134 | 0.00 | 0.00 | – s | – |
| 13 (retrained) | 0.0090 | 800 | 0.0131 | 0.01 | 0.01 | 3.11 s | 2.85–3.15 s |

## Ramping and recruitment

Per area (mean rate over units). *slope/r*: ramp onset (20% of rise, ≥1 s after cue) regressed on RT across trials; slope ≈ 1 means the ramp shifts with movement time. *lead*: median RT − ramp onset. *units*: fraction of units whose peak rate (1 s after cue → RT) exceeds pre-cue by > 0.05.

| seed | D1 slope / r / lead / units | D2 slope / r / lead / units | Cortex slope / r / lead / units | Thalamus slope / r / lead / units | SNc slope / r / lead / units |
|---|---|---|---|---|---|
| **baseline** (13, frozen params) | 0.79 / 0.87 / 0.5 s / 0.80 | 0.69 / 0.65 / 0.2 s / 1.00 | 0.81 / 0.90 / 0.4 s / 0.95 | 0.80 / 0.90 / 0.5 s / 0.85 | -0.10 / -0.10 / 2.0 s / 0.80 |
| 1 | 0.25 / 0.57 / 1.5 s / 0.20 | -0.02 / -0.03 / 1.7 s / 0.20 | -0.00 / -0.01 / 1.6 s / 0.85 | 0.25 / 0.46 / 1.4 s / 0.65 | 0.10 / 0.34 / 1.6 s / 0.40 |
| 2 | 0.39 / 0.59 / 1.8 s / 0.30 | 0.03 / 0.04 / 2.0 s / 0.90 | 0.40 / 0.58 / 1.8 s / 0.95 | 0.36 / 0.54 / 1.9 s / 1.00 | -0.01 / -0.02 / 2.1 s / 0.40 |
| 3 | 0.13 / 0.27 / 1.6 s / 0.80 | – / – / 2.1 s / 0.80 | 0.25 / 0.27 / 2.0 s / 0.90 | 0.02 / 0.04 / 1.5 s / 0.75 | – / – / 2.1 s / 0.60 |
| 4 | 0.43 / 0.59 / 1.1 s / 0.70 | – / – / 2.0 s / 0.80 | 0.28 / 0.29 / 1.1 s / 1.00 | 0.49 / 0.68 / 1.1 s / 0.70 | 0.22 / 0.62 / 1.8 s / 0.60 |
| 5 | – / – / – s / 0.80 | – / – / – s / 1.00 | – / – / – s / 0.90 | – / – / – s / 0.40 | – / – / – s / 0.80 |
| 13 (retrained) | -0.03 / -0.06 / 1.2 s / 0.80 | 1.15 / 0.56 / 1.2 s / 1.00 | 0.01 / 0.02 / 1.2 s / 0.90 | 0.62 / 0.79 / 1.0 s / 0.30 | 0.05 / 0.29 / 2.0 s / 0.60 |

## Opto (1000 trials; RT from cue onset; strongest strength vs. no stim)

| seed | control RT | inh dSPN RT (respond) | inh iSPN RT (respond) | stim dSPN RT (respond) | stim iSPN RT (respond) | inh iSPN faster & inh dSPN slower? |
|---|---|---|---|---|---|---|
| **baseline** (13, frozen params) | 3.07 s (0.90) | 4.56 s (0.87) | 1.88 s (1.00) | 1.34 s (1.00) | 5.07 s (0.64) | yes |
| 1 | 3.10 s (0.44) | 4.36 s (0.54) | 1.32 s (1.00) | 1.29 s (1.00) | 5.19 s (0.46) | yes |
| 2 | 3.44 s (0.39) | – s (0.00) | 1.36 s (1.00) | 1.16 s (1.00) | – s (0.00) | – |
| 3 | 3.43 s (0.01) | 6.37 s (0.00) | 1.16 s (1.00) | 1.10 s (1.00) | – s (0.00) | yes |
| 4 | 3.12 s (0.15) | – s (0.00) | 1.34 s (1.00) | 1.19 s (1.00) | – s (0.00) | – |
| 5 | – s (0.00) | 3.80 s (0.02) | – s (0.00) | – s (0.00) | 4.58 s (0.00) | – |
| 13 (retrained) | 3.09 s (0.01) | 5.20 s (0.01) | 1.41 s (1.00) | 1.37 s (1.00) | – s (0.00) | yes |

## Point of no return (test_pnr_nm, 150 noise seeds)

*ambiguous peaks*: fraction of perturbed trials whose peak output lands in 0.4–0.5 (just under threshold); low = all-or-none. Response probability by stop-pulse delay after cue.

| seed | no-cue respond | ambiguous peaks | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1.0 s | 1.2 s | 1.4 s | 1.6 s | 1.8 s | 2.0 s | 2.2 s | 2.4 s | 2.6 s | 2.8 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** (13, frozen params) | 0.55 | 0.07 | 0.51 | 0.76 | 0.81 | 0.83 | 0.87 | 0.93 | 0.98 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 0.09 | 0.24 | 0.04 | 0.05 | 0.01 | 0.10 | 0.31 | 0.63 | 0.89 | 0.93 | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
| 2 | 0.29 | 0.32 | 0.29 | 0.16 | 0.26 | 0.33 | 0.39 | 0.53 | 0.73 | 0.84 | 0.91 | 0.92 | 0.95 | 0.97 | 0.99 | 0.99 |
| 3 | 0.00 | 0.41 | 0.03 | 0.07 | 0.05 | 0.12 | 0.21 | 0.37 | 0.62 | 0.82 | 0.95 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| 4 | 0.00 | 0.31 | 0.08 | 0.05 | 0.01 | 0.00 | 0.05 | 0.27 | 0.54 | 0.86 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | 0.00 | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.72 | 0.75 |
| 13 (retrained) | 0.00 | 0.11 | 0.50 | 0.61 | 0.71 | 0.81 | 0.90 | 0.96 | 0.95 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
