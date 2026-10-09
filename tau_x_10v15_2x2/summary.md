# tau_x 10 vs 15: offset × tau_x × seed (2026-10-08)

Trained exactly as training_script.py (fit_nm_rnn, 2000 iters, log_interval 200, lr 1e-3, tau_z 150), saving `best_params`.
- head = commit 3f9844b, nln = max(0, tanh(x + 0.1))
- current = working tree on 2026-10-08, nln = max(0, tanh(x))

Checks: head/tau15/seed13 reproduces the committed params_nm.pkl exactly (max |diff| 0.0);
current/tau10/seed13 reproduces the working-tree params_nm.pkl exactly.

Evaluated on the training batch with fit_nm_rnn's frozen noise. "SNc clamp" = s held at its trial mean (m -> 0, c -> logit(mean s)).

| code | tau_x | seed | best step | loss | crossed | on-time | loss with SNc clamped | cue effect on s | SNc needed? |
|---|---|---|---|---|---|---|---|---|---|
| head | 15 | 13 | 800 | 0.0089 | 0.40 | 0.17 | 0.0150 | 0.131 | yes |
| head | 15 | 1 | 400 | 0.0115 | 0.00 | 0.00 | 0.0149 | 0.076 | yes |
| head | 10 | 13 | 1600 | 0.0090 | 0.30 | 0.22 | 0.0091 | 0.000 | **no** |
| head | 10 | 1 | 1000 | 0.0093 | 0.67 | 0.07 | 0.0154 | 0.024 | yes |
| current | 15 | 13 | 1600 | 0.0092 | 0.00 | 0.00 | 0.0159 | 0.011 | yes |
| current | 15 | 1 | 800 | 0.0148 | 0.20 | 0.01 | 0.0169 | 0.333 | yes |
| current | 10 | 13 | 1400 | 0.0077 | 0.67 | 0.32 | 0.0141 | 0.110 | yes |
| current | 10 | 1 | 1400 | 0.0115 | 0.00 | 0.00 | 0.0202 | 0.015 | yes |

Note: the current/tau15/seed1 row's loss (0.0148) is from the analysis's re-evaluation; fit_nm_rnn logged 0.0083 at that chunk boundary.

## Fresh-noise test (as testing_script: 100 noise seeds x test_start_t 270-330)

| code | tau_x | seed | respond | on-time (RT 2.5-3.5 s) | RT median | cue-driven shift in s |
|---|---|---|---|---|---|---|
| head | 15 | 13 | 0.34 | 0.30 | 3.28 s | +0.150 |
| head | 15 | 1 | 0.00 | 0.00 | - | +0.054 |
| head | 10 | 13 | 0.27 | 0.25 | 3.21 s | +0.000 |
| head | 10 | 1 | 0.71 | 0.26 | 3.61 s | +0.187 |
| current | 15 | 13 | 0.00 | 0.00 | - | +0.150 |
| current | 15 | 1 | 0.28 | 0.00 | 4.27 s | +0.273 |
| current | 10 | 13 | 0.74 | 0.65 | 3.09 s | +0.167 |
| current | 10 | 1 | 0.00 | 0.00 | - | +0.027 |

fit_nm_rnn's saved best_params are one update past the step whose loss selected them (logged best vs re-evaluated: identical to the first loss of the next chunk in all 8 runs).
