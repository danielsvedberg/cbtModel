# seeds/2k_2d fixed-gain sweep (seeds/gain_sweep_2d.py)

DA-driven SPN gains replaced by fixed gains for the whole trial via the model switch params[fixed_gain_ctx/tau]. Control = intact DA network. Each cell: noise-free t80 / control t80 (stall = never reaches 80% of the control rise), scale-only / shift-only warp R² on the 60 cortex/BG/thalamus rates, fraction responding, median paired noisy RT ratio vs intact DA (500 trials).

## diag / both: dSPN = iSPN = g (the nonm sweep); cortex->BG gain and BG tau gain fixed

| seed | verdict | DA t80 (s) | DA resp | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 3.24 | 94% | 0.00 100% RT×0.27 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 2 | no | 3.45 | 91% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 3 | no | 2.98 | 82% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 4 | no ramp | 0.00 | 98% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | stall 0% | 0.65 (-0.63/-0.87) 97% RT×0.64 | 0.00 100% RT×0.53 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 6 | no | 3.08 | 99% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 7 | no | 3.01 | 95% | stall 0% | stall 0% | stall 0% | 0.55 (-135.42/-111.64) 99% RT×0.91 | 0.00 100% RT×0.03 |
| 8 | no | 2.96 | 87% | stall 0% | 0.00 100% RT×0.15 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | stall 0% |
| 9 | no ramp | 0.00 | 100% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 10 | no | 3.16 | 97% | stall 0% | stall 20% RT×0.97 | stall 100% RT×0.04 | stall 97% RT×0.14 | stall 61% RT×0.25 |

## diag / ctx: dSPN = iSPN = g (the nonm sweep); cortex->BG gain fixed, tau gain DA-driven

| seed | verdict | DA t80 (s) | DA resp | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 3.24 | 94% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 2 | no | 3.45 | 91% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 3 | no | 2.98 | 82% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 4 | no ramp | 0.00 | 98% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | 0.62 (-0.69/-0.93) 100% RT×0.61 | 0.64 (0.29/0.19) 100% RT×0.63 | 0.63 (0.55/0.50) 100% RT×0.63 | 0.65 (0.42/0.41) 100% RT×0.65 | 0.69 (-0.09/-0.09) 100% RT×0.70 |
| 6 | no | 3.08 | 99% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 7 | no | 3.01 | 95% | stall 5% RT×1.07 | stall 29% RT×1.07 | stall 0% | stall 0% | 0.00 26% RT×0.62 |
| 8 | no | 2.96 | 87% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 0% | stall 0% | stall 0% |
| 9 | no ramp | 0.00 | 100% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 10 | no | 3.16 | 97% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |

## diag / tau: dSPN = iSPN = g (the nonm sweep); BG tau gain fixed, cortex->BG gain DA-driven

| seed | verdict | DA t80 (s) | DA resp | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 3.24 | 94% | 0.00 100% RT×0.03 | stall 0% | stall 0% | stall 0% | stall 0% |
| 2 | no | 3.45 | 91% | stall 0% | stall 1% | stall 0% | stall 0% | stall 0% |
| 3 | no | 2.98 | 82% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 4 | no ramp | 0.00 | 98% | stall 0% | stall 0% | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | stall 0% | stall 0% | 0.91 (0.95/0.94) 74% RT×0.93 | 0.77 (0.86/0.86) 100% RT×0.74 | 0.00 100% RT×0.03 |
| 6 | no | 3.08 | 99% | 0.22 (-12.57/-9.56) 100% RT×0.19 | stall 0% | stall 0% | stall 0% | stall 0% |
| 7 | partial | 3.01 | 95% | stall 0% | 0.94 (0.57/0.49) 70% RT×0.92 | 1.03 (0.90/0.90) 27% RT×1.04 | 1.43 (0.85/0.47) 2% RT×1.34 | stall 0% |
| 8 | no | 2.96 | 87% | 0.00 3% RT×0.24 | stall 0% | stall 12% RT×0.83 | stall 0% | stall 0% |
| 9 | no ramp | 0.00 | 100% | stall 0% | stall 10% RT×0.89 | stall 15% RT×1.27 | stall 5% RT×1.26 | stall 1% |
| 10 | no | 3.16 | 97% | stall 0% | stall 0% | stall 2% | stall 0% | stall 0% |

## da / both: dSPN = g, iSPN = 1/g (DA direction); cortex->BG gain and BG tau gain fixed

| seed | verdict | DA t80 (s) | DA resp | 0.5/2 | 0.75/1.33333 | 1/1 | 1.5/0.666667 | 2/0.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 3.24 | 94% | stall 0% | 1.06 (0.43/0.45) 75% RT×1.02 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 2 | no | 3.45 | 91% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 3 | no | 2.98 | 82% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 4 | no ramp | 0.00 | 98% | stall 0% | stall 97% RT×1.02 | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | stall 0% | stall 6% RT×1.10 | 0.00 100% RT×0.53 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 6 | no | 3.08 | 99% | stall 0% | stall 83% RT×1.11 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 7 | no | 3.01 | 95% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 8 | no | 2.96 | 87% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 9 | no ramp | 0.00 | 100% | stall 2% RT×1.27 | stall 100% RT×1.00 | stall 0% | stall 0% | stall 0% |
| 10 | no | 3.16 | 97% | stall 0% | stall 0% | stall 100% RT×0.04 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |

## da / ctx: dSPN = g, iSPN = 1/g (DA direction); cortex->BG gain fixed, tau gain DA-driven

| seed | verdict | DA t80 (s) | DA resp | 0.5/2 | 0.75/1.33333 | 1/1 | 1.5/0.666667 | 2/0.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 3.24 | 94% | stall 0% | 0.91 (0.36/0.49) 90% RT×0.96 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 2 | no | 3.45 | 91% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 3 | no | 2.98 | 82% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 4 | no ramp | 0.00 | 98% | stall 1% | stall 98% RT×1.02 | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | stall 0% | stall 2% RT×1.12 | 0.63 (0.55/0.50) 100% RT×0.63 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 6 | no | 3.08 | 99% | stall 0% | 1.09 (0.98/0.96) 98% RT×1.07 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 7 | no | 3.01 | 95% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 8 | no | 2.96 | 87% | stall 0% | stall 0% | 0.00 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 9 | no ramp | 0.00 | 100% | stall 21% RT×1.21 | stall 100% RT×1.00 | stall 0% | stall 0% | stall 0% |
| 10 | no | 3.16 | 97% | stall 0% | stall 0% | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |

## da / tau: dSPN = g, iSPN = 1/g (DA direction); BG tau gain fixed, cortex->BG gain DA-driven

| seed | verdict | DA t80 (s) | DA resp | 0.5/2 | 0.75/1.33333 | 1/1 | 1.5/0.666667 | 2/0.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | partial | 3.24 | 94% | stall 0% | 0.99 (0.95/0.93) 94% RT×1.00 | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 2 | no | 3.45 | 91% | stall 0% | stall 0% | stall 0% | 0.92 (-0.26/-0.04) 99% RT×0.97 | 0.70 (-0.93/-0.38) 100% RT×0.72 |
| 3 | no | 2.98 | 82% | stall 0% | 1.10 (0.94/0.79) 70% RT×1.08 | stall 0% | 0.63 (-42.65/-37.12) 100% RT×0.57 | 0.00 100% RT×0.03 |
| 4 | no ramp | 0.00 | 98% | stall 88% RT×1.22 | stall 97% RT×1.00 | stall 0% | stall 0% | stall 0% |
| 5 | no ramp | 3.24 | 35% | stall 2% | 0.99 (0.99/0.99) 51% RT×0.99 | 0.91 (0.95/0.94) 74% RT×0.93 | 0.78 (0.45/0.38) 100% RT×0.78 | 0.00 100% RT×0.03 |
| 6 | no | 3.08 | 99% | stall 0% | 1.03 (1.00/0.99) 100% RT×1.03 | stall 0% | 0.00 100% RT×0.03 | 0.00 100% RT×0.03 |
| 7 | no | 3.01 | 95% | stall 0% | stall 0% | 1.03 (0.90/0.90) 27% RT×1.04 | stall 0% | 0.00 100% RT×0.05 |
| 8 | no | 2.96 | 87% | stall 0% | stall 0% | stall 12% RT×0.83 | stall 0% | 0.00 100% RT×0.03 |
| 9 | no ramp | 0.00 | 100% | stall 100% RT×1.12 | stall 100% RT×1.00 | stall 15% RT×1.27 | stall 0% | stall 0% |
| 10 | no | 3.16 | 97% | stall 0% | stall 0% | stall 2% | 0.13 (-30.02/-18.82) 100% RT×0.03 | 0.00 100% RT×0.03 |

## Notes (2026-11-01)

- Model switch: `model_functions.py` now takes optional `params['fixed_gain_ctx']` / `params['fixed_gain_tau']` (scalar or per-SPN). Without them the output is bit-identical to `seeds/2k_2d/model_functions_snapshot.py` (max |diff| 0.0 at noise 0 and 0.05, seed 2).
- Unlike nonm, g = 1 is not the trained operating point. The noise-free DA gains are dSPN ≈ 0.76–0.86 and iSPN ≈ 1.22–1.42, flat over the trial in seeds 1, 3–6 and 9. They move after the cue in seeds 2, 7, 8 and 10, e.g. seed 10: dSPN 0.77 → 1.25 and iSPN 1.29 → 1.15 at 0–3 s. That is why the column nearest the intact network is da 0.75/1.33, and why most other gains either stall or fire at cue offset (RT×0.03).
- So this sweep mostly tests distance from the operating point, not temporal scaling around it.
