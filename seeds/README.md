# Init-seed test of the baseline config (2026-10-09)

**Question:** do other weight-init seeds, trained with the baseline config (`baseline-2026-10-08`, 2000 iterations), give the same desirable model?

**What varied:** only `keyind`. The `T_start` draw (`PRNGKey(13)`) and the training noise (`PRNGKey(0)`) are the baseline's.

**Files:**
- `summary.md` holds the tables and `seed_comparison.png` the figure.
- Each `seed_<k>/` holds `params_nm.pkl`, its `config_script.py`, `losses.npy`, logs, metrics and plots. Plots are 300-dpi PNG only.
- `baseline_12a8ade/` holds the same tests run on the frozen baseline params.

**Rerunning:**
- One seed: `bash seeds/run_all.sh <seed> ...`, then `python seeds/summarize.py`.
- **Run one job at a time on the GPU** (see Reproducibility).

## Results

| | baseline | seeds 1–5, and 13 retrained |
|---|---|---|
| Responds to cue (fresh noise) | 90% (81% on time) | 0–44% |
| Ramp onset tracks RT (slope ≈ 0.8, r ≈ 0.9 in D1, D2, cortex, thalamus) | yes | weak (slope ≤ 0.4) or absent |
| iSPN inhibition → faster, dSPN inhibition → slower | yes | yes in seeds 1 and 3 and in 13 retrained (seed 3 and 13 retrained only on rare trials); no in 2, 4, 5 |
| All-or-none peaks (ambiguous 0.4–0.5) | 7% | 11–52% |

**None of the other seeds reproduces the baseline.**
- **Seed 1 comes closest, and the user judges it pretty good** from `seed_comparison.png`.
  - It responds on 44% of trials and has the right opto signs, including iSPN stimulation slowing responses.
  - Its dynamic PNR curve is a clean sigmoid with no spontaneous responses: 9% on no-cue trials, against the baseline's 55%.
  - Its ramps track RT more weakly (slope 0.25), and fewer SPNs are recruited.
- **Seeds 3, 5 and 13 retrained almost never respond** (2%, 0%, 1%). They sit on the 0.013 "never move" plateau.
- **Things every seed shares:**
  - Strong iSPN inhibition and dSPN stimulation trigger early responses at about 1.2–1.9 s, with 100% responding.
  - A stop pulse to all SPNs late in the delay (≥ 1.8 s) triggers responses in nearly every trial.

**The baseline is not a robust outcome of this config; it is one lucky training trajectory.**
- Retraining seed 13 in this batch differed from the original run only in the 8th digit of the step-200 loss (0.013350893 vs 0.013350898).
- That tiny difference grew into a different solution that almost never responds. Training is chaotic enough that numerical noise acts like a new seed.

**Caveats:**
- **The baseline moves spontaneously.** In the PNR tests it reaches threshold on 55% of no-cue trials within 6 s; the other seeds do so on 0–29%. This was not visible in the mean-trace plots.
- **The "lead" column measures something narrow.** It is time from 20% of the rise to threshold, which is shorter than the visible 1–2 s ramp.

## Reproducibility

- **Run alone, results match exactly.** Training `seed_13` by itself (including through `run_seed.py`) reproduces the baseline log at every checked step, whether or not GPU memory preallocation is on.
- **This batch did not match.** It ran two lanes at once (seeds 13/1/2 and 3/4/5), and the retrained seed 13 diverged.
- **The cause is unconfirmed.** The likely one is XLA's GPU autotuning, which picks kernels by timing them, so contention can change the picks. A quick two-process test did not trigger it.
- **So:** seeds 1–5 here may not be bit-reproducible. Rerun them one at a time if exact copies are needed.

## 3000-iteration batch (`3k/`, 2026-10-09)

**Setup:** seeds 13 (control), 1 and 6–29 (25 seeds), 3000 iterations each, on 2026-10-09. Seeds 30 and 2–5 were added on 2026-10-10, for 30 in all.
- From batch 2 on, `run_3k.sh` runs PNR in the GPU lane straight after each training run.
- Training ran one seed at a time on the GPU (`run_3k.sh`), with test and opto stages on the CPU.
- PNR was run on the GPU after all training (`run_3k_pnr.sh`). On the CPU, XLA's LLVM compile runs out of memory and the stage silently writes nothing.
- **Files:**
  - `3k/summary.md` and `3k/seed_comparison.png`: full tables (`python seeds/summarize.py 3k`).
  - `3k/scorecard.md`: pass/fail against the five target properties (`python seeds/scorecard.py 3k`).
  - `3k/init_features.csv` and `.png`: features of the untrained network vs. outcome (`JAX_PLATFORMS=cpu python seeds/init_features.py 3k`).

**Reproducibility:** seed 13 trained alone gave params bit-identical to the baseline. Seed 1 also matched its 2k-batch result (best step 1600, 44% responding).

### Only the baseline has all five properties (`3k/scorecard.md`)

Thresholds are set so the baseline passes each:
- respond on ≥ 50% of trials;
- ramp onset tracks RT (slope ≥ 0.5 and r ≥ 0.6) in ≥ 3 of D1, D2, cortex and thalamus;
- ≥ 40% of units recruited in every area;
- iSPN inhibition gives earlier responses on ≥ 90% of trials, and dSPN inhibition gives later ones;
- PNR: stop pulses ≥ 2 s after the cue trigger ≥ 95% responses, with ≤ 25% ambiguous peaks.

| Score (of 5) | Seeds |
|---|---|
| 5 | 13 (baseline) only |
| 4 | 21 (no RT-tracking ramps), 17 (ramps in 2 areas), 9 (all except respond: only 21%) |
| 3 | 27, 10, 6, 29 |
| ≤ 2 | the other 22 |

- **Seed 9 has the baseline's mechanism but not its reliability.** Ramps track RT in all four areas (slope 0.5–0.8), every population is recruited, opto signs are correct, and the PNR curve is a clean sigmoid with few spontaneous moves (5%). It responds on only 21% of trials.
- **Seed 29 ramps like the baseline** in D1, cortex and thalamus (slope about 0.8, r about 0.89), but SNc is not recruited and its PNR is not all-or-none.
- **High response rate alone is a different solution.**
  - Seed 19 responds on 99% of trials with a very tight RT spread, too narrow for ramps to track. iSPN inhibition *prevents* its responses rather than speeding them up.
  - Seed 27 (88%) behaves the same way.
- **Only the baseline scales with cue strength.** Its response probability rises steadily with cue magnitude, to about 0.95 (`3k/seed_comparison.png`, right panel). Every other seed plateaus at 0.8 or below.
- **The same seed can land on very different solutions.**
  - Seeds 2–4 were trained in both batches. The 2k batch ran two training jobs on the GPU at once, which introduces ~1e-8 numerical differences; the 3k batch ran one at a time.
  - Seed 2 went from 41% responding (2k, best step 400) to 0% (3k, best step 2800).
  - Seeds 3 and 4 improved slightly (2% → 17% and 14% → 23%).
  - This is more evidence that tiny numerical differences act like a new seed.
- **More iterations helped:** 10 of 25 seeds had their best checkpoint after step 2000.
- **Every responding seed shares the opto direction.** iSPN inhibition gives earlier responses in 19 of 21 seeds that respond at all.

### The untrained network predicts trainability

`init_features.py` rebuilds each seed's initial weights and simulates them noiselessly before training.
- **Two kinds of starting network.** Some are **silent**: every SPN, cortical and thalamic unit sits at zero before the cue, and the output peaks briefly after the cue. The rest are **self-active**: most units fire with no input, and the output drifts upward through the trial.

| Starting network | Seeds | Trained response rate |
|---|---|---|
| Silent | 9 (13, 27, 12, 10, 7, 23, 29, 3, 30) | 7 of 9 at 50–90%; seed 3 at 17%, seed 30 at 0%; median 59% |
| Self-active | 21 | median 21%; 9 of 21 at ≤ 1% |

- The difference is significant (Mann–Whitney p = 0.039).
- It was p = 0.011 at 25 seeds, before seeds 3 and 30 broke the pattern. So a silent start is a strong tendency, not a guarantee.
- Untrained output level, SNc baseline `s`, and fraction of active SPNs all correlate negatively with trained response rate (Spearman rho −0.5 to −0.7).
- **Weight statistics predict little:** recurrent eigenvalues, input and readout norms all have |rho| ≤ 0.4.
- **Silence is not sufficient.** Silent starts score 0–3 of 5, except the baseline. Three self-active seeds (9, 17, 21) score 4.
  - So a silent start makes a responding solution likely.
  - The baseline's full set of properties, RT-scaled ramps plus iSPN-triggered early responses, still looks like trajectory luck: 1 of 25 seeds.

**Next steps to try:**
- Bias initialization toward a silent start, e.g. a negative bias or a smaller `g_bg`.
- Use seed 9's dynamics as a second reference for the ramp mechanism.
