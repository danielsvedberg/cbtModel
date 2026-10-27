# Perturbation and lesion experiments (2026-10-10)

All tests start from the frozen baseline params (`baselines/2026-10-08_12a8ade/params_nm.pkl`). Full tables are in `summary.md` (`python experiments/summarize.py`).

## How it works

- `perturb_model.py` is a copy of `multiregion_nmrnn` with three hooks: an SNc clamp, an SPN lesion mask and an SNc lesion mask. `model_functions.py` is unchanged.
  - With the hooks off, the copy matches the original exactly (max |diff| 0.0, `verify()`).
  - `install()` swaps `mf.batched_nm_rnn`, so training and the repo's test scripts run unchanged on a lesioned network.
- `run_experiment.py <net> <stage>` runs one test; `run_all.sh` ran everything, one GPU job at a time.
- **Lesion:** the unit's state is held at −10 (rate 0) on every step, so it sends no output.
  - 75% doesn't divide 5 or 10 units evenly, so both lesions are 80%, drawn with `default_rng(2026)` and recorded in `lesion.json`.
  - **PD:** SNc units 0, 1, 3 and 4 lesioned; unit 2 survives.
  - **HD:** 8 of 10 iSPNs lesioned (x_bg 10, 12, 13, 14, 16, 17, 18, 19); 11 and 15 survive.
- **Acute** = baseline params with the lesion applied, no retraining.
- **Retrained** = 3000 more iterations of `fit_nm_rnn` from the baseline params with the lesion in place.

## Folders

| Folder | Contents |
|---|---|
| `baseline/{dspn_pnr,snc_inhibition,snc_stimulation}/` | New tests on the intact baseline. Its test/opto/PNR results are in `seeds/3k/seed_13/` (bit-identical params). |
| `pd_snc_lesion/`, `hd_ispn_lesion/` | `lesion.json`, `training/` (params, loss curve), and `acute/<test>/` and `retrained/<test>/` |

The tests are `test`, `opto`, `pnr` (stop pulse), `dspn_pnr`, `snc_inhibition` and `snc_stimulation`. Each test folder holds `metrics_<test>.json`, a copy of the params it used, and its plots.
- **RT plots for every PNR test:** `pnr/` and `dspn_pnr/` also hold `rt_<test>.png`, with the RT CDF and per-condition RT histograms (the vertical tick on each row marks the pulse). The RTs themselves are in `rts_<test>.npz`.
  - The tests covered are cue magnitude, cue duration, stop pulse and dSPN pulse.
  - The baseline's are in `baseline/pnr/` and `baseline/dspn_pnr/`.
  - The figures are made by `experiments/pnr_rt_plots.py [net ...]`, which re-simulates with the original noise keys and checks its response rates against the stored metrics. `run_experiment.py` now also makes them.

## The new tests

- **dSPN pulse PNR (`dspn_pnr`)**
  - A +1 input to dSPNs only (units 0–9) for 50 ms, at 0.2–2.8 s after the cue, on 150 noise seeds.
  - Each pulse trial has a **twin**: the same noise key and cue, with no pulse.
  - *P(triggered)*: P(the pulse trial crosses threshold within 0.5 s of the pulse), minus the same for its twin.
  - *Relaxation*: d(t) = ‖rates(pulse) − rates(twin)‖ over all 65 units.
    - τ = time from pulse end until d falls below peak/e.
    - The window ends 300 ms before either trial crosses threshold, so the movement itself doesn't count.
    - *censored*: d never falls to 1/e within the window.
    - *residual 500 ms*: d 500 ms after the pulse divided by peak d. Above 1 means the perturbation grew.
- **SNc inhibition / stimulation:** every SNc state is clamped to −10 (rate 0) or +10 (rate about 1) from cue onset to trial end, on 1000 trials. The control is the same trials unclamped.
  - Even with SNc off, the DA signal is not zero: `s` falls to `sigmoid(c)`.

## Results

**Baseline**
- **SNc inhibition:** responding drops from 90% to 22%, and RT is slower (3.07 → 3.42 s). The ramp still peaks at about the same time but at about half the amplitude.
  - At `tau_z` = 40, SNc acts as a gain on the ramp, not as the timer. That differs from the `tau_z` = 150 models.
- **SNc stimulation:** 100% respond, and responses come earlier (2.49 s).
- **dSPN pulse:**
  - A pulse 2.2–2.8 s after the cue triggers an immediate response on 8–23% of trials.
  - A pulse 1.4–1.8 s after the cue *suppresses* responding, down to 49% at 1.6 s.
  - **Critical slowing:** perturbations decay within 50–160 ms up to 1.2 s. From 1.4 s on, more and more never decay (69% at 2.0 s). From 1.8 s they grow (residual 1.1–1.7).
  - So the network loses its restoring force as it approaches commitment.

**PD (SNc lesion)**

| | Acute | Retrained |
|---|---|---|
| Responding | 17%, slow (3.45 s) | 0% |
| Activity | Ramps still track RT in D1, cortex and thalamus | Static state: iSPNs saturated, thalamus silent |
| Opto | iSPN inhibition and dSPN stimulation still trigger responses (100%) | No perturbation produces a response, including SNc stimulation and iSPN inhibition |
| Stop pulse | No longer all-or-none: 53% ambiguous peaks | – |
| SNc inhibition / stimulation | Inhibition changes nothing (the remaining unit is already silent); stimulation brings responding back to 88% at 3.05 s, like L-DOPA rescue | No effect |
| Critical slowing | Still present | None: perturbations decay in 140 ms at every delay |

- **Retraining makes the PD network worse.** The lesioned baseline starts at loss 0.0115. Training falls onto the 0.013 never-move plateau and stays there; only 3% of steps go below 0.010.
- The retrained network has also lowered `sigmoid(c)` to 0.37, so its DA signal is below the baseline's floor.
- This is consistent with the training fragility seen across seeds, rather than evidence that a PD network can't learn the task.

**HD (iSPN lesion)**

| | Acute | Retrained |
|---|---|---|
| Responding | Output passes threshold about 2 s *before* the cue on every trial, and stays there | 30%, on time (RT 2.98 s, very narrow spread) |
| Activity | Hyperkinetic; nothing perturbs it | Ramps weaker; iSPN population mostly gone, as expected |
| Opto | – | iSPN inhibition triggers early responses (1.43 s); dSPN inhibition slows (4.17 s); **dSPN stimulation abolishes responding** (the baseline responds at 1.34 s) |
| SNc inhibition | – | Abolishes responding |
| Stop pulse | – | All-or-none in the late delay (≥ 95% from 1.8 s) |
| Critical slowing | – | Present from 1.6 s; residual peaks at about 1.0 around 2.0 s |
| dSPN pulse | – | Triggers responses only after 2.4 s (10–33%) |

- Retraining recovers part of the behaviour: best loss 0.0082 at step 1000.

## SNc activity and critical slowing (`snc_critical_slowing.py`, 2026-10-10)

`python experiments/snc_critical_slowing.py [baseline|pd_acute|hd]` writes to `<network>/snc_critical_slowing/`: metrics JSON, `trials.csv` (one row per trial), `snc_critical_slowing.png` and `area_decomposition.png`. The script's docstring has the full definitions.

**Method:** the dSPN-pulse test is repeated with an SNc manipulation applied identically to each pulse trial and its twin.

| Condition | What SNc does |
|---|---|
| intact | free |
| open-loop | Follows the intact twin's own trajectory, so `s(t)` is unchanged but the pulse can't travel through SNc. The clamped twin's output stays within 1e-4 of the intact twin's; RTs agree on 100% of trials. |
| frozen | Held at its mean post-cue state from the cue: average gain, no dynamics |
| off | Rate 0 from the cue |
| saturated | Rate ~1 from the cue |

Other analyses:
- Critical slowing aligned to **time to movement** (twin RT − pulse).
- Trial-by-trial regression.
- The pulse's effect split by area.
- **Local stability:** the largest eigenvalue magnitude of the noise-free one-step Jacobian along intact twin trajectories. It is computed for the full system, with SNc held fixed, and for the SNc block alone. The `e_d1`/`e_d2` trace memory is held fixed.
  - The step function reproduces the model to 6e-8.
  - |λ| > 1 means small perturbations grow.

**Baseline**
1. **SNc dynamics are not needed for critical slowing.** Under open-loop or frozen SNc, the residual, censored fraction and τ by pulse time match intact: e.g. residual 1.27 and 1.25 vs 1.30 at 2.0 s, responding 0.93 and 0.92.
   - The PD-acute network shows the same. Its surviving SNc unit is silent, so `s` is constant, yet the slowing and instability are intact.
2. **The instability lives in the striatum–cortex–thalamus loop.**
   - With SNc held fixed, |λ| crosses 1 about 1.2 s before movement and peaks at 1.045 about 0.5 s before. The full system's curve lies on top of it.
   - Before that, the non-SNc loop relaxes with an effective τ of about 400 ms, rising to about 800 ms at 1.6 s before movement.
   - After a late pulse, the difference from the twin grows mainly in D1 and cortex.
3. **SNc is a slow, stable integrator alongside it.**
   - The SNc block alone has |λ| ≈ 0.995, an effective τ of about 2 s, five times `tau_z` because of SNc recurrence. It never goes unstable.
   - Early pulses leave a small, long-lasting trace in `s`, which grows to about 2.5× its value at the pulse over 0.8 s.
   - Early pulses' residual floor is SNc-mediated, through `s` or the `e_d1` trace (which ratchets up with any rise in `s`): 0.15 intact vs 0.08 open-loop. Saturated SNc reaches a similar floor (0.13) some other way. This has no measurable effect on RT (≤ 0.01 s).
4. **DA doesn't change *when* the loop goes unstable; it changes how soon the network escapes afterwards.**
   - By pulse time, the intact, open-loop, frozen and SNc-off curves almost overlap. The residual passes about 0.3 near 1.4 s after the cue, and perturbations grow (residual ≥ 1) from about 1.8–2.0 s.
   - This happens even in SNc-off trials that never move. So the circuit drifts into the unstable region on a schedule set by time since the cue.
   - DA shifts RT (off 3.48 s, intact 3.10 s, saturated 2.47 s). With onset fixed to the cue, that alone explains why, aligned to time-to-movement, the residual first passes 0.3 at about 2.0 s before movement with SNc off, 1.6 s intact and 1.0 s saturated. Fixed onset predicts 2.0, 1.7 and 1.1 s.
   - Low DA: the network sits unstable for longer before crossing threshold, or never crosses (23% respond).
   - High DA: it escapes soon after. With saturated SNc the residual stays below 1 at every time-to-movement, and late-delay pulses find most trials already moved.
   - Caveat: the SNc-off time-to-movement curve uses only the 23% of trials that move.
   - **Trial-by-trial, `s` doesn't behave like the clamp.**
     - In the intact network, higher `s` at the pulse goes with *later* movement (Spearman +0.26).
     - With delay and time-to-movement controlled, at delays of 1.4–2.2 s, higher `s` predicts a slightly smaller residual (standardized β = −0.12, 95% CI −0.16 to −0.08, n = 485 trials). Time-to-movement has the larger effect (β = −0.32).
     - Natural fluctuations in `s` are a weak and ambiguous signal compared with the clamps.
   - **PD acute agrees.** Its `s` is constant at the floor, and only saturation changes it: earlier RT. Unlike the baseline, its saturated condition *does* exceed 1 (1.29 at 0.9–1.2 s before movement).

**HD retrained: a different solution in which SNc matters**
- **SNc dynamics carry the timing.** Freezing SNc at its mean, silencing it or saturating it all abolish responding (0%). Open-loop SNc (the same `s(t)`) keeps 28%.
- **The SNc block is itself marginally stable** (|λ| ≈ 1.00) from about 1.8 to 1.2 s before movement, and the full system is unstable there while the non-SNc loop is not. In this network SNc–circuit coupling creates the slow unstable mode, closer to the `tau_z` = 150 "SNc as timer" models.
- The measured pulse residual is still unchanged under open-loop SNc. The SNc-carried unstable mode is weak (|λ| ≈ 1.005), so it adds little within 500 ms.

**Caveats**
- Residual and τ come only from trials that stay at least 300 ms from movement after the pulse. So late delays, and the late parts of the area curves, are biased toward slow trials. Counts are in the JSON.
- The Jacobian ignores noise and holds the `e` traces' memory fixed.

## Same tests on seeds 9, 17 and 21 (2026-10-11)

**Which seeds:** the three next-best seeds from the 3k batch (4 of 5 on `seeds/3k/scorecard.md`).

**Run with:** `bash experiments/run_seeds.sh <seed>`, which sets `CBT_SEED` for `run_experiment.py`. Each seed takes about 1.5 h on the GPU.

**What ran:**
- Intact network: `pnr`, `dspn_pnr`, `snc_inhibition` and `snc_stimulation`.
- Lesions: PD and HD lesions, each retrained for 3000 iterations, then all six tests on the acute and retrained networks.
- The lesioned units are the same as the baseline's: SNc [0, 1, 3, 4] and iSPN [10, 12, 13, 14, 16, 17, 18, 19].

**Output:** `seeds/seed_<k>/`, with tables in `seeds/seed_<k>/summary.md` (`python experiments/summarize.py <k>`).

| | baseline (13) | seed 9 | seed 17 | seed 21 |
|---|---|---|---|---|
| **Intact:** respond | 0.90 | 0.22 | 0.54 | 0.72 |
| SNc inhibition: respond (RT) | 0.22 (3.42 s) | 0.00 | 0.00 | 0.00 |
| SNc stimulation: respond (RT) | 1.00 (2.49 s) | 0.99 (1.76 s) | 1.00 (1.01 s) | 1.00 (2.05 s) |
| Stop pulse at 0.2 / 0.6 / 1.0 / ≥ 1.6 s | 0.51 / 0.81 / 0.87 / ≥ 0.98 | 0.00 / 0.23 / 0.83 / 1.00 | 0.59 / 0.39 / 0.99 / 1.00 | 0.77 / 0.54 / 0.69 / 1.00 |
| dSPN pulse: P(respond), control → minimum (delay of minimum) | 0.93 → 0.49 (1.6 s) | 0.27 → 0.05 (1.2 s) | 0.64 → 0.25 (1.8 s) | 0.70 → 0.47 (2.0–2.2 s) |
| dSPN pulse: delay where the censored fraction peaks | 2.0 s (0.69) | 1.2 s (0.99) | 1.8 s (0.83) | 2.0 s (0.70) |
| **PD acute:** respond | 0.17 | 0.00 | 0.00 | 0.00 |
| PD acute + SNc stimulation | 0.88 | 0.00 | 0.00 | 0.00 |
| **PD retrained:** respond | 0.00 | 0.69 | 0.19 | 0.98 |
| **HD acute** | saturated | saturated | saturated | saturated |
| **HD retrained:** respond | 0.30 | 0.68 | 0.41 | 0.35 |
| HD retrained: iSPN inhibition respond (RT) | 1.00 (1.43 s) | 0.00 | 1.00 (2.04 s) | 0.97 (1.79 s) |
| HD retrained: SNc inhibition / stimulation | 0.00 / 0.00 | 0.44 / 0.00 | 0.41 / 0.47 | 0.00 / 0.66 |

**What generalizes across all four networks:**
- **SNc stimulation from the cue speeds responses** and drives responding to about 100%.
- **Late stop pulses (≥ 1.6 s) trigger responses in every trial.**
- **Critical slowing after a dSPN pulse.**
  - Up to about 1 s after the cue, +1 dSPN pulses decay within a few hundred ms.
  - Later in the delay, pulses increasingly fail to decay within the 2.5 s window (censored).
  - The censored fraction peaks at 1.8–2.0 s in the baseline and seeds 17 and 21, and earlier, at 1.2 s, in seed 9.
- **A mid-delay dSPN pulse *lowers* the chance of responding, and only late pulses trigger.**
  - This is seen in all four networks. The window where responding drops lines up with the critical-slowing window.
  - The +1 pulse seems to knock the network off its trajectory while the loop is slow, rather than push it over threshold.
  - This is an inference from the tables, not tested directly.
- **The HD lesion saturates the output before the cue every time.**

**Where the baseline is the odd one out:**
- **The baseline is the only network where responding survives SNc silencing (22%).** In seeds 9, 17 and 21, silencing SNc from the cue abolishes responding.
  - So "SNc is a gain, not the timer" holds for the baseline. The seeds depend more on SNc, but this test doesn't separate gain from timer in them.
  - Running `snc_critical_slowing.py` on a seed would settle it.
- **Acute PD:**
  - **SNc stimulation doesn't rescue it in any seed**; the baseline went from 17% to 88%.
  - **It isn't the level of s.** In seeds 9 and 17 the clamp raises mean s after the cue to the intact network's level (0.649 vs 0.633; 0.539 vs 0.540), yet nothing responds.
  - **One untested candidate:** the e traces. They start from s computed with 4 of 5 SNc units dead and fall with τ = 15000, so iSPN gain stays raised all trial; the clamp only starts at the cue.
  - **Responding without stimulation** drops to zero in all three seeds; the baseline kept 17%.
- **PD retrained:**
  - The baseline collapsed to never moving. Every seed recovered: 19–98% responding.
  - So the baseline's collapse is a property of that network or training trajectory, not of the lesion.
- **HD retrained:** all four recover some on-time responding (30–68%). Their SNc dependence differs:
  - **Baseline and seed 21:** SNc inhibition abolishes responding.
  - **Seed 9 and the baseline:** SNc stimulation abolishes responding.
  - **Seed 17:** neither changes responding much.
- **Early stop pulses (0.2–0.6 s): compare with each network's cue-only rate** (the dSPN control column), not with later pulses. From `rts_stop_pulse.npz`:
  - **Baseline:** 0.51–0.81 against 0.93 cue-only, so early pulses suppress responding.
  - **Seed 9:** 0.00–0.23 against 0.27. Suppressed; the few responses come about 1.1 s after the pulse.
  - **Seed 17:** 0.39–0.59 against 0.64. Rate roughly unchanged, but the responses are pushed late (median about 5 s after the cue).
  - **Seed 21:** 0.54–0.77 against 0.70. Rate unchanged, but responses come *earlier* (median 2.3–2.5 s against 3.05 s cue-only).
- **Seed 17's responses are largely not cue-driven.** It responds on 53% of no-cue PNR trials (59% in the dSPN test), close to its 56% intact rate. Read its responding, SNc-stimulation and PD rows with that in mind.

## Cortex off and PNR determinants (2026-10-11)

The full write-up with figures is in `report/pnr_bistability_report.md`; `report/make_figs.py` builds its figures.

**Cortex off:**
- **Stages:** `ctx_inh_test`, `ctx_inh_pnr` and `ctx_inh_dspn_pnr` in `run_experiment.py`. Every cortical unit is held at rate 0 from 100 ms after cue onset to trial end, via `perturb_model` `ctx_off` / `CTX_DELAY`.
- **Run with:** `bash run_ctx.sh` on the baseline and seeds 9, 17 and 21.
- **Result:** no network responds and no pulse triggers; critical slowing disappears.

**PNR determinants** (`pnr_determinants.py <seed> [intact|ctx_off|snc_off|snc_sat]`):
- **Design:** a paired-pulse analysis with 13 pulse types and 14 delays, where every condition shares one noise key per seed. It runs on the CPU (`run_pnr_determinants.sh`).
- **Results:**
  - The cancellable window coincides with each network's critical-slowing peak.
  - Late in that window the PNR sits about 0.6–1.0 s before each trial's own movement.
  - Commitment is triggered via iSPN → thalamus disinhibition.
  - Dopamine sets how easily the switch flips.

## Nine more seeds: opto score > 0.75 (2026-10-12)

The full series (intact PNR/dSPN-PNR/SNc clamps, PD and HD lesions acute and retrained, cortex off, paired-pulse determinants) was run on seeds 29, 22, 7, 23, 1, 3, 4, 24 and 6 (`run_opto_seeds.sh`; selection by `seeds/similarity.py`). Tables and Fig 8: Section 9 of `report/pnr_bistability_report.md`, built by `report/seed_comparison.py`.
- **General (13 networks):** iSPN −5 triggers and dSPN −5 never does in all 13. Cortex off → no responses in 12 (seed 23 responds ~0.2 s after cue when cortex is silenced, with or without a cue). The cancellation trough lies within 0.4 s of the critical-slowing peak in 10 of 11 measurable.
- **Variable:** late stop pulse triggers ≥ 0.94 in 7 of 13 (seed 4: 0.01; seed 23 inverted, triggers early). Movement-locked late edge (0.6–1.0 s) clear in 6; clock-locked in 24, 4, 23. SNc stimulation *suppresses* responding in 22, 1, 3, 7.
- **The baseline's lesion results are no longer unique:** PD acute is rescued by SNc stimulation in 22 and 24 as well; PD retraining collapses to 0 in 23, 4 and 24 as well; responding survives SNc inhibition in 22 (0.44) and 29 (whose SNc is inert: Δs = 0 after the cue). The "baseline is the outlier" points in the seeds 9/17/21 section hold only against those three seeds.

## Caveats

- **The lesion sizes are 80%, not 75%.**
- **Lesions are a single random draw.** In PD only one SNc unit survives (unit 2), so which unit survives may matter a lot.
- **The retraining outcomes are single runs.** Given how chaotic training is, they behave like one seed each.
- **Acute HD RTs of 0.00 s** mean the output was already above threshold at cue onset.
