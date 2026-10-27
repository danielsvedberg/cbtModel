# cbtModel

A JAX model of the cortico–basal ganglia–thalamic (CBT) loop with a dopaminergic (SNc) neuromodulator, trained on a self-timed movement task. The aim is a network whose dynamics and perturbation responses mirror the real circuit, and then to understand the latent mechanisms behind those properties.

## Model (`model_functions.py`, `config_script.py`)

Leaky rate units, `x ← (1 − 1/τ)x + (1/τ)(input)`, with `nln(x) = max(0, tanh(x))`.

| Region | Size | Recurrence | Inputs |
|---|---|---|---|
| Cortex `x_c` | 20 | `J_c` (unsigned) | cue `B_cu`, thalamus `exc(B_ct)` |
| Striatum `x_bg` | 20: first 10 = D1/dSPN, last 10 = D2/iSPN | `inh(J_bg)`, modulated by gain `G` | cortex `exc(B_bgc)`, modulated by gain `G` |
| Thalamus `x_t` | 20 | `J_t` | dSPN excitatory, iSPN inhibitory (`B_tbg` sign-split) |
| SNc `x_nm` | 5 | `J_nm` | cortex excitatory `B_nmc`, BG inhibitory `B_nmbg` |

- **Time constants:** cortex, BG and thalamus use `tau_x`; SNc uses `tau_z`.
- **Output:** `y = exc(C)·nln(x_t) + rb`, read out from thalamus.
- **Dopamine signal:** `s = sigmoid(exc(m)·nln(x_nm) + c)`. It scales BG recurrent and corticostriatal weights by `G = exp(±s)`: up for dSPN rows, down for iSPN rows.
- **SPN excitability traces:** `e_d1` tracks `s` and `e_d2` tracks `1 − s`. They rise fast (10 or 20 steps) and fall very slowly (15000 steps). They multiply SPN population gain by `(1 + e)`.
  - These time constants are **hard-coded defaults in `multiregion_nmrnn`**, not in config or params.
- **Opto / perturbation input:** `stim` is added to `x_bg`.
- **Noise:** process noise of `noise_std/√(2τ)` per step, plus noise on the initial state.

**Task** (`self_timed_movement_task`):
- Time step is 10 ms; a trial is 900 steps (9 s).
- The cue lasts 10 steps.
- The target is a baseline of 0.25, rising to 1.0 from 300 steps (3 s) after the cue until the end of the 300-step movement window.

**Loss:** masked MSE. The target window is placed at the output's own first crossing of 0.5, clamped to be no earlier than the 3 s mark. A trial that never crosses scores about 0.013, which is the "never move" plateau.

## Workflow

Run everything from the repo root:

| Step | Script | Output |
|---|---|---|
| Train | `training_script.py` | `params_nm.pkl` (best params from `fit_nm_rnn`) |
| Test | `testing_script.py` | output and area activity, RT distribution, response-time-binned activity, D1/D2 ratio correlograms |
| Opto | `opto_script.py` | inhibit/stimulate dSPN or iSPN at 1–2.75 s after the cue, then plot activity and RT CDFs |
| Point of no return | `test_pnr_nm.py` (needs the GPU; on CPU the XLA compile runs out of memory) | cue-magnitude sweep, cue-duration sweep, and dynamic stop-signal test (a −5 pulse into all SPNs for 50 ms at 0.2–2.8 s after the cue); output in `plots/pnr_test/` and `plots/png/pnr_*` |

Other scripts:
- `test_pnr.py`, `*_ffin.py` and `params_nm_ffin.pkl` belong to the older ffin model variant.
- `safehouse/` is an older, January 2026 snapshot. It is **not** the current baseline.

## Baseline: commit `12a8ade` (2026-10-08). Return here before making changes.

This config and parameter set produced every property the network is meant to show. These were judged by the user from the testing, opto and PNR plots of this run:

1. **Slow ramping before movement, scaling with movement-time variability.** Activity in dSPN, iSPN, cortex, thalamus and SNc ramps up over about 1–2 s before the response. Ramp onset shifts with response time (`binned_responses_by_area.png`).
2. **Every cell population is recruited.** All five areas show both a cue transient and a pre-movement ramp. SNc steps up after the cue, decays slowly, then ramps with movement.
3. **iSPN inhibition speeds timing.** Responses move from about 3 s to about 1.8 s (`response_times_cdf_opto.png`).
4. **dSPN inhibition slows timing.** Responses move to about 4.5 s. As a bonus, strong dSPN stimulation triggers early responses at about 1.3 s, and iSPN stimulation delays them to about 5 s.
5. **Bistable, point-of-no-return-like responses to perturbation** (`pnr_dyn_outputs.png`, `dynamic_pnr.png`). A brief striatal inhibitory pulse triggers an all-or-none response. Pulses late in the delay produce full, immediate responses. The earliest pulses produce only weak, late responses.

**Config** (`config_script.py` at 12a8ade):
- Task: `dt` 10 ms, `T` 900, `T_cue` 10, `T_wait` 300, `T_movement` 300; `T_start` = 200 random draws in [100, 400) from `PRNGKey(13)`.
- Sizes and gains: `n_bg` 20, `n_nm` 5, `g_bg` = `g_nm` = 0.5.
- Time constants: `tau_x` 15, `tau_z` 40.
- Noise: training `noise_std` 0.05, test noise 0.05.
- Training: `num_full_train_iters` 5000, `keyind` 13 (init seed); optimizer `clip_by_global_norm(1.0)` then `adamw(1e-3)`.
- Model code: `nln = max(0, tanh(x))` (no +0.1 offset); `tau_d1_fall` = `tau_d2_fall` = 15000; `tau_d1_rise` 10, `tau_d2_rise` 20; `gain_e_d1` = `gain_e_d2` = 1.

**Seeds for this run.** All of them are hard-coded; keep them fixed to reproduce:

| What | Seed | Where |
|---|---|---|
| Training trial onsets `T_start` | `jr.PRNGKey(13)` | `config_script.py:62`, separate from `keyind` |
| Weight initialization | `keyind` = 13, i.e. `jr.PRNGKey(13)` | `config_script.py:96` |
| Training noise | `jr.PRNGKey(0)` | `fit_nm_rnn`, `model_functions.py:238` |
| Test trials | `PRNGKey(0..99)` | `test_model` |
| Opto | `PRNGKey(0)`, split over 1000 trials | `simulate_opto` |
| PNR tests | `PRNGKey(s)`, `s + 100`, `s + 200` for s in 0..149 | `test_pnr_nm.py` |

**Params:** `params_nm.pkl` at 12a8ade, sha256 `b88f9339a1974285f2c8abcc3e80c3c8581ea198cec69504e81258cd3f137d6c`.

**Reproducible.** Re-running `training_script.py` at 12a8ade on the GPU (2026-10-08) gave bit-identical params: max |diff| 0.0. The log is in `baselines/2026-10-08_12a8ade/training_log.txt`.
- The loss bounces between the 0.013 never-move plateau and dips.
- The best chunk ends at step 800, with logged loss 0.0068. The saved params are the update right after it.
- From about step 2200 the loss sits flat on the plateau. **So the baseline is effectively an ~800-iteration model**, and the extra iterations add nothing.
- From 2026-10-08, `config_script.py` sets `num_full_train_iters` = 2000.
  - The first 2000 steps of training are deterministic and identical to the 5000-iteration run, and the step-800 checkpoint (0.0068) still beats every later one.
  - So 2000 iterations select the same baseline params.

**Frozen copy:** `baselines/2026-10-08_12a8ade/` holds the scripts, `params_nm.pkl` and the plots from this run. The plots are copied there because `.gitignore` excludes every png/svg under `plots/`. Three very large 900-dpi figures were downsampled.

**To restore:**
```
git checkout baseline-2026-10-08 -- config_script.py model_functions.py params_nm.pkl
```
Or copy the files back from `baselines/2026-10-08_12a8ade/`.

**Fragility. The baseline depends on all of the following:**
- **Time constants:** the user observed it is sensitive to `tau_x`, `tau_z` and `tau_d*_fall`. Across seeds, training outcome is highly variable.
  - In 2×2 tests at `tau_z` 150, whether a run found a good solution flipped with init seed and with small code changes. See `tau_x_10v15_2x2/summary.md`.
  - Treat the init seed (`keyind` 13) and the `T_start` draw as part of the baseline.
- **The quirks of `fit_nm_rnn`, exactly as they are.** Fixing any of them changes which solution training lands on:
  - **Frozen noise:** `nonlocal rng_key` inside a jitted `lax.scan` reuses one noise draw for every step.
  - **Adam reset:** Adam's state is reset every `log_interval` (200).
  - **best_params one step late:** the saved `best_params` are one update after the step whose loss selected them.
- **Code defaults:** the `tau_d*` defaults live in `model_functions.py`, so editing them silently changes the baseline even with the same `params_nm.pkl`.
- **Other seeds do not reproduce it** (`seeds/README.md`, 2026-10-09).
  - Init seeds 1–5 under the same config respond on 0–44% of trials, against 90% for the baseline, and show weaker or no RT-scaled ramps.
  - **Seed 1 comes closest, and the user rates it pretty good:** 44% responding, correct opto directions, and a clean PNR curve with few spontaneous moves. It is a candidate second reference.
  - Retraining seed 13 with ~1e-8 numerical differences (two GPU jobs at once) also failed.
  - The baseline is a lucky trajectory, not a robust outcome of the config.
- **The 3000-iteration batch** (30 seeds, `seeds/README.md` and `seeds/3k/scorecard.md`) gave the same verdict.
  - **Only the baseline scores 5 of 5** target properties. Seeds 9, 17 and 21 score 4.
  - **Seed 9** has RT-tracking ramps in all areas, a clean PNR curve and the right opto signs, but responds on only 21% of trials.
  - **A silent untrained network predicts trainability:** no SPN, cortical or thalamic activity before the cue. 7 of 9 silent inits, including 13, reached ≥ 50% responding, against a median of 21% for self-active inits (p = 0.039).
  - **Silence doesn't give the full set of properties**; that still looks like trajectory luck.
- **Train one job at a time on the GPU.** Run alone, training reproduced bit-identically every time; concurrent jobs did not.
- **Spontaneous moves:** the baseline reaches threshold on about 55% of no-cue trials within 6 s (PNR tests).

## Perturbation and lesion experiments (`experiments/README.md`, 2026-10-10)

`experiments/perturb_model.py` adds SNc-clamp and lesion hooks to a copy of the model (identical output with hooks off); `model_functions.py` is unchanged. On the baseline:
- **At `tau_z` = 40, SNc is a gain, not the timer.** Silencing SNc from the cue cuts responding from 90% to 22% but leaves ramp timing almost unchanged. Saturating it gives 100% responding at 2.49 s.
- **Critical slowing before commitment.** After a +1 dSPN pulse, perturbations decay within 50–160 ms up to 1.2 s after the cue. From 1.4 s on they increasingly fail to decay, and from 1.8 s they grow.
- **SNc and critical slowing** (`experiments/snc_critical_slowing.py`): the instability before movement is in the striatum–cortex–thalamus loop. It is unchanged with SNc frozen or open-loop, and the Jacobian with SNc held fixed crosses |λ| = 1 about 1.2 s before RT. The loop goes unstable on a schedule set by time since the cue, whatever the DA level; DA shifts how soon the network escapes to threshold afterwards (RT). SNc itself is a stable ~2 s integrator. In the HD-retrained network, by contrast, SNc dynamics carry the timing.
- **PD (4 of 5 SNc units lesioned):** acutely 17% responding, rescued to 88% by SNc stimulation. Retraining collapsed it onto the never-move plateau.
- **HD (8 of 10 iSPNs lesioned):** acutely the output saturates before the cue. Retraining recovered 30% on-time responding.
- **Seeds 9, 17, 21** (same tests and lesioned units, 2026-10-11; `experiments/README.md`). Every network shares:
  - SNc stimulation speeds responses;
  - late stop pulses always trigger responses;
  - critical slowing after a dSPN pulse, peaking at 1.2–2.0 s after the cue;
  - a mid-delay dSPN pulse *lowers* responding.

  The baseline is the outlier on three counts:
  - only it keeps responding (22%) with SNc silenced; the seeds drop to 0%;
  - only acute PD in the baseline is rescued by SNc stimulation;
  - only PD in the baseline collapses after retraining; the seeds recover 19–98% responding.
  - **Among nine more seeds (opto score > 0.75, 2026-10-12) none of the three is unique:** seeds 22/24 are also rescued from PD, 23/4/24 also collapse after PD retraining, 22 also keeps responding with SNc off. See report Section 9.
- **PNR and bistability report** (`experiments/report/pnr_bistability_report.md`, 2026-10-11):
  - **Cancellation window:** a response can only be cancelled inside a window that coincides with each network's critical-slowing peak.
  - **PNR:** late in that window it sits about 0.6–1.0 s before each trial's own movement. In the baseline that matches the Jacobian instability about 1.2 s before movement.
  - **Pathway:** commitment is triggered via iSPN → thalamus disinhibition. dSPN inhibition never triggers.
  - **Dopamine:** saturating SNc removes the cancellable window; silencing SNc deepens it.
  - **Cortex off from cue + 100 ms:** no responses, no critical slowing.
  - **What "bistable" means here:** a threshold with transient all-or-none excursions. Coexisting stable states were not tested.
  - **Across 13 networks** (Section 9; `experiments/report/seed_comparison.py`): the iSPN pathway holds in 13/13, cortex-off in 12/13 (seed 23 responds when cortex is silenced), trough–critical-slowing alignment in 10/11. The movement-locked edge holds in only 6/13, and SNc stimulation suppresses responding in 4 seeds.
- **Thalamic E/I and commitment** (`experiments/thalamic_ei/README.md`, 2026-10-13): the hypothesis that commitment is set by E/I balance (or its rate of change) at the readout thalamus was not supported. Balance measures add nothing beyond time since cue and state, and pulse types trigger at different E−I. Structurally, readout-weighted dSPN/iSPN weight onto thalamus predicts the PNR edge (ρ −0.83, n = 11). Connectivity matrices: `experiments/connectivity/`.

## Mechanistic leads (from earlier configs; re-check on the baseline)

Investigated 2026-10-07/08 with `tau_z` = 150, mostly 2000 iterations. **None of this has been verified at the baseline's `tau_z` = 40.**
- **SNc as timer.** In 7 of 8 trained models, the slow SNc dynamics carry the cue-to-movement delay: holding `s` constant at its mean abolishes responding. One model instead held the delay in cortical recurrence.
- **The SPN excitability traces were not the memory.** Freezing them improved timing slightly.
- **Recurrence stretches timescales.** Effective timescales were about 4–10× `tau_x` at initialization. Trained cortex reached about 130–170 steps regardless of `tau_x`.
- **The loss gradient is biased near good solutions.** Autodiff ignores the self-referential argmax target, so training repeatedly finds good solutions and then loses them.

Tooling for this: `tau_x_sweep.py` (`--verify` reproduces `fit_nm_rnn` exactly; `--keep_opt_state`), `tau_z_sweep.py`; results are in `tau_x_sweep*/`, `tau_z_sweep/` and `tau_x_10v15_2x2/`.

## Practical notes

- **Running headless:** `plotting_functions.py` calls `matplotlib.use('Qt5Agg')`, which hangs on `plt.show()`. To run headless, force Agg first:
  ```
  import matplotlib; matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
  ```
- **GPU memory:** about 4.5 GB per training run on the shared GPU. PyCharm consoles can hold most of it, so check `nvidia-smi` first.
- **Comparing settings:** use several init seeds, and score each with fresh-noise tests (`test_model`), not with one run's training loss.
