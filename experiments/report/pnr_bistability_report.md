# Bistability, dynamics and the point of no return in the CBT model

*2026-10-11. Networks: the baseline (`baseline-2026-10-08`, init seed 13) and the three next-best 3k-batch seeds (9, 17, 21; 4 of 5 on `seeds/3k/scorecard.md`). All intact; lesioned networks are summarised in `experiments/README.md`.*

*2026-10-12: Section 9 adds nine more seeds (every 3k seed with opto score > 0.75 on `seeds/similarity.py`) and checks which findings hold across all 13 networks. Sections 1–8 are about the original four.*

This report pulls together every perturbation test run on these four networks, with an emphasis on the question: **is the commitment to move a bistable, dynamical event, and what determines the point of no return (PNR)?**

---

## 1. Summary

1. **Commitment is a threshold event in a dynamical system, not a readout of a fixed timer.**
   - A brief (50 ms) striatal pulse can trigger, advance or cancel a movement, and its effect depends on where the trial is along its own pre-movement ramp.
   - Before commitment, the network loses its restoring force (critical slowing). The baseline's linearised loop becomes unstable (|λ| > 1) about 1.2 s before movement.
2. **The point of no return has two parts: a window in time since the cue, and the trial's own progress inside it.**
   - **The window.** A response can only be cancelled while the loop is near-unstable.
     - Each network's cancellation trough lines up with its own critical-slowing peak: baseline 1.6 s vs 2.0 s, seed 9 1.2 vs 1.2 s, seed 17 1.8 vs 1.8 s, seed 21 2.2 vs 2.0 s after the cue (Figs 4, 5).
     - Pulses well before the window cancel nothing even in trials 2–3 s from moving.
     - The pulse delay alone explains most of whether a response is cancelled (CV AUC 0.79–0.86).
   - **The edge.** Late in the window (pulses at 2.0–2.4 s), a +1 dSPN pulse cancels only trials still more than a fixed time from their own movement, and that time is the same at each of those delays:
     - baseline 1.0 s, seed 9 1.0 s, seed 17 0.8 s, seed 21 0.6 s (seed 21 at 2.2–2.4 s; at 2.0 s its first populated bin is already below 0.9, so no clean edge);
     - in the baseline a stop pulse likewise stops changing RT once the trial is within 1.0 s of moving (same edge at pulse delays 1.8, 2.0 and 2.2 s).
     - So late in the window, the PNR sits at a fixed distance before each trial's own movement.
   - **Caveat on the edge.** Earlier in the window the edge is larger (e.g. seed 17: 2.4, 1.8 and 1.6 s at pulse delays 0.8, 1.0 and 1.2 s), and delay + edge stays at about 3 s. Because cue-only RTs are tightly clustered (baseline IQR 2.9–3.3 s), time-to-movement and time since the cue are hard to separate there. Within a fixed delay, the trial's own time to movement predicts cancellation only modestly (AUC 0.64–0.78).
3. **The ~1 s PNR matches the loop's instability horizon.** In the baseline the linearised loop crosses |λ| = 1 about 1.2 s before movement. After that the trajectory is being carried away from the stable region, and a pulse can speed it but not reverse it.
4. **What determines whether a pulse pushes the network over:**
   - **(a) How far along the ramp the trial is.** At a fixed time since the cue, the pre-pulse cortical, thalamic and D1 rates and the ramp "progress" predict triggering almost perfectly in the baseline (AUC 0.99, delays 2.4–2.8 s) and seed 9 (0.93, 1.8–2.2 s), and moderately in seeds 17/21 (progress AUC 0.80/0.73). The output y predicts equally well, but that is close to circular: it is the distance to threshold.
   - **(b) The barrier shrinks with time.** In seeds 17 and 21 the smallest stop pulse that triggers on ≥ 50% of trials falls about 40-fold, from |a| = 10 at 0.6 s after the cue (seed 21 until 1.0 s) to 0.25 at 2.6 s (Fig 3).
   - **(c) Pathway.** Commitment is triggered by **removing iSPN inhibition of thalamus**: an iSPN-only pulse triggers in every network, a dSPN-only inhibitory pulse never does. In the baseline, stop pulses stronger than |a| ≈ 1 trigger *less*, because they also silence dSPNs.
   - **(d) Dopamine sets how hard the switch is to flip.**
     - With SNc saturated from the cue, every pulse triggers fast and nothing cancels.
     - With SNc off, the baseline responds slowly and its cancellation window deepens (P(survive) 0.15 vs 0.56). The seeds stop responding at all; only iSPN inhibition can still trigger them.
   - **(e) Cortex is required after the cue.**
     - With cortex silenced from 100 ms after cue onset, no network ever responds, and no striatal pulse of any type or timing can trigger a response.
     - Critical slowing disappears.
     - Striatum and SNc fall silent because cortex is their excitatory drive, so this shows cortex is necessary, not by itself that it is part of the unstable loop. The baseline Jacobian, where the growing mode is carried by D1 and cortex, supports the stronger reading.
5. **Bistability, precisely.** The tests show a threshold (separatrix) with all-or-none excursions. The "move" state is transient: pulse-triggered output peaks and returns to baseline within 1–1.5 s (Fig 7, top). Two coexisting stable states were not tested.
6. **Two regimes among the four networks.**
   - **Baseline and seed 9, "progressive":** a stop pulse advances the trial along its ramp, and the response follows 1–1.8 s (baseline) or 0.2–1.1 s (seed 9) later, the latency shrinking steadily with time since the cue.
   - **Seeds 17 and 21, "switch":** from about 0.8 s (17) or 1.2 s (21) after the cue (the first delays where ≥ 50% of stop pulses trigger within 500 ms), a stop pulse flips the network to a response within 0.1–0.3 s, irrespective of when the trial would otherwise have moved. Earlier pulses instead *delay* responding.
   - The baseline's responses are also the most stereotyped. Its cue-only peak output is a tight cluster at about 0.62, well above threshold. In the seeds, natural responses barely cross 0.5 (seed 9: 76% of cue-only peaks fall in the ambiguous 0.4–0.5 band).
7. **Across 13 networks (Section 9).** The iSPN pathway holds in all 13; the cortex requirement in 12 (seed 23 responds when cortex is silenced); cancellation aligned with critical slowing in 10 of 11 measurable. The movement-locked edge at 0.6–1.0 s is clear in only 6 of 13 (clock-locked in 3), and SNc stimulation suppresses rather than raises responding in 4.

---

## 2. Tests and methods

| Test | What it does | Where |
|---|---|---|
| Stop-pulse PNR (`test_pnr_nm.py`) | −5 into all SPNs for 50 ms at 0.2–2.8 s after the cue; cue-magnitude and cue-duration sweeps. Each condition has its own noise. 150 seeds | GPU; `seeds/3k/seed_<k>/`, `experiments/.../pnr/` |
| dSPN-pulse PNR (`run_experiment.py dspn_pnr`) | +1 into dSPNs for 50 ms; every pulse trial has a noise-identical "twin" without the pulse. Relaxation time, residual at 500 ms, fraction that never decays (censored) | GPU; `.../dspn_pnr/` |
| SNc and critical slowing (`snc_critical_slowing.py`) | dSPN-pulse test under SNc free / open-loop / frozen / off / saturated; Jacobian of the one-step map along trajectories | baseline only; `experiments/baseline/snc_critical_slowing/` |
| SNc inhibition / stimulation | SNc clamped off or saturated from cue onset to trial end | GPU; `.../snc_inhibition`, `.../snc_stimulation` |
| **Cortex off (new)** | every cortical unit held at rate 0 from 100 ms after cue onset (cue offset) to trial end; then the regular test, both PNR tests | GPU; `.../ctx_inh_test`, `ctx_inh_pnr`, `ctx_inh_dspn_pnr` |
| **PNR determinants (new, `pnr_determinants.py`)** | paired design: 200 noise seeds × 183 conditions (cue-only twin + 13 pulse types × 14 delays), all sharing one noise key per seed. Pulse types: stop (all SPNs) at −0.1 … −10, dSPN −5, iSPN −1/−5, dSPN +0.25/+0.5/+1. Run intact, with SNc off or saturated from the cue, and with cortex off from cue + 100 ms | CPU; `experiments/pnr_determinants/seed<k>_<cond>/` |

Definitions used in the paired analysis (`pnr_determinants.py` docstring has the full text):
- **twin**: the same trial (same noise) with the cue and no pulse. Before the pulse the two are identical.
- **time to movement (ttr)**: twin RT − pulse onset. A future quantity, used to *locate* the PNR, not as a mechanism.
- **triggered**: crosses 0.5 within 500 ms of the pulse (only trials whose twin has not moved by then).
- **survives**: the pulse trial still responds at all (only trials whose twin moves after the pulse ends).
- **state features** at the step before the pulse: output y; mean rates of D1, D2, cortex, thalamus, SNc; D1 − D2; DA signal s; e_d1/e_d2 traces; ramp progress (projection of all 65 rates on the axis from cue end to just before movement, 0 → 1) and its velocity.
- **Clock-fixed AUC**: AUC computed within each pulse delay, then averaged. This removes the trivial fact that almost everything ramps with time.

The paired analysis was checked against the stored GPU stop-pulse curve. For the baseline, P(respond) by delay is 0.49 / 0.79 / 0.82 / 0.82 / 0.86 / 0.89 / 0.95 … against 0.51 / 0.76 / 0.81 / 0.83 / 0.87 / 0.93 / 0.98 … from `test_pnr_nm.py`, within sampling error (different keys and platform).

To regenerate: `bash experiments/run_ctx.sh` (GPU), `bash experiments/run_pnr_determinants.sh` (CPU, ~30 min), `python experiments/report/make_figs.py`.

---

## 3. Evidence for bistability and all-or-none commitment

![Fig 1](figs/fig1_stop_pulse.png)

**Fig 1, top row: the stop-pulse PNR curve** (GPU `test_pnr_nm.py`).
- Every network goes from low response probability for early pulses to 100% for pulses ≥ 1.6 s after the cue.
- Early pulses suppress responding relative to the cue-only rate (dotted line) in the baseline (0.51 vs 0.91) and seed 9 (0.00 vs 0.23).
- In seeds 17 and 21 early pulses barely change the rate, but they *delay* responses (middle row).

**Middle row: latency from pulse to threshold.** This separates the two regimes:
- **Baseline:** latency falls steadily from 1.76 s (pulse at 0.2 s) to 0.97 s (2.0 s) and 0.30 s (2.8 s). The pulse does not flip a switch; it pushes the trial forward along a slow trajectory, and the trajectory still has to be travelled.
- **Seed 9:** the same shape, faster (1.1 → 0.18 s).
- **Seeds 17 and 21:** from 0.6 s / 1.0 s after the cue on, a stop pulse produces a response 0.1–0.3 s later, always: a genuine flip between two states. Before that time, pulses delay responding by 1–5 s.
- **Dopamine** (blue = SNc off, red = saturated) shifts the whole latency curve: baseline latency at 1.0 s is 1.96 s with SNc off, 1.32 s intact and 0.99 s saturated.

**Bottom row: peak output after the pulse** (to trial end), against the cue-only twins.
- **Baseline:** both distributions are a single tight peak at about 0.6. Every response is a full, stereotyped event, and only 7–9% of trials fall in the ambiguous 0.4–0.5 band.
- **Seeds:** the cue-only responses pile up *at* threshold (ambiguous fraction 0.76 / 0.48 / 0.29 for seeds 9 / 17 / 21). Their spontaneous "responses" are marginal threshold crossings.
- **Pulse-triggered responses in the seeds are bimodal:** one cluster at threshold and one clearly above (0.6–0.9), which ordinary trials rarely reach. In seeds 17 and 21 the response follows the pulse within 0.1–0.3 s, so part of the high cluster may be the pulse's own direct thalamic-disinhibition transient rather than a separate network state.
- **The excursion is transient**, not a persistent second state: triggered output peaks and returns to its baseline within 1–1.5 s. What the tests demonstrate is a threshold with all-or-none excursions; coexisting stable states were not tested.

Earlier baseline-only results that also point to bistability (`experiments/README.md`):
- the original `pnr_dyn_outputs.png` (below), where late stop pulses give full responses and early ones weak late responses;
- the RT histograms of the stop-pulse test, where early pulses produce responses clustered about 1–2 s after the pulse rather than a smear.

![baseline stop pulse outputs](figs/baseline_pnr_dyn_outputs.png)
![baseline stop pulse RTs](figs/baseline_rt_stop_pulse.png)

---

## 4. Critical slowing: the loop loses stability before commitment

![Fig 5](figs/fig5_critical_slowing.png)

**Fig 5** (GPU dSPN-pulse test; the pulse trial is compared with its noise-identical twin).
- Early in the delay a +1 dSPN pulse decays within a few hundred ms (baseline 50–160 ms).
- Later, an increasing fraction of perturbations **never decays** (censored).
  - It peaks at 2.0 s after the cue in the baseline (0.69) and seed 21 (0.70), 1.8 s in seed 17 (0.83), and 1.2 s in seed 9 (0.99).
  - The residual at 500 ms rises above 1 in the baseline: perturbations grow.
- This is the classic signature of approaching a bifurcation or separatrix.

The **response probability** curve (left) dips in this window in all four networks. An excitatory pulse mid-delay *reduces* responding, and only late pulses trigger. The deepest point of the dip (from the paired analysis) is close to each network's own censored-fraction peak:

| | baseline | seed 9 | seed 17 | seed 21 |
|---|---|---|---|---|
| dSPN +1 cancellation trough (paired, CPU) | 1.6 s (P(survive) 0.56) | 1.2 s (0.21) | 1.8 s (0.32) | 2.2 s (0.56) |
| critical-slowing peak (censored fraction, GPU) | 2.0 s (0.69) | 1.2 s (0.99) | 1.8 s (0.83) | 2.0 s (0.70) |

![SNc and critical slowing, baseline](figs/baseline_snc_critical_slowing.png)

**Baseline, SNc dissection** (`snc_critical_slowing.py`; full write-up in `experiments/README.md`):
- **The slowing is unchanged with SNc frozen or open-loop.** The instability lives in the striatum–cortex–thalamus loop.
- **Jacobian:** with SNc held fixed, the largest eigenvalue magnitude crosses 1 about 1.2 s before movement and peaks at 1.045 about 0.5 s before. Before that, the non-SNc loop relaxes with an effective τ of about 400 ms, rising to about 800 ms.
- **SNc itself is a stable ~2 s integrator.**
- **The loop goes unstable on a schedule set by time since the cue.** Its onset is the same with SNc free, frozen or off. DA changes how soon the network escapes to threshold afterwards.

The ~1.2 s instability horizon from the Jacobian lines up with the ~1.0 s PNR found below.

---

## 5. Where is the point of no return?

![Fig 2](figs/fig2_pnr_vs_time_to_movement.png)

**Fig 2** plots each pulse's effect against the trial's own time to movement (twin RT − pulse onset), with one line per pulse delay. If the lines collapse, the outcome depends on where the trial is relative to its own movement and not on the clock.

**Row 1 (stop −5):**
- **Baseline:**
  - Trials within about **1.0 s** of their own movement are unaffected by the pulse (median RT shift 0).
  - Trials further away are advanced, by an amount that grows with distance (shift −0.12 s at ttr 1.1 s, −0.64 s at 1.9 s, −1.26 s at 2.9 s).
  - The edge is at 1.0 s for pulses at 1.8, 2.0 and 2.2 s after the cue. So **the last ~1 s before movement is committed**: nothing a stop pulse does changes when the trial moves.
  - Note that RT shift and time-to-movement both contain the twin's RT, so part of the lines' common slope is built in. The informative part is where the edge sits.
- **Seeds 17 and 21:** the shift lies almost on the RT = pulse-time diagonal down to ttr ≈ 0.2 s. The pulse triggers an immediate response at almost any point after the switch becomes available. Early pulses with long ttr (top right of the panels) are delayed instead.
- **Seed 9:** intermediate; RT unaffected only within about 0.4 s.

**Row 2 (iSPN −5):** an iSPN-only pulse advances RT in every network right up to 0.1–0.2 s before movement. Disinhibiting thalamus is the most direct way to the threshold.

**Row 3 (dSPN +1): cancellation.**
- Pooled over delays 1.0–2.4 s, P(survive) stays ≥ 0.9 below a time-to-movement of **1.0 s (baseline), 1.0 s (seed 9), 0.8 s (seed 17), 0.6 s (seed 21)**, then drops (to 0.59, 0.14, 0.28 and 0.52 at its lowest).
- **Per pulse delay, the edge (the time-to-movement bin where P(survive) first falls below 0.9) is:**

| pulse delay | 0.8 s | 1.0 s | 1.2 s | 1.4 s | 1.6 s | 1.8 s | 2.0 s | 2.2 s | 2.4 s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | – | – | – | – | – | – | 1.0 | 1.0 | 1.0 |
| seed 9 | – | – | – | – | – | 1.0 | 1.0 | – | – |
| seed 17 | 2.4 | 1.8 | 1.6 | – | – | – | 0.8 | 0.8 | 0.8 |
| seed 21 | – | – | – | 1.6 | 1.4 | – | – | 0.6 | 0.6 |

  "–" means no clean edge: either the bins never fall below 0.9, or they are below 0.9 from the start, or there are fewer than 8 trials per bin. Edges are on a 0.2 s grid. (Regenerated 2026-10-12 by `seed_comparison.py`, which applies this rule strictly; the first version listed baseline 1.8 at 1.2 s and seed 21 0.6 at 2.0 s, where the first populated bin was already below 0.9.)
- **Late in the window (2.0–2.4 s; seed 21 2.2–2.4 s) the edge is constant across delays.** That is the signature of a PNR defined relative to each trial's own movement.
- **Earlier the edge is larger, and delay + edge ≈ 3 s**, about the median RT. There, cancellability is limited by the clock window rather than by the trial's progress. With cue-only RTs this tightly clustered, the two can't be fully separated.
- **Cancellability needs both conditions:**
  1. the pulse arrives inside the network's critical-slowing window, where the loop is near-unstable and a perturbation can redirect it. The window is network-specific: from about 0.8 s after the cue in seed 9 (P(survive) 0.57 at 0.8 s, 0.34 at 1.0 s), and from about 1.2–1.4 s in the others. It ends around 2.0–2.4 s;
  2. the trial is still more than ~0.6–1.0 s from its own movement, i.e. not yet past the PNR.

**Dopamine and cancellation** (Fig 4, bottom row): cancellation by the dSPN +1 pulse is
- deeper and wider with SNc off (baseline minimum P(survive) 0.15 vs 0.56 intact);
- abolished with SNc saturated (0.93–1.00 in every network).

High DA makes the network commit early and irreversibly; low DA keeps it in the cancellable region longer.

---

## 6. What determines the PNR?

### 6a. State versus clock

![Fig 6](figs/fig6_determinants.png)

**Fig 6, top: triggering by a stop −5 pulse, with the clock held fixed.**
- **Baseline** (only pulse delays 2.4–2.8 s have ≥ 10 triggered and ≥ 10 non-triggered trials): cortex, thalamus, D1 and ramp progress each predict triggering with AUC 0.99. D2 gives 0.97, velocity 0.94, SNc 0.86, s 0.81. The output y also gives 0.99, but that is nearly circular, since it measures distance to threshold.
- **Seed 9** (delays 1.8–2.2 s): these give 0.92–0.93.
- **What doesn't predict:** the slow traces e_d1 / e_d2 (≈ 0.5–0.56 in the baseline).
- **Seeds 17 / 21** (delays 0.6–0.8 s / 0.6–1.4 s, i.e. while the switch is becoming available): only ramp progress (0.80 / 0.73) and cortex (0.76 / 0.64) carry much. Later their triggering is near 100%, so there is nothing to predict.

So at any given time since the cue, the trials that a pulse pushes over are the ones already furthest along the shared pre-movement trajectory. The logistic models agree: delay alone (as a category) gives CV AUC 0.93 in the baseline, delay + state 0.999.

**Fig 6, bottom: cancellation by a dSPN +1 pulse.**
- The clock dominates: delay alone (categorical) gives CV AUC 0.79–0.86, and the state adds at most 0.03.
- The best single within-delay predictor is the trial's own time to movement (AUC 0.64–0.78), not any instantaneous rate. At a given delay, trials that would have moved later are more cancellable.
- Instantaneous rates are weak predictors (≤ 0.68). The cancellable window is defined by the clock (critical-slowing window) and then by how close the trial is to its own movement.

### 6b. The barrier shrinks through the delay; the iSPN pathway carries commitment

![Fig 3](figs/fig3_pulse_type_amplitude.png)

**Fig 3, top: triggering (within 500 ms) by pulse type.**
- **iSPN −5:** the most effective trigger in every network, from 2.0 s in the baseline (0.89 at 2.8 s), 0.8 s in seed 9, and from the earliest delay in seeds 17 and 21.
- **iSPN −1 and stop −5:** trigger later.
- **dSPN +1:** triggers only late (from 1.8 s in seed 9, 2.0–2.2 s in the others).
- **dSPN −5:** never triggers (0.00 at every delay in every network).

Commitment is reached by releasing thalamus from iSPN inhibition. Inhibiting dSPNs works against it.

**Bottom: amplitude dependence of the stop pulse.**
- **Seeds 17 and 21:** a clean picture of a shrinking barrier.
  - The smallest amplitude that triggers on ≥ 50% of trials falls from |a| = 10 to 0.25. Seed 21: 10 (0.6–1.0 s), 5 (1.2–1.4 s), 2.5 (1.6–2.0 s), 1.0 (2.2 s), 0.5 (2.4 s), 0.25 (2.6–2.8 s). Seed 17 is about 0.2–0.4 s earlier: 10 (0.2–0.6 s), 5 (0.8–1.2 s), 2.5 (1.4–2.0 s), then the same as seed 21.
  - The amplitude needed falls about 40-fold between 0.6 and 2.6 s after the cue, on a coarse 7-point amplitude grid: the distance to the separatrix shrinks steadily through the delay.
- **Seed 9:** similar from 2.0 s on (2.5 → 0.5), with a fall-off at |a| = 10.
- **Baseline:** strongly **non-monotonic**. At 2.8 s, triggering rises from 0.16 (|a| = 0.1) to 0.44 (|a| = 1), then falls to 0.24 (|a| = 10).
  - A big all-SPN pulse removes dSPN drive as well as iSPN inhibition, and in the baseline the two effects nearly cancel.
  - The baseline needs time after the pulse (median latency ~1 s) and rarely crosses within 500 ms.

### 6c. Dopamine

![Fig 4](figs/fig4_dopamine.png)

**Top row: P(respond) after a stop −5 pulse (solid) or an iSPN −1 pulse (dotted)**, with SNc free (black), off (blue) or saturated (red) from the cue.
- **Saturated SNc:** cue-only responding is 99–100%, and mid-delay pulses can no longer suppress it. The one exception is the baseline: a stop pulse in the first 0.4 s still prevents most responses (P(respond) 0.13 at 0.2 s, 0.70 at 0.4 s).
- **SNc off:**
  - The baseline still responds on 24% of cue-only trials, and pulses lift that to about 0.5–0.75 late in the delay.
  - **The seeds never respond** with SNc off. Only an iSPN −5 pulse can still trigger them (seed 9 0.98, 17 1.00, 21 0.63, within 500 ms) — dopamine supplies the drive that brings the seeds into the switch's range, and without it only direct thalamic disinhibition works.
- **Mid-delay iSPN −1 pulses suppress responding** in every intact network (dotted dips: baseline 0.76, seed 9 0.05, seed 17 0.03, seed 21 0.57). The same pulse triggers late in the delay. So the sign of a perturbation's effect is set by *when* it arrives.

### 6d. Cortex

![Fig 7](figs/fig7_cortex_off.png)

**What the test does:** every cortical unit is held at rate 0 from 100 ms after cue onset (when the cue ends) to the end of the trial. The cue gets into cortex first; with silencing from cue onset itself the cue never reaches the network. Run on the GPU with the regular test (600 trials), the stop-pulse PNR test, and the dSPN-pulse PNR test, and on the CPU in the paired-pulse analysis.

**Result: no network ever responds, in any test.**

| network | regular test: respond | stop-pulse PNR: max P(respond) | cue magnitude 0–2: max P(respond) | dSPN-pulse PNR: max P(respond) | dSPN-pulse relaxation τ / censored |
|---|---|---|---|---|---|
| baseline | 0.00 | 0.007 | 0.013 | 0.00 | 50–70 ms / 0 at every delay |
| seed 9 | 0.00 | 0.00 | 0.00 | 0.00 | 80–180 ms / 0 |
| seed 17 | 0.00 | 0.00 | 0.00 | 0.00 | 220–400 ms / 0 |
| seed 21 | 0.00 | 0.00 | 0.00 | 0.00 | 120–200 ms / 0 |

- **Activity** (regular-operation plots below): once cortex is silenced, D1, D2 and SNc fall to zero within about 0.5 s. Cortex is the only excitatory input to striatum, and SNc's excitation also comes from cortex. Thalamus stays at a low tonic level, and the output falls to or below its pre-cue level (Fig 7, top).
- **No pulse of any of the 13 types triggers a response** at any delay in the paired analysis (Fig 7, bottom). That includes iSPN −5, which triggers in every intact network.
- **Critical slowing disappears entirely.** Every dSPN +1 pulse decays, with a short, flat relaxation time across the whole delay.

**What this does and doesn't show:**
- **It does show** that commitment needs ongoing cortical activity after the cue: no cortex, no ramp, no instability, no response. The cue transient alone (100 ms into cortex) never carries the network over threshold.
- **It doesn't separate "cortex is part of the bistable loop" from "cortex is just the excitatory drive that keeps SPNs active".**
  - With SPNs silent, inhibitory pulses into them change nothing: their rates are already 0, and `nln` clips at 0.
  - The excitatory dSPN +1 pulse does reach them but decays without effect.
- The Jacobian analysis of the intact baseline (Section 4) is the stronger evidence that the unstable mode involves cortex. There, the growing perturbation after a late pulse is carried mainly by D1 and cortex.
- A sharper test would silence cortex later (e.g. from 1.5 s after the cue) to ask whether a trial past the PNR still needs cortex to finish the movement.

Regular operation with cortex off (baseline; the seeds' plots are in `figs/ctx_off_seed<k>_*.png`):

![cortex off, baseline, activity by area](figs/ctx_off_seed13_activity_by_area.png)
![cortex off, baseline, output](figs/ctx_off_seed13_output_activity.png)

Dynamic PNR (stop pulse) with cortex off, for each network:

| baseline | seed 9 | seed 17 | seed 21 |
|---|---|---|---|
| ![](figs/ctx_off_seed13_pnr_dyn_outputs.png) | ![](figs/ctx_off_seed9_pnr_dyn_outputs.png) | ![](figs/ctx_off_seed17_pnr_dyn_outputs.png) | ![](figs/ctx_off_seed21_pnr_dyn_outputs.png) |
| ![](figs/ctx_off_seed13_dynamic_pnr.png) | ![](figs/ctx_off_seed9_dynamic_pnr.png) | ![](figs/ctx_off_seed17_dynamic_pnr.png) | ![](figs/ctx_off_seed21_dynamic_pnr.png) |

dSPN-pulse PNR with cortex off (baseline):

![cortex off dSPN pulse](figs/ctx_off_seed13_dspn_pnr.png)

---

## 7. Putting it together: a picture of the PNR

1. **After the cue the network rides a slow, shared trajectory** (the ramp) through the striatum–cortex–thalamus loop. The loop is stable early: perturbations decay within ~0.1–0.5 s and a trial cannot be pushed over.
2. **From about 1 s after the cue the loop approaches instability.**
   - The restoring force weakens (critical slowing).
   - The barrier to the "move" state shrinks (smaller pulses suffice).
   - The trajectory becomes sensitive to perturbations in *both* directions: a dSPN-excitatory or weak iSPN-inhibitory pulse can now knock a trial off course (cancel), and a strong iSPN-inhibitory pulse can push it over.
   - When this starts is set by time since the cue. It is the same with SNc free, frozen or off (baseline). The cancellable window coincides with each network's own critical-slowing peak.
3. **Late in that window, about 0.6–1.0 s before each trial's own movement, the trial passes the PNR.**
   - From then on no pulse cancels it, and in the baseline no stop pulse even shifts its timing.
   - This edge is fixed relative to the trial's own movement at pulse delays of 2.0–2.4 s (seed 21: 2.2–2.4 s). Earlier in the delay the clock window, not the edge, limits what can be cancelled.
   - This matches the Jacobian: the loop's linearisation becomes unstable about 1.2 s before movement, after which the trajectory is being carried away from the stable region.
4. **What sets the PNR for a given trial:**
   - the time since the cue, which opens and closes the cancellable window (the largest single factor for cancellation);
   - its position along the ramp (at a fixed delay, the state predicts triggering almost perfectly in the baseline, and time to movement predicts cancellation modestly);
   - the size of the barrier, which shrinks with time since the cue;
   - dopamine, which lowers the barrier, shortens the cancellable phase and speeds the escape;
   - **continuing cortical activity**, without which there is no ramp, no instability and no commitment.
5. **Commitment is expressed through iSPN → thalamus disinhibition.** dSPN inhibition cannot produce it.

---

## 8. Caveats

- **Seed 17's responding is largely not cue-driven** (53% of no-cue trials respond in the PNR test). Its ramp and switch statistics include spontaneous moves.
- **The valid-trial filters bias late delays toward slow trials.** "Triggered" requires the twin not to have moved within 500 ms of the pulse, and the relaxation metrics require 300 ms before movement.
- **CPU and GPU numbers come from different noise keys and platforms.** The paired-pulse analysis (Figs 1 middle/bottom, 2–4, 6, 7) ran on the CPU; the stop-pulse curves (Fig 1 top), critical slowing (Fig 5) and the cortex-off stages ran on the GPU. They agree within sampling error where they overlap but are not interchangeable trial by trial.
- **Cortex-off cuts long cues.** Cortex is silenced 100 ms after cue onset, so in the cue-duration sweep, cues longer than 100 ms are cut short.
- **The e-trace features are recomputed** from s and ignore the per-step noise on the SNc state, so they are approximate.
- **Time to movement and time since the cue are confounded** because cue-only RTs are tightly clustered (IQR about 0.2–0.4 s). The PNR "edge" is a constant distance from movement only at pulse delays of 2.0–2.4 s (seed 21: 2.2–2.4 s); earlier, it moves with delay.
- **Time to movement is a future quantity.** It locates the PNR but is not a mechanism. The mechanistic statements rest on the clock-fixed state AUCs, the pulse-type and amplitude dissection, the SNc and cortex clamps, and the Jacobian.
- **The Jacobian and the SNc dissection were run on the baseline only.**
- **Four networks, all from one config.** Seeds 9, 17 and 21 differ from the baseline in important ways (Section 1, point 6). The PNR timing (~0.6–1.0 s before movement) and the iSPN pathway are the features that hold across all four. Section 9 tests nine more seeds: the iSPN pathway holds in all 13, the cortex requirement in 12, the movement-locked edge in about half.

## 9. Do the findings generalise? Thirteen networks

*Added 2026-10-12.* The full test series (Sections 2–6 plus the PD/HD lesions) was run on every 3k-batch seed with an opto score above 0.75 on `seeds/similarity.py`: seeds **29, 22, 7, 23, 1, 3, 4, 24, 6** (new) and 13, 17 (already done). Seeds 9 and 21 (opto 0.75 and 0.50) are kept from the earlier runs. Seeds 10 and 14 score exactly 0.75 and were not run. "Similarity" is the one-number index from `seeds/similarity.py` (0–100, baseline = 100): graded versions of the five scorecard properties, averaged. Tables are printed by `experiments/report/seed_comparison.py`; networks are ordered by similarity.

**Definitions used here** (where they differ from earlier sections):
- The critical-slowing peak and the cancellation trough are searched only over pulse delays 0.8–2.4 s, the mid-delay window the report is about. "none" = censored fraction < 0.1, or P(survive) never below 0.9, in that window. Early effects (0.2–0.6 s after the cue) are listed separately.
- "Triggers ≥ 50%" = delays where a stop −5 pulse makes the trial cross threshold within 500 ms on at least half the valid trials. This replaces the latency-based "switch" description, which only counts responders.

![Fig 8](figs/seed_comparison.png)

**Fig 8.** All 13 networks (baseline in black, the rest coloured by similarity rank). Top: stop-pulse P(respond) (GPU), stop −5 latency, iSPN −5 triggering. Bottom: critical slowing (censored fraction), dSPN +1 cancellation, and cancellation trough vs critical-slowing peak.

### 9a. Behaviour and the stop-pulse PNR

| network | similarity | opto score | respond | RT median (s) | no-cue moves | cue-only ambiguous peaks | late stop pulse P(respond) | stop −5 median latency at 1.0 / 2.0 s | stop −5 triggers ≥ 50% within 500 ms at |
|---|---|---|---|---|---|---|---|---|---|
| **13 (baseline)** | 100 | 1.00 | 0.90 | 2.96 | 0.55 | 0.07 | 1.00 | 1.32 / 0.97 | never |
| 17 | 77 | 0.92 | 0.56 | 2.93 | 0.53 | 0.48 | 1.00 | 0.26 / 0.17 | 0.8–2.8 s |
| 9 | 70 | 0.75 | 0.21 | 2.82 | 0.05 | 0.76 | 1.00 | 0.94 / 0.48 | 2.0–2.8 s |
| 21 | 68 | 0.50 | 0.75 | 2.93 | 0.04 | 0.29 | 1.00 | 0.29 / 0.21 | 1.2–2.8 s |
| 29 | 66 | 1.00 | 0.53 | 3.12 | 0.39 | 0.49 | 0.63 | 2.12 / 1.17 | never |
| 22 | 65 | 1.00 | 0.69 | 2.96 | 0.57 | 0.34 | 0.94 | 1.58 / 0.78 | 2.4–2.8 s |
| 7 | 57 | 1.00 | 0.59 | 3.32 | 0.00 | 0.45 | 0.85 | 2.38 / 1.11 | 2.8 s |
| 23 | 55 | 1.00 | 0.50 | 3.35 | 0.00 | 0.59 | 0.30 | 0.15 / 0.19 | 0.2–1.6, 2.8 s |
| 1 | 54 | 0.96 | 0.44 | 2.87 | 0.09 | 0.51 | 0.99 | 0.82 / 0.23 | 1.8–2.8 s |
| 3 | 52 | 0.82 | 0.17 | 2.82 | 0.00 | 0.79 | 0.98 | 1.02 / 0.60 | 2.2–2.8 s |
| 4 | 52 | 0.83 | 0.23 | 3.28 | 0.19 | 0.74 | 0.01 | 2.13 / 0.84 | never |
| 24 | 46 | 0.98 | 0.30 | 3.41 | 0.00 | 0.74 | 0.31 | 1.70 / 1.18 | 2.4–2.8 s |
| 6 | 38 | 0.94 | 0.56 | 3.92 | 0.00 | 0.27 | 0.79 | 0.51 / 0.34 | 1.6–2.8 s |

- **The late stop pulse triggers reliably (≥ 0.94) in only 7 of 13:** the original four plus 22, 1 and 3. It is weaker in 29 (0.63), 6 (0.79) and 7 (0.85), and weak or reversed in 24 (0.31), 23 (0.30) and 4 (0.01).
  - **Seed 23 is inverted:** stop pulses trigger immediately early in the delay (≥ 50% at 0.2–1.6 s) and less and less later.
  - **Seed 4:** a stop pulse almost never triggers, yet an iSPN-only pulse does (0.99, next table). In this network the dSPN half of the all-SPN pulse dominates.
- **Fast "switch" networks are a minority.** Only 17 and 21 (and inverted 23) trigger on ≥ 50% of stop pulses within 500 ms from ≤ 1.2 s after the cue. Most seeds trigger only late (from 1.6–2.8 s) or never (baseline, 29, 4), like the baseline's "progressive" regime.
- **Natural responses barely cross threshold in most seeds** (cue-only ambiguous peaks 0.27–0.79, against 0.07 in the baseline). The baseline remains the only network with a clean, stereotyped response.

### 9b. Critical slowing, cancellation and the edge

| network | critical-slowing peak, 0.8–2.4 s (censored) | dSPN +1 cancellation trough, 0.8–2.4 s (P(survive)) | early (0.2–0.6 s): max censored / min P(survive) | late edge at 2.0 / 2.2 / 2.4 s | iSPN −5 max P(trig) | dSPN −5 max P(trig) | min stop |a| for 50% trig, 1.0 → 2.6 s |
|---|---|---|---|---|---|---|---|
| **13 (baseline)** | 2.0 s (0.69) | 1.6 s (0.56) | 0.00 / 1.00 | 1.0 / 1.0 / 1.0 | 0.89 | 0.00 | – → – |
| 17 | 1.8 s (0.83) | 1.8 s (0.32) | 0.07 / 0.96 | 0.8 / 0.8 / 0.8 | 1.00 | 0.00 | 5 → 0.25 |
| 9 | 1.2 s (0.99) | 1.2 s (0.21) | 0.19 / 0.81 | 1.0 / – / – | 1.00 | 0.00 | – → 0.5 |
| 21 | 2.0 s (0.70) | 2.2 s (0.56) | 0.00 / 0.96 | – / 0.6 / 0.6 | 1.00 | 0.00 | 10 → 0.25 |
| 29 | 1.8 s (0.95) | 1.6 s (0.07) | 0.79 / 0.55 | – / – / 0.6 | 0.99 | 0.00 | – → – |
| 22 | 1.6 s (0.74) | 1.8 s (0.79) | 0.00 / 0.98 | 1.0 / 0.8 / 0.8 | 1.00 | 0.00 | – → 0.5 |
| 7 | 2.4 s (0.75) | none (0.95) | 0.11 / 0.63 | – / – / – | 1.00 | 0.00 | – → – |
| 23 | 2.2 s (0.53) | 2.2 s (0.80) | 0.00 / 0.95 | 1.2 / 1.0 / 1.0 | 1.00 | 0.00 | 5 → 10 |
| 1 | 1.6 s (0.47) | 1.8 s (0.64) | 0.00 / 0.94 | 0.8 / 1.0 / 0.8 | 1.00 | 0.00 | – → 1 |
| 3 | 0.8 s (0.63) | 1.6 s (0.16) | 0.56 / 0.72 | 0.8 / 0.8 / – | 1.00 | 0.00 | – → 1 |
| 4 | 2.4 s (0.35) | 2.2 s (0.38) | 0.00 / 0.90 | – / – / – | 0.99 | 0.00 | – → – |
| 24 | none (0.00) | 1.6 s (0.76) | 0.00 / 0.76 | 1.6 / 1.4 / – | 1.00 | 0.00 | – → 5 |
| 6 | 2.4 s (0.57) | 2.2 s (0.66) | 0.91 / 0.29 | – / – / 1.4 | 1.00 | 0.00 | 10 → 2.5 |

Per-delay cancellation edge (time to own movement below which a dSPN +1 pulse no longer cancels; same rule as Section 5):

| pulse delay | 0.8 s | 1.0 s | 1.2 s | 1.4 s | 1.6 s | 1.8 s | 2.0 s | 2.2 s | 2.4 s |
|---|---|---|---|---|---|---|---|---|---|
| **13 (baseline)** | – | – | – | – | – | – | 1.0 | 1.0 | 1.0 |
| 17 | 2.4 | 1.8 | 1.6 | – | – | – | 0.8 | 0.8 | 0.8 |
| 9 | – | – | – | – | – | 1.0 | 1.0 | – | – |
| 21 | – | – | – | 1.6 | 1.4 | – | – | 0.6 | 0.6 |
| 29 | – | – | – | – | – | – | – | – | 0.6 |
| 22 | – | – | – | 1.6 | – | – | 1.0 | 0.8 | 0.8 |
| 7 | – | – | – | – | – | – | – | – | – |
| 23 | 2.8 | 2.6 | – | – | – | 1.4 | 1.2 | 1.0 | 1.0 |
| 1 | – | – | 1.6 | 1.4 | – | 1.0 | 0.8 | 1.0 | 0.8 |
| 3 | – | – | – | – | – | – | 0.8 | 0.8 | – |
| 4 | 2.6 | 2.4 | 2.2 | 2.0 | – | – | – | – | – |
| 24 | – | 2.6 | 2.4 | 2.2 | 2.0 | 1.8 | 1.6 | 1.4 | – |
| 6 | 2.8 | – | 2.4 | – | – | – | – | – | 1.4 |

- **Commitment through iSPNs holds in all 13 networks.** An iSPN −5 pulse triggers in every one (max P(trig) 0.89–1.00); a dSPN −5 pulse never triggers in any (0.00 at every delay). This is the most robust result in the report.
- **The cancellation trough lines up with critical slowing in 10 of the 11 networks that have both** (within 0.4 s). Seed 3 is the exception on this rule (critical slowing peaks at 0.8 s, cancellation at 1.6 s), but its 0.8 s value is the tail of its early post-cue period (censored 0.48 → 0.63 from 0.2 to 0.8 s); its only mid-delay bump (0.32 at 1.4 s) is 0.2 s from its trough. Seed 7 has critical slowing (0.75 at 2.4 s) but cancels almost nothing in the window; seed 24 cancels (trough 0.76) without any trial failing to decay.
- **Some seeds have a second, early unstable period right after the cue:** perturbations at 0.2–0.6 s fail to decay in 29 (0.79), 3 (0.56) and 6 (0.91), and in 6 an early dSPN pulse cancels most responses (P(survive) 0.29). The original four have no such period.
- **The edge.** Three patterns:
  - **Constant late edge, 0.6–1.0 s before movement (movement-locked PNR):** baseline, 17, 21, 22, 1, 3 (6 of 13; seed 9 and 29 have only one or two cells, consistent with 0.6–1.0).
  - **Clock-locked:** in 24 (1.0–2.2 s) and 4 (0.8–1.4 s), the edge falls by 0.2 s for every 0.2 s of delay, so delay + edge is constant (3.6 s and 3.4 s). Cancellable trials are simply those that would have moved after a fixed time since the cue. Seed 23 is clock-locked at 1.8–2.2 s (delay + edge = 3.2 s), then flattens at 1.0 s.
  - **No edge:** 7 and 6.
  - So the "fixed distance before movement" reading holds in about half the networks. Elsewhere the data cannot separate it from a clock (the confound in Section 8).
- **The shrinking barrier** (stop amplitude needed for 50% triggering falls from 1.0 to 2.6 s) is measurable only where stop pulses trigger at both delays: 17 (5 → 0.25), 21 (10 → 0.25), 6 (10 → 2.5). Seed 23 runs backwards (5 → 10), matching its inverted stop-pulse curve.

### 9c. Dopamine and cortex

| network | SNc off: cue-only respond | SNc off: iSPN −5 max P(trig) | min P(survive) dSPN +1: intact / SNc off / SNc sat | SNc sat: respond | cortex off: max P(respond), any test | cortex off: max censored | SNc Δs after cue |
|---|---|---|---|---|---|---|---|
| **13 (baseline)** | 0.24 | 0.63 | 0.56 / 0.15 / 0.93 | 0.99 | 0.013 | 0.00 | +0.068 |
| 17 | 0.00 | 1.00 | 0.32 / – / 1.00 | 1.00 | 0.000 | 0.00 | +0.041 |
| 9 | 0.00 | 0.98 | 0.21 / – / 0.97 | 0.99 | 0.000 | 0.00 | +0.024 |
| 21 | 0.00 | 0.62 | 0.56 / – / 1.00 | 1.00 | 0.000 | 0.00 | +0.137 |
| 29 | 0.51 | 0.99 | 0.07 / 0.07 / 1.00 | 1.00 | 0.000 | 0.00 | +0.000 |
| 22 | 0.43 | 1.00 | 0.79 / 0.67 / – | 0.00 | 0.000 | 0.00 | +0.006 |
| 7 | 0.00 | 0.00 | 0.95 / – / 0.34 | 0.27 | 0.000 | 0.00 | +0.078 |
| 23 | 0.01 | 1.00 | 0.80 / – / 0.95 | 0.98 | 1.000 | – | +0.012 |
| 1 | 0.00 | 1.00 | 0.64 / – / 0.65 | 0.12 | 0.000 | 0.00 | +0.055 |
| 3 | 0.00 | 1.00 | 0.16 / – / – | 0.01 | 0.000 | 0.00 | +0.018 |
| 4 | 0.00 | 0.00 | 0.38 / – / 0.96 | 0.98 | 0.000 | 0.19 | +0.065 |
| 24 | 0.00 | 0.26 | 0.76 / – / 1.00 | 1.00 | 0.000 | 0.00 | +0.052 |
| 6 | 0.00 | 1.00 | 0.66 / – / 1.00 | 1.00 | 0.000 | 0.00 | +0.080 |

- **Cortex off: no responses in 12 of 13 networks** (max P(respond) ≤ 0.013 in every test). With cortex off, critical slowing is absent in 11 of the 12 where it can be measured; seed 4 keeps a small censored fraction (0.19) but never responds.
- **Seed 23 is the exception: silencing cortex itself triggers a response.** It responds at about 0.2 s after the cue, 0.1 s after the silencing starts, on 97% of cue trials and on 98% of no-cue trials (where silencing starts at the same time). So its cortex-off dSPN-pulse metrics are undefined (every twin has moved before the pulses). The mechanism was not examined.
- **SNc stimulation suppresses responding in 4 seeds**: 22 (0.69 → 0.00), 1 (0.44 → 0.03), 3 (0.17 → 0.00) and 7 (0.59 → 0.21). The GPU clamp and the CPU paired run agree. In these networks, cue-only peak output falls with SNc saturated (about 0.50 → 0.40–0.43), whereas in the baseline it rises (0.57 → 0.68). In the other 9 networks SNc stimulation gives ≥ 98% responding, as in the original four. The opto score does not test SNc, so selecting on it did not guarantee dopamine's sign. No mechanism is claimed here.
- **Saturating SNc abolishes cancellation (min P(survive) ≥ 0.93) in 9 of the 11 networks where it can be measured**, but not in 7 (0.34) or 1 (0.65). In 22 and 3 nothing responds with SNc saturated, so cancellation is undefined.
- **SNc off:** responding survives only in the baseline (0.24), 22 (0.43) and 29 (0.51).
  - **Seed 29's SNc is inert:** its dopamine signal does not move after the cue (Δs = 0.000) and clamping SNc off leaves responding unchanged (0.52 vs 0.53), so its survival says nothing about dopamine.
  - Seed 22's SNc barely moves after the cue (Δs = +0.006), but SNc inhibition still lowers responding (0.69 → 0.44), so its SNc does matter.
  - With SNc off, an iSPN −5 pulse still triggers in 10 of 13 (≥ 0.62); not in 7, 4 (0.00) or 24 (0.26).

### 9d. Lesions

| network | intact | SNc inhibited | SNc stimulated (RT) | PD acute | PD acute + SNc stim | PD retrained | HD acute (on time) | HD retrained (on time) |
|---|---|---|---|---|---|---|---|---|
| **13 (baseline)** | 0.90 | 0.22 | 1.00 (2.49 s) | 0.17 | 0.88 | 0.00 | 1.00 (0.00) | 0.30 (0.30) |
| 17 | 0.56 | 0.00 | 1.00 (1.01 s) | 0.00 | 0.00 | 0.19 | 1.00 (0.00) | 0.41 (0.39) |
| 9 | 0.21 | 0.00 | 0.99 (1.76 s) | 0.00 | 0.00 | 0.69 | 1.00 (0.00) | 0.68 (0.67) |
| 21 | 0.75 | 0.00 | 1.00 (2.05 s) | 0.00 | 0.00 | 0.98 | 1.00 (0.00) | 0.35 (0.35) |
| 29 | 0.53 | 0.52 | 1.00 (0.88 s) | 0.39 | 0.00 | 0.53 | 1.00 (0.00) | 0.37 (0.36) |
| 22 | 0.69 | 0.44 | 0.00 (– s) | 0.31 | 1.00 | 0.58 | 1.00 (0.00) | 0.52 (0.52) |
| 7 | 0.59 | 0.00 | 0.21 (2.40 s) | 0.00 | 0.02 | 0.30 | 1.00 (0.00) | 0.00 (0.00) |
| 23 | 0.50 | 0.02 | 0.98 (3.07 s) | 0.09 | 0.18 | 0.00 | 1.00 (0.00) | 0.41 (0.34) |
| 1 | 0.44 | 0.00 | 0.03 (1.92 s) | 0.00 | 0.33 | 0.33 | 1.00 (0.00) | 0.00 (0.00) |
| 3 | 0.17 | 0.00 | 0.00 (1.11 s) | 0.00 | 0.05 | 0.21 | 1.00 (0.00) | 0.69 (0.19) |
| 4 | 0.23 | 0.00 | 0.99 (1.81 s) | 0.00 | 0.00 | 0.00 | 0.00 (0.00) | 0.72 (0.72) |
| 24 | 0.30 | 0.00 | 1.00 (2.50 s) | 0.00 | 0.95 | 0.00 | 1.00 (0.00) | 0.27 (0.25) |
| 6 | 0.56 | 0.00 | 1.00 (0.56 s) | 0.00 | 0.04 | 0.73 | 1.00 (0.00) | 0.28 (0.22) |

- **HD acute** (8 of 10 iSPNs lesioned) saturates the output before the cue in 12 of 13 (respond 1.00, on time 0.00). Seed 4 is the exception (no responses). Retraining recovers 0–72% on-time responding.
- **The baseline's lesion behaviour is not unique among these seeds.** Earlier (`experiments/README.md`) only the baseline, of 9/17/21, was rescued from acute PD by SNc stimulation and collapsed after PD retraining. In the larger set:
  - PD acute is rescued by SNc stimulation in 22 (1.00) and 24 (0.95) too, partly in 1 (0.33);
  - PD retraining collapses to 0 in 23, 4 and 24 too;
  - responding survives SNc inhibition in 22 (0.44) and 29 (trivially, inert SNc).

### 9e. What generalises

| finding (Sections 1–7) | holds in |
|---|---|
| iSPN −5 triggers; dSPN −5 never does | **13 / 13** |
| cortex off from cue + 100 ms → no responses | **12 / 13** (seed 23: silencing itself triggers) |
| cortex off → no critical slowing | 11 / 12 measurable (seed 4 keeps 0.19 censored) |
| cancellation trough within 0.4 s of the critical-slowing peak | 10 / 11 measurable (not 3; 7 and 24 lack one of the two) |
| SNc saturation abolishes cancellation | 9 / 11 measurable |
| SNc stimulation raises responding | 9 / 13 (suppresses in 22, 1, 3, 7) |
| late stop pulse triggers ≥ 0.94 | 7 / 13 |
| movement-locked late edge at 0.6–1.0 s | 6 / 13 clearly (baseline, 17, 21, 22, 1, 3); clock-locked in 24, 4, 23 |
| fast "switch" regime (stop pulse triggers ≥ 50% from ≤ 1.2 s) | 2 / 13 (17, 21) |
| clean, stereotyped responses (cue-only ambiguous peaks < 0.1) | 1 / 13 (baseline) |

The pathway (iSPN → thalamus disinhibition), the need for ongoing cortical activity, and a cancellable window aligned with critical slowing are general properties of networks trained on this task. The movement-locked PNR, the all-or-none stop-pulse response and the dopamine sign vary between networks, and the baseline is the clearest case of each.

---

## Files

- **This report and figures:** `experiments/report/` (`make_figs.py` builds `figs/`; `figs/numbers.json` holds the numbers quoted here).
- **Paired-pulse analysis:** `experiments/pnr_determinants.py`; one folder per network and condition in `experiments/pnr_determinants/`, each with a nine-panel `pnr_determinants.png`, `metrics.json` and `data.npz`.
- **Cortex-off stages:** `experiments/baseline/ctx_inh_*` and `experiments/seeds/seed_<k>/baseline/ctx_inh_*`; hook in `experiments/perturb_model.py` (`ctx_off`, `CTX_DELAY` = 10 steps).
- **Cross-network comparison (Section 9):** `experiments/report/seed_comparison.py` (prints the tables; writes `figs/seed_comparison.png` and `figs/seed_comparison.json`). Seed similarity index: `seeds/similarity.py`. Runs: `experiments/run_opto_seeds.sh`.
- **Per-network determinant figures** (copied into `figs/`): `determinants_seed13.png`, `determinants_seed9.png`, `determinants_seed17.png`, `determinants_seed21.png`.
