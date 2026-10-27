# Thalamic excitation/inhibition and commitment (2026-10-13)

**Hypothesis tested:** commitment (the point of no return, PNR) is set by the balance of excitation and inhibition, and by how fast each is changing, at the thalamic units the output is read from.

**Verdict: not supported as stated, with one structural exception.**
- The net drive onto the readout thalamus tracks commitment about as well as output y and ramp progress do, but no better.
- A balance measure (E/I ratio, d(E−I)/dt, gain balance) adds nothing once time since the cue and circuit state are known.
- Pulse types do not share one E−I threshold.
- The timing of the E−I rise does not follow the PNR across networks.
- **What does hold:** across 11 networks, a readout thalamus whose striatal input weights favour iSPNs over dSPNs predicts an earlier PNR (Spearman ρ = −0.83, p = 0.001; robust to dropping any one network).

## What was measured

Scripts:
- `experiments/thalamic_ei.py <seed>`: one network.
- `experiments/run_thalamic_ei.sh`: all 13 networks (CPU).
- `experiments/thalamic_ei_summary.py`: tables (`summary_tables.md`), `summary.png` and `structural_ei_vs_edge.png`.

**Design.** The paired-pulse simulation from `pnr_determinants.py` was re-run with the same noise keys and conditions (200 noise seeds; a cue-only twin plus 13 pulse types × 14 delays). Every pulse and twin trial reproduces the PNR-determinants response times exactly, in all 13 networks.

**Inputs to thalamic unit i** (exactly as in `multiregion_nmrnn`; r = nln(x)):
- E_i = Σ |B_tbg(i, dSPN)| r_dSPN + Σ max(J_t, 0) r_thal
- I_i = Σ |B_tbg(i, iSPN)| r_iSPN + Σ max(−J_t, 0) r_thal

Each is averaged over thalamic units with weights w_i = |C_i| / Σ|C|, since y = |C|·nln(x_t) + rb. Both are split into a striatal part (E_bg, I_bg) and a thalamic recurrent part (E_rec, I_rec).

**Derived quantities:**
- **Rates:** dE/dt and dI/dt over 100 ms.
- **Gains:** G_E and G_I, the same weights multiplied by nln′ of the presynaptic state.
- **Thalamic loop gain:** λ_t, the largest real eigenvalue of J_t·diag(nln′(x_t)).

## Results

The full tables are in `summary_tables.md`.

**1. The E−I rise is not a commitment event.**
- In the baseline, excitation outpaces inhibition for almost the whole ramp: mean dE/dt > dI/dt from 2.6 s before movement. In 11 of 13 networks, inhibition is rising as fast as or faster than excitation at the moment of movement.
- The E−I rise that peaks just before movement begins 0.2–1.75 s before RT. That is no earlier than the rise in y itself (y's rise starts earlier than E−I's in 7 of 13 networks).
- Across networks, its onset does not track the PNR edge (ρ = +0.05, n = 11).

**2. At a fixed time since the cue, balance measures predict triggering worse than ramp progress.** For a stop −5 pulse, the within-delay AUCs are:

| network | E−I | E/I | d(E−I)/dt | progress |
|---|---|---|---|---|
| baseline | 0.99 | 0.68 | 0.75 | 0.99 |
| other 11 measurable | 0.42–0.97 | 0.46–0.91 | 0.39–0.69 | 0.52–0.95 |

- The gain balance G_E − G_I runs the *wrong* way in 9 of 12 networks (AUC < 0.5); only seeds 17 and 9 go the predicted way: networks are less likely to trigger when gain favours excitation.
- High E−I AUCs say little in themselves. E−I is the input that moves x_t, so it is close to a lagged y.

**3. Cancellation: E/I adds nothing beyond time since the cue and state.** For dSPN +1 survival, the 5-fold CV AUC is:
- delay + state: 0.696–0.882;
- adding the E/I features changes it by −0.002 to +0.015;
- adding E/I features shuffled within delay gives −0.008 to +0.004 (seed 23 alone: +0.015 real against −0.019 shuffled).

**4. No single E−I threshold across pulse types.** Measured 100 ms after pulse onset, excluding trials that had already crossed:
- Adding pulse identity to E−I raises the triggering CV AUC by up to +0.18 (e.g. 0.79 → 0.97 in seed 6, 0.88 → 0.98 in seed 21).
- E−I does need less help from pulse type than y or thalamic rate. As the thalamic input, it registers a pulse's effect before y does.
- **The thresholds depend on pulse type.** In the baseline, the E−I at which a type triggers 50% falls with stop-pulse strength (0.15 for stop −0.1 down to 0.03 for stop −10), with iSPN pulses near 0.08–0.09 and dSPN-excitatory pulses near 0.13–0.16. So a strong stop pulse commits the network at a much lower net drive.
- Disinhibitory (iSPN) and excitatory (dSPN +) pulses at matched E−I trigger with similar probability in most networks; the mean difference is −0.13 to +0.08 in 10 of 13. The exceptions are seed 9 (+0.29), seed 17 (−0.60, only 3 bins) and seed 21 (−0.25, 4 bins).

**5. Stop pulses do not trigger by removing net inhibition at the readout thalamus.**
- A stop pulse removes both E_bg and I_bg. Its predicted net effect is therefore I_bg − E_bg just before the pulse.
- **Within every network, that quantity is *anti*-correlated with stop-pulse triggering across delays** (ρ −0.13 to −0.97). As the delay grows, the striatal input to the readout thalamus becomes net excitatory, yet stop pulses trigger more often. The triggering must come from the slower loop dynamics after the pulse (the rebound), not from the instantaneous change in thalamic E/I.
- The prediction also fails for the odd networks:
  - **Seed 4** never triggers, yet its I_bg − E_bg changes sign over the delay, from +0.15 to −0.18.
  - **Seed 23's** inverted curve has no matching sign change in I_bg − E_bg.

**6. The thalamic loop alone is not the instability.**
- Taking the maximum λ_t from cue to RT, its median exceeds 1 on every trial in only 4 networks (22, 7, 23, 1), and in some trials of the baseline (39%) and seed 4 (37%).
- It stays below 1 in the rest (0.31–0.95).
- This fits the cortex-off result: the instability needs the cortex–striatum–thalamus loop.

**7. Structural E/I at the readout thalamus predicts where the PNR sits.** This is the readout-weighted dSPN/iSPN weight ratio onto thalamus (`structural_ei_vs_edge.png`):

| | baseline | 9 | 24 | others |
|---|---|---|---|---|
| dSPN/iSPN ratio | 0.75 | 0.99 | 0.25 | 1.52–1.93 |
| PNR edge (s before movement) | 1.0 | 1.0 | 1.5 | 0.6–1.07, except seed 6 (1.4) |

- The three networks with iSPN-dominated readout input (baseline, 9, 24) have edges of 1.0–1.5 s. The networks with dSPN-dominated input commit later, 0.6–0.87 s before movement, with seeds 23 and 6 as exceptions.
- Spearman ρ = −0.83 (p = 0.0015, n = 11). Dropping any one network leaves ρ between −0.78 and −0.91. With only the clearly movement-locked networks, ρ = −0.88 (n = 8).
- This was one of 5 exploratory cross-network tests; it survives a Bonferroni correction (0.05/5).
- It is a weight property and does not depend on how the E/I time courses evolve. It fits the earlier finding that commitment runs through iSPN → thalamus disinhibition.
- Structural E/I does *not* predict late stop-pulse triggering (ρ −0.13).

## Caveats

- **Edges are on a 0.2 s grid**, averaged over the 2.0–2.4 s delays. For seeds 24, 23 and 6, the edge tracks time since the cue rather than time to movement (report Section 9b).
- **The per-network figures (`seed<k>/thalamic_ei.png`) use older rules:**
  - their "final rise" panel requires the rise to last until RT, which gives 0 s in networks where inhibition catches up just before movement;
  - their single-threshold panel uses the pulse-end state.

  The summary uses the peak-based rise rule and the +100 ms state; use the summary numbers.
- **Readout weighting** uses |C|; rb is a constant offset.
- **DA scaling:** the striatal rates feeding E_bg and I_bg already include the DA gain and the excitability traces. The thalamic weights themselves are not DA-modulated.

## Connectivity matrices

`experiments/connectivity.py` writes to `experiments/connectivity/`:
- `conn_seed<k>.png`: one signed matrix per network;
- `all_networks.png`: all 13 in a grid;
- `block_means.png`: area-level summed input;
- `matrices.npz`.

**How the matrices are drawn:**
- Rows are targets and columns are sources. Red is excitatory and blue is inhibitory, with signs as the model applies them. Each network has its own colour scale.
- Thalamic units are sorted by readout weight |C|.
- Striatal blocks are shown at DA gain 1. In the model they are scaled by exp(±s) and by (1 + e) excitability, which differs several-fold between dSPN and iSPN. Striatum → thalamus weights are not scaled.
- Only the 13 intact networks are plotted; the retrained PD/HD networks are not.

**Note on area-level net input:** averaged over *all* thalamic units, striatal input is net inhibitory in the baseline (−0.4), seed 9 (−0.1) and seed 24 (−1.3). Weighted by readout, as in Result 7, the dSPN/iSPN ratio is below 1 in the baseline (0.75) and seed 24 (0.25), and about 1 in seed 9 (0.99).
