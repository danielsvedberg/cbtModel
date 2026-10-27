"""Collect seeds/seed_*/metrics_*.json into seeds/summary.md and seeds/seed_comparison.png.

usage (from repo root): python seeds/summarize.py [subfolder, e.g. 3k]
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), *sys.argv[1:2])
AREAS = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']


def load(s, stage):
    p = os.path.join(HERE, s, f'metrics_{stage}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, fmt='{:.2f}'):
    return '–' if v is None else fmt.format(v)


# rows are directory names; the frozen baseline params first, then each trained seed
seeds = (['baseline_12a8ade'] if os.path.isdir(os.path.join(HERE, 'baseline_12a8ade')) else []) + \
    [f'seed_{k}' for k in sorted(int(d.split('_')[-1]) for d in glob.glob(os.path.join(HERE, 'seed_*')) if os.path.isdir(d))]
THREEK = sys.argv[1:2] == ['3k']
LABELS = {'baseline_12a8ade': '**baseline** (13, frozen params)', 'seed_13': '13 (= baseline)' if THREEK else '13 (retrained)'}
lab = lambda s: LABELS.get(s, s.split('_')[-1])
# the baseline's training numbers come from its original log (baselines/2026-10-08_12a8ade/training_log.txt)
BASE_TRAIN = dict(best_logged_loss=0.006769335828721523, best_step=800, final_logged_loss=0.013384486548602581)
L = ['# Init-seed comparison', '',
     'Weight-init seed (`keyind`) varied; everything else is the baseline config '
     f'(T_start draw PRNGKey(13), training noise PRNGKey(0), {3000 if THREEK else 2000} iterations). '
     'Produced by `seeds/run_seed.py`; per-seed plots in `seeds/seed_<k>/plots/`. ' +
     ('Seed 13 is the baseline seed retrained alone; its params are bit-identical to the baseline\'s. ' if THREEK else
      '"13 (retrained)" uses the baseline seed but trained in this batch; it did not reproduce the baseline (see below). ') +
     'The baseline row evaluates the frozen params from `baselines/2026-10-08_12a8ade/`.', '']

L += ['## Training and timing (testing_script, 100 noise seeds × 6 cue times)', '',
      '| seed | best logged loss | best step | final loss | respond | on time (RT 2.5–3.5 s) | RT median | RT IQR |',
      '|---|---|---|---|---|---|---|---|']
for s in seeds:
    tr, te = (BASE_TRAIN if s == 'baseline_12a8ade' else load(s, 'train') or {}), load(s, 'test') or {}
    iqr = te.get('rt_iqr')
    L.append(f"| {lab(s)} | {f(tr.get('best_logged_loss'), '{:.4f}')} | {tr.get('best_step', '–')} | "
             f"{f(tr.get('final_logged_loss'), '{:.4f}')} | {f(te.get('respond'))} | {f(te.get('on_time_2p5_3p5'))} | "
             f"{f(te.get('rt_median'))} s | {'–' if not iqr else f'{iqr[0]:.2f}–{iqr[1]:.2f} s'} |")

L += ['', '## Ramping and recruitment', '',
      'Per area (mean rate over units). *slope/r*: ramp onset (20% of rise, ≥1 s after cue) regressed on RT across trials; '
      'slope ≈ 1 means the ramp shifts with movement time. *lead*: median RT − ramp onset. '
      '*units*: fraction of units whose peak rate (1 s after cue → RT) exceeds pre-cue by > 0.05.', '',
      '| seed | ' + ' | '.join(f'{a} slope / r / lead / units' for a in AREAS) + ' |',
      '|---|' + '---|' * len(AREAS)]
for s in seeds:
    te = load(s, 'test') or {}
    cells = []
    for a in AREAS:
        d = te.get(a) or {}
        cells.append(f"{f(d.get('ramp_onset_vs_rt_slope'))} / {f(d.get('ramp_onset_vs_rt_r'))} / "
                     f"{f(d.get('ramp_lead_median_s'), '{:.1f}')} s / {f(d.get('frac_units_modulated'))}")
    L.append(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')

L += ['', '## Opto (1000 trials; RT from cue onset; strongest strength vs. no stim)', '',
      '| seed | control RT | inh dSPN RT (respond) | inh iSPN RT (respond) | stim dSPN RT (respond) | stim iSPN RT (respond) | inh iSPN faster & inh dSPN slower? |',
      '|---|---|---|---|---|---|---|']
for s in seeds:
    op = load(s, 'opto')
    if not op:
        L.append(f'| {lab(s)} |' + ' – |' * 6); continue
    ctrl = op['inh dMSN'][0]
    cells, rts = [], {}
    for g in ['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']:
        r = max(op[g], key=lambda x: abs(x['strength']))
        rts[g] = r['rt_median']
        cells.append(f"{f(r['rt_median'])} s ({r['respond']:.2f})")
    c = ctrl['rt_median']
    ok = None if None in (c, rts['inh dMSN'], rts['inh iMSN']) else (rts['inh iMSN'] < c < rts['inh dMSN'])
    L.append(f"| {lab(s)} | {f(c)} s ({ctrl['respond']:.2f}) | " + ' | '.join(cells) +
             f" | {'–' if ok is None else ('yes' if ok else 'no')} |")

L += ['', '## Point of no return (test_pnr_nm, 150 noise seeds)', '',
      '*ambiguous peaks*: fraction of perturbed trials whose peak output lands in 0.4–0.5 (just under threshold); '
      'low = all-or-none. Response probability by stop-pulse delay after cue.', '']
pn = {s: load(s, 'pnr') for s in seeds}
ref = next((p for p in pn.values() if p), None)
if ref:
    delays = ref['pert_delay_ms']
    L += ['| seed | no-cue respond | ambiguous peaks | ' + ' | '.join(f'{d / 1000:.1f} s' for d in delays) + ' |',
          '|---|---|---|' + '---|' * len(delays)]
    for s in seeds:
        p = pn[s]
        if not p:
            L.append(f'| {lab(s)} | – | – |' + ' – |' * len(delays)); continue
        L.append(f"| {lab(s)} | {p['no_cue_prob']:.2f} | {p['peak_frac_ambiguous_0p4_0p5']:.2f} | " +
                 ' | '.join(f'{v:.2f}' for v in p['pert_prob']) + ' |')
open(os.path.join(HERE, 'summary.md'), 'w').write('\n'.join(L) + '\n')

# comparison figure: training loss, opto RT shifts, PNR curve, cue-magnitude curve
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
cmap = plt.cm.tab10
# with many seeds, colour only the baseline and the best-scoring seeds (seeds/<sub>/scorecard.md); the rest grey
HL = ['seed_13', 'seed_9', 'seed_21', 'seed_17', 'seed_29', 'seed_1'] if len(seeds) > 10 else seeds
order = [s for s in seeds if s not in HL] + [s for s in HL if s in seeds]  # highlighted drawn on top
for s in order:
    i = HL.index(s) if s in HL else None
    col = ('k' if s in ('seed_13', 'baseline_12a8ade') else cmap(i)) if i is not None else (0.8, 0.8, 0.8)
    lw = 2.5 if s in ('seed_13', 'baseline_12a8ade') else 1.2 if i is not None else 0.6
    lp = os.path.join(HERE, s, 'losses.npy')
    if os.path.exists(lp):
        ax[0].plot(np.load(lp), color=col, lw=lw * 0.5, label=lab(s) if i is not None else None)
    op = load(s, 'opto')
    if op:
        c = op['inh dMSN'][0]['rt_median']
        for j, g in enumerate(['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']):
            r = max(op[g], key=lambda x: abs(x['strength']))['rt_median']
            if r is not None and c is not None:
                ax[1].scatter(j + (seeds.index(s) - len(seeds) / 2) * 0.02, r - c, color=col, s=60 if s == 'baseline_12a8ade' else 25)
    p = pn.get(s)
    if p:
        ax[2].plot(np.array(p['pert_delay_ms']) / 1000, p['pert_prob'], 'o-', color=col, lw=lw, ms=3)
        ax[3].plot(p['cue_mag'], p['cue_mag_prob'], 'o-', color=col, lw=lw, ms=3)
ax[0].set(xlabel='iteration', ylabel='training loss', ylim=(0, 0.03), title='training')
ax[0].legend(fontsize=7, title='init seed' + (' (grey: others)' if len(seeds) > 10 else ''))
ax[1].axhline(0, color='gray', lw=0.8)
ax[1].set_xticks(range(4), ['inh dSPN', 'inh iSPN', 'stim dSPN', 'stim iSPN'])
ax[1].set(ylabel='Δ median RT vs. control (s)', title='opto (strongest)')
ax[2].set(xlabel='stop-pulse delay after cue (s)', ylabel='P(respond)', ylim=(-0.03, 1.03), title='dynamic PNR')
ax[3].set(xlabel='cue magnitude', ylabel='P(respond)', ylim=(-0.03, 1.03), title='cue magnitude')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'seed_comparison.png'), dpi=150)
print('\n'.join(L))
