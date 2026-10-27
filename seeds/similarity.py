"""One-number similarity of each seed to the baseline (seed 13); prints a markdown table.

usage (from repo root): python seeds/similarity.py [subfolder, default 3k]
Graded versions of the five scorecard.py criteria, each in [0, 1] with the baseline = 1; index = 100 x their mean.
  timing  : on-time responding (RT 2.5-3.5 s) relative to the baseline's
  ramp    : mean over D1, D2, cortex, thalamus of min(1, slope/0.5) * min(1, r/0.6), ramp onset vs RT
  recruit : mean over the 5 areas of fraction of units modulated, relative to the baseline's (capped at 1)
  opto    : mean over inh/stim dSPN/iSPN (strongest) of the RT shift as a fraction of the baseline's, same sign only
  pnr     : (1 - mean |stop-pulse P(respond) curve - baseline's|) x (1 - excess ambiguous-peak fraction over baseline's / 0.5)
"""
import os, sys, json, glob
import numpy as np
SUB = sys.argv[1] if len(sys.argv) > 1 else '3k'
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUB)
ld = lambda d, s: json.load(open(os.path.join(d, f'metrics_{s}.json')))
cl = lambda x: float(np.clip(x, 0, 1))
AREAS = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']
OPTO = ['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']


def opto_shifts(op):
    c = op['inh dMSN'][0]['rt_median']
    out = {}
    for g in OPTO:
        x = max(op[g], key=lambda x: abs(x['strength']))
        out[g] = None if c is None or x['rt_median'] is None else x['rt_median'] - c
    return out


def components(d, b):
    te, op, pn = ld(d, 'test'), ld(d, 'opto'), ld(d, 'pnr')
    bte, bop, bpn = b
    timing = cl(te['on_time_2p5_3p5'] / bte['on_time_2p5_3p5'])
    ramp = np.mean([cl((te[a]['ramp_onset_vs_rt_slope'] or 0) / 0.5) * cl((te[a]['ramp_onset_vs_rt_r'] or 0) / 0.6)
                    for a in AREAS[:4]])
    rec = np.mean([cl(te[a]['frac_units_modulated'] / bte[a]['frac_units_modulated']) for a in AREAS])
    sh, bsh = opto_shifts(op), opto_shifts(bop)
    opto = np.mean([0.0 if sh[g] is None else cl(sh[g] / bsh[g]) for g in OPTO])
    curve = 1 - np.mean(np.abs(np.array(pn['pert_prob']) - bpn['pert_prob']))
    pnr = curve * cl(1 - max(0, pn['peak_frac_ambiguous_0p4_0p5'] - bpn['peak_frac_ambiguous_0p4_0p5']) / 0.5)
    return te['respond'], [timing, ramp, rec, opto, pnr]


dirs = sorted(filter(os.path.isdir, glob.glob(os.path.join(HERE, 'seed_*'))), key=lambda d: int(d.split('_')[-1]))
b13 = os.path.join(HERE, 'seed_13')
base = tuple(ld(b13, s) for s in ('test', 'opto', 'pnr'))
rows = []
for d in dirs:
    s = int(d.split('_')[-1])
    r, c = components(d, base)
    rows.append((100 * np.mean(c), s, r, c))
print('| rank | seed | similarity (0-100) | respond | timing | ramp | recruit | opto | PNR |')
print('|---|---|---|---|---|---|---|---|---|')
for i, (idx, s, r, c) in enumerate(sorted(rows, key=lambda t: -t[0])):
    lab = '**13 (baseline)**' if s == 13 else str(s)
    print(f'| {i} | {lab} | **{idx:.0f}** | {r:.2f} | ' + ' | '.join(f'{x:.2f}' for x in c) + ' |')
