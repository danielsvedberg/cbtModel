"""Score each seed against the baseline's target properties; prints a markdown table.

usage (from repo root): python seeds/scorecard.py [subfolder, default 3k]
Criteria (thresholds chosen so the baseline passes each):
  respond  : responds on >= 50% of test trials
  ramp     : ramp onset tracks RT (slope >= 0.5 and r >= 0.6) in >= 3 of D1, D2, cortex, thalamus
  recruit  : >= 40% of units modulated in every area (D1, D2, cortex, thalamus, SNc)
  opto     : inh iSPN -> earlier with >= 90% responding, and inh dSPN -> later
  pnr      : stop pulse >= 2.0 s after cue triggers >= 95% responses, and <= 25% ambiguous peaks
"""
import os, sys, json, glob, csv
SUB = sys.argv[1] if len(sys.argv) > 1 else '3k'
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUB)
ld = lambda d, s: json.load(open(os.path.join(d, f'metrics_{s}.json')))
init = {}
p = os.path.join(HERE, 'init_features.csv')
if os.path.exists(p):
    init = {int(r['seed']): float(r['active_bg']) == 0 for r in csv.DictReader(open(p))}
rows = []
for d in sorted(filter(os.path.isdir, glob.glob(os.path.join(HERE, 'seed_*'))), key=lambda d: int(d.split('_')[-1])):
    s = int(d.split('_')[-1])
    te, op, pn = ld(d, 'test'), ld(d, 'opto'), ld(d, 'pnr')
    ramp = sum(1 for a in ['D1', 'D2', 'Cortex', 'Thalamus']
               if (te[a]['ramp_onset_vs_rt_slope'] or 0) >= 0.5 and (te[a]['ramp_onset_vs_rt_r'] or 0) >= 0.6)
    rec = min(te[a]['frac_units_modulated'] for a in ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc'])
    strongest = lambda g: max(op[g], key=lambda x: abs(x['strength']))
    c, di, ii = op['inh dMSN'][0]['rt_median'], strongest('inh dMSN'), strongest('inh iMSN')
    opto = c is not None and ii['rt_median'] is not None and ii['rt_median'] < c and ii['respond'] >= 0.9 and \
        (di['rt_median'] is None or di['rt_median'] > c)
    late = [pr for dl, pr in zip(pn['pert_delay_ms'], pn['pert_prob']) if dl >= 2000]
    pnr = min(late) >= 0.95 and pn['peak_frac_ambiguous_0p4_0p5'] <= 0.25
    crit = [te['respond'] >= 0.5, ramp >= 3, rec >= 0.4, opto, pnr]
    rows.append((sum(crit), s, te['respond'], ramp, rec, opto, pnr, pn['no_cue_prob'], crit, init.get(s)))
ok = lambda b: '✓' if b else '·'
print('| seed | silent at init | respond | ramp areas (of 4) | min units recruited | opto | PNR | no-cue moves | score (of 5) |')
print('|---|---|---|---|---|---|---|---|---|')
for sc, s, r, ramp, rec, opto, pnr, nc, crit, sil in sorted(rows, key=lambda t: (-t[0], -t[2])):
    lab = '**13 (baseline)**' if s == 13 else str(s)
    print(f"| {lab} | {'yes' if sil else 'no'} | {r:.2f} {ok(crit[0])} | {ramp} {ok(crit[1])} | {rec:.2f} {ok(crit[2])} | "
          f"{ok(opto)} | {ok(pnr)} | {nc:.2f} | {sc} |")
