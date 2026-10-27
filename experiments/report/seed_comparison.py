"""Cross-network comparison for the PNR report: the baseline and every 3k seed with the full test series (CPU, no simulation).

usage (repo root): python experiments/report/seed_comparison.py
Reads the same outputs as make_figs.py (paired-pulse analysis in experiments/pnr_determinants/, GPU stages under
experiments/[seeds/seed_<k>/]...) and seeds/3k/ (intact test/opto/PNR, similarity index via seeds/similarity.py).
Writes experiments/report/figs/seed_comparison.png and seed_comparison.json, and prints the
markdown tables used in section 9 of the report. Definitions follow make_figs.py; see the report for details.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(EXP)
FIG = os.path.join(EXP, 'report', 'figs')
DET = os.path.join(EXP, 'pnr_determinants')
START, NP = 300, 14
PT = np.arange(320, 600, 20); DL = (PT - START) / 100
TTB = np.arange(0, 3.01, 0.2)
NETS = sorted(int(n[4:].split('_')[0]) for n in os.listdir(DET) if n.endswith('_intact') and n.startswith('seed'))

# similarity index (seeds/similarity.py)
sys.argv = ['similarity.py', '3k']
_src = open(os.path.join(ROOT, 'seeds', 'similarity.py')).read().split("print('| rank")[0]
_ns = {'__file__': os.path.join(ROOT, 'seeds', 'similarity.py')}
exec(_src, _ns)
SIM = {s: (idx, c[3]) for idx, s, r, c in _ns['rows']}
NETS = sorted(NETS, key=lambda k: -SIM[k][0])


def lesion_root(k):
    return EXP if k == 13 else os.path.join(EXP, 'seeds', f'seed_{k}')


def jl(*p):
    p = os.path.join(*p)
    return json.load(open(p)) if os.path.exists(p) else None


def load(k, c):
    d = np.load(os.path.join(DET, f'seed{k}_{c}', 'data.npz'), allow_pickle=True)
    return d, jl(DET, f'seed{k}_{c}', 'metrics.json')


def binned(x, y, f=np.mean, nmin=8):
    b = np.digitize(x, TTB) - 1
    ok = [i for i in range(len(TTB) - 1) if (b == i).sum() >= nmin]
    return TTB[ok] + 0.1, np.array([f(y[b == i]) for i in ok])


def edge(d, ip):
    """dSPN +1 cancellation edge at one pulse delay: lower edge of the first ttr bin with P(survive) < 0.9, if earlier bins are >= 0.9."""
    c = {n: i for i, n in enumerate(d['cols'])}; R = d['rows']
    mk = (R[:, 0] == list(d['types']).index('dSPN +1')) & (R[:, c['svalid']] == 1) & (R[:, 1] == ip)
    x, y = binned(R[mk, c['ttr']], R[mk, c['surv']])
    i = next((i for i, v in enumerate(y) if v < 0.9), None)
    return None if i is None or i == 0 else round(float(x[i] - 0.1), 1)


def latency(d, name):
    rt, it = d['rt'], list(d['types']).index(name)
    out = []
    for ip, p in enumerate(PT):
        rp = rt[:, 1 + it * NP + ip]; v = (rt[:, 0] >= p + 5) & np.isfinite(rp)
        out.append(float(np.median(rp[v] - p) / 100) if v.sum() >= 10 else np.nan)
    return np.array(out)


def min_amp(ty, ip):
    a = [abs(ty[n]['amp']) for n in ty if n.startswith('stop') and ty[n]['p_triggered'][ip] is not None and ty[n]['p_triggered'][ip] >= 0.5]
    return min(a) if a else None


def surv_curve(ty):
    return np.array([np.nan if v is None or n < 10 else v for v, n in zip(ty['dSPN +1']['p_survive'], ty['dSPN +1']['n_survive_valid'])])


R = {}
for k in NETS:
    r = R[k] = {'similarity': SIM[k][0], 'opto_score': SIM[k][1]}
    te, gp = jl(ROOT, 'seeds', '3k', f'seed_{k}', 'metrics_test.json'), jl(ROOT, 'seeds', '3k', f'seed_{k}', 'metrics_pnr.json')
    dsp = jl(lesion_root(k), 'baseline', 'dspn_pnr', 'metrics_dspn_pnr.json')
    d, m = load(k, 'intact'); ty = m['types']
    r['respond'] = te['respond']; r['rt_median'] = te['rt_median']; r['no_cue'] = gp['no_cue_prob']
    r['amb_cue_only'] = m['twin_frac_peak_ambiguous_0p4_0p5']
    late = [p for dl, p in zip(gp['pert_delay_ms'], gp['pert_prob']) if dl >= 2000]
    r['stop_late_min'] = min(late)
    r['stop_curve'] = gp['pert_prob']
    # critical slowing (GPU dSPN +1, censored fraction) and cancellation trough (paired, CPU)
    # both searched inside the report's mid-delay window (0.8-2.4 s); 'none' if nothing fails to decay / nothing cancels
    cens = [p['frac_censored'] for p in dsp['pulses']]
    r['cens_curve'] = cens
    W = range(3, 12)
    i = max(W, key=lambda i: cens[i])
    r['cs_peak_delay'], r['cs_peak'] = (float(DL[i]), cens[i]) if cens[i] >= 0.1 else (None, cens[i])
    r['early_cens_max'] = max(cens[:3])
    sv = surv_curve(ty); r['surv_curve'] = sv.tolist()
    sw = np.array([sv[i] if i in W else np.nan for i in range(NP)])
    if np.all(np.isnan(sw)):
        r['trough_delay'] = r['trough'] = None
    else:
        i = int(np.nanargmin(sw)); r['trough'] = float(sw[i]); r['trough_delay'] = float(DL[i]) if sw[i] < 0.9 else None
    r['early_surv_min'] = None if np.all(np.isnan(sv[:3])) else float(np.nanmin(sv[:3]))
    r['edges'] = {f'{DL[ip]:.1f}': edge(d, ip) for ip in range(NP)}
    r['late_edges'] = [r['edges'][f'{x:.1f}'] for x in (2.0, 2.2, 2.4)]
    # latency regime (stop −5)
    lat = latency(d, 'stop −5'); r['lat_curve'] = lat.tolist()
    r['lat_1p0'], r['lat_2p0'] = lat[4], lat[9]
    trig = ty['stop −5']['p_triggered']
    r['stop5_trig'] = trig
    r['fast_trig_delays'] = [float(DL[i]) for i in range(NP) if trig[i] is not None and trig[i] >= 0.5]
    # pathway, amplitude
    r['ispn5_trig_max'] = max(v for v in ty['iSPN −5']['p_triggered'] if v is not None)
    r['ispn5_curve'] = ty['iSPN −5']['p_triggered']
    r['dspn_inh_trig_max'] = max(v for v in ty['dSPN −5']['p_triggered'] if v is not None)
    r['amp50_1p0'], r['amp50_2p6'] = min_amp(ty, 4), min_amp(ty, 12)
    # dopamine and cortex (paired)
    for c in ['snc_off', 'snc_sat', 'ctx_off']:
        dc, mc = load(k, c); tc = mc['types']
        r[f'{c}_respond'] = mc['twin_respond']
        r[f'{c}_min_surv'] = None if np.all(np.isnan(surv_curve(tc))) else float(np.nanmin(surv_curve(tc)))
        r[f'{c}_ispn5_max'] = max(v for v in tc['iSPN −5']['p_triggered'] if v is not None) if any(v is not None for v in tc['iSPN −5']['p_triggered']) else None
        r[f'{c}_max_any'] = max(max(v for v in tc[n]['p_respond'] if v is not None) for n in tc)
    # cortex off on the GPU
    ct, cp, cd = (jl(lesion_root(k), 'baseline', f'ctx_inh_{s}', f'metrics_{s}.json') for s in ('test', 'pnr', 'dspn_pnr'))
    r['ctx_gpu_respond'] = ct['respond']; r['ctx_gpu_pnr_max'] = max(cp['pert_prob'] + cp['cue_mag_prob'])
    r['ctx_gpu_censored_max'] = max([p['frac_censored'] for p in cd['pulses'] if p['n_tau'] >= 10] or [None], key=lambda v: -1 if v is None else v)
    # SNc clamps (GPU, 1000 trials) and lesions
    si, ss = jl(lesion_root(k), 'baseline', 'snc_inhibition', 'metrics_snc_inhibition.json'), jl(lesion_root(k), 'baseline', 'snc_stimulation', 'metrics_snc_stimulation.json')
    r['snc_inh_respond'], r['snc_stim_respond'], r['snc_stim_rt'] = si['clamped']['respond'], ss['clamped']['respond'], ss['clamped']['rt_median_s']
    r['snc_ds_after_cue'] = si['control']['s_after_cue'] - si['control']['s_precue']
    L = lesion_root(k)
    r['pd_acute'] = jl(L, 'pd_snc_lesion', 'acute', 'test', 'metrics_test.json')['respond']
    r['pd_acute_stim'] = jl(L, 'pd_snc_lesion', 'acute', 'snc_stimulation', 'metrics_snc_stimulation.json')['clamped']['respond']
    r['pd_retrained'] = jl(L, 'pd_snc_lesion', 'retrained', 'test', 'metrics_test.json')['respond']
    for h in ('acute', 'retrained'):
        t = jl(L, 'hd_ispn_lesion', h, 'test', 'metrics_test.json')
        r[f'hd_{h}'], r[f'hd_{h}_ontime'], r[f'hd_{h}_rt'] = t['respond'], t['on_time_2p5_3p5'], t['rt_median']

json.dump(R, open(os.path.join(FIG, 'seed_comparison.json'), 'w'), indent=1, default=float)


# ---------------- tables ----------------
def f(v, fmt='{:.2f}'):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def rng(v):
    if not v:
        return 'never'
    runs, cur = [], [v[0]]
    for x in v[1:]:
        if abs(x - cur[-1] - 0.2) < 1e-6:
            cur.append(x)
        else:
            runs.append(cur); cur = [x]
    runs.append(cur)
    return ', '.join(f'{a[0]:.1f}' if len(a) == 1 else f'{a[0]:.1f}–{a[-1]:.1f}' for a in runs) + ' s'


name = lambda k: '**13 (baseline)**' if k == 13 else str(k)
out = []
out += ['| network | similarity | opto score | respond | RT median (s) | no-cue moves | cue-only ambiguous peaks | late stop pulse P(respond) | stop −5 median latency at 1.0 / 2.0 s | stop −5 triggers ≥ 50% within 500 ms at |',
        '|---|---|---|---|---|---|---|---|---|---|']
for k in NETS:
    r = R[k]
    out.append(f"| {name(k)} | {r['similarity']:.0f} | {r['opto_score']:.2f} | {r['respond']:.2f} | {f(r['rt_median'])} | {r['no_cue']:.2f} | "
               f"{r['amb_cue_only']:.2f} | {r['stop_late_min']:.2f} | {f(r['lat_1p0'])} / {f(r['lat_2p0'])} | {rng(r['fast_trig_delays'])} |")
out += ['', '| network | critical-slowing peak, 0.8–2.4 s (censored) | dSPN +1 cancellation trough, 0.8–2.4 s (P(survive)) | early (0.2–0.6 s): max censored / min P(survive) | late edge at 2.0 / 2.2 / 2.4 s | iSPN −5 max P(trig) | dSPN −5 max P(trig) | min stop |a| for 50% trig, 1.0 → 2.6 s |',
        '|---|---|---|---|---|---|---|---|']
for k in NETS:
    r = R[k]
    out.append(f"| {name(k)} | {f(r['cs_peak_delay'], '{:.1f} s') if r['cs_peak_delay'] else 'none'} ({r['cs_peak']:.2f}) | "
               f"{f(r['trough_delay'], '{:.1f} s') if r['trough_delay'] else 'none'} ({f(r['trough'])}) | {r['early_cens_max']:.2f} / {f(r['early_surv_min'])} | "
               + ' / '.join(f(e, '{:.1f}') for e in r['late_edges']) +
               f" | {r['ispn5_trig_max']:.2f} | {r['dspn_inh_trig_max']:.2f} | {f(r['amp50_1p0'], '{:g}')} → {f(r['amp50_2p6'], '{:g}')} |")
out += ['', '| network | SNc off: cue-only respond | SNc off: iSPN −5 max P(trig) | min P(survive) dSPN +1: intact / SNc off / SNc sat | SNc sat: respond | cortex off: max P(respond), any test | cortex off: max censored | SNc Δs after cue |',
        '|---|---|---|---|---|---|---|---|']
for k in NETS:
    r = R[k]
    out.append(f"| {name(k)} | {r['snc_off_respond']:.2f} | {f(r['snc_off_ispn5_max'])} | {f(r['trough'])} / {f(r['snc_off_min_surv'])} / {f(r['snc_sat_min_surv'])} | "
               f"{r['snc_sat_respond']:.2f} | {max(r['ctx_off_max_any'], r['ctx_gpu_respond'], r['ctx_gpu_pnr_max']):.3f} | {f(r['ctx_gpu_censored_max'])} | {r['snc_ds_after_cue']:+.3f} |")
out += ['', '| network | intact | SNc inhibited | SNc stimulated (RT) | PD acute | PD acute + SNc stim | PD retrained | HD acute (on time) | HD retrained (on time) |',
        '|---|---|---|---|---|---|---|---|---|']
for k in NETS:
    r = R[k]
    out.append(f"| {name(k)} | {r['respond']:.2f} | {r['snc_inh_respond']:.2f} | {r['snc_stim_respond']:.2f} ({f(r['snc_stim_rt'])} s) | {r['pd_acute']:.2f} | "
               f"{r['pd_acute_stim']:.2f} | {r['pd_retrained']:.2f} | {r['hd_acute']:.2f} ({r['hd_acute_ontime']:.2f}) | {r['hd_retrained']:.2f} ({r['hd_retrained_ontime']:.2f}) |")
print('\n'.join(out))

# ---------------- figures ----------------
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(NETS) - 1))
COL = {13: 'k', **{k: cmap[i] for i, k in enumerate([k for k in NETS if k != 13])}}
lab = lambda k: f"{'baseline' if k == 13 else f'seed {k}'} ({R[k]['similarity']:.0f})"
fig, ax = plt.subplots(2, 3, figsize=(19, 11)); ax = ax.ravel()
for k in NETS:
    r = R[k]; kw = dict(color=COL[k], lw=2.5 if k == 13 else 1.2, ms=3, label=lab(k), zorder=3 if k == 13 else 2)
    ax[0].plot(DL, r['stop_curve'], 'o-', **kw)
    ax[1].plot(DL, r['lat_curve'], 'o-', **kw)
    ax[2].plot(DL, [np.nan if v is None else v for v in r['ispn5_curve']], 'o-', **kw)
    ax[3].plot(DL, r['cens_curve'], 'o-', **kw)
    ax[4].plot(DL, r['surv_curve'], 'o-', **kw)
ax[0].set(ylim=(-0.02, 1.02), ylabel='P(respond)', title='stop pulse (−5, all SPNs; GPU test_pnr_nm)')
ax[1].set(ylim=(0, 3), ylabel='median latency pulse → threshold (s)', title='stop −5: time from pulse to response')
ax[2].set(ylim=(-0.02, 1.02), ylabel='P(cross 0.5 within 500 ms)', title='iSPN −5 pulse: triggering')
ax[3].set(ylim=(-0.02, 1.02), ylabel='fraction never relaxing', title='critical slowing (dSPN +1, censored fraction; GPU)')
ax[4].set(ylim=(-0.02, 1.02), ylabel='P(response survives)', title='dSPN +1: cancellation')
for a in ax[:5]:
    a.set_xlabel('pulse delay after cue (s)')
ax[0].legend(fontsize=7, title='network (similarity)', title_fontsize=7)
a = ax[5]
for k in NETS:
    r = R[k]
    if r['trough_delay'] is not None and r['cs_peak_delay'] is not None:
        a.scatter(r['cs_peak_delay'], r['trough_delay'], s=40 + 160 * (1 - r['trough']), color=COL[k], edgecolor='k', zorder=3)
        a.annotate('base' if k == 13 else str(k), (r['cs_peak_delay'], r['trough_delay']), xytext=(4, 4), textcoords='offset points', fontsize=8)
a.plot([0, 3], [0, 3], color='gray', ls='--', lw=0.8)
a.set(xlim=(0, 3), ylim=(0, 3), xlabel='critical-slowing peak (s after cue)', ylabel='dSPN +1 cancellation trough (s after cue)',
      title='cancellation trough vs critical-slowing peak, 0.8–2.4 s (size = depth)')
fig.suptitle('Fig 8. All networks with the full test series, ordered by similarity to the baseline', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'seed_comparison.png'), dpi=94); plt.close(fig)

print()
print('| pulse delay | ' + ' | '.join(f'{x:.1f} s' for x in DL[3:12]) + ' |')
print('|---|' + '---|' * 9)
for k in NETS:
    print(f'| {name(k)} | ' + ' | '.join(f(R[k]['edges'][f'{x:.1f}'], '{:.1f}') for x in DL[3:12]) + ' |')
