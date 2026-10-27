"""Composite figures for experiments/report/pnr_bistability_report.md (CPU, no simulation).

usage (repo root): python experiments/report/make_figs.py
Reads: experiments/pnr_determinants/seed<k>_<cond>/ (paired-pulse analysis), the GPU test_pnr_nm / dspn_pnr metrics of
each intact network, and the cortex-off stages (ctx_inh_*). Writes experiments/report/figs/*.png (150 dpi) and
experiments/report/figs/numbers.json (every number quoted in the report that is not read straight off a figure).
"""
import os, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(EXP)
FIG = os.path.join(EXP, 'report', 'figs')
os.makedirs(FIG, exist_ok=True)
DET = os.path.join(EXP, 'pnr_determinants')
NETS = [13, 9, 17, 21]
NAME = {13: 'baseline (seed 13)', 9: 'seed 9', 17: 'seed 17', 21: 'seed 21'}
COND = ['intact', 'snc_off', 'snc_sat', 'ctx_off']
CC = {'intact': 'k', 'snc_off': 'tab:blue', 'snc_sat': 'tab:red', 'ctx_off': 'tab:green'}
CL = {'intact': 'intact', 'snc_off': 'SNc off from cue', 'snc_sat': 'SNc saturated from cue', 'ctx_off': 'cortex off from cue+100 ms'}
START, NP = 300, 14
PT = np.arange(320, 600, 20); DL = (PT - START) / 100
NUM = {}


def netdir(k):
    return os.path.join(EXP, 'baseline') if k == 13 else os.path.join(EXP, 'seeds', f'seed_{k}', 'baseline')


def intact_pnr(k):
    return json.load(open(os.path.join(ROOT, 'seeds', '3k', f'seed_{k}', 'metrics_pnr.json')))


def load(k, c):
    d = np.load(os.path.join(DET, f'seed{k}_{c}', 'data.npz'), allow_pickle=True)
    m = json.load(open(os.path.join(DET, f'seed{k}_{c}', 'metrics.json')))
    return d, m


def col(d):
    return {n: i for i, n in enumerate(d['cols'])}


def tidx(d, name):
    return list(d['types']).index(name)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150); plt.close(fig)


def latency(d, name):
    """median pulse -> threshold latency (s) per delay, trials whose twin has not crossed before the pulse ends."""
    rt, it = d['rt'], tidx(d, name)
    out = []
    for ip, p in enumerate(PT):
        rp = rt[:, 1 + it * NP + ip]; v = (rt[:, 0] >= p + 5) & np.isfinite(rp)
        out.append(float(np.median(rp[v] - p) / 100) if v.sum() >= 10 else np.nan)
    return np.array(out)


def binned(x, y, bins, f=np.mean, nmin=8):
    b = np.digitize(x, bins) - 1
    ok = [i for i in range(len(bins) - 1) if (b == i).sum() >= nmin]
    return bins[ok] + (bins[1] - bins[0]) / 2, np.array([f(y[b == i]) for i in ok])


# ---------------- Fig 1: stop-pulse PNR, latency, all-or-none ----------------
fig, ax = plt.subplots(3, 4, figsize=(18, 12.5))
for j, k in enumerate(NETS):
    g = intact_pnr(k)
    a = ax[0, j]
    a.plot(np.array(g['pert_delay_ms']) / 1000, g['pert_prob'], 'ko-', ms=4, label='stop pulse (test_pnr_nm, GPU)')
    a.axhline(g['no_cue_prob'], color='gray', ls='--', label=f'no cue ({g["no_cue_prob"]:.2f})')
    dd, mm = load(k, 'intact')
    a.axhline(mm['twin_respond'], color='k', ls=':', label=f'cue only ({mm["twin_respond"]:.2f})')
    a.set(ylim=(-0.02, 1.02), xlabel='stop-pulse delay after cue (s)', ylabel='P(respond)', title=f'{NAME[k]}: stop-pulse PNR curve')
    a.legend(fontsize=7, loc='lower right')
    a = ax[1, j]
    for c in ['intact', 'snc_off', 'snc_sat']:
        d, m = load(k, c)
        a.plot(DL, latency(d, 'stop −5'), 'o-', ms=3, color=CC[c], label=CL[c])
    a.plot(DL, latency(dd, 'iSPN −5'), 's--', ms=3, color='tab:orange', label='iSPN −5 pulse (intact)')
    a.set(xlabel='pulse delay after cue (s)', ylabel='median latency pulse → threshold (s)', ylim=(0, 3), title='stop −5 pulse: time from pulse to response')
    a.legend(fontsize=7)
    NUM.setdefault('latency_stop5_intact', {})[k] = dict(zip([f'{x:.1f}' for x in DL], np.round(latency(dd, 'stop −5'), 2).tolist()))
    a = ax[2, j]
    pk, pk0 = dd['peak'], dd['peak_twin']
    hb = np.linspace(0.2, max(1.2, float(max(pk.max(), pk0.max())) + 0.05), 50)
    a.hist(pk, hb, color='gray', density=True, label='stop −5 pulse trials'); a.hist(pk0, hb, histtype='step', color='k', density=True, lw=1.5, label='cue only')
    a.axvspan(0.4, 0.5, color='tab:red', alpha=0.1)
    a.set(xlabel='peak output after pulse (to trial end)', ylabel='density',
          title=f'peak output; ambiguous (0.4–0.5):\n{mm["stop5_frac_peak_ambiguous_0p4_0p5"]:.2f} stop-pulse trials, {mm["twin_frac_peak_ambiguous_0p4_0p5"]:.2f} cue-only')
    a.legend(fontsize=7)
fig.suptitle('Fig 1. Stop pulse (−5 into all SPNs, 50 ms): response probability, response latency and peak-output distribution', fontsize=13)
fig.tight_layout(); save(fig, 'fig1_stop_pulse.png')

# ---------------- Fig 2: where is the PNR? outcome vs own time-to-movement ----------------
fig, ax = plt.subplots(3, 4, figsize=(18, 13))
cm = plt.cm.plasma(np.linspace(0, 0.9, NP))
TTB = np.arange(0, 3.01, 0.2)
NUM['pnr_ttr_stop5'] = {}
for j, k in enumerate(NETS):
    d, m = load(k, 'intact'); c = col(d); R = d['rows']
    for r, (name, key, ylab) in enumerate([('stop −5', 'shift', 'median RT shift vs twin (s)'), ('iSPN −5', 'shift', 'median RT shift vs twin (s)'),
                                          ('dSPN +1', 'surv', 'P(response survives pulse)')]):
        a = ax[r, j]; it = tidx(d, name)
        msk = (R[:, 0] == it) & (R[:, c['svalid']] == 1)
        for ip in range(NP):
            mk = msk & (R[:, 1] == ip)
            if key == 'shift':
                mk &= np.isfinite(R[:, c['shift']])
                x, y = binned(R[mk, c['ttr']], R[mk, c['shift']], TTB, np.median)
            else:
                x, y = binned(R[mk, c['ttr']], R[mk, c['surv']], TTB)
            if len(x) >= 2:
                a.plot(x, y, 'o-', ms=2.5, lw=1, color=cm[ip], label=f'{DL[ip]:.1f} s')
        if key == 'shift':
            a.plot([0, 3], [0, -3], color='gray', lw=0.6, ls='--', label='RT = pulse time')
            a.axhline(0, color='k', lw=0.5); a.set_ylim(-3, 1.5)
            # PNR: twin time-to-movement below which the pulse no longer advances RT by > 0.1 s (pooled over delays)
            mk = msk & np.isfinite(R[:, c['shift']])
            x, y = binned(R[mk, c['ttr']], R[mk, c['shift']], TTB, np.median, 15)
            pnr = next((float(xx) for xx, yy in zip(x, y) if yy < -0.1), None)
            NUM['pnr_ttr_stop5' if name == 'stop −5' else 'pnr_ttr_ispn5'] = {**NUM.get('pnr_ttr_stop5' if name == 'stop −5' else 'pnr_ttr_ispn5', {}), k: pnr}
            if pnr is not None:
                a.axvline(pnr - 0.1, color='tab:green', lw=1.2)
                a.text(pnr - 0.05, 1.2, f'no RT change below ≈ {pnr - 0.1:.1f} s', color='tab:green', fontsize=8)
        else:
            a.set_ylim(-0.02, 1.02); a.axhline(0.5, color='k', lw=0.5)
            # cancellation PNR: pooled over delays 1.0-2.4 s, smallest ttr bin where P(survive) < 0.9
            mk = msk & (R[:, 1] >= 4) & (R[:, 1] <= 11)
            x, y = binned(R[mk, c['ttr']], R[mk, c['surv']], TTB, np.mean, 15)
            cp = next((float(xx) for xx, yy in zip(x, y) if yy < 0.9), None)
            NUM.setdefault('cancel_pnr_ttr_dspn1', {})[k] = None if cp is None else round(cp - 0.1, 1)
            NUM.setdefault('cancel_curve_dspn1', {})[k] = [(round(float(xx), 1), round(float(yy), 2)) for xx, yy in zip(x, y)]
            if cp is not None:
                a.axvline(cp - 0.1, color='tab:green', lw=1.2)
                a.text(cp - 0.05, 0.05, f'cannot cancel below ≈ {cp - 0.1:.1f} s', color='tab:green', fontsize=8)
        a.set(xlabel="twin's own time to movement at pulse onset (s)", ylabel=ylab, title=f'{NAME[k]}: {name}')
        if j == 0:
            a.legend(fontsize=6, ncol=2, title='pulse delay', title_fontsize=7)
fig.suptitle('Fig 2. Pulse outcome against each trial\'s own time to movement (twin RT − pulse onset); one line per pulse delay after the cue.\n'
             'Lines that collapse onto each other = the outcome is set by the trial\'s progress, not by the clock.', fontsize=12)
fig.tight_layout(); save(fig, 'fig2_pnr_vs_time_to_movement.png')

# ---------------- Fig 3: which population, which amplitude ----------------
fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
for j, k in enumerate(NETS):
    d, m = load(k, 'intact'); ty = m['types']
    a = ax[0, j]
    for n, cc in [('iSPN −5', 'tab:orange'), ('iSPN −1', 'navajowhite'), ('stop −5', 'k'), ('dSPN −5', 'tab:blue'), ('dSPN +1', 'tab:green'), ('dSPN +0.25', 'yellowgreen')]:
        a.plot(DL, ty[n]['p_triggered'], 'o-', ms=3, color=cc, label=n)
    a.set(ylim=(-0.02, 1.02), xlabel='pulse delay after cue (s)', ylabel='P(cross 0.5 within 500 ms)', title=f'{NAME[k]}: triggering by pulse type')
    a.legend(fontsize=7)
    a = ax[1, j]
    stops = [n for n in ty if n.startswith('stop')]
    amps = [abs(ty[n]['amp']) for n in stops]
    for ip, cc in zip([6, 9, 11, 13], plt.cm.viridis([0.1, 0.4, 0.7, 0.95])):
        a.plot(amps, [ty[n]['p_triggered'][ip] for n in stops], 'o-', color=cc, label=f'{DL[ip]:.1f} s')
    a.set(xscale='log', ylim=(-0.02, 1.02), xlabel='stop-pulse amplitude |a| (all SPNs, 50 ms)', ylabel='P(triggered within 500 ms)', title='amplitude dependence')
    a.legend(fontsize=7, title='delay', title_fontsize=7)
    NUM.setdefault('trig_by_type_at_2p6', {})[k] = {n: round(ty[n]['p_triggered'][12], 2) for n in ty}
fig.suptitle('Fig 3. Commitment is triggered through iSPNs: iSPN inhibition triggers, dSPN inhibition never does; stronger stop pulses are not better', fontsize=12)
fig.tight_layout(); save(fig, 'fig3_pulse_type_amplitude.png')

# ---------------- Fig 4: dopamine sets how easily the switch flips ----------------
fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
for j, k in enumerate(NETS):
    for c in ['intact', 'snc_off', 'snc_sat']:
        d, m = load(k, c); ty = m['types']
        lab = f'{CL[c]} (cue-only resp {m["twin_respond"]:.2f})'
        ax[0, j].plot(DL, ty['stop −5']['p_respond'], 'o-', ms=3, color=CC[c], label=lab)
        ax[0, j].plot(DL, ty['iSPN −1']['p_respond'], 's:', ms=2.5, color=CC[c])
        sv = [np.nan if v is None or n < 10 else v for v, n in zip(ty['dSPN +1']['p_survive'], ty['dSPN +1']['n_survive_valid'])]
        ax[1, j].plot(DL, sv, 'o-', ms=3, color=CC[c], label=CL[c])
        NUM.setdefault('dspn1_min_survive', {}).setdefault(k, {})[c] = None if np.all(np.isnan(sv)) else round(float(np.nanmin(sv)), 2)
    ax[0, j].set(ylim=(-0.02, 1.02), xlabel='pulse delay after cue (s)', ylabel='P(respond)', title=f'{NAME[k]}: stop −5 (solid), iSPN −1 (dotted)')
    ax[0, j].legend(fontsize=6.5)
    ax[1, j].set(ylim=(-0.02, 1.02), xlabel='pulse delay after cue (s)', ylabel='P(response survives)', title='dSPN +1 pulse: can it cancel a response?')
    ax[1, j].legend(fontsize=6.5)
fig.suptitle('Fig 4. Dopamine (SNc clamped from the cue) changes how easily the switch flips: response probability after a stop pulse, and cancellation by a dSPN +1 pulse', fontsize=12)
fig.tight_layout(); save(fig, 'fig4_dopamine.png')

# ---------------- Fig 5: critical slowing (GPU dspn_pnr metrics) ----------------
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
for k, cc in zip(NETS, ['k', 'tab:purple', 'tab:olive', 'tab:cyan']):
    g = json.load(open(os.path.join(netdir(k), 'dspn_pnr', 'metrics_dspn_pnr.json')))
    P = g['pulses']; dl = [p['pulse_delay_ms'] / 1000 for p in P]
    ax[0].plot(dl, [p['respond'] for p in P], 'o-', ms=3, color=cc, label=f'{NAME[k]} (cue only {g["control_respond"]:.2f})')
    ax[1].plot(dl, [p['frac_censored'] for p in P], 'o-', ms=3, color=cc, label=NAME[k])
    ax[2].plot(dl, [p['residual_500ms_median'] for p in P], 'o-', ms=3, color=cc, label=NAME[k])
    ax[3].plot(dl, [p['relax_tau_ms_median'] for p in P], 'o-', ms=3, color=cc, label=NAME[k])
    NUM.setdefault('critical_slowing', {})[k] = dict(peak_censored_delay=dl[int(np.argmax([p['frac_censored'] for p in P]))],
                                                     peak_censored=max(p['frac_censored'] for p in P))
ax[0].set(ylabel='P(respond)', title='dSPN +1 pulse: response probability'); ax[0].legend(fontsize=7)
ax[1].set(ylabel='fraction never relaxing to 1/e', title='perturbation fails to decay (censored)')
ax[2].set(ylabel='|Δ rates| at 500 ms / peak', title='residual 500 ms after pulse'); ax[2].axhline(1, color='gray', ls=':')
ax[3].set(ylabel='ms', title='relaxation time (decaying trials only)', yscale='log')
for a in ax:
    a.set_xlabel('pulse delay after cue (s)')
fig.suptitle('Fig 5. Critical slowing: a +1 dSPN pulse decays quickly early in the delay and stops decaying (or grows) late in the delay', fontsize=12)
fig.tight_layout(); save(fig, 'fig5_critical_slowing.png')

# ---------------- Fig 6: determinants at a fixed delay ----------------
FEATS = ['y', 'D1', 'D2', 'D1−D2', 'cortex', 'thalamus', 'SNc', 's', 'e_d1', 'e_d2', 'progress', 'velocity']


def auc(x, yb):
    yb = yb.astype(bool); n1, n0 = yb.sum(), (~yb).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(x)
    return float((rk[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_cv(X, yb, k=5, lam=1e-2):
    if yb.sum() < 10 or (~yb).sum() < 10:
        return np.nan
    fold = np.random.default_rng(0).permutation(len(yb)) % k; pred = np.zeros(len(yb))
    for f in range(k):
        Xa, ya = X[fold != f], yb[fold != f]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Z = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]; w = np.zeros(Z.shape[1])
        for _ in range(50):
            pz = 1 / (1 + np.exp(-Z @ w)); W = pz * (1 - pz) + 1e-6
            H = Z.T @ (Z * W[:, None]) + lam * len(ya) * np.eye(len(w)); H[0, 0] -= lam * len(ya)
            w = w + np.linalg.solve(H, Z.T @ (ya - pz) - lam * len(ya) * np.r_[0, w[1:]])
        pred[fold == f] = np.c_[np.ones((fold == f).sum()), (X[fold == f] - mu) / sd] @ w
    return auc(pred, yb)


fig, ax = plt.subplots(2, 2, figsize=(17, 10)); ax = ax.ravel()
NUM['cv_auc_onehot'] = {}
for r, (name, outc, vcol) in enumerate([('stop −5', 'trig', 'valid'), ('dSPN +1', 'surv', 'svalid')]):
    w = 0.2
    for j, k in enumerate(NETS):
        d, m = load(k, 'intact'); c = col(d); R = d['rows']
        msk = (R[:, 0] == tidx(d, name)) & (R[:, c[vcol]] == 1)
        yb = R[msk, c[outc]].astype(bool); ip = R[msk, 1].astype(int); X = R[msk, c['y']:]
        ttr = R[msk, c['ttr']]
        vals = []
        for f in FEATS + ['ttr']:
            v = X[:, FEATS.index(f)] if f != 'ttr' else -ttr
            a_, wt = [], []
            for q in range(NP):
                mk = ip == q
                if yb[mk].sum() >= 10 and (~yb[mk]).sum() >= 10 and np.isfinite(v[mk]).all():
                    a_.append(auc(v[mk], yb[mk])); wt.append(mk.sum())
            vals.append(max(np.average(a_, weights=wt), 1 - np.average(a_, weights=wt)) if a_ else np.nan)
        ax[2 * r].bar(np.arange(len(vals)) + (j - 1.5) * w, vals, w, label=NAME[k])
        oh = (ip[:, None] == np.arange(NP)[None]).astype(float)
        cv = {'delay (categorical)': logit_cv(oh, yb), 'state': logit_cv(X, yb), 'delay (cat.) + state': logit_cv(np.c_[oh, X], yb)}
        if outc == 'surv':
            cv['delay (cat.) + ttr'] = logit_cv(np.c_[oh, ttr], yb)
        NUM['cv_auc_onehot'].setdefault(name, {})[k] = {kk: (None if not np.isfinite(v) else round(v, 3)) for kk, v in cv.items()}
        NUM.setdefault('within_delay_auc', {}).setdefault(name, {})[k] = dict(zip(FEATS + ['ttr'], [None if not np.isfinite(v) else round(float(v), 3) for v in vals]))
        ax[2 * r + 1].plot(range(len(cv)), list(cv.values()), 'o-', label=NAME[k])
        ax[2 * r + 1].set_xticks(range(len(cv)), list(cv.keys()), fontsize=8)
    ax[2 * r].set_xticks(range(len(FEATS) + 1), FEATS + ['ttr'], rotation=45, fontsize=8); ax[2 * r].axhline(0.5, color='k', lw=0.5)
    ax[2 * r].set(ylim=(0.45, 1.0), ylabel='AUC at fixed delay, max(A, 1−A)',
                  title=f'{name}: {"triggered within 500 ms" if outc == "trig" else "response survives"} — single variables, clock held fixed')
    ax[2 * r].legend(fontsize=7)
    ax[2 * r + 1].set(ylim=(0.45, 1.0), ylabel='5-fold CV AUC', title=f'{name}: logistic models (delay as a category)'); ax[2 * r + 1].legend(fontsize=7)
fig.suptitle('Fig 6. At a fixed time since the cue, does the circuit state before the pulse predict the outcome?', fontsize=12)
fig.tight_layout(); save(fig, 'fig6_determinants.png')

# ---------------- Fig 7: cortex off from cue + 100 ms ----------------
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
tt = (np.arange(900) - START) / 100
for j, k in enumerate(NETS):
    a = ax[0, j]
    for c in ['intact', 'ctx_off']:
        d, m = load(k, c)
        a.plot(tt, d['y_twin'].mean(0), color=CC[c], label=f'{CL[c]}: cue only (resp {m["twin_respond"]:.2f})')
        y5 = d['y_stop5']
        for ip, ls in [(9, '--'), (13, ':')]:
            a.plot(tt, y5[:, ip].mean(0), color=CC[c], ls=ls, lw=1, label=f'{CL[c]}: stop −5 at {DL[ip]:.1f} s')
    a.axhline(0.5, color='gray', lw=0.5); a.axvline(0, color='gray', lw=0.5)
    a.set(xlabel='time from cue (s)', ylabel='mean output y', title=f'{NAME[k]}: output', xlim=(-1, 6)); a.legend(fontsize=6)
    a = ax[1, j]
    d, m = load(k, 'ctx_off'); ty = m['types']
    for n, cc in [('iSPN −5', 'tab:orange'), ('stop −5', 'k'), ('dSPN +1', 'tab:green'), ('stop −0.1', 'gray')]:
        a.plot(DL, ty[n]['p_respond'], 'o-', ms=3, color=cc, label=n)
    g = os.path.join(netdir(k), 'ctx_inh_pnr', 'metrics_pnr.json')
    if os.path.exists(g):
        g = json.load(open(g)); a.plot(np.array(g['pert_delay_ms']) / 1000, g['pert_prob'], 'k^', ms=5, mfc='none', label='stop −5 (test_pnr_nm, GPU)')
        NUM.setdefault('ctx_off_gpu_pnr', {})[k] = dict(max_pert_prob=max(g['pert_prob']), no_cue=g['no_cue_prob'], cue_mag_max=max(g['cue_mag_prob']))
    a.set(ylim=(-0.02, 1.02), xlabel='pulse delay after cue (s)', ylabel='P(respond)', title='cortex off: any pulse still triggers?'); a.legend(fontsize=7)
fig.suptitle('Fig 7. Cortex silenced from 100 ms after cue onset to trial end: no responses, and no striatal pulse can trigger one', fontsize=12)
fig.tight_layout(); save(fig, 'fig7_cortex_off.png')

# ---------------- copies of single-network plots (downscaled to <= 1800 px wide) ----------------
def copy(src, dst, width=1800):
    if not os.path.exists(src):
        print('missing', src); return
    im = Image.open(src)
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(FIG, dst), optimize=True)


copy(os.path.join(EXP, 'baseline', 'snc_critical_slowing', 'snc_critical_slowing.png'), 'baseline_snc_critical_slowing.png')
copy(os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'plots', 'pnr_dyn_outputs.png'), 'baseline_pnr_dyn_outputs.png')
copy(os.path.join(EXP, 'baseline', 'pnr', 'rt_stop_pulse.png'), 'baseline_rt_stop_pulse.png')
copy(os.path.join(EXP, 'baseline', 'dspn_pnr', 'rt_dspn_pulse.png'), 'baseline_rt_dspn_pulse.png')
for k in NETS:
    copy(os.path.join(DET, f'seed{k}_intact', 'pnr_determinants.png'), f'determinants_seed{k}.png')
    nd = netdir(k)
    for sub, f in [('ctx_inh_test', 'output_activity.png'), ('ctx_inh_test', 'activity_by_area.png'), ('ctx_inh_test', 'cue_aligned_activity.png'),
                   ('ctx_inh_pnr', 'pnr_dyn_outputs.png'), ('ctx_inh_pnr', 'dynamic_pnr.png'), ('ctx_inh_pnr', 'pnr_dyn_areas.png'),
                   ('ctx_inh_pnr', 'rt_stop_pulse.png'), ('ctx_inh_dspn_pnr', 'rt_dspn_pulse.png'), ('ctx_inh_dspn_pnr', 'dspn_pnr.png')]:
        src = next((q for q in [os.path.join(nd, sub, 'plots', 'png', f), os.path.join(nd, sub, 'plots', 'pnr_test', f),
                                os.path.join(nd, sub, f)] if os.path.exists(q)), os.path.join(nd, sub, f))
        copy(src, f'ctx_off_seed{k}_{f}')
json.dump(NUM, open(os.path.join(FIG, 'numbers.json'), 'w'), indent=1, default=float)
print(json.dumps(NUM, indent=1, default=float))
