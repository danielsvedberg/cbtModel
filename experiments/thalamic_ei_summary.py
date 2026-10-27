"""Cross-network summary of the thalamic E/I test (CPU; reads experiments/thalamic_ei/seed<k>/ and the report's seed_comparison.json).

usage (repo root): python experiments/thalamic_ei_summary.py
Writes experiments/thalamic_ei/summary.png, summary.json and prints markdown tables.

Single-threshold test: if the readout-weighted net drive E - I is the variable that commits the network, all pulse types
should trigger at the same value of it. For each pulse type with >= 10 triggered and >= 10 non-triggered valid trials,
a 1-D logistic fit of 'triggered' on the variable at +100 ms from pulse onset gives the 50% point; the spread (SD) of
those points across types, divided by the SD of the variable over all valid pulse trials, is the threshold dispersion
(0 = one threshold for all types). Compared against the same for output y and mean thalamic rate.
"""
import os, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

EXP = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(EXP)
D = os.path.join(EXP, 'thalamic_ei')
CMP = {int(k): v for k, v in json.load(open(os.path.join(EXP, 'report', 'figs', 'seed_comparison.json'))).items()}
NETS = [k for k in sorted(CMP, key=lambda k: -CMP[k]['similarity']) if os.path.exists(os.path.join(D, f'seed{k}', 'data.npz'))]
NP, nT = 14, 13
TN = ['stop −0.1', 'stop −0.25', 'stop −0.5', 'stop −1', 'stop −2.5', 'stop −5', 'stop −10', 'dSPN −5', 'iSPN −1', 'iSPN −5',
      'dSPN +0.25', 'dSPN +0.5', 'dSPN +1']


def thr50(x, yb):
    yb = yb.astype(float); mu, sd = x.mean(), x.std() + 1e-12; z = (x - mu) / sd; w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(w[0] + w[1] * z))); W = p * (1 - p) + 1e-9
        H = np.array([[W.sum(), (W * z).sum()], [(W * z).sum(), (W * z * z).sum()]]) + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, np.array([(yb - p).sum(), ((yb - p) * z).sum()]))
    return None if w[1] <= 0 else mu - sd * w[0] / w[1]


PT = np.arange(320, 600, 20)


def logit_cv(X, yb, k=5, lam=1e-2, seed=0):
    yb = yb.astype(bool)
    if yb.sum() < 10 or (~yb).sum() < 10:
        return None
    fold = np.random.default_rng(seed).permutation(len(yb)) % k; pred = np.zeros(len(yb))
    for f_ in range(k):
        Xa, ya = X[fold != f_], yb[fold != f_]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Z = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]; w = np.zeros(Z.shape[1])
        for _ in range(50):
            pz = 1 / (1 + np.exp(-Z @ w)); W = pz * (1 - pz) + 1e-6
            H = Z.T @ (Z * W[:, None]) + lam * len(ya) * np.eye(len(w)); H[0, 0] -= lam * len(ya)
            w = w + np.linalg.solve(H, Z.T @ (ya - pz) - lam * len(ya) * np.r_[0, w[1:]])
        pred[fold == f_] = np.c_[np.ones((fold == f_).sum()), (X[fold == f_] - mu) / sd] @ w
    from scipy.stats import rankdata
    n1, n0 = yb.sum(), (~yb).sum(); rk = rankdata(pred)
    return float((rk[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def c_types(k, name):
    return json.load(open(os.path.join(EXP, 'pnr_determinants', f'seed{k}_intact', 'metrics.json')))['types'][name]['p_triggered']


def rise_lead(X, rt0, start=320):
    """median over responding trials of RT minus the onset of the rise that ends at the signal's peak in the last 0.5 s
    before RT (onset = last 100-ms difference <= 0 before that peak)."""
    out = []
    for s in np.where(np.isfinite(rt0))[0]:
        r = int(rt0[s]); x = X[s, start - 10:r + 1]
        if len(x) < 62:
            continue
        pk = len(x) - 51 + int(np.argmax(x[-51:]))
        dx = x[10:pk + 1] - x[:pk + 1 - 10]; neg = np.where(dx <= 0)[0]
        out.append((r - (start + (neg[-1] + 1 if len(neg) else 0))) / 100)
    return float(np.median(out)) if out else None


def surv_cv(k, DTW):
    dd = np.load(os.path.join(EXP, 'pnr_determinants', f'seed{k}_intact', 'data.npz'), allow_pickle=True)
    cols = {n: i for i, n in enumerate(dd['cols'])}; R = dd['rows']
    mk = (R[:, 0] == TN.index('dSPN +1')) & (R[:, cols['svalid']] == 1); R = R[mk]
    yb = R[:, cols['surv']].astype(bool); ip = R[:, 1].astype(int); sd_ = R[:, cols['seed']].astype(int)
    X0 = R[:, cols['y']:]; t = PT[ip] - 1
    E = DTW[sd_, t, 0] + DTW[sd_, t, 2]; I = DTW[sd_, t, 1] + DTW[sd_, t, 3]
    t10 = t - 10; E1 = DTW[sd_, t10, 0] + DTW[sd_, t10, 2]; I1 = DTW[sd_, t10, 1] + DTW[sd_, t10, 3]
    XE = np.c_[E, I, E - I, E / np.maximum(I, 1e-6), (E - E1) * 10, (I - I1) * 10, DTW[sd_, t, :4]]
    oh = (ip[:, None] == np.arange(NP)[None]).astype(float)
    ok = np.isfinite(X0).all(1) & np.isfinite(XE).all(1)
    if yb[ok].sum() < 10 or (~yb[ok]).sum() < 10:
        return None
    rng = np.random.default_rng(1); XS = XE.copy()
    for q in range(NP):                               # shuffle E/I features within delay
        ii = np.where(ip == q)[0]; XS[ii] = XE[rng.permutation(ii)]
    base = [logit_cv(np.c_[oh, X0][ok], yb[ok], seed=f) for f in range(3)]
    add = [logit_cv(np.c_[oh, X0, XE][ok], yb[ok], seed=f) for f in range(3)]
    shuf = [logit_cv(np.c_[oh, X0, XS][ok], yb[ok], seed=f) for f in range(3)]
    return dict(state=float(np.mean(base)), state_EI=float(np.mean(add)), state_EIshuffled=float(np.mean(shuf)))


def params_of(k):
    p = (os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl') if k == 13
         else os.path.join(ROOT, 'seeds', '3k', f'seed_{k}', 'params_nm.pkl'))
    return {a: np.asarray(v, np.float64) for a, v in pickle.load(open(p, 'rb')).items()}


S = {}
for k in NETS:
    m = json.load(open(os.path.join(D, f'seed{k}', 'metrics.json'))); d = np.load(os.path.join(D, f'seed{k}', 'data.npz'))
    r = S[k] = {}
    r['rate_cross'] = m['mean_rate_crossing_lead_s']
    for name, key in [('stop −5', 'trig'), ('dSPN +1', 'surv'), ('iSPN −1', 'trig')]:
        b = m['B_prepulse'][name]; a = b['within_delay_auc']
        r[f'{key}_{name}'] = dict(auc={f: a[f] for f in ['E−I', 'd(E−I)/dt', 'E/I', 'G_E−G_I', 'progress', 'y']}, cv=b['cv'], p=b['p'])
    # single-threshold tests at +100 ms from pulse onset (POST index 2); trials already past threshold at that time dropped
    POST, POSTY, valid, trig, RT = d['POST'], d['POSTY'], d['valid'], d['trig'], d['rt']
    q, off = 2, 10
    cidx = np.arange(1, 1 + nT * NP); tix = (cidx - 1) // NP; pix = (cidx - 1) % NP
    notyet = RT[:, 1:] >= (PT[pix] + off)[None]
    var = {'E−I': POST[:, 1:, q, 0] + POST[:, 1:, q, 2] - POST[:, 1:, q, 1] - POST[:, 1:, q, 3], 'y': POSTY[:, 1:, q, 0], 'thalamus': POSTY[:, 1:, q, 1]}
    m_ = valid[:, 1:] & notyet; TT = np.broadcast_to(tix, m_.shape)[m_]; yb = trig[:, 1:][m_]
    oh = (TT[:, None] == np.arange(nT)[None]).astype(float)
    r['thr'] = {}
    for vn, V in var.items():
        x = V[m_]; th = {}
        for t in range(nT):
            mk = TT == t
            if yb[mk].sum() >= 10 and (~yb[mk]).sum() >= 10:
                v = thr50(x[mk], yb[mk])
                if v is not None:
                    th[TN[t]] = float(v)
        r['thr'][vn] = dict(by_type=th, dispersion=float(np.std(list(th.values())) / x.std()) if len(th) >= 3 else None, n_types=len(th),
                            cv_var=logit_cv(x[:, None], yb), cv_var_type=logit_cv(np.c_[x, oh], yb))
    # matched net drive: disinhibitory (iSPN -1/-5) vs excitatory (dSPN +0.25/+0.5/+1) pulses, P(trig) in shared E-I bins
    x = var['E−I'][m_]; dis = np.isin(TT, [TN.index('iSPN −1'), TN.index('iSPN −5')]); ex = np.isin(TT, [TN.index(n) for n in ('dSPN +0.25', 'dSPN +0.5', 'dSPN +1')])
    bins = np.quantile(x[dis | ex], np.linspace(0, 1, 16)); diffs, wts, curves = [], [], []
    for i in range(15):
        inb = (x >= bins[i]) & (x < bins[i + 1])
        if (inb & dis).sum() >= 10 and (inb & ex).sum() >= 10:
            pd_, pe_ = yb[inb & dis].mean(), yb[inb & ex].mean(); diffs.append(pd_ - pe_); wts.append((inb & (dis | ex)).sum())
            curves.append((float(x[inb].mean()), float(pd_), float(pe_)))
    r['matched'] = dict(mean_diff=float(np.average(diffs, weights=wts)) if diffs else None, n_bins=len(diffs), curves=curves)
    # stop pulse removes both E_bg and I_bg: predicted net effect on readout thalamus = I_bg - E_bg just before the pulse (twin)
    DTW = d['DTW']; ib_eb = np.array([np.mean(DTW[:, p - 1, 1] - DTW[:, p - 1, 0]) for p in PT])
    stopc = np.array([np.nan if v is None else v for v in c_types(k, 'stop −5')])
    r['stop_net'] = dict(I_bg_minus_E_bg=ib_eb.tolist(), stop5_trig=stopc.tolist(),
                         rho=(float(spearmanr(ib_eb[np.isfinite(stopc)], stopc[np.isfinite(stopc)])[0]) if np.isfinite(stopc).sum() >= 5 and np.nanstd(stopc) > 0 else None))
    # timing controls: final sustained rise of y and of ramp progress (same rule as E-I)
    Yt = d['Y_twin']; prog = np.load(os.path.join(EXP, 'pnr_determinants', f'seed{k}_intact', 'data.npz'))['prog'][:Yt.shape[0]]
    NETt = DTW[..., 0] + DTW[..., 2] - DTW[..., 1] - DTW[..., 3]
    r['lead'], r['lead_y'], r['lead_prog'] = rise_lead(NETt, RT[:, 0]), rise_lead(Yt, RT[:, 0]), rise_lead(prog, RT[:, 0])
    # thalamic loop gain from cue to RT (responders)
    LAM = d['LAM']; lm = [LAM[s, 30:int(RT[s, 0]) // 10 + 1].max() for s in range(LAM.shape[0]) if np.isfinite(RT[s, 0]) and int(RT[s, 0]) // 10 >= 31]
    r['lam_max'], r['lam_frac1'] = (float(np.median(lm)), float(np.mean(np.array(lm) > 1))) if lm else (None, None)
    # cancellation (dSPN +1 survive): does E/I add to delay + state? 3 fold seeds, and E/I shuffled within delay as control
    r['surv_cv'] = surv_cv(k, DTW)
    # structural readout-weighted E/I onto thalamus
    P = params_of(k); w = np.abs(P['C']).ravel(); w /= w.sum()
    r['W_E_bg'] = float(w @ np.abs(P['B_tbg'][:, :10]).sum(1)); r['W_I_bg'] = float(w @ np.abs(P['B_tbg'][:, 10:]).sum(1))
    r['W_E_rec'] = float(w @ np.maximum(P['J_t'], 0).sum(1)); r['W_I_rec'] = float(w @ np.maximum(-P['J_t'], 0).sum(1))
    r['W_EI_bg'] = r['W_E_bg'] / r['W_I_bg']; r['W_EI_all'] = (r['W_E_bg'] + r['W_E_rec']) / (r['W_I_bg'] + r['W_I_rec'])
    c = CMP[k]
    r['stop_net_late'] = float(np.mean(r['stop_net']['I_bg_minus_E_bg'][9:]))   # 2.0-2.8 s
    r['edge'] = np.nanmean([np.nan if e is None else e for e in c['late_edges']]) if any(e is not None for e in c['late_edges']) else None
    r['cs_peak'] = c['cs_peak_delay']; r['stop_late'] = c['stop_late_min']; r['similarity'] = c['similarity']
    r['respond'] = c['respond']; r['amb'] = c['amb_cue_only']; r['lat2'] = c['lat_2p0']

json.dump(S, open(os.path.join(D, 'summary.json'), 'w'), indent=1, default=float)


def f(v, fmt='{:.2f}'):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


name = lambda k: '**13 (baseline)**' if k == 13 else str(k)
L = ['| network | rise to the pre-movement peak, onset before RT (median s): E−I / y / progress | mean dE/dt > dI/dt from (s before RT) | PNR late edge (s) | thalamic loop λ max, cue→RT (frac > 1) | '
     'stop −5 trig, within-delay AUC: E−I / d(E−I)/dt / E/I / G_E−G_I / progress |',
     '|---|---|---|---|---|---|']
for k in NETS:
    r = S[k]; a = r['trig_stop −5']['auc']
    L.append(f"| {name(k)} | {f(r['lead'])} / {f(r['lead_y'])} / {f(r['lead_prog'])} | {f(r['rate_cross'])} | {f(r['edge'])} | {f(r['lam_max'])} ({f(r['lam_frac1'])}) | "
             + ' / '.join(f(a[x]) for x in ['E−I', 'd(E−I)/dt', 'E/I', 'G_E−G_I', 'progress']) + ' |')
L += ['', '| network | dSPN +1 survive, CV AUC: delay+state / +E/I / +E/I shuffled within delay | trigger CV AUC at +100 ms, variable alone → + pulse type: E−I / y / thalamic rate | disinhibitory − excitatory pulse P(trig) at matched E−I (bins) | stop −5: ρ(I_bg − E_bg before pulse, P(trig)) over delays |',
      '|---|---|---|---|---|']
for k in NETS:
    r = S[k]; t = r['thr']; sc = r['surv_cv']; mt = r['matched']
    L.append(f"| {name(k)} | " + ('–' if sc is None else f"{sc['state']:.3f} / {sc['state_EI']:.3f} / {sc['state_EIshuffled']:.3f}") + ' | '
             + ' / '.join(f"{f(t[v]['cv_var'])} → {f(t[v]['cv_var_type'])}" for v in ['E−I', 'y', 'thalamus'])
             + f" | {f(mt['mean_diff'], '{:+.2f}')} ({mt['n_bins']}) | {f(r['stop_net']['rho'], '{:+.2f}')} |")
L += ['', '| network | readout-weighted structural input onto thalamus: dSPN / iSPN (ratio) | recurrent J_t: + / − | late stop pulse P(respond) |', '|---|---|---|---|']
for k in NETS:
    r = S[k]
    L.append(f"| {name(k)} | {r['W_E_bg']:.2f} / {r['W_I_bg']:.2f} ({r['W_EI_bg']:.2f}) | {r['W_E_rec']:.2f} / {r['W_I_rec']:.2f} | {r['stop_late']:.2f} |")
# cross-network correlations
pairs = []
for xa, ya in [('lead', 'edge'), ('lead_y', 'edge'), ('W_EI_bg', 'stop_late'), ('stop_net_late', 'stop_late'), ('W_EI_bg', 'edge')]:
    xs = [S[k][xa] for k in NETS]; ys = [S[k][ya] for k in NETS]
    ok = [i for i in range(len(NETS)) if xs[i] is not None and ys[i] is not None and np.isfinite(xs[i]) and np.isfinite(ys[i])]
    if len(ok) >= 5:
        rho, p = spearmanr([xs[i] for i in ok], [ys[i] for i in ok]); pairs.append((xa, ya, len(ok), rho, p))
L += ['', f'Exploratory: {len(pairs)} tests on <= {len(NETS)} networks, no correction.', '', '| across networks | n | Spearman ρ | p |', '|---|---|---|---|'] + [f'| {a} vs {b} | {n} | {rho:+.2f} | {p:.3f} |' for a, b, n, rho, p in pairs]
print('\n'.join(L))
json.dump({'tables': L}, open(os.path.join(D, 'summary_tables.json'), 'w'), indent=1)

# ---------------- figure ----------------
cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(NETS) - 1)))
COL = {13: 'k', **{k: cmap[i] for i, k in enumerate([k for k in NETS if k != 13])}}
fig, ax = plt.subplots(2, 3, figsize=(19, 11)); ax = ax.ravel()
for k in NETS:
    d = np.load(os.path.join(D, f'seed{k}', 'data.npz')); DTW, rt = d['DTW'], d['rt'][:, 0]
    NET = DTW[..., 0] + DTW[..., 2] - DTW[..., 1] - DTW[..., 3]
    resp = np.isfinite(rt); lags = np.arange(-300, 51)
    al = np.full((resp.sum(), len(lags)), np.nan)
    for i, s in enumerate(np.where(resp)[0]):
        t = int(rt[s]) + lags; ok = (t >= 0) & (t < NET.shape[1]); al[i, ok] = NET[s, t[ok]]
    kw = dict(color=COL[k], lw=2.5 if k == 13 else 1, label=('baseline' if k == 13 else f'seed {k}'))
    if resp.sum() >= 5:
        mn = np.nanmean(al, 0); ax[0].plot(lags / 100, mn - np.nanmean(mn[:20]), **kw)
        dn = np.full_like(mn, np.nan); dn[10:] = (mn[10:] - mn[:-10]) / 0.1; ax[1].plot(lags / 100, dn, **kw)
ax[0].set(xlabel='time from movement (s)', ylabel='E − I (relative to −3 s)', title='net drive onto readout thalamus, movement-aligned'); ax[0].legend(fontsize=6, ncol=2)
ax[1].axhline(0, color='gray', lw=0.5); ax[1].set(xlabel='time from movement (s)', ylabel='d(E − I)/dt (per s)', title='rate of change of net drive')
x = [S[k]['lead'] for k in NETS]; y = [S[k]['edge'] for k in NETS]
for k, xx, yy in zip(NETS, x, y):
    if xx is not None and yy is not None and np.isfinite(yy):
        ax[2].scatter(xx, yy, color=COL[k], s=50, edgecolor='k'); ax[2].annotate('base' if k == 13 else str(k), (xx, yy), xytext=(3, 3), textcoords='offset points', fontsize=8)
ax[2].plot([0, 2], [0, 2], color='gray', ls='--', lw=0.8)
ax[2].set(xlabel='onset of the E−I rise to its pre-movement peak, median s before RT', ylabel='PNR late edge (s before own movement)', title='does the E−I rise mark the PNR?')
fe = ['E−I', 'd(E−I)/dt', 'E/I', 'progress', 'y']; wdt = 0.8 / len(NETS)
for j, k in enumerate(NETS):
    a = S[k]['trig_stop −5']['auc']
    ax[3].bar(np.arange(len(fe)) + (j - len(NETS) / 2) * wdt, [np.nan if a[x] is None else a[x] for x in fe], wdt, color=COL[k])
ax[3].axhline(0.5, color='k', lw=0.5); ax[3].set_xticks(range(len(fe)), fe); ax[3].set(ylim=(0, 1), ylabel='within-delay AUC', title='stop −5 triggering: pre-pulse predictors (clock fixed)')
for j, vn in enumerate(['E−I', 'y', 'thalamus']):
    vals = [None if S[k]['thr'][vn]['cv_var'] is None else S[k]['thr'][vn]['cv_var_type'] - S[k]['thr'][vn]['cv_var'] for k in NETS]
    ax[4].scatter(np.full(len(NETS), j) + np.linspace(-0.2, 0.2, len(NETS)), [np.nan if v is None else v for v in vals], c=[COL[k] for k in NETS], s=40, edgecolor='k')
ax[4].axhline(0, color='gray', lw=0.5)
ax[4].set_xticks(range(3), ['E − I', 'output y', 'thalamic rate']); ax[4].set(ylabel='CV AUC gain from adding pulse type', title='one threshold for all pulse types? (0 = yes; state at +100 ms)')
for k in NETS:
    cv_ = np.array(S[k]['matched']['curves'])
    if len(cv_):
        ax[5].plot(cv_[:, 0], cv_[:, 1], '-', color=COL[k], lw=2.5 if k == 13 else 1)
        ax[5].plot(cv_[:, 0], cv_[:, 2], '--', color=COL[k], lw=2.5 if k == 13 else 1)
ax[5].plot([], [], 'k-', label='disinhibitory (iSPN −1, −5)'); ax[5].plot([], [], 'k--', label='excitatory (dSPN +0.25 … +1)'); ax[5].legend(fontsize=8)
ax[5].set(xlabel='E − I onto readout thalamus, +100 ms', ylabel='P(triggered within 500 ms)', ylim=(-0.02, 1.02), title='same net drive, different outcome?')
fig.suptitle('Fig 9. Is commitment set by the excitation/inhibition balance at the readout thalamus? (13 networks)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, 'summary.png'), dpi=95); plt.close(fig)

# structural E/I onto readout thalamus vs PNR edge
fig, ax = plt.subplots(figsize=(6.5, 5))
for k in NETS:
    if S[k]['edge'] is not None and np.isfinite(S[k]['edge']):
        clock = k in (24, 4, 23, 6)
        ax.scatter(S[k]['W_EI_bg'], S[k]['edge'], color=COL[k], s=60, edgecolor='k', marker='s' if clock else 'o')
        ax.annotate('base' if k == 13 else str(k), (S[k]['W_EI_bg'], S[k]['edge']), xytext=(4, 3), textcoords='offset points', fontsize=8)
ok = [k for k in NETS if S[k]['edge'] is not None and np.isfinite(S[k]['edge'])]
rho, p = spearmanr([S[k]['W_EI_bg'] for k in ok], [S[k]['edge'] for k in ok])
ax.axvline(1, color='gray', lw=0.5)
ax.set(xlabel='readout-weighted dSPN / iSPN weight onto thalamus', ylabel='PNR late edge (s before own movement)',
       title=f'structural E/I at the readout thalamus vs PNR\nSpearman ρ = {rho:+.2f}, p = {p:.3f}, n = {len(ok)} (squares: clock-locked or no clean edge)')
fig.tight_layout(); fig.savefig(os.path.join(D, 'structural_ei_vs_edge.png'), dpi=110); plt.close(fig)
