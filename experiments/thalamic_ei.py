"""Is commitment set by the excitation/inhibition balance at the readout thalamus? Paired-pulse re-simulation.

usage (repo root, CPU): JAX_PLATFORMS=cpu python experiments/thalamic_ei.py <seed>
Same design, conditions and noise keys as pnr_determinants.py (intact): 200 noise seeds x (cue-only twin + 13 pulse
types x 14 delays), one key per seed. Output: experiments/thalamic_ei/seed<k>/ (data.npz, metrics.json, thalamic_ei.png).

Thalamic unit i at step t is driven by (model_functions.multiregion_nmrnn; rates r = nln(x), BG rates already updated):
  excitation E_i = sum_j |B_tbg[i, dSPN j]| r_dSPN_j  +  sum_j max(J_t[i, j], 0) r_thal_j(t-1)
  inhibition I_i = sum_j |B_tbg[i, iSPN j]| r_iSPN_j  +  sum_j max(-J_t[i, j], 0) r_thal_j(t-1)
Everything is read out through y = |C| . nln(x_t) + rb, so each quantity is averaged over thalamic units with weights
w_i = |C_i| / sum |C|  ("readout-weighted"). Split into feedforward BG parts (E_bg, I_bg) and thalamic recurrent parts
(E_rec, I_rec). Rates of change dE/dt, dI/dt over the previous 100 ms (per s).
Recurrent gain = sensitivity of that readout-weighted drive to the presynaptic state: G_E = sum_j (w.|B_d1| or w.J+)_j
nln'(x_j), likewise G_I; and lam_t = largest real part of eig(J_t diag(nln'(x_t))) (thalamic loop gain; > 1 means the
thalamic loop alone amplifies).

Tests (outcomes exactly as pnr_determinants.py):
 A. movement-aligned time course of E, I, E-I, dE/dt - dI/dt, gains, lam_t in cue-only twins; per trial, the onset of
    the final sustained rise of E-I (d(E-I)/dt > 0 from then until RT) and its lead before RT.
 B. at a fixed pulse delay, does the pre-pulse E/I state of the twin predict triggering (stop -5) and survival
    (dSPN +1)? AUC per feature (within delay), and 5-fold CV logistic: delay + state (pnr_determinants FEATS) vs
    delay + state + E/I.
 C. one threshold across all pulse types: at pulse end, does the readout-weighted net drive E-I (or its rate) of the
    pulse trial predict triggering pooled over all 13 types and 14 delays, better than y or thalamic rate?
"""
import os, sys, json, pickle
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
sys.path[:0] = [EXP, ROOT]
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
import config_script as cs
import model_functions as mf
import perturb_model as pm

seed = int(sys.argv[1])
PP = (os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl') if seed == 13
      else os.path.join(ROOT, 'seeds', '3k', f'seed_{seed}', 'params_nm.pkl'))
params = pickle.load(open(PP, 'rb'))
OUT = os.path.join(EXP, 'thalamic_ei', f'seed{seed}')
os.makedirs(OUT, exist_ok=True)
DET = os.path.join(EXP, 'pnr_determinants', f'seed{seed}_intact')

T, N_BG, N1 = cs.config['T'], cs.config['n_bg'], cs.n_d1_cells
TAU_X, TAU_Z = cs.config['tau_x'], cs.config['tau_z']
START, DUR, NS = 300, 5, int(os.environ.get('PNR_NS', 200))
PT = np.arange(320, 600, 20); DL = (PT - START) / 100; NP = len(PT)
D1, D2, ALL = slice(0, N1), slice(N1, N_BG), slice(0, N_BG)
TYPES = [('stop −0.1', ALL, -0.1), ('stop −0.25', ALL, -0.25), ('stop −0.5', ALL, -0.5), ('stop −1', ALL, -1.0),
         ('stop −2.5', ALL, -2.5), ('stop −5', ALL, -5.0), ('stop −10', ALL, -10.0),
         ('dSPN −5', D1, -5.0), ('iSPN −1', D2, -1.0), ('iSPN −5', D2, -5.0),
         ('dSPN +0.25', D1, 0.25), ('dSPN +0.5', D1, 0.5), ('dSPN +1', D1, 1.0)]
TN = [t[0] for t in TYPES]; nT = len(TYPES); nC = 1 + nT * NP
stim = np.zeros((nC, T, N_BG), np.float32)
for it, (_, sl, a) in enumerate(TYPES):
    for ip, p in enumerate(PT):
        stim[1 + it * NP + ip, p:p + DUR, sl] = a
base, _, _ = mf.self_timed_movement_task(jnp.array([START]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)
inputs = jnp.repeat(base, nC, 0); stim = jnp.asarray(stim)
bat = pm.make_batched()
sim = jax.jit(lambda k: bat(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim, cs.test_noise_std, jnp.repeat(k[None], nC, 0)))

# ---- readout-weighted connectivity onto thalamus ----
P = {k: np.asarray(v, np.float64) for k, v in params.items()}
w = np.abs(P['C']).ravel(); w = w / w.sum()
Jt = P['J_t']; Jp, Jn = np.maximum(Jt, 0), np.maximum(-Jt, 0)
Bd1, Bd2 = np.abs(P['B_tbg'][:, :N1]), np.abs(P['B_tbg'][:, N1:])
v_e_bg, v_i_bg, v_e_rec, v_i_rec = w @ Bd1, w @ Bd2, w @ Jp, w @ Jn       # (presyn,) readout-weighted weights


def nln(x):
    return np.maximum(0, np.tanh(x))


def dnln(x):
    return (1 - np.tanh(x) ** 2) * (x > 0)


def drive(xbg, xt):
    """xbg (..., T, 20), xt (..., T, 20) -> (..., T, 8): E_bg, I_bg, E_rec, I_rec, G_E_bg, G_I_bg, G_E_rec, G_I_rec."""
    rb, rt = nln(xbg), nln(xt)
    rtp = np.concatenate([rt[..., :1, :], rt[..., :-1, :]], -2)
    xtp = np.concatenate([xt[..., :1, :], xt[..., :-1, :]], -2)
    return np.stack([rb[..., D1] @ v_e_bg, rb[..., D2] @ v_i_bg, rtp @ v_e_rec, rtp @ v_i_rec,
                     dnln(xbg[..., D1]) @ v_e_bg, dnln(xbg[..., D2]) @ v_i_bg, dnln(xtp) @ v_e_rec, dnln(xtp) @ v_i_rec], -1)


def first_cross(y, t0):
    above = y[..., t0:] > 0.5
    return np.where(above.any(-1), above.argmax(-1) + t0, np.inf)


LS = np.arange(0, T, 10)
Y = np.zeros((NS, nC, T), np.float32)
DTW = np.zeros((NS, T, 8), np.float32)              # twin time series
LAM = np.zeros((NS, len(LS)), np.float32)          # twin thalamic loop gain
RTW = np.zeros((NS, T, 65), np.float32)            # twin rates (for ramp progress)
POST = np.zeros((NS, nC, 4, 8), np.float32)         # pulse trials: drive 10 ms before onset, at pulse end (+50 ms), +100, +150 ms from onset
POSTY = np.zeros((NS, nC, 4, 2), np.float32)        # y and mean thalamic rate at the same times
TOFF = [-1, DUR, DUR + 5, DUR + 10]
for s in range(NS):
    ys, xs, zs = sim(jr.PRNGKey(1000 + s))
    Y[s] = np.asarray(ys[..., 0])
    xbg, xc, xt = (np.asarray(a, np.float64) for a in xs)
    D = drive(xbg, xt)                              # (nC, T, 8)
    DTW[s] = D[0]
    RTW[s] = np.concatenate([nln(xbg[0]), nln(xc[0]), nln(xt[0]), nln(np.asarray(zs[0]))], -1)
    for j, t in enumerate(LS):
        LAM[s, j] = np.linalg.eigvals(Jt * dnln(xt[0, max(t - 1, 0)])[None, :]).real.max()
    for c in range(1, nC):
        p = PT[(c - 1) % NP]
        for q, o in enumerate(TOFF):
            POST[s, c, q] = D[c, p + o]; POSTY[s, c, q] = [Y[s, c, p + o], nln(xt[c, p + o]).mean()]
    if s == 0:
        print('first sim done', flush=True)

rt = first_cross(Y, START); rt0 = rt[:, 0]
if os.path.exists(os.path.join(DET, 'data.npz')):
    ref = np.load(os.path.join(DET, 'data.npz'), allow_pickle=True)['rt']
    print('twin/pulse RT identical to pnr_determinants:', bool(np.array_equal(ref[:NS], rt)), flush=True)

E = DTW[..., 0] + DTW[..., 2]; I = DTW[..., 1] + DTW[..., 3]; NET = E - I
GE = DTW[..., 4] + DTW[..., 6]; GI = DTW[..., 5] + DTW[..., 7]


def rate(x, lag=10):
    out = np.full_like(x, np.nan); out[:, lag:] = (x[:, lag:] - x[:, :-lag]) / (lag / 100); return out


dE, dI, dNET = rate(E), rate(I), rate(NET)
resp = np.isfinite(rt0)
M = {'seed': seed, 'n_seeds': NS, 'twin_respond': float(resp.mean()), 'readout_weights_|C|': w.tolist()}

# ---------- A. movement-aligned ----------
LAGS = np.arange(-300, 51)
def align(x):
    out = np.full((resp.sum(), len(LAGS)), np.nan)
    for i, s in enumerate(np.where(resp)[0]):
        t = int(rt0[s]) + LAGS; ok = (t >= 0) & (t < T); out[i, ok] = x[s, t[ok]]
    return out
AL = {n: align(x) for n, x in [('E', E), ('I', I), ('E-I', NET), ('dE/dt', dE), ('dI/dt', dI), ('d(E-I)/dt', dNET),
                                ('E_bg', DTW[..., 0]), ('I_bg', DTW[..., 1]), ('E_rec', DTW[..., 2]), ('I_rec', DTW[..., 3]),
                                ('G_E', GE), ('G_I', GI), ('y', Y[:, 0])]}
# onset of the final sustained rise of E-I before RT (smoothed d(E-I)/dt > 0 from onset to RT)
lead = []
for s in np.where(resp)[0]:
    r = int(rt0[s]); d = dNET[s, START + 20:r]
    if len(d) == 0 or not np.isfinite(d).all():
        continue
    neg = np.where(d <= 0)[0]
    lead.append((r - (START + 20 + (neg[-1] + 1 if len(neg) else 0))) / 100)
lead = np.array(lead)
M['net_rise_lead_s'] = dict(median=float(np.median(lead)) if len(lead) else None, iqr=np.percentile(lead, [25, 75]).tolist() if len(lead) else None, n=len(lead))
# lead vs the RT-aligned crossing of mean dE/dt and dI/dt
mdE, mdI = np.nanmean(AL['dE/dt'], 0), np.nanmean(AL['dI/dt'], 0)
pos = (mdE - mdI) > 0; pre = np.where(LAGS <= 0)[0]
lastneg = [i for i in pre if not pos[i]]
M['mean_rate_crossing_lead_s'] = float(-LAGS[lastneg[-1] + 1] / 100) if lastneg and lastneg[-1] + 1 <= pre[-1] else None
mI_bg, mI_rec = np.nanmean(AL['I_bg'], 0), np.nanmean(AL['I_rec'], 0)
M['aligned_means'] = {n: {f'{l / 100:.1f}': (None if not np.isfinite(v) else round(float(v), 4))
                          for l, v in zip(LAGS[::10], np.nanmean(AL[n], 0)[::10])} for n in AL}
# responders vs non-responders, cue-locked
M['cue_locked_net'] = {k: np.nanmean(NET[m_], 0)[START:START + 300:20].round(4).tolist() for k, m_ in [('resp', resp), ('nonresp', ~resp)] if m_.sum() >= 5}
M['lam_t'] = dict(max_median=float(np.median(LAM[:, START // 10:].max(1))),
                  frac_trials_above_1=float((LAM[:, START // 10:].max(1) > 1).mean()))
LAMAL = []
for s in np.where(resp)[0]:
    j = int(rt0[s]) // 10; LAMAL.append(LAM[s, max(j - 30, 0):j + 1][::-1][:31])
lam_al = np.full((len(LAMAL), 31), np.nan)
for i, v in enumerate(LAMAL):
    lam_al[i, :len(v)] = v
lam_al = lam_al[:, ::-1]  # -3.0 .. 0 s


# ---------- B. pre-pulse E/I state at a fixed delay ----------
def auc(x, yb):
    yb = yb.astype(bool); n1, n0 = yb.sum(), (~yb).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(x); return float((rk[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_cv(X, yb, k=5, lam=1e-2):
    yb = yb.astype(bool)
    if yb.sum() < 10 or (~yb).sum() < 10:
        return np.nan
    fold = np.random.default_rng(0).permutation(len(yb)) % k; pred = np.zeros(len(yb))
    for f in range(k):
        Xa, ya = X[fold != f], yb[fold != f]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Z = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]; wv = np.zeros(Z.shape[1])
        for _ in range(50):
            pz = 1 / (1 + np.exp(-Z @ wv)); W = pz * (1 - pz) + 1e-6
            H = Z.T @ (Z * W[:, None]) + lam * len(ya) * np.eye(len(wv)); H[0, 0] -= lam * len(ya)
            wv = wv + np.linalg.solve(H, Z.T @ (ya - pz) - lam * len(ya) * np.r_[0, wv[1:]])
        pred[fold == f] = np.c_[np.ones((fold == f).sum()), (X[fold == f] - mu) / sd] @ wv
    return auc(pred, yb)


EIF = ['E', 'I', 'E−I', 'E/I', 'dE/dt', 'dI/dt', 'd(E−I)/dt', 'E_bg', 'I_bg', 'E_rec', 'I_rec', 'G_E', 'G_I', 'G_E−G_I']


def eifeats(s, t):
    return [E[s, t], I[s, t], NET[s, t], E[s, t] / max(I[s, t], 1e-6), dE[s, t], dI[s, t], dNET[s, t], DTW[s, t, 0], DTW[s, t, 1],
            DTW[s, t, 2], DTW[s, t, 3], GE[s, t], GI[s, t], GE[s, t] - GI[s, t]]


dd = np.load(os.path.join(DET, 'data.npz'), allow_pickle=True)
cols = {n: i for i, n in enumerate(dd['cols'])}; ROWS = dd['rows']; F0 = cols['y']
ROWS = ROWS[ROWS[:, cols['seed']] < NS]
B = {}
for name, outc, vcol in [('stop −5', 'trig', 'valid'), ('dSPN +1', 'surv', 'svalid'), ('iSPN −1', 'trig', 'valid')]:
    mk = (ROWS[:, 0] == TN.index(name)) & (ROWS[:, cols[vcol]] == 1)
    R_ = ROWS[mk]; yb = R_[:, cols[outc]].astype(bool); ip = R_[:, 1].astype(int); sd_ = R_[:, cols['seed']].astype(int)
    X0 = R_[:, F0:]
    XE = np.array([eifeats(s, PT[q] - 1) for s, q in zip(sd_, ip)])
    per = {}
    for fi, fn in enumerate(EIF + ['progress', 'thalamus', 'y']):
        v = XE[:, fi] if fi < len(EIF) else X0[:, list(dd['cols'][F0:]).index(fn)]
        a_, wt = [], []
        for q in range(NP):
            m_ = ip == q
            if yb[m_].sum() >= 10 and (~yb[m_]).sum() >= 10 and np.isfinite(v[m_]).all():
                a_.append(auc(v[m_], yb[m_])); wt.append(m_.sum())
        per[fn] = None if not a_ else round(float(np.average(a_, weights=wt)), 3)   # signed: > 0.5 = higher value -> outcome
    oh = (ip[:, None] == np.arange(NP)[None]).astype(float)
    ok = np.isfinite(XE).all(1)
    B[name] = dict(outcome=outc, n=int(mk.sum()), p=float(yb.mean()), within_delay_auc=per,
                   cv=dict(delay=logit_cv(oh[ok], yb[ok]), delay_state=logit_cv(np.c_[oh, X0][ok], yb[ok]),
                           delay_EI=logit_cv(np.c_[oh, XE][ok], yb[ok]), delay_state_EI=logit_cv(np.c_[oh, X0, XE][ok], yb[ok])))
M['B_prepulse'] = B

# ---------- C. one threshold across pulse types: drive right after the pulse ----------
valid = np.zeros((NS, nC), bool); trig = np.zeros((NS, nC), bool)
for c in range(1, nC):
    p = PT[(c - 1) % NP]; valid[:, c] = rt0 >= p + 50; trig[:, c] = rt[:, c] < p + 50
C_ = {}
cidx = np.arange(1, nC); tix = (cidx - 1) // NP; pix = (cidx - 1) % NP
for q, o in enumerate(TOFF):
    Ev, Iv = POST[:, 1:, q, 0] + POST[:, 1:, q, 2], POST[:, 1:, q, 1] + POST[:, 1:, q, 3]
    m_ = valid[:, 1:]
    yb = trig[:, 1:][m_]
    feats = {'E−I': (Ev - Iv)[m_], 'E': Ev[m_], 'I': Iv[m_], 'G_E−G_I': (POST[:, 1:, q, 4] + POST[:, 1:, q, 6] - POST[:, 1:, q, 5] - POST[:, 1:, q, 7])[m_],
             'y': POSTY[:, 1:, q, 0][m_], 'thalamus': POSTY[:, 1:, q, 1][m_]}
    if q >= 1:
        E0, I0 = POST[:, 1:, 0, 0] + POST[:, 1:, 0, 2], POST[:, 1:, 0, 1] + POST[:, 1:, 0, 3]
        feats['Δ(E−I) since pulse'] = ((Ev - Iv) - (E0 - I0))[m_]
    C_[f'{o * 10:+d} ms'] = {k: round(auc(v, yb), 3) for k, v in feats.items()}
    if q == 1:
        TT = np.broadcast_to(tix, m_.shape)[m_]; PP_ = np.broadcast_to(pix, m_.shape)[m_]
        oh = np.c_[(TT[:, None] == np.arange(nT)).astype(float), (PP_[:, None] == np.arange(NP)).astype(float)]
        C_['cv_at_pulse_end'] = dict(type_delay=logit_cv(oh, yb), type_delay_EI=logit_cv(np.c_[oh, feats['E−I'], feats['E'], feats['I']], yb),
                                     type_delay_y=logit_cv(np.c_[oh, feats['y']], yb), EI_only=logit_cv(np.c_[feats['E−I'], feats['E'], feats['I']], yb))
        # per type: E-I at pulse end vs triggering
        C_['per_type_EI_auc'] = {TN[t]: (round(auc(feats['E−I'][TT == t], yb[TT == t]), 3) if 10 <= yb[TT == t].sum() <= (TT == t).sum() - 10 else None) for t in range(nT)}
        SC = dict(net=feats['E−I'], trig=yb, type=TT, ip=PP_)
M['C_postpulse_pooled_auc'] = C_
json.dump(M, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1, default=float)
np.savez_compressed(os.path.join(OUT, 'data.npz'), DTW=DTW, LAM=LAM, Y_twin=Y[:, 0], rt=rt, POST=POST, POSTY=POSTY, valid=valid, trig=trig,
                    lags=LAGS, lam_al=lam_al, lead=lead, w=w)

# ---------- figure ----------
fig, ax = plt.subplots(3, 3, figsize=(17, 13)); ax = ax.ravel(); tl = LAGS / 100
for n, cc in [('E', 'tab:red'), ('I', 'tab:blue'), ('E-I', 'k')]:
    m_ = np.nanmean(AL[n], 0); ax[0].plot(tl, m_, color=cc, label=n)
for n, cc, ls in [('E_bg', 'tab:red', '--'), ('I_bg', 'tab:blue', '--'), ('E_rec', 'tab:red', ':'), ('I_rec', 'tab:blue', ':')]:
    ax[0].plot(tl, np.nanmean(AL[n], 0), color=cc, ls=ls, lw=1, label=n)
ax[0].set(xlabel='time from movement (s)', ylabel='readout-weighted input', title='E and I onto readout thalamus (cue-only twins)'); ax[0].legend(fontsize=7)
for n, cc in [('dE/dt', 'tab:red'), ('dI/dt', 'tab:blue'), ('d(E-I)/dt', 'k')]:
    ax[1].plot(tl, np.nanmean(AL[n], 0), color=cc, label=n)
ax[1].axhline(0, color='gray', lw=0.5)
ax[1].set(xlabel='time from movement (s)', ylabel='per s', title='rates of change (100 ms)'); ax[1].legend(fontsize=7)
ax[2].plot(tl, np.nanmean(AL['G_E'], 0), color='tab:red', label='G_E'); ax[2].plot(tl, np.nanmean(AL['G_I'], 0), color='tab:blue', label='G_I')
a2 = ax[2].twinx(); a2.plot(np.linspace(-3, 0, 31), np.nanmedian(lam_al, 0), 'g-', label='λ_t (thalamic loop)'); a2.axhline(1, color='g', ls=':', lw=0.8); a2.set_ylabel('max Re λ, thalamic loop', color='g')
ax[2].set(xlabel='time from movement (s)', ylabel='readout-weighted gain', title='E vs I gain; thalamic loop gain'); ax[2].legend(fontsize=7, loc='upper left')
if len(lead):
    ax[3].hist(lead, np.arange(0, 3.01, 0.1), color='gray')
    ax[3].axvline(np.median(lead), color='k')
ax[3].set(xlabel='lead of final sustained E−I rise before RT (s)', ylabel='trials', title=f'final rise of net drive: median {M["net_rise_lead_s"]["median"]}')
tc = (np.arange(T) - START) / 100
for k_, m_, cc in [('responders', resp, 'k'), ('non-responders', ~resp, 'tab:orange')]:
    if m_.sum() >= 5:
        ax[4].plot(tc, np.nanmean(NET[m_], 0), color=cc, label=f'{k_} (n={m_.sum()})')
ax[4].set(xlim=(-0.5, 5), xlabel='time from cue (s)', ylabel='E − I', title='net drive, cue-locked'); ax[4].legend(fontsize=7)
for r_, (name, b) in enumerate(B.items()):
    if r_ > 1:
        break
    fn = list(b['within_delay_auc']); vals = [np.nan if b['within_delay_auc'][f] is None else b['within_delay_auc'][f] for f in fn]
    ax[5 + r_].bar(range(len(fn)), vals, color=['tab:purple'] * len(EIF) + ['gray'] * 3)
    ax[5 + r_].axhline(0.5, color='k', lw=0.5); ax[5 + r_].set_xticks(range(len(fn)), fn, rotation=60, fontsize=7)
    cv = b['cv']
    ax[5 + r_].set(ylim=(0, 1), ylabel='AUC within delay (>0.5: higher → outcome)',
                   title=f"{name} → {b['outcome']}: pre-pulse twin state\nCV: delay {cv['delay']:.2f}, +state {cv['delay_state']:.2f}, +E/I {cv['delay_EI']:.2f}, +both {cv['delay_state_EI']:.2f}")
x = SC['net']; yb = SC['trig']
bins = np.quantile(x, np.linspace(0, 1, 21))
for t, cc in [(TN.index('stop −5'), 'k'), (TN.index('iSPN −5'), 'tab:orange'), (TN.index('dSPN +1'), 'tab:green'), (TN.index('iSPN −1'), 'navajowhite'), (TN.index('stop −0.5'), 'gray')]:
    mk = SC['type'] == t
    if mk.sum() > 50:
        b_ = np.digitize(x[mk], bins); xs_ = [x[mk][b_ == i].mean() for i in range(1, 21) if (b_ == i).sum() >= 10]
        ys_ = [yb[mk][b_ == i].mean() for i in range(1, 21) if (b_ == i).sum() >= 10]
        ax[7].plot(xs_, ys_, 'o-', ms=3, color=cc, label=TN[t])
ax[7].set(xlabel='E − I at pulse end (pulse trial)', ylabel='P(triggered within 500 ms)', ylim=(-0.02, 1.02),
          title=f"one threshold across types? pooled AUC E−I {C_['+50 ms']['E−I']}, y {C_['+50 ms']['y']}"); ax[7].legend(fontsize=7)
cv = C_['cv_at_pulse_end']
ax[8].bar(range(4), [cv['type_delay'], cv['type_delay_y'], cv['type_delay_EI'], cv['EI_only']], color=['gray', 'gray', 'tab:purple', 'tab:purple'])
ax[8].set_xticks(range(4), ['type+delay', '+ y', '+ E, I, E−I', 'E, I, E−I only'], fontsize=8)
ax[8].set(ylim=(0.5, 1), ylabel='5-fold CV AUC', title='triggering, all pulse types and delays (pulse-end state)')
fig.suptitle(f'Readout-thalamus excitation vs inhibition — seed {seed}{" (baseline)" if seed == 13 else ""}; twin respond {resp.mean():.2f}', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'thalamic_ei.png'), dpi=100); plt.close(fig)
print(json.dumps({k: M[k] for k in ['net_rise_lead_s', 'mean_rate_crossing_lead_s', 'lam_t', 'B_prepulse', 'C_postpulse_pooled_auc']}, indent=1, default=float))
