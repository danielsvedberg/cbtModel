"""What determines the point of no return (PNR)? Paired-pulse analysis.

usage (repo root, GPU): python experiments/pnr_determinants.py <seed> [intact|ctx_off|snc_off|snc_sat]
  seed 13 = the baseline params (baselines/2026-10-08_12a8ade); 9, 17, 21 = seeds/3k/seed_<k>.
  ctx_off: cortex silenced from cue onset + 100 ms (perturb_model ctx_off); snc_off / snc_sat: SNc state clamped
  to -10 / +10 from cue onset. The manipulation is applied identically to every condition.
Output: experiments/pnr_determinants/seed<k>_<cond>/ : data.npz, metrics.json, pnr_determinants.png

Design (unlike test_pnr_nm, where each condition has its own noise): every condition of a noise seed shares ONE rng
key, so condition 0 (cue at t=300, no pulse) is an exact twin of every pulse trial; before the pulse they are identical.
Pulses: 50 ms (5 steps) at 0.2..2.8 s after the cue, of 13 types (TYPES; stop = all SPNs). 200 noise seeds.

Per pulse trial:
  valid      twin has not crossed 0.5 before pulse onset + 500 ms (so a crossing in that window is caused by the pulse)
  triggered  pulse trial crosses 0.5 within 500 ms of pulse onset (valid trials only)
  ttr        twin's own time to movement, twin RT - pulse onset (s); inf if the twin never moves
  svalid     twin moves and has not crossed 0.5 before the pulse ends (ttr kept down to 0)
  survive    (svalid trials) the pulse trial still crosses 0.5 at some point. The PNR of a pulse type at a delay is
             the twin time-to-movement at which P(survive) passes 0.5 (ttr50): pulses arriving later than that
             can no longer cancel the trial's own response.
Features read from the twin at the step before the pulse: output y; mean rate of D1, D2, cortex, thalamus, SNc;
D1 - D2; DA signal s; e_d1 / e_d2 traces (recomputed from s with the model's rise/fall constants; approximate,
ignores the per-step noise on x_nm); ramp progress (projection of all 65 unit rates on the cue-end -> pre-movement
axis of responding twins, 0 at cue end and 1 just before movement) and its velocity over the previous 100 ms.
Determinants are scored by AUC for 'triggered' and 'survive', pooled over delays and within each delay (time since cue fixed),
and by 5-fold cross-validated logistic regression (numpy IRLS, L2 1e-2) on delay only, state only, or both.
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
from model_functions import nln, exc

seed = int(sys.argv[1]); cond = sys.argv[2] if len(sys.argv) > 2 else 'intact'
PP = (os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl') if seed == 13
      else os.path.join(ROOT, 'seeds', '3k', f'seed_{seed}', 'params_nm.pkl'))
params = pickle.load(open(PP, 'rb'))
OUT = os.path.join(EXP, 'pnr_determinants', f'seed{seed}_{cond}')
os.makedirs(OUT, exist_ok=True)

T, N_BG, N_NM, N1 = cs.config['T'], cs.config['n_bg'], cs.config['n_nm'], cs.n_d1_cells
TAU_X, TAU_Z = cs.config['tau_x'], cs.config['tau_z']
START, DUR, WIN, NS = 300, 5, 50, int(os.environ.get('PNR_NS', 200))  # PNR_NS: smoke tests only
PT = np.arange(320, 600, 20)                       # pulse onsets, 0.2..2.8 s after the cue
DL = (PT - START) / 100
D1, D2, ALL = slice(0, N1), slice(N1, N_BG), slice(0, N_BG)
TYPES = [('stop −0.1', ALL, -0.1), ('stop −0.25', ALL, -0.25), ('stop −0.5', ALL, -0.5), ('stop −1', ALL, -1.0),
         ('stop −2.5', ALL, -2.5), ('stop −5', ALL, -5.0), ('stop −10', ALL, -10.0),
         ('dSPN −5', D1, -5.0), ('iSPN −1', D2, -1.0), ('iSPN −5', D2, -5.0),
         ('dSPN +0.25', D1, 0.25), ('dSPN +0.5', D1, 0.5), ('dSPN +1', D1, 1.0)]
nT, nP = len(TYPES), len(PT)
nC = 1 + nT * nP                                   # 0 = cue only twin; 1 + it*nP + ip = type it, pulse ip

stim = np.zeros((nC, T, N_BG), np.float32)
for it, (_, sl, a) in enumerate(TYPES):
    for ip, p in enumerate(PT):
        stim[1 + it * nP + ip, p:p + DUR, sl] = a
base, _, _ = mf.self_timed_movement_task(jnp.array([START]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)
inputs = jnp.repeat(base, nC, 0)
stim = jnp.asarray(stim)
if cond in ('intact', 'ctx_off'):
    bat = pm.make_batched(ctx_off=cond == 'ctx_off')
    sim = jax.jit(lambda k: bat(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim, cs.test_noise_std, jnp.repeat(k[None], nC, 0)))
else:
    bat = pm.make_batched(clamp=True)
    cl = jnp.full((T, N_NM), jnp.nan).at[START:].set(pm.OFF if cond == 'snc_off' else pm.ON)
    sim = jax.jit(lambda k: bat(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim, cs.test_noise_std, jnp.repeat(k[None], nC, 0), cl))

Y, R = np.zeros((NS, nC, T), np.float32), np.zeros((NS, T, 65), np.float32)
for s in range(NS):
    ys, xs, zs = sim(jr.PRNGKey(1000 + s))
    Y[s] = np.asarray(ys[..., 0])
    R[s] = np.asarray(jnp.concatenate([nln(xs[0][0]), nln(xs[1][0]), nln(xs[2][0]), nln(zs[0])], -1))
    if s == 0:
        print('first sim done', flush=True)


def first_cross(y, t0):
    above = y[..., t0:] > 0.5
    return np.where(above.any(-1), above.argmax(-1) + t0, np.inf)


rt = first_cross(Y, START)                         # (NS, nC) absolute index
rt0 = rt[:, 0]
mexc = np.asarray(exc(params['m'])).ravel(); c0 = float(np.ravel(params['c'])[0])
S = 1 / (1 + np.exp(-(R[..., -N_NM:] @ mexc + c0)))  # DA signal s(t) of the twin (from post-update x_nm)


def trace(sv, rise, fall):
    e = np.empty_like(sv); e[:, 0] = sv[:, 0]
    for t in range(1, sv.shape[1]):
        tgt = sv[:, t - 1]; tau = np.where(tgt > e[:, t - 1], rise, fall)
        e[:, t] = e[:, t - 1] + (tgt - e[:, t - 1]) / tau
    return np.clip(e, 0, 1)


E1, E2 = trace(S, 10.0, 15000.0), trace(1 - S, 20.0, 15000.0)
resp = np.isfinite(rt0)
r_ce = R[:, START + 10].mean(0)
pre = np.stack([R[s, int(rt0[s]) - 10] for s in np.where(resp)[0]]).mean(0) if resp.sum() > 5 else R[:, -1].mean(0)
axis = pre - r_ce
PROG = ((R - r_ce) @ axis) / (axis @ axis)          # (NS, T): 0 at cue end, 1 just before movement

FEATS = ['y', 'D1', 'D2', 'D1−D2', 'cortex', 'thalamus', 'SNc', 's', 'e_d1', 'e_d2', 'progress', 'velocity']


def feats(s, p):
    r = R[s, p - 1]
    return [Y[s, 0, p - 1], r[:N1].mean(), r[N1:N_BG].mean(), r[:N1].mean() - r[N1:N_BG].mean(), r[N_BG:N_BG + 20].mean(),
            r[N_BG + 20:N_BG + 40].mean(), r[N_BG + 40:].mean(), S[s, p - 1], E1[s, p - 1], E2[s, p - 1],
            PROG[s, p - 1], PROG[s, p - 1] - PROG[s, p - 11]]


def auc(x, yb):
    yb = yb.astype(bool); n1, n0 = yb.sum(), (~yb).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = np.argsort(np.argsort(x, kind='stable'), kind='stable') + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)  # average ties
    for v in np.where(cnt > 1)[0]:
        rk[inv == v] = rk[inv == v].mean()
    return float((rk[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_cv(X, yb, k=5, lam=1e-2):
    if yb.sum() < 10 or (~yb.astype(bool)).sum() < 10:
        return np.nan, None
    rng = np.random.default_rng(0); fold = rng.permutation(len(yb)) % k
    pred = np.zeros(len(yb))
    def fit(Xa, ya):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Z = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]; w = np.zeros(Z.shape[1])
        for _ in range(50):
            pz = 1 / (1 + np.exp(-Z @ w)); W = pz * (1 - pz) + 1e-6
            H = Z.T @ (Z * W[:, None]) + lam * len(ya) * np.eye(len(w)); H[0, 0] -= lam * len(ya)
            w = w + np.linalg.solve(H, Z.T @ (ya - pz) - lam * len(ya) * np.r_[0, w[1:]])
        return w, mu, sd
    for f in range(k):
        w, mu, sd = fit(X[fold != f], yb[fold != f])
        pred[fold == f] = np.c_[np.ones((fold == f).sum()), (X[fold == f] - mu) / sd] @ w
    w, _, _ = fit(X, yb)
    return auc(pred, yb), w[1:]


res = {'seed': seed, 'cond': cond, 'params': os.path.relpath(PP, ROOT), 'n_seeds': NS, 'delays_s': DL.tolist(),
       'twin_respond': float(resp.mean()), 'twin_rt_median_s': float(np.median((rt0[resp] - START) / 100)) if resp.any() else None,
       'types': {}}
rows = []                                          # one row per (seed, type, pulse) for the npz
for it, (name, _, amp) in enumerate(TYPES):
    trig_p, resp_p, valid_n, surv_p, sv_n, adv = [], [], [], [], [], []
    for ip, p in enumerate(PT):
        rp = rt[:, 1 + it * nP + ip]
        valid = rt0 >= p + WIN
        trig = valid & (rp < p + WIN)
        sval = np.isfinite(rt0) & (rt0 >= p + DUR)
        surv = sval & np.isfinite(rp)
        trig_p.append(float(trig.sum() / max(valid.sum(), 1))); valid_n.append(int(valid.sum()))
        resp_p.append(float(np.isfinite(rp).mean()))
        surv_p.append(float(surv.sum() / sval.sum()) if sval.any() else None); sv_n.append(int(sval.sum()))
        both = sval & np.isfinite(rp)
        adv.append(float(np.median((rp[both] - rt0[both]) / 100)) if both.any() else None)
        for s in range(NS):
            rows.append([it, ip, s, valid[s], trig[s], (rt0[s] - p) / 100, np.isfinite(rp[s]), sval[s], surv[s],
                         (rp[s] - rt0[s]) / 100 if np.isfinite(rp[s]) and np.isfinite(rt0[s]) else np.nan, *feats(s, p)])
    res['types'][name] = dict(amp=amp, p_triggered=trig_p, n_valid=valid_n, p_respond=resp_p,
                              p_survive=surv_p, n_survive_valid=sv_n, rt_shift_median_s=adv)
rows = np.array(rows, dtype=float)
COL = ['type', 'ip', 'seed', 'valid', 'trig', 'ttr', 'resp', 'svalid', 'surv', 'shift'] + FEATS
F0 = COL.index(FEATS[0]); ci = COL.index
TTB = np.arange(0, 3.01, 0.2)                      # ttr bins for survive curves


def ttr50(ttr, sv):
    """ttr at which P(survive) first reaches 0.5 going from late (small ttr) to early pulses; bins of 0.2 s, >= 8 trials."""
    b = np.digitize(ttr, TTB) - 1
    pts = [(TTB[i] + 0.1, sv[b == i].mean()) for i in range(len(TTB) - 1) if (b == i).sum() >= 8]
    if not pts:
        return None
    x, y = np.array(pts).T
    if y.min() >= 0.5:
        return 0.0                                 # never cancelled (pulse can't stop the response at any ttr)
    if y.max() < 0.5:
        return float('inf')
    for j in range(1, len(x)):                    # first downward crossing from short ttr towards long ttr
        if y[j - 1] >= 0.5 > y[j]:
            return float(x[j - 1] + (x[j] - x[j - 1]) * (y[j - 1] - 0.5) / (y[j - 1] - y[j]))
    return None


def score(yb, X, delay, ip, ttr=None):
    pooled = {f: auc(X[:, j], yb) for j, f in enumerate(FEATS)}
    pooled['delay'] = auc(delay, yb)
    within = {}
    cols = [(f, X[:, j]) for j, f in enumerate(FEATS)] + ([('ttr', -ttr)] if ttr is not None else [])
    for f, v in cols:
        a, w = [], []
        for k in range(nP):
            mk = ip == k; yk = yb[mk]
            if yk.sum() >= 10 and (~yk).sum() >= 10:
                a.append(auc(v[mk], yk)); w.append(mk.sum())
        within[f] = float(np.average(a, weights=w)) if a else None
    if ttr is not None:
        fin = np.isfinite(ttr); pooled['ttr'] = auc(-ttr[fin], yb[fin])
    cv = dict(delay=logit_cv(delay[:, None], yb)[0], state=logit_cv(X, yb)[0])
    a_both, w_both = logit_cv(np.c_[delay, X], yb); cv['delay+state'] = a_both
    if ttr is not None:
        cv['ttr'] = logit_cv(ttr[:, None], yb)[0]; cv['delay+ttr'] = logit_cv(np.c_[delay, ttr], yb)[0]
    return dict(n=int(len(yb)), n_pos=int(yb.sum()), auc_pooled=pooled, auc_within_delay=within, cv_auc=cv,
                logit_coef_std=dict(zip(['delay'] + FEATS, w_both.tolist())) if w_both is not None else None)


det, pnr = {}, {}
for it, (name, _, _) in enumerate(TYPES):
    m = (rows[:, 0] == it) & (rows[:, ci('valid')] == 1)
    det[name] = {'triggered': score(rows[m, ci('trig')].astype(bool), rows[m, F0:], DL[rows[m, 1].astype(int)], rows[m, 1].astype(int))}
    m = (rows[:, 0] == it) & (rows[:, ci('svalid')] == 1)
    det[name]['survive'] = score(rows[m, ci('surv')].astype(bool), rows[m, F0:], DL[rows[m, 1].astype(int)], rows[m, 1].astype(int), rows[m, ci('ttr')])
    pnr[name] = [ttr50(rows[m & (rows[:, 1] == k), ci('ttr')], rows[m & (rows[:, 1] == k), ci('surv')]) for k in range(nP)]
res['determinants'] = det
res['pnr_ttr50_s'] = pnr                           # per pulse type, per delay: twin time-to-movement where P(survive) = 0.5
# minimum amplitude (stop family / dSPN+ family) that triggers on >= 50% of valid trials, per delay
fam = {'stop': [n for n, _, _ in TYPES if n.startswith('stop')], 'dSPN+': [n for n, _, _ in TYPES if n.startswith('dSPN +')]}
res['min_amp_trigger50'] = {f: [next((abs(res['types'][n]['amp']) for n in ns if res['types'][n]['p_triggered'][k] >= 0.5), None)
                                 for k in range(nP)] for f, ns in fam.items()}
res['min_amp_cancel50'] = {f: [next((abs(res['types'][n]['amp']) for n in ns if (res['types'][n]['p_survive'][k] or 1) < 0.5), None)
                                for k in range(nP)] for f, ns in fam.items()}
# bistability: peak output from a stop −5 pulse to trial end (valid trials), fraction ambiguous (0.4–0.5) as in test_pnr_nm
it5 = [n for n, _, _ in TYPES].index('stop −5')
pk = np.concatenate([Y[rt0 >= p + WIN, 1 + it5 * nP + ip, p:].max(-1) for ip, p in enumerate(PT)])
pk0 = Y[:, 0, START:].max(-1)  # twin (cue only)
res['stop5_frac_peak_ambiguous_0p4_0p5'] = float(np.mean((pk > 0.4) & (pk <= 0.5)))
res['twin_frac_peak_ambiguous_0p4_0p5'] = float(np.mean((pk0 > 0.4) & (pk0 <= 0.5)))
json.dump(res, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'data.npz'), rows=rows, cols=np.array(COL), rt=rt, delays=DL,
                    types=np.array([n for n, _, _ in TYPES]), y_twin=Y[:, 0], prog=PROG, s=S, peak=pk, peak_twin=pk0,
                    y_stop5=Y[:, 1 + it5 * nP:1 + (it5 + 1) * nP], y_dspn1=Y[:, 1 + (nT - 1) * nP:])

# --- figure ---
fig, ax = plt.subplots(3, 3, figsize=(18, 15)); ax = ax.ravel()
stops = fam['stop']; others = [n for n, _, _ in TYPES if not n.startswith('stop')]
cs_ = plt.cm.viridis(np.linspace(0, 0.9, len(stops)))
for j, n in enumerate(stops):
    ax[0].plot(DL, res['types'][n]['p_triggered'], 'o-', ms=3, color=cs_[j], label=n)
for n in others:
    ax[1].plot(DL, res['types'][n]['p_triggered'], 'o--', ms=3, label=n)
for a, t in zip(ax[:2], ['stop pulses (all SPNs), by amplitude', 'population-specific pulses']):
    a.set(xlabel='pulse delay after cue (s)', ylabel='P(cross 0.5 within 500 ms)', title=f'triggered commitment: {t}', ylim=(-0.02, 1.02)); a.legend(fontsize=7)
hb = np.linspace(0, max(1.2, float(pk.max()) + 0.05), 49)
ax[2].hist(pk, hb, color='gray', density=True, label='stop −5 trials'); ax[2].hist(pk0, hb, histtype='step', color='k', density=True, label='twins (cue only)')
ax[2].axvline(0.5, color='k', ls=':'); ax[2].legend(fontsize=7)
ax[2].set(xlabel='peak output, pulse to trial end', ylabel='density', title=f'stop −5 peak output (ambiguous 0.4–0.5: {res["stop5_frac_peak_ambiguous_0p4_0p5"]:.2f})')
cmap = plt.cm.plasma(np.linspace(0, 0.9, nP))
def survive_panel(a, name, key='surv', ylab='P(response survives pulse)'):
    it = [n for n, _, _ in TYPES].index(name)
    m = (rows[:, 0] == it) & (rows[:, ci('svalid')] == 1)
    for k in range(nP):
        mk = m & (rows[:, 1] == k)
        b = np.digitize(rows[mk, ci('ttr')], TTB) - 1
        ok = [i for i in range(len(TTB) - 1) if (b == i).sum() >= 8]
        if len(ok) < 2:
            continue
        v = rows[mk, ci(key)]
        yv = [np.nanmedian(v[b == i]) if key == 'shift' else v[b == i].mean() for i in ok]
        a.plot(TTB[ok] + 0.1, yv, 'o-', ms=3, color=cmap[k], label=f'{DL[k]:.1f} s')
    a.set(xlabel="twin's time to movement at pulse onset (s)", ylabel=ylab, title=f'{name}: {ylab}\n(one line per pulse delay after cue)')
    if key == 'surv':
        a.set_ylim(-0.02, 1.02); a.axhline(0.5, color='k', lw=0.5)
    else:
        a.axhline(0, color='k', lw=0.5)
    a.legend(fontsize=6, ncol=2, title='delay', title_fontsize=7)
survive_panel(ax[3], 'stop −5'); survive_panel(ax[4], 'dSPN +1'); survive_panel(ax[5], 'stop −5', 'shift', 'median RT shift vs twin (s)')
keys = ['delay'] + FEATS + ['ttr']
xx = np.arange(len(keys))
sym = lambda v: np.nan if v is None or not np.isfinite(v) else max(v, 1 - v)
for off, (o, lab) in zip([-0.2, 0.2], [('triggered', 'triggered'), ('survive', 'survive')]):
    d = det['stop −5'][o]
    ax[6].bar(xx + off, [np.nan if k == 'delay' else sym(d['auc_within_delay'].get(k)) for k in keys], 0.4, label=f'{lab}, within delay')
ax[6].set_xticks(xx, keys, rotation=60, fontsize=7); ax[6].axhline(0.5, color='k', lw=0.5)
ax[6].set(ylim=(0.45, 1.0), ylabel='AUC, clock fixed (max(A, 1−A))', title='stop −5: what predicts the outcome at a fixed delay?'); ax[6].legend(fontsize=7)
cvn = ['delay', 'state', 'delay+state', 'ttr', 'delay+ttr']
for n, c in [('stop −5', 'k'), ('iSPN −5', 'tab:orange'), ('dSPN +1', 'tab:green')]:
    for o, ls in [('triggered', '-'), ('survive', ':')]:
        ax[7].plot(range(len(cvn)), [det[n][o]['cv_auc'].get(k, np.nan) for k in cvn], 'o', ls=ls, color=c, label=f'{n} {o}')
ax[7].set_xticks(range(len(cvn)), cvn); ax[7].set(ylim=(0.45, 1.0), ylabel='5-fold CV AUC', title='logistic: clock vs state vs own time-to-movement')
ax[7].legend(fontsize=6)
tt = (np.arange(T) - START) / 100
for k, c in zip([int(np.argmin(np.abs(DL - d))) for d in (1.0, 1.6, 2.2)], ['tab:blue', 'tab:orange', 'tab:red']):
    p = PT[k]; mk = (rows[:, 0] == it5) & (rows[:, 1] == k) & (rows[:, 3] == 1)
    sd = rows[mk, 2].astype(int); tr = rows[mk, 4].astype(bool)
    for sel, ls in [(tr, '-'), (~tr, ':')]:
        if sel.sum() >= 3:
            ax[8].plot(tt[START:p], PROG[sd[sel], START:p].mean(0), color=c, ls=ls, label=f'{DL[k]:.1f} s {"trig" if ls == "-" else "not"} (n={sel.sum()})')
ax[8].set(xlabel='time from cue (s)', ylabel='ramp progress of twin (0 cue end, 1 pre-movement)', title='twin ramp progress up to the pulse,\ntriggered vs not (stop −5)')
ax[8].legend(fontsize=6)
fig.suptitle(f'PNR determinants — seed {seed}{" (baseline)" if seed == 13 else ""}, {cond}; twin responds {resp.mean():.2f}')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pnr_determinants.png'), dpi=110)
for n in ['stop −5', 'iSPN −5', 'dSPN +1']:
    print(n, {o: {k: (None if v is None or not np.isfinite(v) else round(v, 3)) for k, v in det[n][o]['cv_auc'].items()} for o in det[n]})
print('stop −5 ttr50 by delay:', pnr['stop −5'])
print('min amp trigger50:', res['min_amp_trigger50'])
