"""How does SNc activity relate to critical slowing after a dSPN pulse?

usage (from repo root): python experiments/snc_critical_slowing.py [net]   net: baseline (default) | pd_acute | hd
Output: experiments/<network folder>/snc_critical_slowing/ (metrics_snc_critical_slowing.json, trials.csv, figures).

Same pulse protocol as run_experiment.py dspn_pnr (+1 into dSPNs for 50 ms at 0.2-2.8 s after a cue at t=300,
150 noise seeds, every condition of a seed shares its key so condition 1 is the pulse trial's exact twin).
The pulse is repeated under SNc manipulations applied identically to pulse trials and twins:
  intact      no clamp
  open_loop   SNc follows the intact twin's own trajectory (clamp[t] = twin x_nm[t-1]), so the pulse cannot
              propagate through SNc but s(t) is unchanged; drift of the clamped twin from the intact twin is reported
  frozen      SNc held from the cue at the intact twins' mean post-cue state (average gain, no dynamics)
  snc_off     SNc state -10 (rate 0) from the cue; s = sigmoid(c)
  snc_sat     SNc state +10 (rate ~1) from the cue
Analyses: (1) critical-slowing metrics per pulse delay; (2) the same aligned to time-to-movement (twin RT - pulse);
(3) trial-by-trial regression of the residual on s at the pulse and time-to-RT (intact); (4) the distance to the twin
split by area plus |delta s|; (5) leading eigenvalue of the noise-free one-step Jacobian along intact twin trajectories,
full system vs. SNc held fixed (non-SNc block), with the e_d1/e_d2 traces' memory held fixed.
"""
import os, sys, json, csv, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')
sys.path[:0] = [EXP, ROOT]
net = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
NETDIR = {'baseline': 'baseline', 'pd_acute': 'pd_snc_lesion/acute', 'pd': 'pd_snc_lesion/retrained',
          'hd_acute': 'hd_ispn_lesion/acute', 'hd': 'hd_ispn_lesion/retrained'}[net]
OUT = os.path.join(EXP, NETDIR, 'snc_critical_slowing')
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
from scipy import stats
import config_script as cs
import model_functions as mf
import perturb_model as pm
from model_functions import nln, exc, inh, _asymmetric_trace_update

TAU_X, TAU_Z, T = cs.config['tau_x'], cs.config['tau_z'], cs.config['T']
N_BG, N_NM, N1 = cs.config['n_bg'], cs.config['n_nm'], cs.n_d1_cells
BASE = os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl')
params = pickle.load(open(BASE if net in ('baseline', 'pd_acute', 'hd_acute') else
                          os.path.join(EXP, NETDIR.split('/')[0], 'training', 'params_nm.pkl'), 'rb'))
les_bg = les_nm = None
if net != 'baseline':
    idx = json.load(open(os.path.join(EXP, NETDIR.split('/')[0], 'lesion.json')))['lesioned_idx']
    if net.startswith('pd'):
        les_nm = jnp.zeros(N_NM, bool).at[jnp.array(idx)].set(True)
    else:
        les_bg = jnp.zeros(N_BG, bool).at[jnp.array(idx)].set(True)
batched_clamp = pm.make_batched(les_bg, les_nm, clamp=True)

START, DUR, AMP, NS, WIN = 300, 5, 1.0, 150, 50
POST = 150                                    # steps of d(t) kept after each pulse onset
ptimes = np.arange(320, 600, 20)
nP, nC = len(ptimes), len(ptimes) + 2
base, _, _ = mf.self_timed_movement_task(jnp.array([START]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)
stim = np.zeros((nC, T, N_BG), np.float32)
for i, p in enumerate(ptimes):
    stim[i + 2, p:p + DUR, :N1] = AMP
stim = jnp.asarray(stim)
inputs = jnp.repeat(base, nC, 0).at[0].set(0.0)
sim = jax.jit(lambda keys, cl: batched_clamp(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim, cs.test_noise_std, keys, cl))
SL = {'D1': slice(0, N1), 'D2': slice(N1, N_BG), 'Cortex': slice(20, 40), 'Thalamus': slice(40, 60), 'SNc': slice(60, 65)}
s_fn = lambda z: jax.nn.sigmoid(nln(z) @ exc(params['m']).T + params['c'])[..., 0]


def first_cross(y, t0):
    above = y[..., t0:] > 0.5
    return np.where(above.any(-1), above.argmax(-1) + t0, np.inf)


def run(cond, clamp_fn):
    """Simulate every seed under one condition; per-trial metrics and post-pulse distance curves."""
    trials, dcurves, twin = [], np.full((NS, nP, 6, POST), np.nan), {}
    for s in range(NS):
        ys, xs, zs = sim(jnp.repeat(jr.PRNGKey(s + 200)[None], nC, 0), clamp_fn(s))
        r = jnp.concatenate([nln(xs[0]), nln(xs[1]), nln(xs[2]), nln(zs)], -1)          # (nC, T, 65)
        diff = np.asarray(r[2:] - r[1:2])                                                # (nP, T, 65)
        ds = np.abs(np.asarray(s_fn(zs[2:]) - s_fn(zs[1:2])))                            # (nP, T)
        D = np.linalg.norm(diff, axis=-1)
        y = np.asarray(ys[..., 0]); rt = first_cross(y, START)
        sv = np.asarray(s_fn(zs[1]))
        twin[s] = dict(y=y[1], xnm=np.asarray(zs[1]), x=[np.asarray(x[1]) for x in xs], s=sv, rt=rt[1])
        rc = rt[1]
        for i, p in enumerate(ptimes):
            rp = rt[i + 2]
            row = dict(cond=cond, seed=s, delay_s=(p - START) / 100, twin_rt_s=(rc - START) / 100,
                       pulse_rt_s=(rp - START) / 100, valid=bool(rc > p), ttr_s=(rc - p) / 100,
                       s_pulse=float(sv[p - 1]), snc_rate_pulse=float(np.asarray(nln(zs[1, p - 1])).mean()),
                       y_pulse=float(y[1, p - 1]), triggered=bool(p <= rp < p + WIN and not p <= rc < p + WIN),
                       residual=np.nan, tau_ms=np.nan, censored=False, short=False)
            if row['valid']:
                end = int(min(rc, rp, p + DUR + 300, T)) - 30
                if end - p < DUR + 30:
                    row['short'] = True
                else:
                    d = D[i, p:end]; k0 = int(np.argmax(d[:DUR + 20]))
                    if end - p > DUR + 50:
                        row['residual'] = float(d[DUR + 50] / d[k0])
                    below = np.where(d[k0:] < d[k0] / np.e)[0]
                    if len(below):
                        row['tau_ms'] = float((k0 + below[0] - DUR) * 10)
                    else:
                        row['censored'] = True
                    n = min(end - p, POST)
                    for j, a in enumerate(SL):
                        dcurves[s, i, j, :n] = np.linalg.norm(diff[i, p:p + n, SL[a]], axis=-1) / d[k0]
                    dcurves[s, i, 5, :n] = ds[i, p:p + n] / max(ds[i, p:p + DUR + 20].max(), 1e-12)
            trials.append(row)
    return trials, dcurves, twin


nanclamp = jnp.full((T, N_NM), jnp.nan)
COND = {}
print('intact'); COND['intact'] = run('intact', lambda s: nanclamp)
TW = COND['intact'][2]
OPEN = {s: jnp.asarray(np.concatenate([np.full((1, N_NM), np.nan), TW[s]['xnm'][:-1]])) for s in TW}
print('open_loop'); COND['open_loop'] = run('open_loop', lambda s: OPEN[s])
frozen_val = np.mean([TW[s]['xnm'][START + 20:START + 250].mean(0) for s in TW], 0)
FROZEN = nanclamp.at[START:].set(jnp.asarray(frozen_val))
print('frozen'); COND['frozen'] = run('frozen', lambda s: FROZEN)
OFF = nanclamp.at[START:].set(pm.OFF); ON = nanclamp.at[START:].set(pm.ON)
print('snc_off'); COND['snc_off'] = run('snc_off', lambda s: OFF)
print('snc_sat'); COND['snc_sat'] = run('snc_sat', lambda s: ON)

# open-loop sanity: how far does the clamped twin drift from the intact twin?
ol = COND['open_loop'][2]
drift = dict(max_abs_dy_median=float(np.median([np.max(np.abs(ol[s]['y'] - TW[s]['y'])[:int(min(TW[s]['rt'], T))]) for s in TW])),
             rt_agree_within_50ms=float(np.mean([(np.isinf(ol[s]['rt']) and np.isinf(TW[s]['rt'])) or abs(ol[s]['rt'] - TW[s]['rt']) <= 5 for s in TW])),
             s_max_abs_diff_median=float(np.median([np.max(np.abs(ol[s]['s'] - TW[s]['s'])) for s in TW])))

with open(os.path.join(OUT, 'trials.csv'), 'w', newline='') as fh:
    keys = list(COND['intact'][0][0].keys())
    w = csv.DictWriter(fh, keys); w.writeheader()
    for c in COND:
        w.writerows(COND[c][0])


def agg(rows):
    v = [r for r in rows if r['valid']]
    res = np.array([r['residual'] for r in v]); res = res[np.isfinite(res)]
    meas = [r for r in v if not r['short']]
    tau = np.array([r['tau_ms'] for r in meas]); tau = tau[np.isfinite(tau)]
    return dict(n_valid=len(v), n_residual=int(len(res)),
                residual_median=float(np.median(res)) if len(res) else None,
                p_grows=float(np.mean(res > 1)) if len(res) else None,
                frac_censored=float(np.mean([r['censored'] for r in meas])) if meas else None,
                tau_ms_median=float(np.median(tau)) if len(tau) else None,
                p_triggered=float(np.mean([r['triggered'] for r in v])) if v else None)


# (1) by pulse delay
by_delay = {c: [] for c in COND}
for c in COND:
    T_ = COND[c][0]
    for p in ptimes:
        d = (p - START) / 100
        rows = [r for r in T_ if r['delay_s'] == d]
        twin_resp = np.mean([np.isfinite(r['twin_rt_s']) for r in rows])
        by_delay[c].append(dict(delay_s=d, twin_respond=float(twin_resp), **agg(rows)))
resp = {c: dict(respond=float(np.mean([np.isfinite(COND[c][2][s]['rt']) for s in range(NS)])),
                rt_median_s=float(np.median([(COND[c][2][s]['rt'] - START) / 100 for s in range(NS) if np.isfinite(COND[c][2][s]['rt'])]))
                if any(np.isfinite(COND[c][2][s]['rt']) for s in range(NS)) else None,
                s_after_cue=float(np.mean([COND[c][2][s]['s'][START + 50:START + 250].mean() for s in range(NS)]))) for c in COND}

# (2) by time to movement (twin RT - pulse onset); 'never' = twin never moves
TB = np.arange(0.3, 3.31, 0.3)
by_ttr = {}
for c in COND:
    v = [r for r in COND[c][0] if r['valid']]
    out = []
    for lo, hi in zip(TB[:-1], TB[1:]):
        out.append(dict(ttr_lo=float(lo), ttr_hi=float(hi), **agg([r for r in v if lo <= r['ttr_s'] < hi])))
    nev = [r for r in v if np.isinf(r['ttr_s'])]
    out.append(dict(ttr_lo='never', **agg(nev), never_by_delay={f"{d:.1f}": agg([r for r in nev if r['delay_s'] == d])['residual_median']
                                                              for d in (ptimes - START) / 100}))
    by_ttr[c] = out

# (3) regression on intact trials, delays 1.4-2.2 s, twin moves later and residual measured
reg = {}
v = [r for r in COND['intact'][0] if r['valid'] and np.isfinite(r['residual']) and np.isfinite(r['ttr_s']) and 1.35 < r['delay_s'] < 2.25]
if len(v) > 20:
    yv = np.log(np.array([r['residual'] for r in v]))
    X = {k: np.array([r[k] for r in v]) for k in ('s_pulse', 'snc_rate_pulse', 'ttr_s', 'y_pulse')}
    dl = np.array([r['delay_s'] for r in v])
    Z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    dummies = np.stack([(dl == d).astype(float) for d in np.unique(dl)], 1)
    def ols(cols):
        A = np.concatenate([np.stack([Z(X[k]) for k in cols], 1), dummies], 1)
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            ix = rng.integers(0, len(yv), len(yv))
            bs.append(np.linalg.lstsq(A[ix], yv[ix], rcond=None)[0][:len(cols)])
        bs = np.array(bs)
        return {k: dict(beta=float(beta[j]), ci95=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j, k in enumerate(cols)}
    reg = dict(n=len(v), outcome='log(residual_500ms), delay fixed effects, standardized predictors, bootstrap CI',
               s_only=ols(['s_pulse']), ttr_only=ols(['ttr_s']), s_and_ttr=ols(['s_pulse', 'ttr_s']),
               s_ttr_y=ols(['s_pulse', 'ttr_s', 'y_pulse']),
               spearman_s_vs_ttr=float(stats.spearmanr(X['s_pulse'], X['ttr_s'])[0]))

# (4) area decomposition: mean normalized per-area distance curves (intact)
dc = COND['intact'][1]
area_share = {}
for i, p in enumerate(ptimes):
    m = np.nanmean(dc[:, i], 0)                                   # (6, POST)
    area_share[f'{(p - START) / 100:.1f}'] = {a: [float(m[j, k]) if np.isfinite(m[j, k]) else None for k in (10, 30, 55)]
                                             for j, a in enumerate(list(SL) + ['|ds|'])}


# (5) Jacobian of the noise-free step along intact twin trajectories
def step(z, e1p, e2p, u=None):
    x_bg, x_c, x_t, x_nm = z[:20], z[20:40], z[40:60], z[60:]
    s = jax.nn.sigmoid(exc(params['m']) @ nln(x_nm) + params['c'])[0]
    e1 = jnp.clip(_asymmetric_trace_update(e1p, s, 10.0, 15000.0), 0, 1)
    e2 = jnp.clip(_asymmetric_trace_update(e2p, 1 - s, 20.0, 15000.0), 0, 1)
    U = jnp.concatenate((jnp.ones(N1), -jnp.ones(N_BG - N1)))[:, None]
    G = jnp.exp(s * U)
    pg = jnp.concatenate((jnp.ones(N1) * (1 + e1), jnp.ones(N_BG - N1) * (1 + e2)))
    xc = (1 - 1 / TAU_X) * x_c + (1 / TAU_X) * (params['J_c'] @ nln(x_c) + exc(params['B_ct']) @ nln(x_t))
    if u is not None:
        xc = xc + (1 / TAU_X) * params['B_cu'] @ u
    xb = (1 - 1 / TAU_X) * x_bg + (1 / TAU_X) * pg * ((G * inh(params['J_bg'])) @ nln(x_bg) + (G * exc(params['B_bgc'])) @ nln(xc))
    xn = (1 - 1 / TAU_Z) * x_nm + (1 / TAU_Z) * (params['J_nm'] @ nln(x_nm) + exc(params['B_nmc']) @ nln(xc) + inh(params['B_nmbg']) @ nln(xb))
    tbg = jnp.concatenate((exc(params['B_tbg'][:, :N1]), inh(params['B_tbg'][:, N1:])), axis=1)
    xt = (1 - 1 / TAU_X) * x_t + (1 / TAU_X) * (params['J_t'] @ nln(x_t) + tbg @ nln(xb))
    return jnp.concatenate([xb, xc, xt, xn]), e1, e2


def e_traj(Z):
    """e_d1, e_d2 entering the step from Z[t] to Z[t+1], rebuilt from the state trajectory. The traces fall with
    tau 15000, so their start matters: e after step 0 = s of the initial SNc state cs.z0 (its init noise is ignored)."""
    s0 = jax.nn.sigmoid(exc(params['m']) @ nln(cs.z0) + params['c'])[0]
    def f(c, z):
        s = jax.nn.sigmoid(exc(params['m']) @ nln(z[60:]) + params['c'])[0]
        e1 = jnp.clip(_asymmetric_trace_update(c[0], s, 10.0, 15000.0), 0, 1)
        e2 = jnp.clip(_asymmetric_trace_update(c[1], 1 - s, 20.0, 15000.0), 0, 1)
        return (e1, e2), (c[0], c[1])
    _, (a, b) = jax.lax.scan(f, (s0, 1 - s0), Z)
    return a, b


# verify the step against the hooked model with noise off (no lesion masks in step(), so baseline only)
jac_ok = None
if net == 'baseline':
    ysv, xsv, zsv = pm.make_batched()(params, cs.x0, cs.z0, base, TAU_X, TAU_Z, True, None, 0.0, jr.split(jr.PRNGKey(0), 1))
    Zv = jnp.concatenate([xsv[0][0], xsv[1][0], xsv[2][0], zsv[0]], -1)
    e1v, e2v = e_traj(Zv)
    nxt = jax.vmap(lambda z, a, b, u: step(z, a, b, u)[0])(Zv[:-1], e1v[:-1], e2v[:-1], base[0, 1:])
    jac_ok = float(jnp.max(jnp.abs(nxt - Zv[1:])))
jac = jax.jit(jax.vmap(jax.jacfwd(lambda z, a, b: step(z, a, b)[0])))
alive = np.ones(65, bool)
if les_bg is not None: alive[:20] &= ~np.asarray(les_bg)
if les_nm is not None: alive[60:] &= ~np.asarray(les_nm)
NJ, TJ = 40, np.arange(START - 50, START + 600, 5)
lam = dict(full=np.full((NJ, len(TJ)), np.nan), no_snc=np.full((NJ, len(TJ)), np.nan), snc_only=np.full((NJ, len(TJ)), np.nan))
twin_rt_j = np.array([TW[s]['rt'] for s in range(NJ)])
for s in range(NJ):
    Z_ = jnp.concatenate([jnp.asarray(TW[s]['x'][0]), jnp.asarray(TW[s]['x'][1]), jnp.asarray(TW[s]['x'][2]), jnp.asarray(TW[s]['xnm'])], -1)
    e1, e2 = e_traj(Z_)
    Js = np.asarray(jac(Z_[TJ - 1], e1[TJ - 1], e2[TJ - 1]))[:, alive][:, :, alive]
    na = int(alive[:60].sum())
    for k, sub in [('full', slice(None)), ('no_snc', slice(0, na)), ('snc_only', slice(na, None))]:
        J_ = Js[:, sub, sub]
        if J_.shape[-1]:
            lam[k][s] = np.max(np.abs(np.linalg.eigvals(J_)), -1)
    lam_valid = TJ[None, :] < twin_rt_j[s] - 20
    for k in lam:
        lam[k][s, ~lam_valid[0]] = np.nan
tau_eff = {k: -10 / np.log(np.clip(v, 1e-6, 1 - 1e-9)) for k, v in lam.items()}  # ms; > 0 only while |lambda| < 1
jac_summary = {k: dict(delay_s=[float(t) for t in (TJ[::10] - START) / 100],
                       lambda_max_median=[float(np.nanmedian(lam[k][:, j])) if np.isfinite(lam[k][:, j]).any() else None for j in range(0, len(TJ), 10)])
               for k in lam}
# Jacobian aligned to the twin's own RT
RB = np.arange(-3.0, -0.19, 0.2)
lam_ttr = {k: [] for k in lam}
for k in lam:
    rel = (TJ[None, :] - twin_rt_j[:, None]) / 100
    for lo, hi in zip(RB[:-1], RB[1:]):
        vv = lam[k][(rel >= lo) & (rel < hi)]
        vv = vv[np.isfinite(vv)]
        lam_ttr[k].append(float(np.median(vv)) if len(vv) else None)

json.dump(dict(net=net, definition=__doc__, open_loop_drift=drift, jacobian_step_check_max_abs=jac_ok,
               frozen_snc_state=[float(v) for v in frozen_val], respond=resp, by_delay=by_delay, by_time_to_rt=by_ttr,
               regression=reg, area_share_at_100_300_550ms_from_onset=area_share,
               jacobian=dict(note='max |eigenvalue| of the one-step map (dt=10 ms); 1 = marginal; tau_eff = -dt/ln|lambda|',
                             by_delay=jac_summary, by_time_to_rt=dict(bins_s=[float(b) for b in RB], **lam_ttr))),
          open(os.path.join(OUT, 'metrics_snc_critical_slowing.json'), 'w'), indent=1, default=float)

# ---------------- figures
col = dict(intact='k', open_loop='tab:purple', frozen='tab:green', snc_off='tab:blue', snc_sat='tab:red')
lab = dict(intact='intact', open_loop='SNc open-loop (twin\'s trajectory)', frozen='SNc frozen at mean', snc_off='SNc off', snc_sat='SNc saturated')
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
dl = (ptimes - START) / 100
for c in COND:
    b = by_delay[c]
    g = lambda k: [np.nan if r[k] is None else r[k] for r in b]
    ax[0].plot(dl, g('residual_median'), 'o-', color=col[c], label=f"{lab[c]} (respond {resp[c]['respond']:.2f})")
    ax[1].plot(dl, g('frac_censored'), 'o-', color=col[c])
    tt = by_ttr[c][:-1]
    mid = [(r['ttr_lo'] + r['ttr_hi']) / 2 for r in tt]
    ax[3].plot(mid, [np.nan if r['residual_median'] is None or r['n_residual'] < 10 else r['residual_median'] for r in tt], 'o-', color=col[c])
    nv = by_ttr[c][-1]
    if nv['residual_median'] is not None and nv['n_residual'] >= 10:
        ax[3].scatter([3.45], [nv['residual_median']], color=col[c], marker='s')
    ax[4].plot(mid, [np.nan if r['frac_censored'] is None or r['n_valid'] < 10 else r['frac_censored'] for r in tt], 'o-', color=col[c])
ax[0].axhline(1, color='gray', ls=':'); ax[0].set(xlabel='pulse delay after cue (s)', ylabel='d(500 ms)/peak, median', title='residual perturbation by pulse time', yscale='log')
ax[0].legend(fontsize=7)
ax[1].set(xlabel='pulse delay after cue (s)', ylabel='fraction never decaying to 1/e', title='censored fraction by pulse time')
ax[3].axhline(1, color='gray', ls=':'); ax[3].set(xlabel='twin RT − pulse onset (s)   [■ = twin never moves]', ylabel='d(500 ms)/peak, median', title='residual by time to movement', yscale='log')
ax[3].invert_xaxis()
ax[4].set(xlabel='twin RT − pulse onset (s)', ylabel='fraction never decaying to 1/e', title='censored by time to movement'); ax[4].invert_xaxis()
# s and SNc along the intact twin, with residual
sv = np.stack([TW[s]['s'] for s in range(NS)])
tt = (np.arange(T) - START) / 100
ax[2].plot(tt, sv.mean(0), 'k', label='s (intact twin)')
for c in ('snc_off', 'snc_sat', 'frozen'):
    ax[2].axhline(resp[c]['s_after_cue'], color=col[c], ls='--', lw=0.8, label=f'{lab[c]}: s')
ax[2].set(xlim=(-0.5, 3.5), xlabel='time from cue (s)', ylabel='DA signal s', title='DA level in each condition'); ax[2].legend(fontsize=7)
for k, c_ in [('full', 'k'), ('no_snc', 'tab:orange'), ('snc_only', 'tab:cyan')]:
    mid = (RB[:-1] + RB[1:]) / 2
    ax[5].plot(mid, [np.nan if v is None else v for v in lam_ttr[k]], 'o-', color=c_, label={'full': 'full system', 'no_snc': 'SNc held fixed', 'snc_only': 'SNc block alone'}[k])
ax[5].axhline(1, color='gray', ls=':'); ax[5].axhline(1 - 1 / TAU_Z, color='tab:cyan', ls=':', lw=0.7)
ax[5].set(xlabel='time relative to twin RT (s)', ylabel='max |eigenvalue| of step Jacobian', title='local stability (intact twins)'); ax[5].legend(fontsize=7)
fig.suptitle(f'{net}: SNc and critical slowing after a dSPN pulse (+1, 50 ms)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'snc_critical_slowing.png'), dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, d in zip(ax, ['0.6', '1.6', '2.0']):
    i = list(np.round(dl, 1)).index(float(d))
    m = np.nanmean(dc[:, i], 0)
    for j, nm in enumerate(list(SL) + ['|Δs| (norm.)']):
        a.plot(np.arange(POST) * 10 - DUR * 10, m[j], label=nm, lw=1.5 if j < 5 else 1, ls='-' if j < 5 else '--')
    a.set(title=f'pulse at {d} s: distance to twin by area', xlabel='ms from pulse end', ylabel='‖Δrate‖ / peak total d')
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'area_decomposition.png'), dpi=130)
print(json.dumps(dict(drift=drift, jac_ok=jac_ok, respond=resp, regression=reg), indent=1, default=float))
