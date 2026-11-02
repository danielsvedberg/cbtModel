"""Does SNc perturbation temporally scale the dynamics, and what in the circuit makes it do so?

usage (from repo root): python analysis/seed9_scaling/scaling_analysis.py <seed> [<seed> ...]   (seeds/2k networks)
Output: analysis/seed9_scaling/out/seed_<k>/ (metrics.json + figures). All runs use hooks.py (verified bit-identical
to jit(mf.batched_nm_rnn) with hooks off). Cue at t=300 (cue onset = time 0 below).

Timing measure: noise-free trajectories peak just under the 0.5 threshold, so speed is read from the trajectory:
t80 = first time y covers 80% of the rise from its pre-cue level to the control peak.

A. Sustained SNc input (from cue offset to trial end, strengths SUST) -- scaling vs shift:
   - warp fits on the 60 non-SNc unit rates over [cue, perturbed t80]: none / shift / scale (about cue onset) / both
   - path test: distance from each perturbed state to the nearest control-path state, vs the same-time distance
   - 1000 paired noisy trials (same keys): RT_pert vs RT_ctrl regression (scaling: slope != 1, intercept ~ 0)
B. Open-loop gain clamp (s overridden from cue onset): constant-g sweep, g frozen at its pre-cue value, control g(t)
   replayed (sanity), and the pathway split (gain on BG recurrence only / on cortex->BG input only).
C. Flow along the control path with g held fixed: for each g, v_g(x) = f_g(x) - x at control states; speed ratio
   |v_g| / |v_ctrl| and direction cosine (pure speed knob: cosine ~ 1, ratio varies with g); leading |eigenvalue| of
   the 60x60 one-step Jacobian.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
from scipy import stats
import hooks
from hooks import T, N_BG, N_NM

CUE = 300
CUE_END = CUE + 10
SUST = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
GS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.4, 2.7]
N_NOISY = 1000
ALPHAS = np.round(np.arange(0.3, 3.01, 0.01), 2)
DELTAS = np.arange(-200, 201, 5)
AREAS = {'dSPN': slice(0, 10), 'iSPN': slice(10, 20), 'Cortex': slice(20, 40), 'Thalamus': slice(40, 60)}
NAN = np.nan


def sim(p, n=1, nm_in=None, s_clamp=None, g_rec=None, g_ctx=None, noise=0.0, key=0):
    stim, nmi, sc, gr, gc = hooks.free_hooks(n)
    nmi = nmi if nm_in is None else jnp.broadcast_to(jnp.asarray(nm_in, jnp.float32), (n, T, N_NM))
    sc = sc if s_clamp is None else jnp.broadcast_to(jnp.asarray(s_clamp, jnp.float32), (n, T, 2))
    gr = gr if g_rec is None else jnp.broadcast_to(jnp.asarray(g_rec, jnp.float32), (n, T))
    gc = gc if g_ctx is None else jnp.broadcast_to(jnp.asarray(g_ctx, jnp.float32), (n, T, 2))
    out = hooks.run_batch(p, hooks.cue_inputs(n, CUE), stim, nmi, sc, gr, gc, noise, jr.split(jr.PRNGKey(key), n))
    y, xbg, xc, xt, xnm, s, grr, gcc = out
    return dict(y=np.asarray(y[..., 0]), x=np.asarray(jnp.concatenate([xbg, xc, xt], -1)), xnm=np.asarray(xnm),
                s=np.asarray(s), g_rec=np.asarray(grr), g_ctx=np.asarray(gcc))


def step_profile(v, start):
    a = np.full(T, np.nan, np.float32)
    a[start:] = v
    return a


def nm_profile(v):
    a = np.zeros((T, N_NM), np.float32)
    a[CUE_END:] = v
    return a


def t80(y, ctrl_peak, y0):
    thr = y0 + 0.8 * (ctrl_peak - y0)
    above = y[CUE:] >= thr
    return float(above.argmax()) if above.any() else np.nan  # steps after cue onset


def warp_fits(rc, rp, t_end):
    """rc, rp: (T, 60) rates; fit rp(t) ~ rc(CUE + (t - CUE - d) / a) over t in [CUE, CUE + t_end]."""
    t = np.arange(CUE, CUE + int(t_end) + 1)
    Y = rp[t]
    sst = ((Y - Y.mean(0)) ** 2).sum()
    best = {}
    for name, A, D in [('none', [1.0], [0]), ('shift', [1.0], DELTAS), ('scale', ALPHAS, [0]), ('scale+shift', ALPHAS, DELTAS)]:
        bb = (np.inf, None, None)
        for a in A:
            for d in D:
                tau = np.clip(CUE + (t - CUE - d) / a, 0, T - 1)
                i0 = np.floor(tau).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (tau - i0)[:, None]
                sse = ((Y - ((1 - w) * rc[i0] + w * rc[i1])) ** 2).sum()
                if sse < bb[0]:
                    bb = (sse, a, d)
        best[name] = dict(r2=float(1 - bb[0] / sst), alpha=float(bb[1]), shift_steps=float(bb[2]))
    return best


def path_test(rc, rp, t_end_c, t_end_p):
    P = rc[CUE:CUE + int(t_end_c) + 60]
    Q = rp[CUE:CUE + int(t_end_p) + 1]
    nearest = np.sqrt(((Q[:, None] - P[None]) ** 2).sum(-1)).min(1)
    n = min(len(Q), T - CUE)
    same_t = np.sqrt(((Q[:n] - rc[CUE:CUE + n]) ** 2).sum(-1))
    extent = np.sqrt(((P - P[0]) ** 2).sum(-1)).max()
    return dict(nearest_median=float(np.median(nearest)), same_time_median=float(np.median(same_t)),
                nearest_over_extent=float(np.median(nearest) / extent), same_time_over_extent=float(np.median(same_t) / extent))


def rts(y):
    a = y[:, CUE_END:] > 0.5
    return np.where(a.any(1), a.argmax(1) + 10, np.nan) / 100.0  # s after cue onset


def rt_regression(rc, rp):
    both = ~np.isnan(rc) & ~np.isnan(rp)
    out = dict(respond_ctrl=float(np.mean(~np.isnan(rc))), respond_pert=float(np.mean(~np.isnan(rp))), n_both=int(both.sum()))
    if both.sum() >= 20 and np.std(rc[both]) > 0:
        r = stats.linregress(rc[both], rp[both])
        out.update(slope=float(r.slope), slope_se=float(r.stderr), intercept=float(r.intercept),
                   intercept_se=float(r.intercept_stderr), ratio_median=float(np.median(rp[both] / rc[both])))
    return out


def main(seed):
    OUT = os.path.join(HERE, 'out', f'seed_{seed}')
    os.makedirs(OUT, exist_ok=True)
    p = pickle.load(open(os.path.join(ROOT, 'seeds', '2k', f'seed_{seed}', 'params_nm.pkl'), 'rb'))
    M = dict(seed=seed, verify_max_abs_diff=hooks.verify(p))
    assert M['verify_max_abs_diff'] == 0.0
    rate = lambda x: np.maximum(0, np.tanh(x))

    ctrl = sim(p)
    yc, rc = ctrl['y'][0], rate(ctrl['x'][0])
    y0 = float(yc[CUE - 50:CUE].mean())
    peak_c = float(yc[CUE_END:].max())
    t80c = t80(yc, peak_c, y0)
    g_c = np.exp(ctrl['s'][0])
    g0 = float(g_c[CUE - 50:CUE].mean())
    M['control'] = dict(y_pre=y0, y_peak=peak_c, t80_s=t80c / 100, g_pre=g0, g_max=float(g_c[CUE:].max()),
                        s_pre=float(ctrl['s'][0, CUE - 50:CUE].mean()), snc_rate_pre=float(rate(ctrl['xnm'][0, CUE - 50:CUE]).mean()))

    # ---------- A. sustained SNc input ----------
    keysN = 0
    rt_c = rts(sim(p, N_NOISY, noise=0.05, key=keysN)['y'])
    A, trajA = [], {}
    for v in SUST:
        r = sim(p, nm_in=nm_profile(v))
        yp, rp = r['y'][0], rate(r['x'][0])
        tp = t80(yp, peak_c, y0)
        row = dict(strength=v, t80_s=tp / 100, t80_ratio=tp / t80c, g_mean_cue_to_t80=float(np.exp(r['s'][0, CUE:CUE + int(np.nan_to_num(tp, nan=T - CUE - 1))]).mean()))
        if not np.isnan(tp) and v != 0:
            row['warp'] = warp_fits(rc, rp, tp)
            row['path'] = path_test(rc, rp, t80c, tp)
        rt_p = rts(sim(p, N_NOISY, nm_in=nm_profile(v), noise=0.05, key=keysN)['y'])
        row['rt'] = rt_regression(rt_c, rt_p)
        A.append(row)
        trajA[v] = (yp, rp, np.exp(r['s'][0]), tp)
    M['sustained_snc'] = A

    # ---------- B. open-loop gain clamp ----------
    B = dict(constant=[], rec_only=[], ctx_only=[])
    trajB = {}
    for g in GS:
        for kind in B:
            kw = dict(constant=dict(g_rec=step_profile(g, CUE), g_ctx=step_profile(g, CUE)),
                      rec_only=dict(g_rec=step_profile(g, CUE), g_ctx=step_profile(g0, CUE)),
                      ctx_only=dict(g_rec=step_profile(g0, CUE), g_ctx=step_profile(g, CUE)))[kind]
            r = sim(p, **kw)
            tp = t80(r['y'][0], peak_c, y0)
            row = dict(g=g, t80_s=tp / 100, t80_ratio=tp / t80c, y_peak=float(r['y'][0, CUE_END:].max()))
            if kind == 'constant':
                if not np.isnan(tp):
                    rp = rate(r['x'][0])
                    row['warp'] = warp_fits(rc, rp, tp)
                    row['path'] = path_test(rc, rp, t80c, tp)
                rt_p = rts(sim(p, 500, noise=0.05, key=keysN, **kw)['y'])
                row['respond'] = float(np.mean(~np.isnan(rt_p)))
                row['rt_median_s'] = float(np.nanmedian(rt_p)) if np.any(~np.isnan(rt_p)) else None
                trajB[g] = (r['y'][0], tp)
            B[kind].append(row)
    fr = sim(p, g_rec=step_profile(g0, CUE), g_ctx=step_profile(g0, CUE))
    B['frozen_pre_cue'] = dict(g=g0, t80_s=t80(fr['y'][0], peak_c, y0) / 100, y_peak=float(fr['y'][0, CUE_END:].max()))
    rp_ = sim(p, g_rec=g_c, g_ctx=g_c)
    B['replay_control_g_max_abs_diff_y'] = float(np.abs(rp_['y'][0] - yc).max())
    M['gain_clamp'] = B

    # ---------- C. flow along the control path with g fixed ----------
    ts = np.arange(CUE_END + 10, CUE + int(t80c), 10)
    X = jnp.asarray(ctrl['x'][0, ts - 1])  # state entering step t
    sc_t = ctrl['s'][0, ts]
    flows = {}
    for g in GS + ['ctrl']:
        def v_at(x, s):
            return hooks.step_map(p, s)(x) - x
        s_arr = jnp.asarray(sc_t) if g == 'ctrl' else jnp.full(len(ts), np.log(g), jnp.float32)
        V = np.asarray(jax.vmap(v_at)(X, s_arr))
        J = np.asarray(jax.vmap(lambda x, s: jax.jacfwd(hooks.step_map(p, s))(x))(X, s_arr))
        lam = np.abs(np.linalg.eigvals(J)).max(-1)
        flows[g] = (V, lam)
    Vc = flows['ctrl'][0]
    C = []
    for g in GS:
        V, lam = flows[g]
        cos = (V * Vc).sum(-1) / (np.linalg.norm(V, axis=-1) * np.linalg.norm(Vc, axis=-1))
        along = (V * Vc).sum(-1) / (Vc ** 2).sum(-1)  # speed along the control direction, relative
        C.append(dict(g=g, cos_median=float(np.median(cos)), cos_min=float(cos.min()),
                      speed_along_median=float(np.median(along)), speed_ratio_median=float(np.median(np.linalg.norm(V, axis=-1) / np.linalg.norm(Vc, axis=-1))),
                      lam_max_median=float(np.median(lam)), lam_max_max=float(lam.max())))
    M['flow'] = dict(times_s=((ts - CUE) / 100).tolist(), rows=C, lam_ctrl=flows['ctrl'][1].tolist())
    json.dump(M, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1)
    figures(M, OUT, seed, yc, rc, g_c, t80c, trajA, trajB, flows, ts)
    return M


def figures(M, OUT, seed, yc, rc, g_c, t80c, trajA, trajB, flows, ts):
    t = (np.arange(T) - CUE) / 100
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(-1, 1)
    # A1: sustained SNc input, raw vs. time rescaled by t80 ratio
    fig, axs = plt.subplots(6, 2, figsize=(5, 8), sharex='col')
    for v, (yp, rp, gp, tp) in trajA.items():
        a = tp / t80c if not np.isnan(tp) else np.nan
        col = cmap(norm(v))
        rows = [('output y', yp), ('g_bg', gp)] + [(k, rp[:, sl].mean(-1)) for k, sl in AREAS.items()]
        for i, (lab, tr) in enumerate(rows):
            axs[i, 0].plot(t, tr, color=col, lw=1)
            if not np.isnan(a):
                axs[i, 1].plot(t / a, tr, color=col, lw=1)
            axs[i, 0].set_ylabel(lab)
    for ax in axs.ravel():
        ax.set_xlim(-0.5, 7); ax.axvline(0, color='r', lw=0.5)
    axs[0, 0].set_title('sustained SNc input from cue\n(blue -1 ... red +1)', fontsize=7)
    axs[0, 1].set_title('time / (t80 / t80_control)', fontsize=7)
    axs[-1, 0].set_xlabel('time after cue (s)'); axs[-1, 1].set_xlabel('rescaled time')
    fig.suptitle(f'seed {seed}: noise-free', fontsize=8); fig.tight_layout()
    fig.savefig(f'{OUT}/A_sustained_snc_scaling.png', dpi=250); plt.close(fig)

    # A2: warp model R2, path ratio, RT regression
    rows = [r for r in M['sustained_snc'] if 'warp' in r]
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.2))
    x = np.arange(len(rows))
    for j, (name, c) in enumerate(zip(['none', 'shift', 'scale', 'scale+shift'], ['gray', 'tab:orange', 'tab:blue', 'k'])):
        axs[0].bar(x + (j - 1.5) * 0.2, [r['warp'][name]['r2'] for r in rows], 0.2, color=c, label=name)
    axs[0].set_xticks(x); axs[0].set_xticklabels([f"{r['strength']:+g}" for r in rows]); axs[0].set_ylim(min(0, min(r['warp']['none']['r2'] for r in rows)), 1)
    axs[0].set_ylabel('warp fit R²'); axs[0].set_xlabel('SNc input'); axs[0].legend(fontsize=5, frameon=False)
    axs[1].bar(x - 0.15, [r['path']['nearest_over_extent'] for r in rows], 0.3, label='to nearest control state')
    axs[1].bar(x + 0.15, [r['path']['same_time_over_extent'] for r in rows], 0.3, label='to control at same time')
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{r['strength']:+g}" for r in rows]); axs[1].set_ylabel('median distance / path extent')
    axs[1].legend(fontsize=5, frameon=False); axs[1].set_xlabel('SNc input')
    for r in M['sustained_snc']:
        rr = r['rt']
        if 'slope' in rr:
            axs[2].errorbar(r['strength'], rr['slope'], yerr=1.96 * rr['slope_se'], fmt='o', color=cmap(norm(r['strength'])), ms=3)
    axs[2].axhline(1, color='k', lw=0.5); axs[2].set_xlabel('SNc input'); axs[2].set_ylabel('slope RT_pert vs RT_ctrl')
    fig.suptitle(f'seed {seed}: scaling vs shift (sustained SNc input)', fontsize=8); fig.tight_layout()
    fig.savefig(f'{OUT}/A_scaling_tests.png', dpi=250); plt.close(fig)

    # B: gain clamp
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.2))
    gn = plt.Normalize(min(GS), max(GS))
    for g, (yp, tp) in trajB.items():
        axs[0].plot(t, yp, color=plt.cm.viridis(gn(g)), lw=1)
    axs[0].plot(t, yc, 'k--', lw=1, label='control (closed loop)'); axs[0].set_xlim(-0.5, 6)
    axs[0].set_xlabel('time after cue (s)'); axs[0].set_ylabel('output y'); axs[0].legend(fontsize=5, frameon=False)
    axs[0].set_title('g clamped from cue (viridis: low→high g)', fontsize=6)
    for kind, c in [('constant', 'k'), ('rec_only', 'tab:purple'), ('ctx_only', 'tab:green')]:
        rr = M['gain_clamp'][kind]
        axs[1].plot([r['g'] for r in rr], [r['t80_s'] for r in rr], 'o-', color=c, ms=3, label=kind)
    axs[1].axhline(t80c / 100, color='gray', ls='--', lw=0.6); axs[1].axvline(M['control']['g_pre'], color='gray', ls=':', lw=0.6)
    axs[1].set_xlabel('clamped g'); axs[1].set_ylabel('t80 (s)'); axs[1].legend(fontsize=5, frameon=False)
    rr = M['gain_clamp']['constant']
    axs[2].plot([r['g'] for r in rr], [r['respond'] for r in rr], 'o-k', ms=3)
    axs[2].set_xlabel('clamped g'); axs[2].set_ylabel('fraction responding (noisy)')
    fig.suptitle(f'seed {seed}: open-loop gain clamp (dashed: control t80, dotted: pre-cue g)', fontsize=7); fig.tight_layout()
    fig.savefig(f'{OUT}/B_gain_clamp.png', dpi=250); plt.close(fig)

    # C: flow
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.2))
    tt = (ts - CUE) / 100
    Vc = flows['ctrl'][0]
    for g in GS:
        V, lam = flows[g]
        col = plt.cm.viridis(gn(g))
        along = (V * Vc).sum(-1) / (Vc ** 2).sum(-1)
        cos = (V * Vc).sum(-1) / (np.linalg.norm(V, axis=-1) * np.linalg.norm(Vc, axis=-1))
        axs[0].plot(tt, along, color=col, lw=1); axs[1].plot(tt, cos, color=col, lw=1); axs[2].plot(tt, lam, color=col, lw=1)
    axs[2].plot(tt, flows['ctrl'][1], 'k--', lw=1)
    axs[0].set_ylabel('speed along control flow\n(relative to control)'); axs[1].set_ylabel('cos(flow_g, flow_ctrl)')
    axs[2].set_ylabel('max |eig| one-step Jacobian'); axs[2].axhline(1, color='gray', lw=0.5)
    for ax in axs:
        ax.set_xlabel('time after cue on control path (s)')
    fig.suptitle(f'seed {seed}: flow on the control path with g fixed (viridis: low→high g; dashed: control g(t))', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/C_flow_vs_gain.png', dpi=250); plt.close(fig)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        M = main(int(s))
        print(json.dumps({k: M[k] for k in ['seed', 'control']}, indent=1))
