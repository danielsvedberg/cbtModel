"""Temporal-scaling screen for every seed in a batch dir: python seeds/scaling_screen.py <dir> [seed ...]
Writes seeds/<dir>/seed_k/metrics_scaling.json; summarize_1k.py adds the verdict to summary.md.

Protocol (analysis/seed9_scaling, section A): sustained SNc input from cue offset to trial end, strengths SUST,
noise-free trajectories through hooks.py (checked bit-identical to jit(mf.batched_nm_rnn) per seed).
t80 = first time the output covers 80% of the rise from its pre-cue level to the control peak (cue onset = 0).
Warp fits on the 60 cortex/BG/thalamus unit rates over [cue, perturbed t80]: shift-only vs scale-only (about cue onset).
Paired noisy RTs (N_NOISY trials, same noise keys): median RT_pert / RT_ctrl on trials responding in both.

Verdict:
  'no ramp'  control noise-free output rises < 0.1 above its pre-cue level, or is already at t80 at cue onset
  'yes'      t80 ratio monotonic over SNc input -0.5..+0.5, >= 5% slower at -0.5 (or stalls) and >= 5% faster at +0.5,
             and at +-0.25 and +-0.5 (where t80 exists) scale-only R2 >= 0.95 and >= shift-only R2 + 0.02
  'partial'  t80 changes >= 5% in the right direction on one side only, or the scale fit misses the R2 bar but
             scale-only still beats shift-only by 0.02 on average
  'no'       otherwise (timing doesn't move, moves the wrong way, or the change is a shift / shape change)
"""
import os, sys, json, pickle, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'analysis', 'seed9_scaling'), ROOT]
os.chdir(ROOT)
import jax.numpy as jnp
import hooks
from scaling_analysis import sim, nm_profile, t80, rts, CUE, CUE_END, ALPHAS, DELTAS

SUST = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
N_NOISY = 500
CHECK = [-0.5, -0.25, 0.25, 0.5]
rate = lambda x: np.maximum(0, np.tanh(x))


def warp_r2(rc, rp, t_end):
    t = np.arange(CUE, CUE + int(t_end) + 1)
    Y = rp[t]
    sst = ((Y - Y.mean(0)) ** 2).sum()
    out = {}
    for name, A, D in [('shift', [1.0], DELTAS), ('scale', ALPHAS, [0])]:
        best = (np.inf, None, None)
        for a in A:
            for d in D:
                tau = np.clip(CUE + (t - CUE - d) / a, 0, hooks.T - 1)
                i0 = np.floor(tau).astype(int); i1 = np.minimum(i0 + 1, hooks.T - 1); w = (tau - i0)[:, None]
                sse = ((Y - ((1 - w) * rc[i0] + w * rc[i1])) ** 2).sum()
                if sse < best[0]:
                    best = (sse, a, d)
        out[name] = dict(r2=float(1 - best[0] / sst), alpha=float(best[1]), shift_steps=float(best[2]))
    return out


def verdict(M):
    if M['control']['rise'] < 0.1 or not M['control']['t80_s'] > 0:  # nan, or already at 80% at cue onset
        return 'no ramp'
    R = {r['strength']: r for r in M['rows']}
    ratio = lambda v: np.inf if np.isnan(R[v]['t80_ratio']) else R[v]['t80_ratio']  # stall = infinitely slow
    seq = [ratio(v) for v in [-0.5, -0.25, 0.0, 0.25, 0.5]]
    mono = all(a >= b for a, b in zip(seq, seq[1:]))
    slow, fast = ratio(-0.5) >= 1.05, ratio(0.5) <= 0.95
    fits = [R[v]['warp'] for v in CHECK if 'warp' in R[v]]
    good_fit = len(fits) > 0 and all(f['scale']['r2'] >= 0.95 and f['scale']['r2'] >= f['shift']['r2'] + 0.02 for f in fits)
    scale_beats = len(fits) > 0 and np.mean([f['scale']['r2'] - f['shift']['r2'] for f in fits]) >= 0.02
    if mono and slow and fast and good_fit:
        return 'yes'
    if (slow or fast) and (ratio(-0.5) >= ratio(0.5)) and scale_beats:
        return 'partial'
    return 'no'


def screen(p):
    M = dict(verify_max_abs_diff=hooks.verify(p), rec_gain_sign=hooks.REC_SIGN, bg_tau_gain=hooks.TAU_GAIN)
    assert M['verify_max_abs_diff'] == 0.0, M
    c = sim(p)
    yc, rc = c['y'][0], rate(c['x'][0])
    y0, peak = float(yc[CUE - 50:CUE].mean()), float(yc[CUE_END:].max())
    tc = t80(yc, peak, y0)
    M['control'] = dict(y_pre=y0, y_peak=peak, rise=peak - y0, t80_s=tc / 100)
    rt_c = rts(sim(p, N_NOISY, noise=0.05, key=0)['y'])
    M['rows'] = []
    for v in SUST:
        r = sim(p, nm_in=nm_profile(v))
        tp = t80(r['y'][0], peak, y0)
        row = dict(strength=v, t80_s=tp / 100, t80_ratio=tp / tc if tc > 0 else np.nan)
        if v != 0 and not np.isnan(tp) and tc > 0:
            row['warp'] = warp_r2(rc, rate(r['x'][0]), tp)
        rt_p = rts(sim(p, N_NOISY, nm_in=nm_profile(v), noise=0.05, key=0)['y'])
        both = ~np.isnan(rt_c) & ~np.isnan(rt_p)
        row.update(respond_ctrl=float(np.mean(~np.isnan(rt_c))), respond_pert=float(np.mean(~np.isnan(rt_p))), n_both=int(both.sum()),
                   rt_ratio_median=float(np.median(rt_p[both] / rt_c[both])) if both.sum() >= 10 else None)
        M['rows'].append(row)
    M['verdict'] = verdict(M)
    return M


if __name__ == '__main__':
    D = os.path.join(ROOT, 'seeds', sys.argv[1])
    seeds = list(map(int, sys.argv[2:])) or sorted(int(q.split('_')[-1]) for q in glob.glob(f'{D}/seed_*'))
    for s in seeds:
        f = f'{D}/seed_{s}/params_nm.pkl'
        if not os.path.exists(f):
            continue
        M = screen(pickle.load(open(f, 'rb')))
        M['seed'] = s
        json.dump(M, open(f'{D}/seed_{s}/metrics_scaling.json', 'w'), indent=1, default=float)
        print(s, M['verdict'], 'ctrl t80', round(M['control']['t80_s'], 2), ' '.join(
            f"{r['strength']:+g}:{r['t80_ratio']:.2f}/{r['warp']['scale']['r2']:.3f}/{r['warp']['shift']['r2']:.3f}" if 'warp' in r
            else f"{r['strength']:+g}:{r['t80_ratio']:.2f}" for r in M['rows']), flush=True)
