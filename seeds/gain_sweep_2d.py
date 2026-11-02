"""Fixed-gain sweep for cbtModel (rank-2 DA) seeds, ported from cbtModel_nonm seeds/gain_sweep.py:
    python seeds/gain_sweep_2d.py <dir> [seed ...]   (from repo root)
Writes seeds/<dir>/seed_k/metrics_gainsweep.json, gain_sweep_rts.npz and plots/png/gain_sweep_{diag,da}.png.

In cbtModel the SPN gains are DA-driven: G_d1 = exp(+s_d1) on dSPN rows, G_d2 = exp(-s_d2) on iSPN rows, applied to the
cortex->BG weights (G_bg_e) and the BG decay tau (g_bg_tau). Here the DA-driven gain is replaced by a fixed gain for the
whole trial (SNc still runs; it only acts through s), on two axes over g in GAINS:
  diag : dSPN = iSPN = g          (the nonm sweep: same g on every SPN)
  da   : dSPN = g, iSPN = 1/g     (the DA direction: DA raises dSPN and lowers iSPN gain)
and in three conditions, as in nonm:
  both : cortex->BG gain and BG tau gain fixed
  ctx  : cortex->BG gain fixed, BG tau gain left DA-driven
  tau  : BG tau gain fixed, cortex->BG gain left DA-driven
Unlike nonm, g = 1 (s = 0) is not the trained network, so the control is the intact DA network: control t80, peak and
noisy RTs come from it, and g = 1 is reported as its own column.

Runs mf.batched_nm_rnn itself with the model's fixed-gain switch (params['fixed_gain_ctx'] / ['fixed_gain_tau'],
per-SPN vectors); without those keys the model is unchanged.

Metrics as in nonm: noise-free t80 (first time after cue onset the output covers 80% of the control rise) and its ratio
to the intact-DA control; shift-only vs scale-only warp R2 of the 60 cortex/BG/thalamus rates over [cue, perturbed t80]
against the control; N_NOISY noisy trials (same keys in every condition): fraction responding, median RT, median paired
RT ratio vs control.

Verdict (per axis x condition), nonm criteria with the control (ratio 1) inserted between g = 0.75 and g = 1.5 in the
monotonicity sequence in place of g = 1:
  'no ramp'  control noise-free output rises < 0.1 above its pre-cue level, or is already at t80 at cue onset
  'yes'      t80 ratio monotonic over 0.5, 0.75, control, 1.5, 2 (stall = slowest), changes >= 5% at both 0.75 and 1.5 in
             opposite directions, and at 0.75 and 1.5 (where t80 exists) scale-only R2 >= 0.95 and >= shift-only + 0.02
  'partial'  t80 changes >= 5% at 0.75 or 1.5 with scale-only beating shift-only by 0.02 on average (over all g != control)
  'no'       otherwise
"""
import os, sys, json, pickle, glob, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import config_script as cs
import model_functions as mf

T = cs.config['T']
CUE = 300
CUE_END = CUE + cs.config['T_cue']
TAU_X, TAU_Z = cs.config['tau_x'], cs.config['tau_z']
N_BG = cs.config['n_bg']; N1 = N_BG // 2
GAINS = [0.5, 0.75, 1.0, 1.5, 2.0]
INNER = [0.75, 1.5]
AXES = {'diag': lambda g: (g, g), 'da': lambda g: (g, 1.0 / g)}
CONDS = {'both': ('fixed_gain_ctx', 'fixed_gain_tau'), 'ctx': ('fixed_gain_ctx',), 'tau': ('fixed_gain_tau',)}
N_NOISY = 500
ALPHAS = np.round(np.arange(0.3, 3.01, 0.01), 2)
DELTAS = np.arange(-200, 201, 5)
rate = lambda x: np.maximum(0, np.tanh(x))
_inp = mf.self_timed_movement_task(jnp.array([CUE]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)[0]


_run = jax.jit(lambda p, inp, ns, keys: mf.batched_nm_rnn(p, cs.x0, cs.z0, inp, TAU_X, TAU_Z, True, None, ns, keys),
               static_argnums=(2,))


def with_gain(p, cond, g12):
    """params with the model's fixed-gain switch set: per-SPN gains (dSPN = g12[0], iSPN = g12[1])."""
    q = dict(p)
    v = jnp.concatenate([jnp.full(N1, g12[0]), jnp.full(N_BG - N1, g12[1])]).astype(jnp.float32)
    for k in CONDS[cond]:
        q[k] = v
    return q


def sim(p, n, noise):
    keys = jr.split(jr.PRNGKey(0), n)
    y, (xbg, xc, xt), _ = _run(p, jnp.repeat(_inp, n, axis=0), noise, keys)
    return np.asarray(y[..., 0]), np.asarray(jnp.concatenate([xbg, xc, xt], -1))


def t80(y, ctrl_peak, y0):
    above = y[CUE:] >= y0 + 0.8 * (ctrl_peak - y0)
    return float(above.argmax()) if above.any() else np.nan


def rts(y):
    a = y[:, CUE_END:] > 0.5
    return np.where(a.any(1), a.argmax(1) + 10, np.nan) / 100.0


def warp_r2(rc, rp, t_end):
    t = np.arange(CUE, CUE + int(t_end) + 1)
    Y = rp[t]
    sst = ((Y - Y.mean(0)) ** 2).sum()
    out = {}
    for name, A, D in [('shift', [1.0], DELTAS), ('scale', ALPHAS, [0])]:
        best = (np.inf, None, None)
        for a in A:
            for d in D:
                tau = np.clip(CUE + (t - CUE - d) / a, 0, T - 1)
                i0 = np.floor(tau).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (tau - i0)[:, None]
                sse = ((Y - ((1 - w) * rc[i0] + w * rc[i1])) ** 2).sum()
                if sse < best[0]:
                    best = (sse, a, d)
        out[name] = dict(r2=float(1 - best[0] / sst) if sst > 0 else float('nan'), alpha=float(best[1]),
                         shift_steps=float(best[2]))
    return out


def verdict(c, rows):
    if c['rise'] < 0.1 or not c['t80_s'] > 0:
        return 'no ramp'
    R = {r['gain']: r for r in rows}
    ratio = lambda g: np.inf if R[g]['t80_ratio'] != R[g]['t80_ratio'] else R[g]['t80_ratio']
    seq = [ratio(0.5), ratio(0.75), 1.0, ratio(1.5), ratio(2.0)]
    mono = all(np.diff(seq) >= 0) or all(np.diff(seq) <= 0)
    lo, hi = ratio(0.75), ratio(1.5)
    both_sides = (lo >= 1.05 and hi <= 0.95) or (lo <= 0.95 and hi >= 1.05)
    fits = [R[g]['warp'] for g in INNER if 'warp' in R[g]]
    good_fit = len(fits) > 0 and all(f['scale']['r2'] >= 0.95 and f['scale']['r2'] >= f['shift']['r2'] + 0.02 for f in fits)
    allfits = [R[g]['warp'] for g in GAINS if 'warp' in R[g]]
    diffs = [f['scale']['r2'] - f['shift']['r2'] for f in allfits if np.isfinite(f['scale']['r2'] - f['shift']['r2'])]
    scale_beats = len(diffs) > 0 and np.mean(diffs) >= 0.02
    if mono and both_sides and good_fit:
        return 'yes'
    if (abs(lo - 1) >= 0.05 or abs(hi - 1) >= 0.05) and scale_beats:
        return 'partial'
    return 'no'


def sweep(p, fig_dir, title):
    yc, xc = sim(p, 1, 0.0)
    yc, rc = yc[0], rate(xc[0])
    y0, peak = float(yc[CUE - 50:CUE].mean()), float(yc[CUE_END:].max())
    tc = t80(yc, peak, y0)
    rt_c = rts(sim(p, N_NOISY, cs.test_noise_std)[0])
    control = dict(y_pre=y0, y_peak=peak, rise=peak - y0, t80_s=tc / 100, respond=float(np.mean(~np.isnan(rt_c))),
                   rt_median=float(np.nanmedian(rt_c)) if np.any(~np.isnan(rt_c)) else None)
    M = dict(gains=GAINS, control=control, axes={})
    rt_store, traces = {'control': rt_c}, {}
    for ax, gfun in AXES.items():
        M['axes'][ax] = {}
        for cond in CONDS:
            rows = []
            for g in GAINS:
                q = with_gain(p, cond, gfun(g))
                yp, xp = sim(q, 1, 0.0)
                tp = t80(yp[0], peak, y0)
                row = dict(gain=g, g_d1=gfun(g)[0], g_d2=gfun(g)[1], t80_s=tp / 100,
                           t80_ratio=tp / tc if tc > 0 else np.nan)
                if not np.isnan(tp) and tc > 0 and tp > 0:
                    row['warp'] = warp_r2(rc, rate(xp[0]), tp)
                rt_p = rts(sim(q, N_NOISY, cs.test_noise_std)[0])
                both = ~np.isnan(rt_c) & ~np.isnan(rt_p)
                row.update(respond=float(np.mean(~np.isnan(rt_p))),
                           rt_median=float(np.nanmedian(rt_p)) if np.any(~np.isnan(rt_p)) else None, n_both=int(both.sum()),
                           rt_ratio_median=float(np.median(rt_p[both] / rt_c[both])) if both.sum() >= 10 else None)
                rows.append(row)
                rt_store[f'{ax}_{cond}_g{g:g}'] = rt_p
                traces[(ax, cond, g)] = yp[0]
            M['axes'][ax][cond] = dict(rows=rows, verdict=verdict(control, rows))
    np.savez_compressed('gain_sweep_rts.npz', **rt_store)
    for ax in AXES:
        plot(M, ax, yc, traces, rt_store, fig_dir, title)
    return M


def plot(M, ax, yc, traces, rt_store, fig_dir, title):
    t = (np.arange(T) - CUE) / 100
    cols = plt.cm.coolwarm(np.linspace(0, 1, len(GAINS)))
    fig, axs = plt.subplots(2, 3, figsize=(8, 4.4), sharex='row', sharey='row')
    rc = rt_store['control']; rcs = np.sort(rc[~np.isnan(rc)])
    for j, cond in enumerate(CONDS):
        axs[0, j].plot(t, yc, color='k', lw=1.6, label='DA')
        if len(rcs):
            axs[1, j].step(rcs, np.arange(1, len(rcs) + 1) / len(rc), where='post', color='k', lw=1.6)
        for g, c in zip(GAINS, cols):
            axs[0, j].plot(t, traces[(ax, cond, g)], color=c, lw=0.8)
            r = rt_store[f'{ax}_{cond}_g{g:g}']
            rr = np.sort(r[~np.isnan(r)])
            if len(rr):
                axs[1, j].step(rr, np.arange(1, len(rr) + 1) / len(r), where='post', color=c, lw=0.8)
        axs[0, j].set_title(f"{cond}: {M['axes'][ax][cond]['verdict']}", fontsize=8)
        axs[0, j].axvline(0, color='k', lw=0.4); axs[1, j].set_xlabel('time after cue onset (s)')
        axs[1, j].set_xlim(0, (T - CUE) / 100); axs[1, j].set_ylim(0, 1)
    axs[0, 0].set_ylabel('output y (noise-free)'); axs[1, 0].set_ylabel('fraction responded')
    lab = [f'{g:g}' if ax == 'diag' else f'{g:g}/{1 / g:g}' for g in GAINS]
    h = axs[0, 2].get_lines()
    axs[0, 2].legend(h[:1] + h[1:1 + len(GAINS)], ['DA'] + lab, title='dSPN/iSPN gain' if ax == 'da' else 'gain',
                     fontsize=5, title_fontsize=6, frameon=False)
    desc = 'dSPN = iSPN = g' if ax == 'diag' else 'dSPN = g, iSPN = 1/g'
    fig.suptitle(f'{title} — fixed SPN gains replacing DA ({desc}); black = intact DA', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{fig_dir}/gain_sweep_{ax}.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    D = os.path.join(ROOT, 'seeds', sys.argv[1])
    seeds = list(map(int, sys.argv[2:])) or sorted(int(q.split('_')[-1]) for q in glob.glob(f'{D}/seed_*'))
    for s in seeds:
        sd = f'{D}/seed_{s}'
        if not os.path.exists(f'{sd}/params_nm.pkl'):
            continue
        p = pickle.load(open(f'{sd}/params_nm.pkl', 'rb'))
        os.chdir(sd)
        os.makedirs('plots/png', exist_ok=True)
        M = sweep(p, 'plots/png', f'seed {s}')
        os.chdir(ROOT)
        M['seed'] = s
        json.dump(M, open(f'{sd}/metrics_gainsweep.json', 'w'), indent=1, default=float)
        print(s, 'ctrl t80', round(M['control']['t80_s'], 2), f"resp {M['control']['respond']:.0%}",
              ' | '.join(f"{a}/{c}:{M['axes'][a][c]['verdict']}" for a in AXES for c in CONDS), flush=True)
