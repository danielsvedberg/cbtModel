"""Fine multiplicative gain sweep (add --from_cue to apply the multiplier only from cue onset to trial end;
outputs then get the suffix _fromcue) for cbtModel (rank-2 DA) seeds: python seeds/gain_sweep_2d_fine.py <dir> [seed ...]
Writes seeds/<dir>/seed_k/metrics_gainsweep_fine.json, gain_sweep_fine_rts.npz, plots/png/gain_sweep_fine_{diag,da}.png.

Unlike seeds/gain_sweep_2d.py (which replaces the DA-driven gain with a fixed value), here a fixed multiplier g is applied
on top of the DA-driven SPN gains for the whole trial (model switch params['gain_mult_ctx'] / ['gain_mult_tau']), so
g = 1 is exactly the trained network, as in cbtModel_nonm. Gains: 1 +- 0.001 * 2^k (k = 0..7, i.e. up to +-0.128), then
0.75 and 1.25. Axes: diag (dSPN = iSPN = g) and da (dSPN = g, iSPN = 1/g); conditions both / ctx / tau as before.

Metrics as in gain_sweep_2d.py (t80 ratio vs g = 1, warp R2, responding, paired noisy RT ratio). Verdict uses the
gain_sweep_2d criteria with the inner pair 0.872 / 1.128 (strict fit) and monotonicity over all gains, plus
'threshold': the smallest |g - 1| at which t80 changes >= 5% (or stalls) or responding changes >= 10 points.
"""
import os, sys, json, pickle, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gain_sweep_2d as G
import jax.numpy as jnp
import matplotlib.pyplot as plt

DEV = [0.001 * 2 ** k for k in range(8)] + [0.25]
GAINS = sorted([round(1 - d, 6) for d in DEV] + [1.0] + [round(1 + d, 6) for d in DEV])
INNER = [0.872, 1.128]
FROM_CUE = '--from_cue' in sys.argv  # multiplier only from cue onset (step CUE) to trial end; g = 1 before
if FROM_CUE:
    sys.argv.remove('--from_cue')
FROMTXT = ' from cue onset' if FROM_CUE else ''
SUFFIX = ''  # from-cue results overwrite the whole-trial ones (user request 2026-11-02)
CONDS = {'both': ('gain_mult_ctx', 'gain_mult_tau'), 'ctx': ('gain_mult_ctx',), 'tau': ('gain_mult_tau',)}


def with_mult(p, cond, g12):
    q = dict(p)
    v = jnp.concatenate([jnp.full(G.N1, g12[0]), jnp.full(G.N_BG - G.N1, g12[1])]).astype(jnp.float32)
    if FROM_CUE:
        v = jnp.where((jnp.arange(G.T) >= G.CUE)[:, None], v[None, :], 1.0).astype(jnp.float32)  # (T, num_bg)
    for k in CONDS[cond]:
        q[k + ('_t' if FROM_CUE else '')] = v
    return q


def verdict(c, rows):
    if c['rise'] < 0.1 or not c['t80_s'] > 0:
        return 'no ramp'
    R = {r['gain']: r for r in rows}
    ratio = lambda g: np.inf if R[g]['t80_ratio'] != R[g]['t80_ratio'] else R[g]['t80_ratio']
    seq = [ratio(g) for g in GAINS]
    mono = all(np.diff(seq) >= 0) or all(np.diff(seq) <= 0)
    lo, hi = ratio(INNER[0]), ratio(INNER[1])
    both_sides = (lo >= 1.05 and hi <= 0.95) or (lo <= 0.95 and hi >= 1.05)
    fits = [R[g]['warp'] for g in INNER if 'warp' in R[g]]
    good_fit = len(fits) > 0 and all(f['scale']['r2'] >= 0.95 and f['scale']['r2'] >= f['shift']['r2'] + 0.02 for f in fits)
    d = [R[g]['warp']['scale']['r2'] - R[g]['warp']['shift']['r2'] for g in GAINS if 'warp' in R[g]]
    d = [x for x in d if np.isfinite(x)]
    if mono and both_sides and good_fit:
        return 'yes'
    if (abs(lo - 1) >= 0.05 or abs(hi - 1) >= 0.05) and len(d) and np.mean(d) >= 0.02:
        return 'partial'
    return 'no'


def threshold(rows, ctrl_resp):
    for dev in DEV:
        for r in rows:
            if abs(abs(r['gain'] - 1) - dev) < 1e-9:
                tr = r['t80_ratio']
                if tr != tr or abs(tr - 1) >= 0.05 or abs(r['respond'] - ctrl_resp) >= 0.10:
                    return dev
    return None


def sweep(p, fig_dir, title):
    yc, xc = G.sim(p, 1, 0.0)
    yc, rc = yc[0], G.rate(xc[0])
    y0, peak = float(yc[G.CUE - 50:G.CUE].mean()), float(yc[G.CUE_END:].max())
    tc = G.t80(yc, peak, y0)
    rt_c = G.rts(G.sim(p, G.N_NOISY, G.cs.test_noise_std)[0])
    control = dict(y_pre=y0, y_peak=peak, rise=peak - y0, t80_s=tc / 100, respond=float(np.mean(~np.isnan(rt_c))),
                   rt_median=float(np.nanmedian(rt_c)) if np.any(~np.isnan(rt_c)) else None)
    M = dict(gains=GAINS, control=control, axes={})
    rt_store, traces = {'control': rt_c}, {}
    for ax, gfun in G.AXES.items():
        M['axes'][ax] = {}
        for cond in CONDS:
            rows = []
            for g in GAINS:
                q = p if g == 1.0 else with_mult(p, cond, gfun(g))
                yp, xp = G.sim(q, 1, 0.0)
                tp = G.t80(yp[0], peak, y0)
                row = dict(gain=g, g_d1=gfun(g)[0], g_d2=gfun(g)[1], t80_s=tp / 100, t80_ratio=tp / tc if tc > 0 else np.nan)
                if g != 1.0 and not np.isnan(tp) and tc > 0 and tp > 0:
                    row['warp'] = G.warp_r2(rc, G.rate(xp[0]), tp)
                rt_p = rt_c if g == 1.0 else G.rts(G.sim(q, G.N_NOISY, G.cs.test_noise_std)[0])
                both = ~np.isnan(rt_c) & ~np.isnan(rt_p)
                row.update(respond=float(np.mean(~np.isnan(rt_p))),
                           rt_median=float(np.nanmedian(rt_p)) if np.any(~np.isnan(rt_p)) else None, n_both=int(both.sum()),
                           rt_ratio_median=float(np.median(rt_p[both] / rt_c[both])) if both.sum() >= 10 else None)
                rows.append(row)
                rt_store[f'{ax}_{cond}_g{g:g}'] = rt_p
                traces[(ax, cond, g)] = yp[0]
            M['axes'][ax][cond] = dict(rows=rows, verdict=verdict(control, rows), threshold=threshold(rows, control['respond']))
    np.savez_compressed(f'gain_sweep_fine{SUFFIX}_rts.npz', **rt_store)
    for ax in G.AXES:
        plot(M, ax, traces, rt_store, fig_dir, title)
    return M


def plot(M, ax, traces, rt_store, fig_dir, title):
    t = (np.arange(G.T) - G.CUE) / 100
    cols = plt.cm.coolwarm(np.linspace(0, 1, len(GAINS)))
    fig, axs = plt.subplots(3, 3, figsize=(8, 6.4), sharex='row', sharey='row')
    for j, cond in enumerate(CONDS):
        rows = M['axes'][ax][cond]['rows']
        for g, c in zip(GAINS, cols):
            axs[0, j].plot(t, traces[(ax, cond, g)], color='k' if g == 1 else c, lw=1.4 if g == 1 else 0.6)
            r = rt_store[f'{ax}_{cond}_g{g:g}']; rr = np.sort(r[~np.isnan(r)])
            if len(rr):
                axs[1, j].step(rr, np.arange(1, len(rr) + 1) / len(r), where='post', color='k' if g == 1 else c,
                               lw=1.4 if g == 1 else 0.6)
        gg = np.array([r['gain'] for r in rows]); tr = np.array([r['t80_ratio'] for r in rows], float)
        resp = np.array([r['respond'] for r in rows])
        dg = gg - 1
        axs[2, j].plot(dg, tr, 'o-', ms=2.5, color='C0', label='t80 ratio (noise-free)')
        axs[2, j].plot(dg[np.isnan(tr)], np.ones(np.isnan(tr).sum()) * 0.5, 'x', color='C3', ms=4, label='stall')
        axs[2, j].plot(dg, resp, 's-', ms=2, color='C2', lw=0.8, label='responding (noisy)')
        axs[2, j].set_xscale('symlog', linthresh=0.001); axs[2, j].axvline(0, color='k', lw=0.4)
        axs[2, j].set_xticks([-0.25, -0.1, -0.01, 0, 0.01, 0.1, 0.25]); axs[2, j].set_xticklabels(['-.25', '-.1', '-.01', '0', '.01', '.1', '.25'], fontsize=6)
        axs[2, j].set_xlabel('g - 1 (dSPN gain)')
        th = M['axes'][ax][cond]['threshold']
        axs[0, j].set_title(f"{cond}: {M['axes'][ax][cond]['verdict']}, threshold {th if th is None else f'{th:g}'}", fontsize=8)
        axs[0, j].axvline(0, color='k', lw=0.4); axs[1, j].set_xlabel('time after cue onset (s)')
        axs[1, j].set_xlim(0, (G.T - G.CUE) / 100); axs[1, j].set_ylim(0, 1)
    axs[0, 0].set_ylabel('output y (noise-free)'); axs[1, 0].set_ylabel('fraction responded'); axs[2, 0].set_ylabel('ratio / fraction')
    axs[2, 2].legend(fontsize=5, frameon=False)
    desc = 'dSPN = iSPN = g' if ax == 'diag' else 'dSPN = g, iSPN = 1/g'
    fig.suptitle(f'{title} — fixed gain multiplier on top of DA{FROMTXT} ({desc}); black = trained (g = 1)', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{fig_dir}/gain_sweep_fine{SUFFIX}_{ax}.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    D = os.path.join(G.ROOT, 'seeds', sys.argv[1])
    seeds = list(map(int, sys.argv[2:])) or sorted(int(q.split('_')[-1]) for q in glob.glob(f'{D}/seed_*'))
    for s in seeds:
        sd = f'{D}/seed_{s}'
        if not os.path.exists(f'{sd}/params_nm.pkl'):
            continue
        p = pickle.load(open(f'{sd}/params_nm.pkl', 'rb'))
        os.chdir(sd); os.makedirs('plots/png', exist_ok=True)
        M = sweep(p, 'plots/png', f'seed {s}')
        os.chdir(G.ROOT)
        M['seed'] = s
        json.dump(M, open(f'{sd}/metrics_gainsweep_fine{SUFFIX}.json', 'w'), indent=1, default=float)
        print(s, 'ctrl t80', round(M['control']['t80_s'], 2), f"resp {M['control']['respond']:.0%}",
              ' | '.join(f"{a}/{c}:{M['axes'][a][c]['verdict']}@{M['axes'][a][c]['threshold']}" for a in G.AXES for c in CONDS), flush=True)
