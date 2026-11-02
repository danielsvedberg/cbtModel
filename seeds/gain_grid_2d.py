"""2-D additive SPN-gain grid for cbtModel (rank-2 DA) seeds: python seeds/gain_grid_2d.py <dir> [seed ...]
Writes seeds/<dir>/seed_k/metrics_gaingrid.npz + .json and plots/png/gain_grid.png.

A constant offset is added to the SPN gains themselves for the whole trial: G = exp(l_bg) + delta, where exp(l_bg) is
the normal DA-driven gain (exp(+s_d1) on dSPN, exp(-s_d2) on iSPN) and G multiplies both the cortex->BG input and the BG
decay tau, as in training (model switch params['gain_add'], per-SPN). delta_dSPN and delta_iSPN are swept independently over
DEL = 0, +-0.001 * 2^k (k = 0..7, up to +-0.128), +-0.25 (19 x 19 grid; (0, 0) = the trained network).

Per cell: noise-free t80 / control t80 (t80 = first time after cue onset the output covers 80% of the control rise; NaN =
stall, 0 = already there at cue onset), scale-only and shift-only warp R2 of the 60 cortex/BG/thalamus rates over
[cue, t80], and over N_NOISY noisy trials (same keys in every cell): fraction responding, median RT, median paired RT
ratio vs control.
"""
import os, sys, json, pickle, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gain_sweep_2d as G
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DEV = [0.001 * 2 ** k for k in range(8)] + [0.25]
DEL = sorted([-d for d in DEV] + [0.0] + DEV)
I0 = DEL.index(0.0)


def with_add(p, d1, d2):
    q = dict(p)
    q['gain_add'] = jnp.concatenate([jnp.full(G.N1, d1), jnp.full(G.N_BG - G.N1, d2)]).astype(jnp.float32)
    return q


def grid(p):
    yc, xc = G.sim(p, 1, 0.0)
    yc, rc = yc[0], G.rate(xc[0])
    y0, peak = float(yc[G.CUE - 50:G.CUE].mean()), float(yc[G.CUE_END:].max())
    tc = G.t80(yc, peak, y0)
    rt_c = G.rts(G.sim(p, G.N_NOISY, G.cs.test_noise_std)[0])
    n = len(DEL)
    out = {k: np.full((n, n), np.nan) for k in ['t80_ratio', 'scale_r2', 'shift_r2', 'alpha', 'respond', 'rt_median', 'rt_ratio']}
    for i, d1 in enumerate(DEL):
        for j, d2 in enumerate(DEL):
            q = p if (d1 == 0 and d2 == 0) else with_add(p, d1, d2)
            yp, xp = G.sim(q, 1, 0.0)
            tp = G.t80(yp[0], peak, y0)
            out['t80_ratio'][i, j] = tp / tc if tc > 0 else np.nan
            if tc > 0 and tp == tp and tp > 0 and not (d1 == 0 and d2 == 0):
                w = G.warp_r2(rc, G.rate(xp[0]), tp)
                out['scale_r2'][i, j], out['shift_r2'][i, j], out['alpha'][i, j] = w['scale']['r2'], w['shift']['r2'], w['scale']['alpha']
            rt_p = rt_c if (d1 == 0 and d2 == 0) else G.rts(G.sim(q, G.N_NOISY, G.cs.test_noise_std)[0])
            both = ~np.isnan(rt_c) & ~np.isnan(rt_p)
            out['respond'][i, j] = np.mean(~np.isnan(rt_p))
            if np.any(~np.isnan(rt_p)):
                out['rt_median'][i, j] = np.nanmedian(rt_p)
            if both.sum() >= 10:
                out['rt_ratio'][i, j] = np.median(rt_p[both] / rt_c[both])
    control = dict(y_pre=y0, y_peak=peak, rise=peak - y0, t80_s=tc / 100, respond=float(np.mean(~np.isnan(rt_c))),
                   rt_median=float(np.nanmedian(rt_c)) if np.any(~np.isnan(rt_c)) else None)
    return out, control


def summarize(out, control):
    """working = noisy responding within 10 points of control and noise-free t80 exists and > 0."""
    ok = (np.abs(out['respond'] - control['respond']) < 0.10) & np.isfinite(out['t80_ratio']) & (out['t80_ratio'] > 0)
    tr = np.where(ok, out['t80_ratio'], np.nan)
    def axis_thr(v):  # smallest |delta| along a 1-D line where t80 moves >= 5%, stalls or responding moves >= 10 pts
        for k, d in enumerate(DEV):
            for idx in (I0 - 1 - k, I0 + 1 + k):
                t, r = v[0][idx], v[1][idx]
                if t != t or abs(t - 1) >= 0.05 or abs(r - control['respond']) >= 0.10:
                    return d
        return None
    S = dict(control=control, frac_working=float(ok.mean()),
             t80_ratio_range_working=[float(np.nanmin(tr)), float(np.nanmax(tr))] if np.any(ok) else None,
             thr_dSPN_only=axis_thr((out['t80_ratio'][:, I0], out['respond'][:, I0])),
             thr_iSPN_only=axis_thr((out['t80_ratio'][I0, :], out['respond'][I0, :])))
    # mean scale-vs-shift advantage over working cells that move t80 >= 5%
    mv = ok & (np.abs(out['t80_ratio'] - 1) >= 0.05) & np.isfinite(out['scale_r2'])
    S['n_working_moved'] = int(mv.sum())
    S['scale_minus_shift_r2_moved'] = float(np.mean(out['scale_r2'][mv] - out['shift_r2'][mv])) if mv.any() else None
    S['scale_r2_moved_median'] = float(np.median(out['scale_r2'][mv])) if mv.any() else None
    return S


def plot(out, S, path, title):
    lab = [('%g' % d) if abs(d) in (0.25, 0.128, 0.016, 0.002) or d == 0 else '' for d in DEL]
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.4))
    t = out['t80_ratio'].copy(); stall = np.isnan(t)
    tl = np.log2(np.clip(np.where(stall, 4, t), 1 / 4, 4))
    im = axs[0].imshow(tl, origin='lower', cmap='coolwarm', norm=TwoSlopeNorm(0, -2, 2))
    axs[0].contour(stall.astype(float), levels=[0.5], colors='k', linewidths=0.6, origin='lower')
    cb = fig.colorbar(im, ax=axs[0], shrink=0.8); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['1/4', '1/2', '1', '2', '4 / stall'])
    axs[0].set_title('noise-free t80 ratio (black: stall edge)', fontsize=8)
    im = axs[1].imshow(out['respond'], origin='lower', cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[1], shrink=0.8); axs[1].set_title(f"responding (control {S['control']['respond']:.0%})", fontsize=8)
    rt0 = S['control']['rt_median']
    vmax = (G.T - G.CUE_END) / 100
    im = axs[2].imshow(out['rt_median'], origin='lower', cmap='coolwarm',
                       norm=TwoSlopeNorm(rt0, 0, vmax) if rt0 and 0 < rt0 < vmax else None)
    cb = fig.colorbar(im, ax=axs[2], shrink=0.8); cb.set_label('s after cue offset', fontsize=6)
    axs[2].set_title(f"median RT, s (ctrl {rt0:.2f}; white = none)" if rt0 else 'median noisy RT', fontsize=8)
    im = axs[3].imshow(np.log2(out['rt_ratio']), origin='lower', cmap='coolwarm', norm=TwoSlopeNorm(0, -1, 1))
    cb = fig.colorbar(im, ax=axs[3], shrink=0.8); cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(['1/2', '1', '2'])
    axs[3].set_title('median paired RT ratio vs control', fontsize=8)
    for a in axs:
        a.set_xticks(range(len(DEL))); a.set_xticklabels(lab, fontsize=5, rotation=90)
        a.set_yticks(range(len(DEL))); a.set_yticklabels(lab, fontsize=5)
        a.set_xlabel('δ iSPN gain'); a.set_ylabel('δ dSPN gain'); a.plot(I0, I0, 'k+', ms=6)
    fig.suptitle(f'{title} — constant offset added to SPN gains (G = exp(l) + δ, on cortex->BG and tau)', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


if __name__ == '__main__' and '--replot' in sys.argv:  # redraw plots from saved metrics_gaingrid.npz/.json
    sys.argv.remove('--replot')
    D = os.path.join(G.ROOT, 'seeds', sys.argv[1])
    for f in sorted(glob.glob(f'{D}/seed_*/metrics_gaingrid.npz')):
        sd = os.path.dirname(f); z = np.load(f); S = json.load(open(f'{sd}/metrics_gaingrid.json'))
        plot({k: z[k] for k in z.files}, S, f'{sd}/plots/png/gain_grid.png', f"seed {S['seed']}")
    sys.exit()

if __name__ == '__main__':
    D = os.path.join(G.ROOT, 'seeds', sys.argv[1])
    seeds = list(map(int, sys.argv[2:])) or sorted(int(q.split('_')[-1]) for q in glob.glob(f'{D}/seed_*'))
    for s in seeds:
        sd = f'{D}/seed_{s}'
        if not os.path.exists(f'{sd}/params_nm.pkl'):
            continue
        p = pickle.load(open(f'{sd}/params_nm.pkl', 'rb'))
        out, control = grid(p)
        S = summarize(out, control); S['seed'] = s; S['deltas'] = DEL
        os.makedirs(f'{sd}/plots/png', exist_ok=True)
        np.savez_compressed(f'{sd}/metrics_gaingrid.npz', deltas=np.array(DEL), **out)
        json.dump(S, open(f'{sd}/metrics_gaingrid.json', 'w'), indent=1, default=float)
        plot(out, S, f'{sd}/plots/png/gain_grid.png', f'seed {s}')
        print(s, f"ctrl t80 {control['t80_s']:.2f} resp {control['respond']:.0%} | working {S['frac_working']:.0%} "
              f"t80 range {S['t80_ratio_range_working']} | thr dSPN {S['thr_dSPN_only']} iSPN {S['thr_iSPN_only']} | "
              f"moved {S['n_working_moved']} scale R2 {S['scale_r2_moved_median']} scale-shift {S['scale_minus_shift_r2_moved']}", flush=True)
