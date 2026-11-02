"""Cue-aligned area activity along the temporally scaling ridge of seeds/gain_grid_2d.py:
    python seeds/gain_grid_ridge.py <dir> [seed ...]   -> seeds/<dir>/seed_k/plots/png/gain_grid_ridge.png, gain_grid_lines.png, metrics_gaingrid_ridge.json

Ridge = of the 12 lines through (0, 0) in the (delta dSPN, delta iSPN) grid (grid-index steps in STEPS: dSPN only, iSPN
only, same-sign and DA-direction (opposite-sign) diagonals at slopes 1:1, 1:2, 2:1, 1:3, 3:1), the one with the widest t80-ratio range over its 'scaling' cells: working
(noisy responding within 10 points of control, noise-free t80 > 0) with scale-only warp R2 >= 0.9 and >= shift-only R2.
The ridge is that line's contiguous run of scaling cells through (0, 0). Activity: noise-free cue trial, mean rate per area
(nln(x)), SNc as mean nln(x_nm), plus the output y; top row vs time after cue onset, bottom row vs time / t80 ratio
(curves collapse if the change is a pure temporal scaling about cue onset).
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gain_sweep_2d as G
import gain_grid_2d as GG
import pickle
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

n = len(GG.DEL); c0 = GG.I0
# candidate lines through (0, 0): grid-index steps (a, b) = (dSPN, iSPN); a/b > 0 same sign, < 0 DA-like (opposite sign)
STEPS = {'dSPN only': (1, 0), 'iSPN only': (0, 1), 'same-sign diag': (1, 1), 'DA direction': (1, -1),
         'same 1:2': (1, 2), 'same 2:1': (2, 1), 'same 1:3': (1, 3), 'same 3:1': (3, 1),
         'DA 1:2': (1, -2), 'DA 2:1': (2, -1), 'DA 1:3': (1, -3), 'DA 3:1': (3, -1)}


def _line(a, b):
    pts = [(c0 + k * a, c0 + k * b) for k in range(-n, n + 1)]
    return [p for p in pts if 0 <= p[0] < n and 0 <= p[1] < n]


LINES = {k: _line(*v) for k, v in STEPS.items()}
AREAS = [('dSPN', slice(0, 10)), ('iSPN', slice(10, 20)), ('Cortex', slice(20, 40)), ('Thalamus', slice(40, 60))]


def scaling_mask(z, S):
    ok = (np.abs(z['respond'] - S['control']['respond']) < 0.10) & np.isfinite(z['t80_ratio']) & (z['t80_ratio'] > 0)
    sc = ok & (z['scale_r2'] >= 0.9) & (z['scale_r2'] >= z['shift_r2'])
    sc[c0, c0] = True  # control
    return sc


def ridge(z, S, all_lines=False):
    sc = scaling_mask(z, S)
    best = None; runs = {}
    for name, pts in LINES.items():
        k0 = pts.index((c0, c0))
        lo = k0
        while lo - 1 >= 0 and sc[pts[lo - 1]]:
            lo -= 1
        hi = k0
        while hi + 1 < len(pts) and sc[pts[hi + 1]]:
            hi += 1
        run = pts[lo:hi + 1]
        tr = [z['t80_ratio'][p] for p in run]
        rng = max(tr) - min(tr)
        runs[name] = (rng, run)
        if best is None or rng > best[1]:
            best = (name, rng, run)
    return (best, runs) if all_lines else best


def plot_lines(z, S, best_name, runs, path, title):
    """All candidate lines: scaling-cell map with the lines drawn, and t80 ratio along each line (scaling cells filled)."""
    sc = scaling_mask(z, S)
    fig = plt.figure(figsize=(13, 6.5))
    ax0 = fig.add_axes([0.03, 0.12, 0.27, 0.76])
    tr = np.where(sc, z['t80_ratio'], np.nan)
    im = ax0.imshow(tr, origin='lower', cmap='coolwarm', vmin=0.7, vmax=1.3)
    fig.colorbar(im, ax=ax0, shrink=0.7, label='t80 ratio (scaling cells)')
    cols = plt.cm.tab20(np.linspace(0, 1, len(LINES)))
    for (name, pts), c in zip(LINES.items(), cols):
        lw = 2.2 if name == best_name else 0.7
        ax0.plot([p[1] for p in pts], [p[0] for p in pts], color=c, lw=lw, alpha=0.9)
    lab = [('%g' % d) if abs(d) in (0.25, 0.064, 0.016, 0.004) or d == 0 else '' for d in GG.DEL]
    ax0.set_xticks(range(n)); ax0.set_xticklabels(lab, fontsize=5, rotation=90); ax0.set_yticks(range(n)); ax0.set_yticklabels(lab, fontsize=5)
    ax0.set_xlabel('δ iSPN'); ax0.set_ylabel('δ dSPN'); ax0.set_xlim(-0.5, n - 0.5); ax0.set_ylim(-0.5, n - 0.5)
    ax0.set_title(f'scaling cells; bold = ridge ({best_name})', fontsize=8)
    for k, ((name, pts), c) in enumerate(zip(LINES.items(), cols)):
        ax = fig.add_axes([0.37 + (k % 4) * 0.158, 0.66 - (k // 4) * 0.27, 0.13, 0.2])
        a, b = STEPS[name]
        x = np.array([(p[0] - c0) / a if a else (p[1] - c0) / b for p in pts])  # signed step index along the line
        t = np.array([z['t80_ratio'][p] for p in pts]); m = np.array([sc[p] for p in pts])
        ax.plot(x, np.where(np.isfinite(t), t, np.nan), '-', color=c, lw=0.8)
        ax.plot(x[m], t[m], 'o', color=c, ms=3); ax.plot(x[~m & np.isfinite(t)], t[~m & np.isfinite(t)], 'o', mfc='none', mec=c, ms=3)
        ax.axhline(1, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3); ax.set_ylim(0.5, 1.6)
        rng = runs[name][0]
        ax.set_title(f"{name} (a,b)=({a},{b}): range {rng:.2f}", fontsize=6.5, fontweight='bold' if name == best_name else 'normal')
        ax.tick_params(labelsize=5); ax.set_xlabel('step along line', fontsize=5)
    fig.suptitle(f'{title} — t80 ratio along 12 lines through (0, 0); filled = scaling cell, open = not scaling', fontsize=9)
    fig.savefig(path, dpi=250); plt.close(fig)


def sim_full(p):
    keys = jr.split(jr.PRNGKey(0), 1)
    y, (xbg, xc, xt), xnm = G._run(p, G._inp, 0.0, keys)
    return np.asarray(y[0, :, 0]), G.rate(np.asarray(jnp.concatenate([xbg, xc, xt], -1)[0])), G.rate(np.asarray(xnm[0]))


def main(sd, s):
    z = np.load(f'{sd}/metrics_gaingrid.npz'); S = json.load(open(f'{sd}/metrics_gaingrid.json'))
    if not S['control']['t80_s'] > 0:
        return None
    (name, rng, run), runs = ridge(z, S, all_lines=True)
    plot_lines(z, S, name, runs, f'{sd}/plots/png/gain_grid_lines.png', f'seed {s}')
    p = pickle.load(open(f'{sd}/params_nm.pkl', 'rb'))
    t = (np.arange(G.T) - G.CUE) / 100
    trs = np.array([z['t80_ratio'][q] for q in run])
    norm = plt.Normalize(min(trs.min(), 0.99), max(trs.max(), 1.01))
    fig, axs = plt.subplots(2, 6, figsize=(15, 5), sharex='row')
    for (i, j), tr in zip(run, trs):
        q = p if (i, j) == (c0, c0) else GG.with_add(p, GG.DEL[i], GG.DEL[j])
        y, x, xnm = sim_full(q)
        col = 'k' if (i, j) == (c0, c0) else plt.cm.coolwarm(norm(tr))
        lw = 1.6 if (i, j) == (c0, c0) else 0.8
        traces = [x[:, sl].mean(1) for _, sl in AREAS] + [xnm.mean(1), y]
        for k, v in enumerate(traces):
            axs[0, k].plot(t, v, color=col, lw=lw)
            axs[1, k].plot(t / tr, v, color=col, lw=lw)
    for k, ttl in enumerate([a for a, _ in AREAS] + ['SNc', 'output y']):
        axs[0, k].set_title(ttl, fontsize=9)
        for r in (0, 1):
            axs[r, k].axvline(0, color='k', lw=0.4); axs[r, k].set_xlim(-0.5, 5)
        axs[0, k].axvline(S['control']['t80_s'], color='k', lw=0.4, ls=':')
        axs[1, k].axvline(S['control']['t80_s'], color='k', lw=0.4, ls=':')
        axs[0, k].set_xlabel('time after cue onset (s)', fontsize=7); axs[1, k].set_xlabel('time / t80 ratio (s)', fontsize=7)
    axs[0, 0].set_ylabel('mean rate (noise-free)'); axs[1, 0].set_ylabel('mean rate, time-rescaled')
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'); fig.colorbar(sm, ax=axs, shrink=0.6, label='t80 ratio')
    dl = [(GG.DEL[i], GG.DEL[j]) for i, j in run]
    fig.suptitle(f'seed {s} — scaling ridge: {name}, {len(run)} cells, δ (dSPN, iSPN) from {dl[0]} to {dl[-1]}, '
                 f't80 ratio {trs.min():.2f}–{trs.max():.2f}; black = trained, dotted = control t80', fontsize=9)
    fig.savefig(f'{sd}/plots/png/gain_grid_ridge.png', dpi=250, bbox_inches='tight'); plt.close(fig)
    R = dict(seed=s, line=name, n_cells=len(run), deltas=dl, t80_ratio=trs.tolist(),
             scale_r2=[float(z['scale_r2'][q]) for q in run], shift_r2=[float(z['shift_r2'][q]) for q in run])
    json.dump(R, open(f'{sd}/metrics_gaingrid_ridge.json', 'w'), indent=1)
    return R


if __name__ == '__main__':
    D = os.path.join(G.ROOT, 'seeds', sys.argv[1])
    seeds = list(map(int, sys.argv[2:])) or sorted(int(os.path.basename(os.path.dirname(q)).split('_')[-1]) for q in glob.glob(f'{D}/seed_*/metrics_gaingrid.npz'))
    for s in seeds:
        R = main(f'{D}/seed_{s}', s)
        print(s, 'no ramp' if R is None else f"{R['line']}: {R['n_cells']} cells, t80 {min(R['t80_ratio']):.2f}-{max(R['t80_ratio']):.2f}", flush=True)
