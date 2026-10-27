"""Signed effective connectivity matrix of each network (CPU, no simulation).

usage (repo root): python experiments/connectivity.py [seed ...]   (default: every network with PNR-determinant data)
Builds W (rows = targets, columns = sources) with the signs the model applies (model_functions.multiregion_nmrnn):
  cortex   <- cortex J_c (unsigned), thalamus exc(B_ct), cue B_cu
  striatum <- striatum inh(J_bg), cortex exc(B_bgc)        (shown at DA gain G = 1 and population gain 1)
  thalamus <- thalamus J_t (unsigned), dSPN exc(B_tbg[:, :10]), iSPN inh(B_tbg[:, 10:])
  SNc      <- SNc J_nm (unsigned), cortex exc(B_nmc), striatum inh(B_nmbg)
  output y <- thalamus exc(C)
Thalamic units are sorted by readout weight |C| (largest first) in rows and columns. Color scale per network is
symmetric at the 99th percentile of |W|.
Writes experiments/connectivity/conn_seed<k>.png (per network), all_networks.png (grid), block_means.png (area-level
mean signed weight, all networks) and matrices.npz.
"""
import os, sys, glob, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
OUT = os.path.join(EXP, 'connectivity'); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(a) for a in sys.argv[1:]] or sorted(int(os.path.basename(d)[4:].split('_')[0]) for d in glob.glob(os.path.join(EXP, 'pnr_determinants', 'seed*_intact')))
AREAS = [('D1', 10), ('D2', 10), ('Cortex', 20), ('Thalamus', 20), ('SNc', 5)]
OFF = np.cumsum([0] + [n for _, n in AREAS]); N = OFF[-1]
sl = {a: slice(OFF[i], OFF[i + 1]) for i, (a, _) in enumerate(AREAS)}
try:
    SIM = {int(k): v['similarity'] for k, v in json.load(open(os.path.join(EXP, 'report', 'figs', 'seed_comparison.json'))).items()}
except Exception:
    SIM = {}


def params_of(k):
    p = (os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl') if k == 13
         else os.path.join(ROOT, 'seeds', '3k', f'seed_{k}', 'params_nm.pkl'))
    return {a: np.asarray(v, np.float64) for a, v in pickle.load(open(p, 'rb')).items()}


def build(P):
    W = np.zeros((N + 1, N + 1))                     # last row = output y, last column = cue
    a = np.abs
    bg = slice(0, 20)
    W[sl['Cortex'], sl['Cortex']] = P['J_c']; W[sl['Cortex'], sl['Thalamus']] = a(P['B_ct']); W[sl['Cortex'], N] = P['B_cu'][:, 0]
    W[bg, bg] = -a(P['J_bg']); W[bg, sl['Cortex']] = a(P['B_bgc'])
    W[sl['Thalamus'], sl['Thalamus']] = P['J_t']; W[sl['Thalamus'], sl['D1']] = a(P['B_tbg'][:, :10]); W[sl['Thalamus'], sl['D2']] = -a(P['B_tbg'][:, 10:])
    W[sl['SNc'], sl['SNc']] = P['J_nm']; W[sl['SNc'], sl['Cortex']] = a(P['B_nmc']); W[sl['SNc'], bg] = -a(P['B_nmbg'])
    W[N, sl['Thalamus']] = a(P['C'][0])
    order = np.argsort(-a(P['C'][0]))                # thalamus sorted by readout weight
    idx = np.r_[np.arange(OFF[3]), OFF[3] + order, np.arange(OFF[4], N + 1)]
    return W[np.ix_(idx, idx)], a(P['C'][0])[order]


def draw(ax, W, title, cbar=True, small=False):
    v = np.percentile(np.abs(W[W != 0]), 99)
    im = ax.imshow(W, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
    for o in OFF[1:]:
        ax.axhline(o - 0.5, color='k', lw=0.6); ax.axvline(o - 0.5, color='k', lw=0.6)
    mids = [(OFF[i] + OFF[i + 1]) / 2 - 0.5 for i in range(len(AREAS))]
    labs = [a for a, _ in AREAS]
    ax.set_xticks(mids + [N], labs + ['cue'], fontsize=6 if small else 8, rotation=90 if small else 0)
    ax.set_yticks(mids + [N], labs + ['y'], fontsize=6 if small else 8)
    ax.set_title(title, fontsize=8 if small else 11)
    if not small:
        ax.set_xlabel('source (presynaptic)'); ax.set_ylabel('target (postsynaptic)')
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='signed weight')
    return im


def lab(k):
    s = f' (similarity {SIM[k]:.0f})' if k in SIM else ''
    return ('baseline, seed 13' if k == 13 else f'seed {k}') + s


Ws, CW, BLK = {}, {}, {}
SEEDS = sorted(SEEDS, key=lambda k: -SIM.get(k, 0))
for k in SEEDS:
    W, cw = build(params_of(k)); Ws[k], CW[k] = W, cw
    fig = plt.figure(figsize=(10.5, 9.5))
    ax = fig.add_axes([0.08, 0.08, 0.78, 0.84])
    draw(ax, W, f'{lab(k)}: signed connectivity (thalamus sorted by readout |C|; BG at DA gain 1)')
    fig.savefig(os.path.join(OUT, f'conn_seed{k}.png'), dpi=110); plt.close(fig)
    blk = np.zeros((len(AREAS) + 1, len(AREAS) + 1))
    names = [a for a, _ in AREAS]
    for i, a in enumerate(names + ['y']):
        for j, b in enumerate(names + ['cue']):
            r_ = sl[a] if a != 'y' else slice(N, N + 1); c_ = sl[b] if b != 'cue' else slice(N, N + 1)
            blk[i, j] = W[r_, c_].sum(1).mean()       # mean over targets of summed input weight from that area
    BLK[k] = blk

nc = 4; nr = int(np.ceil(len(SEEDS) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr))
for ax, k in zip(axs.ravel(), SEEDS):
    draw(ax, Ws[k], lab(k), cbar=False, small=True)
for ax in axs.ravel()[len(SEEDS):]:
    ax.axis('off')
fig.suptitle('Signed connectivity, all networks (rows = targets; red excitatory, blue inhibitory; per-network scale)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'all_networks.png'), dpi=90); plt.close(fig)

names = [a for a, _ in AREAS]
fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.8 * nr))
vmax = np.percentile(np.abs(np.stack(list(BLK.values()))), 98)
for ax, k in zip(axs.ravel(), SEEDS):
    b = BLK[k]; ax.imshow(b, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] != 0:
                ax.text(j, i, f'{b[i, j]:.1f}', ha='center', va='center', fontsize=6)
    ax.set_xticks(range(len(names) + 1), names + ['cue'], fontsize=6, rotation=90); ax.set_yticks(range(len(names) + 1), names + ['y'], fontsize=6)
    ax.set_title(lab(k), fontsize=8)
for ax in axs.ravel()[len(SEEDS):]:
    ax.axis('off')
fig.suptitle('Area-level connectivity: summed signed input from each source area, averaged over target units (shared scale)', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'block_means.png'), dpi=90); plt.close(fig)
np.savez(os.path.join(OUT, 'matrices.npz'), seeds=np.array(SEEDS), W=np.stack([Ws[k] for k in SEEDS]), readout_sorted=np.stack([CW[k] for k in SEEDS]),
         blocks=np.stack([BLK[k] for k in SEEDS]), areas=np.array(names))
print('wrote', len(SEEDS), 'networks to', OUT)
