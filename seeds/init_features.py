"""Features of each seed's *initial* weights and untrained dynamics, joined with its trained outcome.

usage (from repo root): JAX_PLATFORMS=cpu python seeds/init_features.py [subfolder, default 3k]
Writes seeds/<subfolder>/init_features.csv and init_features.png (each feature vs. trained response rate).
Initial weights are rebuilt with config_script.init_params(PRNGKey(seed)), exactly as training_script does.
"""
import os, sys, json, glob, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
import config_script as cs
import model_functions as mf

SUB = sys.argv[1] if len(sys.argv) > 1 else '3k'
HERE = os.path.join(ROOT, 'seeds', SUB)
cfg = cs.config


def eig_max(W):
    return float(np.max(np.real(np.linalg.eigvals(np.asarray(W)))))


def features(seed):
    p = cs.init_params(jr.PRNGKey(seed), cfg['n_bg'], cfg['n_nm'], cfg['g_bg'], cfg['g_nm'], cfg['U'], cfg['O'])
    n1 = cs.n_d1_cells
    f = dict(
        eig_Jc=eig_max(p['J_c']), eig_Jt=eig_max(p['J_t']), eig_Jnm=eig_max(p['J_nm']),
        eig_Jbg_inh=eig_max(mf.inh(p['J_bg'])),
        cue_in_norm=float(jnp.linalg.norm(p['B_cu'])), cue_in_pos_frac=float(jnp.mean(p['B_cu'] > 0)),
        readout_norm=float(jnp.linalg.norm(p['C'])), rb=float(p['rb'][0]),
        snc_c=float(p['c']), snc_m_norm=float(jnp.linalg.norm(p['m'])),
        d1_to_thal=float(jnp.abs(p['B_tbg'][:, :n1]).sum()), d2_to_thal=float(jnp.abs(p['B_tbg'][:, n1:]).sum()),
    )
    # untrained, noiseless response to one cue at t=300
    inp, _, _ = mf.self_timed_movement_task(jnp.array([300]), cfg['T_cue'], cfg['T_wait'], cfg['T_movement'], cfg['T'])
    ys, xs, zs = mf.batched_nm_rnn(p, cs.x0, cs.z0, inp, cfg['tau_x'], cfg['tau_z'], True, None, 0.0, jr.split(jr.PRNGKey(0), 1))
    y = np.asarray(ys[0, :, 0])
    s = np.asarray(jax.nn.sigmoid(mf.nln(zs[0]) @ mf.exc(p['m']).T + p['c']))[:, 0]
    f.update(y_baseline=float(y[250:300].mean()), y_peak_post_cue=float(y[310:].max()),
             y_peak_time_s=float((np.argmax(y[310:]) + 10) / 100), s_baseline=float(s[250:300].mean()),
             s_cue_shift=float(s[310:600].mean() - s[250:300].mean()))
    for i, a in enumerate(['bg', 'c', 't']):
        r = np.asarray(mf.nln(xs[i][0]))
        f[f'active_{a}'] = float(np.mean(r[250:300].mean(0) > 0.01))
    return f


rows = []
for d in sorted(filter(os.path.isdir, glob.glob(os.path.join(HERE, 'seed_*'))), key=lambda d: int(d.split('_')[-1])):
    seed = int(d.split('_')[-1])
    mt, tr = os.path.join(d, 'metrics_test.json'), os.path.join(d, 'metrics_train.json')
    if not os.path.exists(mt):
        continue
    te, trn = json.load(open(mt)), json.load(open(tr))
    losses = np.load(os.path.join(d, 'losses.npy'))
    row = dict(seed=seed, respond=te['respond'], on_time=te['on_time_2p5_3p5'],
               best_loss=trn['best_logged_loss'], best_step=trn['best_step'],
               early_loss_400=float(losses[:400].mean()), frac_below_0p010=trn['frac_steps_below_0p010'])
    row.update(features(seed))
    rows.append(row)
    print(row['seed'], f"respond {row['respond']:.2f}")

keys = list(rows[0].keys())
with open(os.path.join(HERE, 'init_features.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)

feat = [k for k in keys if k not in ('seed', 'respond', 'on_time')]
resp = np.array([r['respond'] for r in rows])
nc = 5
fig, axs = plt.subplots(int(np.ceil(len(feat) / nc)), nc, figsize=(3 * nc, 2.4 * np.ceil(len(feat) / nc)))
print('\nSpearman rho of each feature with trained response rate (n=%d):' % len(rows))
from scipy.stats import spearmanr
out = []
for ax, k in zip(axs.flat, feat):
    v = np.array([r[k] for r in rows])
    rho, pval = spearmanr(v, resp) if np.std(v) > 0 else (np.nan, np.nan)
    out.append((k, rho, pval))
    ax.scatter(v, resp, s=14, c=['red' if r['seed'] == 13 else 'k' for r in rows])
    ax.set_title(f'{k}\nrho={rho:.2f} p={pval:.2g}', fontsize=8); ax.tick_params(labelsize=7)
for ax in list(axs.flat)[len(feat):]:
    ax.axis('off')
fig.supylabel('trained response rate'); fig.tight_layout()
fig.savefig(os.path.join(HERE, 'init_features.png'), dpi=120)
for k, rho, pval in sorted(out, key=lambda t: -abs(np.nan_to_num(t[1]))):
    print(f'{k:20s} rho={rho:+.2f} p={pval:.3g}')
