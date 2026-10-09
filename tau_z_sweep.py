"""Sweep tau_z and compare trainability.

Train one value (run from the code version you want to test):
    python tau_z_sweep.py --tau_z 50 --iters 5000 --out_dir tau_z_sweep [--lr 1e-3] [--seed 1]
(--lr / --seed override the config's learning rate / parameter-init seed; omitted = config defaults)
Summarize all runs in out_dir (table + loss-curve plot):
    python tau_z_sweep.py --summarize --out_dir tau_z_sweep
"""
import argparse
import os
import sys
import time
import pickle as pkl

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tau_z', type=float)
parser.add_argument('--iters', type=int, default=5000)
parser.add_argument('--lr', type=float)
parser.add_argument('--seed', type=int)
parser.add_argument('--out_dir', default='tau_z_sweep')
parser.add_argument('--summarize', action='store_true')
parser.add_argument('--summarize_grid', action='store_true',
                    help='summarize runs tagged with _lr_ and _seed_, grouped by (tau_z, lr) across seeds')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def train(tau_z):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.use = lambda *a, **k: None  # plotting_functions forces Qt5Agg, which blocks on plt.show()
    sys.path.insert(0, os.getcwd())
    import jax.random as jr
    import optax
    from model_functions import fit_nm_rnn, self_timed_movement_task
    from config_script import config, params, optimizer, x0, z0, init_params

    tag = f'tau_z_{tau_z:g}'
    if args.lr is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=args.lr))
        tag += f'_lr_{args.lr:g}'
    if args.seed is not None:
        params = init_params(jr.PRNGKey(args.seed), config['n_bg'], config['n_nm'],
                             config['g_bg'], config['g_nm'], config['U'], config['O'])
        tag += f'_seed_{args.seed}'

    inputs, outputs, masks = self_timed_movement_task(
        config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T'])
    t = time.time()
    best_params, losses = fit_nm_rnn(inputs, outputs, masks, params, optimizer, x0, z0, args.iters,
                                     config['tau_x'], tau_z, wandb_log=False, modulation=True,
                                     log_interval=100)
    losses = np.concatenate([np.asarray(l) for l in losses])
    np.save(os.path.join(args.out_dir, tag + '_losses.npy'), losses)
    with open(os.path.join(args.out_dir, tag + '_params.pkl'), 'wb') as f:
        pkl.dump(best_params, f)
    print(f'{tag} done in {time.time() - t:.0f}s, final loss {losses[-500:].mean():.4f}')


def summarize():
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = {}
    for fn in glob.glob(os.path.join(args.out_dir, 'tau_z_*_losses.npy')):
        if '_lr_' in fn or '_seed_' in fn:
            continue
        runs[float(os.path.basename(fn).split('_')[2])] = np.load(fn)
    taus = sorted(runs)

    lines = ['tau_z  first100  last500   min     min_step  log-slope/1k(after 200)  late-window CV']
    for tz in taus:
        L = runs[tz]
        steps = np.arange(200, len(L))
        slope = np.polyfit(steps, np.log(L[200:]), 1)[0] * 1000
        late = L[-1000:]
        lines.append(f'{tz:5g}  {L[:100].mean():8.4f}  {L[-500:].mean():7.4f}  {L.min():.4f}  '
                     f'{L.argmin():8d}  {slope:+23.3f}  {late.std() / late.mean():14.2f}')
    table = '\n'.join(lines)
    print(table)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(table + '\n')

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))
    for c, tz in zip(colors, taus):
        L = runs[tz]
        smooth = np.convolve(L, np.ones(100) / 100, mode='valid')
        ax.plot(np.arange(len(smooth)) + 100, smooth, color=c, label=f'tau_z={tz:g}')
    ax.set_yscale('log')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss (100-step moving avg)')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'tau_z_sweep_losses.png'), dpi=200)
    fig.savefig(os.path.join(args.out_dir, 'tau_z_sweep_losses.svg'))


def summarize_grid():
    import glob
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = {}  # (tau_z, lr) -> {seed: losses}
    for fn in glob.glob(os.path.join(args.out_dir, 'tau_z_*_lr_*_seed_*_losses.npy')):
        m = re.match(r'tau_z_([\d.]+)_lr_([\de.-]+)_seed_(\d+)_losses.npy', os.path.basename(fn))
        runs.setdefault((float(m[1]), float(m[2])), {})[int(m[3])] = np.load(fn)

    def smooth(L):
        return np.convolve(L, np.ones(100) / 100, mode='valid')

    # rebound = final loss / best 100-step-smoothed loss; >>1 means a good solution was found then lost
    lines = ['tau_z  lr      seed  last500  min_smoothed  rebound']
    for key in sorted(runs):
        stats = []
        for seed in sorted(runs[key]):
            L = runs[key][seed]
            last, best = L[-500:].mean(), smooth(L).min()
            stats.append((last, best, last / best))
            lines.append(f'{key[0]:5g}  {key[1]:<6g}  {seed:4d}  {last:7.4f}  {best:12.4f}  {last / best:7.2f}')
        stats = np.array(stats)
        lines.append(f'{key[0]:5g}  {key[1]:<6g}  mean  {stats[:, 0].mean():7.4f}  '
                     f'{stats[:, 1].mean():12.4f}  {stats[:, 2].mean():7.2f}   (n={len(stats)})')
        lines.append('')
    table = '\n'.join(lines)
    print(table)
    with open(os.path.join(args.out_dir, 'summary_grid.txt'), 'w') as f:
        f.write(table + '\n')

    taus = sorted({k[0] for k in runs})
    lrs = sorted({k[1] for k in runs})
    colors = dict(zip(lrs, plt.cm.tab10(np.arange(len(lrs)))))
    fig, axs = plt.subplots(1, len(taus), figsize=(5 * len(taus), 4), sharey=True, squeeze=False)
    for ax, tz in zip(axs[0], taus):
        for lr in lrs:
            for i, (seed, L) in enumerate(sorted(runs.get((tz, lr), {}).items())):
                S = smooth(L)
                ax.plot(np.arange(len(S)) + 100, S, color=colors[lr], alpha=0.8, linewidth=1,
                        label=f'lr={lr:g}' if i == 0 else None)
        ax.set_yscale('log')
        ax.set_title(f'tau_z={tz:g}')
        ax.set_xlabel('training step')
        ax.legend(fontsize=8)
    axs[0, 0].set_ylabel('loss (100-step moving avg)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'lr_seed_grid_losses.png'), dpi=200)
    fig.savefig(os.path.join(args.out_dir, 'lr_seed_grid_losses.svg'))


if __name__ == '__main__':
    if args.summarize:
        summarize()
    elif args.summarize_grid:
        summarize_grid()
    else:
        train(args.tau_z)
