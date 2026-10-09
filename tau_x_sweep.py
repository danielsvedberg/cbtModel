"""Sweep tau_x and record why trainability changes, not just whether.

Train one value with per-step gradient diagnostics (run from the code version you want to test):
    python tau_x_sweep.py --tau_x 50 --iters 3000 --seed 1 --out_dir tau_x_sweep
Summarize all runs in out_dir:
    python tau_x_sweep.py --summarize --out_dir tau_x_sweep

The training loop mirrors fit_nm_rnn, including its noise keys: fit_nm_rnn splits its rng key
inside a jitted scan via `nonlocal`, which freezes the key at trace time, so every training step
reuses one noise realization. This loop uses that same fixed realization (verified with --verify).
fit_nm_rnn also discards the optimizer state returned by each lax.scan chunk, so Adam restarts from
its initial state every log_interval steps; that is mirrored too unless --keep_opt_state is given.

Per step it records: loss, raw (pre-clip) gradient norm, cosine to previous step's gradient,
first-order predicted loss change (g . update) vs actual change, and how many trials' threshold-
crossing index (which sets the loss target window) moved. At checkpoints it saves params and
|dL/dx_bg(t)| per trial, i.e. how far back in time the error signal reaches.
"""
import argparse
import os
import sys
import time
import pickle as pkl

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tau_x', type=float)
parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--noise_std', type=float, default=0.05)  # fit_nm_rnn default
parser.add_argument('--checkpoints', type=str, default='0,400,1000,3000')  # multiples of --chunk
parser.add_argument('--chunk', type=int, default=200, help="fit_nm_rnn's log_interval (training_script uses 200)")
parser.add_argument('--keep_opt_state', action='store_true',
                    help="carry Adam state across chunks; default mirrors fit_nm_rnn, which resets it every chunk")
parser.add_argument('--out_dir', default='tau_x_sweep')
parser.add_argument('--verify', action='store_true', help='check the loop reproduces fit_nm_rnn exactly')
parser.add_argument('--summarize', action='store_true')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

CHUNK = args.chunk


def setup():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.use = lambda *a, **k: None  # plotting_functions forces Qt5Agg, which blocks on plt.show()
    sys.path.insert(0, os.getcwd())
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import model_functions as mf
    import config_script as cs

    params, optimizer = cs.params, cs.optimizer
    if args.seed is not None:
        params = cs.init_params(jr.PRNGKey(args.seed), cs.config['n_bg'], cs.config['n_nm'],
                                cs.config['g_bg'], cs.config['g_nm'], cs.config['U'], cs.config['O'])
    if args.lr is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=args.lr))
    inputs, targets, masks = mf.self_timed_movement_task(
        cs.config['T_start'], cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T'])
    # the keys fit_nm_rnn actually uses on every step (frozen at trace time)
    _, subkey = jr.split(jr.PRNGKey(0))
    keys = jr.split(subkey, inputs.shape[0])
    return jax, jnp, optax, mf, cs, params, optimizer, inputs, targets, masks, keys


def make_loss(jnp, mf, cs, inputs, targets, masks, keys, tau_x):
    tau_z = cs.config['tau_z']
    n_bg = cs.config['n_bg']

    def loss_fn(params, stim):
        # same computation as mf.batched_nm_rnn_loss, but with an injectable stim (for dL/dx_bg) and
        # returning the threshold-crossing indices that set the target window
        ys, _, _ = mf.batched_nm_rnn(params, cs.x0, cs.z0, inputs, tau_x, tau_z, True, stim,
                                     args.noise_std, keys)
        T_start_move = jnp.argmax(targets, axis=1)
        Tarray = jnp.arange(inputs.shape[1])
        first_cross = jnp.argmax(ys >= 0.5, axis=1)
        idxs = jnp.where(first_cross < T_start_move, T_start_move, first_cross)
        value_mask = jnp.where(Tarray > (idxs + 50), 0, 1)
        value_mask = jnp.where(Tarray < idxs, 0, value_mask)
        tgt = 0.25 + value_mask[..., None] * (0.75 - 0.25)
        loss = jnp.sum(((ys - tgt) ** 2) * masks) / jnp.sum(masks)
        return loss, first_cross[:, 0]

    zero_stim = jnp.zeros((inputs.shape[0], inputs.shape[1], n_bg))
    return loss_fn, zero_stim


def train():
    jax, jnp, optax, mf, cs, params, optimizer, inputs, targets, masks, keys = setup()
    tau_x = args.tau_x
    loss_fn, zero_stim = make_loss(jnp, mf, cs, inputs, targets, masks, keys, tau_x)

    # sanity: our loss == the repo's loss
    ref = mf.batched_nm_rnn_loss(params, cs.x0, cs.z0, inputs, tau_x, cs.config['tau_z'], targets, masks,
                                 keys, modulation=True, noise_std=args.noise_std)
    ours = loss_fn(params, zero_stim)[0]
    assert abs(float(ref) - float(ours)) < 1e-6 * max(1.0, abs(float(ref))), (ref, ours)

    vg = jax.value_and_grad(lambda p: loss_fn(p, zero_stim), has_aux=True)
    credit = jax.jit(jax.grad(lambda s, p: loss_fn(p, s)[0]))

    def flat(tree):
        return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])

    @jax.jit
    def chunk(carry, _=None):
        def step(c, _):
            params, opt_state, prev_g, prev_cross = c
            (loss, cross), grads = vg(params)
            g = flat(grads)
            gnorm = jnp.linalg.norm(g)
            cos_prev = jnp.vdot(g, prev_g) / (gnorm * jnp.linalg.norm(prev_g) + 1e-12)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            pred = jnp.vdot(g, flat(updates))  # first-order predicted change in loss
            params = optax.apply_updates(params, updates)
            n_cross_moved = jnp.sum(cross != prev_cross)
            return (params, opt_state, g, cross), (loss, gnorm, cos_prev, pred, n_cross_moved)
        return jax.lax.scan(step, carry, None, length=CHUNK)

    checkpoints = sorted({int(c) for c in args.checkpoints.split(',') if c and int(c) <= args.iters})
    tag = (f'tau_x_{tau_x:g}' + (f'_seed_{args.seed}' if args.seed is not None else '')
           + ('_keepopt' if args.keep_opt_state else ''))
    opt_state0 = optimizer.init(params)
    (l0, cross0), g0 = vg(params)
    carry = (params, opt_state0, flat(g0), cross0)
    rec, ckpt = [], {}

    def save_ckpt(step, p):
        g_stim = credit(zero_stim, p) * tau_x  # stim enters as (1/tau_x)*stim, so this is dL/dx_bg(t)
        ckpt[step] = dict(params=jax.device_get(p),
                          credit=np.asarray(jnp.linalg.norm(g_stim, axis=-1)),  # (trials, T)
                          cross=np.asarray(loss_fn(p, zero_stim)[1]))

    if 0 in checkpoints:
        save_ckpt(0, params)
    t = time.time()
    for n in range(args.iters // CHUNK):
        if not args.keep_opt_state:
            carry = (carry[0], opt_state0) + carry[2:]  # fit_nm_rnn behaviour: Adam state reset each chunk
        carry, out = chunk(carry)
        rec.append([np.asarray(o) for o in out])
        step = (n + 1) * CHUNK
        print(f'step {step}, loss {float(out[0][-1]):.5f}, gnorm {float(out[1][-1]):.3f}', flush=True)
        if step in checkpoints:
            save_ckpt(step, carry[0])
    names = ['loss', 'gnorm', 'cos_prev', 'pred_dloss', 'n_cross_moved']
    metrics = {k: np.concatenate([r[i] for r in rec]) for i, k in enumerate(names)}
    metrics['cos_prev'][0] = np.nan  # step 0 compares with itself
    np.savez(os.path.join(args.out_dir, tag + '_metrics.npz'), tau_x=tau_x, **metrics,
             cue_start=np.asarray(jnp.argmax(inputs[..., 0], axis=1)),
             move_start=np.asarray(jnp.argmax(targets[..., 0], axis=1)))
    with open(os.path.join(args.out_dir, tag + '_ckpts.pkl'), 'wb') as f:
        pkl.dump(ckpt, f)
    print(f'{tag} done in {time.time() - t:.0f}s, final loss {metrics["loss"][-500:].mean():.4f}')


def verify():
    """Run fit_nm_rnn and this loop side by side for a few hundred steps; losses must match."""
    jax, jnp, optax, mf, cs, params, optimizer, inputs, targets, masks, keys = setup()
    n = 3 * CHUNK
    _, ref = mf.fit_nm_rnn(inputs, targets, masks, params, optimizer, cs.x0, cs.z0, n, args.tau_x,
                           cs.config['tau_z'], modulation=True, log_interval=CHUNK, noise_std=args.noise_std)
    ref = np.concatenate([np.asarray(r) for r in ref])
    args.iters, args.checkpoints = n, ''
    args.out_dir = os.path.join(args.out_dir, '_verify')
    os.makedirs(args.out_dir, exist_ok=True)
    train()
    tag = (f'tau_x_{args.tau_x:g}' + (f'_seed_{args.seed}' if args.seed is not None else '')
           + ('_keepopt' if args.keep_opt_state else ''))
    ours = np.load(os.path.join(args.out_dir, tag + '_metrics.npz'))['loss']
    print('max |fit_nm_rnn - ours| over', n, 'steps:', np.abs(ref - ours).max())


def summarize():
    import glob
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = {}
    for fn in glob.glob(os.path.join(args.out_dir, 'tau_x_*_metrics.npz')):
        m = re.match(r'tau_x_([\d.]+)(?:_seed_(\d+))?_metrics.npz', os.path.basename(fn))
        runs.setdefault(float(m[1]), {})[m[2]] = (dict(np.load(fn)), fn.replace('_metrics.npz', '_ckpts.pkl'))
    taus = sorted(runs)

    def smooth(x, w=100):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    hdr = ('tau_x  seed  loss@300  last500  min_sm   clip%  cos_prev(<0 %)  actual/pred dL (median)  '
           'cross-moved/step')
    lines = [hdr]
    for tx in taus:
        for seed, (M, _) in sorted(runs[tx].items(), key=lambda kv: str(kv[0])):
            L = M['loss']
            actual = np.diff(L)
            pred = M['pred_dloss'][:-1]
            ok = np.abs(pred) > 1e-9
            ratio = np.median(actual[ok] / pred[ok])
            late = slice(300, None)
            lines.append(f'{tx:5g}  {str(seed):>4}  {L[200:300].mean():8.4f}  {L[-500:].mean():7.4f}  '
                         f'{smooth(L).min():.4f}  {100 * np.mean(M["gnorm"][late] > 1):5.1f}  '
                         f'{np.nanmean(M["cos_prev"][late]):+.2f} ({100 * np.nanmean(M["cos_prev"][late] < 0):3.0f}%)'
                         f'          {ratio:+7.2f}                {M["n_cross_moved"][late].mean():6.1f}')
    table = '\n'.join(lines)
    print(table)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(table + '\n')

    colors = dict(zip(taus, plt.cm.viridis(np.linspace(0, 1, len(taus)))))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    panels = [('loss', 'loss (100-step avg)', True), ('gnorm', 'raw grad norm (100-step avg)', True),
              ('cos_prev', 'cos(g_t, g_t-1) (100-step avg)', False),
              ('n_cross_moved', 'trials whose crossing moved / step', False)]
    for ax, (key, label, logy) in zip(axs.flat, panels):
        for tx in taus:
            for i, (seed, (M, _)) in enumerate(sorted(runs[tx].items(), key=lambda kv: str(kv[0]))):
                y = np.nan_to_num(M[key], nan=0.0)
                S = smooth(y)
                ax.plot(np.arange(len(S)) + 100, S, color=colors[tx], alpha=0.8, linewidth=1,
                        label=f'tau_x={tx:g}' if i == 0 else None)
        ax.set_ylabel(label)
        ax.set_xlabel('training step')
        if logy:
            ax.set_yscale('log')
    axs[0, 0].legend(fontsize=7, ncol=2)

    # credit-assignment profile: |dL/dx_bg| aligned to movement-window start, at init and end
    for ax, which in zip(axs.flat[4:], ['first', 'last']):
        for tx in taus:
            profs = []
            for seed, (M, ck_fn) in runs[tx].items():
                if not os.path.exists(ck_fn):
                    continue
                ck = pkl.load(open(ck_fn, 'rb'))
                if not ck:
                    continue
                step = min(ck) if which == 'first' else max(ck)
                C = ck[step]['credit']
                for trial, ms in enumerate(M['move_start']):
                    lo = ms - 300
                    seg = C[trial, max(lo, 0):ms + 100]
                    if lo < 0:
                        seg = np.concatenate([np.full(-lo, np.nan), seg])
                    profs.append(seg / (np.nanmax(C[trial]) + 1e-12))
            if profs:
                P = np.nanmean(np.array(profs), axis=0)
                ax.plot(np.arange(len(P)) - 300, P, color=colors[tx], label=f'tau_x={tx:g}')
        ax.axvline(0, color='k', lw=0.5)
        ax.axvline(-300, color='gray', lw=0.5, ls='--')
        ax.set_yscale('log')
        ax.set_xlabel('step relative to movement-window start (cue at -300)')
        ax.set_ylabel(f'|dL/dx_bg(t)| / max ({which} checkpoint)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'tau_x_sweep_diagnostics.png'), dpi=200)
    fig.savefig(os.path.join(args.out_dir, 'tau_x_sweep_diagnostics.svg'))


if __name__ == '__main__':
    if args.summarize:
        summarize()
    elif args.verify:
        verify()
    else:
        train()
