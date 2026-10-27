"""Response-time CDFs and histograms for every PNR test (cue magnitude, cue duration, stop pulse, dSPN pulse).

usage (from repo root, GPU): python experiments/pnr_rt_plots.py [net ...]   nets: baseline pd_acute pd hd_acute hd (default all)
Re-simulates each test with the exact noise keys of test_pnr_nm.py / run_experiment.py dspn_pnr (solo GPU runs are
deterministic), then writes, per test, rt_<test>.png and rts_<test>.npz into <network>/pnr/ and <network>/dspn_pnr/.
RT = first output crossing of 0.5 at or after cue onset (t=300; cue magnitude from cue end, t=310, as test_pnr_nm
counts it), in seconds from cue onset. A trial already above threshold at that time has RT 0.
CDFs are over all trials, so each curve plateaus at P(respond). rt_figure() is also called by run_experiment.py.
"""
import os, sys
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)


def rts_from(ys, t0):
    """ys (seeds, conds, T) -> RT in s from cue onset (t=300), NaN = never crosses; search starts at t0."""
    above = ys[..., t0:] > 0.5
    return np.where(above.any(-1), (above.argmax(-1) + t0 - 300) / 100, np.nan)


def rt_figure(ys, labels, t0, title, path, legend_title, marks=None):
    """CDF (all conditions) + one histogram row per condition. marks: {cond index: pulse onset in s} drawn on its row."""
    import matplotlib.pyplot as plt
    rt = rts_from(np.asarray(ys), t0)
    n, nc = rt.shape
    np.savez(path.replace('.png', '.npz').replace('/rt_', '/rts_'), rt_s=rt, labels=np.array(labels, dtype=object), t0=t0)
    col = plt.cm.plasma(np.linspace(0, 0.9, nc))
    fig = plt.figure(figsize=(13, max(4.5, 0.32 * nc + 1.5)))
    a0 = fig.add_axes([0.06, 0.12, 0.38, 0.78]); a1 = fig.add_axes([0.53, 0.12, 0.44, 0.78])
    bins = np.arange(0, 6.01, 0.1)
    for j in range(nc):
        r = np.sort(rt[:, j][np.isfinite(rt[:, j])])
        if len(r):
            a0.step(np.concatenate([[0], r, [6]]), np.concatenate([[0], np.arange(1, len(r) + 1), [len(r)]]) / n,
                    where='post', color=col[j], lw=1.2, label=labels[j])
        h, _ = np.histogram(r, bins)
        h = h / n / 0.1                                            # density per trial: area = P(respond in window)
        scale = 0.9 / max(1e-9, max((np.histogram(rt[:, k][np.isfinite(rt[:, k])], bins)[0] / n / 0.1).max() for k in range(nc)))
        a1.fill_between(bins[:-1], j, j + h * scale, step='post', color=col[j], alpha=0.75, lw=0)
        a1.text(6.05, j + 0.1, f'{np.isfinite(rt[:, j]).mean():.2f}', fontsize=7, va='bottom')
        if marks and j in marks:
            a1.plot([marks[j]] * 2, [j, j + 0.9], color='k', lw=1)
    a0.set(xlim=(0, 6), ylim=(0, 1.02), xlabel='RT from cue onset (s)', ylabel='cumulative P(respond)', title=f'{title}: RT CDF')
    a0.legend(fontsize=6.5, title=legend_title, title_fontsize=7, ncol=2 if nc > 12 else 1, loc='upper left')
    a1.set(xlim=(0, 6), ylim=(-0.2, nc + 0.2), xlabel='RT from cue onset (s)', title=f'{title}: RT histograms (100 ms bins)')
    a1.set_yticks(np.arange(nc) + 0.3, labels, fontsize=7); a1.set_ylabel(legend_title)
    a1.text(6.05, nc, 'P(resp)', fontsize=7)
    if marks:
        a1.plot([], [], color='k', lw=1, label='pulse onset'); a1.legend(fontsize=7, loc='upper right')
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == '__main__':
    import json, pickle
    sys.path[:0] = [EXP, ROOT]
    import matplotlib
    matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
    import jax, jax.numpy as jnp, jax.random as jr
    os.chdir(ROOT)
    import config_script as cs
    import model_functions as mf
    import perturb_model as pm
    import test_pnr_nm as tp
    NETDIR = {'baseline': 'baseline', 'pd_acute': 'pd_snc_lesion/acute', 'pd': 'pd_snc_lesion/retrained',
              'hd_acute': 'hd_ispn_lesion/acute', 'hd': 'hd_ispn_lesion/retrained'}
    BASE = os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl')
    T, N_BG, N1 = cs.config['T'], cs.config['n_bg'], cs.n_d1_cells
    for net in sys.argv[1:] or list(NETDIR):
        les = os.path.join(EXP, NETDIR[net].split('/')[0], 'lesion.json')
        lb = ln = None
        if net != 'baseline':
            idx = jnp.array(json.load(open(les))['lesioned_idx'])
            if net.startswith('pd'):
                ln = jnp.zeros(cs.config['n_nm'], bool).at[idx].set(True)
            else:
                lb = jnp.zeros(N_BG, bool).at[idx].set(True)
        pm.install(lb, ln)
        params = pickle.load(open(BASE if net in ('baseline', 'pd_acute', 'hd_acute') else
                                  os.path.join(EXP, NETDIR[net].split('/')[0], 'training', 'params_nm.pkl'), 'rb'))
        pdir, ddir = os.path.join(EXP, NETDIR[net], 'pnr'), os.path.join(EXP, NETDIR[net], 'dspn_pnr')
        os.makedirs(pdir, exist_ok=True); os.makedirs(ddir, exist_ok=True)
        print(net, 'cue magnitude'); mags = jnp.linspace(0.0, 2.0, 21)
        ys = np.asarray(tp.test_cue_magnitude(params, mags, 150)[3][..., 0])
        rt_figure(ys, [f'{m:.1f}' for m in np.asarray(mags)], 310, f'{net} cue magnitude', os.path.join(pdir, 'rt_cue_magnitude.png'), 'cue magnitude')
        print(net, 'cue duration'); durs = jnp.concatenate((jnp.array([0]), jnp.arange(2, 32, 2)))
        ys = np.asarray(tp.test_cue_duration(params, durs, 150)[3][..., 0])
        rt_figure(ys, [f'{int(d) * 10} ms' for d in np.asarray(durs)], 300, f'{net} cue duration', os.path.join(pdir, 'rt_cue_duration.png'), 'cue duration')
        print(net, 'stop pulse'); pts = jnp.arange(320, 600, 20)
        ys = ys_stop = np.asarray(tp.test_dynamic_perturbation(params, pts, n_seeds=150)[3][..., 0])
        dl = (np.asarray(pts) - 300) / 100
        rt_figure(ys, ['no cue'] + [f'{d:.1f} s' for d in dl], 300, f'{net} stop pulse (−5, all SPNs)',
                  os.path.join(pdir, 'rt_stop_pulse.png'), 'pulse delay', marks={i + 1: d for i, d in enumerate(dl)})
        # dSPN pulse: same layout and keys as run_experiment.py dspn_pnr
        nC = len(pts) + 2
        base, _, _ = mf.self_timed_movement_task(jnp.array([300]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)
        stim = np.zeros((nC, T, N_BG), np.float32)
        for i, p in enumerate(np.asarray(pts)):
            stim[i + 2, p:p + 5, :N1] = 1.0
        inputs = jnp.repeat(base, nC, 0).at[0].set(0.0)
        sim = jax.jit(lambda k: mf.batched_nm_rnn(params, cs.x0, cs.z0, inputs, cs.config['tau_x'], cs.config['tau_z'], True,
                                                  jnp.asarray(stim), cs.test_noise_std, k))
        print(net, 'dSPN pulse')
        ys = np.stack([np.asarray(sim(jnp.repeat(jr.PRNGKey(s + 200)[None], nC, 0))[0][..., 0]) for s in range(150)])
        rt_figure(ys, ['no cue', 'cue only'] + [f'{d:.1f} s' for d in dl], 300, f'{net} dSPN pulse (+1)',
                  os.path.join(ddir, 'rt_dspn_pulse.png'), 'pulse delay', marks={i + 2: d for i, d in enumerate(dl)})
        # consistency with the stored metrics
        mp = os.path.join(ddir, 'metrics_dspn_pnr.json')
        if os.path.exists(mp):
            m = json.load(open(mp)); r = np.isfinite(rts_from(ys, 300)).mean(0)
            print(net, 'dSPN respond max diff vs stored metrics:', float(np.max(np.abs(r[2:] - [p['respond'] for p in m['pulses']]))))
        mp = os.path.join(pdir, 'metrics_pnr.json')
        if os.path.exists(mp):
            r = np.isfinite(rts_from(ys_stop, 300)).mean(0)
            print(net, 'stop-pulse respond max diff vs stored metrics:', float(np.max(np.abs(r[1:] - json.load(open(mp))['pert_prob']))))
