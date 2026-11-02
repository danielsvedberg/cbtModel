"""SNc (nm) opto test and paired timing-significance test, mirroring mf.simulate_opto for striatum.

Run through run_seed.py (stage 'nmopto'), so cwd is the seed folder and config_script is the seed's.

Model: mf.multiregion_nmrnn is copied by patching its source so that the opto input has n_bg + n_nm
columns; the first n_bg go to x_bg exactly as before, the last n_nm are added to x_nm as (1/tau_z) * stim.
The patch tracks whatever model_functions.py currently contains. verify() checks that with zero SNc input
the output is bit-identical to mf.batched_nm_rnn on the same keys before anything else runs.

Protocol (same as the striatal opto): cue at cs.opto_tstart, stimulus from 1.0 to 2.75 s after cue onset,
cs.n_opto_seeds trials with keys split from PRNGKey(0), so the 0-strength control here is the same trials
as the striatal opto control. The stimulus is uniform over all SNc units.

Timing test (per network): trials are paired by noise key across conditions. Only trials that have not
responded before stim onset are analysed (earlier trials are identical in every condition).
  - response probability: exact McNemar test (binomial on discordant pairs)
  - timing, censored: paired Wilcoxon on RT with no-response set to the trial end (6.5 s)
  - timing, responders: paired Wilcoxon on trials that respond in both conditions
  - effect sizes: median paired RT shift in ms; Holm correction across the conditions within each test
  - dose-response: Spearman of signed strength vs. median censored shift across conditions
"""
import inspect
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from scipy import stats
import matplotlib.pyplot as plt
import model_functions as mf
import config_script as cs

STRENGTHS = [6.0, 3.0, 2.0, 1.0, 0.5]  # inh uses -s, stim +s
CONDS = [('control', 0.0)] + [('inh SNc', -s) for s in STRENGTHS] + [('stim SNc', s) for s in STRENGTHS]
T_END_S = (cs.config['T'] - cs.opto_tstart) / 100  # censoring value for no response


def _build_model():
    src = inspect.getsource(mf.multiregion_nmrnn)
    subs = [
        ('def multiregion_nmrnn(', 'def multiregion_nmrnn_nmstim('),
        ('opto_stimulation = jnp.zeros((T, num_bg_cells))', 'opto_stimulation = jnp.zeros((T, num_bg_cells + J_nm.shape[0]))'),
        ('x_bg += (1. / tau_bg) * stim  # simulate stimulation', 'x_bg += (1. / tau_bg) * stim[:num_bg_cells]  # simulate stimulation'),
        ('x_nm += (1. / tau_nm) * inh(B_nmbg) @ nln(x_bg)  # input from BG, inhibitory',
         'x_nm += (1. / tau_nm) * inh(B_nmbg) @ nln(x_bg)  # input from BG, inhibitory\n'
         '        x_nm += (1. / tau_nm) * stim[num_bg_cells:]  # SNc opto'),
    ]
    for old, new in subs:
        assert src.count(old) == 1, f'model source changed, patch anchor not found: {old}'
        src = src.replace(old, new)
    ns = dict(vars(mf))
    exec(src, ns)
    return vmap(ns['multiregion_nmrnn_nmstim'], in_axes=(None, None, None, 0, None, None, None, 0, None, 0))


batched_nmstim = _build_model()


def da_signal(params, zs):
    """DA signal s exactly as in the model step (tanh in the current model), from trained params."""
    return jnp.tanh(mf.nln(zs) @ mf.exc(params['m']).T + mf.nm_bias(params['c']))  # (..., 2): s_d1, s_d2 (cbtModel rank-2)


def _inputs():
    inp, _, _ = mf.self_timed_movement_task(jnp.array([cs.opto_tstart]), cs.config['T_cue'], cs.config['T_wait'],
                                            cs.config['T_movement'], cs.config['T'])
    return jnp.repeat(inp, cs.n_opto_seeds, axis=0)


def _keys():
    return jr.split(jr.PRNGKey(0), cs.n_opto_seeds)


def _stim(strength, n_bg, n_nm):
    T = cs.config['T']
    idx = jnp.arange(T)
    prof = ((idx >= cs.opto_start) & (idx < cs.opto_end)).astype(jnp.float32)
    spatial = jnp.concatenate([jnp.zeros(n_bg), jnp.full(n_nm, strength)])
    return jnp.repeat((prof[:, None] * spatial[None])[None], cs.n_opto_seeds, axis=0)


def verify(params):
    n_bg, n_nm = params['J_bg'].shape[0], params['J_nm'].shape[0]
    inp, keys = _inputs()[:50], _keys()[:50]
    ref = mf.batched_nm_rnn(params, cs.x0, cs.z0, inp, cs.config['tau_x'], cs.config['tau_z'], True,
                            jnp.zeros((50, cs.config['T'], n_bg)), cs.test_noise_std, keys)
    new = batched_nmstim(params, cs.x0, cs.z0, inp, cs.config['tau_x'], cs.config['tau_z'], True,
                         _stim(0.0, n_bg, n_nm)[:50], cs.test_noise_std, keys)
    diffs = [float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(new))]
    return max(diffs)


def rts_from_ys(ys):
    """RT in s after cue onset, NaN = no response (same rule as mf.get_response_times_opto)."""
    post = np.asarray(ys[:, cs.opto_tstart:, 0]) > 0.5
    hit = post.any(1)
    return np.where(hit, post.argmax(1) * 0.01, np.nan)


def simulate(params):
    n_bg, n_nm = params['J_bg'].shape[0], params['J_nm'].shape[0]
    inp, keys = _inputs(), _keys()
    out = {}
    for lab, s in CONDS:
        ys, xs, zs = batched_nmstim(params, cs.x0, cs.z0, inp, cs.config['tau_x'], cs.config['tau_z'], True,
                                    _stim(s, n_bg, n_nm), cs.test_noise_std, keys)
        traces = {a: np.asarray(mf.get_brain_area(a, xs, zs, as_rate=True).mean(-1)) for a in ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']}
        s_da = np.asarray(da_signal(params, zs))
        traces['DA s_d1'], traces['DA s_d2'] = s_da[..., 0], s_da[..., 1]
        traces['output y'] = np.asarray(ys[..., 0])
        out[(lab, s)] = dict(rt=rts_from_ys(ys), mean={k: v.mean(0) for k, v in traces.items()},
                             sem={k: v.std(0) / np.sqrt(v.shape[0] - 1) for k, v in traces.items()})
        del ys, xs, zs
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def timing_stats(sim):
    ctrl = sim[('control', 0.0)]['rt']
    onset_s = (cs.opto_start - cs.opto_tstart) / 100
    waiting = ~(ctrl < onset_s)  # not yet responded at stim onset (identical across conditions)
    rows = []
    for (lab, s), d in sim.items():
        if lab == 'control':
            continue
        a, b = ctrl[waiting], d['rt'][waiting]
        ra, rb = ~np.isnan(a), ~np.isnan(b)
        n01, n10 = int((~ra & rb).sum()), int((ra & ~rb).sum())  # gained / lost responses
        p_mc = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        ac, bc = np.where(ra, a, T_END_S), np.where(rb, b, T_END_S)
        dc = bc - ac
        p_cens = stats.wilcoxon(dc).pvalue if np.any(dc != 0) else 1.0
        both = ra & rb
        db = b[both] - a[both]
        p_both = stats.wilcoxon(db).pvalue if both.sum() > 5 and np.any(db != 0) else np.nan
        rows.append(dict(cond=lab, strength=s, n_waiting=int(waiting.sum()),
                         respond_ctrl=float(ra.mean()), respond_pert=float(rb.mean()),
                         gained=n01, lost=n10, p_mcnemar=float(p_mc),
                         median_rt_ctrl=float(np.nanmedian(a)) if ra.any() else None,
                         median_rt_pert=float(np.nanmedian(b)) if rb.any() else None,
                         shift_cens_median_ms=float(np.median(dc) * 1000), shift_cens_mean_ms=float(np.mean(dc) * 1000),
                         p_wilcoxon_cens=float(p_cens),
                         n_both=int(both.sum()),
                         shift_both_median_ms=float(np.median(db) * 1000) if both.any() else None,
                         p_wilcoxon_both=float(p_both)))
    for key in ['p_mcnemar', 'p_wilcoxon_cens', 'p_wilcoxon_both']:
        p = np.array([r[key] for r in rows])
        ok = ~np.isnan(p)
        adj = np.full_like(p, np.nan)
        adj[ok] = holm(p[ok])
        for r, v in zip(rows, adj):
            r[key + '_holm'] = None if np.isnan(v) else float(v)
    sg = np.array([r['strength'] for r in rows])
    sh = np.array([r['shift_cens_mean_ms'] for r in rows])
    rho, p_rho = stats.spearmanr(sg, sh)
    return dict(n_waiting=int(waiting.sum()), conditions=rows,
                dose_response=dict(spearman_rho=float(rho), p=float(p_rho), measure='signed strength vs mean censored shift'))


def plot(sim, st, fig_dir, title):
    cmap = plt.cm.get_cmap('berlin').reversed()
    norm = plt.Normalize(-max(STRENGTHS), max(STRENGTHS))
    ops, ope = (cs.opto_start - cs.opto_tstart) / 100, (cs.opto_end - cs.opto_tstart) / 100
    groups = ['inh SNc', 'stim SNc']

    # RT CDFs (all trials; CDF over responders, as in the striatal opto plot) + response fraction
    fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.8), sharey=True)
    for ax, g in zip(axs, groups):
        for (lab, s), d in sim.items():
            if lab not in (g, 'control'):
                continue
            r = np.sort(d['rt'][~np.isnan(d['rt'])])
            if len(r):
                ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=cmap(norm(s)), lw=1.5 if s == 0 else 1,
                        label=f'{s:+g} ({np.mean(~np.isnan(d["rt"])):.0%})')
        ax.axvspan(3, 6.5, color='green', alpha=0.2); ax.axvspan(ops, ope, color='blue', alpha=0.2)
        ax.set_title(g); ax.set_xlim(0, 6.5); ax.set_xticks([0, 3, 6]); ax.set_yticks([0, 0.5, 1])
        ax.legend(fontsize=5, frameon=False, title='strength (resp.)', title_fontsize=5, loc='lower right')
    axs[0].set_xlabel('Response time\nafter cue (s)'); axs[0].set_ylabel('Proportion (CDF)')
    fig.suptitle(title, fontsize=8); fig.tight_layout()
    fig.savefig(f'{fig_dir}/nm_opto_response_times_cdf.png', dpi=300); plt.close(fig)

    # activity by area
    areas = ['DA s_d1', 'DA s_d2', 'SNc', 'D1', 'D2', 'Cortex', 'Thalamus', 'output y']
    names = ['DA s_d1 (dSPN)', 'DA s_d2 (iSPN)', 'SNc rate', 'dSPN', 'iSPN', 'Cortex', 'Thalamus', 'Output y']
    fig, axs = plt.subplots(len(areas), 2, figsize=(4.2, 8), sharex=True, sharey='row')
    t = (np.arange(cs.config['T']) - cs.opto_tstart) / 100
    m = (t > -0.5) & (t < 6.4)
    for j, g in enumerate(groups):
        for i, a in enumerate(areas):
            ax = axs[i, j]
            for (lab, s), d in sim.items():
                if lab not in (g, 'control'):
                    continue
                mu, se = d['mean'][a][m], d['sem'][a][m]
                ax.plot(t[m], mu, color=cmap(norm(s)), lw=1)
                ax.fill_between(t[m], mu - se, mu + se, color=cmap(norm(s)), alpha=0.3)
            ax.axvspan(0, cs.config['T_cue'] / 100, color='red', alpha=0.2)
            ax.axvspan(ops, ope, color='blue', alpha=0.1)
            if j == 0:
                ax.set_ylabel(names[i])
            if i == 0:
                ax.set_title(g)
    axs[-1, 0].set_xlabel('time after cue (s)')
    fig.suptitle(title, fontsize=8); fig.tight_layout()
    fig.savefig(f'{fig_dir}/nm_opto_activity.png', dpi=300); plt.close(fig)

    # paired timing test summary
    rows = st['conditions']
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9))
    for ax, key, pkey, lab in [(axs[0], 'shift_cens_median_ms', 'p_wilcoxon_cens_holm', 'censored RT shift (ms)'),
                               (axs[1], 'shift_both_median_ms', 'p_wilcoxon_both_holm', 'RT shift, both respond (ms)')]:
        for r in rows:
            v = r[key]
            if v is None:
                continue
            p = r[pkey]
            ax.bar(r['strength'], v, width=0.4, color=cmap(norm(r['strength'])))
            star = '' if p is None else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
            ax.text(r['strength'], v, star, ha='center', va='bottom' if v >= 0 else 'top', fontsize=5)
        ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('SNc opto strength'); ax.set_ylabel(lab)
    fig.suptitle(f'{title}: median paired shift vs control (Holm)', fontsize=7); fig.tight_layout()
    fig.savefig(f'{fig_dir}/nm_opto_timing_test.png', dpi=300); plt.close(fig)


def run(params, fig_dir, title):
    diff = verify(params)
    assert diff == 0.0, f'patched model differs from mf.batched_nm_rnn with zero SNc input: {diff}'
    sim = simulate(params)
    st = timing_stats(sim)
    st['verify_max_abs_diff'] = diff
    st['calibration'] = {f'{lab} {s:+g}': dict(
        snc_rate_mean_during_stim=float(d['mean']['SNc'][cs.opto_start + 20:cs.opto_end].mean()),
        da_s_d1_mean_during_stim=float(d['mean']['DA s_d1'][cs.opto_start + 20:cs.opto_end].mean()),
        da_s_d2_mean_during_stim=float(d['mean']['DA s_d2'][cs.opto_start + 20:cs.opto_end].mean()))
        for (lab, s), d in sim.items()}
    plot(sim, st, fig_dir, title)
    np.savez_compressed('nm_opto_rts.npz', **{f'{lab}_{s:+g}': d['rt'] for (lab, s), d in sim.items()})
    return st
