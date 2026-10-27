"""Perturbation and lesion experiments on the baseline network (baseline-2026-10-08).

usage (from repo root): python experiments/run_experiment.py <net> <stage>
  net  : baseline | pd_acute | pd | hd_acute | hd
         pd = 75% SNc lesion (4 of 5 units), hd = 75% iSPN lesion (8 of 10 units);
         *_acute = baseline params with the lesion, pd/hd = after retraining with the lesion.
  stage: train (pd/hd only) | test | opto | pnr | dspn_pnr | snc_inhibition | snc_stimulation
         | ctx_inh_test | ctx_inh_pnr | ctx_inh_dspn_pnr : test / pnr / dspn_pnr with every cortical unit silenced
           (state -10, rate 0) from 100 ms after each trial's cue onset (= cue offset) to trial end
           (perturb_model ctx_off, CTX_DELAY = 10 steps; step 310 on no-cue trials)

Every stage runs with cwd = its own folder, experiments/<network folder>/<stage>/, so the repo scripts'
plots land in <stage>/plots/png/; metrics go to <stage>/metrics_<stage>.json.
The model is experiments/perturb_model.py, a hooked copy of model_functions.multiregion_nmrnn that is
bit-identical with its hooks off. model_functions.py is not modified.
"""
import os, sys, json, runpy, shutil, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')
net, stage = sys.argv[1], sys.argv[2]
BASE_PARAMS = os.path.join(ROOT, 'baselines', '2026-10-08_12a8ade', 'params_nm.pkl')
# CBT_SEED=<k>: start from seeds/3k/seed_<k>/params_nm.pkl instead of the baseline and write under experiments/seeds/seed_<k>/
SEED = os.environ.get('CBT_SEED')
OUT = EXP
if SEED:
    BASE_PARAMS = os.path.join(ROOT, 'seeds', '3k', f'seed_{SEED}', 'params_nm.pkl')
    OUT = os.path.join(EXP, 'seeds', f'seed_{SEED}')
LESION_DIR = {'pd': 'pd_snc_lesion', 'hd': 'hd_ispn_lesion'}
kind = net.split('_')[0] if net != 'baseline' else None
if net == 'baseline':
    NDIR = os.path.join(OUT, 'baseline')
elif stage == 'train':
    assert net in ('pd', 'hd')
    NDIR = os.path.join(OUT, LESION_DIR[kind])
else:
    NDIR = os.path.join(OUT, LESION_DIR[kind], 'acute' if net.endswith('_acute') else 'retrained')
SDIR = os.path.join(NDIR, 'training' if stage == 'train' else stage)
os.makedirs(SDIR, exist_ok=True)

os.chdir(SDIR)
CTX = stage.startswith('ctx_inh_')  # cortex silenced from cue onset; then run the plain stage of that name
if CTX:
    stage = stage[len('ctx_inh_'):]
TAG = net + (' + cortex off from cue+100 ms' if CTX else '')
sys.path[:0] = [EXP, ROOT]
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr
import plotting_functions as pf
import config_script as cs
import model_functions as mf
import perturb_model as pm
from model_functions import nln, exc


def _save_fig(fig, name):
    fig.savefig(os.path.join(cs.png_folder, name + '.png'), dpi=300)
pf.save_fig = _save_fig
_orig_savefig = matplotlib.figure.Figure.savefig
def _savefig(self, fname, *a, **k):  # 300-dpi PNG only, like seeds/run_seed.py
    if str(fname).endswith('.svg'):
        return
    k['dpi'] = min(k.get('dpi', 300) or 300, 300)
    return _orig_savefig(self, fname, *a, **k)
matplotlib.figure.Figure.savefig = _savefig

TAU_X, TAU_Z, T = cs.config['tau_x'], cs.config['tau_z'], cs.config['T']
N_BG, N_NM, N1 = cs.config['n_bg'], cs.config['n_nm'], cs.n_d1_cells


# ---- lesion definition (fixed seed; written to <lesion folder>/lesion.json) ----
def lesion(kind):
    if kind is None:
        return None, None, None
    rng = np.random.default_rng(2026)
    if kind == 'pd':
        idx = np.sort(rng.choice(N_NM, 4, replace=False))
        m = np.zeros(N_NM, bool); m[idx] = True
        info = dict(kind='PD: SNc lesion', units='SNc (x_nm)', n_lesioned=4, n_total=N_NM, fraction=0.8,
                    lesioned_idx=idx.tolist(), note='75% of 5 units rounded up to 4 (80%); rng default_rng(2026)')
        return None, jnp.asarray(m), info
    idx = np.sort(rng.choice(np.arange(N1, N_BG), 8, replace=False))
    m = np.zeros(N_BG, bool); m[idx] = True
    info = dict(kind='HD: iSPN lesion', units='iSPN (x_bg indices 10-19)', n_lesioned=8, n_total=N_BG - N1,
                fraction=0.8, lesioned_idx=idx.tolist(), note='75% of 10 iSPNs rounded up to 8 (80%); rng default_rng(2026)')
    return jnp.asarray(m), None, info


les_bg, les_nm, les_info = lesion(kind)
if les_info:
    json.dump(les_info, open(os.path.join(OUT, LESION_DIR[kind], 'lesion.json'), 'w'), indent=1)
pm.install(les_bg, les_nm, ctx_off=CTX)  # every mf.batched_nm_rnn call now uses the (lesioned) hooked model
batched = mf.batched_nm_rnn
batched_clamp = pm.make_batched(les_bg, les_nm, clamp=True)

if stage == 'train' or net in ('baseline', 'pd_acute', 'hd_acute'):
    PARAMS_SRC = BASE_PARAMS
else:
    PARAMS_SRC = os.path.join(OUT, LESION_DIR[kind], 'training', 'params_nm.pkl')
params = pickle.load(open(PARAMS_SRC, 'rb'))
if stage != 'train':
    shutil.copy(PARAMS_SRC, 'params_nm.pkl')  # the repo scripts load ./params_nm.pkl


def dump(obj):
    obj = dict(net=net, stage=stage, cortex_off_from_cue=CTX, params=os.path.relpath(PARAMS_SRC, ROOT), lesion=les_info, **obj)
    json.dump(obj, open(f'metrics_{stage}.json', 'w'), indent=1)
    print(json.dumps(obj, indent=1))


def run(script):
    return runpy.run_path(os.path.join(ROOT, script), run_name='__main__')


def first_cross(y, t0):
    """index (absolute) of the first y > 0.5 at or after t0, else inf. y: (..., T)"""
    above = y[..., t0:] > 0.5
    return np.where(above.any(-1), above.argmax(-1) + t0, np.inf)


def s_of(zs):
    return np.asarray(jax.nn.sigmoid(nln(zs) @ exc(params['m']).T + params['c']))[..., 0]


AREAS = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']

# ------------------------------------------------------------------ stages
if stage == 'train':
    ITERS = 3000
    inp, out, msk = mf.self_timed_movement_task(cs.config['T_start'], cs.config['T_cue'], cs.config['T_wait'],
                                                cs.config['T_movement'], T)
    best, losses = mf.fit_nm_rnn(inp, out, msk, params, cs.optimizer, cs.x0, cs.z0, ITERS, TAU_X, TAU_Z,
                                 wandb_log=False, modulation=True)
    pickle.dump(best, open('params_nm.pkl', 'wb'))
    losses = np.concatenate([np.asarray(l) for l in losses])
    np.save('losses.npy', losses)
    chunk_last = losses[199::200]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, lw=0.6); ax.set(xlabel='iteration', ylabel='loss', ylim=(0, 0.03), title=f'{net} retraining from baseline')
    fig.tight_layout(); fig.savefig('training_loss.png', dpi=150)
    dump(dict(iters=ITERS, start='baseline params with lesion applied', first_step_loss=float(losses[0]),
              best_logged_loss=float(chunk_last.min()), best_step=int((chunk_last.argmin() + 1) * 200),
              final_logged_loss=float(chunk_last[-1]), frac_steps_below_0p010=float((losses < 0.010).mean())))

elif stage == 'test':  # same metrics as seeds/run_seed.py
    try:
        g = run('testing_script.py')
        ys, xs, zs = g['all_ys'], g['all_xs'], g['all_zs']
    except Exception as e:
        print('testing_script failed, computing metrics directly:', repr(e))
        ys, xs, zs = mf.test_model(params, noise=True)
    rt = np.asarray(mf.get_response_times(ys, exclude_nan=False))
    out = dict(respond=float(np.mean(~np.isnan(rt))), on_time_2p5_3p5=float(np.mean((rt >= 2.5) & (rt <= 3.5))),
               rt_median=float(np.nanmedian(rt)) if np.any(~np.isnan(rt)) else None,
               rt_iqr=[float(v) for v in np.nanpercentile(rt, [25, 75])] if np.any(~np.isnan(rt)) else None)
    ns, nc = ys.shape[:2]
    starts = np.asarray(cs.test_start_t)
    for area in AREAS:
        rate = np.asarray(mf.get_brain_area(area, xs, zs, as_rate=True))
        a_mean = rate.mean(-1)
        onsets, rts, mods, unit_mod = [], [], [], []
        for c in range(nc):
            ce = starts[c] + cs.config['T_cue']
            pre = rate[:, c, starts[c] - 50:starts[c]].mean(1)
            for s in range(ns):
                end = int(ce + (rt[s, c] * 100 if not np.isnan(rt[s, c]) else T - ce))
                win = rate[s, c, ce + 100:min(end + 50, T)]
                if win.shape[0] < 2:
                    continue
                unit_mod.append(win.max(0) - pre[s])
                mods.append(a_mean[s, c, ce + 100:min(end + 50, T)].max() - pre[s].mean())
                if not np.isnan(rt[s, c]):
                    tr = a_mean[s, c, ce + 100:]
                    lo, hi = tr[:int(rt[s, c] * 100) - 100 + 1].min(), tr.max()
                    if hi - lo > 0.02:
                        onsets.append((np.argmax(tr > lo + 0.2 * (hi - lo)) + 100) / 100)
                        rts.append(rt[s, c])
        onsets, rts = np.array(onsets), np.array(rts)
        unit_mod = np.mean(unit_mod, 0) if unit_mod else np.zeros(1)
        ok = len(rts) > 5 and np.std(rts) > 0 and np.std(onsets) > 0
        out[area] = dict(modulation=float(np.mean(mods)) if mods else None,
                         frac_units_modulated=float(np.mean(unit_mod > 0.05)),
                         ramp_onset_vs_rt_r=float(np.corrcoef(onsets, rts)[0, 1]) if ok else None,
                         ramp_onset_vs_rt_slope=float(np.polyfit(rts, onsets, 1)[0]) if ok else None,
                         ramp_lead_median_s=float(np.median(rts - onsets)) if len(rts) else None)
    dump(out)

elif stage == 'opto':
    g = run('opto_script.py')
    oys = g['opto_ys']
    labels, strengths = list(cs.stim_labels), np.asarray(cs.stim_strengths)
    out = {}
    for lab in ['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']:
        rows = []
        for i in [i for i, l in enumerate(labels) if l == lab]:
            r = np.asarray(mf.get_response_times_opto(oys[i], exclude_nan=False, flatten=True))
            rows.append(dict(strength=float(strengths[i]), respond=float(np.mean(~np.isnan(r))),
                             rt_median=float(np.nanmedian(r)) if np.any(~np.isnan(r)) else None))
        out[lab] = rows
    dump(out)

elif stage == 'pnr':
    g = run('test_pnr_nm.py')
    pt_ys = np.asarray(g['pt_ys'])[..., 0]
    peak = pt_ys[:, 1:, 300:].max(-1)
    dump(dict(pert_delay_ms=[float(v) for v in np.asarray(g['pt_conditions'])[1:]],
              pert_prob=[float(v) for v in np.asarray(g['pt_prob'])[1:]],
              no_cue_prob=float(np.asarray(g['pt_prob'])[0]),
              peak_frac_ambiguous_0p4_0p5=float(np.mean((peak > 0.4) & (peak <= 0.5))),
              cue_mag=[float(v) for v in np.asarray(g['mags'])], cue_mag_prob=[float(v) for v in np.asarray(g['m_prob'])],
              cue_dur_ms=[float(v) for v in np.asarray(g['durs']) * cs.config['dt']],
              cue_dur_prob=[float(v) for v in np.asarray(g['d_prob'])]))
    from pnr_rt_plots import rt_figure  # RT CDFs + histograms for each PNR test
    dl = (np.asarray(g['pert_times']) - 300) / 100
    rt_figure(np.asarray(g['m_ys'])[..., 0], [f'{m:.1f}' for m in np.asarray(g['mags'])], 310, f'{TAG} cue magnitude', os.path.abspath('rt_cue_magnitude.png'), 'cue magnitude')
    rt_figure(np.asarray(g['d_ys'])[..., 0], [f'{int(d) * 10} ms' for d in np.asarray(g['durs'])], 300, f'{TAG} cue duration', os.path.abspath('rt_cue_duration.png'), 'cue duration')
    rt_figure(pt_ys, ['no cue'] + [f'{d:.1f} s' for d in dl], 300, f'{TAG} stop pulse (−5, all SPNs)', os.path.abspath('rt_stop_pulse.png'),
              'pulse delay', marks={i + 1: d for i, d in enumerate(dl)})

elif stage == 'dspn_pnr':
    # Dynamic PNR with excitation: +1 into dSPNs only for 5 steps (50 ms) at the test_pnr_nm pulse times.
    # Every condition of a noise seed shares the SAME rng key, so condition 1 (cue, no pulse) is an exact
    # twin of each pulse trial and their difference is the effect of the pulse alone.
    START, DUR, AMP, NS, WIN = 300, 5, 1.0, 150, 50
    ptimes = np.arange(320, 600, 20)
    nP, nC = len(ptimes), len(ptimes) + 2  # 0: no cue, 1: cue only (paired control), 2..: cue + pulse
    base, _, _ = mf.self_timed_movement_task(jnp.array([START]), cs.config['T_cue'], cs.config['T_wait'],
                                             cs.config['T_movement'], T)
    stim = np.zeros((nC, T, N_BG), np.float32)
    for i, p in enumerate(ptimes):
        stim[i + 2, p:p + DUR, :N1] = AMP
    stim = jnp.asarray(stim)
    inputs = jnp.repeat(base, nC, 0).at[0].set(0.0)
    sim = jax.jit(lambda keys: batched(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim, cs.test_noise_std, keys))
    Y, D, XS, ZS = [], [], [], []
    for s in range(NS):
        ys, xs, zs = sim(jnp.repeat(jr.PRNGKey(s + 200)[None], nC, 0))
        r = jnp.concatenate([nln(xs[0]), nln(xs[1]), nln(xs[2]), nln(zs)], -1)  # all units' rates
        D.append(np.asarray(jnp.linalg.norm(r[2:] - r[1:2], axis=-1)))      # (nP, T) distance to twin
        Y.append(np.asarray(ys[..., 0])); XS.append([np.asarray(x) for x in xs]); ZS.append(np.asarray(zs))
    Y, D = np.stack(Y), np.stack(D)                      # (NS, nC, T), (NS, nP, T)
    rt = first_cross(Y, START)                           # (NS, nC), absolute index
    resp = np.isfinite(rt).mean(0)
    rows, curves = [], []
    for i, p in enumerate(ptimes):
        rc, rp = rt[:, 1], rt[:, i + 2]
        valid = rc > p                                   # twin had not moved before the pulse
        trig = float(np.mean((rp[valid] >= p) & (rp[valid] < p + WIN)) - np.mean((rc[valid] >= p) & (rc[valid] < p + WIN)))
        both = valid & np.isfinite(rc) & np.isfinite(rp)
        shift = float(np.median(rp[both] - rc[both]) / 100) if both.any() else None
        taus, cens, short, cv = [], 0, 0, []
        resid = []
        for s in np.where(valid)[0]:
            end = int(min(rc[s], rp[s], p + DUR + 300, T)) - 30  # stop 300 ms before either trial crosses 0.5
            if end - p < DUR + 30:                       # too little time before movement (e.g. pulse-triggered)
                short += 1; continue
            d = D[s, i, p:end]
            k0 = int(np.argmax(d[:DUR + 20]))
            if end - p > DUR + 50 and d[k0] > 0:
                resid.append(d[DUR + 50] / d[k0])
            below = np.where(d[k0:] < d[k0] / np.e)[0]
            if d[k0] <= 0 or not len(below):
                cens += 1; continue
            taus.append((k0 + below[0] - DUR) * cs.config['dt'])  # ms from pulse end
            if end - p >= DUR + 100:
                cv.append(d[:DUR + 100] / d[k0])
        nv = int(valid.sum())
        rows.append(dict(pulse_delay_ms=float((p - START) * cs.config['dt']), n_valid=nv,
                         respond=float(resp[i + 2]), p_triggered_within_0p5s=trig, rt_shift_median_s=shift,
                         relax_tau_ms_median=float(np.median(taus)) if taus else None,
                         relax_tau_ms_iqr=[float(v) for v in np.percentile(taus, [25, 75])] if taus else None,
                         n_tau=len(taus), residual_500ms_median=float(np.median(resid)) if resid else None, n_residual=len(resid), frac_short_window=short / nv if nv else None, frac_censored=cens / nv if nv else None))
        curves.append(np.mean(cv, 0) if cv else np.full(DUR + 100, np.nan))
    dump(dict(definition=dict(
        pulse=f'+{AMP} into dSPNs (units 0-{N1 - 1}) for {DUR} steps; cue at t={START}; {NS} noise seeds',
        twin='condition with the same noise key, cue and no pulse',
        p_triggered=f'P(pulse trial crosses 0.5 within {WIN * 10} ms of pulse onset) minus the same for its twin; trials whose twin moved before the pulse excluded',
        relax_tau='state distance d(t) = ||rates(pulse) - rates(twin)|| over all 65 units; tau = time from pulse end until d falls below peak/e; '
                  'window = pulse onset to 300 ms before the first threshold crossing of either trial (max 3 s); short = window < 350 ms (not measured, mostly trials that move soon after the pulse); censored = no 1/e decay within the window; residual_500ms = d 500 ms after pulse end / peak d (>1 = the perturbation grew)'),
        no_cue_respond=float(resp[0]), control_respond=float(resp[1]),
        control_rt_median_s=float(np.median(rt[np.isfinite(rt[:, 1]), 1] - START) / 100), pulses=rows))
    # figures
    dl = (ptimes - START) / 100
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].plot(dl, [r['respond'] for r in rows], 'o-', label='P(respond)')
    ax[0].plot(dl, [r['p_triggered_within_0p5s'] for r in rows], 's-', label='P(triggered ≤0.5 s) − twin')
    ax[0].axhline(resp[1], color='k', ls=':', label='cue, no pulse'); ax[0].axhline(resp[0], color='gray', ls='--', label='no cue')
    ax[0].set(xlabel='pulse delay after cue (s)', ylim=(-0.05, 1.05), title='dSPN pulse (+1, 50 ms)'); ax[0].legend(fontsize=7)
    ax[1].plot(dl, [r['rt_shift_median_s'] if r['rt_shift_median_s'] is not None else np.nan for r in rows], 'o-')
    ax[1].axhline(0, color='gray', lw=0.8); ax[1].set(xlabel='pulse delay after cue (s)', ylabel='RT(pulse) − RT(twin), median (s)', title='RT shift')
    med = np.array([r['relax_tau_ms_median'] if r['relax_tau_ms_median'] is not None else np.nan for r in rows])
    iq = np.array([r['relax_tau_ms_iqr'] if r['relax_tau_ms_iqr'] else [np.nan, np.nan] for r in rows])
    ax[2].plot(dl, med, 'o-', color='tab:purple'); ax[2].fill_between(dl, iq[:, 0], iq[:, 1], color='tab:purple', alpha=0.2)
    ax[2].set(xlabel='pulse delay after cue (s)', ylabel='relaxation τ (ms, median, IQR)', title='critical slowing')
    a2 = ax[2].twinx(); a2.plot(dl, [r['frac_censored'] for r in rows], 'x--', color='tab:red', label='frac. never decays to 1/e')
    a2.plot(dl, [r['residual_500ms_median'] if r['residual_500ms_median'] is not None else np.nan for r in rows], 'd--', color='tab:green', label='d(500 ms)/peak, median')
    a2.axhline(1, color='tab:green', lw=0.6, ls=':'); a2.set_ylim(bottom=0); a2.legend(fontsize=7, loc='upper left')
    cmap = plt.cm.plasma
    for i, cvv in enumerate(curves):
        ax[3].plot(np.arange(len(cvv)) * 10 - DUR * 10, cvv, color=cmap(i / max(nP - 1, 1)), lw=1)
    ax[3].axhline(1 / np.e, color='gray', ls=':'); ax[3].set(xlabel='ms from pulse end', ylabel='d(t) / peak (mean)', title='distance to twin (colour: delay)', ylim=(0, 1.1))
    fig.tight_layout(); fig.savefig('dspn_pnr.png', dpi=150)
    from pnr_rt_plots import rt_figure
    rt_figure(Y, ['no cue', 'cue only'] + [f'{d:.1f} s' for d in dl], START, f'{TAG} dSPN pulse (+1)', os.path.abspath('rt_dspn_pulse.png'),
              'pulse delay', marks={i + 2: d for i, d in enumerate(dl)})
    pts = {i + 2: ((p - START) / 100, (p + DUR - START) / 100) for i, p in enumerate(ptimes)}
    xs_st = tuple(np.stack([x[j] for x in XS]) for j in range(3))
    pf.plot_binned_pnr(jnp.asarray(Y[..., None]), xs_st, np.stack(ZS), np.concatenate([[-1.0, 0.0], (ptimes - START) * 10.0]),
                       'Pulse delay (ms)', 'dSPN-pulse PNR', 'dspn_pnr', perturbation_boxes=pts)

elif stage in ('snc_inhibition', 'snc_stimulation'):
    # Clamp every SNc unit's state from cue onset to the end of the trial: -10 (rate 0) or +10 (rate ~1).
    # Trial layout and noise keys are the opto experiment's (cue at opto_tstart, 1000 trials, PRNGKey(0) split);
    # the control is the same trials unclamped.
    val = pm.OFF if stage == 'snc_inhibition' else pm.ON
    t0, N = cs.opto_tstart, cs.n_opto_seeds
    inp, _, _ = mf.self_timed_movement_task(jnp.array([t0]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], T)
    inputs, keys = jnp.repeat(inp, N, 0), jr.split(jr.PRNGKey(0), N)
    free = jnp.full((T, N_NM), jnp.nan)
    res = {}
    for name, cl in [('control', free), ('clamped', free.at[t0:].set(val))]:
        ys, xs, zs = batched_clamp(params, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, None, cs.test_noise_std, keys, cl)
        y = np.asarray(ys[..., 0]); rt = first_cross(y, t0)
        rts = (rt[np.isfinite(rt)] - t0) / 100
        area = {a: np.asarray(mf.get_brain_area(a, xs, zs, as_rate=True)).mean(-1) for a in AREAS}
        res[name] = dict(y=y, s=s_of(zs), rts=rts, area=area,
                         m=dict(respond=float(np.isfinite(rt).mean()),
                                rt_median_s=float(np.median(rts)) if len(rts) else None,
                                rt_iqr_s=[float(v) for v in np.percentile(rts, [25, 75])] if len(rts) else None,
                                s_precue=float(s_of(zs)[:, t0 - 50:t0].mean()), s_after_cue=float(s_of(zs)[:, t0 + 50:].mean())))
    mabs = np.abs(np.asarray(params['m'])).ravel()
    alive = np.ones(N_NM, bool) if les_nm is None else ~np.asarray(les_nm)
    dump(dict(definition=f'SNc state clamped to {val:+.0f} (rate {"0" if val < 0 else "~1"}) for all units from cue onset (t={t0}) '
                         f'to trial end; {N} trials, opto noise keys; control = same trials unclamped',
              s_theory=dict(snc_silent=float(jax.nn.sigmoid(params['c'])),
                            snc_saturated_live_units=float(jax.nn.sigmoid(mabs[alive].sum() * np.tanh(pm.ON) + params['c']))),
              control=res['control']['m'], clamped=res['clamped']['m']))
    tt = (np.arange(T) - t0) / 100
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5)); ax = ax.ravel()
    col = {'control': 'k', 'clamped': 'tab:blue' if val < 0 else 'tab:red'}
    lab = {'control': 'control', 'clamped': 'SNc off' if val < 0 else 'SNc saturated'}
    for name, r in res.items():
        for a, (key, ttl) in zip(ax, [('y', 'output y'), ('s', 'DA signal s')]):
            mu, se = r[key].mean(0), r[key].std(0) / np.sqrt(N)
            a.plot(tt, mu, color=col[name], label=lab[name]); a.fill_between(tt, mu - se, mu + se, color=col[name], alpha=0.2); a.set_title(ttl)
        if len(r['rts']):
            ax[2].plot(np.sort(r['rts']), np.arange(1, len(r['rts']) + 1) / N, color=col[name], label=lab[name])
        for a, an in zip(ax[3:], AREAS):
            a.plot(tt, r['area'][an].mean(0), color=col[name]); a.set_title(f'{an} mean rate')
    ax[0].axhline(0.5, color='gray', ls=':'); ax[0].legend(fontsize=8)
    ax[2].set(xlabel='RT from cue (s)', ylabel='cumulative P(respond)', title='RT CDF', ylim=(0, 1))
    for a in ax:
        a.axvspan(0, (T - t0) / 100, color=col['clamped'], alpha=0.04)
        if a is not ax[2]:
            a.set_xlabel('time from cue (s)')
    fig.suptitle(f'{net}: {lab["clamped"]} from cue to trial end'); fig.tight_layout(); fig.savefig(f'{stage}.png', dpi=150)
else:
    raise SystemExit(f'unknown stage {stage}')
