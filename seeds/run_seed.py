"""Train/test one init seed in seeds/seed_<k>/ using the repo's scripts unchanged.

usage (from repo root): python seeds/run_seed.py <seed> <stage> [<dir name>]   stage: train | test | opto | pnr | nmopto
(dir name, relative to seeds/, defaults to seed_<seed>; env ITERS overrides num_full_train_iters; seeds/baseline_12a8ade holds the frozen baseline params, test stages only)

Only the weight-init seed (config 'keyind') changes; the T_start draw (PRNGKey(13)),
training noise, and every other config value are the baseline's. Each stage runs the
corresponding repo script with cwd = seeds/seed_<k>/ (so params_nm.pkl and plots/ land
there) and writes metrics_<stage>.json. Figures are saved as 300-dpi PNG only (repo
default is 900-dpi PNG + SVG, which is ~1 GB per run).
"""
import os, sys, json, re, runpy, io, contextlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seed, stage = int(sys.argv[1]), sys.argv[2]
SDIR = os.path.join(ROOT, 'seeds', sys.argv[3] if len(sys.argv) > 3 else f'seed_{seed}')
os.makedirs(SDIR, exist_ok=True)

# per-seed config: baseline config_script.py with keyind replaced
cfg_src = open(os.path.join(ROOT, 'config_script.py')).read()
cfg_new, n = re.subn(r'keyind=\d+,', f'keyind={seed},', cfg_src)
assert n == 1
if os.environ.get('ITERS'):  # optional override of num_full_train_iters
    cfg_new, n = re.subn(r'num_full_train_iters=\d+,', f"num_full_train_iters={int(os.environ['ITERS'])},", cfg_new)
    assert n == 1
open(os.path.join(SDIR, 'config_script.py'), 'w').write(cfg_new)

os.chdir(SDIR)
sys.path[:0] = [SDIR, ROOT]
import matplotlib
matplotlib.use('Agg'); matplotlib.use = lambda *a, **k: None
import plotting_functions as pf
import config_script as cs
import model_functions as mf
assert cs.config['keyind'] == seed


def _save_fig(fig, name):
    fig.savefig(os.path.join(cs.png_folder, name + '.png'), dpi=300)
pf.save_fig = _save_fig
_orig_savefig = matplotlib.figure.Figure.savefig
def _savefig(self, fname, *a, **k):  # cap any direct 900-dpi saves, skip svg
    if str(fname).endswith('.svg'):
        return
    k['dpi'] = min(k.get('dpi', 300) or 300, 300)
    return _orig_savefig(self, fname, *a, **k)
matplotlib.figure.Figure.savefig = _savefig


def run(script):
    return runpy.run_path(os.path.join(ROOT, script), run_name='__main__')


def dump(obj):
    json.dump(obj, open(f'metrics_{stage}.json', 'w'), indent=1)
    print(json.dumps(obj, indent=1))


AREAS = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']

if stage == 'train':
    g = run('training_script.py')
    losses = np.concatenate([np.asarray(l) for l in g['losses_nm']])
    np.save('losses.npy', losses)
    chunk_last = losses[199::200]
    if 'param_hist' in __import__('inspect').getsource(mf.fit_nm_rnn):
        # per-step selection: losses[k] is the loss of the params after k updates
        best = dict(best_logged_loss=float(losses.min()), best_step=int(losses.argmin()), selection='every step')
    else:
        best = dict(best_logged_loss=float(chunk_last.min()), best_step=int((chunk_last.argmin() + 1) * 200),
                    selection='200-step checkpoints')
    dump(dict(seed=seed, **best, final_logged_loss=float(chunk_last[-1]),
              frac_steps_below_0p010=float((losses < 0.010).mean())))

elif stage == 'test':
    # testing_script.py's plots, each guarded (testing_script asserts when no trial responds)
    import pickle
    params = pickle.load(open('params_nm.pkl', 'rb'))
    ys, xs, zs = mf.test_model(params, noise=True)
    rt = np.asarray(mf.get_response_times(ys, exclude_nan=False))  # s after cue end, (seeds, conds)
    skipped = {}
    def guard(name, f, *a):
        try:
            f(*a)
        except Exception as e:
            skipped[name] = repr(e)
            print(f'{name} skipped: {e!r}')
    guard('plot_output', pf.plot_output, ys)
    guard('plot_activity_by_area', pf.plot_activity_by_area, xs, zs)
    guard('plot_cue_algn_activity', pf.plot_cue_algn_activity, xs, zs, False)
    if np.any(~np.isnan(rt)):
        guard('plot_response_times', pf.plot_response_times, rt[~np.isnan(rt)])
    else:
        skipped['plot_response_times'] = 'no responding trials'
    d1d2 = mf.get_d1_d2_ratio(xs, avg_time=True, remove_outliers=False)
    guard('plot_ratio_rt_correlogram', pf.plot_ratio_rt_correlogram, d1d2, rt)
    guard('plot_d1d2ratio_SNc_correlogram', pf.plot_d1d2ratio_SNc_correlogram, d1d2, zs, rt)
    guard('plot_d1d2ratio_slope_correlogram', pf.plot_d1d2ratio_slope_correlogram, xs, rt)
    if np.sum(~np.isnan(rt)) >= 3:
        bg_gain = mf.get_bg_gain(params, zs)  # rank-2 DA gains exp(+s_d1), exp(-s_d2), replayed exactly
        guard('plot_binned_responses', lambda: pf.plot_binned_responses(ys, xs, zs, bg_gain=bg_gain))
    else:
        skipped['plot_binned_responses'] = f'only {int(np.sum(~np.isnan(rt)))} responding trials'
    out = dict(seed=seed, respond=float(np.mean(~np.isnan(rt))),
               on_time_2p5_3p5=float(np.mean((rt >= 2.5) & (rt <= 3.5))),
               rt_median=float(np.nanmedian(rt)) if np.any(~np.isnan(rt)) else None,
               rt_iqr=[float(v) for v in np.nanpercentile(rt, [25, 75])] if np.any(~np.isnan(rt)) else None)
    ns, nc, T = ys.shape[:3]
    starts = np.asarray(cs.test_start_t)
    for area in AREAS:
        rate = np.asarray(mf.get_brain_area(area, xs, zs, as_rate=True))  # (seeds, conds, T, N)
        a_mean = rate.mean(-1)
        onsets, rts, mods, unit_mod = [], [], [], []
        for c in range(nc):
            ce = starts[c] + cs.config['T_cue']
            pre = rate[:, c, starts[c] - 50:starts[c]].mean(1)  # (seeds, N)
            for s in range(ns):
                end = int(ce + (rt[s, c] * 100 if not np.isnan(rt[s, c]) else T - ce))
                win = rate[s, c, ce + 100:min(end + 50, T)]  # skip cue transient (first 1 s)
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
        out[area] = dict(
            modulation=float(np.mean(mods)) if mods else None,
            frac_units_modulated=float(np.mean(unit_mod > 0.05)),
            ramp_onset_vs_rt_r=float(np.corrcoef(onsets, rts)[0, 1]) if ok else None,
            ramp_onset_vs_rt_slope=float(np.polyfit(rts, onsets, 1)[0]) if ok else None,
            ramp_lead_median_s=float(np.median(rts - onsets)) if len(rts) else None)
    out['plots_skipped'] = skipped
    dump(out)

elif stage == 'opto':
    g = run('opto_script.py')
    oys = g['opto_ys']
    labels, strengths = list(cs.stim_labels), np.asarray(cs.stim_strengths)
    out = dict(seed=seed)
    for lab in ['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']:
        idx = [i for i, l in enumerate(labels) if l == lab]
        rows = []
        for i in idx:
            r = np.asarray(mf.get_response_times_opto(oys[i], exclude_nan=False, flatten=True))
            rows.append(dict(strength=float(strengths[i]), respond=float(np.mean(~np.isnan(r))),
                             rt_median=float(np.nanmedian(r)) if np.any(~np.isnan(r)) else None))
        out[lab] = rows
    dump(out)

elif stage == 'pnr':
    g = run('test_pnr_nm.py')
    pt_ys = np.asarray(g['pt_ys'])[..., 0]  # (seeds, conds, T)
    peak = pt_ys[:, 1:, 300:].max(-1)        # cue + perturbation conditions
    out = dict(seed=seed,
               pert_delay_ms=[float(v) for v in np.asarray(g['pt_conditions'])[1:]],
               pert_prob=[float(v) for v in np.asarray(g['pt_prob'])[1:]],
               no_cue_prob=float(np.asarray(g['pt_prob'])[0]),
               peak_frac_ambiguous_0p4_0p5=float(np.mean((peak > 0.4) & (peak <= 0.5))),
               cue_mag=[float(v) for v in np.asarray(g['mags'])],
               cue_mag_prob=[float(v) for v in np.asarray(g['m_prob'])],
               cue_dur_ms=[float(v) for v in np.asarray(g['durs']) * cs.config['dt']],
               cue_dur_prob=[float(v) for v in np.asarray(g['d_prob'])])
    dump(out)

elif stage == 'nmopto':
    import pickle
    sys.path.insert(0, os.path.join(ROOT, 'seeds'))
    import nm_opto
    dump(dict(seed=seed, **nm_opto.run(pickle.load(open('params_nm.pkl', 'rb')), cs.png_folder, f'seed {seed}')))
