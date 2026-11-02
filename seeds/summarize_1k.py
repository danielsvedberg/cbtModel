"""Cross-seed summary for seeds/1k (from repo root): python seeds/summarize_1k.py
Usage: python seeds/summarize_1k.py [1k|5k]. Writes seeds/<dir>/loss_curves.png, seeds/1k/rt_cdf_all_seeds.png, seeds/1k/nm_timing_all_seeds.png, seeds/1k/summary.md."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else '1k')
seeds = sorted(int(p.split('_')[-1]) for p in glob.glob(f'{D}/seed_*'))
load = lambda s, st: json.load(open(f'{D}/seed_{s}/metrics_{st}.json')) if os.path.exists(f'{D}/seed_{s}/metrics_{st}.json') else None
colors = plt.cm.tab10(np.arange(len(seeds)))
PLATEAU = 0.013

# loss curves: one panel per seed + overlay
fig, axs = plt.subplots(1, len(seeds) + 1, figsize=(2.0 * (len(seeds) + 1), 2.0), sharey=True)
for ax, s, c in zip(axs, seeds, colors):
    L = np.load(f'{D}/seed_{s}/losses.npy')
    tr = load(s, 'train')
    for a in (ax, axs[-1]):
        a.plot(np.arange(1, len(L) + 1), L, color=c, lw=0.6, label=f'seed {s}')
    ck = np.arange(200, len(L) + 1, 200)
    ax.plot(ck, L[ck - 1], 'o', color='k', ms=2.5)
    ax.plot(tr['best_step'], tr['best_logged_loss'], 'o', mfc='none', mec='r', ms=6)
    ax.set_title(f'seed {s}', fontsize=8)
for ax in axs:
    ax.axhline(PLATEAU, color='gray', ls='--', lw=0.6)
    ax.set_yscale('log'); ax.set_xlabel('iteration')
axs[0].set_ylabel('training loss'); axs[-1].set_title('all seeds', fontsize=8); axs[-1].legend(fontsize=5, frameon=False)
fig.suptitle('dots: 200-step checkpoints (best_params chosen here; red ring = selected). dashed: 0.013 never-move plateau', fontsize=6)
fig.tight_layout(); fig.savefig(f'{D}/loss_curves.png', dpi=300); plt.close(fig)

# RT CDF per seed (unperturbed striatal-opto control trials, 1000 per seed; from the SNc opto run, same keys)
fig, ax = plt.subplots(figsize=(2.4, 2.0))
for s, c in zip(seeds, colors):
    f = f'{D}/seed_{s}/nm_opto_rts.npz'
    if not os.path.exists(f):
        continue
    r = np.load(f)['control_+0']
    rr = np.sort(r[~np.isnan(r)])
    if len(rr):
        ax.plot(rr, np.arange(1, len(rr) + 1) / len(r), color=c, label=f'seed {s} ({len(rr) / len(r):.0%})')
ax.axvspan(3, 6.5, color='green', alpha=0.2); ax.set_xlim(0, 6.5)
ax.set_xlabel('Response time after cue (s)'); ax.set_ylabel('fraction of trials responded')
ax.legend(fontsize=5, frameon=False); fig.tight_layout(); fig.savefig(f'{D}/rt_cdf_all_seeds.png', dpi=300); plt.close(fig)

# SNc timing test across seeds
fig, axs = plt.subplots(1, 2, figsize=(5, 2.0))
lines = []
for s, c in zip(seeds, colors):
    nm = load(s, 'nmopto')
    if nm is None:
        continue
    rows = sorted(nm['conditions'], key=lambda r: r['strength'])
    x = [r['strength'] for r in rows]
    axs[0].plot(x, [r['shift_cens_median_ms'] for r in rows], 'o-', color=c, ms=3, label=f'seed {s}')
    axs[1].plot(x, [r['respond_pert'] - r['respond_ctrl'] for r in rows], 'o-', color=c, ms=3)
    for r in rows:
        if r['p_wilcoxon_cens_holm'] is not None and r['p_wilcoxon_cens_holm'] < 0.05:
            axs[0].plot(r['strength'], r['shift_cens_median_ms'], 'o', mfc='none', mec='k', ms=6)
for ax in axs:
    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('SNc opto strength')
axs[0].set_ylabel('median paired RT shift (ms)\n(no response = 6.5 s)'); axs[1].set_ylabel('Δ fraction responding')
axs[0].legend(fontsize=5, frameon=False)
fig.suptitle('SNc opto 1–2.75 s after cue, trials still waiting at onset; ring = Holm p<0.05', fontsize=6)
fig.tight_layout(); fig.savefig(f'{D}/nm_timing_all_seeds.png', dpi=300); plt.close(fig)

# markdown table
fmtp = lambda p: 'n/a' if p is None else f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
out = [f'# seeds/{os.path.basename(D)} summary', '', 'Init seeds ' + ', '.join(map(str, seeds)) + f', {len(np.load(f"{D}/seed_{seeds[0]}/losses.npy"))} iterations, working-tree model (see model_functions.diff).', '',
       '## Training and test', '', 'Temporal scaling: verdict of seeds/scaling_screen.py (criteria in its docstring; details in the scaling section below).', '',
       '| seed | best logged loss (step) | final loss | responding (test) | on time 2.5–3.5 s | median RT (s) | temporal scaling |', '|---|---|---|---|---|---|---|']
for s in seeds:
    tr, te, sc = load(s, 'train'), load(s, 'test'), load(s, 'scaling')
    out.append(f"| {s} | {tr['best_logged_loss']:.4f} ({tr['best_step']}) | {tr['final_logged_loss']:.4f} | "
               + (f"{te['respond']:.0%} | {te['on_time_2p5_3p5']:.0%} | {te['rt_median'] if te['rt_median'] is None else round(te['rt_median'], 2)} | " if te else 'n/a | n/a | n/a | ')
               + (f"**{sc['verdict']}** |" if sc and sc['verdict'] == 'yes' else f"{sc['verdict']} |" if sc else 'n/a |'))
if any(load(s, 'scaling') for s in seeds):
    out += ['', '## Temporal scaling under sustained SNc input', '',
            'Sustained SNc input from cue offset, noise-free. Each cell: t80 / control t80 (stall = never reaches 80% of the control rise), '
            'then scale-only / shift-only warp R² on the 60 cortex/BG/thalamus rates, then median paired noisy RT ratio (500 trials).', '',
            '| seed | verdict | control t80 (s) | ' + ' | '.join(f'{v:+g}' for v in [-1, -0.5, -0.25, 0.25, 0.5, 1]) + ' |', '|---|---|---|' + '---|' * 6]
    for s in seeds:
        sc = load(s, 'scaling')
        if not sc:
            continue
        cells = []
        for r in sorted(sc['rows'], key=lambda r: r['strength']):
            if r['strength'] == 0:
                continue
            c = 'stall' if r['t80_ratio'] != r['t80_ratio'] else f"{r['t80_ratio']:.2f}"
            if 'warp' in r:
                c += f" ({r['warp']['scale']['r2']:.2f}/{r['warp']['shift']['r2']:.2f})"
            if r['rt_ratio_median'] is not None:
                c += f" RT×{r['rt_ratio_median']:.2f}"
            cells.append(c)
        t = sc['control']['t80_s']
        out.append(f"| {s} | {sc['verdict']} | {'–' if t != t else round(t, 2)} | " + ' | '.join(cells) + ' |')
out += ['', '## Striatal opto (fraction responding / median RT s)', '', '| seed | condition | ' + ' | '.join(['s1', 's2', 's3', 's4', 's5', 's6']) + ' |', '|---|---|' + '---|' * 6]
for s in seeds:
    op = load(s, 'opto')
    if op is None:
        continue
    for lab in ['inh dMSN', 'inh iMSN', 'stim dMSN', 'stim iMSN']:
        cells = [f"{r['strength']:+.2f}: {r['respond']:.0%}/{'–' if r['rt_median'] is None else round(r['rt_median'], 2)}" for r in op[lab]]
        out.append(f'| {s} | {lab} | ' + ' | '.join(cells) + ' |')
out += ['', '## SNc opto timing test', '',
        'Paired by noise key against the unperturbed control; only trials not yet responded at stim onset (1 s). Holm-corrected within each test and seed.', '',
        '| seed | strength | n | resp ctrl→pert | McNemar p | median shift censored (ms) | p | n both | median shift both (ms) | p | SNc rate / DA s_d1 / s_d2 during stim |', '|---|---|---|---|---|---|---|---|---|---|---|']
for s in seeds:
    nm = load(s, 'nmopto')
    if nm is None:
        continue
    for r in sorted(nm['conditions'], key=lambda r: r['strength']):
        cal = nm['calibration'][f"{r['cond']} {r['strength']:+g}"]
        out.append(f"| {s} | {r['strength']:+g} | {r['n_waiting']} | {r['respond_ctrl']:.0%}→{r['respond_pert']:.0%} | {fmtp(r['p_mcnemar_holm'])} | "
                   f"{r['shift_cens_median_ms']:+.0f} | {fmtp(r['p_wilcoxon_cens_holm'])} | {r['n_both']} | "
                   f"{'–' if r['shift_both_median_ms'] is None else format(r['shift_both_median_ms'], '+.0f')} | {fmtp(r['p_wilcoxon_both_holm'])} | "
                   f"{cal['snc_rate_mean_during_stim']:.2f} / {cal['da_s_d1_mean_during_stim']:.2f} / {cal['da_s_d2_mean_during_stim']:.2f} |")
    dr = nm['dose_response']
    out.append(f"| {s} | dose-response | | | | Spearman ρ = {dr['spearman_rho']:.2f}, p = {fmtp(dr['p'])} | | | | | |")
open(f'{D}/summary.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
