"""Markdown tables of seeds/gain_sweep_2d.py results: python seeds/2k_2d_m05/gainsweep_tables.py > seeds/2k_2d/gain_sweep.md"""
import json, glob, os
D = os.path.dirname(os.path.abspath(__file__))
Ms = sorted((json.load(open(f)) for f in glob.glob(f'{D}/seed_*/metrics_gainsweep.json')), key=lambda m: m['seed'])
G = Ms[0]['gains']
desc = {'diag': 'dSPN = iSPN = g (the nonm sweep)', 'da': 'dSPN = g, iSPN = 1/g (DA direction)'}
cdesc = {'both': 'cortex->BG gain and BG tau gain fixed', 'ctx': 'cortex->BG gain fixed, tau gain DA-driven',
         'tau': 'BG tau gain fixed, cortex->BG gain DA-driven'}
def cell(r):
    rt = f" RT×{r['rt_ratio_median']:.2f}" if r['rt_ratio_median'] is not None else ''
    if r['t80_ratio'] != r['t80_ratio']:
        return f"stall {r['respond']:.0%}{rt}"
    w = r.get('warp')
    ws = f" ({w['scale']['r2']:.2f}/{w['shift']['r2']:.2f})" if w else ''
    return f"{r['t80_ratio']:.2f}{ws} {r['respond']:.0%}{rt}"
print('# seeds/2k_2d_m05 fixed-gain sweep (seeds/gain_sweep_2d.py)\n')
print('DA-driven SPN gains replaced by fixed gains for the whole trial via the model switch params[fixed_gain_ctx/tau]. '
      'Control = intact DA network. Each cell: noise-free t80 / control t80 (stall = never reaches 80% of the control rise), '
      'scale-only / shift-only warp R² on the 60 cortex/BG/thalamus rates, fraction responding, median paired noisy RT ratio '
      'vs intact DA (500 trials).\n')
for ax in ['diag', 'da']:
    for c in ['both', 'ctx', 'tau']:
        print(f'## {ax} / {c}: {desc[ax]}; {cdesc[c]}\n')
        hdr = [f'{g:g}' if ax == 'diag' else f'{g:g}/{1/g:g}' for g in G]
        print('| seed | verdict | DA t80 (s) | DA resp | ' + ' | '.join(hdr) + ' |')
        print('|' + '---|' * (4 + len(G)))
        for m in Ms:
            a = m['axes'][ax][c]
            print(f"| {m['seed']} | {a['verdict']} | {m['control']['t80_s']:.2f} | {m['control']['respond']:.0%} | "
                  + ' | '.join(cell(r) for r in a['rows']) + ' |')
        print()
