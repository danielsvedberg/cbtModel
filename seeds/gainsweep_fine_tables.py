"""Markdown summary of seeds/gain_sweep_2d_fine.py: python seeds/gainsweep_fine_tables.py <dir> > seeds/<dir>/gain_sweep_fine.md"""
import json, glob, sys, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
Ms = sorted((json.load(open(f)) for f in glob.glob(f'{D}/seed_*/metrics_gainsweep_fine.json')), key=lambda m: m['seed'])
K = [(a, c) for a in ['diag', 'da'] for c in ['both', 'ctx', 'tau']]
print(f'# seeds/{sys.argv[1]} fine gain sweep (seeds/gain_sweep_2d_fine.py)\n')
print('Fixed multiplier g on top of the DA-driven SPN gains (g = 1 = trained); gains 1 ± 0.001·2^k up to ±0.128, then 0.75 / 1.25. '
      'diag: dSPN = iSPN = g; da: dSPN = g, iSPN = 1/g. Each cell: verdict, threshold |g − 1| (first deviation where noise-free '
      't80 changes ≥ 5% or stalls, or noisy responding changes ≥ 10 points; – = never), and t80 ratio at 0.75 / 1.25 '
      '(stall = never reaches 80% of the control rise).\n')
print('| seed | ctrl t80 (s) | resp | ' + ' | '.join(f'{a}/{c}' for a, c in K) + ' |')
print('|' + '---|' * (3 + len(K)))
f = lambda v: 'stall' if v != v else f'{v:.2f}'
for m in Ms:
    cells = []
    for a, c in K:
        x = m['axes'][a][c]; R = {r['gain']: r for r in x['rows']}
        th = '–' if x['threshold'] is None else f"{x['threshold']:g}"
        cells.append(f"{x['verdict']} @{th} ({f(R[0.75]['t80_ratio'])}/{f(R[1.25]['t80_ratio'])})")
    print(f"| {m['seed']} | {m['control']['t80_s']:.2f} | {m['control']['respond']:.0%} | " + ' | '.join(cells) + ' |')
