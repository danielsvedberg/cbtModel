# seeds/2k_2d_m05 fine gain sweep (seeds/gain_sweep_2d_fine.py)

Fixed multiplier g on top of the DA-driven SPN gains, applied from cue onset to trial end (g = 1 before the cue) (g = 1 = trained); gains 1 ± 0.001·2^k up to ±0.128, then 0.75 / 1.25. diag: dSPN = iSPN = g; da: dSPN = g, iSPN = 1/g. Each cell: verdict, threshold |g − 1| (first deviation where noise-free t80 changes ≥ 5% or stalls, or noisy responding changes ≥ 10 points; – = never), and t80 ratio at 0.75 / 1.25 (stall = never reaches 80% of the control rise).

| seed | ctrl t80 (s) | resp | diag/both | diag/ctx | diag/tau | da/both | da/ctx | da/tau |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.22 | 81% | no @0.064 (stall/stall) | no @0.032 (0.80/stall) | no @0.032 (stall/0.98) | partial @0.008 (stall/0.18) | no @0.016 (stall/0.34) | partial @0.032 (stall/stall) |
| 2 | 3.16 | 97% | no @0.032 (stall/1.04) | no @0.064 (stall/1.09) | no @0.064 (stall/1.00) | partial @0.032 (stall/stall) | partial @0.064 (stall/stall) | partial @0.128 (1.22/stall) |
| 3 | 4.22 | 100% | partial @0.032 (stall/0.89) | partial @0.064 (stall/0.94) | no @0.064 (stall/0.98) | partial @0.008 (stall/stall) | partial @0.016 (stall/stall) | partial @0.032 (stall/stall) |
| 4 | 2.86 | 89% | partial @0.032 (0.52/0.85) | partial @0.032 (0.43/0.91) | yes @0.032 (stall/0.92) | partial @0.016 (stall/stall) | partial @0.016 (stall/stall) | partial @0.032 (1.49/stall) |
| 5 | 3.11 | 41% | no @0.016 (stall/1.01) | no @0.25 (0.95/1.01) | no @0.016 (stall/0.97) | no @0.004 (stall/0.12) | no @0.004 (stall/0.17) | no @0.016 (stall/0.75) |
| 6 | 3.05 | 94% | no @0.064 (stall/stall) | no @0.064 (1.01/stall) | no @0.25 (stall/1.10) | partial @0.016 (stall/stall) | partial @0.016 (stall/stall) | partial @0.032 (1.37/stall) |
| 7 | 3.24 | 100% | partial @0.064 (0.83/stall) | no @0.064 (0.89/stall) | no @0.128 (0.91/1.16) | partial @0.032 (stall/0.73) | partial @0.032 (stall/0.84) | no @0.128 (stall/0.96) |
| 8 | 3.43 | 67% | no @0.032 (stall/1.04) | no @0.064 (1.02/1.07) | no @0.032 (stall/0.99) | partial @0.008 (stall/0.38) | partial @0.008 (stall/0.50) | partial @0.016 (stall/0.72) |
| 9 | 0.00 | 100% | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) |
| 10 | 3.53 | 99% | no @0.128 (1.16/stall) | no @0.25 (1.06/stall) | no @– (1.04/0.98) | partial @0.016 (stall/0.21) | partial @0.016 (stall/0.45) | partial @0.032 (stall/0.88) |
| 11 | 3.14 | 100% | partial @0.032 (0.80/stall) | partial @0.032 (0.82/stall) | partial @0.064 (0.83/stall) | no @0.032 (stall/0.74) | partial @0.032 (stall/0.71) | yes @0.064 (stall/0.84) |
| 12 | 2.97 | 96% | no @0.064 (0.89/stall) | no @0.064 (0.97/stall) | yes @0.128 (0.88/1.11) | no @0.008 (stall/0.23) | no @0.016 (stall/0.39) | partial @0.032 (stall/stall) |
| 13 | 3.14 | 59% | no @0.064 (stall/stall) | no @0.064 (0.91/stall) | no @0.032 (stall/0.97) | no @0.004 (stall/0.31) | no @0.004 (stall/0.50) | no @0.008 (stall/0.70) |
| 14 | 3.38 | 99% | partial @0.032 (0.49/stall) | no @0.032 (0.76/stall) | yes @0.128 (0.87/stall) | partial @0.032 (stall/0.22) | no @0.032 (stall/stall) | no @0.128 (stall/stall) |
| 15 | 3.18 | 53% | no @0.016 (stall/0.92) | no @0.032 (stall/1.05) | no @0.016 (stall/0.86) | partial @0.008 (stall/0.48) | partial @0.008 (stall/0.55) | no @0.016 (stall/0.81) |
| 16 | 2.99 | 60% | partial @0.032 (stall/1.18) | partial @0.064 (0.82/1.38) | no @0.016 (stall/0.93) | partial @0.004 (stall/0.32) | partial @0.008 (stall/0.41) | partial @0.016 (stall/0.63) |
| 17 | 3.04 | 100% | no @0.064 (0.89/stall) | no @0.128 (0.97/stall) | no @0.25 (stall/1.07) | no @0.016 (stall/0.28) | no @0.016 (stall/0.42) | no @0.064 (stall/stall) |
| 18 | 2.97 | 76% | partial @0.008 (stall/1.84) | no @0.032 (stall/stall) | no @0.016 (stall/0.89) | partial @0.004 (stall/0.14) | no @0.008 (stall/0.22) | partial @0.032 (stall/1.94) |
| 19 | 3.10 | 100% | no @0.128 (stall/stall) | no @0.25 (stall/1.15) | no @0.25 (stall/1.01) | partial @0.016 (stall/0.14) | partial @0.016 (stall/0.22) | partial @0.032 (stall/stall) |
| 20 | 0.00 | 0% | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) |
| 21 | 3.19 | 99% | no @0.032 (stall/0.74) | no @0.064 (stall/0.87) | no @0.032 (stall/0.88) | partial @0.008 (stall/0.24) | partial @0.008 (stall/0.38) | yes @0.064 (stall/0.86) |
| 22 | 2.90 | 100% | no @0.032 (0.92/stall) | no @0.032 (0.82/stall) | no @0.25 (stall/1.04) | partial @0.016 (stall/0.33) | no @0.032 (stall/stall) | no @0.032 (1.08/stall) |
| 23 | 3.67 | 100% | no @0.032 (stall/stall) | no @0.032 (0.89/stall) | no @0.128 (1.02/stall) | partial @0.004 (stall/0.22) | partial @0.008 (stall/0.30) | partial @0.016 (stall/0.54) |
| 24 | 3.12 | 79% | partial @0.032 (0.70/stall) | partial @0.032 (0.71/stall) | yes @0.128 (0.94/1.22) | partial @0.008 (stall/0.36) | partial @0.008 (stall/0.46) | partial @0.016 (stall/0.67) |
| 25 | 2.86 | 35% | partial @0.016 (stall/stall) | partial @0.064 (0.49/1.63) | no @0.032 (stall/0.93) | partial @0.008 (stall/stall) | no @0.008 (stall/stall) | partial @0.016 (stall/stall) |
| 26 | 3.07 | 94% | partial @0.016 (0.73/stall) | no @0.016 (0.82/stall) | no @0.032 (0.83/stall) | partial @0.008 (stall/0.44) | partial @0.016 (stall/0.52) | partial @0.032 (stall/0.71) |
| 27 | 3.38 | 97% | no @0.032 (stall/1.15) | no @0.128 (stall/0.99) | no @0.064 (stall/1.14) | partial @0.016 (stall/stall) | partial @0.032 (stall/stall) | no @0.064 (stall/1.17) |
| 28 | 0.00 | 99% | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) |
| 29 | 3.32 | 99% | no @0.064 (stall/stall) | no @0.064 (stall/stall) | no @0.128 (stall/1.15) | no @0.016 (stall/0.22) | no @0.016 (stall/0.32) | partial @0.032 (stall/stall) |
| 30 | 3.27 | 58% | no @0.016 (stall/0.97) | no @0.032 (stall/0.96) | no @0.032 (stall/0.98) | no @0.004 (stall/0.54) | no @0.008 (stall/0.62) | no @0.008 (stall/0.78) |
