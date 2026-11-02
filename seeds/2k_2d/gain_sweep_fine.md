# seeds/2k_2d fine gain sweep (seeds/gain_sweep_2d_fine.py)

Fixed multiplier g on top of the DA-driven SPN gains, applied from cue onset to trial end (g = 1 before the cue) (g = 1 = trained); gains 1 ± 0.001·2^k up to ±0.128, then 0.75 / 1.25. diag: dSPN = iSPN = g; da: dSPN = g, iSPN = 1/g. Each cell: verdict, threshold |g − 1| (first deviation where noise-free t80 changes ≥ 5% or stalls, or noisy responding changes ≥ 10 points; – = never), and t80 ratio at 0.75 / 1.25 (stall = never reaches 80% of the control rise).

| seed | ctrl t80 (s) | resp | diag/both | diag/ctx | diag/tau | da/both | da/ctx | da/tau |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.24 | 94% | partial @0.032 (0.80/stall) | no @0.032 (stall/stall) | no @0.128 (stall/stall) | no @0.008 (stall/0.23) | no @0.008 (stall/0.60) | no @0.032 (stall/stall) |
| 2 | 3.45 | 91% | no @0.032 (0.83/stall) | no @0.064 (0.89/stall) | yes @0.128 (0.87/stall) | partial @0.016 (stall/0.64) | partial @0.016 (stall/0.73) | partial @0.032 (stall/0.81) |
| 3 | 2.98 | 82% | partial @0.032 (stall/0.81) | partial @0.064 (stall/0.92) | yes @0.032 (stall/0.88) | partial @0.016 (stall/0.79) | partial @0.016 (stall/stall) | partial @0.032 (stall/stall) |
| 4 | 0.00 | 98% | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) |
| 5 | 3.24 | 35% | no ramp @0.016 (stall/0.88) | no ramp @0.128 (stall/1.00) | no ramp @0.016 (stall/0.85) | no ramp @0.016 (stall/0.57) | no ramp @0.016 (stall/0.63) | no ramp @0.128 (stall/0.94) |
| 6 | 3.08 | 99% | no @0.064 (0.37/stall) | no @0.128 (stall/0.93) | no @0.128 (stall/0.97) | no @0.016 (stall/0.47) | partial @0.032 (stall/0.88) | no @0.064 (stall/stall) |
| 7 | 3.01 | 95% | partial @0.016 (0.63/stall) | no @0.032 (0.79/stall) | yes @0.064 (0.81/1.21) | partial @0.016 (stall/0.72) | partial @0.016 (stall/0.67) | yes @0.064 (stall/0.79) |
| 8 | 2.96 | 87% | partial @0.004 (0.46/stall) | partial @0.008 (0.55/stall) | partial @0.016 (0.61/stall) | partial @0.004 (stall/0.29) | partial @0.004 (stall/0.38) | partial @0.016 (stall/0.62) |
| 9 | 0.00 | 100% | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) | no ramp @0.001 (stall/stall) |
| 10 | 3.16 | 97% | partial @0.032 (0.73/stall) | partial @0.064 (0.83/stall) | yes @0.128 (0.85/1.26) | partial @0.016 (stall/0.40) | partial @0.016 (stall/0.54) | yes @0.064 (stall/0.76) |
