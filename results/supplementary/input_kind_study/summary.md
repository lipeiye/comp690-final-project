# Which extra inputs cost EEGNet accuracy

Extra inputs after the 16 most relevant electrodes (2a: 6, Lee2019: 46). Accuracy in %, costs in percentage points against the same 16 electrodes alone, with the 95% participant interval. The next day uses the final models; held-out runs of the training day use the fold models. Lee2019 tests were fixed in the plan (2026-11-22T15:30); BCI IV 2a is exploratory.

## Lee2019 tests (54 people, final models, next day)

- K1, copies cost less than independent noise (dup minus noise): +13.95 [+10.65, +17.46]: holds. Copies against 16 electrodes alone: +0.62 [-0.05, +1.27]; no visible cost (lower bound above -2).
- K2, half noise lies between (dup minus mix50 +8.95 [+7.16, +10.84]; mix50 minus noise +5.00 [+2.69, +7.56]): holds
- K3, one shared waveform costs less than independent noise (shared minus noise): +11.50 [+8.27, +14.86]: holds

## Every condition

| dataset | extra inputs | next day % | cost | 95% interval | worse | held-out runs, cost | training set % | extra inputs' share of the spatial filters' output | extra / kept weight size |
|---|---|---:|---:|---|---:|---|---:|---:|---:|
| bnci2a | 16 electrodes alone | 65.15 | | | | 62.99 % | 92.5 | | |
| bnci2a | copies of kept electrodes | 64.01 | -1.14 | [-2.14, -0.23] | 7.0/9.0 | -0.91 [-1.70, -0.07] | 91.9 | 42% | 0.97 |
| bnci2a | copy + noise, a = 0.25 | 59.99 | -5.15 | [-6.60, -3.87] | 9.0/9.0 | -4.90 [-6.31, -3.31] | 93.1 | 33% | 0.74 |
| bnci2a | copy + noise, a = 0.5 | 59.34 | -5.81 | [-7.89, -4.17] | 9.0/9.0 | -6.30 [-8.24, -4.37] | 93.7 | 32% | 0.69 |
| bnci2a | copy + noise, a = 0.75 | 58.27 | -6.88 | [-9.26, -4.81] | 9.0/9.0 | -6.83 [-9.04, -4.75] | 93.4 | 32% | 0.68 |
| bnci2a | independent noise (a = 1) | 57.86 | -7.29 | [-9.80, -5.26] | 9.0/9.0 | -7.36 [-9.53, -5.31] | 93.9 |  |  |
| bnci2a | one noise waveform, repeated | 62.90 | -2.24 | [-3.88, -0.97] | 8.0/9.0 | -2.01 [-3.19, -0.79] | 91.9 | 10% | 0.69 |
| lee2019 | 16 electrodes alone | 67.02 | | | | 62.60 % | 96.1 | | |
| lee2019 | copies of kept electrodes | 67.64 | +0.62 | [-0.05, +1.27] | 21.0/54.0 | +0.39 [-0.27, +1.09] | 96.0 | 79% | 0.99 |
| lee2019 | copy + noise, a = 0.5 | 58.69 | -8.33 | [-10.22, -6.52] | 49.0/54.0 | -7.90 [-10.54, -5.42] | 99.9 | 68% | 0.99 |
| lee2019 | independent noise (a = 1) | 53.69 | -13.33 | [-16.84, -9.99] | 52.0/54.0 | -10.83 [-14.18, -7.69] | 100.0 |  |  |
| lee2019 | one noise waveform, repeated | 65.20 | -1.83 | [-2.82, -0.94] | 35.0/54.0 | -2.48 [-3.72, -1.32] | 94.5 | 33% | 0.45 |

## Paired contrasts

| dataset | contrast | measure | mean | 95% interval | seed error | worse |
|---|---|---|---:|---|---:|---:|
| bnci2a | dup minus noise | next day | +6.15 | [+4.29, +8.21] | 0.53 | 0/9 |
| bnci2a | dup minus noise | held-out runs | +6.45 | [+4.80, +8.20] | 0.71 | 0/9 |
| bnci2a | dup minus none | next day | -1.14 | [-2.14, -0.23] | 0.63 | 7/9 |
| bnci2a | dup minus none | held-out runs | -0.91 | [-1.70, -0.07] | 0.60 | 7/9 |
| bnci2a | dup minus mix50 | next day | +4.67 | [+3.42, +6.09] | 0.50 | 0/9 |
| bnci2a | dup minus mix50 | held-out runs | +5.39 | [+3.99, +6.80] | 0.57 | 0/9 |
| bnci2a | mix50 minus noise | next day | +1.48 | [+0.64, +2.43] | 0.40 | 1/9 |
| bnci2a | mix50 minus noise | held-out runs | +1.06 | [+0.17, +1.96] | 0.35 | 3/9 |
| bnci2a | shared minus noise | next day | +5.04 | [+3.97, +6.12] | 0.51 | 0/9 |
| bnci2a | shared minus noise | held-out runs | +5.35 | [+3.68, +7.15] | 0.41 | 0/9 |
| bnci2a | shared minus none | next day | -2.24 | [-3.88, -0.97] | 0.59 | 8/9 |
| bnci2a | shared minus none | held-out runs | -2.01 | [-3.19, -0.79] | 1.04 | 7/9 |
| bnci2a | mix25 minus dup | next day | -4.01 | [-5.02, -2.96] | 0.59 | 9/9 |
| bnci2a | mix25 minus dup | held-out runs | -3.98 | [-5.03, -3.03] | 0.55 | 9/9 |
| bnci2a | mix75 minus mix50 | next day | -1.07 | [-1.90, -0.30] | 0.26 | 7/9 |
| bnci2a | mix75 minus mix50 | held-out runs | -0.53 | [-1.04, +0.07] | 0.39 | 8/9 |
| lee2019 | dup minus noise | next day | +13.95 | [+10.65, +17.46] |  | 3/54 |
| lee2019 | dup minus noise | held-out runs | +11.22 | [+7.92, +14.68] |  | 9/54 |
| lee2019 | dup minus none | next day | +0.62 | [-0.05, +1.27] |  | 21/54 |
| lee2019 | dup minus none | held-out runs | +0.39 | [-0.27, +1.09] |  | 24/54 |
| lee2019 | dup minus mix50 | next day | +8.95 | [+7.16, +10.84] |  | 3/54 |
| lee2019 | dup minus mix50 | held-out runs | +8.29 | [+5.72, +10.97] |  | 12/54 |
| lee2019 | mix50 minus noise | next day | +5.00 | [+2.69, +7.56] |  | 16/54 |
| lee2019 | mix50 minus noise | held-out runs | +2.93 | [+1.64, +4.39] |  | 14/54 |
| lee2019 | shared minus noise | next day | +11.50 | [+8.27, +14.86] |  | 7/54 |
| lee2019 | shared minus noise | held-out runs | +8.35 | [+5.61, +11.33] |  | 10/54 |
| lee2019 | shared minus none | next day | -1.83 | [-2.82, -0.94] |  | 35/54 |
| lee2019 | shared minus none | held-out runs | -2.48 | [-3.72, -1.32] |  | 37/54 |

## The real dropped electrodes on the same axis (session 1 only)

Share of each dropped electrode's variance on the held-out runs that a least-squares fit from the 16 kept electrodes (fitted on the training runs) cannot predict: 0 for a copy, about a for a mix, about 1 for noise.

| dataset | people | mean share | 95% interval | range over people | cost of mixes with that a, next day | same, held-out runs |
|---|---:|---:|---|---|---:|---:|
| bnci2a | 9 | 0.037 | [0.031, 0.044] | 0.026-0.057 | -1.73 | -1.50 |
| lee2019 | 54 | 0.286 | [0.261, 0.315] | 0.122-0.784 | -4.49 | -4.34 |
