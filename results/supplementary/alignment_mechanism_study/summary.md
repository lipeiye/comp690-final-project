# Why alignment hurts with 62 electrodes

Accuracy in %, differences in points with the 95% participant interval. Exploratory on both datasets; the readings M1-M3 were written in the plan (2026-11-22T16:36) before training.

## Readings

- M1, plain alignment hurts within the training day too (Lee2019 fold models, ea minus unaligned): -4.02 [-6.27, -2.05] (19/54 better): yes
- M2, harm on Lee2019 (unaligned minus ea, next day): +4.58 [+2.71, +6.69] (42/54 better); share removed: ea-top 48%, ea-shrink10 88%, ea-floor 102%, ea-lw 23%: no (ea-top and ea-shrink10 both at least half)
- M3, Ledoit-Wolf coefficient (Lee2019, 62 electrodes): mean 0.00606, largest 0.17470: yes (below 0.01)

## The training-day covariance

| dataset | montage | condition number (median) | directions below 1% of the largest (mean) | largest eigenvalue's share | Ledoit-Wolf (mean) | split-half whitener difference (mean) |
|---|---|---:|---:|---:|---:|---:|
| bnci2a | full-22 | 9122 | 17.2 of 22 | 0.80 | 0.00049 | 0.028 |
| bnci2a | relevance-16 | 5445 | 11.3 of 16 | 0.82 | 0.00051 | 0.028 |
| lee2019 | full-62 | 9445 | 49.1 of 62 | 0.54 | 0.00606 | 0.068 |
| lee2019 | relevance-16 | 1526 | 8.2 of 16 | 0.65 | 0.01368 | 0.052 |

## Accuracy

| dataset | alignment | next day (all trials aligned) | next day (secondary) | held-out runs |
|---|---|---:|---:|---:|
| bnci2a | ea | 66.24 | 65.07 | 62.52 |
| bnci2a | ea-floor | 67.41 | 68.12 | 63.93 |
| bnci2a | ea-lw | 65.73 | 65.63 | 61.97 |
| bnci2a | ea-shrink10 | 67.32 | 67.66 | 64.36 |
| bnci2a | ea-top | 57.07 | 57.06 | 54.88 |
| bnci2a | none | 65.42 | 65.67 | 62.58 |
| lee2019 | ea | 62.28 | 62.81 | 56.32 |
| lee2019 | ea-floor | 66.97 | 67.99 | 60.85 |
| lee2019 | ea-lw | 63.32 | 64.15 | 56.88 |
| lee2019 | ea-shrink10 | 66.32 | 67.29 | 59.87 |
| lee2019 | ea-top | 64.47 | 65.25 | 59.25 |
| lee2019 | none | 66.86 | 68.32 | 60.34 |

## Contrasts

| dataset | measure | contrast | mean | 95% interval | better |
|---|---|---|---:|---|---:|
| bnci2a | next day | ea minus unaligned | +0.82 | [-4.68, +5.74] | 6/9 |
| bnci2a | next day, secondary | ea minus unaligned | -0.60 | [-5.34, +3.77] | 5/9 |
| bnci2a | held-out runs | ea minus unaligned | -0.06 | [-3.06, +2.61] | 5/9 |
| bnci2a | next day | ea-shrink10 minus unaligned | +1.90 | [-1.84, +5.38] | 6/9 |
| bnci2a | next day | ea-shrink10 minus ea | +1.08 | [-3.17, +5.07] | 4/9 |
| bnci2a | next day, secondary | ea-shrink10 minus unaligned | +1.99 | [-2.22, +5.70] | 6/9 |
| bnci2a | next day, secondary | ea-shrink10 minus ea | +2.59 | [-0.31, +6.02] | 5/9 |
| bnci2a | held-out runs | ea-shrink10 minus unaligned | +1.79 | [-0.29, +3.55] | 7/9 |
| bnci2a | held-out runs | ea-shrink10 minus ea | +1.84 | [+0.42, +3.15] | 7/9 |
| bnci2a | next day | ea-lw minus unaligned | +0.31 | [-4.38, +4.67] | 6/9 |
| bnci2a | next day | ea-lw minus ea | -0.51 | [-1.97, +0.74] | 3/9 |
| bnci2a | next day, secondary | ea-lw minus unaligned | -0.04 | [-4.10, +3.91] | 4/9 |
| bnci2a | next day, secondary | ea-lw minus ea | +0.56 | [-0.81, +2.05] | 5/9 |
| bnci2a | held-out runs | ea-lw minus unaligned | -0.61 | [-3.52, +2.01] | 4/9 |
| bnci2a | held-out runs | ea-lw minus ea | -0.55 | [-0.84, -0.25] | 2/9 |
| bnci2a | next day | ea-top minus unaligned | -8.35 | [-12.49, -4.71] | 0/9 |
| bnci2a | next day | ea-top minus ea | -9.17 | [-15.49, -2.30] | 2/9 |
| bnci2a | next day, secondary | ea-top minus unaligned | -8.60 | [-12.12, -5.70] | 0/9 |
| bnci2a | next day, secondary | ea-top minus ea | -8.00 | [-13.25, -2.33] | 2/9 |
| bnci2a | held-out runs | ea-top minus unaligned | -7.70 | [-9.50, -5.64] | 0/9 |
| bnci2a | held-out runs | ea-top minus ea | -7.64 | [-11.25, -3.65] | 1/9 |
| bnci2a | next day | ea-floor minus unaligned | +1.99 | [-1.60, +5.34] | 6/9 |
| bnci2a | next day | ea-floor minus ea | +1.17 | [-3.11, +5.32] | 4/9 |
| bnci2a | next day, secondary | ea-floor minus unaligned | +2.45 | [-1.56, +5.78] | 7/9 |
| bnci2a | next day, secondary | ea-floor minus ea | +3.05 | [+0.05, +6.55] | 7/9 |
| bnci2a | held-out runs | ea-floor minus unaligned | +1.35 | [-0.73, +3.33] | 6/9 |
| bnci2a | held-out runs | ea-floor minus ea | +1.41 | [-0.39, +3.23] | 6/9 |
| lee2019 | next day | ea minus unaligned | -4.58 | [-6.69, -2.71] | 12/54 |
| lee2019 | next day, secondary | ea minus unaligned | -5.51 | [-7.88, -3.29] | 12/54 |
| lee2019 | held-out runs | ea minus unaligned | -4.02 | [-6.27, -2.05] | 19/54 |
| lee2019 | next day | ea-shrink10 minus unaligned | -0.54 | [-2.56, +1.47] | 24/54 |
| lee2019 | next day | ea-shrink10 minus ea | +4.04 | [+2.69, +5.53] | 45/54 |
| lee2019 | next day, secondary | ea-shrink10 minus unaligned | -1.03 | [-3.20, +1.09] | 23/54 |
| lee2019 | next day, secondary | ea-shrink10 minus ea | +4.48 | [+2.99, +6.06] | 43/54 |
| lee2019 | held-out runs | ea-shrink10 minus unaligned | -0.47 | [-1.90, +0.92] | 27/54 |
| lee2019 | held-out runs | ea-shrink10 minus ea | +3.55 | [+1.88, +5.55] | 36/54 |
| lee2019 | next day | ea-lw minus unaligned | -3.54 | [-5.34, -1.66] | 13/54 |
| lee2019 | next day | ea-lw minus ea | +1.04 | [+0.16, +2.12] | 27/54 |
| lee2019 | next day, secondary | ea-lw minus unaligned | -4.17 | [-6.28, -2.02] | 14/54 |
| lee2019 | next day, secondary | ea-lw minus ea | +1.34 | [+0.30, +2.56] | 31/54 |
| lee2019 | held-out runs | ea-lw minus unaligned | -3.46 | [-5.63, -1.56] | 17/54 |
| lee2019 | held-out runs | ea-lw minus ea | +0.56 | [+0.14, +1.04] | 29/54 |
| lee2019 | next day | ea-top minus unaligned | -2.39 | [-3.98, -0.88] | 22/54 |
| lee2019 | next day | ea-top minus ea | +2.19 | [+0.41, +4.03] | 36/54 |
| lee2019 | next day, secondary | ea-top minus unaligned | -3.07 | [-5.06, -1.07] | 19/54 |
| lee2019 | next day, secondary | ea-top minus ea | +2.44 | [+0.12, +4.77] | 30/54 |
| lee2019 | held-out runs | ea-top minus unaligned | -1.08 | [-2.29, +0.13] | 20/54 |
| lee2019 | held-out runs | ea-top minus ea | +2.93 | [+0.64, +5.41] | 32/54 |
| lee2019 | next day | ea-floor minus unaligned | +0.11 | [-1.59, +1.86] | 24/54 |
| lee2019 | next day | ea-floor minus ea | +4.69 | [+3.23, +6.23] | 46/54 |
| lee2019 | next day, secondary | ea-floor minus unaligned | -0.33 | [-2.17, +1.50] | 22/54 |
| lee2019 | next day, secondary | ea-floor minus ea | +5.18 | [+3.66, +6.80] | 46/54 |
| lee2019 | held-out runs | ea-floor minus unaligned | +0.52 | [-0.59, +1.65] | 31/54 |
| lee2019 | held-out runs | ea-floor minus ea | +4.53 | [+2.41, +6.94] | 40/54 |

Lee2019, across participants: correlation of the harm (unaligned minus ea, next day) with condition_number +0.09, below_1pct -0.03, top_share -0.06, ledoit_wolf +0.07, split_half +0.32
