# Overfitting study: why can fewer electrodes decode the next day better?

Accuracy in %, contrasts in percentage points with the 95% participant interval; seed error for BCI IV 2a (ten seeds). Lee2019 tests were fixed in the plan; BCI IV 2a is exploratory.

## Lee2019 tests (54 people, seeds 0-2)

| prediction | contrast | mean | 95% interval | holds |
|---|---|---:|---|---|
| P1: Less training data widens the full montage's deficit | P1: gap with 1 of 2 runs minus gap with all | +0.77 | [-0.62, +2.11] | no |
| P2: The deficit also appears on held-out trials of the training day, not only on the next day | P2: reduced minus full, held-out runs of the training day, 1 of 2 runs | +0.35 | [-0.96, +1.74] | no |
| P3: Inputs that carry nothing hurt: noise in place of the dropped electrodes lowers accuracy | P3: reduced plus noise minus reduced | -4.77 | [-6.61, -3.02] | yes |
| P4: Channel dropout helps the full montage more than the reduced one | P4: dropout gain of full minus dropout gain of reduced | -0.13 | [-1.16, +0.93] | no |

## bnci2a: all contrasts

| contrast | mean | 95% interval | seed error | better/worse |
|---|---:|---|---:|---|
| reduced minus full, next day, all of the training day | +0.32 | [-0.42, +1.04] | 0.45 | 6/3 |
| reduced minus full, next day, 4 of 6 runs | +0.34 | [-1.08, +1.69] | 0.42 | 5/4 |
| P1: gap with 4 of 6 runs minus gap with all | +0.01 | [-1.50, +1.61] | 0.47 | 4/5 |
| P2: reduced minus full, held-out runs of the training day, 4 of 6 runs | +0.20 | [-0.65, +1.08] | 0.50 | 5/4 |
| next-day gap minus within-day gap, 4 of 6 runs | +0.13 | [-1.05, +1.27] | 0.49 | 6/3 |
| reduced minus full, next day, 2 of 6 runs | +0.22 | [-0.73, +1.20] | 0.37 | 4/5 |
| P1: gap with 2 of 6 runs minus gap with all | -0.10 | [-1.25, +1.13] | 0.53 | 5/4 |
| P2: reduced minus full, held-out runs of the training day, 2 of 6 runs | +0.56 | [-0.19, +1.31] | 0.53 | 7/2 |
| next-day gap minus within-day gap, 2 of 6 runs | -0.34 | [-1.17, +0.47] | 0.34 | 4/5 |
| P3: reduced plus noise minus reduced | -4.96 | [-6.55, -3.57] | 0.59 | 0/9 |
| information in the dropped electrodes: full minus reduced plus noise | +4.64 | [+3.21, +6.02] | 0.49 | 9/0 |
| reduced plus noise minus full | -4.64 | [-6.02, -3.21] | 0.49 | 0/9 |
| reduced plus shuffled minus reduced | -4.74 | [-5.87, -3.63] | 0.57 | 0/9 |
| reduced plus shuffled minus full | -4.41 | [-5.83, -3.01] | 0.56 | 0/9 |
| full with channel dropout minus full | -11.07 | [-13.79, -8.82] | 0.38 | 0/9 |
| reduced with channel dropout minus reduced | -11.82 | [-14.15, -9.72] | 0.41 | 0/9 |
| P4: dropout gain of full minus dropout gain of reduced | +0.75 | [-0.37, +1.80] | 0.45 | 6/3 |
| full with channel dropout minus reduced | -11.39 | [-13.74, -9.33] | 0.39 | 0/9 |

| data | montage | training set | held-out runs, same day | next day |
|---|---|---:|---:|---:|
| 6 of 6 runs | full-22 | 93.05 | - | 64.82 |
| 6 of 6 runs | relevance-16 | 92.46 | - | 65.15 |
| 4 of 6 runs | full-22 | 96.98 | 62.15 | 59.91 |
| 4 of 6 runs | relevance-16 | 95.74 | 62.35 | 60.24 |
| 2 of 6 runs | full-22 | 98.25 | 51.09 | 49.37 |
| 2 of 6 runs | relevance-16 | 97.50 | 51.65 | 49.60 |

## lee2019: all contrasts

| contrast | mean | 95% interval | seed error | better/worse |
|---|---:|---|---:|---|
| reduced minus full, next day, all of the training day | +0.27 | [-0.87, +1.47] | 0.23 | 20/33 |
| reduced minus full, next day, 1 of 2 runs | +1.04 | [-0.06, +2.20] | 0.37 | 27/26 |
| P1: gap with 1 of 2 runs minus gap with all | +0.77 | [-0.62, +2.11] | 0.48 | 32/22 |
| P2: reduced minus full, held-out runs of the training day, 1 of 2 runs | +0.35 | [-0.96, +1.74] | 0.30 | 21/32 |
| next-day gap minus within-day gap, 1 of 2 runs | +0.69 | [-0.31, +1.64] | 0.39 | 35/19 |
| P3: reduced plus noise minus reduced | -4.77 | [-6.61, -3.02] | 0.05 | 14/39 |
| information in the dropped electrodes: full minus reduced plus noise | +4.50 | [+2.75, +6.34] | 0.25 | 41/11 |
| reduced plus noise minus full | -4.50 | [-6.34, -2.75] | 0.25 | 11/41 |
| reduced plus shuffled minus reduced | -3.35 | [-4.81, -2.00] | 0.28 | 12/39 |
| reduced plus shuffled minus full | -3.07 | [-4.78, -1.51] | 0.39 | 12/41 |
| full with channel dropout minus full | -2.74 | [-4.16, -1.41] | 0.41 | 17/37 |
| reduced with channel dropout minus reduced | -2.61 | [-3.88, -1.35] | 0.47 | 17/37 |
| P4: dropout gain of full minus dropout gain of reduced | -0.13 | [-1.16, +0.93] | 0.14 | 22/31 |
| full with channel dropout minus reduced | -3.01 | [-4.37, -1.74] | 0.64 | 17/37 |

| data | montage | training set | held-out runs, same day | next day |
|---|---|---:|---:|---:|
| 2 of 2 runs | full-62 | 99.02 | - | 66.90 |
| 2 of 2 runs | relevance-45 | 98.84 | - | 67.17 |
| 1 of 2 runs | full-62 | 99.94 | 60.34 | 60.98 |
| 1 of 2 runs | relevance-45 | 99.97 | 60.69 | 62.02 |
