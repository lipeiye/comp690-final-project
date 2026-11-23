# How much the dropped electrodes add, and what they cost

Extra inputs after the 16 kept electrodes (2a: the 6 dropped, Lee2019: the 46 dropped): the real electrodes; their least-squares prediction from the 16 (fitted on training trials only); and the prediction plus another trial's residual (surr, mean of the draws). Points of accuracy, 95% participant intervals. Lee2019 tests were fixed in the plan (2026-11-22T19:07); BCI IV 2a is exploratory.

## Lee2019 tests (54 people, final models, next day)

- S1, the unpredictable part carries information (real minus surr): +4.30 [+3.09, +5.50]: holds
- S2, a residual unrelated to the trial still costs (surr minus 16 alone): -4.12 [-5.31, -3.02]: holds
- S3, a fully predictable input has no visible cost (pred minus 16 alone): +0.27 [-0.49, +1.02]: holds (lower bound above -2)
- S4, more unexplained, larger cost (Spearman over 54 people): rho -0.079 [-0.340, +0.200]: does not hold
- Post hoc, for S4: a person's cost from seed 0 against seeds 1-2, Spearman over 54 people: rho +0.493 (does a person's cost repeat at all?)

## Decomposition

| dataset | measure | term | contrast | mean | 95% interval | worse | seed error |
|---|---|---|---|---:|---|---:|---:|
| bnci2a | next_day | cost | surr - none | -1.50 | [-2.29, -0.70] | 7/9 | 0.34 |
| bnci2a | next_day | information | real - surr | +0.34 | [-0.53, +1.18] | 4/9 | 0.71 |
| bnci2a | next_day | sum | real - none | -1.16 | [-2.45, +0.18] | 6/9 | 0.81 |
| bnci2a | next_day | pred_cost | pred - none | +0.48 | [-0.49, +1.63] | 5/9 | 0.67 |
| bnci2a | next_day | order | real - full | -0.84 | [-2.32, +0.53] | 6/9 | 0.65 |
| bnci2a | next_day | cost_d0 | surr-d0 - none | -1.53 | [-2.67, -0.57] | 8/9 | 0.55 |
| bnci2a | next_day | cost_d1 | surr-d1 - none | -1.46 | [-2.32, -0.57] | 6/9 | 0.43 |
| bnci2a | next_day | cost_d2 | surr-d2 - none | -1.52 | [-2.60, -0.35] | 7/9 | 0.35 |
| bnci2a | within | cost | surr - none | -2.62 | [-3.93, -1.19] | 7/9 | 0.86 |
| bnci2a | within | information | real - surr | +2.20 | [+1.14, +3.21] | 2/9 | 0.35 |
| bnci2a | within | sum | real - none | -0.42 | [-1.49, +0.96] | 6/9 | 0.79 |
| bnci2a | within | pred_cost | pred - none | +0.34 | [-1.05, +1.78] | 5/9 | 1.05 |
| bnci2a | within | order | real - full | -0.11 | [-1.71, +1.51] | 4/9 | 0.41 |
| bnci2a | within | cost_d0 | surr-d0 - none | -3.05 | [-5.11, -1.02] | 6/9 | 1.03 |
| bnci2a | within | cost_d1 | surr-d1 - none | -1.48 | [-2.47, -0.52] | 7/9 | 1.02 |
| bnci2a | within | cost_d2 | surr-d2 - none | -3.34 | [-4.72, -1.69] | 7/9 | 0.65 |
| lee2019 | next_day | cost | surr - none | -4.12 | [-5.31, -3.02] | 45/54 |  |
| lee2019 | next_day | information | real - surr | +4.30 | [+3.09, +5.50] | 7/54 |  |
| lee2019 | next_day | sum | real - none | +0.18 | [-1.25, +1.56] | 23/54 |  |
| lee2019 | next_day | pred_cost | pred - none | +0.27 | [-0.49, +1.02] | 21/54 |  |
| lee2019 | next_day | order | real - full | +0.31 | [-0.73, +1.42] | 30/54 |  |
| lee2019 | next_day | cost_d0 | surr-d0 - none | -4.42 | [-5.82, -3.20] | 45/54 |  |
| lee2019 | next_day | cost_d1 | surr-d1 - none | -3.82 | [-5.07, -2.63] | 43/54 |  |
| lee2019 | within | cost | surr - none | -5.36 | [-7.27, -3.60] | 44/54 |  |
| lee2019 | within | information | full - surr | +3.10 | [+1.74, +4.42] | 15/54 |  |
| lee2019 | within | sum | full - none | -2.27 | [-4.23, -0.51] | 35/54 |  |
| lee2019 | within | pred_cost | pred - none | +0.12 | [-0.63, +0.91] | 29/54 |  |
| lee2019 | within | cost_d0 | surr-d0 - none | -5.36 | [-7.27, -3.60] | 44/54 |  |

Interpolated in the input-kind study (cost of the real electrodes, next day): bnci2a -1.73, lee2019 -4.49. On Lee2019's held-out runs, full-62 stands in for real.

## Every condition

| dataset | condition | next day % | held-out runs % | training set % | residual share (train) | extra inputs' share of the spatial filters |
|---|---|---:|---:|---:|---:|---:|
| bnci2a | 16 electrodes alone | 65.15 | 62.99 | 92.5 | |  |
| bnci2a | real dropped electrodes | 63.98 | 62.57 | 93.2 | 0.033 | 36% |
| bnci2a | their prediction from the 16 alone | 65.63 | 63.33 | 92.3 | 0.033 | 36% |
| bnci2a | prediction + another trial's residual (mean of draws) | 63.64 | 60.37 | 93.0 | 0.033 | 34% |
| bnci2a | all electrodes (another input order) | 64.82 | 62.68 | 93.1 | |  |
| lee2019 | 16 electrodes alone | 67.02 | 62.60 | 96.1 | |  |
| lee2019 | real dropped electrodes | 67.20 | | 98.9 | 0.250 | 61% |
| lee2019 | their prediction from the 16 alone | 67.30 | 62.73 | 96.0 | 0.250 | 61% |
| lee2019 | prediction + another trial's residual (mean of draws) | 62.90 | 57.24 | 99.0 | 0.250 | 60% |
| lee2019 | all electrodes (another input order) | 66.90 | 60.34 | 99.0 | |  |
