# Noise drawn afresh in every epoch

relevance-16-plus-k with the noise drawn once (fixed, scripts/input_cost_study.py) or afresh in every training epoch (fresh); scored inputs identical. Points of accuracy, 95% participant intervals. Lee2019 tests were fixed in the plan (2026-11-22T19:43); BCI IV 2a is exploratory.

## Lee2019 tests (54 people, final models, next day)

- F1, fresh noise costs less than fixed noise (fresh minus fixed): +12.18 [+8.77, +15.70]: holds
- F2, fresh noise has no visible cost (fresh minus 16 alone): -1.15 [-2.12, -0.23]: does not hold (lower bound above -2)
- F3, the network turns fresh noise down (weight ratio, fresh minus fixed): -0.88 [-0.92, -0.85]: holds; fresh median ratio 0.19 (below 1/3)

## Contrasts

| dataset | noise inputs | measure | contrast | mean | 95% interval | better | seed error |
|---|---:|---|---|---:|---|---:|---:|
| bnci2a | 6 | next_day | fresh - fixed | +5.37 | [+3.89, +6.95] | 9/9 | 0.31 |
| bnci2a | 6 | next_day | fresh - none | -1.92 | [-3.12, -0.90] | 1/9 | 0.65 |
| bnci2a | 6 | next_day | fixed - none | -7.29 | [-9.80, -5.26] | 0/9 | 0.49 |
| bnci2a | 6 | within | fresh - fixed | +4.31 | [+3.01, +5.85] | 9/9 | 0.51 |
| bnci2a | 6 | within | fresh - none | -3.05 | [-5.02, -1.31] | 1/9 | 0.84 |
| bnci2a | 6 | within | fixed - none | -7.36 | [-9.53, -5.31] | 0/9 | 1.16 |
| bnci2a | 16 | next_day | fresh - fixed | +8.81 | [+6.77, +10.72] | 9/9 | 0.57 |
| bnci2a | 16 | next_day | fresh - none | -3.70 | [-5.98, -2.00] | 0/9 | 0.63 |
| bnci2a | 16 | next_day | fixed - none | -12.51 | [-15.53, -9.48] | 0/9 | 0.56 |
| bnci2a | 16 | within | fresh - fixed | +11.01 | [+8.82, +13.74] | 9/9 | 1.25 |
| bnci2a | 16 | within | fresh - none | -3.62 | [-5.28, -2.02] | 1/9 | 0.89 |
| bnci2a | 16 | within | fixed - none | -14.63 | [-18.62, -11.58] | 0/9 | 0.74 |
| lee2019 | 46 | next_day | fresh - fixed | +12.18 | [+8.77, +15.70] | 47/54 |  |
| lee2019 | 46 | next_day | fresh - none | -1.15 | [-2.12, -0.23] | 21/54 |  |
| lee2019 | 46 | next_day | fixed - none | -13.33 | [-16.84, -9.99] | 2/54 |  |
| lee2019 | 46 | within | fresh - fixed | +8.74 | [+5.84, +11.86] | 44/54 |  |
| lee2019 | 46 | within | fresh - none | -2.09 | [-3.47, -0.96] | 14/54 |  |
| lee2019 | 46 | within | fixed - none | -10.83 | [-14.18, -7.69] | 7/54 |  |

## What the networks learned (final models, training day)

| dataset | noise inputs | noise | training trials, draw-0 noise % | training trials, unseen noise % | noise share of the spatial filters | noise / electrode weight size (median) |
|---|---:|---|---:|---:|---:|---:|
| bnci2a | 6 | fixed | 93.9 | 79.5 | 32% | 0.65 |
| bnci2a | 6 | fresh | 91.2 | 91.2 | 1% | 0.10 |
| bnci2a | 16 | fixed | 97.1 | 68.1 | 55% | 0.69 |
| bnci2a | 16 | fresh | 89.6 | 89.9 | 3% | 0.12 |
| lee2019 | 46 | fixed | 100.0 | 61.5 | 86% | 1.07 |
| lee2019 | 46 | fresh | 92.3 | 92.3 | 8% | 0.19 |
