# Letting EEGNet switch inputs off: a group-lasso penalty on the spatial weights

## lam chosen on held-out runs of the training day (accuracy %)

| dataset | condition | lam 0 | lam 0.001 | lam 0.003 | lam 0.01 | lam 0.03 | lam 0.1 | selected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bnci2a | full-22 | 62.68 | 62.21 | 62.15 | 61.94 | 60.94 | 56.96 | 0 |
| bnci2a | relevance-16 | 62.99 | 63.05 | 63.24 | 63.94 | 61.58 | 58.08 | 0.01 |
| bnci2a | relevance-16-plus-6 | 55.63 | 56.38 | 56.46 | 55.95 | 55.94 | 54.50 | 0.003 |
| bnci2a | relevance-16-plus-16 | 48.36 | 49.00 | 49.44 | 50.25 | 48.83 | 47.91 | 0.01 |
| lee2019 | full-62 | 59.37 | 59.69 | 60.10 | 59.78 | 58.47 | 56.33 | 0.003 |
| lee2019 | relevance-16 | 63.51 | 63.44 | 63.45 | 63.07 | 62.67 | 62.22 | 0 |
| lee2019 | relevance-16-plus-46 | 51.65 | 51.36 | 51.33 | 51.35 | 51.13 | 51.25 | 0 |

## Next day, final models (accuracy %)

| dataset | condition | selected lam | no penalty | penalty |
|---|---|---:|---:|---:|
| bnci2a | full-22 | 0 | 64.82 | 64.82 |
| bnci2a | relevance-16 | 0.01 | 65.15 | 65.29 |
| bnci2a | relevance-16-plus-16 | 0.01 | 52.63 | 53.36 |
| bnci2a | relevance-16-plus-6 | 0.003 | 57.86 | 58.01 |
| lee2019 | full-62 | 0.003 | 66.90 | 67.33 |
| lee2019 | relevance-16 | 0 | 67.02 | 67.02 |
| lee2019 | relevance-16-plus-46 | 0 | 53.69 | 53.69 |

## Contrasts (next day, pp, 95% participant interval)

### bnci2a

- full-22 minus relevance-16, unpenalized: -0.32 [-1.04, +0.42] (3/9 better)
- full-22 minus relevance-16, penalized: -0.47 [-1.43, +0.62] (3/9 better)
- relevance-16: penalized minus unpenalized: +0.15 [-0.40, +0.78] (4/9 better)
- full-22: penalized minus unpenalized: +0.00 [+0.00, +0.00] (0/9 better)
- relevance-16-plus-6: cost unpenalized: -7.29 [-9.80, -5.26] (0/9 better)
- relevance-16-plus-6: cost penalized: -7.28 [-9.65, -5.25] (0/9 better)
- relevance-16-plus-6: penalized cost minus unpenalized cost: +0.01 [-0.76, +0.59] (6/9 better)
- relevance-16-plus-16: cost unpenalized: -12.51 [-15.53, -9.48] (0/9 better)
- relevance-16-plus-16: cost penalized: -11.94 [-14.49, -9.06] (0/9 better)
- relevance-16-plus-16: penalized cost minus unpenalized cost: +0.58 [-0.43, +1.59] (6/9 better)

### lee2019

- full-62 minus relevance-16, unpenalized: -0.13 [-1.74, +1.46] (29/54 better)
- full-62 minus relevance-16, penalized: +0.31 [-1.11, +1.71] (26/54 better)
- relevance-16: penalized minus unpenalized: +0.00 [+0.00, +0.00] (0/54 better)
- full-62: penalized minus unpenalized: +0.44 [-0.20, +1.15] (25/54 better)
- relevance-16-plus-46: cost unpenalized: -13.33 [-16.84, -9.99] (2/54 better)
- relevance-16-plus-46: cost penalized: -13.33 [-16.84, -9.99] (2/54 better)
- relevance-16-plus-46: penalized cost minus unpenalized cost: +0.00 [+0.00, +0.00] (0/54 better)

## Lee2019 tests

- G1 (inputs that carry nothing cost less with the penalty): +0.00 [+0.00, +0.00] (0/54 better): does not hold
- G2 (with the penalty all 62 electrodes beat the best 16): +0.31 [-1.11, +1.71] (26/54 better): does not hold
- G3 (noise inputs switched off): median noise-input norm / electrode norm = nan (nan: lam = 0 was selected, so no penalized model): not applicable

## Input norms of the penalized models (medians)

| dataset | condition | lam | final | electrodes | noise inputs | electrodes switched off |
|---|---|---:|---|---:|---:|---:|
| bnci2a | full-22 | 0.001 | folds | 0.282 |  | 0% |
| bnci2a | full-22 | 0.003 | folds | 0.225 |  | 0% |
| bnci2a | full-22 | 0.01 | folds | 0.142 |  | 0% |
| bnci2a | full-22 | 0.03 | folds | 0.074 |  | 1% |
| bnci2a | full-22 | 0.1 | folds | 0.028 |  | 7% |
| bnci2a | relevance-16 | 0.001 | folds | 0.318 |  | 0% |
| bnci2a | relevance-16 | 0.003 | folds | 0.257 |  | 0% |
| bnci2a | relevance-16 | 0.01 | folds | 0.166 |  | 0% |
| bnci2a | relevance-16 | 0.01 | yes | 0.178 |  | 0% |
| bnci2a | relevance-16 | 0.03 | folds | 0.090 |  | 0% |
| bnci2a | relevance-16 | 0.1 | folds | 0.037 |  | 5% |
| bnci2a | relevance-16-plus-16 | 0.001 | folds | 0.275 | 0.175 | 0% |
| bnci2a | relevance-16-plus-16 | 0.003 | folds | 0.212 | 0.123 | 0% |
| bnci2a | relevance-16-plus-16 | 0.01 | folds | 0.130 | 0.071 | 0% |
| bnci2a | relevance-16-plus-16 | 0.01 | yes | 0.152 | 0.089 | 0% |
| bnci2a | relevance-16-plus-16 | 0.03 | folds | 0.073 | 0.045 | 0% |
| bnci2a | relevance-16-plus-16 | 0.1 | folds | 0.031 | 0.025 | 4% |
| bnci2a | relevance-16-plus-6 | 0.001 | folds | 0.301 | 0.178 | 0% |
| bnci2a | relevance-16-plus-6 | 0.003 | folds | 0.235 | 0.126 | 0% |
| bnci2a | relevance-16-plus-6 | 0.003 | yes | 0.256 | 0.140 | 0% |
| bnci2a | relevance-16-plus-6 | 0.01 | folds | 0.146 | 0.074 | 0% |
| bnci2a | relevance-16-plus-6 | 0.03 | folds | 0.082 | 0.046 | 0% |
| bnci2a | relevance-16-plus-6 | 0.1 | folds | 0.035 | 0.026 | 4% |
| lee2019 | full-62 | 0.001 | folds | 0.160 |  | 0% |
| lee2019 | full-62 | 0.003 | folds | 0.114 |  | 0% |
| lee2019 | full-62 | 0.003 | yes | 0.140 |  | 0% |
| lee2019 | full-62 | 0.01 | folds | 0.058 |  | 2% |
| lee2019 | full-62 | 0.03 | folds | 0.027 |  | 6% |
| lee2019 | full-62 | 0.1 | folds | 0.010 |  | 11% |
| lee2019 | relevance-16 | 0.001 | folds | 0.301 |  | 0% |
| lee2019 | relevance-16 | 0.003 | folds | 0.235 |  | 0% |
| lee2019 | relevance-16 | 0.01 | folds | 0.138 |  | 0% |
| lee2019 | relevance-16 | 0.03 | folds | 0.068 |  | 1% |
| lee2019 | relevance-16 | 0.1 | folds | 0.027 |  | 4% |
| lee2019 | relevance-16-plus-46 | 0.001 | folds | 0.143 | 0.118 | 0% |
| lee2019 | relevance-16-plus-46 | 0.003 | folds | 0.093 | 0.067 | 0% |
| lee2019 | relevance-16-plus-46 | 0.01 | folds | 0.041 | 0.034 | 1% |
| lee2019 | relevance-16-plus-46 | 0.03 | folds | 0.017 | 0.020 | 2% |
| lee2019 | relevance-16-plus-46 | 0.1 | folds | 0.006 | 0.011 | 2% |
