# A penalty no rescaling can lower

Each spatial filter's weights scaled to unit length, then lam times the sum over inputs of each input's norm over the filters (mibci.scalefree_penalty). lam chosen per dataset and condition on held-out runs of the training day. Accuracy in %, differences in points with the 95% participant interval. Lee2019 tests were fixed in the plan (2026-11-22T16:21); BCI IV 2a is exploratory.

## Lee2019 tests

- N1, noise inputs cost less with the penalty: +1.19 [-0.24, +2.63] (32/54 better): does not hold
- N2, all 62 electrodes beat relevance-16 with the penalty: -0.12 [-1.65, +1.38] (27/54 better): does not hold
- N3, some strength switches the noise off (ratio below 1/3) within 1 point of lam = 0 on held-out runs: does not hold

## Selected strengths (training day only)

| dataset | condition | selected lam | second reading |
|---|---|---:|---:|
| bnci2a | full-22 | 0.0 | 0.01 |
| bnci2a | relevance-16 | 0.0 | 0.003 |
| bnci2a | relevance-16-plus-6 | 0.01 |  |
| bnci2a | relevance-16-plus-16 | 0.01 |  |
| lee2019 | full-62 | 0.003 |  |
| lee2019 | relevance-16 | 0.003 |  |
| lee2019 | relevance-16-plus-46 | 0.1 |  |

## Fold models by strength (training day only)

| dataset | condition | lam | held-out % | minus lam 0 | 95% interval | noise/electrode g | plain norms | top-16 overlap with relevance-16 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| bnci2a | full-22 | 0.0 | 62.68 | +0.00 |  |  |  | 12.2 (chance 11.6) |
| bnci2a | full-22 | 0.001 | 62.59 | -0.08 | [-0.41, +0.26] |  |  | 12.2 (chance 11.6) |
| bnci2a | full-22 | 0.003 | 62.43 | -0.25 | [-1.01, +0.45] |  |  | 12.2 (chance 11.6) |
| bnci2a | full-22 | 0.01 | 62.36 | -0.32 | [-1.69, +0.98] |  |  | 11.8 (chance 11.6) |
| bnci2a | full-22 | 0.03 | 58.86 | -3.81 | [-5.47, -2.29] |  |  | 11.5 (chance 11.6) |
| bnci2a | full-22 | 0.1 | 51.89 | -10.79 | [-12.38, -9.26] |  |  | 11.4 (chance 11.6) |
| bnci2a | relevance-16 | 0.0 | 62.99 | +0.00 |  |  |  |  |
| bnci2a | relevance-16 | 0.001 | 62.80 | -0.20 | [-0.50, +0.09] |  |  |  |
| bnci2a | relevance-16 | 0.003 | 62.65 | -0.34 | [-1.23, +0.41] |  |  |  |
| bnci2a | relevance-16 | 0.01 | 61.52 | -1.47 | [-2.41, -0.48] |  |  |  |
| bnci2a | relevance-16 | 0.03 | 58.16 | -4.83 | [-6.54, -3.12] |  |  |  |
| bnci2a | relevance-16 | 0.1 | 51.63 | -11.36 | [-14.45, -8.70] |  |  |  |
| bnci2a | relevance-16-plus-16 | 0.0 | 48.36 | +0.00 |  | 0.695 | 0.706 |  |
| bnci2a | relevance-16-plus-16 | 0.001 | 48.92 | +0.56 | [+0.03, +1.06] | 0.656 | 0.667 |  |
| bnci2a | relevance-16-plus-16 | 0.003 | 48.86 | +0.49 | [-0.15, +1.18] | 0.558 | 0.569 |  |
| bnci2a | relevance-16-plus-16 | 0.01 | 48.97 | +0.61 | [-1.30, +2.21] | 0.502 | 0.509 |  |
| bnci2a | relevance-16-plus-16 | 0.03 | 48.02 | -0.34 | [-2.57, +1.58] | 1.482 | 1.589 |  |
| bnci2a | relevance-16-plus-16 | 0.1 | 40.75 | -7.61 | [-11.83, -3.48] | 0.802 | 0.824 |  |
| bnci2a | relevance-16-plus-6 | 0.0 | 55.63 | +0.00 |  | 0.644 | 0.638 |  |
| bnci2a | relevance-16-plus-6 | 0.001 | 55.98 | +0.35 | [-0.04, +0.72] | 0.592 | 0.594 |  |
| bnci2a | relevance-16-plus-6 | 0.003 | 56.12 | +0.48 | [-0.01, +0.96] | 0.521 | 0.512 |  |
| bnci2a | relevance-16-plus-6 | 0.01 | 56.63 | +1.00 | [-0.11, +2.12] | 0.476 | 0.488 |  |
| bnci2a | relevance-16-plus-6 | 0.03 | 54.08 | -1.55 | [-3.13, +0.10] | 1.435 | 1.475 |  |
| bnci2a | relevance-16-plus-6 | 0.1 | 46.47 | -9.17 | [-12.58, -5.52] | 0.913 | 0.941 |  |
| lee2019 | full-62 | 0.0 | 59.37 | +0.00 |  |  |  | 4.5 (chance 4.1) |
| lee2019 | full-62 | 0.001 | 59.40 | +0.03 | [-0.37, +0.42] |  |  | 4.4 (chance 4.1) |
| lee2019 | full-62 | 0.003 | 59.61 | +0.24 | [-0.42, +0.89] |  |  | 4.5 (chance 4.1) |
| lee2019 | full-62 | 0.01 | 58.42 | -0.95 | [-2.49, +0.60] |  |  | 4.2 (chance 4.1) |
| lee2019 | full-62 | 0.03 | 57.04 | -2.33 | [-4.19, -0.54] |  |  | 4.1 (chance 4.1) |
| lee2019 | full-62 | 0.1 | 53.98 | -5.39 | [-8.09, -2.83] |  |  | 4.0 (chance 4.1) |
| lee2019 | relevance-16 | 0.0 | 63.51 | +0.00 |  |  |  |  |
| lee2019 | relevance-16 | 0.001 | 63.57 | +0.06 | [-0.28, +0.45] |  |  |  |
| lee2019 | relevance-16 | 0.003 | 63.79 | +0.28 | [-0.21, +0.78] |  |  |  |
| lee2019 | relevance-16 | 0.01 | 63.68 | +0.17 | [-0.71, +0.99] |  |  |  |
| lee2019 | relevance-16 | 0.03 | 61.81 | -1.69 | [-2.86, -0.56] |  |  |  |
| lee2019 | relevance-16 | 0.1 | 57.82 | -5.69 | [-7.78, -3.70] |  |  |  |
| lee2019 | relevance-16-plus-46 | 0.0 | 51.65 | +0.00 |  | 0.962 | 0.960 |  |
| lee2019 | relevance-16-plus-46 | 0.001 | 51.26 | -0.39 | [-0.66, -0.12] | 0.957 | 0.953 |  |
| lee2019 | relevance-16-plus-46 | 0.003 | 51.53 | -0.12 | [-0.58, +0.35] | 0.897 | 0.893 |  |
| lee2019 | relevance-16-plus-46 | 0.01 | 51.95 | +0.31 | [-0.51, +1.12] | 0.885 | 0.897 |  |
| lee2019 | relevance-16-plus-46 | 0.03 | 51.74 | +0.09 | [-1.03, +1.21] | 1.778 | 1.797 |  |
| lee2019 | relevance-16-plus-46 | 0.1 | 53.21 | +1.56 | [-0.21, +3.31] | 0.631 | 0.633 |  |

## Next day, selected strengths

| contrast | mean | 95% interval |
|---|---:|---|
| bnci2a|full minus relevance-16, penalized | -0.32 | [-1.04, +0.42] |
| bnci2a|full minus relevance-16, unpenalized | -0.32 | [-1.04, +0.42] |
| bnci2a|relevance-16-plus-6 cost, penalized | -6.49 | [-8.64, -4.54] |
| bnci2a|relevance-16-plus-6 cost, unpenalized | -7.29 | [-9.80, -5.26] |
| bnci2a|relevance-16-plus-6 cost change | +0.79 | [-0.20, +1.78] |
| bnci2a|relevance-16-plus-16 cost, penalized | -11.17 | [-14.24, -8.11] |
| bnci2a|relevance-16-plus-16 cost, unpenalized | -12.51 | [-15.53, -9.48] |
| bnci2a|relevance-16-plus-16 cost change | +1.35 | [+0.82, +2.00] |
| bnci2a|relevance-16-plus-6 at lam 0.01 minus lam 0 | +0.79 | [-0.20, +1.78] |
| bnci2a|relevance-16-plus-16 at lam 0.01 minus lam 0 | +1.35 | [+0.82, +2.00] |
| lee2019|full minus relevance-16, penalized | -0.12 | [-1.65, +1.38] |
| lee2019|full minus relevance-16, unpenalized | -0.13 | [-1.74, +1.46] |
| lee2019|relevance-16-plus-46 cost, penalized | -12.15 | [-15.40, -9.06] |
| lee2019|relevance-16-plus-46 cost, unpenalized | -13.33 | [-16.84, -9.99] |
| lee2019|relevance-16-plus-46 cost change | +1.19 | [-0.24, +2.63] |
| lee2019|full-62 at lam 0.003 minus lam 0 | +0.07 | [-0.50, +0.65] |
| lee2019|relevance-16 at lam 0.003 minus lam 0 | +0.06 | [-0.29, +0.41] |
| lee2019|relevance-16-plus-46 at lam 0.1 minus lam 0 | +1.25 | [-0.21, +2.73] |

## Next day, second reading

| contrast | mean | 95% interval |
|---|---:|---|
| bnci2a|full minus relevance-16, penalized | -2.07 | [-3.56, -0.64] |
| bnci2a|full minus relevance-16, unpenalized | -0.32 | [-1.04, +0.42] |
| bnci2a|relevance-16-plus-6 cost, penalized | -6.10 | [-8.41, -3.93] |
| bnci2a|relevance-16-plus-6 cost, unpenalized | -7.29 | [-9.80, -5.26] |
| bnci2a|relevance-16-plus-6 cost change | +1.19 | [-0.02, +2.32] |
| bnci2a|relevance-16-plus-16 cost, penalized | -10.77 | [-14.05, -7.48] |
| bnci2a|relevance-16-plus-16 cost, unpenalized | -12.51 | [-15.53, -9.48] |
| bnci2a|relevance-16-plus-16 cost change | +1.74 | [+0.77, +2.72] |
| bnci2a|full-22 at lam 0.01 minus lam 0 | -2.14 | [-3.54, -0.96] |
| bnci2a|relevance-16 at lam 0.003 minus lam 0 | -0.40 | [-0.88, +0.11] |
| bnci2a|relevance-16-plus-6 at lam 0.01 minus lam 0 | +0.79 | [-0.20, +1.78] |
| bnci2a|relevance-16-plus-16 at lam 0.01 minus lam 0 | +1.35 | [+0.82, +2.00] |
| lee2019|full minus relevance-16, penalized | -0.12 | [-1.65, +1.38] |
| lee2019|full minus relevance-16, unpenalized | -0.13 | [-1.74, +1.46] |
| lee2019|relevance-16-plus-46 cost, penalized | -12.15 | [-15.40, -9.06] |
| lee2019|relevance-16-plus-46 cost, unpenalized | -13.33 | [-16.84, -9.99] |
| lee2019|relevance-16-plus-46 cost change | +1.19 | [-0.24, +2.63] |
| lee2019|full-62 at lam 0.003 minus lam 0 | +0.07 | [-0.50, +0.65] |
| lee2019|relevance-16 at lam 0.003 minus lam 0 | +0.06 | [-0.29, +0.41] |
| lee2019|relevance-16-plus-46 at lam 0.1 minus lam 0 | +1.25 | [-0.21, +2.73] |
