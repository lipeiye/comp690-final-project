# Electrode count x training data

## Best count at each amount of training data

| dataset | training data | best count | its accuracy % | all electrodes % | bootstrap share of best counts |
|---|---|---:|---:|---:|---|
| bnci2a | 6 of 6 runs | 16 | 65.15 | 64.82 | 16: 66%, 22: 16%, 14: 15%, 18: 3% |
| bnci2a | 4 of 6 runs | 16 | 60.66 | 60.34 | 16: 49%, 22: 24%, 20: 17%, 18: 7% |
| bnci2a | 2 of 6 runs | 20 | 50.51 | 49.49 | 20: 41%, 12: 29%, 18: 13%, 16: 10% |
| lee2019 | 2 of 2 runs | 20 | 67.50 | 66.90 | 20: 55%, 45: 21%, 62: 14%, 16: 6% |
| lee2019 | 1 of 2 runs | 20 | 62.39 | 60.98 | 20: 70%, 45: 23%, 32: 4%, 8: 1% |

## Gap changes

| dataset | contrast | mean | 95% interval |
|---|---|---:|---|
| lee2019 | (16 minus 62) with 1 of 2 runs minus the same with all | +0.48 | [-1.05, +2.04] |
| bnci2a | (16 minus 22) with 4 of 6 runs minus the same with all | -0.00 | [-1.84, +2.22] |
| bnci2a | (16 minus 22) with 2 of 6 runs minus the same with all | +0.32 | [-0.99, +1.62] |

## Lee2019 tests

- D1: +0.48 [-1.05, +2.04]: does not hold
- D2: best count 20 with half the data, 20 with all; lower in 31% of resamples: does not hold

## Curves (next-day accuracy %)

| dataset | training data | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 32 | 45 | 62 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bnci2a | 6 of 6 runs | 55.63 | 58.20 | 59.83 | 62.23 | 63.65 | 64.50 | 65.15 | 64.56 | 63.55 | 64.82 |  |  |  |
| bnci2a | 4 of 6 runs | 52.47 | 55.35 | 56.52 | 57.28 | 59.53 | 60.18 | 60.66 | 60.29 | 60.36 | 60.34 |  |  |  |
| bnci2a | 2 of 6 runs | 46.16 | 48.27 | 49.04 | 49.57 | 50.41 | 50.10 | 50.14 | 50.34 | 50.51 | 49.49 |  |  |  |
| lee2019 | 2 of 2 runs |  |  | 64.42 |  |  |  | 67.02 |  | 67.50 |  | 66.95 | 67.17 | 66.90 |
| lee2019 | 1 of 2 runs |  |  | 61.47 |  |  |  | 61.59 |  | 62.39 |  | 61.73 | 62.02 | 60.98 |
