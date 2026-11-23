# Channel dropout: the batch norms and a failed electrode

Accuracy in %, differences in points with the 95% participant interval. p = 0 without dropout; p = 0.2 final models are the archived ones of the overfitting study. Re-estimated: batch norms after the spatial filters re-estimated on the training day with every electrode present. Lee2019 tests were fixed in the plan (2026-11-22T16:30); BCI IV 2a is exploratory.

Checks: archived final models scored as trained reproduce their archived predictions in 1008/1008; fold models without dropout reproduce their held-out predictions in 459/459.

## Lee2019 tests

- B1, re-estimating recovers the p = 0.2 models: +0.38 [-0.04, +0.81] (35/54 positive): does not hold; it recovers 14% of the 2.74-point loss
- B2, p = 0.05 costs less than p = 0.2 (as trained): +2.45 [+1.56, +3.37] (37/54 positive): holds; re-estimated: +2.26 [+1.43, +3.14] (40/54 positive)
- B3, a failed electrode costs the dropout model (p = 0.05, chosen on held-out runs) less: +0.49 [+0.28, +0.72] (38/54 positive): holds

## Next day (final models) and held-out runs of the training day (fold models)

| dataset | montage | p | version | next day % | held-out runs % |
|---|---|---:|---|---:|---:|
| bnci2a | full | 0.0 | as trained | 64.82 | 62.68 |
| bnci2a | full | 0.0 | recalibrated | 65.29 | 62.96 |
| bnci2a | full | 0.05 | as trained | 60.07 | 57.59 |
| bnci2a | full | 0.05 | recalibrated | 60.46 | 57.58 |
| bnci2a | full | 0.1 | as trained | 57.33 | 55.39 |
| bnci2a | full | 0.1 | recalibrated | 57.77 | 55.68 |
| bnci2a | full | 0.2 | as trained | 53.75 | 52.23 |
| bnci2a | full | 0.2 | recalibrated | 54.35 | 52.75 |
| bnci2a | reduced | 0.0 | as trained | 65.15 |  |
| bnci2a | reduced | 0.0 | recalibrated | 65.66 |  |
| bnci2a | reduced | 0.2 | as trained | 53.33 |  |
| bnci2a | reduced | 0.2 | recalibrated | 53.78 |  |
| lee2019 | full | 0.0 | as trained | 66.90 | 60.34 |
| lee2019 | full | 0.0 | recalibrated | 67.19 | 60.52 |
| lee2019 | full | 0.05 | as trained | 66.60 | 60.46 |
| lee2019 | full | 0.05 | recalibrated | 66.80 | 60.52 |
| lee2019 | full | 0.1 | as trained | 65.56 | 59.67 |
| lee2019 | full | 0.1 | recalibrated | 65.78 | 59.67 |
| lee2019 | full | 0.2 | as trained | 64.16 | 58.27 |
| lee2019 | full | 0.2 | recalibrated | 64.54 | 58.49 |
| lee2019 | reduced | 0.0 | as trained | 67.17 |  |
| lee2019 | reduced | 0.0 | recalibrated | 67.16 |  |
| lee2019 | reduced | 0.2 | as trained | 64.56 |  |
| lee2019 | reduced | 0.2 | recalibrated | 64.73 |  |

## Stored over actual variance, per batch norm (median over channels, mean over models)

| dataset | montage | p | version | temporal | after spatial (bnorm_1) | after separable (bnorm_2) |
|---|---|---:|---|---:|---:|---:|
| bnci2a | full | 0.0 | as trained | 0.985 | 0.983 | 1.257 |
| bnci2a | full | 0.0 | recalibrated | 0.985 | 1.000 | 1.000 |
| bnci2a | full | 0.05 | as trained | 0.934 | 0.976 | 1.272 |
| bnci2a | full | 0.05 | recalibrated | 0.934 | 1.000 | 1.000 |
| bnci2a | full | 0.1 | as trained | 0.887 | 0.919 | 1.181 |
| bnci2a | full | 0.1 | recalibrated | 0.887 | 1.000 | 1.000 |
| bnci2a | full | 0.2 | as trained | 0.787 | 0.758 | 0.986 |
| bnci2a | full | 0.2 | recalibrated | 0.787 | 1.000 | 1.000 |
| bnci2a | reduced | 0.0 | as trained | 0.983 | 0.988 | 1.268 |
| bnci2a | reduced | 0.0 | recalibrated | 0.983 | 1.000 | 1.000 |
| bnci2a | reduced | 0.2 | as trained | 0.790 | 0.798 | 1.022 |
| bnci2a | reduced | 0.2 | recalibrated | 0.790 | 1.000 | 1.000 |
| lee2019 | full | 0.0 | as trained | 0.984 | 1.012 | 1.329 |
| lee2019 | full | 0.0 | recalibrated | 0.984 | 1.000 | 1.000 |
| lee2019 | full | 0.05 | as trained | 0.933 | 0.966 | 1.327 |
| lee2019 | full | 0.05 | recalibrated | 0.933 | 1.000 | 1.000 |
| lee2019 | full | 0.1 | as trained | 0.881 | 0.885 | 1.218 |
| lee2019 | full | 0.1 | recalibrated | 0.881 | 1.000 | 1.000 |
| lee2019 | full | 0.2 | as trained | 0.783 | 0.720 | 1.005 |
| lee2019 | full | 0.2 | recalibrated | 0.783 | 1.000 | 1.000 |
| lee2019 | reduced | 0.0 | as trained | 0.981 | 1.002 | 1.345 |
| lee2019 | reduced | 0.0 | recalibrated | 0.981 | 1.000 | 1.000 |
| lee2019 | reduced | 0.2 | as trained | 0.783 | 0.725 | 1.004 |
| lee2019 | reduced | 0.2 | recalibrated | 0.783 | 1.000 | 1.000 |

## A failed electrode, next day (all-electrode models; points lost)

| dataset | p | version | one electrode, mean | one electrode, worst | three electrodes, mean |
|---|---:|---|---:|---:|---:|
| bnci2a | 0.0 | as trained | 14.80 | 31.74 | 28.11 |
| bnci2a | 0.0 | recalibrated | 14.78 | 31.83 | 28.26 |
| bnci2a | 0.05 | as trained | 1.70 | 9.91 | 5.76 |
| bnci2a | 0.05 | recalibrated | 1.93 | 10.34 | 6.08 |
| bnci2a | 0.1 | as trained | 0.92 | 7.32 | 2.88 |
| bnci2a | 0.1 | recalibrated | 1.04 | 7.94 | 3.23 |
| bnci2a | 0.2 | as trained | 0.44 | 6.44 | 1.23 |
| bnci2a | 0.2 | recalibrated | 0.85 | 9.03 | 2.26 |
| lee2019 | 0.0 | as trained | 0.81 | 10.74 | 1.92 |
| lee2019 | 0.0 | recalibrated | 0.75 | 10.95 | 2.01 |
| lee2019 | 0.05 | as trained | 0.27 | 6.02 | 0.67 |
| lee2019 | 0.05 | recalibrated | 0.32 | 6.36 | 0.80 |
| lee2019 | 0.1 | as trained | 0.09 | 4.89 | 0.31 |
| lee2019 | 0.1 | recalibrated | 0.15 | 4.92 | 0.38 |
| lee2019 | 0.2 | as trained | 0.07 | 3.54 | 0.19 |
| lee2019 | 0.2 | recalibrated | 0.18 | 4.12 | 0.43 |

## BCI IV 2a contrasts (exploratory)

| contrast | mean | 95% interval |
|---|---:|---|
| full: p 0.2 re-estimated minus as trained | +0.60 | [+0.15, +1.07] |
| full: p 0 re-estimated minus as trained | +0.47 | [+0.11, +0.86] |
| reduced: p 0.2 re-estimated minus as trained | +0.45 | [+0.07, +0.79] |
| reduced: p 0 re-estimated minus as trained | +0.51 | [+0.06, +1.06] |
| full: p 0.05 minus p 0, as_trained | -4.75 | [-7.04, -2.66] |
| full: p 0.05 minus p 0, recalibrated | -4.36 | [-6.41, -2.51] |
| full: p 0.1 minus p 0, as_trained | -7.49 | [-10.14, -5.23] |
| full: p 0.1 minus p 0, recalibrated | -7.05 | [-9.52, -4.88] |
| full: p 0.2 minus p 0, as_trained | -11.07 | [-13.79, -8.82] |
| full: p 0.2 minus p 0, recalibrated | -10.47 | [-13.11, -8.26] |
| one failed electrode: p 0 as trained minus p 0.05 re-estimated (cost) | +12.87 | [+9.09, +16.96] |
