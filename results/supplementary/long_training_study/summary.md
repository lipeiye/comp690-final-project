# Training four times as long

The frozen recipe run for 1900 epochs; checkpoints at 475, 950, 1900. Accuracy in %, differences in percentage points with the 95% participant interval. Next day: final models; held-out runs: fold models of the training day. Lee2019 tests were fixed in the plan (2026-11-22T15:38); BCI IV 2a is exploratory.

475-epoch checkpoints with exactly the earlier study's weights: 495 of 1485 compared (largest difference 0.404).

## Lee2019 tests (54 people)

- L1, the noise inputs cost less after 1900 epochs ((plus-46 minus relevance-16) at 1900 minus at 475): +0.52 [-0.40, +1.49]: does not hold
- L2, all 62 electrodes beat relevance-16 after 1900 epochs: +0.68 [-0.61, +1.94]: does not hold
- L3, held-out accuracy with noise inputs still rises after epoch 475 (epochs 1800-1900 minus 425-525): +0.19 [-0.48, +0.85]: does not hold

## Next-day accuracy at each checkpoint (final models)

| dataset | condition | 475 epochs | 950 epochs | 1900 epochs | training set at 1900 |
|---|---|---:|---:|---:|---:|
| bnci2a | full-22 | 64.89 | 66.31 | 66.93 | 99.5 |
| bnci2a | relevance-16 | 65.15 | 66.17 | 66.83 | 99.2 |
| bnci2a | relevance-16-plus-6 | 57.86 | 59.09 | 60.34 | 99.8 |
| lee2019 | full-62 | 66.83 | 67.11 | 67.32 | 99.9 |
| lee2019 | relevance-16 | 67.01 | 66.85 | 66.64 | 99.9 |
| lee2019 | relevance-16-plus-46 | 53.69 | 53.94 | 53.85 | 100.0 |

## Contrasts

| dataset | contrast | epochs | mean | 95% interval | seed error | worse |
|---|---|---:|---:|---|---:|---:|
| bnci2a | noise cost | 475 | -7.29 | [-9.76, -5.29] | 0.50 | 9/9 |
| bnci2a | all electrodes minus relevance-16 | 475 | -0.26 | [-1.00, +0.51] | 0.43 | 6/9 |
| bnci2a | noise cost | 950 | -7.08 | [-9.19, -5.38] | 0.48 | 9/9 |
| bnci2a | all electrodes minus relevance-16 | 950 | +0.14 | [-0.92, +1.31] | 0.53 | 5/9 |
| bnci2a | noise cost | 1900 | -6.49 | [-8.86, -4.47] | 0.78 | 9/9 |
| bnci2a | all electrodes minus relevance-16 | 1900 | +0.10 | [-1.17, +1.42] | 0.63 | 4/9 |
| bnci2a | relevance-16: 1900 minus 475 epochs | 1900 | +1.68 | [+0.39, +3.21] | 0.33 | 2/9 |
| bnci2a | relevance-16-plus-6: 1900 minus 475 epochs | 1900 | +2.48 | [+0.71, +4.28] | 0.48 | 3/9 |
| bnci2a | full-22: 1900 minus 475 epochs | 1900 | +2.04 | [+0.01, +4.43] | 0.44 | 2/9 |
| bnci2a | noise cost at 1900 minus at 475 epochs | 1900 | +0.80 | [-0.60, +2.26] | 0.63 | 2/9 |
| lee2019 | noise cost | 475 | -13.31 | [-16.81, -9.96] |  | 52/54 |
| lee2019 | all electrodes minus relevance-16 | 475 | -0.17 | [-1.78, +1.42] |  | 25/54 |
| lee2019 | noise cost | 950 | -12.91 | [-16.36, -9.61] |  | 50/54 |
| lee2019 | all electrodes minus relevance-16 | 950 | +0.26 | [-1.26, +1.78] |  | 24/54 |
| lee2019 | noise cost | 1900 | -12.79 | [-16.11, -9.62] |  | 51/54 |
| lee2019 | all electrodes minus relevance-16 | 1900 | +0.68 | [-0.61, +1.94] |  | 23/54 |
| lee2019 | relevance-16: 1900 minus 475 epochs | 1900 | -0.36 | [-1.25, +0.48] |  | 24/54 |
| lee2019 | relevance-16-plus-46: 1900 minus 475 epochs | 1900 | +0.15 | [-0.30, +0.62] |  | 26/54 |
| lee2019 | full-62: 1900 minus 475 epochs | 1900 | +0.49 | [-0.39, +1.45] |  | 26/54 |
| lee2019 | noise cost at 1900 minus at 475 epochs | 1900 | +0.52 | [-0.40, +1.49] |  | 30/54 |

## Held-out runs of the training day: epochs 1800-1900 minus 425-525

| dataset | condition | mean | 95% interval |
|---|---|---:|---|
| bnci2a | relevance-16 | +3.14 | [+1.32, +5.18] |
| bnci2a | relevance-16-plus-6 | +1.90 | [+0.95, +3.12] |
| bnci2a | full-22 | +3.72 | [+1.67, +6.40] |
| lee2019 | relevance-16 | -0.08 | [-0.77, +0.61] |
| lee2019 | relevance-16-plus-46 | +0.19 | [-0.48, +0.85] |
| lee2019 | full-62 | +1.25 | [+0.46, +2.04] |
