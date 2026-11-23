# Why an input that carries nothing costs EEGNet accuracy

## Mechanism (held-out runs of the training day)

Held-out accuracy recomputed from the saved models equals the record for 3834 of 3834 models.

| dataset | training data | added | none % | as trained | cost | batch norms re-estimated | noise weights 0 | noise weights 0, re-estimated | recovered | still lost | share of the cost recovered |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| bnci2a | 4 of 6 runs | 1 | 62.99 | 62.03 | -0.96 [-2.63, +0.48] | 62.03 | 62.01 | 62.41 | +0.38 [-0.02, +0.78] | -0.90 [-2.10, +0.15] | 30% |
| bnci2a | 4 of 6 runs | 2 | 62.99 | 61.37 | -1.62 [-2.99, -0.30] | 61.37 | 61.31 | 61.84 | +0.47 [+0.08, +0.90] | -1.47 [-2.61, -0.23] | 24% |
| bnci2a | 4 of 6 runs | 4 | 62.99 | 57.77 | -5.22 [-7.37, -3.04] | 57.93 | 56.82 | 58.40 | +0.47 [-0.03, +0.95] | -4.91 [-6.70, -2.97] | 9% |
| bnci2a | 4 of 6 runs | 6 | 62.99 | 55.63 | -7.36 [-9.53, -5.31] | 55.96 | 54.72 | 56.87 | +0.91 [+0.08, +1.71] | -6.44 [-8.65, -4.29] | 12% |
| bnci2a | 4 of 6 runs | 10 | 62.99 | 52.41 | -10.59 [-13.67, -7.99] | 52.64 | 50.55 | 54.26 | +1.62 [+0.66, +2.56] | -9.05 [-11.80, -6.57] | 15% |
| bnci2a | 4 of 6 runs | 16 | 62.99 | 48.36 | -14.63 [-18.62, -11.58] | 48.81 | 44.28 | 51.57 | +2.76 [+1.69, +3.86] | -11.73 [-15.50, -8.60] | 19% |
| bnci2a | 2 of 6 runs | 1 | 52.13 | 51.30 | -0.82 [-2.03, +0.47] | 51.68 | 51.23 | 51.93 | +0.25 [-0.22, +0.73] | -0.81 [-1.77, +0.29] | 23% |
| bnci2a | 2 of 6 runs | 2 | 52.13 | 49.74 | -2.39 [-3.28, -1.57] | 50.27 | 49.14 | 50.85 | +0.57 [+0.15, +0.99] | -1.89 [-2.66, -1.13] | 23% |
| bnci2a | 2 of 6 runs | 4 | 52.13 | 47.11 | -5.02 [-6.80, -3.22] | 47.57 | 46.12 | 48.12 | +0.55 [+0.33, +0.75] | -4.62 [-6.07, -3.03] | 11% |
| bnci2a | 2 of 6 runs | 6 | 52.13 | 46.27 | -5.85 [-7.45, -4.20] | 46.51 | 44.46 | 46.88 | +0.37 [-0.14, +0.86] | -5.86 [-7.07, -4.63] | 6% |
| bnci2a | 2 of 6 runs | 10 | 52.13 | 41.94 | -10.19 [-13.52, -7.17] | 42.24 | 40.67 | 43.64 | +1.40 [+0.59, +2.31] | -9.10 [-12.27, -6.21] | 13% |
| bnci2a | 2 of 6 runs | 16 | 52.13 | 38.93 | -13.19 [-16.91, -9.42] | 39.24 | 35.25 | 40.93 | +1.69 [+0.75, +2.45] | -11.81 [-15.37, -8.39] | 12% |
| lee2019 | 1 of 2 runs | 4 | 62.60 | 60.51 | -2.09 [-3.13, -1.09] | 60.54 | 60.32 | 60.93 | +0.39 [-0.09, +0.89] | -1.90 [-2.77, -1.03] | 17% |
| lee2019 | 1 of 2 runs | 8 | 62.60 | 58.09 | -4.51 [-5.91, -3.19] | 58.34 | 57.86 | 58.84 | +0.50 [-0.08, +1.06] | -3.98 [-5.29, -2.71] | 11% |
| lee2019 | 1 of 2 runs | 17 | 62.60 | 55.43 | -7.17 [-9.65, -4.85] | 55.55 | 55.17 | 55.98 | +0.43 [-0.50, +1.35] | -6.85 [-9.00, -4.82] | 6% |
| lee2019 | 1 of 2 runs | 32 | 62.60 | 53.45 | -9.15 [-12.31, -6.16] | 53.48 | 53.38 | 54.37 | +0.89 [-0.11, +1.89] | -8.45 [-11.31, -5.76] | 9% |
| lee2019 | 1 of 2 runs | 46 | 62.60 | 51.77 | -10.83 [-14.18, -7.69] | 51.87 | 51.67 | 51.92 | +0.05 [-0.89, +1.02] | -10.90 [-14.10, -7.86] | 0% |

"Recovered": noise weights 0 and batch norms re-estimated, minus batch norms re-estimated only. "Still lost": the former minus the no-noise models with re-estimated batch norms.

## How much of a spatial filter's output is noise

| dataset | added | after training | at initialization | trained / initial | noise weight rms / electrode weight rms |
|---|---:|---:|---:|---:|---:|
| bnci2a | 1 | 7.9% | 15.2% | 0.52 | 0.60 |
| bnci2a | 2 | 15.1% | 21.1% | 0.71 | 0.63 |
| bnci2a | 4 | 27.8% | 37.2% | 0.75 | 0.69 |
| bnci2a | 6 | 37.6% | 43.1% | 0.87 | 0.71 |
| bnci2a | 10 | 51.5% | 56.1% | 0.92 | 0.73 |
| bnci2a | 16 | 62.2% | 59.7% | 1.04 | 0.76 |
| lee2019 | 4 | 30.6% | 32.0% | 0.96 | 0.84 |
| lee2019 | 8 | 48.0% | 42.8% | 1.12 | 0.88 |
| lee2019 | 17 | 67.1% | 58.9% | 1.14 | 0.89 |
| lee2019 | 32 | 79.7% | 71.6% | 1.11 | 0.92 |
| lee2019 | 46 | 85.0% | 79.3% | 1.07 | 0.94 |

Within participant and condition, correlation of the noise share after training with held-out accuracy (M2): bnci2a r = -0.29 [-0.39, -0.18]; lee2019 r = -0.29 [-0.42, -0.11]

## Noise draws (next day, BCI IV 2a final models, seeds 0-9)

Reference models rescored: 270 of 270 equal their recorded next-day accuracy.

| arm | accuracy % | cost vs. 16 electrodes | 95% interval |
|---|---:|---:|---|
| input-cost | 57.86 | -7.29 | [-9.80, -5.26] |
| overfit-draw-appended | 59.00 | -6.14 | [-7.56, -4.78] |
| draw-1 | 57.72 | -7.43 | [-9.31, -5.32] |
| draw-2 | 58.88 | -6.26 | [-8.73, -4.18] |
| draw-3 | 58.66 | -6.49 | [-8.58, -4.57] |
| draw-4 | 59.43 | -5.71 | [-7.47, -4.05] |
| overfit-draw-in-place | 60.18 | -4.96 | [-6.55, -3.57] |
| input-cost-draw-in-place | 57.63 | -7.51 | [-9.84, -5.39] |

- B1: spread between the six appended draws beyond seed noise: SD 0.98 pp [0.10, 1.40] (seed SD of one draw's mean 1.15); per participant, lowest and highest draw cost: {1: [-8.75, -5.16], 2: [-15.44, -9.15], 3: [-7.51, -4.29], 4: [-6.84, -3.11], 5: [-9.28, -3.08], 6: [-9.16, -7.02], 7: [-11.08, -6.39], 8: [-5.39, -3.17], 9: [-2.42, -1.21]}
- B2: original difference (input-cost appended minus overfit-study in place) -2.32 [-3.82, -1.06]; with the placements swapped -1.37 [-2.54, -0.24]; the input-cost draw moved into the dropped places -0.23 [-0.87, +0.37]; the overfit-study draw appended -1.18 [-2.26, -0.18]

Next day, the same mechanism measurements (mean over participants and seeds, accuracy %):

| arm | as_trained | recalibrated | noise_removed | noise_removed_recalibrated |
|---|---:|---:|---:|---:|
| draw-1 | 57.72 | 58.06 | 56.99 | 59.39 |
| draw-2 | 58.88 | 59.16 | 58.85 | 60.12 |
| draw-3 | 58.66 | 59.14 | 58.49 | 60.03 |
| draw-4 | 59.43 | 59.68 | 58.42 | 60.61 |
| input-cost | 57.86 | 57.94 | 58.13 | 60.15 |
| input-cost-draw-in-place | 57.63 | 57.83 | 57.95 | 59.94 |
| overfit-draw-appended | 59.00 | 59.12 | 58.73 | 60.53 |
| overfit-draw-in-place | 60.18 | 60.53 | 59.16 | 61.13 |
