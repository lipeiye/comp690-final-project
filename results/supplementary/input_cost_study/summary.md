# The cost of inputs that carry nothing

Relevance-16 plus k inputs of band-limited noise; accuracy in %, costs in percentage points against the same 16 electrodes alone, with the 95% participant interval. Lee2019 tests were fixed in the plan; BCI IV 2a is exploratory.

## Lee2019 tests (54 people, seeds 0-2)

- C1, the cost grows with the number of such inputs (8 added minus 46 added, both runs): +9.80 [+6.64, +13.13]: holds
- C2, the cost is larger with fewer training trials ((17 added minus none) with one run minus the same with both): +2.49 [+0.88, +4.12]: does not hold. Share of the accuracy above chance that 17 added inputs remove: 45% with one run, 45% with both.

## Cost per added input (least squares through k = 0)

| dataset | training data | 16 electrodes alone % | pp per added input | 95% interval | share of the margin above chance per input |
|---|---|---:|---:|---|---:|
| bnci2a | 6 of 6 runs | 65.15 | -0.89 | [-1.13, -0.67] | 2.2% |
| bnci2a | 4 of 6 runs | 60.66 | -0.93 | [-1.16, -0.72] | 2.6% |
| bnci2a | 2 of 6 runs | 50.14 | -0.84 | [-1.04, -0.63] | 3.3% |
| lee2019 | 2 of 2 runs | 67.02 | -0.32 | [-0.40, -0.24] | 1.9% |
| lee2019 | 1 of 2 runs | 61.59 | -0.22 | [-0.29, -0.16] | 1.9% |

## Every point

| dataset | training data | added | accuracy % | next-day cost | 95% interval | seed error | worse | held-out runs, same day | training-set change | share of margin |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| bnci2a | 6 of 6 runs | 1 | 63.40 | -1.74 | [-2.42, -1.09] | 0.51 | 9/9 | - | +0.01 | 4% |
| bnci2a | 6 of 6 runs | 2 | 63.07 | -2.08 | [-3.71, -0.77] | 0.45 | 8/9 | - | +0.81 | 5% |
| bnci2a | 6 of 6 runs | 4 | 60.72 | -4.43 | [-5.99, -3.07] | 0.53 | 9/9 | - | +0.82 | 11% |
| bnci2a | 6 of 6 runs | 6 | 57.86 | -7.29 | [-9.80, -5.26] | 0.49 | 9/9 | - | +1.43 | 18% |
| bnci2a | 6 of 6 runs | 10 | 55.09 | -10.05 | [-13.05, -7.52] | 0.81 | 9/9 | - | +2.76 | 25% |
| bnci2a | 6 of 6 runs | 16 | 52.63 | -12.51 | [-15.53, -9.48] | 0.56 | 9/9 | - | +4.66 | 31% |
| bnci2a | 4 of 6 runs | 1 | 59.71 | -0.95 | [-2.11, +0.04] | 0.41 | 6/9 | -0.96 [-2.63, +0.48] | +0.24 | 3% |
| bnci2a | 4 of 6 runs | 2 | 58.74 | -1.92 | [-3.39, -0.57] | 0.79 | 7/9 | -1.62 [-2.99, -0.30] | +0.91 | 5% |
| bnci2a | 4 of 6 runs | 4 | 55.52 | -5.14 | [-6.61, -3.81] | 0.36 | 9/9 | -5.22 [-7.37, -3.04] | +1.49 | 14% |
| bnci2a | 4 of 6 runs | 6 | 53.35 | -7.31 | [-9.34, -5.25] | 0.90 | 9/9 | -7.36 [-9.53, -5.31] | +2.01 | 20% |
| bnci2a | 4 of 6 runs | 10 | 50.78 | -9.88 | [-12.38, -7.53] | 1.29 | 9/9 | -10.59 [-13.67, -7.99] | +2.94 | 28% |
| bnci2a | 4 of 6 runs | 16 | 47.22 | -13.44 | [-16.88, -10.61] | 0.42 | 9/9 | -14.63 [-18.62, -11.58] | +3.81 | 38% |
| bnci2a | 2 of 6 runs | 1 | 49.57 | -0.57 | [-1.46, +0.51] | 0.38 | 7/9 | -0.82 [-2.03, +0.47] | +0.52 | 2% |
| bnci2a | 2 of 6 runs | 2 | 48.46 | -1.68 | [-2.35, -0.96] | 0.31 | 9/9 | -2.39 [-3.28, -1.57] | +0.95 | 7% |
| bnci2a | 2 of 6 runs | 4 | 45.89 | -4.25 | [-5.61, -3.02] | 0.45 | 9/9 | -5.02 [-6.80, -3.22] | +1.64 | 17% |
| bnci2a | 2 of 6 runs | 6 | 44.25 | -5.88 | [-6.87, -4.74] | 0.50 | 9/9 | -5.85 [-7.45, -4.20] | +2.00 | 23% |
| bnci2a | 2 of 6 runs | 10 | 40.76 | -9.37 | [-12.01, -6.84] | 1.01 | 9/9 | -10.19 [-13.52, -7.17] | +2.34 | 37% |
| bnci2a | 2 of 6 runs | 16 | 37.92 | -12.21 | [-15.55, -8.87] | 0.37 | 9/9 | -13.19 [-16.91, -9.42] | +2.59 | 49% |
| lee2019 | 2 of 2 runs | 4 | 64.85 | -2.17 | [-3.18, -1.22] | 0.34 | 33/54 | - | +1.74 | 13% |
| lee2019 | 2 of 2 runs | 8 | 63.49 | -3.53 | [-4.81, -2.34] | 0.66 | 41/54 | - | +2.88 | 21% |
| lee2019 | 2 of 2 runs | 17 | 59.36 | -7.66 | [-9.90, -5.57] | 1.50 | 48/54 | - | +3.74 | 45% |
| lee2019 | 2 of 2 runs | 32 | 56.38 | -10.65 | [-13.32, -8.10] | 0.54 | 51/54 | - | +3.89 | 63% |
| lee2019 | 2 of 2 runs | 46 | 53.69 | -13.33 | [-16.84, -9.99] | 0.65 | 52/54 | - | +3.90 | 78% |
| lee2019 | 1 of 2 runs | 4 | 60.92 | -0.68 | [-1.41, -0.00] | 0.29 | 28/54 | -2.09 [-3.13, -1.09] | +0.40 | 6% |
| lee2019 | 1 of 2 runs | 8 | 58.87 | -2.72 | [-3.62, -1.88] | 0.83 | 43/54 | -4.51 [-5.91, -3.19] | +0.48 | 23% |
| lee2019 | 1 of 2 runs | 17 | 56.42 | -5.17 | [-6.94, -3.53] | 0.80 | 44/54 | -7.17 [-9.65, -4.85] | +0.48 | 45% |
| lee2019 | 1 of 2 runs | 32 | 54.16 | -7.44 | [-9.68, -5.36] | 0.64 | 49/54 | -9.15 [-12.31, -6.16] | +0.48 | 64% |
| lee2019 | 1 of 2 runs | 46 | 52.40 | -9.20 | [-11.90, -6.73] | 0.43 | 48/54 | -10.83 [-14.18, -7.69] | +0.48 | 79% |

## BCI IV 2a: the cost of 16 added inputs with less training data, minus the same with all

| training data | mean | 95% interval |
|---|---:|---|
| 4 of 6 runs | -0.92 | [-3.30, +1.57] |
| 2 of 6 runs | +0.30 | [-4.79, +5.23] |

## BCI IV 2a: two noise draws of the same condition

Final models, the same 16 electrodes plus 6 noise inputs, seeds 0-9. The overfit study put its noise in the six dropped places; this study appends it. EEGNet treats input positions alike (its spatial filters span every input), so the two differ only in the noise drawn, one draw per participant shared by all seeds.

- this study's draw: -7.29 [-9.80, -5.26]
- the overfit study's draw: -4.96 [-6.55, -3.57]
- difference: -2.32 [-3.82, -1.06]; per participant {1: -4.45, 2: -7.21, 3: -2.27, 4: -0.83, 5: -1.52, 6: 0.19, 7: -3.07, 8: -1.37, 9: -0.38}
