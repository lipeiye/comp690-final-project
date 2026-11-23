# Pretraining on other people, then fine-tuning

Accuracy in %, differences in points with the 95% participant interval. group-16 is the 16 electrodes with the best mean relevance rank among the pretraining participants, shared by everyone in that run. Lee2019 tests were fixed in the plan (2026-11-22T16:28); BCI IV 2a is exploratory.

Pretraining epochs chosen on the targets' training days (the project's budget rule): bnci2a 125 (smoothed maximum 48.60% at epoch 168); lee2019 33 (smoothed maximum 71.58% at epoch 62)

## Lee2019 tests

- T1, with all 62 electrodes, pretraining and fine-tuning beat training from scratch: +7.65 [+5.39, +9.94] (40/54 better): holds
- T2, after pretraining and fine-tuning the 46 noise inputs cost less than from scratch: +9.91 [+6.72, +13.36] (44/54 better): holds
- T3, after pretraining and fine-tuning, all 62 electrodes beat the shared 16: +4.53 [+2.84, +6.44] (44/54 better): holds

## Next-day accuracy

| dataset | configuration | from scratch | pretrained only | pretrained + fine-tuned |
|---|---|---:|---:|---:|
| bnci2a | full-22 | 65.30 | 45.37 | 56.14 |
| bnci2a | group-16 | 63.88 | 44.84 | 56.57 |
| bnci2a | group-16-plus-6 | 57.76 | 44.45 | 54.11 |
| lee2019 | full-62 | 66.90 | 71.89 | 74.55 |
| lee2019 | group-16 | 65.71 | 67.34 | 70.02 |
| lee2019 | group-16-plus-46 | 53.63 | 66.06 | 67.85 |

## Contrasts, lee2019

| contrast | mean | 95% interval | better |
|---|---:|---|---:|
| scratch: all electrodes minus group-16 | +1.19 | [-0.77, +3.18] | 30/54 |
| scratch: noise cost (group-16 plus noise minus group-16) | -12.08 | [-15.53, -8.85] | 6/54 |
| pretrain: all electrodes minus group-16 | +4.55 | [+2.66, +6.55] | 40/54 |
| pretrain: noise cost (group-16 plus noise minus group-16) | -1.28 | [-2.11, -0.42] | 18/54 |
| finetune: all electrodes minus group-16 | +4.53 | [+2.84, +6.44] | 44/54 |
| finetune: noise cost (group-16 plus noise minus group-16) | -2.17 | [-2.97, -1.40] | 10/54 |
| full: finetune minus scratch | +7.65 | [+5.39, +9.94] | 40/54 |
| full: pretrain only minus scratch | +4.99 | [+2.29, +7.66] | 38/54 |
| full: finetune minus pretrain only | +2.66 | [+1.38, +4.01] | 42/54 |
| group-16: finetune minus scratch | +4.31 | [+2.07, +6.60] | 39/54 |
| group-16: pretrain only minus scratch | +1.63 | [-1.06, +4.23] | 32/54 |
| group-16: finetune minus pretrain only | +2.68 | [+1.33, +4.19] | 35/54 |
| noise: finetune minus scratch | +14.22 | [+10.68, +17.84] | 48/54 |
| noise: pretrain only minus scratch | +12.43 | [+9.06, +15.90] | 47/54 |
| noise: finetune minus pretrain only | +1.79 | [+0.73, +2.86] | 35/54 |
| noise cost after finetune minus from scratch | +9.91 | [+6.72, +13.36] | 44/54 |
| (all minus group-16) after finetune minus from scratch | +3.34 | [+1.29, +5.53] | 27/54 |

## Contrasts, bnci2a

| contrast | mean | 95% interval | better |
|---|---:|---|---:|
| scratch: all electrodes minus group-16 | +1.41 | [-0.33, +3.40] | 5/9 |
| scratch: noise cost (group-16 plus noise minus group-16) | -6.13 | [-7.98, -4.42] | 0/9 |
| pretrain: all electrodes minus group-16 | +0.52 | [-1.20, +2.18] | 5/9 |
| pretrain: noise cost (group-16 plus noise minus group-16) | -0.40 | [-1.44, +0.77] | 3/9 |
| finetune: all electrodes minus group-16 | -0.43 | [-1.59, +0.68] | 4/9 |
| finetune: noise cost (group-16 plus noise minus group-16) | -2.46 | [-3.85, -1.22] | 1/9 |
| full: finetune minus scratch | -9.15 | [-13.97, -5.29] | 0/9 |
| full: pretrain only minus scratch | -19.93 | [-25.02, -15.59] | 0/9 |
| full: finetune minus pretrain only | +10.78 | [+7.23, +14.63] | 9/9 |
| group-16: finetune minus scratch | -7.31 | [-10.14, -4.57] | 0/9 |
| group-16: pretrain only minus scratch | -19.04 | [-23.66, -14.59] | 0/9 |
| group-16: finetune minus pretrain only | +11.73 | [+8.58, +14.98] | 9/9 |
| noise: finetune minus scratch | -3.65 | [-8.42, +1.32] | 2/9 |
| noise: pretrain only minus scratch | -13.31 | [-19.17, -7.91] | 0/9 |
| noise: finetune minus pretrain only | +9.66 | [+6.96, +12.51] | 9/9 |
| noise cost after finetune minus from scratch | +3.66 | [+1.02, +6.58] | 7/9 |
| (all minus group-16) after finetune minus from scratch | -1.84 | [-4.82, +0.74] | 3/9 |
