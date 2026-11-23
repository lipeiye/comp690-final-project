# Microcontroller study (CMSIS-NN on emulated Cortex-M cores)

Tools: arm-none-eabi-gcc (Arm GNU Toolchain 15.3.Rel2 (Build arm-15.181)) 15.3.1 20260627; QEMU emulator version 11.1.3; CMSIS-NN 13c97dbb6f78.
Counts are instructions executed per 2-s window in QEMU (participant 1, seed 0; mean of 8 windows). QEMU models no timing: these are not cycles, time or energy.

Bit-exactness: 136 of 136 INT8 builds gave logits identical to the NumPy integer reference on all 1088 windows; FP32 builds differ from NumPy float by at most 3.1e-06.

## Million instructions per window

| condition | order | precision | m0 | m4 | m4-nodsp | m55 | m55-nomve |
|---|---|---|---:|---:|---:|---:|---:|
| full-22 | original | f32 | 1251.02 | 66.71 | - | 24.17 | 61.01 |
| full-22 | original | int8 | 79.42 | 40.77 | 46.53 | 20.44 | 39.58 |
| full-22 | swapped | f32 | 191.31 | 15.97 | - | 2.34 | 14.60 |
| full-22 | swapped | int8 | 18.45 | 6.92 | 8.89 | 1.37 | 6.43 |
| relevance-16 | original | f32 | 920.26 | 49.14 | - | 17.89 | 44.94 |
| relevance-16 | original | int8 | 58.71 | 30.08 | 34.37 | 15.14 | 29.18 |
| relevance-16 | swapped | f32 | 181.73 | 15.73 | - | 2.30 | 14.41 |
| relevance-16 | swapped | int8 | 18.15 | 6.74 | 8.77 | 1.35 | 6.26 |
| relevance-16-prune50 | original | f32 | 915.43 | 48.99 | - | 17.84 | 44.81 |
| relevance-16-prune50 | original | int8 | 58.46 | 30.00 | 34.27 | 15.12 | 29.11 |
| relevance-16-prune50 | swapped | f32 | 176.74 | 15.59 | - | 2.24 | 14.28 |
| relevance-16-prune50 | swapped | int8 | 17.92 | 6.66 | 8.67 | 1.33 | 6.19 |
| relevance-8 | original | f32 | 480.03 | 25.69 | - | 9.52 | 23.49 |
| relevance-8 | original | int8 | 31.07 | 15.83 | 18.16 | 8.08 | 15.32 |
| relevance-8 | swapped | f32 | 167.96 | 15.40 | - | 2.23 | 14.14 |
| relevance-8 | swapped | int8 | 17.77 | 6.67 | 8.61 | 1.35 | 6.18 |

## Instructions per multiply-accumulate, whole network (participant 1)

Best case if every multiply-accumulate went through the core's widest MAC instruction and nothing else ran: M0 int8 (MULS + ADDS) 2.0; M4 int8 (SMLAD, 2 per instruction) 0.5; M4 float (VFMA.F32) 1.0; M55 int8 (VMLADAV.S8, 16 per instruction) 0.0625; M55 float (VFMA.F32 on 4 lanes) 0.25.

| condition | order | precision | m0 | m4 | m4-nodsp | m55 | m55-nomve |
|---|---|---|---:|---:|---:|---:|---:|
| full-22 | original | f32 | 212.58 | 11.34 | - | 4.11 | 10.37 |
| full-22 | original | int8 | 13.50 | 6.93 | 7.91 | 3.47 | 6.73 |
| full-22 | swapped | f32 | 253.57 | 21.17 | - | 3.10 | 19.35 |
| full-22 | swapped | int8 | 24.45 | 9.18 | 11.78 | 1.82 | 8.52 |
| relevance-16 | original | f32 | 214.12 | 11.43 | - | 4.16 | 10.46 |
| relevance-16 | original | int8 | 13.66 | 7.00 | 8.00 | 3.52 | 6.79 |
| relevance-16 | swapped | f32 | 257.23 | 22.27 | - | 3.25 | 20.39 |
| relevance-16 | swapped | int8 | 25.69 | 9.54 | 12.41 | 1.91 | 8.86 |
| relevance-16-prune50 | original | f32 | 213.82 | 11.44 | - | 4.17 | 10.47 |
| relevance-16-prune50 | original | int8 | 13.65 | 7.01 | 8.00 | 3.53 | 6.80 |
| relevance-16-prune50 | swapped | f32 | 256.19 | 22.59 | - | 3.25 | 20.70 |
| relevance-16-prune50 | swapped | int8 | 25.98 | 9.66 | 12.57 | 1.93 | 8.97 |
| relevance-8 | original | f32 | 220.02 | 11.77 | - | 4.36 | 10.77 |
| relevance-8 | original | int8 | 14.24 | 7.25 | 8.32 | 3.70 | 7.02 |
| relevance-8 | swapped | f32 | 261.42 | 23.97 | - | 3.47 | 22.01 |
| relevance-8 | swapped | int8 | 27.66 | 10.38 | 13.40 | 2.10 | 9.63 |

## FP32 / INT8 instruction ratio

| core | full-22 original | full-22 swapped | frozen original | frozen swapped |
|---|---:|---:|---:|---:|
| m0 | 15.75 | 10.37 | 15.66 | 9.86 |
| m4 | 1.64 | 2.31 | 1.63 | 2.34 |
| m55 | 1.18 | 1.71 | 1.18 | 1.69 |
| m55-nomve | 1.54 | 2.27 | 1.54 | 2.31 |

## Per layer (frozen model, instructions per multiply-accumulate)

| core | order | precision | l1 | l2 | elu1 | pool1 | conv_d | conv_p | elu2 | pool2 | fc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m0 | original | f32 | 850711k (207.3) | 31098k (242.5) | 19523k | 1079k | 7439k (230.6) | 3398k (210.7) | 1978k | 107k | 97k (201.3) |
| m0 | original | int8 | 48929k (11.9) | 7617k (59.4) | 64k | 404k | 1214k (37.6) | 190k (11.8) | 8k | 31k | 3k (7.3) |
| m0 | swapped | f32 | 27126k (211.9) | 115995k (226.1) | 19523k | 1081k | 7439k (230.6) | 3398k (210.7) | 1978k | 107k | 97k (201.3) |
| m0 | swapped | int8 | 1495k (11.7) | 14513k (28.3) | 64k | 404k | 1214k (37.6) | 190k (11.8) | 8k | 31k | 3k (7.3) |
| m4 | original | f32 | 43922k (10.7) | 3506k (27.3) | 419k | 115k | 870k (27.0) | 102k (6.3) | 44k | 10k | 3k (5.4) |
| m4 | original | int8 | 25176k (6.1) | 4185k (32.6) | 48k | 86k | 420k (13.0) | 67k (4.2) | 6k | 9k | 1k (2.6) |
| m4 | swapped | f32 | 800k (6.2) | 13225k (25.8) | 419k | 115k | 870k (27.0) | 102k (6.3) | 44k | 10k | 3k (5.4) |
| m4 | swapped | int8 | 528k (4.1) | 5495k (10.7) | 48k | 86k | 420k (13.0) | 67k (4.2) | 6k | 9k | 1k (2.6) |
| m4-nodsp | original | int8 | 29085k (7.1) | 4209k (32.8) | 48k | 170k | 655k (20.3) | 79k (4.9) | 6k | 15k | 2k (4.6) |
| m4-nodsp | swapped | int8 | 621k (4.9) | 7073k (13.8) | 48k | 170k | 655k (20.3) | 79k (4.9) | 6k | 15k | 2k (4.6) |
| m55 | original | f32 | 13775k (3.4) | 3447k (26.9) | 416k | 29k | 104k (3.2) | 21k (1.3) | 44k | 3k | 1k (1.5) |
| m55 | original | int8 | 10540k (2.6) | 4413k (34.4) | 27k | 31k | 86k (2.7) | 17k (1.1) | 3k | 5k | 0k (0.8) |
| m55 | swapped | f32 | 152k (1.2) | 1475k (2.9) | 416k | 29k | 104k (3.2) | 21k (1.3) | 44k | 3k | 1k (1.5) |
| m55 | swapped | int8 | 127k (1.0) | 1033k (2.0) | 27k | 31k | 86k (2.7) | 17k (1.1) | 3k | 5k | 0k (0.8) |
| m55-nomve | original | f32 | 39912k (9.7) | 3462k (27.0) | 416k | 89k | 795k (24.6) | 85k (5.3) | 44k | 8k | 2k (4.4) |
| m55-nomve | original | int8 | 24131k (5.9) | 4387k (34.2) | 40k | 80k | 394k (12.2) | 63k (3.9) | 5k | 9k | 1k (2.6) |
| m55-nomve | swapped | f32 | 667k (5.2) | 12178k (23.7) | 416k | 89k | 795k (24.6) | 85k (5.3) | 44k | 8k | 2k (4.4) |
| m55-nomve | swapped | int8 | 491k (3.8) | 5105k (10.0) | 40k | 80k | 394k (12.2) | 63k (3.9) | 5k | 9k | 1k (2.6) |

## Memory (participant 1): flash = code + constants; RAM = data + bss + measured stack

| condition | order | precision | core | flash bytes | RAM bytes | scratch | stack |
|---|---|---|---|---:|---:|---:|---:|
| full-22 | original | f32 | m0 | 34,000 | 401,592 | 4,096 | 680 |
| full-22 | original | f32 | m4 | 26,940 | 401,416 | 4,096 | 504 |
| full-22 | original | f32 | m55 | 36,712 | 401,472 | 4,096 | 560 |
| full-22 | original | f32 | m55-nomve | 27,776 | 401,400 | 4,096 | 488 |
| full-22 | original | int8 | m0 | 20,140 | 100,136 | 256 | 656 |
| full-22 | original | int8 | m4 | 24,888 | 100,432 | 512 | 696 |
| full-22 | original | int8 | m4-nodsp | 19,104 | 100,132 | 256 | 652 |
| full-22 | original | int8 | m55 | 20,392 | 108,036 | 7,936 | 876 |
| full-22 | original | int8 | m55-nomve | 25,236 | 100,432 | 512 | 696 |
| full-22 | swapped | f32 | m0 | 36,080 | 93,176 | 16,384 | 680 |
| full-22 | swapped | f32 | m4 | 29,020 | 92,920 | 16,384 | 424 |
| full-22 | swapped | f32 | m55 | 38,792 | 93,056 | 16,384 | 560 |
| full-22 | swapped | f32 | m55-nomve | 29,856 | 92,936 | 16,384 | 440 |
| full-22 | swapped | int8 | m0 | 20,732 | 19,608 | 0 | 576 |
| full-22 | swapped | int8 | m4 | 25,496 | 21,492 | 2,048 | 396 |
| full-22 | swapped | int8 | m4-nodsp | 19,700 | 19,392 | 0 | 360 |
| full-22 | swapped | int8 | m55 | 21,000 | 51,456 | 31,744 | 664 |
| full-22 | swapped | int8 | m55-nomve | 25,844 | 21,492 | 2,048 | 396 |
| relevance-16 | original | f32 | m0 | 33,616 | 293,464 | 4,096 | 680 |
| relevance-16 | original | f32 | m4 | 26,556 | 293,288 | 4,096 | 504 |
| relevance-16 | original | f32 | m55 | 36,328 | 293,344 | 4,096 | 560 |
| relevance-16 | original | f32 | m55-nomve | 27,392 | 293,272 | 4,096 | 488 |
| relevance-16 | original | int8 | m0 | 20,044 | 73,104 | 256 | 656 |
| relevance-16 | original | int8 | m4 | 24,792 | 73,400 | 512 | 696 |
| relevance-16 | original | int8 | m4-nodsp | 19,008 | 73,100 | 256 | 652 |
| relevance-16 | original | int8 | m55 | 20,296 | 81,004 | 7,936 | 876 |
| relevance-16 | original | int8 | m55-nomve | 25,140 | 73,400 | 512 | 696 |
| relevance-16 | swapped | f32 | m0 | 35,696 | 81,240 | 16,384 | 680 |
| relevance-16 | swapped | f32 | m4 | 28,636 | 80,984 | 16,384 | 424 |
| relevance-16 | swapped | f32 | m55 | 38,408 | 81,120 | 16,384 | 560 |
| relevance-16 | swapped | f32 | m55-nomve | 29,472 | 81,000 | 16,384 | 440 |
| relevance-16 | swapped | int8 | m0 | 20,636 | 16,624 | 0 | 576 |
| relevance-16 | swapped | int8 | m4 | 25,400 | 18,508 | 2,048 | 396 |
| relevance-16 | swapped | int8 | m4-nodsp | 19,604 | 16,408 | 0 | 360 |
| relevance-16 | swapped | int8 | m55 | 20,904 | 48,472 | 31,744 | 664 |
| relevance-16 | swapped | int8 | m55-nomve | 25,748 | 18,508 | 2,048 | 396 |
| relevance-16-prune50 | original | f32 | m0 | 31,152 | 293,464 | 4,096 | 680 |
| relevance-16-prune50 | original | f32 | m4 | 24,092 | 293,288 | 4,096 | 504 |
| relevance-16-prune50 | original | f32 | m55 | 33,864 | 293,344 | 4,096 | 560 |
| relevance-16-prune50 | original | f32 | m55-nomve | 24,928 | 293,272 | 4,096 | 488 |
| relevance-16-prune50 | original | int8 | m0 | 19,340 | 73,104 | 256 | 656 |
| relevance-16-prune50 | original | int8 | m4 | 24,088 | 73,400 | 512 | 696 |
| relevance-16-prune50 | original | int8 | m4-nodsp | 18,304 | 73,100 | 256 | 652 |
| relevance-16-prune50 | original | int8 | m55 | 19,592 | 81,004 | 7,936 | 876 |
| relevance-16-prune50 | original | int8 | m55-nomve | 24,436 | 73,400 | 512 | 696 |
| relevance-16-prune50 | swapped | f32 | m0 | 33,232 | 81,240 | 16,384 | 680 |
| relevance-16-prune50 | swapped | f32 | m4 | 26,172 | 80,984 | 16,384 | 424 |
| relevance-16-prune50 | swapped | f32 | m55 | 35,944 | 81,120 | 16,384 | 560 |
| relevance-16-prune50 | swapped | f32 | m55-nomve | 27,008 | 81,000 | 16,384 | 440 |
| relevance-16-prune50 | swapped | int8 | m0 | 19,932 | 16,624 | 0 | 576 |
| relevance-16-prune50 | swapped | int8 | m4 | 24,696 | 18,508 | 2,048 | 396 |
| relevance-16-prune50 | swapped | int8 | m4-nodsp | 18,900 | 16,408 | 0 | 360 |
| relevance-16-prune50 | swapped | int8 | m55 | 20,200 | 48,472 | 31,744 | 664 |
| relevance-16-prune50 | swapped | int8 | m55-nomve | 25,044 | 18,508 | 2,048 | 396 |
| relevance-8 | original | f32 | m0 | 33,104 | 165,208 | 4,096 | 680 |
| relevance-8 | original | f32 | m4 | 26,044 | 165,032 | 4,096 | 504 |
| relevance-8 | original | f32 | m55 | 35,816 | 165,088 | 4,096 | 560 |
| relevance-8 | original | f32 | m55-nomve | 26,880 | 165,016 | 4,096 | 488 |
| relevance-8 | original | int8 | m0 | 19,916 | 41,040 | 256 | 656 |
| relevance-8 | original | int8 | m4 | 24,664 | 41,336 | 512 | 696 |
| relevance-8 | original | int8 | m4-nodsp | 18,880 | 41,036 | 256 | 652 |
| relevance-8 | original | int8 | m55 | 20,168 | 48,940 | 7,936 | 876 |
| relevance-8 | original | int8 | m55-nomve | 25,012 | 41,336 | 512 | 696 |
| relevance-8 | swapped | f32 | m0 | 35,184 | 81,240 | 16,384 | 680 |
| relevance-8 | swapped | f32 | m4 | 28,124 | 80,984 | 16,384 | 424 |
| relevance-8 | swapped | f32 | m55 | 37,896 | 81,120 | 16,384 | 560 |
| relevance-8 | swapped | f32 | m55-nomve | 28,960 | 81,000 | 16,384 | 440 |
| relevance-8 | swapped | int8 | m0 | 20,508 | 16,624 | 0 | 576 |
| relevance-8 | swapped | int8 | m4 | 25,272 | 18,508 | 2,048 | 396 |
| relevance-8 | swapped | int8 | m4-nodsp | 19,476 | 16,408 | 0 | 360 |
| relevance-8 | swapped | int8 | m55 | 20,776 | 48,472 | 31,744 | 664 |
| relevance-8 | swapped | int8 | m55-nomve | 25,620 | 18,508 | 2,048 | 396 |

## Session-2 accuracy (artifact-free trials; 9 participants x 3 seeds)

| condition | order | FP32 % | INT8 CMSIS-NN % | INT8 ONNX % | INT8 minus FP32 (pp) | 95% interval | worst participant | losing > 2 pp |
|---|---|---:|---:|---:|---:|---|---:|---:|
| full-22 | original | 65.04 | 65.34 | 65.30 | +0.30 | [-0.16, +0.76] | -1.09 | 0 |
| full-22 | swapped | 65.04 | 65.20 | 65.30 | +0.16 | [-0.26, +0.56] | -1.09 | 0 |
| relevance-16 | original | 66.21 | 65.96 | 66.17 | -0.24 | [-0.66, +0.16] | -1.47 | 0 |
| relevance-16 | swapped | 66.21 | 65.89 | 66.17 | -0.31 | [-0.67, +0.03] | -1.02 | 0 |
| relevance-16-prune50 | original | 64.99 | 64.91 | 65.01 | -0.08 | [-0.47, +0.36] | -0.95 | 0 |
| relevance-16-prune50 | swapped | 64.99 | 64.94 | 65.01 | -0.05 | [-0.46, +0.38] | -1.07 | 0 |
| relevance-8 | original | 59.92 | 59.59 | 59.61 | -0.32 | [-0.78, +0.14] | -1.46 | 0 |
| relevance-8 | swapped | 59.92 | 59.64 | 59.61 | -0.28 | [-0.71, +0.19] | -1.17 | 0 |
