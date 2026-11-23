# The int8 FIR kernel for the temporal layer on Cortex-M

136 of 136 builds give logits bit-identical to the NumPy integer reference (136 identical to the CMSIS-NN-only build). The layers both builds share differ by at most 12 instructions per window. Instruction counts from QEMU, mean per 2-s window; not cycles, time or energy.

## Participant 1, seed 0

| condition | order | core | temporal layer, CMSIS-NN | with the FIR kernel | per MAC: CMSIS-NN / FIR / best case | whole network, CMSIS-NN | with the FIR kernel | fewer | flash bytes (CMSIS-NN / FIR) | RAM bytes (CMSIS-NN / FIR) |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|
| full-22 | original | m0 | 67.30 M | 39.26 M | 11.93 / 6.96 / 2 | 79.42 M | 51.39 M | 1.55x | 20,140 / 21,968 | 100,136 / 101,616 |
| full-22 | original | m4 | 34.62 M | 12.99 M | 6.13 / 2.30 / 0.5 | 40.77 M | 19.14 M | 2.13x | 24,888 / 26,736 | 100,432 / 101,932 |
| full-22 | original | m4-nodsp | 39.99 M | 17.68 M | 7.09 / 3.13 / 2 | 46.53 M | 24.21 M | 1.92x | 19,104 / 20,688 | 100,132 / 101,368 |
| full-22 | original | m55 | 14.49 M | 3.94 M | 2.57 / 0.70 / 0.0625 | 20.44 M | 9.88 M | 2.07x | 20,392 / 21,888 | 108,036 / 109,608 |
| full-22 | original | m55-nomve | 33.18 M | 11.53 M | 5.88 / 2.04 / 0.5 | 39.58 M | 17.93 M | 2.21x | 25,236 / 27,056 | 100,432 / 101,932 |
| full-22 | swapped | m0 | 14.51 M | 3.68 M | 28.29 / 7.18 / 2 | 18.45 M | 7.62 M | 2.42x | 20,732 / 23,632 | 19,608 / 21,432 |
| full-22 | swapped | m4 | 5.49 M | 1.27 M | 10.71 / 2.47 / 0.5 | 6.92 M | 2.70 M | 2.57x | 25,496 / 28,432 | 21,492 / 21,756 |
| full-22 | swapped | m4-nodsp | 7.07 M | 1.70 M | 13.79 / 3.30 / 2 | 8.89 M | 3.51 M | 2.53x | 19,700 / 22,384 | 19,392 / 21,192 |
| full-22 | swapped | m55 | 1.03 M | 0.45 M | 2.01 / 0.87 / 0.0625 | 1.37 M | 0.78 M | 1.75x | 21,000 / 23,568 | 51,456 / 29,424 |
| full-22 | swapped | m55-nomve | 5.11 M | 1.14 M | 9.95 / 2.21 / 0.5 | 6.43 M | 2.46 M | 2.61x | 25,844 / 28,752 | 21,492 / 21,756 |
| relevance-16 | original | m0 | 48.95 M | 28.55 M | 11.93 / 6.96 / 2 | 58.71 M | 38.31 M | 1.53x | 20,044 / 21,872 | 73,104 / 74,584 |
| relevance-16 | original | m4 | 25.18 M | 9.45 M | 6.13 / 2.30 / 0.5 | 30.08 M | 14.35 M | 2.10x | 24,792 / 26,640 | 73,400 / 74,900 |
| relevance-16 | original | m4-nodsp | 29.08 M | 12.86 M | 7.09 / 3.13 / 2 | 34.37 M | 18.14 M | 1.89x | 19,008 / 20,592 | 73,100 / 74,336 |
| relevance-16 | original | m55 | 10.54 M | 2.86 M | 2.57 / 0.70 / 0.0625 | 15.14 M | 7.46 M | 2.03x | 20,296 / 21,776 | 81,004 / 82,576 |
| relevance-16 | original | m55-nomve | 24.13 M | 8.38 M | 5.88 / 2.04 / 0.5 | 29.18 M | 13.44 M | 2.17x | 25,140 / 26,960 | 73,400 / 74,900 |
| relevance-16 | swapped | m0 | 14.51 M | 3.68 M | 28.28 / 7.17 / 2 | 18.15 M | 7.32 M | 2.48x | 20,636 / 23,536 | 16,624 / 18,448 |
| relevance-16 | swapped | m4 | 5.49 M | 1.27 M | 10.71 / 2.47 / 0.5 | 6.74 M | 2.52 M | 2.68x | 25,400 / 28,336 | 18,508 / 18,772 |
| relevance-16 | swapped | m4-nodsp | 7.07 M | 1.70 M | 13.79 / 3.30 / 2 | 8.77 M | 3.39 M | 2.58x | 19,604 / 22,272 | 16,408 / 18,208 |
| relevance-16 | swapped | m55 | 1.03 M | 0.45 M | 2.01 / 0.87 / 0.0625 | 1.35 M | 0.76 M | 1.77x | 20,904 / 23,472 | 48,472 / 26,440 |
| relevance-16 | swapped | m55-nomve | 5.11 M | 1.14 M | 9.95 / 2.21 / 0.5 | 6.26 M | 2.29 M | 2.73x | 25,748 / 28,656 | 18,508 / 18,772 |
| relevance-16-prune50 | original | m0 | 48.93 M | 28.54 M | 11.92 / 6.95 / 2 | 58.46 M | 38.07 M | 1.54x | 19,340 / 21,168 | 73,104 / 74,584 |
| relevance-16-prune50 | original | m4 | 25.18 M | 9.45 M | 6.13 / 2.30 / 0.5 | 30.00 M | 14.27 M | 2.10x | 24,088 / 25,936 | 73,400 / 74,900 |
| relevance-16-prune50 | original | m4-nodsp | 29.08 M | 12.86 M | 7.09 / 3.13 / 2 | 34.27 M | 18.04 M | 1.90x | 18,304 / 19,888 | 73,100 / 74,336 |
| relevance-16-prune50 | original | m55 | 10.54 M | 2.86 M | 2.57 / 0.70 / 0.0625 | 15.12 M | 7.45 M | 2.03x | 19,592 / 21,088 | 81,004 / 82,576 |
| relevance-16-prune50 | original | m55-nomve | 24.13 M | 8.38 M | 5.88 / 2.04 / 0.5 | 29.11 M | 13.36 M | 2.18x | 24,436 / 26,256 | 73,400 / 74,900 |
| relevance-16-prune50 | swapped | m0 | 14.51 M | 3.68 M | 28.29 / 7.18 / 2 | 17.92 M | 7.09 M | 2.53x | 19,932 / 22,832 | 16,624 / 18,448 |
| relevance-16-prune50 | swapped | m4 | 5.49 M | 1.27 M | 10.71 / 2.47 / 0.5 | 6.66 M | 2.44 M | 2.73x | 24,696 / 27,632 | 18,508 / 18,772 |
| relevance-16-prune50 | swapped | m4-nodsp | 7.07 M | 1.70 M | 13.79 / 3.30 / 2 | 8.67 M | 3.29 M | 2.63x | 18,900 / 21,568 | 16,408 / 18,208 |
| relevance-16-prune50 | swapped | m55 | 1.03 M | 0.45 M | 2.01 / 0.87 / 0.0625 | 1.33 M | 0.74 M | 1.79x | 20,200 / 22,768 | 48,472 / 26,440 |
| relevance-16-prune50 | swapped | m55-nomve | 5.11 M | 1.14 M | 9.95 / 2.21 / 0.5 | 6.19 M | 2.22 M | 2.79x | 25,044 / 27,952 | 18,508 / 18,772 |
| relevance-8 | original | m0 | 24.45 M | 14.26 M | 11.92 / 6.95 / 2 | 31.07 M | 20.88 M | 1.49x | 19,916 / 21,760 | 41,040 / 42,520 |
| relevance-8 | original | m4 | 12.59 M | 4.72 M | 6.13 / 2.30 / 0.5 | 15.83 M | 7.96 M | 1.99x | 24,664 / 26,512 | 41,336 / 42,836 |
| relevance-8 | original | m4-nodsp | 14.54 M | 6.43 M | 7.09 / 3.13 / 2 | 18.16 M | 10.05 M | 1.81x | 18,880 / 20,464 | 41,036 / 42,272 |
| relevance-8 | original | m55 | 5.27 M | 1.43 M | 2.57 / 0.70 / 0.0625 | 8.08 M | 4.24 M | 1.91x | 20,168 / 21,648 | 48,940 / 50,504 |
| relevance-8 | original | m55-nomve | 12.07 M | 4.19 M | 5.88 / 2.04 / 0.5 | 15.32 M | 7.45 M | 2.06x | 25,012 / 26,848 | 41,336 / 42,836 |
| relevance-8 | swapped | m0 | 14.51 M | 3.68 M | 28.29 / 7.18 / 2 | 17.77 M | 6.94 M | 2.56x | 20,508 / 23,408 | 16,624 / 18,448 |
| relevance-8 | swapped | m4 | 5.49 M | 1.27 M | 10.71 / 2.47 / 0.5 | 6.67 M | 2.44 M | 2.73x | 25,272 / 28,208 | 18,508 / 18,772 |
| relevance-8 | swapped | m4-nodsp | 7.07 M | 1.70 M | 13.79 / 3.30 / 2 | 8.61 M | 3.23 M | 2.66x | 19,476 / 22,160 | 16,408 / 18,208 |
| relevance-8 | swapped | m55 | 1.03 M | 0.45 M | 2.01 / 0.87 / 0.0625 | 1.35 M | 0.76 M | 1.77x | 20,776 / 23,344 | 48,472 / 26,440 |
| relevance-8 | swapped | m55-nomve | 5.11 M | 1.14 M | 9.95 / 2.21 / 0.5 | 6.18 M | 2.22 M | 2.79x | 25,620 / 28,528 | 18,508 / 18,772 |

## Participants 2-9 (mean per window)

| condition | order | core | CMSIS-NN | with the FIR kernel | fewer |
|---|---|---|---:|---:|---:|
| full-22 | original | m0 | 79.40 M | 51.36 M | 1.55x |
| full-22 | original | m4 | 40.77 M | 19.14 M | 2.13x |
| full-22 | original | m55 | 20.44 M | 9.88 M | 2.07x |
| full-22 | swapped | m0 | 18.44 M | 7.61 M | 2.42x |
| full-22 | swapped | m4 | 6.92 M | 2.70 M | 2.57x |
| full-22 | swapped | m55 | 1.37 M | 0.78 M | 1.75x |
| relevance-16-prune50 | original | m0 | 58.47 M | 38.08 M | 1.54x |
| relevance-16-prune50 | original | m4 | 30.00 M | 14.27 M | 2.10x |
| relevance-16-prune50 | original | m55 | 15.12 M | 7.45 M | 2.03x |
| relevance-16-prune50 | swapped | m0 | 17.92 M | 7.09 M | 2.53x |
| relevance-16-prune50 | swapped | m4 | 6.66 M | 2.44 M | 2.73x |
| relevance-16-prune50 | swapped | m55 | 1.33 M | 0.74 M | 1.79x |
