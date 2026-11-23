# The missing INT8 kernels (ONNX Runtime custom operators)

- grouped spatial convolution, original order (kernels/ort_grouped_conv.c): 459 of 459 archived models give bit-identical logits on all 132,192 session-2 windows (largest difference 0.0); 132,192 predictions equal the archived ones.
- spatial convolution after the swap (kernels/ort_spatial_first_conv.c): 459 of 459 archived models give bit-identical logits on all 132,192 session-2 windows (largest difference 0.0); 132,192 predictions equal the archived ones.

Latency: ONNX Runtime 1.31.1, arm64, one thread, batch 1, 15 interleaved rounds of 500 runs, every variant in one set; load average 4.3 before and 3.4 after. Microseconds per 2-s window.

| condition | variant | microseconds | relative to FP32 | parameter bytes |
|---|---|---:|---:|---:|
| full-22 | FP32 | 273 | 1.00x | 9,576 |
| full-22 | INT8 (ONNX Runtime) | 357 | 1.31x | 3,260 |
| full-22 | INT8 + new kernel | 180 | 0.66x | 2,908 |
| full-22 | FP32, layers swapped | 94 | 0.34x | 11,592 |
| full-22 | INT8, layers swapped | 66 | 0.24x | 3,716 |
| full-22 | INT8, layers swapped + new kernel | 47 | 0.17x | 3,428 |
| relevance-16 | FP32 | 201 | 1.00x | 9,192 |
| relevance-16 | INT8 (ONNX Runtime) | 259 | 1.29x | 3,164 |
| relevance-16 | INT8 + new kernel | 138 | 0.68x | 2,812 |
| relevance-16 | FP32, layers swapped | 92 | 0.46x | 11,208 |
| relevance-16 | INT8, layers swapped | 58 | 0.29x | 3,620 |
| relevance-16 | INT8, layers swapped + new kernel | 45 | 0.22x | 3,332 |
| relevance-8 | FP32 | 114 | 1.00x | 8,680 |
| relevance-8 | INT8 (ONNX Runtime) | 148 | 1.29x | 3,036 |
| relevance-8 | INT8 + new kernel | 80 | 0.70x | 2,684 |
| relevance-8 | FP32, layers swapped | 90 | 0.78x | 10,696 |
| relevance-8 | INT8, layers swapped | 49 | 0.43x | 3,492 |
| relevance-8 | INT8, layers swapped + new kernel | 42 | 0.37x | 3,204 |
| relevance-16-prune50 | FP32 | 203 | 1.00x | 6,728 |
| relevance-16-prune50 | INT8 (ONNX Runtime) | 258 | 1.27x | 2,420 |
| relevance-16-prune50 | INT8 + new kernel | 137 | 0.67x | 2,132 |
| relevance-16-prune50 | FP32, layers swapped | 90 | 0.44x | 8,744 |
| relevance-16-prune50 | INT8, layers swapped | 56 | 0.28x | 2,876 |
| relevance-16-prune50 | INT8, layers swapped + new kernel | 44 | 0.22x | 2,652 |

## Where the time goes (ONNX Runtime profiler, microseconds per window)

| condition | variant | temporal | spatial | other | total |
|---|---|---:|---:|---:|---:|
| full-22 | FP32 | 241.0 | 13.0 | 24.2 | 278.2 |
| full-22 | INT8 (ONNX Runtime) | 152.8 | 189.3 | 35.5 | 377.7 |
| full-22 | INT8 + new kernel | 153.6 | 9.9 | 34.9 | 198.5 |
| full-22 | FP32, layers swapped | 66.9 | 7.0 | 24.7 | 98.5 |
| full-22 | INT8, layers swapped | 19.0 | 25.7 | 35.0 | 79.7 |
| full-22 | INT8, layers swapped + new kernel | 18.7 | 7.1 | 34.3 | 60.1 |
| relevance-16 | FP32 | 173.6 | 9.2 | 24.1 | 206.9 |
| relevance-16 | INT8 (ONNX Runtime) | 110.8 | 129.0 | 35.0 | 274.8 |
| relevance-16 | INT8 + new kernel | 110.7 | 8.0 | 33.8 | 152.5 |
| relevance-16 | FP32, layers swapped | 64.9 | 5.0 | 23.7 | 93.7 |
| relevance-16 | INT8, layers swapped | 18.6 | 18.0 | 34.5 | 71.1 |
| relevance-16 | INT8, layers swapped + new kernel | 18.6 | 6.0 | 34.3 | 59.0 |
| relevance-8 | FP32 | 87.2 | 5.2 | 23.3 | 115.7 |
| relevance-8 | INT8 (ONNX Runtime) | 56.3 | 72.4 | 34.0 | 162.7 |
| relevance-8 | INT8 + new kernel | 56.2 | 5.8 | 32.1 | 94.1 |
| relevance-8 | FP32, layers swapped | 66.3 | 3.0 | 23.8 | 93.2 |
| relevance-8 | INT8, layers swapped | 18.8 | 11.1 | 34.2 | 64.0 |
| relevance-8 | INT8, layers swapped + new kernel | 18.8 | 5.0 | 33.5 | 57.3 |
| relevance-16-prune50 | FP32 | 175.8 | 9.7 | 23.1 | 208.6 |
| relevance-16-prune50 | INT8 (ONNX Runtime) | 110.5 | 128.3 | 34.1 | 272.9 |
| relevance-16-prune50 | INT8 + new kernel | 110.6 | 8.0 | 32.6 | 151.3 |
| relevance-16-prune50 | FP32, layers swapped | 65.6 | 4.9 | 22.7 | 93.1 |
| relevance-16-prune50 | INT8, layers swapped | 18.9 | 18.1 | 33.9 | 70.9 |
| relevance-16-prune50 | INT8, layers swapped + new kernel | 18.5 | 6.0 | 33.4 | 57.9 |

## Earlier measurement (scripts/ort_kernel_study.py, without the second kernel)

| condition | variant | microseconds |
|---|---|---:|
| full-22 | FP32 | 274 |
| full-22 | INT8 (ONNX Runtime) | 358 |
| full-22 | INT8 + new kernel | 181 |
| full-22 | INT8, dense rewrite | 209 |
| full-22 | FP32, layers swapped | 94 |
| full-22 | INT8, layers swapped | 66 |
| relevance-16 | FP32 | 202 |
| relevance-16 | INT8 (ONNX Runtime) | 259 |
| relevance-16 | INT8 + new kernel | 138 |
| relevance-16 | INT8, dense rewrite | 160 |
| relevance-16 | FP32, layers swapped | 92 |
| relevance-16 | INT8, layers swapped | 57 |
| relevance-8 | FP32 | 114 |
| relevance-8 | INT8 (ONNX Runtime) | 147 |
| relevance-8 | INT8 + new kernel | 80 |
| relevance-8 | INT8, dense rewrite | 90 |
| relevance-8 | FP32, layers swapped | 90 |
| relevance-8 | INT8, layers swapped | 49 |
| relevance-16-prune50 | FP32 | 203 |
| relevance-16-prune50 | INT8 (ONNX Runtime) | 258 |
| relevance-16-prune50 | INT8 + new kernel | 137 |
| relevance-16-prune50 | INT8, dense rewrite | 160 |
| relevance-16-prune50 | FP32, layers swapped | 91 |
| relevance-16-prune50 | INT8, layers swapped | 56 |
