# INT8 portability: the same INT8 files on different CPUs

Reference: the archived predictions made on the original Apple M4. FP32 is the control.

| machine | CPU | 8-bit related extensions | FP32 models changed | INT8 models changed | INT8 mean agreement | INT8 accuracy change (pp) | 7-bit weights: accuracy vs FP32 (pp) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| c141601 | AMD EPYC 75F3 32-Core Processor | AVX2 | 0/459 | 457/459 | 0.8557 | -3.97 |  |
| c151411 | AMD EPYC 9654 96-Core Processor | AVX-512, AVX-512 VNNI, AVX2 | 0/459 | 0/459 | 1.0000 | +0.00 |  |
| c151417 | AMD EPYC 9654 96-Core Processor | AVX-512, AVX-512 VNNI, AVX2 | 0/459 | 0/459 | 1.0000 | +0.00 |  |
| c151608 | AMD EPYC 9654 96-Core Processor | AVX-512, AVX-512 VNNI, AVX2 | 0/459 | 0/459 | 1.0000 | +0.00 |  |
| intel-no-avx512 | Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz | AVX2 | 0/459 | 457/459 | 0.8557 | -3.97 |  |
| intel-xeon-gold-6140 | Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz | AVX-512, AVX2 | 0/459 | 457/459 | 0.8557 | -3.97 |  |
| m4-reference | Apple M4 | DotProd, I8MM | 0/459 | 0/459 | 1.0000 | +0.00 |  |
| rr-amd-epyc-75f3 | AMD EPYC 7702 64-Core Processor | AVX2 | 0/459 | 457/459 | 0.8557 | -3.97 | -0.17 |
| rr-amd-epyc-9654 | AMD EPYC 9654 96-Core Processor | AVX-512, AVX-512 VNNI, AVX2 | 0/459 | 0/459 | 1.0000 | +0.00 | -0.17 |
| rr-intel-gold-6140 | Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz | AVX-512, AVX2 | 0/459 | 457/459 | 0.8557 | -3.97 | -0.17 |
| rr-m4 | Apple M4 | DotProd, I8MM | 0/459 | 0/459 | 1.0000 | +0.00 | -0.17 |

Machines that reproduce every INT8 prediction: AMD EPYC 9654 96-Core Processor, Apple M4.
Machines that change INT8 predictions: AMD EPYC 75F3 32-Core Processor, AMD EPYC 7702 64-Core Processor, Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz, Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz.
Largest per-participant INT8 accuracy change on those machines: participant 8, -7.89 pp (c141601).
7-bit weights: models whose accuracy and agreement with FP32 equal the M4's to the last trial: rr-amd-epyc-75f3 459/459, rr-amd-epyc-9654 459/459, rr-intel-gold-6140 459/459.

FP32 agrees on every machine, so the environment is not the cause. On x86 CPUs without VNNI, ONNX Runtime's 8-bit kernels multiply with VPMADDUBSW, which adds pairs of products in 16 bits and can saturate; Arm dot products and VNNI accumulate in 32 bits. ONNX Runtime documents 7-bit weights (reduce_range) as the remedy for non-VNNI machines; the last column tests it.

Exploratory: run after the main study's test results were known; it concerns deployment, not the frozen analysis.
