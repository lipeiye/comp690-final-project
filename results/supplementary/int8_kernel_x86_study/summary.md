# The two INT8 kernels on x86 CPUs

459 archived INT8 graphs, next day; the grouped-convolution kernel's portable C path swapped into each machine's optimized graph. 'Changed' counts models with any next-day prediction unlike the archived M4 INT8 one.

| machine | CPU | kernel = runtime | changed vs M4: runtime / kernel | accuracy vs M4 (points): runtime / kernel |
|---|---|---:|---|---|
| novnni-intel-6140 | Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz | 9/459 | 457 / 441 | -3.97 / -1.88 |
| vnni-amd-9654 | AMD EPYC 9654 96-Core Processor | 459/459 | 0 / 0 | +0.00 / +0.00 |

## Latency (microseconds per window, one thread)

| machine | architecture | fp32 | int8 | int8-kernel | fp32-spatial-first | int8-spatial-first | int8-spatial-first-kernel |
|---|---|---:|---:|---:|---:|---:|---:|
| novnni-intel-6140 | full-22 | 561 | 1823 | 952 | 115 | 324 | 431 |
| novnni-intel-6140 | relevance-16 | 429 | 1375 | 725 | 114 | 293 | 364 |
| novnni-intel-6140 | relevance-16-prune50 | 419 | 1363 | 719 | 107 | 284 | 359 |
| novnni-intel-6140 | relevance-8 | 256 | 752 | 425 | 111 | 243 | 293 |
| vnni-amd-9654 | full-22 | 297 | 725 | 392 | 49 | 125 | 175 |
| vnni-amd-9654 | relevance-16 | 225 | 530 | 295 | 48 | 110 | 147 |
| vnni-amd-9654 | relevance-16-prune50 | 223 | 520 | 293 | 46 | 108 | 146 |
| vnni-amd-9654 | relevance-8 | 129 | 302 | 173 | 47 | 94 | 111 |
