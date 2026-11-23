# Where INT8 computes differently without VNNI (the graphs each machine runs)

Archived INT8 graphs and 7-bit variants, optimized on each machine as a normal session does; integer tensors read from that optimized graph for the first 32 artifact-free training-day windows; predictions of normal sessions on all artifact-free training-day windows. A VNNI CPU is the reference.

| machine | CPU | captures equal to a normal session's logits |
|---|---|---:|
| novnni-intel-6140 | Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz | 126/126 |
| vnni-amd-9654 | AMD EPYC 9654 96-Core Processor | 126/126 |

## novnni-intel-6140 vs vnni-amd-9654

First tensor that differs (archived graphs, 18 models): temporal conv 18

In that layer (18 models checked) 1,876,973 of 43,863,552 outputs differ; the u8 x s8 int16-pairing model predicts 2,098,302 to saturate; 1,876,973 differ and are predicted (100% of those that differ, 89% of those predicted).

| tensor | values differing (all models) | largest difference |
|---|---:|---:|
| input (as uint8) | 0.00% | 0 |
| input (as uint8) (layout) | 0.00% | 0 |
| temporal conv | 4.28% | 72 |
| spatial conv | 9.55% | 231 |
| spatial conv (layout) | 9.55% | 231 |
| pool 1 (as uint8) | 11.40% | 255 |
| pool 1 (as uint8) (layout) | 11.40% | 255 |
| depthwise conv | 23.34% | 184 |
| pointwise conv | 64.04% | 107 |
| pointwise conv (layout) | 64.04% | 107 |
| pool 2 (as uint8) | 64.29% | 188 |
| pool 2 (as uint8) (layout) | 64.29% | 188 |
| dense (logits) | 83.42% | 103 |
| logits | 83.42% | 6.28494 |

| variant | models identical in every tensor | training-day predictions the same |
|---|---:|---:|
| 7bit-all | 18/18 | 4656/4656 (100.0%) |
| 7bit-conv_separable_depth | 0/18 | 3952/4656 (84.9%) |
| 7bit-conv_separable_point | 0/18 | 4173/4656 (89.6%) |
| 7bit-conv_spatial | 0/18 | 4326/4656 (92.9%) |
| 7bit-conv_temporal | 0/18 | 3958/4656 (85.0%) |
| 7bit-final_layer | 0/18 | 3944/4656 (84.7%) |
| int8 | 0/18 | 3949/4656 (84.8%) |

