# Where INT8 computes differently without VNNI

Archived INT8 graphs and 7-bit variants on training-day windows; a VNNI CPU as reference.

**Caution.** This capture added the int8 tensors to the graph's outputs, which on x86 kept ONNX Runtime from rewriting the graph as U8S8 before fusing it, so these tensors are not what a normal session computes there (a normal session on the Gold 6140 changes 144 of P1's 273 training-day predictions; the capture changes none). scripts/int8_capture_study.py repeats the comparison on the graphs each machine actually runs.

## novnni-intel-6140 vs vnni-amd-9654

First tensor that differs (archived graphs, 18 models): none 18

In that layer 0 of 0 outputs differ; the int16-pairing model predicts 0 to saturate; 0 differ and are predicted (0% of those that differ, 0% of those predicted).

| tensor | values differing (all models) | largest difference |
|---|---:|---:|
| input | 0.00% | 0 |
| temporal conv | 0.00% | 0 |
| spatial conv | 0.00% | 0 |
| pool 1 | 0.00% | 0 |
| depthwise conv | 0.00% | 0 |
| pointwise conv | 0.00% | 0 |
| pool 2 | 0.00% | 0 |
| logits (int8) | 0.00% | 0 |
| logits (float) | 0.00% | 0 |

| variant | models identical in every tensor | predictions the same |
|---|---:|---:|
| 7bit-all | 18/18 | 576/576 |
| 7bit-conv_separable_depth | 18/18 | 576/576 |
| 7bit-conv_separable_point | 18/18 | 576/576 |
| 7bit-conv_spatial | 18/18 | 576/576 |
| 7bit-conv_temporal | 18/18 | 576/576 |
| 7bit-final_layer | 18/18 | 576/576 |
| int8 | 18/18 | 576/576 |

