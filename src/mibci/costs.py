"""Analytical cost model of EEGNet for one 2-s window (batch 1).

MACs count multiply-accumulates in convolutions and the dense layer. Activation
memory assumes an explicit layer-by-layer schedule: while a layer runs, only its
input and output tensors are live, so the peak is the largest input + output pair.
"""

from __future__ import annotations

import torch
from torch import nn

from mibci.data import N_TIMES
from mibci.model import deploy_copy

_COUNTED = (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.ELU)


def layer_costs(model: nn.Module) -> list[dict]:
    model = deploy_copy(model)
    rows = []

    def record(name, module, inputs, output):
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs = output.numel() * module.in_channels // module.groups * kh * kw
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
        else:
            macs = 0
        rows.append({
            "layer": name,
            "type": type(module).__name__,
            "out_shape": list(output.shape[1:]),
            "in_elems": inputs[0].numel(),
            "out_elems": output.numel(),
            "macs": macs,
            "params": sum(p.numel() for p in module.parameters(recurse=False)),
        })

    handles = [
        m.register_forward_hook(lambda mod, i, o, name=name: record(name, mod, i, o))
        for name, m in model.named_modules()
        if isinstance(m, _COUNTED)
    ]
    with torch.no_grad():
        model(torch.zeros(1, model.conv_spatial.kernel_size[0], N_TIMES))
    for h in handles:
        h.remove()
    return rows


def cost_summary(model: nn.Module) -> dict:
    rows = layer_costs(model)
    peak = max(r["in_elems"] + r["out_elems"] for r in rows)
    return {
        "macs": sum(r["macs"] for r in rows),
        "params": sum(p.numel() for p in deploy_copy(model).parameters()),
        "input_elems": rows[0]["in_elems"],
        "peak_activation_elems": peak,
        # Both exported graphs accept float32 EEG. Quantization occurs inside QDQ.
        "input_bytes": {"fp32": 4 * rows[0]["in_elems"], "int8": 4 * rows[0]["in_elems"]},
        "modeled_input_bytes": {"fp32": 4 * rows[0]["in_elems"], "int8": rows[0]["in_elems"]},
        "peak_activation_bytes": {"fp32": 4 * peak, "int8": peak},
        "activation_memory_kind": "idealized layer input+output, uniformly 4 or 1 byte per element; not measured QDQ/ORT memory",
    }
