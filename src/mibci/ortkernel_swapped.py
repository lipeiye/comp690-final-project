"""The second missing INT8 kernel: EEGNet's spatial convolution once the first two layers are swapped.

mibci.ortkernel replaced the grouped spatial convolution of the original layer order. After the swap
(mibci.variants.spatial_first) the network starts with a different spatial convolution, one input
channel and a kernel spanning all electrodes, and ONNX Runtime 1.31.1 runs that INT8 QLinearConv on a
generic path too. kernels/ort_spatial_first_conv.c implements it (domain "mibci.swapped", NHWC in and
out, optional bias); this module builds it and swaps it into ONNX Runtime's optimized graph, in the
same way as mibci.ortkernel.with_kernel, which stays unchanged because a finished study pins it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import onnx
from onnx import helper

from mibci.ortkernel import FLAGS
from mibci.paths import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "kernels" / "ort_spatial_first_conv.c"
DOMAIN, OP = "mibci.swapped", "QLinearSpatialConvNhwc"


def build(headers: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        ["cc", *FLAGS, f"-I{headers}", "-o", str(out), str(SOURCE)], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"kernel build failed:\n{done.stdout}{done.stderr}")
    return out


def with_kernel(optimized_path: Path, out: Path) -> dict:
    """Replace the single-group QLinearConv whose kernel spans the electrodes by the custom operator."""
    model = onnx.load(str(optimized_path))
    nodes = list(model.graph.node)
    found = []
    for node in nodes:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        single = node.op_type == "QLinearConv" and attrs.get("channels_last") == 1 and attrs.get("group", 1) == 1
        if single and attrs["kernel_shape"][1] == 1 and attrs["kernel_shape"][0] > 1:
            found.append((node, attrs))
    if len(found) != 1:
        raise ValueError(
            f"{optimized_path}: expected one spatial QLinearConv spanning the electrodes, found {len(found)}"
        )
    conv, attrs = found[0]
    if any(attrs.get("pads", [0] * 4)):
        raise ValueError(f"unexpected padding {attrs.get('pads')}")
    custom = helper.make_node(OP, list(conv.input), list(conv.output), name="spatial_first_int8", domain=DOMAIN)
    position = nodes.index(conv)
    del model.graph.node[:]
    model.graph.node.extend([*nodes[:position], custom, *nodes[position + 1 :]])
    if not any(o.domain == DOMAIN for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    onnx.save(model, str(out))
    return {"replaced": conv.name, "kernel_shape": attrs["kernel_shape"], "inputs": len(conv.input)}
