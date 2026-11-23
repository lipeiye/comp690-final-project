"""The two EEGNet INT8 kernels for x86 graphs, whose activations ONNX Runtime keeps as uint8 (WP7 C).

On x86, ONNX Runtime 1.31 rewrites an S8S8 QDQ graph as U8S8 before fusing it (mibci.int8_capture shows the
rewrite in the optimized graphs), so the int8 operators of kernels/ort_grouped_conv.c and
kernels/ort_spatial_first_conv.c cannot be swapped into an x86 graph. kernels/ort_conv_u8.c holds the same
two operators for uint8 activations, in domain "mibci.u8".

build        compile kernels/ort_conv_u8.c against ONNX Runtime's C API headers, for the machine it runs on
with_kernels swap the operator into an optimized graph: the grouped spatial convolution of the original order
             (as mibci.ortkernel.with_kernel) or the first convolution of the swapped network (as
             mibci.ortkernel_swapped.with_kernel), whichever the graph has
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import onnx
from onnx import helper

from mibci import ortkernel, ortkernel_swapped
from mibci.paths import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "kernels" / "ort_conv_u8.c"
DOMAIN = "mibci.u8"
FLAGS = [*ortkernel.FLAGS, "-march=native"]  # plain C loops: let the compiler use this machine's vector units


def build(headers: Path, out: Path) -> Path:
    """Compile the operator library against ONNX Runtime's C API headers (directory `headers`)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cc", *FLAGS, f"-I{headers}", "-o", str(out), str(SOURCE)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"kernel build failed:\n{done.stdout}{done.stderr}")
    return out


def with_kernels(optimized_path: Path, out: Path) -> dict:
    """The optimized graph with its grouped spatial (original order) or first (swapped) convolution replaced by
    the uint8 operator; the replacement is made by the int8 swap functions and moved to the uint8 domain."""
    with tempfile.TemporaryDirectory() as tmp:
        swapped = Path(tmp) / "swapped.onnx"
        try:
            info = {"layer": "grouped spatial", **ortkernel.with_kernel(optimized_path, swapped)}
            old = ortkernel.DOMAIN
        except ValueError:
            info = {"layer": "spatial first", **ortkernel_swapped.with_kernel(optimized_path, swapped)}
            old = ortkernel_swapped.DOMAIN
        model = onnx.load(str(swapped))
    for node in model.graph.node:
        if node.domain == old:
            node.domain = DOMAIN
    kept = [o for o in model.opset_import if o.domain != old]
    del model.opset_import[:]
    model.opset_import.extend([*kept, helper.make_opsetid(DOMAIN, 1)])
    onnx.save(model, str(out))
    return info
