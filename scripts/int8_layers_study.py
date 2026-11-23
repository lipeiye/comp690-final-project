"""Post-test study: in which layer does the same INT8 graph compute differently on x86 CPUs without VNNI, and why?

scripts/int8_portability.py found that the archived INT8 graphs predict identically on the Apple M4 and on
AVX-512 VNNI CPUs but differently (457 of 459 models, -4 points) on x86 CPUs without VNNI, and that 7-bit
weights everywhere remove the difference. This is part B of the plan's WP7 (mibci.int8_layers):

  capture  on one machine: for 18 archived models (participants 1-9, seed 0, full-22 and the frozen
           relevance-16-prune50), every int8 tensor of the graph and the logits for the first 32 artifact-free
           windows of the training day (session 1 only), for the archived graph and for six variants with
           7-bit weights in one weighted layer, or in all five. The tensors go to the data directory
           (int8_layers/<label>/); a record of the machine to results.
  compare  two captures (a VNNI reference and a machine without VNNI): per model, variant and tensor, how
           many values differ and by how much; the first tensor that differs; for the archived graph, whether
           the outputs that differ in that layer are those whose x86 pairing of products (two adjacent 8-bit
           products summed in 16 bits) leaves the int16 range; and which single 7-bit layer is enough to make
           the two machines agree.

Exploratory; no session-2 data. Tables: scripts/analyze_int8_layers_study.py.
"""

from __future__ import annotations

import argparse
import socket
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import onnxruntime as ort

from mibci import data, int8_layers
from mibci.artifacts import model_exports, sha256, write_json
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import DATA_DIR
from mibci.study import Study
from mibci.timing import cpu

STUDY = Study(
    "int8_layers_study",
    "scripts/int8_layers_study.py",
    sources=["src/mibci/int8_layers.py", "src/mibci/deploy.py", "src/mibci/study.py"],
)
CONDITIONS = ("full-22", "relevance-16-prune50")
N_WINDOWS = 32
VARIANTS = ("int8", *(f"7bit-{layer}" for layer in int8_layers.WEIGHTED), "7bit-all")


def model_name(subject: int, condition: str) -> str:
    return f"sub-{subject:02d}_seed-0_{condition}"


def plan_jobs() -> list[dict]:
    return [{"name": model_name(s, c), "subject": s, "condition": c} for s in data.SUBJECTS for c in CONDITIONS]


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Find the layer where the archived INT8 graphs compute differently on x86 CPUs without VNNI, test "
        "the explanation (adjacent 8-bit products summed in a saturating 16-bit register), and find the smallest "
        "7-bit change that makes the machines agree.",
        claim_boundary="Exploratory, planned after the portability results were known. Training-day windows only; the "
        "archived graphs are not changed (the 7-bit variants are new copies). The int16-pairing model of the x86 "
        "kernel is a hypothesis this study checks against the observed differences.",
        variants=list(VARIANTS),
        windows=f"the first {N_WINDOWS} artifact-free session-1 windows of each participant",
        pairing="activations + 128 as uint8, int8 weights, adjacent products along the reduction axis in the order "
        "(kh, kw, input channel) summed into int16 (x86 VPMADDUBSW without VNNI)",
    )
    print(f"{len(plan['jobs'])} models x {len(VARIANTS)} variants")


def windows(subject: int, condition: str) -> np.ndarray:
    _, scaler, bundle = load_bundle(FINAL_MODELS / f"{model_name(subject, condition)}.pt")
    ses = data.load_session(subject, 1)
    X = ses["X"][~ses["artifact"]][:N_WINDOWS][:, bundle["channels"]]
    return scaler.transform(X)


def graph(job: dict, variant: str) -> onnx.ModelProto:
    archived = onnx.load(str(model_exports(FINAL_MODELS / f"{job['name']}.pt")[1]))
    if variant == "int8":
        return archived
    layers = int8_layers.WEIGHTED if variant == "7bit-all" else (variant.removeprefix("7bit-"),)
    for layer in layers:
        archived = int8_layers.seven_bit(archived, layer)
    return archived


def store(label: str):
    return DATA_DIR / "int8_layers" / label


def capture(label: str) -> None:
    plan = STUDY.load()
    out = STUDY.out / "capture" / f"{label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    folder = store(label)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for job in plan["jobs"]:
        X = windows(job["subject"], job["condition"])
        for variant in VARIANTS:
            tensors = int8_layers.capture(graph(job, variant), X)
            path = folder / f"{job['name']}__{variant}.npz"
            np.savez_compressed(path, **tensors)
            files[path.name] = sha256(path)
        print(job["name"], flush=True)
    write_json(out, {"label": label, "host": socket.gethostname(), "cpu": cpu(), "onnxruntime": ort.__version__,
                     "finished": datetime.now(ZoneInfo("America/New_York")).isoformat(), "files": files,
                     "protocol_sha256": sha256(STUDY.protocol_path)})  # fmt: skip


def load(label: str, job: dict, variant: str) -> dict[str, np.ndarray]:
    with np.load(store(label) / f"{job['name']}__{variant}.npz") as f:  # noqa: TID251  our own captures
        return {k: f[k] for k in f.files}


def layer_map(model: onnx.ModelProto) -> dict[str, dict]:
    """For each weighted layer: its int8 input and output tensors, input zero point, groups and pads."""
    out = {}
    quantized = {n.input[0]: n.output[0] for n in model.graph.node if n.op_type == "QuantizeLinear"}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue
        layer = node.input[1].removesuffix(".weight_DequantizeLinear_Output")
        source = node.input[0].removesuffix("_DequantizeLinear_Output")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        out[layer] = {"input": f"{source}_QuantizeLinear_Output", "zero": f"{source}_zero_point",
                      "output": quantized[node.output[0]], "op": node.op_type,
                      "groups": int(attrs.get("group", 1)), "pads": tuple(attrs.get("pads", (0, 0, 0, 0)))}  # fmt: skip
    return out


def saturation_check(model: onnx.ModelProto, layer: dict, name: str, ref: dict, other: dict) -> dict:
    """Do the outputs that differ in this layer coincide with the outputs the int16-pairing model predicts?"""
    x = ref[layer["input"]]
    w = int8_layers.initializer(model, f"{name}.weight_quantized")
    zero = int(int8_layers.initializer(model, layer["zero"]))
    if layer["op"] == "Gemm":
        x, w = x.reshape(len(x), -1, 1, 1), w.reshape(w.shape[0], -1, 1, 1)
    predicted = int8_layers.pair_saturation(x, zero, w, layer["groups"], layer["pads"])
    differ = (ref[layer["output"]] != other[layer["output"]]).reshape(predicted.shape)
    return {"outputs": int(differ.size), "differ": int(differ.sum()), "predicted": int(predicted.sum()),
            "differ_and_predicted": int((differ & predicted).sum()),
            "differ_not_predicted": int((differ & ~predicted).sum()),
            "predicted_not_differ": int((~differ & predicted).sum())}  # fmt: skip


def compare(reference: str, other: str) -> None:
    plan = STUDY.load()
    rows = []
    for job in plan["jobs"]:
        archived = graph(job, "int8")
        layers = layer_map(archived)
        order = int8_layers.quantized_outputs(archived)
        for variant in VARIANTS:
            a, b = load(reference, job, variant), load(other, job, variant)
            tensors = {}
            for name in [*order, "logits"]:
                diff = np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))
                tensors[name] = {"values": int(diff.size), "differ": int((diff > 0).sum()),
                                 "max_abs": float(diff.max())}  # fmt: skip
            first = next((n for n in order if tensors[n]["differ"]), None)
            row = {"model": job["name"], "variant": variant, "first_differing": first, "tensors": tensors,
                   "same_predictions": int((a["logits"].argmax(1) == b["logits"].argmax(1)).sum()),
                   "windows": len(a["logits"])}  # fmt: skip
            if variant == "int8" and first is not None:
                name = next((k for k, v in layers.items() if v["output"] == first), None)
                row["first_layer"] = name  # None: the first tensor that differs is not a weighted layer's output
                if name is not None:
                    row["saturation"] = saturation_check(archived, layers[name], name, a, b)
            rows.append(row)
        print(job["name"], flush=True)
    out = {"reference": reference, "other": other, "rows": rows, "protocol_sha256": sha256(STUDY.protocol_path)}
    write_json(STUDY.out / "compare" / f"{other}_vs_{reference}.json", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "capture", "compare"], required=True)
    parser.add_argument("--label", default=socket.gethostname().split(".")[0], help="capture: name of this machine")
    parser.add_argument("--reference", help="compare: label of the VNNI capture")
    parser.add_argument("--other", help="compare: label of the capture without VNNI")
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
    elif args.stage == "capture":
        capture(args.label)
    else:
        if not (args.reference and args.other):
            raise SystemExit("compare needs --reference and --other")
        compare(args.reference, args.other)


if __name__ == "__main__":
    main()
