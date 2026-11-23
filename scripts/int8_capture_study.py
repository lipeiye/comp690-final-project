"""Post-test study: in which layer does the same INT8 graph compute differently on x86 CPUs without VNNI?

Part B of the plan's WP7, second design. scripts/int8_layers_study.py read the QDQ graph's int8 tensors by
adding them as graph outputs; on x86 that stopped ONNX Runtime from rewriting the graph as U8S8 before fusing
it, so both x86 machines ran an exact int8 path and agreed in every tensor, although a normal session on the
same Intel Gold 6140 changes 144 of P1's 273 training-day predictions (diagnostic, 2026-11-22). Here each
machine first optimizes the graph as a normal session does (mibci.int8_capture) and the integer tensors are
read from that optimized graph with further optimization off; every capture's logits are checked against a
normal session's. Models, windows and 7-bit variants are those of int8_layers_study:

  capture  on one machine: for 18 archived models (participants 1-9, seed 0, full-22 and the frozen
           relevance-16-prune50) and 7 variants (the archived graph; 7-bit weights in one weighted layer, or in
           all five), the integer tensors and logits for the first 32 artifact-free training-day windows, the
           machine's optimized graph, and a normal session's logits for all artifact-free training-day windows.
           Tensors and graphs go to the data directory (int8_capture/<label>/), a record to results.
  compare  two captures (a VNNI reference and a machine without VNNI): per model, variant and tensor, how many
           values differ and by how much; the first tensor that differs; for the archived graph, whether the
           outputs that differ in that layer are those whose x86 pairing of products (two adjacent u8 x s8
           products summed in 16 bits) leaves the int16 range; and agreement of the normal sessions'
           predictions on all training-day windows, for every variant.

Exploratory; no session-2 data. Tables: scripts/analyze_int8_capture_study.py.
"""

from __future__ import annotations

import argparse
import socket
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from mibci import data, deploy, int8_capture, int8_layers
from mibci.artifacts import model_exports, sha256, write_json
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import DATA_DIR
from mibci.study import Study
from mibci.timing import cpu

STUDY = Study(
    "int8_capture_study",
    "scripts/int8_capture_study.py",
    sources=["src/mibci/int8_capture.py", "src/mibci/int8_layers.py", "src/mibci/deploy.py", "src/mibci/study.py"],
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
        purpose="Find the layer where the archived INT8 graphs compute differently on x86 CPUs without VNNI, reading "
        "the integer tensors of the graph each machine actually runs, test the explanation (adjacent u8 x s8 "
        "products summed in a saturating 16-bit register) and find the smallest 7-bit change that makes the "
        "machines agree.",
        claim_boundary="Exploratory, planned after the portability results and after int8_layers_study, whose capture "
        "changed the computation on x86 (both machines agreed exactly while a normal session on the Gold 6140 "
        "changed 144 of P1's 273 training-day predictions). Training-day windows only; the archived graphs are not "
        "changed (the 7-bit variants are new copies). Every capture is checked against a normal session's logits.",
        variants=list(VARIANTS),
        windows=f"tensors: the first {N_WINDOWS} artifact-free session-1 windows of each participant; predictions: "
        "all artifact-free session-1 windows",
        pairing="activations as the optimized graph stores them (uint8 on x86), int8 weights, adjacent products along "
        "the reduction axis in the order (kh, kw, input channel) summed into int16 (x86 VPMADDUBSW without VNNI)",
        replaces="int8_layers_study (kept as it is; its capture did not run the deployed x86 path)",
    )
    print(f"{len(plan['jobs'])} models x {len(VARIANTS)} variants")


def windows(subject: int, condition: str) -> np.ndarray:
    _, scaler, bundle = load_bundle(FINAL_MODELS / f"{model_name(subject, condition)}.pt")
    ses = data.load_session(subject, 1)
    return scaler.transform(ses["X"][~ses["artifact"]][:, bundle["channels"]])


def graph(job: dict, variant: str) -> onnx.ModelProto:
    archived = onnx.load(str(model_exports(FINAL_MODELS / f"{job['name']}.pt")[1]))
    if variant == "int8":
        return archived
    layers = int8_layers.WEIGHTED if variant == "7bit-all" else (variant.removeprefix("7bit-"),)
    for layer in layers:
        archived = int8_layers.seven_bit(archived, layer)
    return archived


def store(label: str):
    return DATA_DIR / "int8_capture" / label


def capture(label: str) -> None:
    plan = STUDY.load()
    out = STUDY.out / "capture" / f"{label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    folder = store(label)
    folder.mkdir(parents=True, exist_ok=True)
    rows, files = [], {}
    for job in plan["jobs"]:
        X = windows(job["subject"], job["condition"])
        for variant in VARIANTS:
            g = graph(job, variant)
            c = int8_capture.capture(g, X[:N_WINDOWS])
            stem = f"{job['name']}__{variant}"
            onnx.save(c["optimized"], str(folder / f"{stem}.onnx"))
            sess = ort.InferenceSession(g.SerializeToString(), _plain_options(), providers=["CPUExecutionProvider"])
            all_windows = deploy.logits(sess, X)
            np.savez_compressed(folder / f"{stem}.npz", **c["tensors"], plain_all=all_windows)
            for suffix in (".npz", ".onnx"):
                files[stem + suffix] = sha256(folder / f"{stem}{suffix}")
            rows.append({"model": job["name"], "variant": variant, "tensors": len(c["tensors"]) - 1,
                         "capture_equals_plain": bool(np.array_equal(c["tensors"]["logits"], c["plain"])),
                         "windows": len(X)})  # fmt: skip
        print(job["name"], flush=True)
    write_json(out, {"label": label, "host": socket.gethostname(), "cpu": cpu(), "onnxruntime": ort.__version__,
                     "finished": datetime.now(ZoneInfo("America/New_York")).isoformat(), "rows": rows,
                     "files": files, "protocol_sha256": sha256(STUDY.protocol_path)})  # fmt: skip


def _plain_options() -> ort.SessionOptions:
    """mibci.deploy.session's options, for a graph held in memory."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def load(label: str, job: dict, variant: str) -> tuple[dict[str, np.ndarray], onnx.ModelProto]:
    stem = store(label) / f"{job['name']}__{variant}"
    with np.load(f"{stem}.npz") as f:  # noqa: TID251  our own captures
        tensors = {k: f[k] for k in f.files}
    return tensors, onnx.load(f"{stem}.onnx")


def saturation_check(model: onnx.ModelProto, node: onnx.NodeProto, ref: dict, other: dict) -> dict | None:
    """Do the outputs of this layer that differ coincide with those the u8 x s8 int16-pairing model predicts?"""
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    x, out = ref.get(node.input[0]), node.output[0]
    if x is None or out not in ref or out not in other:
        return None
    zero = int(inits[node.input[2]]) if node.input[2] in inits else 0
    if x.dtype == np.uint8:  # mibci.int8_layers.pair_saturation takes int8 and adds 128
        x, zero = (x.astype(np.int16) - 128).astype(np.int8), zero - 128
    if node.op_type == "QGemm":
        b = inits.get(node.input[3])
        if b is None:
            return None
        w = b if attrs.get("transB", 0) else b.T
        x4, w4 = x.reshape(len(x), -1, 1, 1), w.reshape(w.shape[0], -1, 1, 1)
        groups, pads, channels_last = 1, (0, 0, 0, 0), 0
    else:
        w4 = inits.get(node.input[3])
        if w4 is None:
            return None
        channels_last = attrs.get("channels_last", 0)
        x4 = x.transpose(0, 3, 1, 2) if channels_last else x
        groups, pads = int(attrs.get("group", 1)), tuple(attrs.get("pads", (0, 0, 0, 0)))
        pads = (pads[0], pads[1], pads[2], pads[3]) if len(pads) == 4 else (0, 0, 0, 0)
    predicted = int8_layers.pair_saturation(x4, zero, w4, groups, pads)
    if channels_last:
        predicted = predicted.transpose(0, 2, 3, 1)
    differ = ref[out] != other[out]
    if differ.size != predicted.size:
        return None
    differ = differ.reshape(predicted.shape)
    return {"layer": out, "op": node.op_type, "outputs": int(differ.size), "differ": int(differ.sum()),
            "predicted": int(predicted.sum()), "differ_and_predicted": int((differ & predicted).sum()),
            "differ_not_predicted": int((differ & ~predicted).sum()),
            "predicted_not_differ": int((~differ & predicted).sum())}  # fmt: skip


def compare(reference: str, other: str) -> None:
    plan = STUDY.load()
    rows = []
    for job in plan["jobs"]:
        for variant in VARIANTS:
            (a, graph_a), (b, graph_b) = load(reference, job, variant), load(other, job, variant)
            order = [n for n in int8_capture.integer_tensors(graph_a) if n in a]
            names = [n for n in order if n in b]
            tensors = {}
            for name in [*names, "logits"]:
                diff = np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))
                tensors[name] = {
                    "values": int(diff.size),
                    "differ": int((diff > 0).sum()),
                    "max_abs": float(diff.max()),
                }
            lab = int8_capture.labels(graph_a)
            for name in names:
                tensors[name]["label"] = lab.get(name, name)
            first = next((n for n in names if tensors[n]["differ"]), None)
            pa, pb = a["plain_all"].argmax(1), b["plain_all"].argmax(1)
            row = {"model": job["name"], "variant": variant, "first_differing": first, "tensors": tensors,
                   "only_reference": [n for n in order if n not in b],
                   "only_other": [n for n in int8_capture.integer_tensors(graph_b) if n in b and n not in a],
                   "same_predictions": int((a["logits"].argmax(1) == b["logits"].argmax(1)).sum()),
                   "windows": len(a["logits"]), "same_predictions_all": int((pa == pb).sum()),
                   "windows_all": len(pa)}  # fmt: skip
            if variant == "int8" and first is not None:
                node = int8_capture.producers(graph_b).get(first)
                row["first_op"] = node.op_type if node is not None else None
                row["saturation"] = saturation_check(graph_b, node, a, b) if node is not None else None
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
