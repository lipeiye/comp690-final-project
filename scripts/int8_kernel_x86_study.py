"""Post-test study: the two missing INT8 kernels on x86 CPUs (part C of the plan's WP7).

scripts/ort_kernel_study.py and scripts/ort_kernel_swapped_study.py supplied the INT8 kernels ONNX Runtime
lacks for EEGNet's grouped spatial convolution and for the swapped network's first convolution, and checked
them on the Apple M4. On x86, ONNX Runtime rewrites the graph as U8S8 before fusing it (activations uint8), so
those int8 operators do not fit an x86 graph (the first plan of this study failed to load them on both x86
machines). kernels/ort_conv_u8.c holds the same two operators for uint8 activations (plain C, exact 32-bit
accumulation, compiled with -march=native on each machine; mibci.ortkernel_u8). On each machine, the stage `run`:

  equivalence  for the 459 archived INT8 graphs: the next-day logits with the grouped-convolution kernel
               swapped into this machine's optimized graph, against ONNX Runtime's own INT8 logits on this
               machine and against the archived M4 INT8 predictions. Where ONNX Runtime's INT8 is exact
               (VNNI) the kernel must give identical logits; where it is not (no VNNI) the count of models
               that return to the M4 predictions shows how much of the difference the spatial layer carries.
  timing       participant 1, seed 0, four architectures as scripts/hw_latency.py: FP32, INT8, INT8 with the
               kernel, and the swapped versions with and without the second kernel, interleaved rounds.

Needs ONNX Runtime's C API headers (--ort-headers) and a C compiler. Session 2 is read only to compare with
the archived predictions. Exploratory. Tables: scripts/analyze_int8_kernel_x86_study.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mibci import data, deploy, ortkernel, ortkernel_u8
from mibci.artifacts import model_exports, sha256, write_json
from mibci.env import environment
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import DATA_DIR, RESULTS_DIR
from mibci.study import Study
from mibci.timing import build_variants, cpu

STUDY = Study(
    "int8_kernel_x86_study",
    "scripts/int8_kernel_x86_study.py",
    sources=["src/mibci/ortkernel.py", "src/mibci/ortkernel_swapped.py", "src/mibci/ortkernel_u8.py",
             "kernels/ort_conv_u8.c", "src/mibci/timing.py", "src/mibci/deploy.py", "src/mibci/study.py"],
)  # fmt: skip
ARCHITECTURES = ("full-22", "relevance-16", "relevance-16-prune50", "relevance-8")
TIMING = {"rounds": 15, "runs": 500, "warmup": 100}


def models() -> list[str]:
    return sorted(p.stem for p in (RESULTS_DIR / "test").glob("sub-*.json"))


def create_plan() -> None:
    plan = STUDY.create(
        [{"name": m} for m in models()],
        purpose="Run the two INT8 kernels written for the M4 on x86 CPUs: exact where ONNX Runtime's INT8 is exact, "
        "how many predictions they restore where it is not, and their speed.",
        claim_boundary="Exploratory, planned after the portability and kernel results were known, and after the first "
        "plan of this study failed (its int8 operators do not fit x86's uint8 graphs). The uint8 operators are plain C "
        "loops compiled for each machine (no hand-written x86 SIMD); timings are batch-1, one thread, on shared nodes.",
        architectures=list(ARCHITECTURES),
        timing=TIMING,
        kernels={"source": str(ortkernel_u8.SOURCE.name), "domain": ortkernel_u8.DOMAIN, "flags": ortkernel_u8.FLAGS},
        replaces="int8_kernel_x86_study_unexecuted_v1 (int8 operators; no records)",
    )
    print(f"{len(plan['jobs'])} archived INT8 graphs")


def equivalence(name: str, lib: Path, work: Path) -> dict:
    bundle_path = FINAL_MODELS / f"{name}.pt"
    _, scaler, b = load_bundle(bundle_path)
    ses = data.load_session(int(b["subject"]), 2, final_evaluation=True)
    X = scaler.transform(ses["X"][:, b["channels"]])
    int8 = model_exports(bundle_path)[1]
    opt, swapped = work / f"{name}_opt.onnx", work / f"{name}_kernel.onnx"
    ortkernel.optimized(int8, opt)
    ortkernel_u8.with_kernels(opt, swapped)
    plain = deploy.logits(ortkernel.session(int8), X)
    kernel = deploy.logits(ortkernel.session(swapped, lib), X)
    archived = np.array(json.loads((RESULTS_DIR / "test" / f"{name}.json").read_text())["pred"]["int8"])
    clean = ~ses["artifact"]
    return {"model": name, "max_abs_kernel_minus_runtime": float(np.abs(kernel - plain).max()),
            "kernel_equals_runtime": bool(np.array_equal(kernel, plain)),
            "runtime_changed_vs_archived": int((plain.argmax(1) != archived).sum()),
            "kernel_changed_vs_archived": int((kernel.argmax(1) != archived).sum()),
            "runtime_primary_acc": float((plain.argmax(1)[clean] == ses["y"][clean]).mean()),
            "kernel_primary_acc": float((kernel.argmax(1)[clean] == ses["y"][clean]).mean()),
            "archived_primary_acc": float((archived[clean] == ses["y"][clean]).mean())}  # fmt: skip


def timing(lib: Path, work: Path) -> dict:
    out = {}
    for arch in ARCHITECTURES:
        name = f"sub-01_seed-0_{arch}"
        bundle_path = FINAL_MODELS / f"{name}.pt"
        _, scaler, b = load_bundle(bundle_path)
        ses1 = data.load_session(1, 1)
        calibration = scaler.transform(ses1["X"][~ses1["artifact"]][:, b["channels"]])
        fp32, int8 = model_exports(bundle_path)
        made = build_variants(fp32, work, name, calibration, ("fp32-spatial-first", "int8-spatial-first"))
        opt, kern = work / f"{name}_opt.onnx", work / f"{name}_kernel.onnx"
        ortkernel_u8.with_kernels(ortkernel.optimized(int8, opt), kern)
        sopt, skern = work / f"{name}_sf_opt.onnx", work / f"{name}_sf_kernel.onnx"
        ortkernel_u8.with_kernels(ortkernel.optimized(made["int8-spatial-first"], sopt), skern)
        sessions = {"fp32": ortkernel.session(fp32), "int8": ortkernel.session(int8),
                    "int8-kernel": ortkernel.session(kern, lib),
                    "fp32-spatial-first": ortkernel.session(made["fp32-spatial-first"]),
                    "int8-spatial-first": ortkernel.session(made["int8-spatial-first"]),
                    "int8-spatial-first-kernel": ortkernel.session(skern, lib)}  # fmt: skip
        inputs = calibration[:8]
        check = deploy.logits(sessions["int8-spatial-first"], inputs)
        same = bool(np.array_equal(deploy.logits(sessions["int8-spatial-first-kernel"], inputs), check))
        out[arch] = {"timed": ortkernel.timed(sessions, inputs, TIMING["rounds"], TIMING["runs"], TIMING["warmup"]),
                     "swapped_kernel_equals_runtime_on_8_windows": same}  # fmt: skip
        print(arch, {k: round(v["median_ms"] * 1000, 1) for k, v in out[arch]["timed"].items()}, flush=True)
    return out


def run(label: str, headers: Path) -> None:
    plan = STUDY.load()
    out = STUDY.out / "run" / f"{label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    if platform.machine() not in ("x86_64", "AMD64"):
        raise SystemExit("this study runs the uint8 operators of x86 graphs; run it on an x86 machine")
    build = DATA_DIR / "int8_kernel_x86" / label
    lib = ortkernel_u8.build(headers, build / "libu8.so")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, job in enumerate(plan["jobs"], 1):
            rows.append(equivalence(job["name"], lib, Path(tmp)))
            if i % 50 == 0:
                print(f"[{i}/{len(plan['jobs'])}]", flush=True)
        timed = timing(lib, Path(tmp))
    libraries = {"u8": sha256(lib)}
    write_json(out, {"label": label, "host": socket.gethostname(), "cpu": cpu(), "onnxruntime": ort.__version__,
                     "environment": environment(), "library_sha256": libraries, "equivalence": rows,
                     "timing": timed, "protocol_sha256": sha256(STUDY.protocol_path)})  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "run"], required=True)
    parser.add_argument("--label", default=socket.gethostname().split(".")[0], help="run: name of this machine")
    parser.add_argument("--ort-headers", type=Path, help="run: directory with onnxruntime_c_api.h")
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
    elif args.ort_headers is None:
        raise SystemExit("run needs --ort-headers")
    else:
        run(args.label, args.ort_headers)


if __name__ == "__main__":
    main()
