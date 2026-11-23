"""Post-test study: EEGNet in INT8 and FP32 on emulated Cortex-M microcontrollers (CMSIS-NN in QEMU).

The review asked why INT8 did not make the decoder faster, and whether integer hardware is the
reason. On the M4 Mac and on server CPUs the answer was the runtime's slow path for the grouped
spatial convolution. Microcontrollers are where an 8-bit decoder would actually be deployed, so
this study runs the trained networks with Arm's CMSIS-NN kernels (v8.0.0, the library TensorFlow
Lite for Microcontrollers uses on Cortex-M) on emulated cores that differ in exactly the hardware in
question (mibci.mcu_build):

  m0         Cortex-M0: no FPU (float is emulated in software), no SIMD
  m4         Cortex-M4F: single-precision FPU, DSP extension (2 x 16-bit MACs per instruction)
  m4-nodsp   the int8 kernels without the DSP extension (CMSIS-NN plain C, Armv7-M) on the M4
  m55        Cortex-M55: FPU and Helium vectors (16 x 8-bit or 4 x float32 lanes)
  m55-nomve  the same core with Helium switched off

Networks (mibci.mcu): the archived final models of full-22, relevance-16, relevance-8 and the frozen
relevance-16-prune50, each in the original and the swapped layer order, BatchNorm folded. INT8 uses
TFLite-style post-training quantization calibrated on the model's own session-1 trials; FP32 uses
CMSIS-NN's experimental float kernels. Stages:

  plan       the jobs, toolchain versions and build flags
  reference  every model (participants 1-9, seeds 0-2, four conditions, two orders): session-2
             accuracy of FP32 and of the CMSIS-NN integer arithmetic (NumPy reference)
  emulate    build and run on QEMU: participant 1 seed 0 on every core and precision; participants
             2-9 (full-22 and the frozen model, int8 on m0, m4, m55) to check that the emulated
             logits equal the reference bit for bit. Each run uses the first 8 session-2 windows and
             records instructions per layer, flash sections, scratch and stack

QEMU models no timing: the results are instruction counts, not cycles or energy. Exploratory (the
test results were known). Tables: scripts/analyze_mcu_study.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mibci import data, mcu, mcu_build
from mibci.artifacts import sha256, write_json
from mibci.model import load_bundle
from mibci.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR
from mibci.study import Study

MCU_FILES = ["mcu/board.ld.in", "mcu/eegnet_f32.c", "mcu/eegnet_int8.c", "mcu/icount.c", "mcu/main.c", "mcu/net.h",
             "mcu/semihost.c", "mcu/semihost.h", "mcu/startup.c"]  # fmt: skip
STUDY = Study(
    "mcu_study",
    "scripts/mcu_study.py",
    sources=["src/mibci/mcu.py", "src/mibci/mcu_build.py", "src/mibci/fakequant.py", "src/mibci/study.py", *MCU_FILES],
)
CONDITIONS = ("full-22", "relevance-16", "relevance-8", "relevance-16-prune50")
CHECKED = ("full-22", "relevance-16-prune50")  # emulated for every participant
SEEDS = (0, 1, 2)
N_WINDOWS = 8
ALL_BUILDS = [("int8", c) for c in ("m0", "m4", "m4-nodsp", "m55", "m55-nomve")] + [
    ("f32", c) for c in ("m0", "m4", "m55", "m55-nomve")
]
CHECK_BUILDS = [("int8", "m0"), ("int8", "m4"), ("int8", "m55")]
STACK_BYTES = 16384  # reserved on the big boards; the stack actually used is measured


def model_name(subject: int, seed: int, condition: str) -> str:
    return f"sub-{subject:02d}_seed-{seed}_{condition}"


def plan_jobs() -> list[dict]:
    jobs = []
    for subject in data.SUBJECTS:
        for condition in CONDITIONS:
            builds = ALL_BUILDS if subject == 1 else CHECK_BUILDS if condition in CHECKED else []
            jobs.extend(
                {"name": f"{model_name(subject, 0, condition)}_{order}_{precision}_{core}", "subject": subject,
                 "seed": 0, "condition": condition, "order": order, "precision": precision, "core": core}
                for order in mcu.ORDERS
                for precision, core in builds
            )  # fmt: skip
    return jobs


def create_plan(t: dict) -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Does INT8 make EEGNet faster on microcontrollers, with and without integer SIMD and an FPU; how much "
        "do the layer order, the electrode count and pruning change; and does the CMSIS-NN integer pipeline keep the "
        "accuracy?",
        claim_boundary="Exploratory, planned after the test results were known. QEMU executes instructions but models "
        "no timing, so the results are instruction counts per 2-s window, not cycles, time or energy, and say nothing "
        "about memory wait states. Only the network runs on the emulated core: band-pass filtering, standardization "
        "and input quantization happen beforehand. Flash and RAM come from the linker and a painted stack on emulated "
        "boards, not from real silicon. INT8 follows the TFLite specification with min-max calibration on the "
        "model's own session-1 trials; it is not the archived ONNX INT8 graph. The frozen configuration does not "
        "change.",
        reference={
            "participants": data.SUBJECTS,
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "orders": list(mcu.ORDERS),
            "session2": "all 288 windows; artifact-free ones are the primary score",
        },
        emulation={
            "windows": f"the first {N_WINDOWS} session-2 windows of the participant (no selection)",
            "all_builds_for": "participant 1, seed 0",
            "checked_conditions": list(CHECKED),
            "check_builds": [list(b) for b in CHECK_BUILDS],
            "all_builds": [list(b) for b in ALL_BUILDS],
            "stack_reserved_bytes": STACK_BYTES,
        },
        cores={k: {"flags": v["flags"], "board": v["board"]} for k, v in mcu_build.CORES.items()},
        boards={k: {r: [hex(o), n] for r, (o, n) in v.items()} for k, v in mcu_build.BOARDS.items()},
        library_flags=mcu_build.LIBRARY_FLAGS,
        program_flags=mcu_build.PROGRAM_FLAGS,
        tools=mcu_build.versions(t),
        quantization="TFLite int8: per-tensor asymmetric activations (min-max over the model's session-1 trials), "
        "per-channel symmetric conv weights, per-tensor fully connected weights, int32 bias, CMSIS-NN default "
        "requantization; ELU as a 256-entry table; average pooling keeps its input scale (mibci.mcu)",
    )
    print(f"{len(plan['jobs'])} emulation jobs")


def _session(subject: int, session: int):
    return data.load_session(subject, session, final_evaluation=session == 2)


def prepared(subject: int, seed: int, condition: str, order: str):
    """The model's float network, its int8 version and the standardized session-2 windows."""
    model, scaler, bundle = load_bundle(MODELS_DIR / "final" / f"{model_name(subject, seed, condition)}.pt")
    s1 = _session(subject, 1)
    calibration = scaler.transform(s1["X"][~s1["artifact"]][:, bundle["channels"]])
    net = mcu.export(model, order)
    qnet = mcu.quantize(net, mcu.calibrate(net, calibration))
    s2 = _session(subject, 2)
    return model, net, qnet, scaler.transform(s2["X"][:, bundle["channels"]]), s2


def reference_job(subject: int, seed: int, condition: str, order: str) -> None:
    torch.set_num_threads(1)
    model, net, qnet, X2, s2 = prepared(subject, seed, condition, order)
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X2)).double().numpy()
    float_logits = mcu.float_forward(net, X2)
    int_logits = mcu.int_forward(qnet, mcu.quantize_input(X2, qnet))
    y, clean = s2["y"], ~s2["artifact"]
    archived = json.loads((RESULTS_DIR / "test" / f"{model_name(subject, seed, condition)}.json").read_text())["pred"]
    preds = {"torch_fp32": torch_logits.argmax(1), "mcu_fp32": float_logits.argmax(1),
             "mcu_int8": int_logits.astype(np.int64).argmax(1), "onnx_fp32": np.array(archived["fp32"]),
             "onnx_int8": np.array(archived["int8"])}  # fmt: skip
    rows = mcu.shapes(net, data.N_TIMES)
    write_json(STUDY.out / "reference" / f"{model_name(subject, seed, condition)}_{order}.json", {
        "subject": subject, "seed": seed, "condition": condition, "order": order,
        "acc_clean": {k: float((p[clean] == y[clean]).mean()) for k, p in preds.items()},
        "acc_all": {k: float((p == y).mean()) for k, p in preds.items()},
        "agreement_with_torch_fp32": {k: float((p == preds["torch_fp32"]).mean()) for k, p in preds.items()},
        "max_abs_float_logit_difference": float(np.abs(float_logits - torch_logits).max()),
        "layers": rows, "macs": sum(r["macs"] for r in rows), "buffers": list(mcu.buffers(rows)),
        "int8_bytes": int(sum(q["w"].size + 4 * q["b"].size + 8 * np.size(q["multiplier"]) for q in qnet["layers"]
                              if "w" in q) + sum(256 for q in qnet["layers"] if q["op"] == "elu")),
        "f32_bytes": int(sum(4 * (layer["w"].size + layer["b"].size) for layer in net["layers"] if "w" in layer)),
        "activation_params": {k: [float(s), int(z)] for k, (s, z) in qnet["params"].items()},
        "pred": {k: p.tolist() for k, p in preds.items()}, "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def emulate_job(job: dict, t: dict, build_dir: Path, plugin: Path) -> None:
    _, net, qnet, X2, _ = prepared(job["subject"], job["seed"], job["condition"], job["order"])
    X = X2[:N_WINDOWS]
    rows = mcu.shapes(net, data.N_TIMES)
    if job["precision"] == "int8":
        windows = mcu.quantize_input(X, qnet)
        expected = mcu.int_forward(qnet, windows)
        network = qnet
    else:
        windows = mcu.to_nhwc(X, job["order"]).astype(np.float32)
        expected = mcu.float_forward(net, X)
        network = net
    out = build_dir / job["name"]
    mcu.write_headers(out / "gen", network, rows, windows, job["precision"])
    board = mcu_build.CORES[job["core"]]["board"]
    start = time.perf_counter()
    elf = mcu_build.link(t, job["core"], job["precision"], out / "gen", out / "probe", board, 0, STACK_BYTES)
    result = mcu_build.emulate(t, elf, board, plugin)
    scratch = 0
    if result.get("scratch_too_small"):  # second build with exactly the scratch the kernels asked for
        scratch = result["scratch_needed"]
        elf = mcu_build.link(t, job["core"], job["precision"], out / "gen", out / "final", board, scratch, STACK_BYTES)
        result = mcu_build.emulate(t, elf, board, plugin)
    if result["exit_status"] != 0 or len(result["outputs"]) != N_WINDOWS or result.get("fault"):
        raise mcu_build.BuildError(f"{job['name']}: emulation failed ({result['exit_status']})")
    dtype = np.int8 if job["precision"] == "int8" else np.float32
    got = np.array([np.frombuffer(result["outputs"][i], dtype=dtype) for i in range(N_WINDOWS)])
    regions = np.array(result["regions"]).reshape(N_WINDOWS, len(mcu.LAYERS))
    sizes = mcu_build.sections(t, elf)
    write_json(STUDY.record("emulate", job), {
        **job, "board": board, "scratch_bytes": scratch, "stack_used": result["stack_used"],
        "sections": sizes, "flash_bytes": sum(sizes.get(s, 0) for s in mcu_build.FLASH_SECTIONS),
        "ram_bytes": sizes.get(".data", 0) + sizes.get(".bss", 0) + result["stack_used"],
        "instructions": {layer: regions[:, i].tolist() for i, layer in enumerate(mcu.LAYERS)},
        "instructions_per_window": regions.sum(1).tolist(),
        "bit_exact": bool(np.array_equal(got, expected)) if job["precision"] == "int8" else None,
        "max_abs_difference": float(np.abs(got.astype(np.float64) - expected).max()),
        "emulated": got.tolist(), "build_and_run_s": time.perf_counter() - start,
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "reference", "emulate"], required=True)
    parser.add_argument("--toolchain", type=Path, help="bin directory of the Arm GNU toolchain (default: PATH)")
    parser.add_argument("--cmsis-nn", type=Path, help="CMSIS-NN source tree (the plan records its commit)")
    parser.add_argument("--build-dir", type=Path, default=DATA_DIR / "mcu_build", help="generated code and binaries")
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    args = parser.parse_args()
    if args.stage == "reference":
        STUDY.load()
        todo = [(s, seed, c, o) for s in args.subjects for seed in SEEDS for c in CONDITIONS for o in mcu.ORDERS]
        todo = [j for j in todo if not (STUDY.out / "reference" / f"{model_name(*j[:3])}_{j[3]}.json").exists()]
        for i, job in enumerate(todo, 1):
            reference_job(*job)
            print(f"[{i}/{len(todo)}] {model_name(*job[:3])} {job[3]}", flush=True)
        return
    t = mcu_build.tools(args.toolchain, args.cmsis_nn)
    if args.stage == "plan":
        create_plan(t)
        return
    plan = STUDY.load()
    if mcu_build.versions(t) != plan["tools"]:
        raise SystemExit(f"Toolchain differs from the plan: {mcu_build.versions(t)} vs {plan['tools']}")
    plugin = mcu_build.plugin(t, args.build_dir)
    jobs = [j for j in plan["jobs"] if j["subject"] in args.subjects]
    pending = STUDY.pending(jobs, "emulate")
    for i, job in enumerate(pending, 1):
        emulate_job(job, t, args.build_dir, plugin)
        print(f"[{i}/{len(pending)}] {job['name']}", flush=True)


if __name__ == "__main__":
    main()
