"""Post-test study: the int8 temporal-convolution kernel CMSIS-NN lacks, on emulated Cortex-M cores.

scripts/mcu_study.py ran EEGNet with Arm's CMSIS-NN kernels in QEMU. With the layers swapped, the
temporal convolution (16 FIR filters of 64 taps) took 5.5 of the Cortex-M4's 6.9 million instructions per
window, 10.7 per multiply-accumulate, because CMSIS-NN runs it as a general depthwise convolution; in the
original order it is a CONV_2D with one input channel. mcu/kernels/fir_s8.c is that layer as an int8 FIR
filter bank with plain-C, DSP (SMLAD) and Helium (VMLADAVA) paths, and mcu/kernels/eegnet_int8_fir.c is
mcu/eegnet_int8.c with only that layer replaced (mibci.mcu_kernels).

The builds are mcu_study's int8 builds: participant 1, seed 0, the four conditions (full-22,
relevance-16, relevance-8, relevance-16-prune50), both layer orders, on m0, m4, m4-nodsp, m55 and
m55-nomve; participants 2-9 (full-22 and the frozen model) on m0, m4 and m55. Each runs the same first 8
session-2 windows as mcu_study, so that every layer's instruction count can be set beside the library's;
the emulated logits must equal mibci.mcu's NumPy integer reference bit for bit (session 2 serves only as
emulator input; nothing is chosen with it, and the kernel cannot change accuracy if it is bit-exact).

QEMU models no timing: these are instruction counts, not cycles, time or energy. Exploratory (the test
results were known). Stages: plan, emulate. Tables: scripts/analyze_mcu_kernel_study.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mibci import data, mcu, mcu_build, mcu_kernels
from mibci.artifacts import sha256, write_json
from mibci.model import load_bundle
from mibci.paths import DATA_DIR, MODELS_DIR
from mibci.study import Study

MCU_FILES = [
    *(f"mcu/{f}" for f in ("board.ld.in", "icount.c", "main.c", "net.h", "semihost.c", "semihost.h", "startup.c")),
    *(f"mcu/kernels/{f}" for f in ("eegnet_int8_fir.c", "fir_s8.c", "fir_s8.h")),
]
STUDY = Study(
    "mcu_kernel_study",
    "scripts/mcu_kernel_study.py",
    sources=[f"src/mibci/{m}.py" for m in ("mcu", "mcu_build", "mcu_kernels", "fakequant", "study")] + MCU_FILES,
)
CONDITIONS = ("full-22", "relevance-16", "relevance-8", "relevance-16-prune50")
CHECKED = ("full-22", "relevance-16-prune50")
ALL_CORES = ("m0", "m4", "m4-nodsp", "m55", "m55-nomve")
CHECK_CORES = ("m0", "m4", "m55")
N_WINDOWS = 8
STACK_BYTES = 16384


def model_name(subject: int, seed: int, condition: str) -> str:
    return f"sub-{subject:02d}_seed-{seed}_{condition}"


def plan_jobs() -> list[dict]:
    jobs = []
    for subject in data.SUBJECTS:
        for condition in CONDITIONS:
            cores = ALL_CORES if subject == 1 else CHECK_CORES if condition in CHECKED else ()
            jobs.extend(
                {"name": f"{model_name(subject, 0, condition)}_{order}_int8-fir_{core}", "subject": subject,
                 "seed": 0, "condition": condition, "order": order, "core": core,
                 "library_job": f"{model_name(subject, 0, condition)}_{order}_int8_{core}"}
                for order in mcu.ORDERS
                for core in cores
            )  # fmt: skip
    return jobs


def create_plan(t: dict) -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Supply the int8 kernel CMSIS-NN lacks for EEGNet's temporal convolution (an FIR filter bank) and "
        "count what it saves on Cortex-M0, M4 and M55, layer by layer, with the logits unchanged bit for bit.",
        claim_boundary="Exploratory, planned after the test results were known. Instruction counts from QEMU, not "
        "cycles, time or energy; memory wait states are not modelled. Same quantization, inputs and other kernels as "
        "scripts/mcu_study.py. The frozen configuration does not change.",
        emulation={
            "windows": f"the first {N_WINDOWS} session-2 windows, as scripts/mcu_study.py",
            "all_cores_for": "participant 1, seed 0",
            "checked_conditions": list(CHECKED),
            "check_cores": list(CHECK_CORES),
            "all_cores": list(ALL_CORES),
            "stack_reserved_bytes": STACK_BYTES,
        },
        kernel="mcu/kernels/fir_s8.c: per input row, zero-point padding, exact int32 sums, arm_nn_requantize, "
        "out_offset, int8 clamp; Helium: VMLADAVA on 16-tap int8 vectors with the taps held in registers; DSP: SMLAD "
        "on int16 pairs, two outputs per pass; plain C: two outputs per tap load",
        cores={k: {"flags": v["flags"], "board": v["board"]} for k, v in mcu_build.CORES.items()},
        library_flags=mcu_build.LIBRARY_FLAGS,
        program_flags=mcu_build.PROGRAM_FLAGS,
        tools=mcu_build.versions(t),
    )
    print(f"{len(plan['jobs'])} emulation jobs")


def prepared(subject: int, seed: int, condition: str, order: str):
    """As scripts/mcu_study.py: the int8 network calibrated on session 1, and the session-2 windows."""
    model, scaler, bundle = load_bundle(MODELS_DIR / "final" / f"{model_name(subject, seed, condition)}.pt")
    s1 = data.load_session(subject, 1)
    calibration = scaler.transform(s1["X"][~s1["artifact"]][:, bundle["channels"]])
    net = mcu.export(model, order)
    qnet = mcu.quantize(net, mcu.calibrate(net, calibration))
    s2 = data.load_session(subject, 2, final_evaluation=True)
    return net, qnet, scaler.transform(s2["X"][:, bundle["channels"]])


def emulate_job(job: dict, t: dict, build_dir: Path, plugin: Path) -> None:
    net, qnet, X2 = prepared(job["subject"], job["seed"], job["condition"], job["order"])
    windows = mcu.quantize_input(X2[:N_WINDOWS], qnet)
    expected = mcu.int_forward(qnet, windows)
    rows = mcu.shapes(net, data.N_TIMES)
    out = build_dir / job["name"]
    mcu.write_headers(out / "gen", qnet, rows, windows, "int8")
    mcu_kernels.write_header(out / "gen", qnet)
    board = mcu_build.CORES[job["core"]]["board"]
    start = time.perf_counter()
    elf = mcu_kernels.link(t, job["core"], out / "gen", out / "probe", board, 0, STACK_BYTES)
    result = mcu_build.emulate(t, elf, board, plugin)
    scratch = 0
    if result.get("scratch_too_small"):
        scratch = result["scratch_needed"]
        elf = mcu_kernels.link(t, job["core"], out / "gen", out / "final", board, scratch, STACK_BYTES)
        result = mcu_build.emulate(t, elf, board, plugin)
    if result["exit_status"] != 0 or len(result["outputs"]) != N_WINDOWS or result.get("fault"):
        raise mcu_build.BuildError(f"{job['name']}: emulation failed ({result['exit_status']})")
    got = np.array([np.frombuffer(result["outputs"][i], dtype=np.int8) for i in range(N_WINDOWS)])
    regions = np.array(result["regions"]).reshape(N_WINDOWS, len(mcu.LAYERS))
    sizes = mcu_build.sections(t, elf)
    write_json(STUDY.record("emulate", job), {
        **job, "board": board, "scratch_bytes": scratch, "stack_used": result["stack_used"], "sections": sizes,
        "flash_bytes": sum(sizes.get(s, 0) for s in mcu_build.FLASH_SECTIONS),
        "ram_bytes": sizes.get(".data", 0) + sizes.get(".bss", 0) + result["stack_used"],
        "instructions": {layer: regions[:, i].tolist() for i, layer in enumerate(mcu.LAYERS)},
        "instructions_per_window": regions.sum(1).tolist(), "macs": {r["name"]: r["macs"] for r in rows},
        "bit_exact": bool(np.array_equal(got, expected)), "emulated": got.tolist(),
        "build_and_run_s": time.perf_counter() - start, "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "emulate"], required=True)
    parser.add_argument("--toolchain", type=Path, help="bin directory of the Arm GNU toolchain (default: PATH)")
    parser.add_argument("--cmsis-nn", type=Path, help="CMSIS-NN source tree (the plan records its commit)")
    parser.add_argument("--build-dir", type=Path, default=DATA_DIR / "mcu_build", help="generated code and binaries")
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    args = parser.parse_args()
    t = mcu_build.tools(args.toolchain, args.cmsis_nn)
    if args.stage == "plan":
        create_plan(t)
        return
    plan = STUDY.load()
    if mcu_build.versions(t) != plan["tools"]:
        raise SystemExit(f"Toolchain differs from the plan: {mcu_build.versions(t)} vs {plan['tools']}")
    plugin = mcu_build.plugin(t, args.build_dir)
    pending = STUDY.pending([j for j in plan["jobs"] if j["subject"] in args.subjects], "emulate")
    for i, job in enumerate(pending, 1):
        emulate_job(job, t, args.build_dir, plugin)
        print(f"[{i}/{len(pending)}] {job['name']}", flush=True)


if __name__ == "__main__":
    main()
