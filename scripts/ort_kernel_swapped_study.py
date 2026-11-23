"""Post-test study: the second missing INT8 kernel (the swapped layer order), and both kernels timed together.

scripts/ort_kernel_study.py wrote the INT8 kernel ONNX Runtime lacks for the original order's grouped
spatial convolution. Its benchmark showed a second gap of the same kind: with the first two layers
swapped, the INT8 graph starts with a spatial convolution (one input channel, kernel over all
electrodes) that ONNX Runtime 1.31.1 runs slower than in FP32 (25 against 7 microseconds).
kernels/ort_spatial_first_conv.c implements that layer (mibci.ortkernel_swapped).

Stages:
  plan         versions and timing settings
  build        compile both operator libraries (this kernel and kernels/ort_grouped_conv.c)
  equivalence  every archived model (459): the swapped INT8 graph rebuilt as scripts/int8_study.py builds
               it, its logits on all session-2 windows with the kernel against ONNX Runtime's (must be
               bit-identical), and its predictions against the int8_study records
  benchmark    batch-1, single-thread latency (participant 1, seed 0) of FP32, INT8 and both kernels in
               both layer orders, in interleaved rounds, with ONNX Runtime's per-node profile

Exploratory (the test results were known). Tables: scripts/analyze_ort_kernel_study.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from mibci import data, deploy, ortkernel, ortkernel_swapped
from mibci.artifacts import sha256, write_json
from mibci.env import environment
from mibci.experiment import session1
from mibci.model import load_bundle
from mibci.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR
from mibci.study import Study
from mibci.timing import build_variants, power_state

STUDY = Study(
    "ort_kernel_swapped_study",
    "scripts/ort_kernel_swapped_study.py",
    sources=[
        "kernels/ort_spatial_first_conv.c",
        "kernels/ort_grouped_conv.c",
        "src/mibci/ortkernel_swapped.py",
        "src/mibci/ortkernel.py",
        "src/mibci/timing.py",
        "src/mibci/variants.py",
        "src/mibci/deploy.py",
        "src/mibci/study.py",
    ],
)
HEADERS = ("onnxruntime_c_api.h", "onnxruntime_error_code.h", "onnxruntime_ep_c_api.h")
INT8_RECORDS = RESULTS_DIR / "supplementary" / "int8_study" / "models"
BENCH = {"subject": 1, "seed": 0, "conditions": ["full-22", "relevance-16", "relevance-8", "relevance-16-prune50"]}
TIMING = {"rounds": 15, "runs": 500, "warmup": 200, "inputs": "the first 100 session-1 training windows"}


def libraries(build_dir: Path) -> dict[str, Path]:
    ext = "dylib" if platform.system() == "Darwin" else "so"
    return {
        "grouped": build_dir / f"libmibci_ortkernel.{ext}",
        "swapped": build_dir / f"libmibci_ortkernel_swapped.{ext}",
    }


def archived_models() -> list[str]:
    stems = sorted(p.name.removesuffix(".json") for p in (RESULTS_DIR / "test").glob("sub-*_seed-*_*.json"))
    return [s for s in stems if (INT8_RECORDS / f"{s}.json").exists()]


def create_plan(headers: Path) -> None:
    STUDY.create(
        [{"name": stem} for stem in archived_models()],
        purpose="Write the second INT8 kernel ONNX Runtime lacks (the swapped order's spatial convolution), check it "
        "bit for bit on every archived model, and time FP32, INT8 and both kernels in both layer orders together.",
        claim_boundary="Exploratory, planned after the test results were known. The kernel covers this layer shape "
        "(one input channel, kernel over all electrodes, stride 1, no padding; NEON for 16 filters, plain C "
        "otherwise) on ARM64, as a custom operator, not a change to ONNX Runtime. Latency is one shared Apple M4 Mac "
        "mini, single thread, batch 1, with the load average recorded. Accuracy cannot change: logits are checked "
        "bit for bit.",
        benchmark=BENCH,
        timing=TIMING,
        compile_flags=ortkernel.FLAGS,
        onnxruntime=ort.__version__,
        headers={h: sha256(headers / h) for h in HEADERS},
    )
    print(f"{len(archived_models())} archived models")


def equivalence_job(stem: str, build_dir: Path, library: Path) -> None:
    out = STUDY.record("equivalence", {"name": stem})
    if out.exists():
        return
    bundle = MODELS_DIR / "final" / f"{stem}.pt"
    _, scaler, b = load_bundle(bundle)
    subject = int(b["subject"])
    X1, _, train, _ = session1(subject, "final")
    s2 = data.load_session(subject, 2, final_evaluation=True)
    X = scaler.transform(s2["X"][:, b["channels"]])
    with tempfile.TemporaryDirectory(dir=build_dir) as tmp:
        work = Path(tmp)
        graph = build_variants(
            bundle.with_suffix(".onnx"),
            work,
            stem,
            scaler.transform(X1[train][:, b["channels"]]),
            ["int8-spatial-first"],
        )["int8-spatial-first"]
        opt = ortkernel.optimized(graph, work / "optimized.onnx")
        swapped = ortkernel_swapped.with_kernel(opt, work / "kernel.onnx")
        reference = deploy.logits(ortkernel.session(graph), X)
        kernel = deploy.logits(ortkernel.session(work / "kernel.onnx", library), X)
    archived = np.array(
        json.loads((INT8_RECORDS / f"{stem}.json").read_text())["variants"]["int8-spatial-first"]["pred"]
    )
    write_json(
        out,
        {
            "name": stem,
            "subject": subject,
            "condition": b["condition"],
            "seed": int(b["seed"]),
            "replaced": swapped,
            "n_windows": len(X),
            "logits_identical": bool(np.array_equal(reference, kernel)),
            "max_abs_logit_difference": float(np.abs(reference - kernel).max()),
            "predictions_equal_archived": int((kernel.argmax(1) == archived).sum()),
            "rebuilt_equal_archived": int((reference.argmax(1) == archived).sum()),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def benchmark(build_dir: Path, libs: dict[str, Path]) -> None:
    out = STUDY.out / "benchmark.json"
    if out.exists():
        raise SystemExit(f"{out} exists. Keep the measurement; move it aside deliberately to measure again.")
    s1 = data.load_session(BENCH["subject"], 1)
    clean = ~s1["artifact"]
    load_before = os.getloadavg()
    rows, profiles = [], {}
    work = build_dir / "bench"
    work.mkdir(parents=True, exist_ok=True)
    for condition in BENCH["conditions"]:
        stem = f"sub-{BENCH['subject']:02d}_seed-{BENCH['seed']}_{condition}"
        bundle = MODELS_DIR / "final" / f"{stem}.pt"
        _, scaler, b = load_bundle(bundle)
        train_inputs = scaler.transform(s1["X"][clean][:, b["channels"]])
        fp32, int8 = bundle.with_suffix(".onnx"), bundle.with_name(f"{stem}_int8.onnx")
        made = build_variants(fp32, work, stem, train_inputs, ["fp32-spatial-first", "int8-spatial-first"])
        ortkernel.with_kernel(ortkernel.optimized(int8, work / f"{stem}_o.onnx"), work / f"{stem}_k1.onnx")
        ortkernel_swapped.with_kernel(
            ortkernel.optimized(made["int8-spatial-first"], work / f"{stem}_so.onnx"), work / f"{stem}_k2.onnx"
        )
        paths = {
            "fp32": fp32,
            "int8": int8,
            "int8-kernel": work / f"{stem}_k1.onnx",
            "fp32-spatial-first": made["fp32-spatial-first"],
            "int8-spatial-first": made["int8-spatial-first"],
            "int8-spatial-first-kernel": work / f"{stem}_k2.onnx",
        }
        lib = {"int8-kernel": libs["grouped"], "int8-spatial-first-kernel": libs["swapped"]}
        sessions = {name: ortkernel.session(path, lib.get(name)) for name, path in paths.items()}
        timing = ortkernel.timed(sessions, train_inputs[:100], TIMING["rounds"], TIMING["runs"], TIMING["warmup"])
        for name, path in paths.items():
            rows.append(
                {
                    "condition": condition,
                    "variant": name,
                    "electrodes": len(b["channels"]),
                    "F2": b["F2"],
                    **timing[name],
                    **deploy.size_bytes(path),
                }
            )
            profiles[f"{condition} {name}"] = ortkernel.profile(path, lib.get(name), train_inputs[:100])
        ref = timing["fp32"]["median_ms"]
        shown = {n: f"{t['median_ms'] * 1000:.0f} us ({t['median_ms'] / ref:.2f}x)" for n, t in timing.items()}
        print(condition, shown, flush=True)
    write_json(
        out,
        {
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "status": "supplementary_post_test",
            "environment": {
                **environment(),
                "onnxruntime": ort.__version__,
                "machine": platform.machine(),
                "power": power_state(),
                "ort_threads": {"intra_op": 1, "inter_op": 1},
                "load_average_before": load_before,
                "load_average_after": os.getloadavg(),
            },
            "method": "Sessions for all variants of one model are created together and timed in interleaved rounds "
            "that alternate direction; the value is the median of the per-round medians.",
            "libraries_sha256": {k: sha256(v) for k, v in libs.items()},
            "rows": rows,
            "op_profile_us": profiles,
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "build", "equivalence", "benchmark"], required=True)
    parser.add_argument("--ort-headers", type=Path, help="directory with the ONNX Runtime C API headers")
    parser.add_argument("--build-dir", type=Path, default=DATA_DIR / "ort_kernel_build")
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan(args.ort_headers)
        return
    plan = STUDY.load()
    libs = libraries(args.build_dir)
    if args.stage == "build":
        if {h: sha256(args.ort_headers / h) for h in HEADERS} != plan["headers"]:
            raise SystemExit("These ONNX Runtime headers differ from the ones recorded in the plan")
        ortkernel.build(args.ort_headers, libs["grouped"])
        ortkernel_swapped.build(args.ort_headers, libs["swapped"])
        print({k: sha256(v)[:12] for k, v in libs.items()})
        return
    if not all(p.exists() for p in libs.values()):
        raise SystemExit("operator libraries missing: run --stage build first")
    if args.stage == "equivalence":
        for i, job in enumerate(plan["jobs"], 1):
            equivalence_job(job["name"], args.build_dir, libs["swapped"])
            if i % 50 == 0 or i == len(plan["jobs"]):
                print(f"[{i}/{len(plan['jobs'])}]", flush=True)
    else:
        benchmark(args.build_dir, libs)


if __name__ == "__main__":
    main()
