"""Post-test study: write the missing INT8 kernel for EEGNet's grouped spatial convolution, then re-measure.

scripts/int8_study.py traced the INT8 slowdown on the Apple M4 to one layer: ONNX Runtime 1.31.1 has
fast INT8 kernels for ordinary and for depthwise convolutions, but EEGNet's spatial convolution (8
groups of 2 filters over all electrodes) is neither and runs on the generic grouped path, 15 times
slower than in FP32. kernels/ort_grouped_conv.c is that missing kernel, an ONNX Runtime custom
operator (NEON widening multiply-accumulates, MLAS's requantization); mibci.ortkernel puts it into
ONNX Runtime's own optimized graph in place of the grouped QLinearConv, so every other node is
unchanged.

Stages:
  plan         the compiler, ONNX Runtime and header versions and the timing settings
  build        compile the operator library
  equivalence  every archived INT8 model (459): logits on all session-2 windows with the kernel
               against ONNX Runtime's INT8 graph (must be bit-identical), and the predictions
               against the archived INT8 predictions
  benchmark    batch-1, single-thread latency on this machine (participant 1, seed 0): FP32, INT8,
               INT8 with the kernel, and the earlier fixes (dense rewrite, swapped layers), in
               interleaved rounds, with ONNX Runtime's per-node profile

Exploratory (the test results were known). Tables: scripts/analyze_ort_kernel_study.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from mibci import data, deploy, ortkernel
from mibci.artifacts import sha256, write_json
from mibci.env import environment
from mibci.model import load_bundle
from mibci.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR
from mibci.study import Study
from mibci.timing import build_variants, power_state

STUDY = Study(
    "ort_kernel_study",
    "scripts/ort_kernel_study.py",
    sources=["kernels/ort_grouped_conv.c", "src/mibci/ortkernel.py", "src/mibci/timing.py", "src/mibci/variants.py",
             "src/mibci/deploy.py", "src/mibci/study.py"],
)  # fmt: skip
HEADERS = ("onnxruntime_c_api.h", "onnxruntime_error_code.h", "onnxruntime_ep_c_api.h")
BENCH = {"subject": 1, "seed": 0, "conditions": ["full-22", "relevance-16", "relevance-8", "relevance-16-prune50"]}
TIMING = {"rounds": 15, "runs": 500, "warmup": 200, "inputs": "the first 100 session-1 training windows"}
EXTRA = ["int8-dense", "fp32-spatial-first", "int8-spatial-first"]


def library_path(build_dir: Path) -> Path:
    return build_dir / ("libmibci_ortkernel.dylib" if platform.system() == "Darwin" else "libmibci_ortkernel.so")


def archived_models() -> list[str]:
    """Stems of the 459 archived final models that have a scored INT8 graph."""
    stems = sorted(p.name.removesuffix(".json") for p in (RESULTS_DIR / "test").glob("sub-*_seed-*_*.json"))
    return [s for s in stems if (MODELS_DIR / "final" / f"{s}_int8.onnx").exists()]


def create_plan(headers: Path) -> None:
    compiler = subprocess.run(["cc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    STUDY.create(
        [{"name": stem} for stem in archived_models()],
        purpose="Write the INT8 kernel that ONNX Runtime lacks for EEGNet's grouped spatial convolution, check that it "
        "gives bit-identical outputs on every archived model, and measure the latency of INT8 with it against FP32.",
        claim_boundary="Exploratory, planned after the test results were known. The kernel covers this layer shape "
        "(groups of 2 filters spanning all electrodes, stride 1, no padding; NEON for 8 groups, plain C otherwise) on "
        "ARM64; it is a custom operator, not a change to ONNX Runtime itself. Latency is measured on one shared Apple "
        "M4 Mac mini, single thread, batch 1, with the load average recorded. Accuracy cannot change: the logits are "
        "checked bit for bit.",
        benchmark=BENCH,
        timing=TIMING,
        compiler=compiler,
        compile_flags=ortkernel.FLAGS,
        onnxruntime=ort.__version__,
        headers={h: sha256(headers / h) for h in HEADERS},
        headers_source="Microsoft's ONNX Runtime C API headers (ORT_API_VERSION 30) as vendored in Mozilla's Firefox "
        "tree, dom/onnx/ (github.com/microsoft/onnxruntime was unreachable on 2026-11-21); the 1.31.1 runtime serves "
        "API version 30",
    )
    print(f"{len(archived_models())} archived INT8 models")


def equivalence_job(stem: str, build_dir: Path, library: Path) -> None:
    out = STUDY.record("equivalence", {"name": stem})
    if out.exists():
        return
    bundle = MODELS_DIR / "final" / f"{stem}.pt"
    _, scaler, b = load_bundle(bundle)
    subject = int(b["subject"])
    s2 = data.load_session(subject, 2, final_evaluation=True)
    X = scaler.transform(s2["X"][:, b["channels"]])
    work = build_dir / "graphs"
    work.mkdir(parents=True, exist_ok=True)
    stored = bundle.with_name(f"{stem}_int8.onnx")
    opt = ortkernel.optimized(stored, work / f"{stem}_int8_optimized.onnx")
    swapped = ortkernel.with_kernel(opt, work / f"{stem}_int8_kernel.onnx")
    reference = deploy.logits(ortkernel.session(stored), X)
    kernel = deploy.logits(ortkernel.session(work / f"{stem}_int8_kernel.onnx", library), X)
    archived = np.array(json.loads((RESULTS_DIR / "test" / f"{stem}.json").read_text())["pred"]["int8"])
    write_json(out, {
        "name": stem, "subject": subject, "condition": b["condition"], "seed": int(b["seed"]), "replaced": swapped,
        "n_windows": len(X), "logits_identical": bool(np.array_equal(reference, kernel)),
        "max_abs_logit_difference": float(np.abs(reference - kernel).max()),
        "predictions_equal_archived_int8": int((kernel.argmax(1) == archived).sum()),
        "reference_equal_archived_int8": int((reference.argmax(1) == archived).sum()),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def benchmark(build_dir: Path, library: Path) -> None:
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
        inputs = train_inputs[:100]
        fp32, int8 = bundle.with_suffix(".onnx"), bundle.with_name(f"{stem}_int8.onnx")
        opt = ortkernel.optimized(int8, work / f"{stem}_int8_optimized.onnx")
        ortkernel.with_kernel(opt, work / f"{stem}_int8_kernel.onnx")
        paths = {"fp32": fp32, "int8": int8, "int8-kernel": work / f"{stem}_int8_kernel.onnx",
                 **build_variants(fp32, work, stem, train_inputs, EXTRA)}  # fmt: skip
        paths["fp32-spatial-first"] = work / f"{stem}__fp32-spatial-first.onnx"
        libraries = {name: library if name == "int8-kernel" else None for name in paths}
        sessions = {name: ortkernel.session(path, libraries[name]) for name, path in paths.items()}
        timing = ortkernel.timed(sessions, inputs, TIMING["rounds"], TIMING["runs"], TIMING["warmup"])
        for name, path in paths.items():
            rows.append({"condition": condition, "variant": name, "electrodes": len(b["channels"]), "F2": b["F2"],
                         **timing[name], **deploy.size_bytes(path)})  # fmt: skip
            profiles[f"{condition} {name}"] = ortkernel.profile(path, libraries[name], inputs)
        ref = timing["fp32"]["median_ms"]
        shown = {n: f"{t['median_ms'] * 1000:.0f} us ({t['median_ms'] / ref:.2f}x)" for n, t in timing.items()}
        print(condition, shown, flush=True)
    write_json(out, {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "status": "supplementary_post_test",
        "environment": {**environment(), "onnxruntime": ort.__version__, "machine": platform.machine(),
                        "power": power_state(), "ort_threads": {"intra_op": 1, "inter_op": 1},
                        "load_average_before": load_before, "load_average_after": os.getloadavg()},
        "method": "Sessions for all variants of one model are created together and timed in interleaved rounds that "
        "alternate direction; the value is the median of the per-round medians.",
        "library_sha256": sha256(library), "rows": rows, "op_profile_us": profiles,
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


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
    library = library_path(args.build_dir)
    if args.stage == "build":
        found = {h: sha256(args.ort_headers / h) for h in HEADERS}
        if found != plan["headers"]:
            raise SystemExit("These ONNX Runtime headers differ from the ones recorded in the plan")
        ortkernel.build(args.ort_headers, library)
        print(f"built {library} ({sha256(library)[:12]})")
        return
    if not library.exists():
        raise SystemExit(f"{library} missing: run --stage build first")
    if args.stage == "equivalence":
        jobs = plan["jobs"]
        for i, job in enumerate(jobs, 1):
            equivalence_job(job["name"], args.build_dir, library)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)
    else:
        benchmark(args.build_dir, library)


if __name__ == "__main__":
    main()
