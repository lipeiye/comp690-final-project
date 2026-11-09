"""Measure isolated ONNX Runtime process RSS, keeping it distinct from tensor estimates.

The parent prepares normalized session-1 inputs. Each precision runs in a fresh
child that never imports PyTorch. Peak RSS includes interpreter, ORT, allocator,
graph initialization and input arrays; it is not per-inference tensor memory or MCU SRAM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def child(model,inputs,runs):
    import resource
    import numpy as np
    import onnxruntime as ort
    import psutil
    process=psutil.Process()
    baseline=process.memory_info().rss
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(model),opts,providers=["CPUExecutionProvider"])
    after_session=process.memory_info().rss
    array=np.load(inputs)
    name=session.get_inputs()[0].name
    for i in range(100+runs):
        session.run(None,{name:array[i%len(array):i%len(array)+1]})
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"baseline_rss_bytes":baseline,"after_session_rss_bytes":after_session,
        "after_inference_rss_bytes":process.memory_info().rss,
        "process_lifetime_peak_rss_bytes":int(peak*(1 if platform.system()=="Darwin" else 1024)),
        "runtime_input_type":session.get_inputs()[0].type,"input_array_bytes":array.nbytes,
        "warmup":100,"runs":runs,"onnxruntime":ort.__version__}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--runs",type=int,default=1000)
    parser.add_argument("--child-model",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--child-input",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.runs<1:parser.error("runs must be positive")
    if args.child_model:
        child(args.child_model,args.child_input,args.runs);return
    if not args.bundle or not args.output:parser.error("--bundle and --output are required")
    from mibci.artifacts import model_exports,require_files,sha256,write_json
    from mibci.model import load_bundle
    from mibci.experiment import session1
    import numpy as np
    paths=model_exports(args.bundle);require_files([args.bundle,*paths],"process memory benchmark")
    if args.output.exists():raise SystemExit("Output exists; choose a new file to retain previous measurements")
    _,scaler,b=load_bundle(args.bundle)
    X,_,tr,_=session1(b["subject"],"final")
    inputs=scaler.transform(X[tr][:32,b["channels"]])
    rows={}
    with tempfile.TemporaryDirectory() as temp:
        input_path=Path(temp)/"inputs.npy";np.save(input_path,inputs)
        for precision,path in zip(["fp32","int8"],paths):
            run=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--child-model",str(path),
                "--child-input",str(input_path),"--runs",str(args.runs)],capture_output=True,text=True,check=True)
            rows[precision]={"model_sha256":sha256(path),**json.loads(run.stdout)}
    write_json(args.output,{"bundle":str(args.bundle.resolve()),"bundle_sha256":sha256(args.bundle),
        "platform":platform.platform(),"rows":rows,
        "definition":"Process RSS in separate children without PyTorch; includes initialization and runtime overhead. Peak is lifetime high-water mark, not incremental inference memory.",
        "limits":"Not MCU SRAM, tensor liveness, allocation tracing, or a proof of a 4x activation-memory reduction."})
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
