"""Portable inference from a model bundle and cached, causally filtered EEG windows.

Input NPZ must contain X: trials x 22 electrodes x 500 samples, microvolts, after
the project's causal filtering/windowing. This tool applies only the stored channel
selection and training normalization. It does not treat arbitrary raw samples as
already preprocessed, use labels, fit a scaler, or demonstrate online closed-loop use.
"""

import argparse
from pathlib import Path

import numpy as np

from mibci import data,deploy
from mibci.artifacts import model_exports,require_files,sha256,write_json
from mibci.model import load_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--input-npz",type=Path,required=True)
    parser.add_argument("--precision",choices=["fp32","int8"],default="int8")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    onnx=model_exports(args.bundle)[0 if args.precision=="fp32" else 1]
    require_files([args.bundle,onnx,args.input_npz],"portable inference")
    if args.output.exists():raise SystemExit("Output exists; choose a new predictions file")
    _,scaler,b=load_bundle(args.bundle)
    with np.load(args.input_npz) as inputs:X=inputs["X"]
    if X.ndim!=3 or X.shape[1:]!=(22,500) or not len(X) or not np.isfinite(X).all():
        raise ValueError("Expected finite, causally filtered X with shape (trials,22,500)")
    logits=deploy.logits(deploy.session(onnx),scaler.transform(X[:,b["channels"]]))
    pred=logits.argmax(1)
    write_json(args.output,{"precision":args.precision,"subject":b.get("subject"),"seed":b.get("seed"),
        "base_origin":b.get("base_origin","not embedded; consult the supplied training record/protocol"),
        "bundle_sha256":sha256(args.bundle),"onnx_sha256":sha256(onnx),"input_sha256":sha256(args.input_npz),
        "channels":[data.EEG_CHANNELS[c] for c in b["channels"]],
        "pred":pred.tolist(),"labels":[data.CLASSES[p] for p in pred],"logits":logits.tolist(),
        "input_contract":"Project-preprocessed EEG windows in microvolts; no test-label access or fitted normalization"})
    print(f"Wrote {len(pred)} predictions to {args.output}")


if __name__=="__main__":main()
