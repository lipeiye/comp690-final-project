"""Supplementary, post-test matched-budget pruning experiment.

Plan first, then train/evaluate/analyze. Original evidence is read-only. Every
base is branched into no pruning, 25% pruning and 50% pruning, each given the same
100 epochs with a fresh Adam optimizer at 1e-4 and the same seed/batch schedule.
With --rebuild-bases the 475-epoch bases are independently reproduced from the
historical channel lists; these are never presented as restored original weights.
"""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from datetime import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np
import torch

from mibci import data, deploy
from mibci.artifacts import completed, model_exports, require_files, sha256, write_json
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, device_for, run_pool, session1, train_run
from mibci.model import load_bundle, save_bundle
from mibci.prune import FINETUNE_EPOCHS, FINETUNE_LR, prune_pointwise
from mibci.train import TrainConfig, fit, predict, scores, set_seed

OUT = RESULTS_DIR / "supplementary/matched_budget"
MODELS = MODELS_DIR / "supplementary/matched_budget"
CORE = ["src/mibci/data.py", "src/mibci/model.py", "src/mibci/train.py", "src/mibci/prune.py",
        "src/mibci/experiment.py", "src/mibci/deploy.py", "scripts/run_matched_control.py"]


def load_plan() -> dict:
    plan = json.loads((OUT / "protocol.json").read_text())
    if sha256(RESULTS_DIR / "frozen_config.json") != plan["historical_frozen_config_sha256"]:
        raise RuntimeError("Historical frozen configuration changed since supplementary plan")
    for relative, digest in plan["source_code_sha256"].items():
        if sha256(data.PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"Planned training/evaluation source changed: {relative}; start a new supplementary directory")
    for relative, digest in plan["ranking_sha256"].items():
        if sha256(RESULTS_DIR / relative) != digest:
            raise RuntimeError(f"Historical electrode ranking changed: {relative}")
    return plan


def create_plan(rebuild: bool) -> None:
    if (OUT / "protocol.json").exists():
        plan = load_plan()
        if plan["base_origin"] != ("independent_reproduction" if rebuild else "original_bundles"):
            raise RuntimeError("Existing protocol has a different base origin")
        print("Existing supplementary protocol retained.")
        return
    frozen_path = RESULTS_DIR / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text())
    inputs = [RESULTS_DIR / f"final/rankings/sub-{subject:02d}_seed-{seed}.json"
              for subject in data.SUBJECTS for seed in frozen["seeds"]]
    inputs += [RESULTS_DIR / f"final/runs/sub-{subject:02d}_seed-{seed}_{base}.json"
               for subject in data.SUBJECTS for seed in frozen["seeds"] for base in frozen["pruning"]["bases"]]
    with ThreadPoolExecutor(max_workers=12) as reader:
        input_contents = dict(zip(inputs, reader.map(lambda p: p.read_bytes(), inputs)))
    rows, ranking_hashes = [], {}
    for subject in data.SUBJECTS:
        for seed in frozen["seeds"]:
            relative = f"final/rankings/sub-{subject:02d}_seed-{seed}.json"
            ranking_path = RESULTS_DIR / relative
            ranking_hashes[relative] = hashlib.sha256(input_contents[ranking_path]).hexdigest()
            subsets = json.loads(input_contents[ranking_path])["subsets"]
            for base in frozen["pruning"]["bases"]:
                name = f"sub-{subject:02d}_seed-{seed}_{base}"
                bundle = (MODELS / "bases" if rebuild else MODELS_DIR / "final") / f"{name}.pt"
                original_record = RESULTS_DIR / "final/runs" / f"{name}.json"
                metadata = json.loads(input_contents[original_record])
                original_channels = [data.EEG_CHANNELS.index(c) for c in metadata["channels"]]
                if subsets[base] != original_channels:
                    raise RuntimeError(f"Ranking and original run channel lists disagree: {name}")
                rows.append({"name":name, "subject":subject, "seed":seed, "condition":base,
                             "channels":subsets[base], "base_bundle":str(bundle),
                             "historical_run_sha256":hashlib.sha256(input_contents[original_record]).hexdigest(),
                             "original_base_sha256":sha256(bundle) if not rebuild else None})
    if not rebuild:
        require_files([r["base_bundle"] for r in rows], "supplementary original bases")
    plan = {"created_at":datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "status":"supplementary_post_test", "base_origin":"independent_reproduction" if rebuild else "original_bundles",
            "historical_frozen_config_sha256":sha256(frozen_path),
            "historical_test_results_already_known":True,
            "purpose":"Separate extra optimization from the effect of removing final pointwise filters",
            "selection":"Both historical pruning bases, all nine participants and all three historical seeds; no new selection",
            "base_epochs":frozen["epochs"], "finetune_epochs":FINETUNE_EPOCHS, "finetune_lr":FINETUNE_LR,
            "batch_size":64, "optimizer":"fresh Adam for every branch", "fractions":[0.0,0.25,0.5],
            "primary":"artifact-free session-2 trials; average seeds within participant",
            "secondary":"all session-2 trials", "precision":["fp32","int8"],
            "bootstrap":{"resamples":10000,"seed":690,"level":0.95},
            "claim_boundary":"Does not change the historical finalist or turn a post-test analysis into prospective evidence",
            "source_code_sha256":{p:sha256(data.PROJECT_ROOT / p) for p in CORE},
            "ranking_sha256":ranking_hashes, "jobs":rows}
    write_json(OUT / "protocol.json", plan)
    print(f"Planned {len(rows)} bases and {3*len(rows)} matched-budget branches.")


def train_base(job: dict, epochs: int) -> None:
    record_path = OUT / "bases" / f"{job['name']}.json"
    path = Path(job["base_bundle"])
    if completed(record_path,[path],{"subject":job["subject"],"seed":job["seed"],"epochs":epochs}):
        return
    record = train_run(job["subject"], "final", job["condition"], job["channels"],
                       TrainConfig(epochs=epochs,seed=job["seed"]), path)
    record.update(base_origin="independent_reproduction", protocol_sha256=sha256(OUT / "protocol.json"),
                  model_sha256=sha256(path))
    write_json(record_path,record)


def branch_name(job: dict, fraction: float) -> str:
    return f"{job['name']}-matched{round(100*fraction)}"


def train_branch(job: dict, fraction: float, plan: dict) -> None:
    base_path = Path(job["base_bundle"])
    model,scaler,bundle = load_bundle(base_path)
    if bundle["channels"] != job["channels"] or bundle["seed"] != job["seed"] or bundle["epochs"] != plan["base_epochs"]:
        raise RuntimeError(f"Base metadata differs from protocol: {base_path}")
    if plan["base_origin"] == "original_bundles" and sha256(base_path) != job["original_base_sha256"]:
        raise RuntimeError(f"Original base changed since planning: {base_path}")
    # Preserve the original pruning procedure: construct the branch before resetting its seed.
    branch = prune_pointwise(model,fraction) if fraction else model
    X,y,tr,_ = session1(job["subject"],"final")
    X = scaler.transform(X[:,job["channels"]])
    set_seed(job["seed"])
    cfg = TrainConfig(epochs=plan["finetune_epochs"],lr=plan["finetune_lr"],seed=job["seed"],batch_size=plan["batch_size"])
    training = fit(branch,X[tr],y[tr],cfg,device=device_for(job["subject"]))
    name = branch_name(job,fraction)
    path = MODELS / "branches" / f"{name}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    save_bundle(path,branch,job["channels"],scaler,subject=job["subject"],seed=job["seed"],split="final",
                condition=job["condition"],epochs=plan["base_epochs"],fraction=fraction,
                finetune_epochs=cfg.epochs,base_sha256=sha256(base_path),base_origin=plan["base_origin"])
    write_json(OUT / "runs" / f"{name}.json", {**job,"fraction":fraction,"base_origin":plan["base_origin"],
               "finetune_epochs":cfg.epochs,"finetune_lr":cfg.lr,"train_time_s":training["train_time_s"],
               "device":device_for(job["subject"]),"model_sha256":sha256(path),
               "base_sha256":sha256(base_path),"protocol_sha256":sha256(OUT / "protocol.json")})


def evaluate(job: dict, bundle_path: str, label: str) -> None:
    path=Path(bundle_path)
    model,scaler,b=load_bundle(path)
    X,y,tr,_ = session1(job["subject"],"final")
    calibration = scaler.transform(X[tr][:,b["channels"]])
    fp32,int8=model_exports(path)
    deploy.export_onnx(model,fp32)
    deploy.quantize_int8(fp32,int8,calibration)
    # Export verification uses session 1, not test labels, and has an explicit failure threshold.
    reference=predict(model,calibration[:32])
    exported=deploy.logits(deploy.session(fp32),calibration[:32])
    error=float(np.abs(reference-exported).max())
    if not np.allclose(reference,exported,atol=1e-5,rtol=1e-4):
        raise RuntimeError(f"FP32 export failed tolerance for {path}: {error}")
    ses=data.load_session(job["subject"],2,final_evaluation=True)
    inputs=scaler.transform(ses["X"][:,b["channels"]])
    clean=~ses["artifact"]
    outputs={p:deploy.logits(deploy.session(q),inputs) for p,q in [("fp32",fp32),("int8",int8)]}
    historical_condition = job["condition"] if label == "base" else (
        f"{job['condition']}-prune{label}" if label in ["25","50"] else None)
    historical_path=RESULTS_DIR / "test" / f"sub-{job['subject']:02d}_seed-{job['seed']}_{historical_condition}.json"
    historical=json.loads(historical_path.read_text()) if historical_condition and historical_path.exists() else None
    write_json(OUT / "test" / f"{path.stem}.json",{**job,"label":label,
        "base_origin":load_plan()["base_origin"],"model_sha256":sha256(path),
        "protocol_sha256":sha256(OUT / "protocol.json"),"export_max_abs_diff":error,
        "primary":{p:scores(lg[clean],ses["y"][clean]) for p,lg in outputs.items()},
        "secondary":{p:scores(lg,ses["y"]) for p,lg in outputs.items()},
        "pred":{p:lg.argmax(1).tolist() for p,lg in outputs.items()},
        "y":ses["y"].tolist(),"artifact":ses["artifact"].astype(int).tolist(),
        "historical_prediction_agreement":{p:float((lg.argmax(1)==np.array(historical["pred"][p])).mean())
                                           for p,lg in outputs.items()} if historical else None,
        "historical_prediction_agreement_note":"Matching predictions do not prove identical weights.",
        "size":{p:deploy.size_bytes(q) for p,q in [("fp32",fp32),("int8",int8)]}})


def analyze(plan):
    rows=[]
    for job in plan["jobs"]:
        for label in ["base","0","25","50"]:
            name=job["name"] if label=="base" else branch_name(job,float(label)/100)
            path=OUT / "test" / f"{name}.json"
            require_files([path],"matched analysis complete grid")
            r=json.loads(path.read_text())
            if r["protocol_sha256"] != sha256(OUT / "protocol.json"):
                raise RuntimeError(f"Result uses another protocol: {path}")
            for precision in ["fp32","int8"]:
                rows.append({"subject":job["subject"],"seed":job["seed"],"base":job["condition"],
                             "branch":label,"precision":precision,"accuracy":r["primary"][precision]["acc"]})
    groups=collections.defaultdict(list)
    for r in rows:
        groups[(r["base"],r["branch"],r["precision"],r["subject"])].append(r["accuracy"])
    means={k:statistics.mean(v) for k,v in groups.items()}
    comparisons=[]
    rng=np.random.default_rng(plan["bootstrap"]["seed"])
    for base in sorted({j["condition"] for j in plan["jobs"]}):
        for precision in ["fp32","int8"]:
            for a,b in [("0","base"),("25","0"),("50","0"),("25","base"),("50","base")]:
                d=np.array([means[(base,a,precision,s)]-means[(base,b,precision,s)] for s in data.SUBJECTS])
                interval=np.quantile(d[rng.integers(0,9,(plan["bootstrap"]["resamples"],9))].mean(1),[.025,.975])
                comparisons.append({"base":base,"precision":precision,"a":a,"b":b,
                    "difference_pp":100*d.mean(),"ci_low_pp":100*interval[0],"ci_high_pp":100*interval[1]})
    for name,values in [("seed_accuracy.csv",rows),("paired_comparisons.csv",comparisons)]:
        with (OUT / name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_json(OUT / "analysis.json",{"base_origin":plan["base_origin"],"status":"supplementary_post_test",
               "protocol_sha256":sha256(OUT / "protocol.json"),"comparisons":comparisons,
               "note":"0 vs base = extra-training effect; 25/50 vs 0 = matched-budget pruning effect. No finalist reselection."})
    print("Matched-budget analysis complete.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["plan","train","evaluate","analyze"],required=True)
    parser.add_argument("--rebuild-bases",action="store_true",help="Plan an independent reproduction when original weights are unavailable")
    parser.add_argument("--subjects",type=int,nargs="+",default=data.SUBJECTS)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if not set(args.subjects)<=set(data.SUBJECTS):
        parser.error("subjects must be in 1..9")
    if args.stage=="plan":
        create_plan(args.rebuild_bases);return
    plan=load_plan();selected=[j for j in plan["jobs"] if j["subject"] in args.subjects]
    if args.stage=="train":
        write_json(OUT / "environments" / ("subjects-"+"-".join(map(str,args.subjects))+".json"),
                   {**environment(),"device_assignment":{str(s):device_for(s) for s in args.subjects}})
        if plan["base_origin"]=="independent_reproduction":
            jobs=[(j["name"],{"job":j,"epochs":plan["base_epochs"]}) for j in selected
                  if not completed(OUT / "bases" / f"{j['name']}.json",[Path(j["base_bundle"])])]
            run_pool(train_base,jobs,args.workers,1)
        require_files([j["base_bundle"] for j in selected],"matched branch inputs")
        jobs=[]
        for j in selected:
            for f in plan["fractions"]:
                name=branch_name(j,f)
                if not completed(OUT / "runs" / f"{name}.json",[MODELS / "branches" / f"{name}.pt"],
                                 {"fraction":f,"finetune_epochs":plan["finetune_epochs"]}):
                    jobs.append((name,{"job":j,"fraction":f,"plan":plan}))
        run_pool(train_branch,jobs,args.workers,1)
    elif args.stage=="evaluate":
        jobs=[]
        for j in selected:
            for label in ["base","0","25","50"]:
                path=Path(j["base_bundle"]) if label=="base" else MODELS / "branches" / f"{branch_name(j,float(label)/100)}.pt"
                require_files([path],"matched evaluation inputs")
                if not completed(OUT / "test" / f"{path.stem}.json",[path,*model_exports(path)]):
                    jobs.append((path.stem,{"job":j,"bundle_path":str(path),"label":label}))
        run_pool(evaluate,jobs,args.workers,1)
    else:
        analyze(plan)


if __name__=="__main__":
    main()
