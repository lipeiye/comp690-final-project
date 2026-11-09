"""Finish the already-authorized supplementary run and record every remaining stage.

This is an immediate local pipeline, not a scheduled job. It waits for the existing
training processes, exports/evaluates the complete fixed grid, validates stored
predictions, measures process RSS and prepares report files for final visual QA.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/supplementary/matched_budget"
MODELS=ROOT/"data.nosync/models/supplementary/matched_budget"
EVIDENCE=ROOT/"docs/evidence"
STATUS=EVIDENCE/"completion_status.json"


def status(phase,**extra):
    value={"at":datetime.now(ZoneInfo("America/New_York")).isoformat(),"phase":phase,**extra}
    temp=STATUS.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(STATUS)
    print(json.dumps(value),flush=True)


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def run(name,command):
    status(name,command=command)
    with (EVIDENCE/(name+".log")).open("w") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith(("[","Wrote","Matched","scored")):
                print(line.rstrip(),flush=True)
        code=process.wait()
    if code:raise RuntimeError(f"Stage {name} failed with exit {code}; inspect its log")


def validate_supplement():
    spec=importlib.util.spec_from_file_location("audit_helpers",ROOT/"scripts/audit_project.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    errors=[];paths=sorted((OUT/"test").glob("sub-*.json"))
    if len(paths)!=216:errors.append(f"Expected 216 evaluated models, found {len(paths)}")
    for path in paths:
        r=json.loads(path.read_text())
        for precision in ["fp32","int8"]:
            for analysis in ["primary","secondary"]:
                metrics=module.metrics(r["pred"][precision],r["y"],r["artifact"],analysis=="primary")
                for key in ["acc","macro_f1"]:
                    if abs(metrics[key]-r[analysis][precision][key])>1e-10:errors.append(f"{path.name}/{precision}/{analysis}/{key}")
                if metrics["confusion"]!=r[analysis][precision]["confusion"]:errors.append(f"{path.name}/confusion")
    target=EVIDENCE/"supplement_validation.json"
    target.write_text(json.dumps({"models":len(paths),"errors":errors,"passed":not errors,
        "scope":"Recomputed supplementary stored prediction metrics; complete 54 bases + 162 branches. No new finalist selection."},indent=2)+"\n")
    if errors:raise RuntimeError(f"Supplementary validation failed: {errors[:5]}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pids",nargs="+",type=int,required=True)
    parser.add_argument("--artifact-python",required=True)
    args=parser.parse_args()
    try:
        last=None
        while True:
            bases=len(list((OUT/"bases").glob("*.json")))
            branches=len(list((OUT/"runs").glob("*.json")))
            progress=(bases,branches)
            if progress!=last:
                status("training",bases=bases,expected_bases=54,branches=branches,expected_branches=162)
                last=progress
            for name in ["matched_train_mps.log","matched_train_cpu.log"]:
                text=(EVIDENCE/name).read_text()
                if "Traceback (most recent call last)" in text:raise RuntimeError(f"Training failed; inspect {name}")
            if bases==54 and branches==162:break
            if not any(alive(pid) for pid in args.training_pids):raise RuntimeError("Training exited before the complete grid was saved")
            time.sleep(20)
        run("matched_evaluation",[sys.executable,"-u","scripts/run_matched_control.py","--stage","evaluate","--workers","6"])
        run("matched_analysis",[sys.executable,"-u","scripts/run_matched_control.py","--stage","analyze"])
        status("validating_supplementary_predictions")
        validate_supplement()
        for name,bundle in [("memory_full22",MODELS/"bases/sub-01_seed-0_full-22.pt"),
                            ("memory_final_candidate",MODELS/"branches/sub-01_seed-0_relevance-16-matched50.pt")]:
            output=OUT/(name+".json")
            if not output.exists():run(name,[sys.executable,"-u","scripts/benchmark_memory.py","--bundle",str(bundle),"--output",str(output)])
        bundle=MODELS/"branches/sub-03_seed-0_relevance-16-matched50.pt"
        expected=json.loads((OUT/"test"/(bundle.stem+".json")).read_text())
        for precision in ["fp32","int8"]:
            output=OUT/("portable_"+precision+".json")
            if not output.exists():run("portable_inference_"+precision,[sys.executable,"-u","scripts/infer_bundle.py",
                "--bundle",str(bundle),"--input-npz",str(ROOT/"data.nosync/processed/sub-03_ses-2.npz"),
                "--precision",precision,"--output",str(output)])
            if json.loads(output.read_text())["pred"]!=expected["pred"][precision]:raise RuntimeError("Portable inference predictions do not match evaluation")
        run("report_final_build",[sys.executable,"scripts/build_report.py"])
        run("report_final_pdf",[args.artifact_python,"scripts/export_report_pdf.py"])
        analysis=json.loads((OUT/"analysis.json").read_text())
        contrast=[r for r in analysis["comparisons"] if r["base"]=="relevance-16" and r["precision"]=="int8" and r["a"]=="50" and r["b"]=="0"][0]
        with (ROOT/"docs/WORK_LOG.md").open("a") as log:
            log.write("\n## 补充实验执行完成（自动流水线记录）\n\n")
            log.write(f"完成时间：{datetime.now(ZoneInfo('America/New_York')).isoformat()}。54 个独立重建基础模型及 162 条等预算分支均完成；216 个模型分别导出/量化并评价 FP32/INT8。\n\n")
            log.write("全量补充预测指标重算通过。基于原固定被试、种子和通道列表，没有重新选择最终模型。新运行环境为记录中的 M5，历史 M4 性能结果未替换。\n\n")
            log.write(f"最终候选对应的 relevance-16 INT8，50% 剪枝相对不剪枝、同样微调 100 epoch 的配对差异为 {contrast['difference_pp']:+.3f} 个百分点，95% 探索性区间 [{contrast['ci_low_pp']:+.3f}, {contrast['ci_high_pp']:+.3f}]。这是事后补充复现实验，不是新增独立被试或原预注册结果。\n\n")
            log.write("已执行独立进程 RSS 测量及可重定位推理 CLI 核验，FP32/INT8 预测与评价记录完全一致。报告 Markdown/PDF 已用补充结果重建，仍待最终逐页视觉检查与交付包校验。相关命令与输出保存在 docs/evidence。\n")
        status("ready_for_final_review",matched_contrast=contrast)
    except BaseException as error:
        status("failed",error=f"{type(error).__name__}: {error}")
        raise


if __name__=="__main__":main()
