"""Build a portable Markdown research report from retained experiment evidence.

Does not train, choose models or overwrite historical tables/figures. New charts
and derived tables are written only under results/maintenance and docs/assets.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/"docs"
ASSETS=DOCS/"assets"
RESULTS=ROOT/"results"


def read_csv(relative):
    with (RESULTS/relative).open(newline="") as stream:return list(csv.DictReader(stream))


def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join(
        "| "+" | ".join(map(str,r))+" |" for r in rows)


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((RESULTS/"frozen_config.json").read_text())
    chosen=frozen["final_configuration"]
    pp=read_csv("analysis/per_participant_primary.csv")
    summary=read_csv("analysis/conditions_primary.csv")
    secondary=read_csv("analysis/conditions_secondary.csv")
    efficiencies=read_csv("analysis/efficiency.csv")
    comparisons=read_csv("analysis/comparisons_primary.csv")
    conditions={(r["condition"],r["precision"]):r for r in summary}
    costs={(r["condition"],r["precision"]):r for r in efficiencies}
    selected_key=(chosen["condition"],chosen["precision"])
    baseline_key=("full-22","fp32")
    b,f=conditions[baseline_key],conditions[selected_key]
    bc,fc=costs[baseline_key],costs[selected_key]
    participant={(r["condition"],r["precision"],int(r["subject"])):r for r in pp}
    participant_rows=[]
    stage_rows=[]
    for subject in range(1,10):
        before=participant[(*baseline_key,subject)];after=participant[(*selected_key,subject)]
        participant_rows.append([subject,float(before["acc"])*100,float(after["acc"])*100,
                                 (float(after["acc"])-float(before["acc"]))*100])
        stages=[float(participant[(c,p,subject)]["acc"]) for c,p in [baseline_key,
                ("relevance-16","fp32"),("relevance-16-prune50","fp32"),selected_key]]
        stage_rows.append([subject,*[100*(stages[i+1]-stages[i]) for i in range(3)]])
    loss_count=sum(r[3]<-2 for r in participant_rows)
    if loss_count!=int(f["n_losing_more_than_2pts"]):raise RuntimeError("Participant loss count disagrees with summary")
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white"})
    fig,ax=plt.subplots(figsize=(8.6,4.9),layout="constrained")
    differences=[r[3] for r in participant_rows]
    ax.barh([f"P{r[0]}" for r in participant_rows],differences,color=["#2e75b6" if v>=0 else "#b5591a" for v in differences])
    for i,value in enumerate(differences):ax.text(value+(.15 if value>=0 else -.15),i,f"{value:+.2f}",va="center",ha="left" if value>=0 else "right")
    ax.axvline(0,color="#333333",lw=.8);ax.axvline(-2,color="#555555",ls="--",lw=1,label="2-point loss threshold")
    ax.set(xlim=(-6,9),xlabel="Final INT8 minus full-22 FP32 accuracy (percentage points)",
           title="Final configuration: participant-level accuracy differences")
    ax.invert_yaxis();ax.legend(loc="lower right",frameon=False)
    fig.savefig(ASSETS/"participant_differences.png",dpi=170);plt.close(fig)
    class_columns=["recall_left_hand","recall_right_hand","recall_feet","recall_tongue"]
    class_rows=[]
    for column in class_columns:
        a=statistics.mean(float(participant[(*baseline_key,s)][column]) for s in range(1,10))*100
        z=statistics.mean(float(participant[(*selected_key,s)][column]) for s in range(1,10))*100
        class_rows.append([column.removeprefix("recall_").replace("_"," "),a,z,z-a])
    fig,ax=plt.subplots(figsize=(8.6,4.3),layout="constrained")
    x=list(range(4))
    ax.bar([v-.18 for v in x],[r[1] for r in class_rows],width=.36,color="#555555",label="Full-22 FP32")
    ax.bar([v+.18 for v in x],[r[2] for r in class_rows],width=.36,color="#2e75b6",hatch="//",label="Frozen final INT8")
    ax.set(xticks=x,xticklabels=[r[0] for r in class_rows],ylim=(0,100),ylabel="Mean participant recall (%)",title="Class recall across nine participants")
    ax.legend(frameon=False)
    fig.savefig(ASSETS/"class_recall.png",dpi=170);plt.close(fig)
    with (RESULTS/"maintenance/participant_differences.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["subject","baseline_accuracy_pct","final_accuracy_pct","difference_pp"]);writer.writerows(participant_rows)
    with (RESULTS/"maintenance/class_recall.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["class","baseline_recall_pct","final_recall_pct","difference_pp"]);writer.writerows(class_rows)
    parameter_ratio=float(bc["parameter_bytes"])/float(fc["parameter_bytes"])
    file_reduction=100*(1-float(fc["file_bytes"])/float(bc["file_bytes"]))
    modeled_ratio=float(bc["peak_activation_bytes"])/float(fc["peak_activation_bytes"])
    shown=[baseline_key,("full-22","int8"),("proposed-16","fp32"),("relevance-16","fp32"),
           ("weightnorm-16","fp32"),("proposed-8","fp32"),selected_key]
    result_table=table(["Condition","Precision","Accuracy (%)","Macro-F1 (%)","Difference vs baseline (pp)","95% paired interval (pp)"],
        [[c,p,f"{100*float(conditions[(c,p)]['acc_mean']):.2f}",f"{100*float(conditions[(c,p)]['f1_mean']):.2f}",
          f"{100*float(conditions[(c,p)]['diff_vs_full_fp32']):+.2f}",
          f"[{100*float(conditions[(c,p)]['diff_ci_low']):+.2f}, {100*float(conditions[(c,p)]['diff_ci_high']):+.2f}]"] for c,p in shown])
    selection_table=table(["Electrodes","Precision","Control","Proposed minus control (pp)","95% paired interval (pp)"],
        [[r["electrodes"],r["precision"],r["family"].removeprefix("proposed vs "),f"{100*float(r['mean_diff']):+.2f}",
          f"[{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}]"] for r in comparisons if r["family"].startswith("proposed vs")])
    size_table=table(["Quantity","Full-22 FP32","Frozen final INT8","Interpretation"],[
        ["Initializer/parameter bytes",bc["parameter_bytes"],fc["parameter_bytes"],f"{parameter_ratio:.2f}x ratio; includes initializers such as scales"],
        ["Serialized ONNX bytes",bc["file_bytes"],fc["file_bytes"],f"{file_reduction:.2f}% file reduction"],
        ["Modeled peak activation bytes",bc["peak_activation_bytes"],fc["peak_activation_bytes"],f"{modeled_ratio:.2f}x ratio under uniform-precision tensor assumptions"],
        ["Actual ONNX float32 input bytes","44000","32000","Both graph interfaces accept float32; quantization is internal"],
        ["Median CPU latency (ms)",bc["latency_median_ms"],fc["latency_median_ms"],"Historical batch-1, one-thread measurement"],
        ["MACs",bc["macs"],fc["macs"],"Analytical convolution/dense multiply-accumulate count"]])
    memory_rows=[]
    for name,label in [("memory_full22","Full-22"),("memory_final_candidate","16 electrodes, 50% pruned")]:
        path=RESULTS/"supplementary/matched_budget"/(name+".json")
        if path.exists():
            result=json.loads(path.read_text())
            for precision,row in result["rows"].items():
                memory_rows.append([label,precision,*[f"{row[key]/1048576:.2f}" for key in
                    ["baseline_rss_bytes","after_session_rss_bytes","after_inference_rss_bytes","process_lifetime_peak_rss_bytes"]]])
    memory_text=("### 4.1 Additional isolated-process measurements on the M5 reproduction\n\n"+
        table(["Model","Precision","Baseline RSS (MiB)","After session (MiB)","After inference (MiB)","Lifetime peak (MiB)"],memory_rows)+
        "\n\nEach row is one fresh process without PyTorch, using 100 warm-up and 1,000 inference calls. "
        "These are M5 measurements of independently reproduced models, not replacements for the historical M4 timing. "
        "Process RSS includes interpreter, runtime, graph setup, input arrays and allocator overhead; its lifetime high-water mark is not incremental tensor memory or MCU SRAM. "
        "No uncertainty interval or pure activation-memory compression ratio is inferred from these process measurements.") if memory_rows else ""
    supplement_path=RESULTS/"supplementary/matched_budget/analysis.json"
    matched_summary=""
    if supplement_path.exists():
        supplement=json.loads(supplement_path.read_text())
        contrasts=[r for r in supplement["comparisons"] if r["a"] in ["25","50"] and r["b"]=="0"]
        fig,ax=plt.subplots(figsize=(8.6,4.9),layout="constrained")
        for i,row in enumerate(contrasts):
            color="#2e75b6" if row["precision"]=="int8" else "#555555"
            ax.hlines(i,row["ci_low_pp"],row["ci_high_pp"],color=color,lw=1.4)
            ax.scatter(row["difference_pp"],i,marker="s" if row["a"]=="50" else "o",s=38,color=color,zorder=3)
        ax.axvline(0,color="#333333",ls="--",lw=.8)
        ax.set_yticks(range(len(contrasts)),[f"{r['base']} | {r['precision'].upper()} | prune {r['a']}%" for r in contrasts])
        ax.invert_yaxis()
        ax.set(title="Pruning versus an equally fine-tuned unpruned model",
               xlabel="Paired accuracy difference (percentage points)\n95% participant bootstrap interval; n = 9")
        fig.savefig(ASSETS/"matched_budget_effects.png",dpi=170);plt.close(fig)
        origin="independent reproduction" if supplement["base_origin"]=="independent_reproduction" else "matched branching from original bundles"
        supplement_text=(f"The additional experiment has been executed as **{origin}**, after the original test results were known. "
            "Both historical pruning bases and all nine participants/three seeds were retained; no finalist was reselected. "
            "Every base was branched into 0%, 25% and 50% pruning, all with the same 100-epoch, 1e-4 Adam fine-tuning budget.\n\n"+
            table(["Base","Precision","Branch A","Branch B","A minus B (pp)","95% paired interval (pp)"],
                [[r['base'],r['precision'],r['a'],r['b'],f"{r['difference_pp']:+.2f}",f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"] for r in supplement['comparisons']])+"\n\n"
            "Here `base` is the 475-epoch model with the recorded origin; `0` is unpruned plus 100 more epochs; `25` and `50` are equally fine-tuned pruned branches. "
            "The `0 - base` contrast estimates the extra-training effect. The `25/50 - 0` contrasts compare structural pruning at equal fine-tuning budgets. "
            "Intervals are exploratory participant-level bootstrap intervals. "
            "See the [supplementary protocol](../results/supplementary/matched_budget/protocol.json) and [all paired comparisons](../results/supplementary/matched_budget/paired_comparisons.csv).")
        supplement_text+="\n\n![Matched-budget pruning contrasts with participant intervals](assets/matched_budget_effects.png)"
        selected_contrast=next(r for r in supplement["comparisons"] if r["base"]=="relevance-16" and r["precision"]=="int8" and r["a"]=="50" and r["b"]=="0")
        matched_summary=(f"The added **M5 matched-budget reproduction** found that 50% pruning of the relevance-16 INT8 model "
            f"reduced mean accuracy by **{abs(selected_contrast['difference_pp']):.2f} percentage points** against an unpruned model given the same 100 extra epochs "
            f"(95% exploratory interval **[{selected_contrast['ci_low_pp']:+.2f}, {selected_contrast['ci_high_pp']:+.2f}]**). "
            "Thus, similar performance to the original shorter-trained baseline does not imply pruning is accuracy-free. "
            "This supplementary analysis was added after test outcomes were known; it preserves the historical finalist.")
        supplement_text+=(f"\n\nFor relevance-16 INT8, the 50%-pruned branch differed from the equally fine-tuned unpruned branch by "
            f"{selected_contrast['difference_pp']:+.2f} points, with a 95% interval "
            f"[{selected_contrast['ci_low_pp']:+.2f}, {selected_contrast['ci_high_pp']:+.2f}]. "
            "This isolates the structural comparison within the recorded fine-tuning procedure more closely than the original baseline contrast. "
            "It remains a post-test, nine-participant reproduction on different hardware and does not replace the original study.")
        agreements=[]
        for path in (RESULTS/"supplementary/matched_budget/test").glob("*.json"):
            record=json.loads(path.read_text())
            agreements.extend((record.get("historical_prediction_agreement") or {}).values())
        if agreements:
            supplement_text+=(f"\n\nAcross {len(agreements)} model/precision pairs comparable with a historical condition, "
                f"mean full-trial prediction agreement was {100*statistics.mean(agreements):.2f}% "
                f"(range {100*min(agreements):.2f}–{100*max(agreements):.2f}%). "
                "Agreement is a reproduction diagnostic, not proof of identical weights; mismatching runs remain included.")
    else:
        supplement_text=("A supplementary matched-budget experiment is implemented in `scripts/run_matched_control.py`. "
            "Its analysis is **pending**, and no additional empirical outcome is claimed here. "
            "It branches each of the two historical bases into unpruned, 25%-pruned and 50%-pruned models, all given 100 epochs at 1e-4 with the same seed and batch schedule. "
            "Missing original weights require explicitly labeled independent base reproduction. "
            "This analysis was added after the original test outcomes were known; it cannot be represented as preregistered or used to reselect the historical finalist.")
    report=f'''# Fewer EEG electrodes preserve average decoding accuracy, while INT8 does not improve M4 CPU latency

Peiye Li · COMP 690-193 · Fall 2026  
Local report assembled from retained experiment evidence; maintenance date: 2026-11-04.

## Executive summary

The frozen 16-electrode, 50%-pruned INT8 configuration reached **{100*float(f['acc_mean']):.2f}%** held-out-session accuracy, compared with **{100*float(b['acc_mean']):.2f}%** for the full 22-electrode FP32 baseline. The mean paired difference was **{100*float(f['diff_vs_full_fp32']):+.2f} percentage points**, with a 95% participant bootstrap interval of **[{100*float(f['diff_ci_low']):+.2f}, {100*float(f['diff_ci_high']):+.2f}]**. This is close average performance, not proof of equivalence: **{loss_count}/9 participants lost more than two points**.

Redundancy-aware electrode selection did not show a clear advantage over relevance-only or spatial-weight selection. It outperformed the mean of three fixed random subsets, but most comparisons against the stronger controls had intervals crossing zero. The results support a controlled negative finding about this particular redundancy proxy, rather than a new superior selection algorithm.

Initializer bytes fell by approximately {parameter_ratio:.1f}x, but the complete ONNX file shrank by only {file_reduction:.1f}%. INT8 inference of the full model was about 28% slower than FP32 on the measured Apple M4 CPU. The large activation-memory ratio is an analytical tensor estimate; actual ONNX Runtime memory and MCU SRAM were not measured in the historical experiment.

{matched_summary}

## 1. Research question and relation to prior work

The question is whether EEG electrodes selected for both class relevance and low band-power redundancy preserve four-class motor-imagery information across recording days, and whether any advantage survives INT8 quantization. Electrode count describes physical measurements; pruning removes learned internal features. Their costs and information loss should be evaluated separately.

EEGNet uses depthwise and separable convolutions for compact EEG decoding [1]. Q-EEGNet studies an 8-bit, hardware-aware embedded EEGNet implementation [2]. MI-BMInet combines spatial-filter channel selection, quantization and embedded evaluation [3]. The present project is a controlled software comparison within one fixed EEGNet pipeline. The spatial-weight baseline is inspired by MI-BMInet; it is not a reproduction of that architecture or its deployment results.

## 2. Data and experimental protocol

BCI Competition IV 2a / BNCI 2014-001 contains nine participants, two recording sessions on different days, 22 EEG channels at 250 Hz and four classes: left hand, right hand, feet and tongue. Each session has 288 trials in six runs. EOG is excluded. The primary analysis removes expert-flagged artifacts, while the secondary analysis evaluates the same models on all trials. Counts and source-file SHA-256 values are retained in [data_summary.csv](../results/data_summary.csv).

Each run is processed with a causal fourth-order Butterworth 4–38 Hz band-pass, resetting state at run boundaries. Each trial supplies a two-second window from 0.5 to 2.5 seconds after the cue. Normalization uses training-partition channel means and standard deviations only.

Session 1 supplies training and three validation folds, each holding out two complete runs. Channel rankings, normalization and PTQ calibration are fitted only to the applicable training partition. Session 2 is the final test day. The source code requires an explicit final-evaluation flag to load it. The retained [frozen configuration](../results/frozen_config.json) fixes the training budget at 475 epochs and seeds at 0, 1 and 2. The stored validation-table hash can be checked; this consistency check is not independent proof of the original chronology.

The decoder is Braindecode 2.0.0 EEGNet with F1=8, D=2, F2=16, temporal kernel 64, dropout 0.5 and max-norm constraints. Adam uses learning rate 1e-3, batch size 64 and the same fixed epoch budget for the electrode-selection conditions. The 22-channel model has 2,420 trainable parameters.

For selection, Welch log power is calculated in 8–13 Hz and 13–30 Hz bands. Relevance is the average multiclass Fisher score, scaled to 0–1 across electrodes. Redundancy is the mean absolute Spearman correlation across bands. Greedy selection maximizes relevance minus 0.5 times mean redundancy to the already chosen electrodes. Controls use relevance alone, spatial-weight L2 norms and three fixed random subsets. Models are retrained with 16 or 8 electrodes. These feature correlations do not identify shared neural sources.

Static INT8 PTQ uses ONNX QDQ format, signed activations, per-channel weights and training-only min-max calibration. The original bounded pruning comparison drops 25% or 50% of final pointwise filters by L1 norm, slices BatchNorm and classifier tensors, and adds 100 fine-tuning epochs at 1e-4. The original unpruned baseline lacks those extra epochs, so pruning-versus-baseline accuracy differences confound removal and additional optimization.

The finalist was chosen using validation: fewest electrodes within two points of full-channel FP32 accuracy, then fewest parameter bytes, then highest accuracy. Test accuracy never changes this historical choice. Accuracy, macro-F1, per-class recall and confusion matrices are retained. Seeds are averaged within participant; participants, not seeds or trials, are the nine independent resampling units. Paired intervals use 10,000 percentile bootstrap resamples and seed 690. Intervals are exploratory and are not adjusted for all comparisons.

## 3. Held-out-session decoding results

{result_table}

The full tables include all 17 conditions in both precisions: [primary](../results/analysis/conditions_primary.csv) and [secondary](../results/analysis/conditions_secondary.csv). These are within-participant transfers to another session, not tests on previously unseen people. Differences below are percentage points, not relative percentage changes.

### 3.1 Redundancy-aware selection against its controls

{selection_table}

The redundancy term often leaves the selected set unchanged: in the final seed-0 rankings, proposed and relevance-only subsets are identical for 6/9 participants at 16 electrodes and 7/9 at 8. The small observed advantages against relevance-only cannot establish a reliable improvement, while spatial-weight ranking is competitive. Beating three random subsets supports task-aware selection in this setting; it does not establish superiority to stronger channel-selection algorithms or robustness over all possible random layouts.

![Paired selection comparisons with participant bootstrap intervals](../results/analysis/slides/fig_forest.png)

### 3.2 Participant variation changes the interpretation of the mean

{table(["Participant","Full-22 FP32 (%)","Frozen final INT8 (%)","Difference (pp)"],[[s,f"{a:.2f}",f"{z:.2f}",f"{d:+.2f}"] for s,a,z,d in participant_rows])}

![Final configuration minus baseline by participant](assets/participant_differences.png)

The losses are concentrated in participants {", ".join("P"+str(r[0]) for r in participant_rows if r[3]<-2)}. A similar group mean can coexist with useful gains for some people and material losses for others. The exploratory interval does not establish a two-point noninferiority margin. Electrode lists are fitted separately per person, so this experiment does not validate a universal headset layout.

The observed path to the final model can also be decomposed arithmetically:

{table(["Participant","22 to 16 electrodes + retraining (pp)","Pruning + 100 extra epochs (pp)","INT8 conversion (pp)"],[[s,f"{a:+.2f}",f"{z:+.2f}",f"{d:+.2f}"] for s,a,z,d in stage_rows])}

These columns sum to each participant's final difference up to displayed rounding. They are an accounting of observed conditions, not a causal decomposition: electrode models were retrained, and the original pruning step also adds optimization. The matched-budget experiment below addresses the latter confounding explicitly.

### 3.3 Class-specific outcomes

{table(["Class","Baseline recall (%)","Final recall (%)","Difference (pp)"],[[c,f"{a:.2f}",f"{z:.2f}",f"{d:+.2f}"] for c,a,z,d in class_rows])}

![Mean participant recall by class](assets/class_recall.png)

Class recall is averaged within participant rather than pooled across all trials. These descriptive differences help locate errors; they do not establish a causal neural mechanism or a statistically reliable class-specific effect. The retained [confusion matrices](../results/analysis/fig_confusion.png) show the complementary pooled error pattern.

## 4. Efficiency: storage, modeled tensors and measured time

{size_table}

The historical measurement used the author's Apple M4 **Mac mini**, ONNX Runtime 1.31.1, batch size one and one inference thread, with 100 warm-up calls and 1,000 timed calls. The original hardware record does not establish the RAM capacity; the machine type is corrected from the author's clarification. Preprocessing is timed separately. These are local CPU measurements, not end-to-end remote-serving latency.

The finalist is about 8% faster than full-22 FP32, but about 25% slower than the unpruned relevance-16 FP32 model (0.257 versus 0.206 ms). Its small speed advantage over the full-channel baseline comes from a smaller input, not evidence that INT8 accelerated this decoder. The validation decision rule prioritizes electrode count and initializer bytes under an accuracy tolerance; it does not select the lowest-latency configuration.

Initializer bytes include weights, biases and quantization metadata. The serialized file also contains the graph and other metadata, explaining why a nearly fourfold initializer reduction produces only a small complete-file reduction.

Activation bytes are estimated under a layer-input-plus-output schedule with uniform four-byte FP32 or one-byte INT8 tensors. The real QDQ graph includes float operations and conversion buffers; both input interfaces receive float32 arrays. The historical `input_bytes` field for INT8 describes the modeled quantized tensor, not the actual interface buffer. New benchmark code records those separately. The estimate excludes live mixed-precision tensors, allocator behavior, runtime overhead and kernel workspaces. It is neither a measured runtime peak nor verified MCU SRAM usage.

The isolated-process tool `scripts/benchmark_memory.py` can measure process RSS separately. Its lifetime peak includes Python, ORT, graph initialization and input arrays; even that measurement must not be labeled tensor memory or MCU SRAM.

The profiler attributes the INT8 slowdown mainly to the grouped spatial convolution: about 185 microseconds versus 12.8 microseconds in FP32. This observation identifies an expensive quantized operator on this runtime; it does not prove an integer kernel is missing. The optimized graph in fact contains QLinearConv operators. Reduced electrode count lowers the expensive early temporal-convolution work, while pruning only the late pointwise filters changes MACs very little. This explains why parameter count alone predicts latency poorly here. ONNX Runtime documents the dependence of quantization performance on hardware, model structure and conversion overhead [4].

![Historical per-layer CPU profile](../results/analysis/slides/fig_int8_layers.png)

The decoder needs a two-second observation window and makes its primary decision 2.5 seconds after the cue plus processing time. Sub-millisecond compute does not imply sub-millisecond intention-to-action latency.

{memory_text}

## 5. Supplementary matched-budget pruning comparison

{supplement_text}

## 6. Demonstration and claim boundaries

The [recorded replay](../results/demo/replay_participant3.mp4) streams prerecorded EEG at 250 Hz through causal preprocessing and displays predictions, cursor movement and compute time. The recorded participant cannot respond to the feedback. This validates a replay implementation, not online closed-loop control, asynchronous intent detection, power consumption or clinical utility.

Nine participants and one dataset limit uncertainty and generalization. Removing artifacts is an expert-assisted offline primary analysis; the all-trials secondary analysis is retained to expose that dependency. There is no MCU latency, SRAM or energy measurement, and no claim about implants or new users. A negative result for one fixed band-power redundancy penalty does not rule out other redundancy measures. The original pruning accuracy comparison needs the matched-budget control above.

## 7. Reproducibility and delivery

The repository retains fixed splits, channel lists, seeds, source hashes, configuration, predictions and measurement scripts. See [reproduction instructions](REPRODUCIBILITY.md), [work log](WORK_LOG.md), [supplementary protocol](SUPPLEMENTARY_PROTOCOL.md) and the [machine-readable audit](../results/maintenance/audit.json). The local [slide deck](../presentation/index.html) replaces private blob references with packaged project assets.

Original missing weights must be restored or independently reproduced in separate paths. New artifact checks refuse to silently retrain over historical records when model files are missing. Matching predictions from a reproduction do not prove identical original weights. Cross-backend bitwise equality is not promised; preserve the recorded software and per-participant device allocation [5].

## References and evidence

1. Lawhern et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J. Neural Eng. 15, 056013 (2018). [Author paper](https://arxiv.org/abs/1611.08024), DOI 10.1088/1741-2552/aace8c.
2. Schneider et al. Q-EEGNet: an Energy-Efficient 8-bit Quantized Parallel EEGNet Implementation for Edge Motor-Imagery Brain–Machine Interfaces (2020). [Author paper](https://arxiv.org/abs/2004.11690).
3. Wang et al. MI-BMInet: An Efficient Convolutional Neural Network for Motor Imagery Brain–Machine Interfaces with EEG Channel Selection. [Author paper](https://arxiv.org/abs/2203.14592).
4. ONNX Runtime. [Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), including mixed graph representation and performance limitations.
5. PyTorch. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Numerical claims are derived from the linked local CSV/JSON evidence and the reproducible generator `scripts/build_report.py`. The audit verifies internal consistency, not independent provenance of the original training run. The full original result tables and figures remain available under `results/analysis/`; no historical result was replaced by this report.
'''
    (DOCS/"REPORT.md").write_text(report)
    print("Wrote docs/REPORT.md and source-backed charts.")


if __name__=="__main__":main()
