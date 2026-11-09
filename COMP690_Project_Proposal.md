# Edge-Ready Neural Decoding: Quantization and Structured Pruning of a Compact CNN for Real-Time Motor-Imagery BCI

**Student:** [Peiye Li]  |  **Course:** COMP 690-193, Special Topics in CS (Fall 2026)  |  **Project Proposal (1 page)**

## 1. Problem & Motivation
Brain-computer interfaces (BCIs) must decode neural signals under strict latency and power budgets, especially for wearable and implantable deployment, where inference must run on MCU-class hardware (tens-to-hundreds of KB of SRAM, no GPU). Compact CNN decoders such as EEGNet make deep decoding practical, but such models are typically benchmarked on accuracy alone, not deployability. This project studies the systems question at the heart of this course: **how far can a neural decoder be compressed — in memory footprint, FLOPs, and latency — before decoding accuracy degrades, and what does the accuracy–efficiency Pareto frontier look like for an edge-deployable motor-imagery (MI) BCI?** The work maps directly onto the course themes of architecture/system design tradeoffs and "application design and optimization … tools, runtimes" for BCI systems.

## 2. Background & Related Work
EEGNet (Lawhern et al., 2018) is a compact CNN (depthwise/separable convolutions, ~a few thousand parameters) that is a standard baseline across EEG paradigms. The BCI Competition IV Dataset 2a (9 subjects, 22 EEG channels @ 250 Hz, 4-class motor imagery), publicly accessible via the MOABB benchmarking library, is the de-facto public benchmark for MI decoding. EEG-TCNet (Ingolfsson et al., 2020) demonstrated that an accurate MI decoder can run on an embedded RISC-V MCU, establishing precedent and reference points. Classic compression work (Deep Compression, Han et al., 2016) introduced pruning + quantization, but a systematic accuracy–latency–memory study for EEG decoders remains thin.

## 3. Proposed Approach
1. **Baseline:** Re-implement EEGNet in PyTorch (using the Braindecode reference implementation) and train subject-specific models on BCI Competition IV 2a; target accuracy within a few points of published reference results.
2. **Quantization:** Post-training quantization (PTQ) to INT8 via PyTorch FX / ONNX Runtime; exploratory INT4 (weights-only), with quantization-aware training (QAT) as a fallback if PTQ accuracy drop exceeds ~2 points.
3. **Structured pruning:** Iterative L1-norm filter pruning (sweep ~10–70% of filters) with fine-tuning. Structured pruning is emphasized over unstructured magnitude pruning because only structured sparsity reduces real FLOPs/latency on dense MCU hardware; unstructured results will be reported as a contrast to highlight this hardware-relevance gap.
4. **Combined model:** The best pruned + INT8 configuration serves as the final "edge candidate."

## 4. Evaluation Plan & Metrics
- **Accuracy:** per-subject, session-wise cross-validation; report mean ± std across all 9 subjects.
- **Efficiency:** model size (KB), FLOPs per inference (fvcore), and single-thread CPU latency (batch = 1, ONNX Runtime, ≥1000 timed runs); deployment check against a Cortex-M7-class budget (≤ 512 KB SRAM, < 50 ms per 1-second EEG window). Optional energy proxy via MAC counts × published per-MAC energy (Horowitz, ISSCC 2014).
- **Analysis:** accuracy–latency and accuracy–size Pareto curves, plus a summary table comparing FP32 baseline vs. pruned vs. INT8 vs. pruned+INT8.

## 5. Deliverables & Demo Plan
Deliverables: a reproducible GitHub repository (data pipeline, training, compression, benchmarking harness), the final report, and the presentation. **Demo (no hardware needed):** (a) live side-by-side terminal run of FP32 vs. compressed-model inference latency on CPU; (b) a simulated closed-loop BCI — pre-recorded EEG epochs are streamed in real time through the compressed decoder, and the decoded motor-imagery class drives an on-screen cursor with a live latency/accuracy HUD; (c) Pareto-curve figures. *Stretch goal:* the same closed-loop demo with a small GRU spike-train decoder on the NLB'21 MC_Maze intracortical dataset.

## 6. Timeline (Fall 2026)
- **Weeks 1–3 (by mid-Sep):** environment + MOABB data pipeline; EEGNet baseline reproduced.
- **Weeks 4–6 (Sep–Oct):** INT8 PTQ pipeline; first accuracy/latency benchmark table.
- **Weeks 7–9 (Oct):** structured-pruning sweep with fine-tuning; combined pruned+INT8 model.
- **Weeks 10–12 (Nov):** full evaluation, Pareto analysis, demo build; report draft.
- **Weeks 13–14 (late Nov–early Dec):** final report and presentation.

## 7. Risks & Mitigations
INT4 PTQ may collapse accuracy → restrict to weights-only INT4 or switch to QAT, and report the result as a Pareto-curve point. Unstructured pruning may show no real speedup → expected, and itself motivates the structured-pruning emphasis. Baseline accuracy below published reference → adopt published Braindecode hyperparameters; the within-subject protocol keeps the target modest.

## 8. References (selected)
[1] Lawhern et al., "EEGNet: a compact CNN for EEG-based BCIs," *J. Neural Eng.* 15(5), 2018.  [2] Tangermann et al., "Review of the BCI Competition IV," *Front. Neurosci.*, 2012.  [3] Ingolfsson et al., "EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces," *IEEE SMC*, 2020.  [4] Han et al., "Deep Compression," *ICLR*, 2016.  [5] Schirrmeister et al., "Deep learning with CNNs for EEG decoding," *Human Brain Mapping*, 2017.  [6] Pei et al., "Neural Latents Benchmark '21," *NeurIPS Datasets & Benchmarks*, 2021.
