# EEG Channel Selection and Model Compression for Motor-Imagery BCI

COMP 690-193 (Fall 2026) final project. The study plan is the revised proposal in
`peer tutoring transcript.pdf`; this repository implements it end to end.

**Question.** At a fixed EEG channel count, does accounting for redundancy in
motor-related band-power features improve held-out-session accuracy over
relevance-only and spatial-weight-based selection, and does any advantage survive
INT8 quantization?

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the project folder:

```bash
uv sync
```

The project sits in an iCloud-synced folder, so the virtual environment and the data
live in `.nosync` folders that iCloud skips: `.venv` is a symlink to `.venv.nosync/`,
and raw data, cached windows and trained models go to `data.nosync/`. Outside iCloud,
a plain `.venv/` created by `uv sync` works the same way.

## Reproducing everything

```bash
scripts/run_all.sh
```

runs the whole pipeline in the order it was run (about 3 hours on an Apple M4 laptop).
Participants 1-6 train on the Apple GPU (MPS) while 7-9 train on the CPU, so both run
at once; every condition of a participant uses the same device, and repeating a run
with its seed reproduces its weights exactly on either device.

| Step | Script | Data it may read |
| --- | --- | --- |
| Download, causal preprocessing, cache | `prepare_data.py` | both sessions (cached only) |
| 22-channel models, 500 epochs, per-epoch validation | `run_validation.py --stage full` | session 1 |
| Fix the epoch budget (rule in the script) | `choose_budget.py` | session 1 |
| Every electrode-selection condition | `run_validation.py --stage subsets` | session 1 |
| ONNX export, INT8 static quantization, validation scores | `quantize_models.py --phase validation` | session 1 |
| Pick the channel-reduced model to prune | `summarize_validation.py --pruning-base` | session 1 |
| Structured pruning + fine-tuning | `run_pruning.py --phase validation` | session 1 |
| Freeze the analysis plan and final configuration | `summarize_validation.py --freeze` | session 1 |
| Final models, 3 seeds, all of session 1 | `train_final.py`, `run_pruning.py --phase final`, `quantize_models.py --phase final` | session 1 |
| Score every final model once | `evaluate_test.py` | **session 2** |
| CPU latency, size, MACs, memory, kernel profile | `benchmark.py` | session 1 |
| Tables and figures | `analyze.py` | results only |
| Real-time replay demonstration | `replay_demo.py` | session 2 (after freezing) |

## Protocol

| Item | Setting |
| --- | --- |
| Data | BCI Competition IV 2a (BNCI 2014-001 `.mat` files) via MOABB; 9 participants, 2 sessions on different days, 6 runs of 48 trials each |
| Channels | 22 EEG channels at 250 Hz; EOG dropped; recorded left-mastoid reference kept |
| Filter | Causal 4th-order Butterworth band-pass, 4-38 Hz (`scipy.signal.sosfilt`), restarted from zero state at every run |
| Window | 0.5-2.5 s after cue onset, 500 samples |
| Artifacts | Primary analysis drops expert-flagged trials everywhere; the secondary analysis scores the same models on all trials |
| Held-out test | Session 2. `load_session(subject, 2)` raises unless `final_evaluation=True`, and `evaluate_test.py` refuses to run before `results/frozen_config.json` exists |
| Validation | Session 1 only: 3 folds holding out runs {1, 2}, {3, 4} and {5, 6} |
| Normalization | Per-channel z-score with training-partition statistics |
| Model | Braindecode 2.0.0 `EEGNet`: F1=8, D=2, F2=16, temporal kernel 64, dropout 0.5, max-norm limits; 2,420 parameters with 22 channels |
| Training | Adam (lr 1e-3), batch 64, fixed budget of 475 epochs (set by `choose_budget.py`), no early stopping |
| Selection | Proposed: greedy relevance (Fisher score of mu and beta log power) minus 0.5 x mean Spearman redundancy; controls: relevance only, spatial-weight norm, three fixed random subsets; 16 or 8 electrodes |
| Quantization | ONNX Runtime static PTQ: QDQ, int8 activations, per-channel int8 weights, min-max calibration on the model's own training trials |
| Pruning | 25% or 50% of the final pointwise convolution's filters by L1 norm, tensors sliced, 100 epochs of fine-tuning at lr 1e-4 |
| Decision rule | Fewest electrodes within 2 points of the 22-channel FP32 validation accuracy, then fewest parameter bytes, then highest accuracy |
| Statistics | Seeds averaged within participant; participant-level bootstrap intervals (10,000 resamples) for paired differences |

## Data provenance

MOABB downloads from `lampx.tugraz.at` without TLS certificate verification, so the
SHA-256 of every source file is recorded in `results/data_summary.csv`.

## Layout

```
src/mibci/   data.py        download, causal preprocessing, session cache and the session-2 lock
             splits.py      run-grouped validation folds
             selection.py   band power, Fisher relevance, Spearman redundancy, greedy rule, controls
             model.py       EEGNet settings, model bundles, deployment copy
             train.py       fixed-budget training (CPU or Apple GPU), scoring
             experiment.py  one training run, device per participant, process pool
             prune.py       structured pruning of the final pointwise convolution
             deploy.py      ONNX export, INT8 quantization, ONNX Runtime sessions, latency, profiling
             costs.py       analytical MACs and activation memory per layer
             env.py         software and hardware details stored with results
scripts/     the pipeline steps above, run_all.sh, and pilot_baseline.py (development pilot)
results/     validation/, final/, test/, benchmark/, analysis/ (tables, figures, summary.md), demo/
             pilot/ holds the first 500-epoch pilot on participant 1 (CPU, an earlier revision of
             pilot_baseline.py with the same training settings); development evidence only
presentation/deck_source/   HTML source of the slide deck (deck.json + one file per slide)
```

## Results

Session 2 (held-out day), artifact-free trials, mean over 9 participants after averaging 3 seeds:

| Condition | FP32 | INT8 |
| --- | --- | --- |
| All 22 electrodes | 65.0% | 65.3% |
| 16 electrodes: redundancy-aware / relevance / spatial-weight / random (mean) | 66.8 / 66.2 / 66.8 / 63.5% | 66.6 / 66.2 / 66.4 / 63.5% |
| 8 electrodes: redundancy-aware / relevance / spatial-weight / random (mean) | 60.6 / 59.9 / 62.1 / 58.3% | 60.5 / 59.6 / 62.3 / 58.1% |
| Final configuration: 16 electrodes (relevance), 50% pruned | 65.0% | **65.0%** |

- The redundancy term did not beat relevance-only selection (+0.6 / +0.7 points at 16 / 8
  electrodes; identical subsets for 6 and 7 of 9 participants) or the spatial-weight rule
  (−0.0 / −1.5 points); it beat random subsets by 3.3 / 2.3 points.
- INT8 changed accuracy by at most 0.5 points but ran 28% slower on the Apple M4 CPU: ONNX
  Runtime's 8-bit grouped spatial convolution took 185 µs against 12.8 µs in FP32.
- The final configuration has 4.0x fewer parameter bytes (2,420 vs 9,576) and 5.5x less peak
  activation memory (72 vs 397 KB) than the 22-electrode FP32 model.

See `results/analysis/summary.md` for every table, `results/analysis/*.png` for the figures
(`slides/` holds large-type copies), `results/frozen_config.json` for the plan that was fixed
before session 2 was scored, and `results/demo/replay_participant3.mp4` for the replay video.
The written report and the slide deck are Claude artifacts:
[report](https://claude.ai/artifact/N6p2t4rHvykJbiuMGfK3E8) and
[slides](https://claude.ai/artifact/PJ8FHGfKYWt5uDsZd2fe4w) (private until shared).
