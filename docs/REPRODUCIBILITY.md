# 复现、恢复与交付

## 原实验与当前副本

作者确认原实验在 Apple M4 Mac mini 上完成。`results/` 中的历史记录是原实验证据；维护前当前副本未包含 `.venv.nosync` 和 `data.nosync`。本次恢复环境及原始数据，不代表原实验曾经没有完成，也不自动恢复原权重。

原始硬件日志记录 Apple M4、macOS、软件版本及电源状态，但没有记录 RAM 容量。当前机器的 RAM 由新环境日志记录；不能无依据补写成原实验配置。

## 安装锁定环境

```bash
UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --locked
.venv.nosync/bin/python -m unittest discover -s tests -v
```

这里显式使用 `.venv.nosync`，兼容项目已有的 `.venv` 符号链接。直接运行 `.venv.nosync/bin/python` 可避免 `uv` 在后台重新解析依赖。`uv.lock` 保留原始版本；本次没有升级训练依赖。

## 只检查已有证据

```bash
python3 scripts/audit_project.py
python3 -u scripts/audit_project.py --deep
python3 scripts/audit_project.py --require-artifacts
```

- 普通检查验证配置覆盖、冻结表哈希、汇总表一致性、模型及预处理文件是否齐全。
- `--deep` 从全部 459 个测试记录的逐条预测重算主/次分析准确率、macro-F1、类别召回率、混淆矩阵，并与结果表核对。
- `--require-artifacts` 还要求模型/数据产物齐全。证据一致但模型缺失，不应误报为可以直接推理。
- 默认写入 `results/maintenance/audit.json`，不覆盖原始结果。文件可能受 iCloud 按需下载影响；日志中的进度与实际完成状态分别记录。

完整性检查不能独立证明原始训练数据、实际训练过程或冻结时间。预测与表格一致，也不等于模型权重已经恢复。

## 恢复原数据

```bash
MIBCI_RESULTS_DIR="$PWD/results/maintenance/reproduction" \
  .venv.nosync/bin/python scripts/prepare_data.py \
  --expected-summary results/data_summary.csv
```

默认数据目录是 `data.nosync/`。下载后先校验每个原始 `.mat` 文件的 SHA-256，再做与原实验一致的因果预处理。重建的数据汇总写入独立结果目录；原始 `results/data_summary.csv` 保持不变。MOABB 的该数据下载路径关闭了 TLS 证书验证，因此这里只能用保留的原始哈希检查文件内容一致性。

## 恢复原权重

优先从原 Mac mini 找回整个 `data.nosync/`，包括：

```text
raw/          原始 .mat
processed/    每个被试/会话的 .npz 和 manifest.json
models/
  validation/ .pt、检查点、FP32 ONNX、INT8 ONNX
  final/      .pt、FP32 ONNX、INT8 ONNX
```

可以直接指定外部路径，无需移动文件：

```bash
export MIBCI_DATA_DIR="/absolute/path/to/original/data.nosync"
python3 scripts/audit_project.py --data-dir "$MIBCI_DATA_DIR" --require-artifacts
```

`.pt` bundle 内保存了通道索引、归一化均值/标准差、模型维度和权重，仅拷贝 ONNX 不足以重现预处理。恢复模型后还应在锁定环境中验证反序列化、导出一致性与推理，不把文件存在性等同于模型正确性。

2026-11-09：原 Mac mini 上的 `data.nosync/` 仍然完整，`--require-artifacts` 审计通过。主实验原模型和补充模型已发布到 GitHub Release `models-v1`，下载与核对见 README 第 6.1 节。发布前做过这些检查：459 个最终模型在 session 2 重算 FP32/INT8 预测，与 `results/test` 逐条一致；459 个验证模型在留出 run 上的混淆矩阵、loss 和 ONNX 文件大小与 `results/validation/quant` 一致；另外 54 个预算选择文件能以 `weights_only=True` 载入。

训练、剪枝、量化和测试脚本现在会同时检查结果与必需产物。已有 JSON 而模型缺失时会明确报错，拒绝悄悄重训并替换已发表分数所对应的权重。

## 从零独立复现主实验

原实验已冻结，不能在原结果目录再次运行整个选择流程。为新实验设置两个独立目录：

```bash
export MIBCI_DATA_DIR="$PWD/reproduction.nosync/data"
export MIBCI_RESULTS_DIR="$PWD/reproduction.nosync/results"
bash scripts/run_all.sh
```

这会重新执行数据准备、验证、预算选择、通道选择、量化、剪枝、冻结、最终训练、测试、性能测试及回放。它是新的复现实验；不会替换原证据。允许在冻结之前恢复失败阶段；冻结后恢复最终训练/导出时，应运行相应单步脚本，而非重新运行预算/选择/冻结步骤。

默认保持原分配：1–6 号被试使用可用的 MPS，7–9 号使用 CPU。可为独立实验显式设置 `MIBCI_DEVICE=cpu|mps|cuda`，不可将后端改变后的权重称为原权重。CUDA 需要可用的对应 PyTorch 环境；本次没有在 NVIDIA 服务器验证 CUDA。

相同种子并不保证不同操作系统、硬件、PyTorch 版本或后端的逐位一致。复现时比较指标、预测与记录的设备/依赖；如需权重一致性，应提供原始 bundle 的哈希。

## 补充相同微调预算对照

见 [补充实验协议](SUPPLEMENTARY_PROTOCOL.md)。该实验与原主分析隔离，不改变原最终模型的选择。

## 生成本地报告和演示

```bash
.venv.nosync/bin/python scripts/build_report.py
python3 scripts/build_presentation.py
python3 -m http.server 8765
```

报告：`docs/REPORT.md`。演示：浏览器打开 `http://localhost:8765/presentation/index.html`，使用方向键翻页、N 查看讲稿、F 全屏。也可直接打开该 HTML；图片及视频均使用本地 assets。原幻灯片设计和顺序保留，私有 blob 引用已替换。尚未导出 PDF/PPTX 时，不将 HTML 源码称作已导出的演示文件。

PDF 导出使用带 ReportLab/Pillow 的文档运行环境，不改动训练锁文件：

```bash
python scripts/export_report_pdf.py
```

本次使用 Codex 捆绑文档 Python 运行上述脚本，并逐页渲染检查 `docs/REPORT.pdf`。在其他机器上应为该独立导出脚本安装 ReportLab/Pillow；不要为排版升级训练依赖。PDF 和 Markdown 的内容均来自同一个生成器，补充实验完成后一起重新生成。

## 使用打包后的模型推理

```bash
.venv.nosync/bin/python scripts/infer_bundle.py \
  --bundle models/branches/sub-03_seed-0_relevance-16-matched50.pt \
  --input-npz data.nosync/processed/sub-03_ses-2.npz \
  --precision int8 --output predictions_unique_name.json
```

示例用于包含补充模型的交付包。输入必须已经按照项目执行因果过滤/截窗，形状为 trials × 22 × 500；该命令只应用 bundle 内的通道选择和训练归一化，不把原始 EEG 当成已过滤窗口，也不读取标签来拟合参数。模型路径可重定位；原协议中的执行路径保留作来源记录，在另一机器上不能冒充原目录的直接续跑。

## 实测进程内存

模型与两个 ONNX 导出齐全后：

```bash
.venv.nosync/bin/python scripts/benchmark_memory.py \
  --bundle /absolute/path/to/model.pt \
  --output results/maintenance/memory_unique_name.json
```

每种精度在不导入 PyTorch 的独立进程中测量 RSS，记录加载前、加载后、推理后与进程生命周期峰值。包括解释器、运行时、图初始化、输入和分配器开销，不能解释成纯激活内存或 MCU SRAM。

## 交付包

```bash
python3 scripts/package_release.py --output release/project_review.zip
```

交付包默认包含源码、锁文件、文档、演示及结果，不包含虚拟环境、原始数据或缺失的模型，也不声称可以开箱推理。模型存在时可用 `--model-dir` 指定要随包保存的模型目录；清单会逐文件记录 SHA-256 与文件大小。
