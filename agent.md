# 项目代理与维护指南

本文件面向在本目录内阅读、修改、调试或扩展项目的编码代理和维护者。项目概览、结果与用户命令见 [README.md](README.md)；标准入口 [AGENTS.md](AGENTS.md) 指向本文件，避免两份指南内容漂移。

本指南基于 2026-11-04 的源码与产物核对。后续任务应重新检查相关状态，不把这里的产物数量当成永久保证。用户的明确任务要求优先；常规读取、检查和已授权的可逆修改直接完成，不额外设置批准流程。

## 1. 先建立正确的项目认识

这是 COMP 690 运动想象 EEG 研究项目，Python 包名 `mibci`，不是 Web 应用。研究组合为电极选择、EEGNet、结构剪枝和 ONNX Runtime INT8 部署。数据为 BCI Competition IV 2a / BNCI 2014-001，9 名被试，分别训练，在同一被试的另一天测试。

必须区分三类材料：

1. **历史主实验**：M4 Mac mini 上的 17 条件研究。`results/validation`、`final`、`test`、`benchmark`、`analysis` 与 `frozen_config.json` 是已保留证据。
2. **当前可执行产物**：原 M4 Mac mini 的 `data.nosync/` 完整保存原始/处理数据和主实验原 `models/validation`、`models/final`（2,808 个文件）；2026-11-09 重算预测与历史记录逐条一致，`--require-artifacts` 审计通过。216 个 `.pt` 和 432 个 ONNX 的 M5 独立补充复现存于 M5 副本和 11-04 交付包，不在 M4 的 `data.nosync/`。两套模型都发布在 GitHub Release `models-v1`（README 第 6.1 节）。M5 副本没有原模型，11-04 及以前的文档和日志据此写成“原权重缺失”。
3. **补充对照**：`supplementary/matched_budget` 是知道测试结果后追加的等微调预算研究，不得倒写成原计划或用于重选历史 finalist。

主实验历史记录完整不等于某个副本具备所有原模型：先看所在机器（`sysctl -n hw.model`：Mac16,10 是原 M4 Mac mini，Mac17,3 是 M5）和 `data.nosync/models` 的实际内容。权重缺失也不表示原实验从未完成。原实验日志没有记录 RAM；2026-11-09 在保存原 `data.nosync/` 的 Mac mini（Mac16,10）上读到 16 GiB。M5 环境记录的 24 GiB 不能补写成原硬件配置。

## 2. 开始任务的检查顺序

1. 读用户要求、README、相关源码；涉及方法或结论再读报告和冻结配置，涉及补充实验再读协议。
2. 检查工作目录和 Git 状态。2026-11-08 起项目根是 Git 仓库：根目录的 `.git` 是指向 `.git.nosync/` 的文本文件（Documents 由 iCloud 同步，`.nosync` 不同步），分支 `main`，远端 `origin` 是公开仓库 https://github.com/lipeiye/comp690-final-project 。先看 `git status`；只在用户要求时提交或推送，不改写已推送的历史。
3. 确定任务是说明性文档、源码修改、生成物更新、数据恢复、历史审计还是新实验。不要把文档任务扩展成完整重训。
4. 明确读写路径和环境变量；特别检查生成器是否使用固定项目根目录。
5. 有历史输出时先看其来源、哈希、配置和必需模型是否齐全，再决定复用、恢复或新开目录。
6. 选择与改动匹配的验证，完成后说明改动、实际验证和仍存在的限制。

检索优先限定范围，避免扫入数千个结果文件和大模型：

```bash
rg --files src scripts tests docs presentation/deck_source \
  -g '!**/__pycache__/**' -g '!*.pyc' -g '!docs/evidence/baseline/**'
rg -n 'load_session|completed|frozen_config|MIBCI_' src scripts tests
```

`docs/evidence/baseline/` 是维护前历史快照，不是运行源码；不要在其中修 bug。`plan.md` 和提案是早期计划，不应优先于已执行代码。工作日志要结合最后的完成条目阅读。

## 3. 代码地图与修改落点

| 需求 | 首先阅读或修改的代码 | 联动检查 |
| --- | --- | --- |
| 原始数据、滤波、切窗、单位 | `src/mibci/data.py`、`scripts/prepare_data.py` | NPZ schema、原始哈希、报告方法、回放滤波 |
| 交叉验证与训练日选择 | `splits.py`、`experiment.session1()` | run 泄漏、artifact mask、归一化拟合范围 |
| 特征、通道排名与对照 | `selection.py` | 验证/最终 rankings、输入顺序、随机种子 |
| EEGNet 结构、模型保存加载 | `model.py` | state dict、F2、剪枝、ONNX、成本模型 |
| 优化器、训练预算、评分 | `train.py`、`experiment.py` | epoch 规则、seed、设备、记录 schema |
| 验证预算与最终选择 | `run_validation.py`、`choose_budget.py`、`summarize_validation.py` | 历史冻结保护、候选范围、统计口径 |
| 最终训练与 held-out 评分 | `train_final.py`、`evaluate_test.py` | session-2 边界、两种精度、逐试次预测 |
| 结构剪枝 | `prune.py`、`run_pruning.py` | pointwise/BN/classifier 张量同步、微调预算 |
| ONNX/PTQ/CPU 推理 | `deploy.py`、`quantize_models.py` | FP32 parity、float32 输入、训练数据校准 |
| 参数、MAC、内存、时延 | `costs.py`、`benchmark.py`、`benchmark_memory.py` | 理论/实测界限、硬件、线程、原 benchmark 保护 |
| 结果恢复和写入安全 | `artifacts.py` | orphan JSON、非空产物、元数据、原子写入 |
| 补充等预算实验 | `run_matched_control.py` | plan、来源、固定源码哈希、原方案不可重选 |
| 统计表和科研图 | `analyze.py` | 按被试平均、配对 bootstrap、单位和分母 |
| 正式报告 | `build_report.py`、`export_report_pdf.py` | 生成的 Markdown/PDF、assets、数值来源 |
| 演示文稿 | `presentation/deck_source/deck.json`、`slides/*.html`、`build_presentation.py` | 当前 19 页、讲稿、视频、离线资产 |
| 项目完整性与交付 | `audit_project.py`、`package_release.py` | 审计范围、包内 manifest、模型来源和排除项 |

其中省略目录的模块位于 `src/mibci/`，脚本位于 `scripts/`。`env.py` 记录环境，`__init__.py` 只有包说明，`py.typed` 为类型标记。完整入口和 CLI 参数见 README。项目没有统一 CLI、Web 路由或服务端入口；不要为文档凭空添加这些概念。

## 4. 研究协议必须保持的边界

### 4.1 数据隔离

- session 1 是训练/验证日；session 2 是留出测试日。
- `data.load_session(subject, 2)` 默认抛 `PermissionError`。`final_evaluation=True` 只能用于最终评估和已明确标记的演示/事后分析，不能放进预算选择、排名、训练或校准路径。
- 预处理两天并缓存不等于可用测试标签调参；不能用绕过 loader 的 `np.load` 消除研究边界。
- 三折验证留出 runs `(0,1)`、`(2,3)`、`(4,5)`，不能改成随机 trial 拆分却沿用原协议名称。
- 训练一律使用无专家 artifact 的 session-1 trials；主测试排除 artifact，次测试用同样模型评价全部试次。
- `ChannelStandardizer.fit` 仅在当前训练分区上执行；验证/测试 `transform` 不更新统计量。
- 通道 Fisher/Spearman 排名、weightnorm 全通道模型和 PTQ calibration 均来自相应训练分区。

### 4.2 选择和统计

- 历史预算 475 是规则输出；`TrainConfig` 的通用默认值是 500。不要把二者混为一谈。
- 原规则：27 个 full 验证运行，居中 25-epoch 平滑，选择距最大值 0.5 个百分点内的最早 25 倍数 checkpoint。
- 最终规则：在 full-22 FP32 验证准确率减 0.02 的门槛内，按电极数、initializer 字节、准确率排序；随机对照不参与候选选择。
- 如果没有满足门槛的候选，实现退回最高验证准确率并记录 `eligible=false`。
- 验证先在被试内平均 fold，最终结果先在被试内平均 seed，再对被试汇总。
- 区间为 9 名被试的配对差异 bootstrap，10,000 resamples，95%，seed 690。seed 运行不是额外独立被试。
- 不根据 session 2 结果筛 seed/被试、调整冗余惩罚、重选通道数或替换历史 finalist。
- 不把“差异区间包含 0”写成已证明相等；“均值近似”也不能写成非劣性已建立。

预算选择和 freeze 脚本当前没有独立强制检查全部预期运行都到齐。运行选择前核查 full 阶段 27 条曲线；冻结前核查 9 人 × 3 折 × 17 条件的训练/量化记录覆盖。成功写出配置不等于实验矩阵完整。全通道 500-epoch 检查点用于预算提取；失败恢复按完整 run 重做，不是优化器/epoch 级恢复。

### 4.3 补充实验

- 来源固定为 `original_bundles` 或 `independent_reproduction`；缺原权重只能显式建立 `--rebuild-bases` 计划。
- 所有被试、3 个种子、两个历史 pruning bases 均保留，不为更好结果筛选子集。
- `base` 是基础训练；`matched0/25/50` 都额外训练 100 epoch，使用新建 Adam、学习率 `1e-4`、相同 seed 和 batch 顺序规则。
- `25/50 − 0` 是预算匹配的剪枝比较；`25/50 − base` 混合额外训练影响。
- 协议保存原配置、排名、核心源码哈希及执行路径；改变被固定的代码会使旧计划拒绝继续。
- 需要新设计或新代码时，创建新结果/模型目录和新计划，保留旧计划。不能修改旧哈希字段来“修复”保护检查。
- 新计划需要参考冻结配置、原 rankings/runs；要比较历史预测还需参考 `test/`。移动后的旧绝对路径仅作来源记录，不假定可直接续跑。
- `evaluate` 保存主/次分析指标，默认补充 `analyze` CSV 汇总主分析 accuracy；不能声称它覆盖全部次分析指标。

## 5. 张量、模型与部署契约

### 5.1 数据

固定协议是 float32 `X[N,22,500]`，250 Hz、微伏、4 类。类别顺序是 left_hand、right_hand、feet、tongue；通道顺序见 `data.EEG_CHANNELS`，不能自行按字母排序。

原始 `.mat` 的 trial 索引从 1 开始，代码转成 0-based。cue 在 marker 后 2 秒，截窗起点是 marker 后 2.5 秒，长度 2 秒。必须按 run 因果 `sosfilt`、每个 run 零状态，然后截窗；不能换成双向零相位滤波并继续声称原在线可用协议。

NPZ 字段包括 `X/y/run/artifact/first_window_s/source_sha256`。预处理保留 artifact 标记；不要在缓存阶段删掉这些试次，以免无法执行次分析。

`ChannelStandardizer` 的 mean/std shape 为 `(1,C,1)`，在 trial/time 轴计算。现有实现没有对零标准差、零功率或恒定 Fisher 特征统一添加 epsilon；不要把它描述成接受任意输入的健壮通用 EEG 服务。新增数值保护会改变计算，应作为明确的实现变更验证。

### 5.2 通道选择和网络

`selection.py` 中 μ/β 频带分别为 `[8,13)`、`[13,30)`；relevance 是 Fisher score 两频带均值再 min-max，redundancy 是平均绝对 Spearman 相关；贪心惩罚 0.5。

排名只决定选谁，`condition_subsets()` 会将选中索引按 montage 顺序排序后输入网络。bundle 的 `channels`、归一化向量和 ONNX 输入列必须同序。三个随机对照来自固定 seed 690，跨被试共享；8/16 通道随机子集分别采样，不保证嵌套。

EEGNet 默认 F1=8、D=2、F2=16、时间 kernel 64、dropout 0.5，四类 logits。选择电极减少输入 C；结构剪枝减少 F2，二者不是同一个操作。

基础训练使用 Adam、batch size 64、学习率 `1e-3`，按固定 epoch 预算执行，无 early stopping。`set_seed()` 设置 Python、NumPy 和 torch 种子；batch shuffle 使用 CPU generator。训练完成后模型移回 CPU，不能据此误判训练设备。

剪枝必须实际缩小 `conv_separable_point`、`bnorm_2` 和 classifier。分类器特征为 filter-major 展平，切片要保留对应关系；用零 mask 不等价于当前结构剪枝。

`deploy_copy()` 特意新建模型再载入状态并移除 max-norm parametrization。不要改成对 deepcopy 直接移除参数化：共享的动态注入类可能破坏原模型。

### 5.3 Bundle 与 ONNX

bundle 必需字段是 `state_dict/F2/channels/mean/std`，调用方添加 subject、split、condition、seed、epochs、device 等来源。加载使用 `torch.load(weights_only=True)`，按通道数与 F2 重建网络；现有 bundle 未把任意 n_times/类别配置通用化。

ONNX 契约：opset 18；`eeg: float32[N,C,500] → logits[N,4]`；仅 batch 动态；归一化和选通道在图外完成。INT8 是 signed QDQ 静态 PTQ、per-channel weights、MinMax calibration，不是 QAT，也不是整个图和外部输入都变成 int8。

模型相关更改至少检查：PyTorch logits、FP32 ONNX parity、INT8 有限输出、真实输入 dtype、bundle 重载与剪枝维度。修改结构后不能只检查导出文件存在。

## 6. 历史产物与恢复规则

历史 `results/frozen_config.json`、validation/final/test 原始记录、benchmark、数据来源表、原报告快照和 `docs/evidence/baseline/` 都应作为来源保留。需要更新派生表或报告时，依据明确任务重新生成并保留来源，不能修改原结果使新表“看起来一致”。

`artifacts.completed(record, artifacts, expected)` 的语义：

| 状态 | 行为 | 正确操作 |
| --- | --- | --- |
| 记录不存在 | 返回 False | 在已授权实验中执行该任务 |
| 记录有效，指定元数据匹配，产物非空 | 返回 True | 复用；仍不等于反序列化/哈希验证 |
| 记录存在，产物缺失或为空 | 抛 `ArtifactError` | 找回原产物或独立复现 |
| JSON 损坏/类型错误/元数据冲突 | 抛 `ArtifactError` | 保留原记录并调查，不删除后静默重训 |

写新 JSON 优先使用 `artifacts.write_json()`：严格拒绝 NaN/Infinity，同目录临时写入、fsync 和原子替换。它提供完整文件写入，不提供“任意文件禁止覆盖”策略；调用方仍需按任务语义检查是否允许覆盖。

`require_files()` 只验证存在与非空；`sha256()` 只返回摘要。要宣称模型完整性，必须说明实际检查了哪些摘要、是否重载和执行推理，不夸大函数的保障范围。

被冻结目录拒绝再次选预算或运行汇总/冻结；已有 benchmark 拒绝覆盖。新实验同时隔离 `MIBCI_RESULTS_DIR` 与 `MIBCI_DATA_DIR`，否则可能隔离了 JSON 却共用模型路径。

Git 历史从 2026-11-08 的首次上传开始；此前的变更只记录在 `docs/evidence` 和 WORK_LOG。编辑前副本不要覆盖 `docs/evidence/baseline/` 作为新“基线”。不要为普通说明任务清理结果、删除大文件或重建交付包。仓库是公开的：`data.nosync/`（原始/处理 EEG 和全部模型）、虚拟环境和 `release/` 不进仓库；提交前确认没有密钥或大文件。模型以 ZIP 附件发布在 Release `models-v1`，附件内的 SHA256SUMS 是核对依据；不要替换已发布的附件，新模型另开 Release。

## 7. 环境、路径和运行策略

安装和最小验证：

```bash
UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --locked
.venv.nosync/bin/python -m unittest discover -s tests -v
python3 scripts/audit_project.py --deep --output /tmp/mibci-agent-audit.json
```

数据根目录 `MIBCI_DATA_DIR` 默认 `data.nosync`；模型是其中的 `models/`。实验结果 `MIBCI_RESULTS_DIR` 默认 `results`。环境变量在 import 时求值，必须在启动进程前设好。

`MIBCI_DEVICE=legacy` 使用可用的 MPS 处理被试 1–6，CPU 处理 7–9；没有 MPS 时回落 CPU。`cpu/mps/cuda` 是显式覆盖；不可用 GPU 应报错而不是伪装成功。保留设备、seed 和库版本记录，不承诺跨硬件 bitwise equality。

按任务选择执行范围：

- 文档、索引、注释修订：检查文件/链接/参数，不启动数据下载或训练。
- 核心逻辑修订：先小模型/合成输入测试，再按需要在独立目录跑真实数据验证。
- 完整复现：使用 README 的分步命令，先验证和冻结，再最终训练、评估、benchmark、分析。
- 冻结后中断：续跑对应阶段；不要重新执行预算选择/冻结。
- 产物恢复：先定位原始文件、核对来源；重新训练属于独立复现。

不并发运行同一个 `(subject, fold/seed, condition)` 输出任务。`ProcessPoolExecutor` 的 worker 数与每进程 torch 线程需控制；不同被试任务可并行，产物存在检查与写入并不是跨进程分布式锁。

### 已知的路径例外

| 脚本 | 例外与处理 |
| --- | --- |
| `run_all.sh` | 最后固定 `--save results/demo/replay.mp4`；严格隔离时改用分步运行，显式传新视频路径 |
| `replay_demo.py` | `--save` 不自动创建父目录；先 `mkdir -p` |
| `pilot_baseline.py` | 默认 `--out` 固定根 `results/pilot`，单独传新目录 |
| `audit_project.py` | 读取 `--root/results`；不认 `MIBCI_RESULTS_DIR`；默认输出固定，独立审计要显式传 `--root --data-dir --output` |
| `prepare_data.py` | 最后一条输出用项目根 `relative_to`；外部结果目录可能在写入后打印失败 |
| 报告/演示/打包/维护生成器 | 多数按源码定位固定项目根目录，不因实验变量自动改读新实验 |
| `run_matched_control.py` | protocol 保存绝对基础模型路径，移动后不能直接假定原计划可恢复 |
| `package_release.py` | 根文件白名单尚不含本轮新增 `agent.md`/`AGENTS.md`，历史 ZIP 不自动同步 |

多个脚本没有 argparse；在检查源码之前，不能把 `--help` 当作无副作用探测。`complete_supplement.py` 是原收尾任务专用编排器，会等待 PID 并运行多个产出步骤，不是普通的状态查看命令。

## 8. 验证要求与测试地图

现有测试使用标准库 unittest，没有项目专用 pytest、lint 或类型检查配置。不要声称运行了未配置或未执行的 CI。

| 测试文件 | 项数 | 覆盖 |
| --- | ---: | --- |
| `tests/test_artifacts.py` | 3 | 孤立记录保护、元数据/非空检查、损坏 JSON 与非有限写入 |
| `tests/test_pipeline.py` | 5 | session-2 限制、run folds、归一化隔离、真实剪枝、ONNX parity/INT8 输入契约 |
| `tests/test_supplement.py` | 2 | 缺原权重必须明确独立复现、历史排名变化不能复用计划 |

按范围运行：

```bash
.venv.nosync/bin/python -m unittest discover -s tests -p 'test_artifacts.py' -v
.venv.nosync/bin/python -m unittest discover -s tests -p 'test_pipeline.py' -v
.venv.nosync/bin/python -m unittest discover -s tests -p 'test_supplement.py' -v
```

`test_pipeline` 会生成临时随机小模型及 ONNX，不需要真实数据或主实验权重。FP32 导出测试的容差是 `atol=1e-5, rtol=1e-4`；INT8 检查输出形状和有限性，不要求 logits 精确等于 FP32。

| 改动范围 | 合适的验证 |
| --- | --- |
| README/agent 文档 | 本地链接存在、命令参数与源码一致、路径和覆盖行为准确、数字回溯到数据源 |
| 产物保护逻辑 | artifacts 测试；确认失败前后旧记录字节保持 |
| 数据/划分/归一化 | pipeline 测试；真实数据契约核验，不把测试标签用于拟合 |
| 模型/剪枝/ONNX | pipeline 测试；受影响 bundle 重载、维度、parity 与输入 dtype |
| 补充协议 | supplement 测试；确保旧计划在哈希变化时拒绝执行 |
| 历史结果解释 | `audit_project.py --deep`，并直接核对相关 CSV/JSON |
| 报告/PDF | 重建受影响输出，核对图表来源，PDF 逐页视觉检查 |
| 幻灯片 | 重建 HTML，在浏览器检查页数、讲稿、图片和视频 |
| 交付包 | 新文件名打包，检查 manifest 哈希、模型数量、来源和排除项 |

不为纯格式修改新增镜像测试。修复会影响科学结论的错误时，应增加针对行为或不变量的测试，并标明其对应的新实验来源。通过必要验证后，不无理由反复跑耗时流程。

普通审计成功主要说明历史覆盖、表哈希和指标一致。`--deep` 在各副本都能通过；`--require-artifacts` 在原 M4 副本（或解压原模型 ZIP 后）通过，在只有补充模型的副本上会因原模型缺失失败。这是两个不同检查目标，不能把后者当作需要删除历史记录才能修复的 bug。审计本身不验证原始训练发生、模型反序列化或 MCU 部署，也不自动审计补充模型。

## 9. 报告与结论写作约束

引用结果时先定位最接近原始计算的来源：

| 内容 | 首选来源 |
| --- | --- |
| 数据数量/来源 | `results/data_summary.csv`、processed manifest、恢复校验记录 |
| 固定方案和统计计划 | `results/frozen_config.json`、`validation/budget.json` |
| 主/次准确率 | `analysis/conditions_primary.csv`、`analysis/conditions_secondary.csv` |
| 个体差异 | `analysis/per_participant_primary.csv`、`maintenance/participant_differences.csv` |
| 配对差异/区间 | `analysis/comparisons_primary.csv` |
| 性能 | `benchmark/benchmark.json`、`analysis/efficiency.csv` |
| 补充剪枝效果 | `supplementary/matched_budget/paired_comparisons.csv`、`supplementary/matched_budget/analysis.json` |
| 环境、恢复和完整性 | `docs/evidence/` 中对应检查 JSON 与日志 |

除已写出 `results/` 或 `docs/` 前缀的路径外，上表结果路径相对于 `results/`。

保持以下措辞准确：

- 65.01% 与 65.04% 的均值近似，不等于等效；4/9 人下降超过 2 个百分点。
- proposed 相比强对照没有明确优势证据，不概括成算法绝对无效或严格相同。
- 参数个数、initializer 字节、完整 ONNX 文件字节分别报告；最终 INT8 的 2,420 bytes 不是 2,420 个模型参数。
- 历史全通道 INT8 在 M4 CPU 上约慢 28%；不要把压缩自动写成加速。
- 理想激活比、外部 float32 输入缓冲、进程 RSS、MCU SRAM 是不同指标。
- 现有图不是全整数图；ELU、pool 等浮点路径和 QDQ 开销不能忽略。
- 亚毫秒是模型调用耗时；不包含 2 秒观察窗口和 cue 时间。
- 回放是 prerecorded EEG replay，不是真实闭环或临床验证。
- 补充精度差 −1.56 pp 指 relevance-16、INT8 条件的等预算 50% vs 0%，不要泛化到所有条件或误写成相对基础 `base`。
- 相同预测、相同 seed 或相近准确率都不能证明恢复了原权重。

手工维护文档与生成文档分清：`README.md`、`agent.md` 为人工维护入口；`docs/REPORT.md` 来自 `build_report.py`；PDF 来自导出脚本；`presentation/index.html` 来自 deck 源文件。持续修订改源文件再重建，不仅修改生成物。

## 10. 交付完成前的检查表

- 改动落在任务范围内，历史结果/权重没有被静默替换。
- 文档命令指向存在的入口和正确环境，说明必要前置条件与输出路径。
- 更新了受影响的文档/生成物，或清楚注明未重建的历史快照。
- 只宣称实际执行并通过的测试/审计；新实验、旧记录、文件存在性和模型验证分别说明。
- 新包使用新名称，清单与包内真实内容一致；不能沿用旧 ZIP 的验证结果为新包背书。
- 向用户简明报告“改了什么、为何这样改、如何验证、还有哪些限制”，提供可点击的文件路径。

本指南首次整理时，现有 10 项测试和 459 份主测试记录的深度一致性审计均通过；未重跑完整训练。后续维护需按实际改动重新选择检查，不直接复用这句话作为新验证记录。
