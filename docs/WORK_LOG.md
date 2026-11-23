# 项目补全工作记录

日期：2026-11-04（America/New_York）。执行者：Codex，经项目作者授权补全。

## 工作边界

- 原实验已在作者的 Apple M4 Mac mini 上完成。当前副本缺少 `data.nosync/` 和虚拟环境，不等同于原实验未完成。
- 保留原始冻结配置、验证记录、最终测试记录和性能测量。追加分析不能重新选择最终模型，不能伪装成测试前预注册的实验。
- 补充实验使用独立目录。原始权重缺失时，不通过静默重训替换原模型。
- 只记录实际执行的检查；代码完成、实验执行、结果确认分开标记。

## 初始状态和实施计划

1. 已检查：训练/量化/测试均有 459 个记录；原始数据、权重及 ONNX 不在当前副本；`.venv` 是指向缺失 `.venv.nosync` 的符号链接。
2. 已向作者询问原 Mac mini 上 `data.nosync` 的可读取路径；等待回复期间继续独立工作。
3. 正在使用 `UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --locked` 恢复锁定环境。
4. 计划修复产物恢复检查，增加路径隔离和完整性审计。
5. 计划增加未剪枝模型相同 100 epoch 微调的补充对照，明确依赖原权重及其来源。
6. 计划从已有证据补充逐被试/类别结果解释，修正参数字节、文件大小和理论激活内存的表述。
7. 计划补全本地报告、演示入口及复现/交付说明，并记录测试结果。

## 证据保存

- `evidence/before.sha256.json`：修改前源码与关键历史证据的 SHA-256。
- `evidence/baseline/`：本次维护所涉及源码及演示源文件的原始副本。
- 后续生成变更清单、差异补丁和验证输出，便于复核本次具体修改。

## 执行记录

### 已完成的环境与数据恢复

- 使用原 `uv.lock` 执行锁定环境安装，恢复 `.venv.nosync`，保留既有 `.venv` 链接；没有升级依赖。
- 新环境成功导入 PyTorch 2.15.0、Braindecode 2.0.0、ONNX Runtime 1.31.1；检测到可用 MPS，确认 22 通道模型有 2,420 个参数。
- 重新下载 18 个原始 `.mat`，逐文件 SHA-256 全部匹配原 `results/data_summary.csv`，重建 18 个预处理缓存。
- 下载/缓存命令使用 `MIBCI_RESULTS_DIR=results/maintenance/reproduction`，没有改写原数据汇总。
- 当前环境依赖冻结、硬件信息另存 `evidence/environment_freeze.txt` 和 `evidence/current_environment.json`。原硬件为 M4 Mac mini 来自作者说明；不把当前 RAM 观测无依据写入原日志。

### 已完成的代码补全

- 新增 `src/mibci/artifacts.py`：原子 JSON 写入、非有限值拒绝、文件缺失/空文件检查、恢复配置一致性检查和 SHA-256 工具。
- 训练、剪枝、导出、测试的恢复逻辑同时检查必需产物；已有 JSON 而缺模型时拒绝静默重训。模型存在性检查不冒充反序列化验证。
- 防止空模型目录被误当作零任务成功；保留验证预算模型和历史通道排名，不在恢复时无条件改写。
- 增加 `MIBCI_DATA_DIR` / `MIBCI_RESULTS_DIR` 隔离路径。默认设备仍保持原被试分配；显式设备覆盖仅用于有记录的新实验，没有验证 NVIDIA CUDA 运行。
- 冻结后禁止重新选择训练预算、改写验证汇总/冻结配置；主流程要求新目录独立复现。历史性能测量也不能被直接覆盖。
- ONNX 导出增加基于训练样本的数值容差检查。
- 性能代码区分真实 float32 接口输入和理论 INT8 张量；保留理论激活峰值的假设与局限。增加不导入 PyTorch 的独立进程 RSS 测量脚本。
- 修复电源记录在非 macOS 的报错及“无电池字符串就判 battery”的误判，未知电源明确标 unknown。

### 历史证据核验

- 全量重算 459 个测试 JSON 的 FP32/INT8、主/次分析准确率、macro-F1、类别召回和混淆矩阵；与保留的统计一致。
- 验证/最终训练及量化记录各 459 个，完整配置网格无缺失；冻结验证表哈希匹配。
- 结论：历史证据内部一致，但原权重/ONNX 仍未恢复，二者是不同的完成状态。
- 最终方案有 4/9 人（P1、P4、P7、P9）下降超过 2 个百分点。本次修正此前聊天中误说的 3/9；不能用总体均值替代逐人结论。
- 首次串行读取因 iCloud 按需下载很慢而中止，随后并发读取遇到 `OSError 89 Operation canceled`；增加仅针对瞬时读取错误的有限重试后通过。保留失败与成功日志，不把中断当成功。
- 全量结果见 `evidence/deep_audit_final.log` 与 `../results/maintenance/audit.json`。

### 报告和演示补全

- 新增可再生成的本地英文研究报告 `REPORT.md`，连接原始表格、图区、协议和复现说明。
- 从保留 CSV 生成逐被试差异和逐类别召回图，明确种子先平均到被试、类别结果仅作描述。两图已实际查看，无裁切或标签遮挡。
- 区分初始化参数字节约缩小 4 倍、完整 ONNX 文件只缩小约 7.1%、激活峰值约 5.5 倍仅为理想化张量估算。
- 修正“缺少 integer kernel”的过度推断：历史执行图实际包含 QLinearConv；能支持的是量化分组空间卷积耗时占主导。
- 修正“均值相近就证明未过拟合/完全等价”等表述，保留个体损失和区间。
- 保留 17 页幻灯片的原设计/顺序，把私有 `/_blob/` 图片和视频链接替换为项目资源，构建 `presentation/index.html` 和本地 assets。
- 在浏览器实际检查封面、准确率图、最终结果页、讲稿切换及导航；全部 5 张图片已加载。没有把源码检查等同于视频完整播放测试。
- README 更正 Apple M4 laptop 为作者实际使用的 M4 Mac mini，并加入本地报告、演示、复现与工作记录入口。

### 自动检查和补充实验进度

- 初始 8 项有意义的测试通过，覆盖恢复保护、JSON 写入失败保护、session-2 访问锁、按 run 划分、训练集归一化、真实结构剪枝、FP32 导出一致性和 INT8 float32 输入契约。
- 新增补充协议测试时，一项断言只接受 RuntimeError，但实际缺失原权重会先抛 FileNotFoundError；放宽到接受这两种明确失败，继续检查不生成误导协议/不静默改成重建。
- 全量 Python 源码通过语法解析，`run_all.sh` 通过 `bash -n`。
- 对缺权重恢复、冻结主流程、重复冻结和历史 benchmark 覆盖执行了真实 CLI 反例检查，全部按预期以非零状态拒绝，没有触发训练或改写旧证据；日志保留。
- 补充计划 v1 尚未执行时，优化了独立文件的读取并发，因此源码哈希变化；旧计划整个目录归档为 `results/supplementary/matched_budget_unexecuted_v1`，在新建的主补充目录重新固定计划。旧计划没有训练或评价结果，不伪装为已执行。
- 已固定新协议：54 个独立重建基础模型，162 条相同预算微调分支；保持原 1–6 MPS、7–9 CPU 分配，分别以 3 / 6 个 worker 启动。训练日志见 `evidence/matched_train_mps.log`、`evidence/matched_train_cpu.log`。
- 补充训练/评价尚在进行；本节后续会记录实际完成时间、统计结果及任何失败。不存在分析文件时，本地报告明确标记 pending。
- 作者回复不知道原 `data.nosync` 路径，因此继续已固定的独立重建路线，不要求作者继续寻找文件。
- 新环境实测为 Apple M5 / 24 GiB / Mac17,3，与历史 M4 Mac mini 不同。补充运行不得描述成在原 M4 上恢复的原权重。相同 CPU/MPS 分配并不消除硬件差异。
- 初批 M5 模型约需 7–10 分钟/个，已向作者说明全量补充按小时计；继续执行固定网格，不因训练速度或结果更换种子/筛选被试。
- 扩展后的 10 项测试全部通过（`evidence/tests_verified.log`）。原始证据快照中的 11 个历史结果文件通过 SHA-256 不变性检查。
- 18 个恢复缓存的维度、有限值、原始文件哈希、run/class 覆盖、artifact 数量全部通过检查；session-2 标签与 artifact 标记也逐人匹配历史测试记录。见 `evidence/restored_data_checks.json`。
- 新增便携推理命令；输入契约明确要求项目预处理后的 22×500 窗口，仅使用存储的训练归一化，避免把原始 EEG 误当已过滤窗口。
- 使用独立文档运行环境导出 9 页初版 PDF，查看了全部页面的渲染拼图；表格、图和脚注未见遮挡。一次渲染早于导出结束导致读取到未完成文件，已改为 PDF 原子替换并在导出完成后顺序重渲染。最终补充结果更新后还会重新检查。
- 新增即时本地后处理流水线 `complete_supplement.py`：等待本轮训练、全网格评价、统计、全量补充预测校验、独立进程 RSS、便携推理一致性核验、报告/PDF 重建。它不是定时任务，不发送消息，不修改原最终方案。阶段状态写入 `evidence/completion_status.json`，最后停在待人工视觉检查状态。

### 实际命令与日志

```bash
UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --locked
MIBCI_RESULTS_DIR="$PWD/results/maintenance/reproduction" .venv/bin/python -u scripts/prepare_data.py --expected-summary results/data_summary.csv
python3 -u scripts/audit_project.py --deep
.venv/bin/python -u -m unittest discover -s tests -v
.venv/bin/python -u scripts/run_matched_control.py --stage plan --rebuild-bases
.venv/bin/python -u scripts/run_matched_control.py --stage train --subjects 1 2 3 4 5 6 --workers 3
.venv/bin/python -u scripts/run_matched_control.py --stage train --subjects 7 8 9 --workers 6
.venv/bin/python -u scripts/build_report.py
python3 scripts/build_presentation.py
python3 -m http.server 8765 --bind 127.0.0.1
```

命令输出、失败尝试、环境快照及最终验证日志保留在 `docs/evidence/`。浏览器只访问本地预览，没有发布或发送项目材料。

## 本轮完成与保留限制

- 已完成软件修复、锁定环境/原数据恢复、历史证据核验、完整补充实验、内存测量、便携推理验证、本地报告和演示交付。
- 原 M4 的原始权重/原 ONNX 仍没有找回；交付的是来源明确的 M5 独立重建模型，不能冒充原文件恢复。旧结果完整保留。
- 补充分析是在旧测试结果已知后追加；没有重新挑选原最终模型，也没有增加独立被试。MCU 和真实在线闭环不在本软件研究的完成声明内。
- CUDA/Linux 训练没有实机验证；MIBCI_DEVICE 的显式覆盖不是经过验证的远程服务器部署承诺。
- 最终变更记录见 evidence/change_inventory.json 和 evidence/changes.patch；交付 ZIP 有独立逐文件 SHA-256 清单。

## 补充实验执行完成（自动流水线记录）

完成时间：2026-11-04T20:06:22.457720-05:00。54 个独立重建基础模型及 162 条等预算分支均完成；216 个模型分别导出/量化并评价 FP32/INT8。

全量补充预测指标重算通过。基于原固定被试、种子和通道列表，没有重新选择最终模型。新运行环境为记录中的 M5，历史 M4 性能结果未替换。

最终候选对应的 relevance-16 INT8，50% 剪枝相对不剪枝、同样微调 100 epoch 的配对差异为 -1.558 个百分点，95% 探索性区间 [-2.864, -0.417]。这是事后补充复现实验，不是新增独立被试或原预注册结果。

已执行独立进程 RSS 测量及可重定位推理 CLI 核验，FP32/INT8 预测与评价记录完全一致。报告 Markdown/PDF 已用补充结果重建，仍待最终逐页视觉检查与交付包校验。相关命令与输出保存在 docs/evidence。


## 最终收尾与验证

- 54 个基础模型、162 条等预算分支全部完成；216 个模型均有 FP32/INT8 导出与测试结果。每个 bundle 的 SHA-256 与训练记录全部匹配。
- 所有补充主/次分析准确率、macro-F1、混淆矩阵和召回率已从预测重算；配对均值、协议引用和冻结数值源码哈希再次核对通过。
- 10 项自动测试在交付前重跑通过；文档本地链接无失效，Python 源码解析通过。
- M5 补充结果：relevance-16 INT8 50% 剪枝相对相同微调预算的未剪枝模型低 1.558 个百分点，95% 探索性区间 [-2.864, -0.417]。已回填摘要和正文，并新增第 18 页演示中的补充结果页（序列第 15 页）。
- 324 个可比模型/精度对的历史预测一致率平均 99.40%，范围 96.18–100%；不把预测一致当成权重逐位相同。
- 独立 ORT 进程峰值约 57–59 MiB，包含解释器/运行时开销，不能确认理论 5.5 倍激活张量缩减，也不代表 MCU SRAM。
- PDF 已生成 11 页并进行逐页渲染检查；修正了 Precision 表头换行和补充表跨页布局，使用大字版历史图提高可读性。最终视觉记录见 evidence/pdf_visual_review.json。
- 演示新增等预算对照页并检查实际浏览器显示；所有图片来自本地资源。保留原回放视频；本轮未声称完成真人在线闭环测试或完整视频观看测试。
- 完成 2,413 个历史结果文件的内容哈希清单；初始证据快照与最终交付的历史结果保持一致。
- presentation/deck_source/deck.json 是既有文件，但最初没有保存其逐字节快照；变更清单明确记录“修改、无原始字节哈希”，只声明插入 matched 页及格式化，不伪称新增文件或精确 before diff。

交付包命令：

```bash
python3 scripts/record_maintenance.py
python3 scripts/package_release.py --output release/COMP690_completed_2026-11-04.zip --model-dir data.nosync/models/supplementary/matched_budget
```

逐文件清单在 ZIP 内的 RELEASE_MANIFEST.json；ZIP CRC、每个文件的 SHA-256 和 ZIP 本身校验结果单独写到 release/package_verification.json，避免把 ZIP 的自身哈希放入 ZIP 造成循环。原始/缓存 EEG 与虚拟环境不进入交付包，数据可按原始 SHA-256 重建。没有创建远程仓库、发布站点或发送材料。

## 2026-11-08 期中准备：新增“下一步”页

- 应用户要求新增 `presentation/deck_source/slides/nextsteps.html`，放在 limits 与 takeaways 之间。演示由 18 页变为 19 页：新页是第 18 页，takeaways 顺延到第 19 页。同步更新 `deck.json` 的顺序和 s4 描述，并用 `python3 scripts/build_presentation.py` 重建 `presentation/index.html`。新页 id 用 `nextsteps`，避免和翻页按钮的 `id="next"` 冲突。
- 新页只列计划，不含新结果。必做：根据期中反馈收尾报告和幻灯片。追加 A：只把分组空间卷积留在 FP32，测混合精度时延，在 M5 上测，与 M4 历史数字分开报告。追加 B：先去掉所有电极共有的功率波动再度量冗余，重点看 8 个电极。两项都在冻结计划之后，第二天的结果只能算探索性。页脚和讲稿都写明尚未运行。
- 验证：重建后共 19 页，每页都有讲稿，7 个本地资源都在，页脚编号正确。用无头 Chrome 截图检查了新页；从第 17 页点翻页按钮，依次到第 18、19 页。回放视频页没有改动，这次没有在浏览器里播放视频。
- 新增中文期中讲解页 `docs/MIDTERM_REVIEW_GUIDE_ZH.html`。README 与 agent.md 里的页数改为 19。
- 没有运行实验，没有改动 `results/` 或任何模型。没有重建 `release/COMP690_completed_2026-11-04.zip`，包内仍是 18 页版。没有重跑 `record_maintenance.py`，`docs/evidence` 里的变更清单仍停在 11-04。

## 2026-11-08 上传到 GitHub

- 应用户要求，把项目上传到用户新建的公开仓库 https://github.com/lipeiye/comp690-final-project （推送前仓库为空）。在项目根初始化 Git，分支 `main`。Documents 由 iCloud 同步，所以 Git 数据放在 `.git.nosync/`；根目录的 `.git` 只是一行 `gitdir: .git.nosync` 的文本文件，做法与 `data.nosync`、`.venv.nosync` 一致。
- 沿用原有 `.gitignore`：`data.nosync/`（原始/处理 EEG，以及 216 个 `.pt`、432 个 ONNX 补充模型）、虚拟环境、`release/` 交付 ZIP、`reproduction.nosync/` 和 Python 缓存都不上传；新增忽略 `.DS_Store` 和 `.git.nosync/`。首次提交 3,044 个文件：源码、脚本、测试、锁文件、文档与报告、`results/`、演示文稿与回放视频、课程提案。
- 推送前检查：最大文件是两份 11.3 MB 的回放视频；在暂存文件中搜索常见密钥格式、密码赋值、个人邮箱和机器序列号，没有命中。
- README 开头加了仓库地址，并说明克隆后没有数据和模型、需要按第 7、8 节重建；agent.md 的 Git 状态说明同步更新。
- 没有运行实验，没有改动 `results/`、模型或演示文稿。没有添加 LICENSE：公开仓库没有许可证时，别人只能查看，默认没有复用授权，是否添加由作者决定。

## 2026-11-09 模型发布到 GitHub Release

- 应用户要求（“传模型”），把模型作为 GitHub Release `models-v1` 的附件上传。
- 发现：本机是原 M4 Mac mini（Mac16,10，16 GiB），`data.nosync/models` 里是 2026-11-03 主实验的原模型：validation 513 个 `.pt` + 918 个 ONNX，final 459 个 `.pt` + 918 个 ONNX，共 2,808 个文件。11-04 的维护在 M5 副本上进行，那份副本没有 `data.nosync/`，所以 README、agent.md 和本日志当时记为原权重缺失。11-08 条目说本机 `data.nosync/` 里是 216 个补充模型，这是照搬了 M5 的状态，写错了；本机没有补充模型目录。
- 核对（只读；脚本放在会话临时目录，不写项目）：459 个最终模型在 session 2 重算 FP32/INT8 预测，918 组、264,384 次预测与 `results/test` 逐条一致，bundle 元数据和通道列表也一致；459 个验证模型在留出 run 上的混淆矩阵、loss（误差 1e-5 内）和 ONNX 文件大小与 `results/validation/quant` 一致；54 个预算选择文件能以 `weights_only=True` 载入；`audit_project.py --deep --require-artifacts` 通过，输出写到临时目录，没有改 `results/maintenance/audit.json`。
- 打包：`comp690-models-original-m4.zip` 含 2,808 个文件，34,620,542 字节，SHA-256 `e16773335a1315b932c252efe1aa877b3eeca55aa62307985960466604a42e49`。`comp690-models-supplementary-m5.zip` 含从 `release/COMP690_completed_2026-11-04.zip` 取出的 216 个 `.pt` + 432 个 ONNX，6,715,548 字节，SHA-256 `4335a20dfb303e004b1b9aadbac69e93bb4cdcf8eeb01a8c4c47b7b156f3c2f6`；每个文件的 SHA-256 与交付包清单、`results/supplementary/matched_budget` 中 432 条记录的 `model_sha256` 一致。两个 ZIP 根目录各有一份 SHA256SUMS，解压到 `data.nosync/` 后路径与本地一致。
- 隐私检查：3,456 个模型文件里没有邮箱、密钥或 token。ONNX 的 PyTorch 导出元数据含本机路径 `/Users/lipeiye/Documents/04-UNC/Fall 2026/comp690/project/...`，仓库里已有 396 个文件含同一路径。模型保持原字节，没有删改元数据。
- 文档：README 第 1 节状态表、目录图、第 6.1 节下载步骤和几处“原模型缺失”的说法；agent.md 的当前产物、机器识别、审计说明和 Release 规则；REPRODUCIBILITY.md 的恢复原权重一节；BEGINNER_GUIDE 第 11 节。没有改 `results/`、模型、演示文稿或 11-04 交付包。`docs/MIDTERM_REVIEW_GUIDE_ZH.html` 中“原权重不在了”的说法没有改。
- 应用户要求改第 18 页：追加 A 原写“在 M5 上测，与 M4 数字分开”，改为在同一台 M4 上用原模型测 FP32、INT8 和混合版本，讲稿同步。改的是 `deck_source/slides/nextsteps.html`，然后用 `build_presentation.py` 重建，仍是 19 页。用无头 Chrome 截图检查了第 18 页排版。讲解页里扩展 A 仍写“用 216 个补充模型在 M5 上测”，没有改。

## 2026-11-09 期中版幻灯片

- 课程通知：期中评审每人 12 分钟，硬性截止，含 3–4 分钟问答；要讲清问题、方案、系统原则、新意、期中进度、与承诺时间表的对照和下一步。19 页完整版讲不完，也缺“新意”和“时间表”两页。
- `build_presentation.py` 加了 `--deck` 和 `--output`。不带参数时仍由 deck.json 生成 index.html，重建结果与之前逐字节相同；换页序必须给 `--output`，且不能写 index.html，两种误用都试过会被拒绝。页脚页码本来就按位置重写。
- 新增 `deck_midterm.json`（15 页）和三页源码：`cover_midterm`（期中封面）、`landscape`（EEGNet、Q-EEGNet、MI-BMInet、mRMR，以及本项目新增之处）、`progress`（逐项对照 proposal 第 6 节时间表：计划 11-03 冻结，报告与演示进行中）。其余 12 页沿用完整版。构建为 `presentation/midterm.html`，另用无头 Chrome 导出 `presentation/midterm.pdf` 备用；截图检查了全部 15 页。
- 进度页原写“pilot check of INT8 operators”。核对后发现试跑只有 FP32，INT8 算子检查记录在 `results/benchmark/benchmark.json`，改成“export and INT8 operator check”。
- README 第 1、3、9、11 节和 agent.md 的演示文稿一行已更新。没有改结果或模型，19 页完整版没有变化。

## 2026-11-09 期中 PPTX（UNC 风格）

- 用户要求：按老师的建议少放字；以前常讲不完，要更短；字体改用 UNC 官方字体，配色尽量用 Carolina Blue；最后做成可直接放映的 PPTX。
- 调研：identity.unc.edu/brand/typography 列出的品牌字体是 Field Gothic（标题用，需 Adobe 付费授权）、Open Sans 和 Source Serif，允许用 Oswald、Arial、Georgia 替代。配色取自 its-commons.unc.edu/brand/colors：Carolina Blue #4B9CD3、Navy #13294B，以及 Bolin Creek、Cloud、Old Well Basin 等辅色；Carolina Blue 用作文字时只用于 18pt 以上。identity.unc.edu 的配色页返回 404。
- 产出：新增 `presentation/pptx/build_midterm_pptx.js`（pptxgenjs 4.0.1），生成 `presentation/midterm.pptx`。共 14 页，比 HTML 期中版少了成本表那一页，要点并入其他页；每页只留标题、大数字和 PowerPoint 原生图表，讲稿写在备注里。准确率、MACs、分层时延和等预算差值直接从 `results/` 读取，其余数字引自完整版和 README。
- 检查：pptx skill 的 validate.py 通过；去掉了 ZIP 里的文件夹条目，因为有些云服务会拒收。本机 LibreOffice 找不到 Open Sans 和 Verdana，所以用比 Open Sans 更宽的 DejaVu Sans 另生成一份，渲染全部 14 页检查。修了第 9 页负值条压住类别标签、第 7 页公式折行、第 13 页标题过长。因为 LibreOffice 不认主题字体（Keynote 也可能不认），所有文字都写明字体。
- 未验证：本机没有 PowerPoint 和 Keynote，没有在它们里实际打开过。HTML 期中版（midterm.html/pdf）保留为草稿。

## 2026-11-09 期中逐页讲稿

- 用户要求：给一份逐页讲稿。
- 产出：新增 `presentation/midterm_script.md`，14 节对应 14 页。每节有照着说的英文、中文意思、这一页开始的时刻；时间紧时可跳过的句子放在方括号里；部分页另有被问到时可补充的数字（`If asked:`）。按每分钟 125 词估算：完整约 7 分 25 秒，去掉方括号约 6 分 56 秒；按每分钟 110 词分别约 8 分 23 秒和 7 分 51 秒。
- 讲稿里要念的数字逐个对过 `results/`：比随机组合时 9 人中 8 人更好（16 和 8 个电极都是）；我的规则与只看有用程度选出完全相同电极的人数，16 个电极时 6/9、8 个电极时 7/9（`results/final/rankings/`，各 seed 排序相同）；与只看有用程度、按权重比较的 95% 区间都跨过 0；被剪的是最后一个 pointwise 卷积层的滤波器。
- `build_midterm_pptx.js` 的 14 页备注改为从讲稿读取（每节 `> ` 行加 `If asked:` 行），节号不是 1 到 14 或缺英文会报错。重建后 validate.py 通过；逐页核对备注与讲稿一致、对应页正确；和重建前的 PPTX 相比，除备注外只有时间戳和 pptxgenjs 随机生成的 section ID 不同，幻灯片和图表 XML 未变。
- 未验证：仍没有在 PowerPoint 或 Keynote 里打开过；计时是按词数估算，没有实际朗读计时。

## 2026-11-09 讲稿 Word 版

- 用户要求：讲稿给一份 Word 版。
- 产出：新增 `presentation/pptx/build_midterm_script_docx.js`（docx 9.13.0），从 `presentation/midterm_script.md` 生成 `presentation/midterm_script.docx`：US Letter，8 页。第 1 页是用法说明和第 1 页讲稿，之后每页两页讲稿。英文 13pt，方括号里可跳过的句子排成灰色斜体；中文意思在下；"If asked" 和中文补充放在浅蓝框里；每页标题右侧是开始时刻，标题用 Heading 1，Word 导航窗格可以跳转。英文用 Arial（UNC 允许的 Open Sans 替代），中文用 PingFang SC。讲稿开头两条说明改成不依赖 Markdown 的说法（"英文部分照着说"，"维护"一条提到 Word 版），所以 Markdown 仍是 PPTX 备注和 Word 版的唯一来源；PPTX 备注不受影响。
- 检查：docx skill 的 validate.py 通过，ZIP 里没有文件夹条目；逐段核对 14 页的标题、英文、中文意思、If asked 和补充都与讲稿一致。无头 LibreOffice 只认自带字体，中文显示为空白；用 `FONTCONFIG_FILE` 指向一份加入 /System/Library/Fonts、Supplemental 和 PingFang.ttc 所在目录的配置后，PDF 里嵌入的是真 Arial 和 PingFang SC，逐页看过版面。第一版的 PositionalTab 在 LibreOffice 里不右对齐，改用普通右对齐制表位；初版 9 页留白多，收紧字号和行距后为 8 页。
- 未验证：没有在 Microsoft Word 或 Pages 里打开过。

## 2026-11-09 PPTX 标题改成直白的主题式标题

- 用户反馈：PPT 的标题太奇怪，要简明。
- 改动：12 个内容页的标题从修辞式、结论式改成主题式，对应课程要求的问题、新意、方法、系统原则、结果、进度和下一步：
  - 2 Every electrode costs twice → Why Fewer Electrodes?
  - 3 The pieces exist; I separate their effects → Related Work and Novelty
  - 4 Research question → Research Question
  - 5 Data, and a test day kept locked → Dataset and Evaluation
  - 6 Where EEGNet's cost lives → Systems View: Model Size and Compute
  - 7 Choosing the electrodes → Method: Electrode Selection
  - 8 16 chosen electrodes match 22; 8 lose 3–5 points → Result 1: Accuracy vs. Electrode Count
  - 9 Redundancy beat random, not simpler rules → Result 2: Comparing Selection Rules
  - 10 INT8 kept accuracy but ran slower on this CPU → Result 3: INT8 Quantization
  - 11 Equal fine-tuning reveals a cost of 50% pruning → Result 4: Pruning
  - 12 All planned experiments done; write-up remains → Progress vs. Schedule
  - 13 Next steps → Next Steps
- 第 2 页两张卡片去掉小标签，标题改为 Hardware cost 和 Compute cost，正文改为 "Each electrode needs a contact, an amplifier channel and setup time" 和 "The model's first layer runs once per electrode"。原来写在标题里的结论，现在由各页的大数字和底部一句话承担。
- 讲稿同步：各节标题换成新标题；第 2 页英文开头改为 "Why use fewer electrodes? Because each electrode has two costs. First, hardware… Second, compute…"，中文意思同步。按每分钟 125 词估算总时长不变（约 7 分 25 秒）。PPTX 备注和 Word 版已重建。
- 检查：两份文件的 validate.py 都通过；备注、Word 段落与讲稿逐段一致；用比 Open Sans 宽的 DejaVu Sans 渲染全部 14 页，每个标题都在一行内。
- 未改：HTML 完整版和 HTML 期中草稿仍用原标题。项目根目录另有一份 13:17 的 `midterm_script.md`（未跟踪，不是本次生成），内容是旧标题；按用户要求移到了废纸篓。系统的 trash 命令因权限被拒，改用 mv 移入 ~/.Trash，所以 Finder 的“放回原处”不可用，需要时手动拖回。

## 2026-11-09 讲稿改成自然口语

- 用户反馈：讲稿的语言太奇怪，要自然流畅。
- 改动：14 页英文全部按口语重写，用完整的常见句式和页间过渡，不再用 "Two scores."、"a match"、"What I add is a fair test: one factor at a time, same budget; …" 这类压缩短语和分号列表。例如第 2 页改为 "Each electrode has two kinds of cost. One is hardware… The other is computation…"，第 9 页改为 "These rhythms rise and fall across the whole scalp together, with an average correlation of 0.74."。"If asked" 也改成可以直接说出口的完整回答；"9 人中 8 人赢过随机" 从正文移到第 9 页的 "If asked"。中文意思和补充逐句同步。
- 时长：自然说法字数更多，删减后按每分钟 125 词估算，完整版约 7 分 40 秒（931 词），去掉方括号约 7 分 01 秒（848 词）；按每分钟 110 词分别约 8 分 41 秒和 7 分 56 秒。原来是 7 分 25 秒和 6 分 56 秒。各页开始时刻和用法说明里的检查点（第 7 页 3:15、第 11 页 5:50）已按新词数重算。
- 数字和事实未变，改动只在措辞。PPTX 备注和 Word 版已重建：两份文件的 validate.py 都通过，备注和 Word 段落与讲稿逐段一致；Word 版仍是 8 页，用 Arial 和 PingFang SC 渲染逐页看过。

## 2026-11-12 期中评审后的三个问题与新计划

- 用户要求：解释期中评审留下的三个问题（16 个电极为什么比 22 个好；电极是怎么选的；INT8 为什么在 M4 上慢，试试 INT16 或针对这台机器的改进），检查项目不足，重新列计划并继续推进。计划和结论写在 `docs/POST_MIDTERM_PLAN.md`。
- 边界：以下新实验都在第二天结果已知之后追加，标记 supplementary / post-test，写入 `results/supplementary/` 的新目录；没有改动历史结果、原模型、冻结配置或旧 benchmark。没有修改被旧协议固定哈希的核心模块（`data.py`、`model.py`、`train.py`、`prune.py`、`experiment.py`、`deploy.py`），新功能放在新文件里。
- 本机是原 M4 Mac mini（Mac16,10），原模型都在，环境仍是 PyTorch 2.15.0、ONNX Runtime 1.31.1。

### INT8（问题 3）

- `sysctl` 读到 `FEAT_DotProd=1`、`FEAT_I8MM=1`。读了 ONNX Runtime v1.31.1 的 `qlinearconv.cc`：depthwise 快路径要求每组输入、输出通道都是 1；EEGNet 的空间卷积每组 2 个输出，走的是逐组 im2col 加 GEMM 的通用路径。
- 初测（00:46，训练开始之前，系统负载约 4；P1 seed 0 的 22 电极原模型；单线程；9 轮交替、每轮 400 次，取各轮中位数的中位数）：FP32 275 微秒；原 INT8 353；激活改无符号 356、权重整层量化 351、QOperator 格式 353；INT16 激活 304（执行的是 4 个浮点 Conv）；空间卷积留 FP32 的混合精度 190；分组卷积改写为等价普通卷积后的 INT8 206；调换前两层后 FP32 95、INT8 65。FP32 和原 INT8 两个数与 11-03 的历史 benchmark（280、358）相符。
- 新增 `src/mibci/variants.py`（`spatial_dense`、`spatial_first`、`quantize`、`graph_costs`）、`scripts/int8_study.py`（build / benchmark / analyze）和 `tests/test_variants.py`（3 项，通过）。在 3 个原模型上核对：等价改写的 INT8 输出与原 INT8 逐位相同；调换顺序的 FP32 输出与原 FP32 最大相差 2e-6；`graph_costs` 对原图给出的乘加次数和激活峰值与 `costs.py` 一致。
- 文献核对：MI-BMInet（Wang 等，2024）在网络设计里先做空间卷积；Q-EEGNet（Schneider 等，2020）交错执行前两层以省内存。这里的调换是对已训练 EEGNet 的等价改写，不声称是新方法。
- 进行中：对全部 459 个原模型生成各版本并在第二天数据上打分。未做：正式计时（要等机器空闲）。

### 电极选择与 16 对 22（问题 1、2）

- 从 `results/final/rankings` 统计：relevance-16 去掉的 54 个电极位里，Fz 7/9、FC 一排 26/45、C 一排 10/63、CP 一排 4/45、P 一排 3/27、POz 4/9。
- 从 `results/test` 和原模型统计：同一人同一条件 3 个种子的标准差平均约 3 pp；relevance-16 减 full-22 逐人为 −3.4、+8.7、+2.1、−0.4、+4.5、+1.6、−5.1、+1.2、+1.4；训练日准确率 full-22 93.4%、relevance-16 92.5%、relevance-8 88.3%；验证集与测试集的逐人差值相关系数 −0.36。
- 用两天数据做的描述（读取了第二天，属于测试后描述，不参与任何选择）：被去掉与保留电极的 Fisher 分数，训练日 0.063 对 0.135，第二天 0.096 对 0.187；跨天功率变化两组都是 0.40 个标准差；P2 第二天的幅度是第一天的 1.52 倍。
- 新增 `scripts/run_electrode_study.py`（plan / train / evaluate）和 `scripts/analyze_electrode_study.py`。协议 `results/supplementary/electrode_count/protocol.json` 固定了 388 次训练：full-22、relevance-16/8 及子集不同的 proposed-16/8 的种子 3–9；relevance 排名下 20、18、14、12、10 个电极（种子 0–2）；最不相关的 16 个电极；2 个流程核对。设备分配沿用原来的被试 1–6 用 MPS、7–9 用 CPU。
- 新增 `scripts/redundancy_check.py`：只用训练日数据，检查去掉全头皮共有功率后再算冗余会换掉几个电极。结果：惩罚 0.5 时 16 个电极每人换 0–1 个（4/9 人有变化），8 个电极 2/9 人各换 1 个。据此不再为期中“追加 B”重训。

### 训练中断与处理

- 00:49 按原配置启动训练（MPS 组 3 个进程，CPU 组 6 个进程）。01:05 两组同时以 `BrokenProcessPool` 退出，当时已完成 41 次。
- 原因判断：这台 Mac 同时在跑别的任务。另一个应用（DeepSeek Harness）反复启动 `ai-video-studio` 的视频渲染（ffmpeg 加 4 个 Python 进程，有时两组同时跑），微信也占着接近一个核。`top` 显示内存用尽（swap 8 GB，压缩内存 6 GB，系统态 CPU 62%），负载超过 90。系统日志里没有查到直接的终止记录，所以“内存压力导致进程被终止”是推断，不是确认。
- 处理：停掉自己的任务，改为 MPS 组 2 个进程、CPU 组 3 个进程、`nice -n 10`，外面套自动重启的循环（已完成的训练会被跳过）；INT8 打分改成 1 个低优先级进程。没有动别的应用的进程。
- 影响：训练要比原计划多花几个小时；这段时间里测的时延不可用，正式计时推迟。

### 同日续：INT8 全量打分与正式计时

- 01:16 起改为低并行续跑训练（MPS 组 2 个进程，CPU 组先 3 个、视频渲染结束后改 5 个）。INT8 打分先停后以 1 个进程续跑，01:28 完成全部 459 个原模型。
- 打分结果（`results/supplementary/int8_study/summary.md`）：打分前先确认原 FP32、原 INT8 图仍然逐条复现 `results/test` 里的预测（459 个模型全部一致，否则脚本会报错）。等价改写的 INT8 在第二天全部试验上的 logits 与原 INT8 完全相同（459/459）。调换前两层的 FP32 与原 FP32 的最大 logits 差 4e-6，预测一致率最低 100%。17 个条件平均的准确率：FP32 63.24%，原 INT8 63.18%，混合精度 63.16%，调换顺序的 INT8 63.18%，INT16 63.22%。
- 第一次正式计时（01:28–01:32）：本想先用 SIGSTOP 暂停自己的训练进程，但 zsh 不会把变量按空格拆成多个 PID，`kill` 报 illegal pid，暂停没有生效，计时是在 7 个训练进程同时运行时完成的（负载约 11）。这份结果改名保存为 `benchmark_while_training.json`，不当作正式数字。
- 第二次正式计时（01:32–01:35）：PID 写入文件后暂停成功（15 个进程全部为 T 状态），计时结束后恢复，训练日志继续前进，没有触发重启。22 个电极：FP32 0.276 ms，原 INT8 0.363，三种换量化设置的版本 0.362–0.365，INT16 0.310（两种），混合精度 0.197，等价改写 0.212，调换顺序 FP32 0.095、INT8 0.067。负载从 9.0 降到 3.0；微信仍占约一个核，没有视频渲染任务。两次计时里各版本相对 FP32 的倍数相差在几个百分点以内。
- `scripts/build_report.py` 增加第 6 节“期中后追加的研究”（原第 6、7 节顺延为 7、8），由 `results/supplementary/` 下的分析文件生成；电极部分在补充训练完成前显示为 pending。已重建 `docs/REPORT.md`，`docs/REPORT.pdf` 还是 11-04 的版本（本机没有 ReportLab，没有重新导出）。
- 验证：13 项 unittest 通过；`audit_project.py --deep --require-artifacts` 通过（输出写到会话临时目录，没有改 `results/maintenance/audit.json`）。
- 未完成：补充训练还在进行（写这一条时 MPS 组 44/218，CPU 组 10/113）；`run_electrode_study.py --stage evaluate` 和 `analyze_electrode_study.py` 尚未运行，所以 10 个种子的结论还没有。

### 同日续二：补充训练完成、10 个种子的结论、第二批训练、CoreML、报告

- 第一批训练 02:57 全部完成（MPS 组 218 次用了 96 分钟，CPU 组 113 次用了 93 分钟，降并行之后没有再中断）。`run_electrode_study.py --stage evaluate` 写出 388 条测试记录。流程核对：重新训练的 2 个原模型（P1、P7 的 seed 0 relevance-8）在第二天的预测与存档 100% 一致。
- `analyze_electrode_study.py` 第一次运行通过。10 个种子的结果否定了我在训练跑完之前的初步判断（“差距主要来自训练随机性”）：relevance-16 减 full-22，种子 0–2 是 +1.17，新种子 3–9 是 +1.38，合计 +1.31 [−0.11, +2.80]；只算训练随机性的标准误是 0.57；把 10 个种子的输出概率平均后差距是 +1.91（67.88% 对 69.79%）。同一个种子的两个模型之间没有相关（被试内相关系数 −0.002），所以种子之间可以当作独立。
- 按人看：P2 +4.2、P5 +3.2、P6 +4.8、P8 +2.3，P1 −1.2、P7 −1.6，其余在 ±0.2 以内。22 个电极时低于 60% 的 4 人平均 +3.1 ± 1.0，其余 5 人 −0.1 ± 0.6；基线准确率与提升的 Spearman 相关 −0.67（p = 0.05）。分组是事后做的。
- 幅度“假如”检验换成 10 个种子：整段乘一个系数后差距从 +1.31 变成 +0.53 [−0.79, +1.96]；P2 两种模型变成 35.3% 和 35.7%（原来 42.7% 和 46.9%），P7 两种模型反而各升 5–6 pp。结论改为：P2 的提升只在输入超出训练幅度时出现；这个缩放不是可采用的改进。
- 03:02 追加第二批（`scripts/run_electrode_curve.py`，协议在 `results/supplementary/electrode_count_wave2/`）：20、18、14、12、10 个电极的种子 3–9，共 315 次。理由：这 5 个点只有 3 个种子，每个点约 ±1 pp 的种子噪声，分不清 16 是不是孤立的尖点。这个决定是看了第一批测试结果之后做的，协议里写明了。训练和打分直接调用第一批脚本里的函数，只把输出目录改掉。04:31 完成（MPS 组 210 次 89 分钟，CPU 组 105 次 81 分钟，没有中断）。
- 两批合并后的曲线（10 个种子，与 22 个电极的差）：20 个 +0.12，18 个 +0.73，16 个 +1.31，14 个 −0.44，12 个 −1.24，10 个 −2.10，8 个 −4.20。高的一组到 16 个为止不变（−0.4、−0.5、−0.1），之后每步都掉；低的一组在 18 和 16 个时涨 2.2 和 3.1，其余在 ±1 以内。参与者区间只在 10 个和 8 个电极时不含 0。
- 试过但没有写进报告的检查：在训练好的 22 电极模型里把 6 个电极的输入置零。任意 6 个置零都会让训练日准确率从 93% 掉到 35%–44%，说明这种做法只是把输入推到了模型没见过的地方，区分不出电极有没有用，不采用。
- 文献：Abdullah、Faye、Islam（2022，Bioengineering 9(12):726）的摘要把“通过避免过拟合提高准确率”列为选电极的目的之一，用 Europe PMC 的记录核对过原文；Alotaiby 等（2015，EURASIP JASP 2015:66）的书目信息用 Crossref 核对过，其摘要内容没能直接取到（出版社页面拒绝自动访问），报告里只按搜索结果转述为同一动机。
- CoreML：新增 `scripts/coreml_probe.py`。先在训练进行中粗测，训练结束后 04:33 正式测（负载约 3）：原 FP32 图 CPU 后端 0.271 ms，CoreML 只用 CPU 0.134 ms，允许 GPU/神经引擎 0.286 ms；原 INT8 图 0.351 / 0.402 / 0.403；调换层顺序的 FP32 0.093 / 0.098 / 0.099；调换层顺序的 INT8 0.067 / 0.151 / 0.152。CPU 后端的四个数与 01:32 的正式计时一致。训练进行中（GPU 被占用）时，允许 GPU 的那一档曾测到约 5 ms，这个数没有采用。
- `int8_study.py` 的哈希：`protocol.json`（`sources_sha256_at_build`）和 `benchmark.json` 里记录的 `scripts/int8_study.py` 哈希与现在的文件不同，因为之后改过出表和画图的部分；`variants.py`、`deploy.py` 的哈希仍然一致。核对：用现在的脚本对 9 个模型（每人 1 个，seed 1，9 种条件）重新执行 build，生成的记录与存档逐字段相同。三份训练协议（`matched_budget`、`electrode_count`、`electrode_count_wave2`）固定的源码哈希全部一致。
- 报告：`scripts/build_report.py` 的第 6 节现在有 5 小节（6.1 选电极，6.2 16 对 22，6.3 冗余度量，6.4 INT8，6.5 冻结方案与测试后可选部署形式并排），新增参考文献 7、8，标题后半句改为 “exported INT8 is slower on the M4 until two layers are swapped”，第 4 节加一句指向 6.4。`scripts/export_report_pdf.py` 对第 6 节的表格改为按内容分配列宽，并让方括号里的区间不折行；第 1–5 节的表格列宽没动。
- PDF：本机没有 ReportLab，用 `uv` 在会话临时目录建了独立环境（Python 3.12，reportlab 5.1.0、pillow 13.0.0，另装 pypdfium2 用来把页面转成图片检查），没有改项目环境和 `uv.lock`。`docs/REPORT.pdf` 已重新导出，18 页，逐页看过第 1、3、7、10–18 页。
- 文档：`docs/POST_MIDTERM_PLAN.md` 的 1.1、1.3、1.4 和第 2–5 节按最终结果重写；README 第 2 节加了期中后的结果，脚本数改为 29；`agent.md` 加了第二批、CoreML 脚本和三条路径说明。
- 验证：13 项 unittest 通过；`audit_project.py --deep --require-artifacts` 通过（输出在会话临时目录）。
- 未做：没有提交或推送 Git。新训练的 703 个模型（约 52 MB）和 INT8 各版本只在本机 `data.nosync/models/supplementary/` 下，被 `.gitignore` 排除，没有别的备份。期末幻灯片和交付包还没开始。

## 2026-11-20 至 11-21 重构、代码护栏和 Longleaf 上的新研究

- 重构与护栏：按 Lauren Tan 的“倒金字塔”思路整理了代码和检查（`docs/GUARDRAILS.md`）。lint 存量从 622 降到 4；`tools/golden.py --all` 对全部 9 个生成器输出逐字节相同。新增 `tools/check.py`、`guard.py`、`golden.py`、编辑钩子、pre-commit 和 GitHub CI。
- Longleaf：11-21 起用专用钥匙登录（`ssh longleaf`，需要 VPN，不需要密码和 Duo）；从未使用 Sycamore，也没有替用户输入密码。
- 新研究（全部是看过 2a 测试结果之后追加的探索性研究，第二个数据集除外），结果和读法见 `docs/POST_MIDTERM_PLAN.md` 1.5 节和报告第 6.6–6.11 节：位宽（459 个模型的训练后量化和 756 次量化感知微调）、按训练日定混合位数、INT8 跨 CPU 的准确率和速度、跨天对齐、脑区（18,396 次训练）、选电极方法（1,350 次）和同一硬件对照（135 次）、Lee 2019 的事先登记复现（1,944 + 486 次）。
- 主要结论：8 位就够，INT16 没用；没有 VNNI 的 x86 CPU 上 INT8 掉 4 pp，7 位权重解决；脑区的 Shapley 值重现了运动皮层的左右交叉分布；黎曼距离法选电极在 2a 上探索、在 54 个新的人身上事先登记后证实（8 个电极 +3.66 pp [+2.10, +5.27]）；少电极不掉准确率推广到了新的人；“解得差的人受益更多”和跨天对齐都没有复现。
- 过程中的问题：
  - 脑区研究的最终模型第一次提交时用了 60 个分片，而计划每 4 个任务一个周期，所有最终模型都挤进了 15 个分片，其中 13 个超时。改用奇数分片数重交，`tools/longleaf.py` 的说明里写了这条规则。
  - 夜里监视脚本在 04:00 左右退出后没有重启，Longleaf 空闲了约 5 小时，结果比预计晚。
  - Lee 2019 的 P50（F9）和 P51（FTT10h）各有一个电极在部分第一天试验里恒为 0，导致相关性分数全变成 NaN；第一次计划因此失败。改为把这样的电极排在最后，决定只依据第一天的数据。
  - 发现训练硬件会改变单个模型（M4 的 GPU 和 Longleaf 的 CPU 训练出的同种子模型，预测只有 50%–94% 一致）；135 个参照模型在同一硬件重训，准确率没有系统性偏差。
  - 10 个复现任务落在负载很高的旧 AMD 节点上，慢 5–8 倍；取消后用 `--constraint 'amd&avx512'` 在新节点上重跑同样的分片。
  - 黎曼法的检验计划是在第一个复现计划之后加的。为了保证它也是事先登记，把复现研究的打分任务暂停，等黎曼法的计划写完（10:32）才放行；放行前测试记录数为 0。
- 存储迁移：11-21 的存储迁移把项目复制到 `/Volumes/MacDev/Developer/courses/comp690/project`（见 `~/Developer/WORKFLOW.md`）。核对两份副本在迁移时逐文件相同（不含环境、数据和缓存）；11:30 之后的修改只在 MacDev 副本上，iCloud 副本保留不动。
- 报告：`scripts/build_report.py` 新增第 6.6–6.11 节；`docs/REPORT.pdf` 重新导出为 27 页（用 `~/.local/bin/macdev uv run --no-project --with reportlab --with pillow` 的临时环境，项目环境和 `uv.lock` 没变），逐页看过新增部分的排版。
- 验证：`python3 tools/check.py` 全部通过（格式、lint 棘轮、规则检查、单元测试）；`python3 tools/golden.py --all` 全部 SAME。
- 未做：没有提交或推送 Git。新研究的模型（约 438 MB）只在 Longleaf 上。期末幻灯片和交付包还没开始。

## 2026-11-21 晚 期末幻灯片初稿、课程要求和单片机模拟环境

- 期末幻灯片初稿（在单独的会话里完成）：`presentation/final.pptx` 22 页加 5 页备用，生成脚本 `presentation/pptx/build_final_pptx.js`，数字都从 `results/` 读取；逐页讲稿 `presentation/final_script.md` 完整约 14 分 51 秒（按每分钟 125 词），去掉方括号约 13 分 27 秒，Word 版 16 页。每页分别用 Open Sans 和更宽的 Verdana 渲染检查过，又整体看了一遍。没有在真正的 PowerPoint 或 Keynote 里测试，包括第 19 页的视频能否播放；放映电脑要装 Open Sans。
- 课程页面（https://www.cs.unc.edu/~raghav/Courses/comp690fa26/fa26.php，11-21 读取）：
  - 汇报在 12-08 下午 4–7 点，每组 25 分钟，其中至少 5 分钟提问。评分：内容 30%、表达 30%、组织 20%、回答问题 20%。
  - 报告同一天交，是 PDF，正文 10–12 页、12 号字、1 英寸页边距。参考文献、Artifacts、附录和伦理一节（最多 1 页）不计页数。前面要写贡献、和其他课程或研究的重叠，摘要少于 200 词。正文章节依次是 Introduction、Background、Related work、Approach/System Design、Experimental setup、Results、Conclusion。
  - 两者都不能迟交；报告会公开给以后的学生看。现有报告 27 页，结构也不同，需要改写。
- 单片机模拟环境：
  - 用 Homebrew 装了 QEMU 11.1.3。从 Arm 官方下载了 Arm GNU Toolchain 15.3.Rel2，校验和与官方一致，放在 `/Volumes/MacDev/Developer/environments/`。
  - 写了一个统计执行指令条数的 QEMU 插件，在模拟的 Cortex-M0（micro:bit）、M4（MPS2 AN386）、M55（MPS3 AN547）上跑通了 1,024 个数的点积。结果（指令条数）：
    - M0：8 位 7,184，浮点 189,368（软件模拟浮点）。
    - M4：8 位 5,136，浮点 6,154，用 DSP 指令的 8 位 3,088。
    - M55：8 位 4,112，浮点 5,135，用 Helium 向量指令的 8 位 272。
  - 每种写法的结果都相同。这些是指令条数，不是时间；模拟器不模拟时钟周期。
  - 测试代码只在临时目录，没有进项目。
- 未做：两项深入研究等你决定；没有提交或推送 Git。

## 2026-11-21 夜到 11-22 两项深入研究、补上的 INT8 内核

- 用户的决定和追问（11-21 晚）：两项深入研究都做（单片机上的 INT8、直接检验过拟合）；数据集要不要整体换成 Lee 2019；为什么峰值正好在 16；INT8 空间卷积为什么慢，“缺内核就补上”；幻灯片不要“局限与未来工作”，能解决的都解决，要写得更详细，把期中之后的过程、用到的工具和 Longleaf 的用法都写进去。
- 数据集：没有整体重构。2a 是领域里最常用的标准数据集，问题在只有 9 个人；Lee 2019 的 54 人已经补上了这一点，而且“16 比 22 好”在 Lee 2019 上不存在，换数据集等于绕开问题。做法是 2a 为主、Lee 2019 同等地位，每个主要结果两个数据集都给。
- 过拟合的直接检验（`scripts/overfit_study.py`，Longleaf，3,240 个模型）：计划 11-21 23:04 写定，Lee 2019 上的 4 条预测（P1–P4）写在计划里，那时新条件的模型一个都没有在第二天上打分。条件：重新训练的全部电极和相关性电极，被去掉的电极换成带限噪声或换成别的试验的信号，通道 dropout（p = 0.2），只用训练日的一部分整组数据训练、在当天留出的组和第二天上同时打分。2a 用 10 个种子，Lee 2019 用 3 个。
- 电极数 × 训练量（`scripts/electrode_data_study.py`，Longleaf，4,824 个模型）：计划 11-21 23:32 写定，Lee 2019 上的 D1、D2 写在计划里。2a 在 2、4、6 组训练数据下画 4 到 22 个电极的曲线，Lee 2019 在 1、2 组下画 8 到 62 个。
- Longleaf 上的做法：`tools/longleaf.py push` 同步代码；计划阶段在登录节点上运行（只读第一天、算排名，约 2 分钟）；训练用 199 个任务的 SLURM 数组，每个任务 4 个 CPU、按 `--shard i/199` 取计划里每第 199 个任务，`--constraint 'amd&avx512'` 只用 AMD EPYC 9654 节点；打分阶段用 `--dependency=afterany` 排在训练数组之后，9 个任务。
- 单片机（`scripts/mcu_study.py`，本机）：从 Arm 的 GitHub 克隆 CMSIS-NN v8.0.0（commit 13c97dbb，53 MB，放在 `/Volumes/MacDev/Developer/environments/`）。新写 `src/mibci/mcu.py`（TFLite 布局导出、TFLite 规范的 INT8 量化、与 CMSIS-NN 逐位一致的 NumPy 整数参考）、`src/mibci/mcu_build.py` 和 `mcu/` 下的 C 程序（CMSIS-NN 的 INT8 和实验性 FP32 网络、测量框架、启动代码、链接脚本模板、QEMU 逐层计数插件）。计划 23:28 写定；216 个模型的参考打分和 168 次模拟（约 10 分钟）。136 个 INT8 版本、1,088 次预测与参考逐位相同；FP32 最多差 3e-6；INT8 与 FP32 的准确率差在 ±0.32 pp 以内。数字见 `docs/POST_MIDTERM_PLAN.md` 1.6.2。
- 补上的 INT8 内核（`scripts/ort_kernel_study.py`、`scripts/ort_kernel_swapped_study.py`，本机）：GitHub 上的 ONNX Runtime 仓库和 NuGet 包当晚都访问不到，C API 头文件改从 Mozilla Firefox 源码里附带的 Microsoft 原版取（API 版本 30，三个文件，SHA-256 记在计划里）。`kernels/ort_grouped_conv.c`（原顺序的分组空间卷积）和 `kernels/ort_spatial_first_conv.c`（调换后的空间卷积）是 ONNX Runtime 自定义算子，换进 ONNX Runtime 自己优化后的图。两个内核在 459 个模型上都逐位相同（各 132,192 个预测）。计时（11-22 00:00，负载 4.3 → 3.4）：22 个电极 FP32 273 微秒、原 INT8 357、加第一个内核 180；调换前两层后 FP32 94、INT8 66、加第二个内核 47。数字见 1.6.1。
- 新增单元测试 `tests/test_mcu.py`（8 项：导出、成本、INT8 跟随浮点、取整、乘数、过拟合研究的三个输入函数）。`python3 tools/check.py` 第一次运行时 tests 一项报失败但输出是 47 项 OK，之后连续两次全部通过，没能复现。
- 幻灯片：改稿要求和全部新数字发给了做幻灯片的会话（11-21 23:51），由它改生成脚本和讲稿；过拟合和曲线两项的结果出来后再补发。
- 未做：没有提交或推送 Git。
- 11-22 00:20 两项 Longleaf 研究全部打完分并拉回（`overfit_study` 3,240 条，`electrode_data_study` 4,824 条），分析见 `docs/POST_MIDTERM_PLAN.md` 1.6.3、1.6.4：被去掉的电极带来的信息（2a +4.64 pp，Lee 2019 +4.50）和多出这些输入的代价（−4.96、−4.77，Lee 2019 上事先登记的 P3 成立）正好抵消，所以 16 个和 22 个一样好；同一硬件重训 10 个种子后差距是 +0.32 pp；曲线先升后平，平台从 2a 的约 16 个、Lee 2019 的 16–20 个开始，训练数据减半时不移动（D1、D2 不成立）；通道 dropout 大幅降低准确率（2a −11 pp）。训练日到第二天的落差大部分发生在同一天之内。
- 用户的新要求（11-22）：幻灯片只讲“用了什么工具、得到什么结果、为什么”，Longleaf 只讲它带来的规模，不讲登录和 SLURM 的细节；不要备用页；每个说不清的地方都查到底，不用数据集的局限来解释。两条要求都转给了做幻灯片的会话（00:12、00:24）。
- 计算量统计：新增 `scripts/analyze_compute.py`，写 `results/supplementary/compute_summary.json`：Longleaf 上 30,456 次训练、2,874.5 CPU 小时，另有 756 次量化感知微调。
- 两周计划写进 `docs/POST_MIDTERM_PLAN.md` 第 3 节。第一项已启动：`scripts/input_cost_study.py`（多一个输入的代价，剂量-反应），计划 00:36 写定，4,590 个模型，Lee 2019 上的 C1、C2 写在计划里；训练任务 10178105，打分任务 10178111。
- 11-22 02:10 `input_cost_study` 的 4,590 个模型全部打完分并拉回；分析脚本 `scripts/analyze_input_cost_study.py`，结果见 `docs/POST_MIDTERM_PLAN.md` 1.6.5：每多一个噪声输入，2a 掉约 0.9 pp、Lee 2019 约 0.3 pp，当天留出组和第二天一样；训练数据多 3 倍也不变小。Lee 2019 上 C1 成立，C2 不成立（方向相反）。两个停不下来的旧轮询（等 `overfit_study`、`electrode_data_study` 的计数，结果早已拉回）已停掉。
- 发现的反常：同样“16 个电极 + 6 个噪声”，1.6.3 的那次噪声掉 4.96 pp、这次掉 7.29 pp，差 2.3 pp（9 人里 8 人同向）。查过的：两项研究的训练环境相同；EEGNet 对输入顺序完全对称（打乱输入并同步打乱权重，输出差 2e-9）；噪声输入分到的初始权重大小和代价无关。看到的：同一条件下种子分成“几乎不受影响”和“掉很多”两类，掉很多的种子训练集准确率也更低。
- 为此新写 `src/mibci/noise_mechanism.py`（另抽噪声、去掉噪声输入的空间权重、在训练数据上重估 BatchNorm、空间滤波器输出中噪声的方差占比）、`tests/test_noise_mechanism.py`（5 项）和 `scripts/noise_mechanism_study.py`。计划 02:33 写定，4,644 个任务：3,834 个只用第一天留出组的机制测量（不训练），540 个新训练（4 次新抽样、两项研究的噪声互换位置），270 个参照模型重新打分。在 Longleaf 上先单跑一个任务，重算的留出组准确率与原记录相同；之后提交：机制 10193444，训练 10193445，打分 10193448（排在训练之后）。
- 02:40 机制部分（3,834 个模型，不训练，只用第一天）拉回并分析（`scripts/analyze_noise_mechanism_study.py`，计划文档 1.6.6）：重算的留出组准确率与原记录 3,834 个全部相同。训练后噪声在空间滤波器输出里的占比和初始化时差不多（2a 加 6 个：38% 对 43%；加 16 个：62% 对 60%；Lee 2019 训练后更高）；训练后把噪声权重设为 0 并重估 BatchNorm，只救回 9%–30% 的损失；同一人同一条件内，噪声占比和准确率的相关是 −0.29（两个数据集都是，区间不含 0）。
- 后台轮询的毛病：zsh 不会把 `$S` 按空格拆开，`$S longleaf ...` 根本没有执行 ssh，等待条件永远不满足。11-21 夜那两个停不下来的旧轮询也是这个原因。改成直接写 `ssh -o ... longleaf`。
- 02:50 新研究 `scripts/input_penalty_study.py`（模块 `src/mibci/input_penalty.py`，测试 `tests/test_input_penalty.py` 3 项，其中一项检查 λ = 0 时与原训练逐位相同；分析 `scripts/analyze_input_penalty_study.py`）：给每个输入的空间权重加 group lasso 惩罚，检验“补上把没用的输入推向 0 的力，代价就消失、全部电极就能胜过最好的 16 个”。计划 02:47 写定，Lee 2019 上 G1–G3 写在计划里；4,320 个 fold 模型选 λ（只用第一天），最终模型只训练选中的 λ。Longleaf 上串成四步：fold 训练 10195830 → 选 λ 10195831 → 最终模型 10195832 → 第二天打分 10195834。
- 03:01 噪声抽样部分完成并拉回（540 个训练、810 个打分；270 个参照模型重新打分与原记录全部相同）。8 个版本的“16 + 6 个噪声”代价在 −5.0 到 −7.5 pp 之间，7 次独立抽样平均 −6.3；同一次噪声换位置几乎不变。2.3 pp 的差主要是抽样的运气，1.6.3 用的那次是最轻的。训练结束时的损失和第二天准确率在同一人同一条件内相关 −0.62。计划文档 1.6.5、1.6.6、1.6.3 的表已更新。`scripts/analyze_compute.py` 重跑：Longleaf 累计 35,586 次训练、3,451.2 CPU 小时（不含正在跑的惩罚研究）。
- 03:04 第三次修改要求发给做幻灯片的会话：第 12 页删掉“代价来自死记训练样本”，换成信息约 6 / 代价约 6、每个输入约 1 pp、训练学不会关掉没用的输入三条；第 13 页最后一句、第 25 页加 C2、第 8–9 页的累计数字。惩罚研究的结果出来后再发一次。
- 04:18 惩罚研究整条链跑完并拉回（4,320 个 fold 模型、432 个最终模型和打分；第一天选 λ：2a 全部 22 个选 0，相关性 16 个 0.01，16 + 6 个噪声 0.003，16 + 16 个噪声 0.01；Lee 2019 全部 62 个 0.003，相关性 16 个和 16 + 46 个噪声都选 0）。Lee 2019 上 G1、G2 不成立，G3 无从检验。惩罚没有关掉噪声输入，而是把所有权重一起缩小（噪声和电极权重之比始终 0.5–0.75）。本机验证原因：空间滤波器后面的 BatchNorm 让整体缩小几乎不改变输出（全部缩到 0.3 倍，输出最多变 0.002；只缩 6 个输入，变 0.11）。分析脚本里 G3 的比值在没有惩罚模型时是 NaN，改成写 null。计划文档 1.6.7。`scripts/analyze_compute.py` 重跑：Longleaf 累计 40,338 次训练、3,961.5 CPU 小时。
- 04:25 第四次修改要求发给做幻灯片的会话：不加新页；第 25 页加惩罚研究不成立及原因，第 26 页视版面加强“EEGNet 自己不会忽略输入，所以要先选电极”，第 12 页讲稿补一句，第 8–9 页累计数字。


## 2026-11-22 下午起 期末冲刺（按 `docs/FINAL_EXECUTION_PLAN.md` 执行）

- 15:13 开工核对：`python3 tools/check.py` 四项 ok；`python3 tools/longleaf.py check` 能连上，home 用 16.1 GB（配额 50 GB），队列为空。
- WP0 发现 `tools/longleaf.py pull` 会用 Longleaf 上较旧的文件覆盖本机较新的：dry-run 显示 `noise_mechanism_study`、`selection_study` 的 `summary.md`、`analysis.json` 和几张图在 Longleaf 上是旧版本（例如 summary.md 02:41 对本机 03:01），`replication_study` 有 4 个环境记录只差时间戳。原因：分析输出在本机生成，以前用 `push`（`--ignore-existing`）推上去之后本机又重新生成过。改法：`pull` 的结果和模型两次 rsync 都加 `--ignore-existing`（记录和模型只写一次，本机已有的就是同一个文件），新增 `tests/test_longleaf.py` 复现这次错误。改完后按计划拉回 11 项研究的模型备份（15:16–15:19），本机文件数与 Longleaf 逐项相同：alignment 81、electrode_data 4,824、input_cost 4,590、input_penalty 4,752、noise_mechanism 540、overfit 3,240、region 18,396、replication_riemann 486、replication 1,944、selection_control 135、selection 1,350，共 862 MB，放在不同步的 `data.nosync/models/supplementary/`。本机较新的分析文件保持不变。
- WP0 改掉 `docs/POST_MIDTERM_PLAN.md` 第 4 节第 6 条的过时说法（“以后请从 MacDev 上的副本开始”）。核对到的实际情况：MacDev 上的 `Developer/courses/comp690/project` 是一个目录，里面每一项都是指回 iCloud 这份的符号链接，只有它自己的 `.venv.nosync` 不是。
- WP1 `input_kind_study`：新模块 `src/mibci/input_kind.py`（拷贝、拷贝和噪声的混合、同一条噪声重复、被去掉的电极线性预测不了的方差比例），测试 `tests/test_input_kind.py` 5 项（其中一项检查 a = 1 时与 `input_cost_study` 的输入逐位相同）。本机在 2a 的 P1 上用临时目录、2 个 epoch 冒烟测试全流程；Lee 2019 的路径在 Longleaf 的临时目录里同样测过（srun，interact 分区），之后删掉临时目录。计划 **15:30:55** 在 Longleaf 上写定（2,583 个任务：2a 1,125，Lee 2019 1,458；协议 SHA-256 6849351398…），那时这项研究的模型一个都还没有训练，Lee 2019 上的 K1–K3 写在计划里。训练数组 10286192（199 个任务、每个 4 个 CPU、`amd&avx512`），第二天打分 10286193 排在训练之后，只用第一天的可预测性分析 10286194。
- 冒烟测试顺带看到的：2a 的 P1 被去掉的 6 个电极，留出组上只有 0.6%–10% 的方差是留下的 16 个电极线性预测不了的；Lee 2019 的 P1、P2 被去掉的 46 个电极，这个比例的中位数是 12%–25%。也就是说真实的被去掉的电极更像“拷贝”，不像独立的噪声。等全部人的结果出来再下结论。
- WP2 `long_training_study`：直接调用 `mibci.train.fit`，训练 1,900 个 epoch，在 475、950、1,900 存权重。新测试 `tests/test_long_training.py` 2 项：训练更久时第 3 个 epoch 的保存点与只训练 3 个 epoch 的权重逐位相同；每个 epoch 记录留出组不改变训练。本机在 2a 的 P1 上用临时目录、2 个 epoch 冒烟测试；和旧模型逐位比较的函数在真实旧模型上判断为相同。计划 **15:38:43** 在 Longleaf 上写定（1,485 个任务：2a 675，Lee 2019 810；协议 SHA-256 9ad2376bc6…），Lee 2019 上 L1–L3 写在计划里。L3 原文是“大于 0”，按 L1、L2 的口径实现为“参与者区间下限大于 0”，写在计划的 test 字段里。训练数组 10287886（199 × 4 CPU，`amd&avx512`，时限 12 小时），打分 10287890 排在训练之后。
- Longleaf 的账户有总 CPU 上限（`AssocGrpCpuLimit`，约 1,000 个 CPU）：WP1 的 199 个任务加 WP2 的 51 个任务占满之后，其余任务都在排队，`scontrol top` 不允许普通用户调整顺序。为了让 WP3 的小 pilot 先跑，暂时 `scontrol hold` 了 WP2 还在排队的任务，pilot 15:55 开始后立刻 `release`（已经在跑的任务不受影响）。以后的研究都会排在这个上限后面，所以提交顺序就是执行顺序。
- WP3 `scalefree_penalty_study`：新模块 `src/mibci/scalefree_penalty.py`（每个空间滤波器先归一化成单位长度，再对每个输入求它在全部滤波器上的范数 g_i，惩罚 λ Σ g_i；Σ g_i² 恒等于滤波器个数），测试 `tests/test_scalefree_penalty.py` 3 项（λ = 0 时与 `mibci.train.fit` 逐位相同；整体或逐个滤波器缩放都不改变惩罚；只缩小一个普通输入，惩罚变小）。λ 的五档按计划只用第一天试出来：pilot（2a 的 P1、P2，fold-0，种子 0，16 + 6 和 16 + 16 个噪声，λ = 0.001–3）在 Longleaf 上跑（计划原文写“在本机试”，按 AGENTS.md 第 5 节扫参放 Longleaf；只用第一天，不影响结论）。结果：0.001 时噪声和电极的归一化范数之比只降 6%–17%、留出组准确率不变；0.1 时 P1 加 16 个噪声的比值降到 0.064，留出组准确率从 47.8% 升到 63.3%；0.3 及以上准确率掉到接近瞎猜、比值回到 1 左右；比值不总是单调（P2，以及 P1 加 6 个噪声在 0.03 时升到 1.9）。五档定为 0.001、0.003、0.01、0.03、0.1，pilot 的全部行写进计划的 `pilot` 字段。pilot 跑的是加 `reference` 阶段和 LAMS 之前的同一个脚本，训练代码相同，这一点也写进了计划。计划 **16:21:53** 写定（协议 SHA-256 2adb1a6f25…），Lee 2019 上 N1–N3 写在计划里。Longleaf 上串成四步：fold 训练 10293425 → 选 λ 10293427 → 最终模型 10293431 → 第二天打分 10293433；λ = 0 的旧模型按本研究的口径重算范数（`reference`）10293439。
- WP8 `pretrain_study`：新模块 `src/mibci/pretrain.py`（按预训练参与者的平均相关性排名选共用的 16 个电极；每人用自己第一天的均值和标准差标准化后合并；项目原来的 epoch 预算规则；Lee 2019 按编号分 6 组），测试 `tests/test_pretrain.py` 7 项，其中一项用原始验证曲线重算出冻结的 475 个 epoch（平滑最大值逐位相同），说明预训练轮数用的是同一条规则。预训练的上限和保存点间隔在计划前写定：2a 300 个 epoch、每 5 个存一次、平滑窗口 5；Lee 2019 75 个 epoch、每 3 个存一次、窗口 3（按梯度步数，分别约为单人训练的 2.9 倍和 5.6 倍）。本机在 2a 上小规模跑通全流程；Lee 2019 的预训练在 Longleaf 的临时目录里测过（srun：45 人合并后内存峰值 6.3 GB，4 线程时 62 个输入每个 epoch 约 0.6 分钟），之后删掉临时目录。计划 **16:28:47** 写定（1,845 个任务；协议 SHA-256 adfe13ce99…），Lee 2019 上 T1–T3 写在计划里。提交：预训练 10296583（191 × 4 CPU，每次运行 4 线程）、从头训练 10296593、选预训练轮数 10296611（只用第一天）→ 微调 10296620 → 第二天打分 10296626。
- WP4 `chdrop_study`：新模块 `src/mibci/chdrop.py`（每个 BatchNorm 层存下的方差和测试方式下实际方差之比、把电极换成训练均值、固定的三电极组），测试 `tests/test_chdrop.py` 4 项。写测试时发现 EEGNet 的 BatchNorm 动量是 0.01：只训练几十步的玩具模型统计量还没收敛，不加 dropout 的比值也很大，所以删掉了一个想在玩具数据上证明机制的测试，机制只在正式研究里判断。本机在 2a 的 P1 上冒烟测试：存档模型按原样重新打分与存档预测 10/10 相同，fold 参照 5/5 相同。计划 **16:30:18** 写定（3,348 个任务；协议 SHA-256 838400bd7c…），Lee 2019 上 B1–B3 写在计划里；那时新模型、存档模型的重估版本和坏电极版本都还没有在第二天上打分（存档模型原样的分数早已知道）。B2 原文没写用原样还是重估后的版本，计划里定为原样，重估后的另外报告。提交：训练 10296899、参照（不带 dropout 的 fold 模型，只用第一天）10296900、第二天打分 10296902 排在训练之后。
- WP5 `alignment_mechanism_study`：新模块 `src/mibci/alignment_variants.py`（普通对齐逐位等于 `mibci.alignment`；向单位阵收缩 10%；Ledoit–Wolf 收缩，把每次试验的协方差当作一个观测，避免把同一次试验里相关的时间点当成独立样本；只白化不小于最大特征值 1% 的方向），测试 `tests/test_alignment_variants.py`。**设计问题：** 本机在 2a 上冒烟测试时，第一天对齐用的协方差里 22 个方向有 16–18 个低于最大特征值的 1%（条件数约 1 万，Ledoit–Wolf 系数约 0.0003）；在 Longleaf 上看了 Lee 2019 的 5 个人（P1、P2、P3、P10、P30），62 个方向里有 45–52 个低于 1%，最大的一个方向占 44%–61% 的方差。按计划的 `ea-top`（小方向置零）在 2a 上只剩 4–6 个方向、Lee 2019 上只剩 10–17 个，它的结果会把“不放大小方向”和“丢掉小方向里的信息”混在一起。处理：`ea-top` 照计划做；另加 `ea-floor`（同样以 1% 为界，小方向不置零，只把特征值抬到 1% 再白化，即保留但不放大），两者合起来才能分开两种解释。M2 仍按原文判定（`ea-top` 和 `ea-shrink10`），`ea-floor` 并列报告。这段原因写进了计划的 `design_note` 字段和进度表。计划 **16:36:10** 写定（2,907 个任务；协议 SHA-256 e46c2ed41b…）。提交：训练 10298348、协方差谱 10298349（只用第一天）、第二天打分 10298351 排在训练之后。
- 11-22 下午重读课程页面（<https://www.cs.unc.edu/~raghav/Courses/comp690fa26/fa26.php>，没有单独的报告模板或项目说明页）：要求与 11-21 记下的一致。补充三点：Introduction 要写“key ideas of your approach including the systems principles you have utilized”；Conclusion 的标题写作 “Conclusion (optionally and future work)”；伦理一节要“clear identification of potential issues, risks, mitigations, deployment measures”，不要空泛的话，可加参考文献。页面上关于 AI 工具只有一段（“AI tools” 一节），允许用来理解概念、找相关工作，但课程材料和其他同学的作品不能交给这些工具；没有说写报告或写代码能不能用。这段留到 WP12 给用户看。
- WP6 `mcu_kernel_study`（本机）：新文件 `mcu/kernels/fir_s8.c`、`fir_s8.h`（INT8 FIR 滤波器组：每个输入行先加 offset、用零点补边，int32 精确求和，加偏置后用 CMSIS-NN 自己的 `arm_nn_requantize`，再加 out_offset、截断；Helium 用 VMLADAVA 做 16 抽头的点积并把 4 个抽头向量留在寄存器里，DSP 用 SMLAD、两个输出共用一次读权重，纯 C 两个输出共用一次读抽头）和 `mcu/kernels/eegnet_int8_fir.c`（被锁定的 `mcu/eegnet_int8.c` 的副本，只换掉时间卷积一层），`src/mibci/mcu_kernels.py`（取抽头、写 `fir.h`、链接）及 `tests/test_mcu_kernels.py` 4 项（NumPy 版的内核算法在两种层顺序下都与 `mibci.mcu` 的整数参考逐位相同）。开发时先在 P1 的第一天窗口上编译运行，6 个构建第一次就逐位相同；之后做了两处优化（Helium 把权重留在寄存器、DSP 和纯 C 两个输出共用读取）。计划 16:47 在本机写定（136 个构建，和 `mcu_study` 的 INT8 构建一一对应，用同样的前 8 个第二天窗口；工具版本与 `mcu_study` 的计划逐项相同），16:48–17:00 跑完。结果：**136/136 逐位相同**；两种构建共用的层每个窗口最多差 12 条指令；整个网络的指令条数少 1.5–2.8 倍（调换层顺序：M4 6.92 → 2.70 百万条，M55 1.37 → 0.78，M0 18.45 → 7.62；冻结方案 M4 6.66 → 2.44）；时间卷积每次乘加的指令数 M4 10.71 → 2.47（硬件上限 0.5），M55 2.01 → 0.87（上限 1/16），M0 28.29 → 7.18（上限 2）；M55 调换顺序时 RAM 51.5 → 29.4 KB（不再需要 CMSIS-NN Helium depthwise 的 31.7 KB 草稿区），flash 多 1.5–3 KB。计划里“有时间再补”的原顺序空间卷积内核没有做：原顺序在每个内核上都比调换后慢，不是推荐的部署形式。这些都是 QEMU 的指令条数，不是时间或能耗。
- 本机写课程报告的生成器 `scripts/build_course_report.py`（英文，数字全部从 `results/` 读，6 张按印刷尺寸画的图放 `docs/course_report/`，还在跑的研究显示为 pending）和导出脚本 `scripts/export_course_report_pdf.py`（Times New Roman 12 号正文、1 英寸页边距、表格和图注 10 号、参考文献另起一页，打印每章所在页）。PDF 用 `uv run --offline --no-project --with reportlab --with pillow` 的临时环境导出，不动 `uv.lock`。第一版：摘要 187 词，正文（标题到结论）10 页，参考文献从第 11 页起；逐页转成图片看过第 1、7 页的排版。新研究的结果出来后再补，预计正文到 11–12 页。
- WP7 A `int8_paths_study`（快慢地图）：新模块 `src/mibci/int8_paths.py`（只有一层卷积的合成模型，G 组、每组 M 个滤波器、覆盖 22 个电极的空间核或 1×64 的时间核，按解码器同样的方式导出和量化），测试 `tests/test_int8_paths.py` 3 项（网格包含 EEGNet 的四种层；乘加次数；INT8 图执行的是 QLinearConv 且跟随 FP32）。计划 17:08 在本机写定（38 种形状；不用任何脑电数据）。本机 M4 的计时在后台跑（开始时负载 3.0）；Longleaf 四种 CPU 各一个任务：`amd&avx512` 10308656、`intel&avx512` 10308658、不带 AVX-512 的 AMD 10308669 和 Intel 10308670（`amd`、`intel` 这两个特征也会匹配带 AVX-512 的节点，所以用 `--exclude` 排除了那些节点，先提交的 10308659、10308660 已取消）。
- `tools/golden.py` 新增期中后每项研究的分析脚本（`study-*`，比较各研究目录顶层生成的 .md/.csv/.png 和 analysis.json）、`hw-latency`、`int8-portability`、`compute` 和 `course-report`（排在最后，读刚重建的表）。新增 `scripts/rebuild_all_tables.py`（课程报告 Artifacts 一节的一键重建：调用 golden 里只需记录的生成器，可用 `python3` 运行，已加进 pyproject 的 py39 目标）。17:11 第一次对照：17 项研究分析、两个跨机器汇总和课程报告全部 SAME；`compute` 不同，原因是新研究已有计划和记录而汇总还没重跑（实验做完后重跑）；最后的“仓库在运行中被改动”提示来自同时在跑的本机 WP7 计时写了 `int8_paths_study/map/m4-mac-mini.json`，与生成器无关。
- WP7 A 本机 M4（17:09 开始，负载 2.3 → 2.9）：38 种形状里 32 种 INT8 比 FP32 慢。INT8 更快的只有不分组的 1×64 时间卷积（每组 ≥ 2 个滤波器，EEGNet 的时间层 0.72 倍）和通道数 ≥ 8 的 depthwise 时间卷积（调换后的时间层 0.28 倍）；覆盖全部电极的 22×1 空间核在任何分组下都慢（EEGNet 的空间层 15.6 倍，调换后的第一层 4.6 倍），分组数 2–8、每组 ≥ 2 个滤波器的分组卷积也都慢 3–6 倍。每个 INT8 图都只执行一个 QLinearConv。等 Longleaf 四种 CPU 的结果再写进计划文档。
- 更正一处口径：1.6.6/1.6.7 写的“7 次独立抽样平均 −6.3、信息 +6.0”不准确。`noise_mechanism_study` 的 8 个版本里只有 6 次是不同的噪声抽样，另 2 个是同一次噪声换了位置；幻灯片一直用 8 次测量的平均（−6.48、+6.15，显示为整数 ±6），报告生成器开始时用了 7 个数。统一成 8 次测量的平均并照实描述（报告、POST_MIDTERM_PLAN 1.6.3 的表、1.6.5、1.6.7）；只算 6 次不同抽样接在后面的版本是 −6.55。
- WP7 B `int8_layers_study`（不准出在哪一层）：为了不改已锁定的 `int8_paths_study.py`（它的四个节点任务还在排队），B 部分写成单独的研究。新模块 `src/mibci/int8_layers.py`（把 QDQ 图里全部 8 个 int8 张量加成图输出一次取出；ONNX Runtime 仍把每层融合成整数内核，测试里核对过 logits 与正常推理逐位相同；按 x86 没有 VNNI 时的做法（激活加 128 成 uint8、相邻两个乘积在 16 位里相加）判断哪些输出会饱和；把某一层的权重改成 7 位网格），测试 `tests/test_int8_layers.py` 3 项。只用第一天：18 个存档模型（P1–P9 种子 0，full-22 和冻结方案）、每人前 32 个无伪迹窗口，存档图加 6 个 7 位变体（五个带权重的层各改一层，或全改）。本机 P1 的检查：每个模型版本的捕获约 2.2 MB；在 M4 的输入上按 x86 的配对方式算，时间卷积约 2.1%、空间卷积约 3.6% 的输出会饱和。计划 17:24 在本机写定（协议 SHA-256 cd9e8789e4…）。Longleaf：VNNI 节点（AMD 9654）捕获 10313410 → 没有 VNNI 的 Intel Gold 6140 捕获 10313411 → 比较 10313412。C 部分（把两个内核移植到 x86）暂不做，先完成必做的报告和幻灯片。
- WP7 C `int8_kernel_x86_study`：发现两个被锁定的内核文件本来就有一条可移植的纯 C 路径（精确的 int32 累加，只有 ARM 上才走 NEON），所以没有按计划“新写可移植的 C”，而是在 x86 上原样编译这两个文件（`mibci.ortkernel.build`、`mibci.ortkernel_swapped.build` 用 `cc -shared -fPIC`，Linux 上也能用）；这是最接近原意的做法，原因写进了计划的 claim_boundary。本机冒烟（M4）：P1 full-22 换上内核后的第二天 logits 与运行库逐位相同。计划 17:29 写定（459 个存档 INT8 图；协议 SHA-256 94023b81b2…）。ONNX Runtime 的 C 头文件从 MacDev 的 `Developer/environments/onnxruntime-headers` 同步到 Longleaf 的 `~/comp690/ort_headers`。提交：VNNI 节点 10314455、没有 VNNI 的 Intel Gold 6140 节点 10314457（先提交的 10314425、10314428 因为 `$HOME` 被单引号包住不会展开而取消，改用相对路径）。
- 17:32 WP1 的 2a fold 模型已全部训练完，先看了当天留出组（只用第一天，探索性；Lee 2019 的第二天预测还没打分）：比只用 16 个电极，6 个拷贝 −0.9 pp，拷贝加 25%/50%/75% 噪声 −4.9/−6.3/−6.8，6 个独立噪声 −7.4，同一条噪声重复 6 次 −2.0。代价主要来自和别的输入无关的独立波动；而且只要 25% 的独立噪声就已付出约三分之二的代价。全部结果出来后用分析脚本重算并写进 1.7.1。
- 17:45 课程报告参考文献逐条核对（Crossref、Europe PMC/PubMed、OpenAlex、arXiv、MLSys 论文集页和 ETH 的论文库；dblp、MDPI、OpenReview 拒绝自动访问）：17 条都存在。改了 6 处：[10] 第一作者只有一个名字 “Abdullah”，原来写的 “M. A.” 没有出处，已删；[1] 补期号；[2][3][8][16] 补会议全名和页码；[4] 补页码。[15] ONNX Runtime 文档确有两点：没有 VNNI 的 x86 上 U8S8 会饱和、建议用 reduce_range（7 位权重）；“旧设备上量化后变慢并不少见”。所以正文在“as ONNX Runtime documents”后补了引用 [15]，并写明 7 位权重是文档推荐的办法，伦理一节不再说成是我们给出的修复。伦理一节的数据说明原来写“两个数据集都是同意参与的成年志愿者”：Lee 2019 的论文写明经高丽大学伦理委员会批准（1040548-KUIRB-16-159-A-2）、所有参与者签了书面知情同意；BCI IV 2a 的数据说明（Brunner 等 2008，全文 6 页）和竞赛综述（Tangermann 等 2012）都没有写同意或伦理审批。已照实改写，并把“健康志愿者”改成“实验室记录”。
- 17:50 WP7 B 第一次提交失败：VNNI 节点的捕获 10313410 运行 9 秒就停了，`final_layer.weight_quantized` 不存在。braindecode 给最后的全连接层套了 max-norm 约束，图里的权重名是 `final_layer.linearconstraint.*`；测试只改过时间卷积一层，所以没发现。本机冒烟时又发现比较阶段的两个问题：最后一层的输出张量叫 `logits_QuantizeLinear_Output`，按卷积层的命名规则推不出来；如果第一个不同的张量不是某个带权重层的输出（例如输入），查找会出错。修改：`mibci.int8_layers` 加层名到权重名的对照（`PREFIX`），比较阶段改为从图里找每层输出接的 QuantizeLinear，找不到对应层时只记名字不做饱和检查；新测试逐层检查五个带权重的层都能改成 7 位。在本机对 P1 的两个模型跑了全部 7 个版本和比较逻辑（人为改动几个输出，确认能找到、能算饱和）。旧计划没有产生任何记录，按 11 月 `matched_budget_unexecuted_v1` 的先例把目录整个归档为 `results/supplementary/int8_layers_study_unexecuted_v1`（本机和 Longleaf 都是），17:49 重新写定计划（设计不变；协议 SHA-256 466ce01be9…）。取消了排在后面的 10313411、10313412，重新提交：VNNI 捕获 10317774 → 没有 VNNI 的 Intel 捕获 10317775 → 比较 10317776。
- 17:55 发现 `tools/longleaf.py push` 只上传 `src`、`scripts`、`tests`、`tools` 和几个配置文件，`kernels/` 和 `mcu/` 从来没有上传过。WP7 C（`int8_kernel_x86_study`）的计划锁定了 `kernels/*.c`，在 Longleaf 上读计划和编译内核时会因为文件不存在而失败；两个任务（10314455、10314457）当时还在排队，没有运行。把 `kernels`、`mcu` 加进上传列表并上传，核对了 Longleaf 上两个 C 文件的 SHA-256 与本机相同；新测试检查每个研究计划锁定的文件都在上传列表里。
- 18:00 WP11 两项本地工作。（1）`docs/REPRODUCIBILITY.md` 新增“重跑期中后的研究”一节：记录和模型在哪、只重建表图的两条命令、在 Longleaf 上重做一项研究的实际命令（push → 远程 plan → train → evaluate `--after` → pull → 分析），以及期末冲刺十项研究的阶段顺序和在哪里跑。（2）旧技术报告 `docs/REPORT.md`：6.2 节的“Reading”原来把 16 对 22 的差别解释成“输入少了，拟合训练试验的余地就小”，并引综述说多余电极导致过拟合；这已被后来的研究否定（掉的电极有信息、每个输入有差不多大的代价、训练数据多 3 倍代价也不变小，Lee 2019 上“解得差的人受益更多”也没复现）。经 `scripts/build_report.py` 改写这一段，新增 6.12 节“Later studies”：指向课程报告，数字全部从各研究的 analysis.json 读，期末冲刺的六项研究按计划里登记的预测逐条列出（还在跑的写明还在跑）；6.11 节“对齐为什么有害，没有检验”改为指向 6.12。重建了 REPORT.md 和 REPORT.pdf。
- 18:02 Longleaf 排队顺序：SLURM 的优先级里有按任务大小的加分，4 个 CPU 的训练任务（5031）比 2 个 CPU 的打分任务（5030）高一点，所以 WP1 的第二天打分（10286193）一直排在 WP4、WP5 的训练后面，只拿到 4 个 CPU。`scontrol top` 对普通用户不开放；改用只降不升的 `nice` 给自己的任务排序，和计划的顺序一致：WP2 +10、WP3 +20、WP8 +30、WP4 +40、WP5 +50（WP1 和 WP7 的小任务不变）。只影响还在排队的任务，依赖关系核对过都还在。
- 18:05 WP7 A 完成：拉回 Longleaf 四种 CPU 的地图（AMD 9654 有 VNNI、AMD 7763、Intel Gold 6140、Intel E5-2680 v3），运行 `scripts/analyze_int8_paths_study.py`。38 种形状里 INT8 比 FP32 慢的：M4 32 种，x86 37–38 种；x86 上每台最多 1 种更快、最多快 6%；M4 上快 3.6 倍的调换后时间卷积（16 通道 depthwise）在 x86 上慢 1.9–3.7 倍。与 1.5.2 的整网计时一致（调换后的 INT8 只在 M4 上比调换后的 FP32 快，24 对里 4 对）。据此改了课程报告：第 6.5 节加一句地图的结果，“Profiling showed” 改成 “Profiling on the M4 showed”；图 5 右边原来只画 FP32、INT8、INT8 调换三组，图注写“调换之后 INT8 处处比 FP32 快”，这只对原来的 FP32 成立，现在补上调换后的 FP32 一组，图注改为“调换后的 INT8 处处比导出的 FP32 快，但只在 M4 上比同样调换的 FP32 快”。POST_MIDTERM 1.7.8 A 写完。
- 18:15 WP1 完成：拉回记录（训练 2,583、第二天打分 2,583、可预测性 63，都等于计划数），运行 `scripts/analyze_input_kind_study.py`。Lee 2019 上事先登记的 K1、K2、K3 全部成立：比只用 16 个电极，拷贝 +0.62 [−0.05, +1.27]、同一条噪声重复 46 次 −1.83、一半噪声 −8.33、独立噪声 −13.33；2a（探索性）分别是 −1.14、−2.24、−5.81、−7.29，25% 噪声已是 −5.15。当天留出组的结果和第二天一样。被去掉的真实电极只有 3.7%（2a）和 28.6%（Lee 2019）的方差不能被留下的 16 个电极线性预测，按混合曲线插值，代价约 1.7 和 4.5 pp；去掉它们准确率只变 +0.32 和 +0.18 pp，倒推出信息约 1.4 和 4.3 pp。**这改了一个已有的说法**：1.6.3、1.6.7 和课程报告草稿用“比同位置的噪声好多少”当作被去掉电极的信息（+4.6，8 次测量平均 +6.2），用噪声的代价当作它们的代价（−6.5）；现在知道真实电极比噪声容易预测、代价小得多，那种算法把两者都高估了。已改：课程报告 6.2 节重写（标题改为“an extra input costs what the others cannot predict”），摘要和贡献第 2 条随之改，图 3 右边加了真实电极的位置（星号），“噪声”和“shared”两个刻度不再重叠，附录 A 的 K1–K3 改为显示计划里的原话；旧报告 6.2 节的读法和 6.12 节改写，重建了 REPORT.md/PDF；POST_MIDTERM 1.7.1 写完，1.6.3 的读法后面加了更正说明。课程报告 PDF 正文仍到第 11 页。幻灯片第 12 页（“信息约 6 / 代价约 6”）还没改，等 WP2、WP3 的结果一起改。
- 18:21 WP10 按 WP1 改幻灯片：第 12 页左边换成“按输入的种类，第二天比只用 16 个电极变化多少”的柱状图（拷贝、同一条噪声重复、一半噪声、独立噪声，两个数据集），右边三个数字改为“被去掉的电极 96% 能被预测（Lee 2019 71%）”“估计的代价 / 信息 −1.7 / +1.4（Lee 2019 −4.5 / +4.3）”“噪声在空间滤波器输出里的占比 43% → 38%”；删掉了偏大的“信息约 6 / 代价约 6”。构建脚本新加检查：K1–K3 任何一条不再成立就报错。第 13 页的“为什么”去掉“每个电极约 1 个点”（真实电极大部分能被预测，代价远小于 1 个点）；第 8 页“为什么 16 个够”一行的模型数加上 WP1，结论改为“输入的代价来自其他输入预测不了的部分”。讲稿第 12 页重写并压到原来的长度；另外发现第 22 页（WP6）加词后第 23–26 页的开始时刻差了 14–17 秒，按每分钟 125 词重新算了全部开始时刻（全文 17:44，去掉方括号 16:22，都在上限内）。问答预案的概述、Q1（原来还写着“7 次抽样 +6.0/−6.3”）和 Q15 已改。重建 `final.pptx`、`final_script.docx`。
- 渲染检查的办法（记下来以后照做）：本机没有安装 Open Sans；无头的 LibreOffice 不经过 macOS 的字体服务，只用自带字体，所以直接转 PDF 时 Open Sans 和 Verdana 都被换成了 Frank Ruhl / Linux Libertine（在沙盒外运行也一样）。做法：写一个 fontconfig 配置（列出 `/System/Library/Fonts`、`Supplemental`、`/Library/Fonts` 和一个放 Open Sans 的目录），运行 soffice 时设 `FONTCONFIG_FILE`；Open Sans 用 Typora 主题里自带的 WOFF2（Regular、Bold、Italic、Bold Italic）经 fontTools 转成 TTF，fontTools 需要的 brotli 用一个调用 Homebrew `brotli` 命令的小替身代替（不下载任何东西）。转出的 PDF 里嵌入的字体核对过确实是 OpenSans-Regular/Bold 和 Verdana/Verdana-Bold。第 8、12、13 页在两种字体下都排得下。
- 18:25 WP7 B 和 C 的第一次结果都不能用，原因相同，记下来：
  - B（`int8_layers_study`）：VNNI 节点（c151612，AMD 9654）和没有 VNNI 的节点（c0317，Intel Gold 6140，捕获记录里的 CPU 型号核对过）上，18 个存档模型、7 个版本的全部 int8 张量和 logits **逐位相同**，没有一个值不同。这和 `int8_portability`（同一种 Gold 6140 上 457/459 个模型的预测变了，平均一致率 85.6%）矛盾。
  - C（`int8_kernel_x86_study`）：两个节点（c151411 AMD 9654 有 VNNI、c0403 Intel）都在加载换上内核的图时报错：x86 上 ONNX Runtime 优化后的图里，分组卷积的输入是 **uint8**，而补上的内核只接受 int8（M4 上优化后的图是 int8）。有 VNNI 的节点也一样。
  - 推断：x86 上 ONNX Runtime 总是把 S8S8 的 QDQ 图改成 U8S8（激活加 128 变成 uint8）；有 VNNI 时 U8S8 的点积是精确的，没有 VNNI 时用 VPMADDUBSW、会饱和。B 的捕获办法把每个 int8 张量都加成图输出，这可能使这一步转换不能做（图输出的类型不能改），x86 于是走了精确的 S8S8 路径，所以两台机器完全一样。证据：本机 M4 上捕获的图和正常推理都是 4 个 QLinearConv 加 1 个 QGemm、logits 逐位相同；但在一个 AMD 9654 节点上（诊断任务本想去 Intel 节点，手写的 `#SBATCH --constraint` 没有生效），P2 的捕获 logits 与正常推理相差最多 0.052，说明 x86 上捕获确实换了计算路径。正在 c0317（Gold 6140）上跑同样的诊断确认：同一批第一天窗口，正常推理（batch 256、32、1）对捕获。
- 18:28 更正：上面 17:55 到 18:25 几条的时刻原来是估的，有几条写成了 18:30–19:15（晚于实际时间）；按文件修改时间改成了实际时刻。以后时刻一律先看 `date`。
- 18:30 诊断结果（Longleaf c0402，Intel Gold 6140）：同一个存档 INT8 图、同一批第一天窗口，**正常推理和 M4 差很多**（P1 full-22：273 个预测里 144 个不同，logits 最多差 6.4；P2 14/270；P1 冻结方案 38/273），batch 256、32、1 都一样；而 `int8_layers_study` 的捕获办法在这台机器上和 M4 逐位相同。所以 WP7 B 第一次的“完全相同”是捕获办法造成的，不是 CPU 的性质。（第一次诊断本想去 Intel 节点，但手写的 sbatch 文件里的 `--constraint` 没有生效，跑到了 AMD 9654 上；改在命令行传 `--constraint` 后正确落在 Gold 6140。）
- 18:33 WP7 B 第二个做法 `int8_capture_study`（新模块 `src/mibci/int8_capture.py`、新脚本和分析脚本、测试 `tests/test_int8_capture.py` 2 项，README 索引已加；第一次的 `int8_layers_study` 原样保留，它的分析表开头加了一段警告，由分析脚本生成）：每台机器先按正常推理的方式优化图（x86 上激活是 uint8），再从优化后的图里取出全部整数张量、关掉进一步优化来运行，每次都核对与正常推理的 logits 逐位相同；另存第一天全部无伪迹窗口的正常推理预测，比较时按 7 个版本统计两台机器预测一致的比例；饱和检验改用 u8×s8 的配对并处理 channels-last 布局。本机端到端冒烟：一个模型 7 个版本的捕获都等于正常推理；人为改动 7 个输出后，比较阶段找到时间卷积、饱和检验能运行。计划 18:32 写定（协议 SHA-256 86df658e4d…）。提交：VNNI 捕获 10325466、Gold 6140 捕获 10325467、比较 10325483（排在两者之后；`tools/longleaf.py submit --after` 现在接受 `a:b` 形式的多个任务号）。
- WP7 C 暂停：x86 上优化后的图在分组卷积前把激活改成 uint8，锁定的两个内核只接受 int8，所以不能原样换进去（两个节点都在加载时报错，没有产生记录）。最接近原意的做法是写 uint8 版本的内核（新文件，锁定文件不动）。WP7 是优先级最低的工作包，B 的第二个做法已经能回答“不准出在哪一层”，所以先做必做的报告和文档，C 等有时间再做；做不了就在进度表里写“未做”和原因。
- 18:40 WP7 B 完成（`int8_capture_study`，比较 10325483 18:37 结束）：两台机器各 126 次捕获都与正常推理逐位相同。没有 VNNI 的 Gold 6140 对 AMD 9654：18 个存档模型第一个不同的层全是时间卷积；该层 1,876,973 个不同的输出 100% 落在“相邻两个 u8×s8 乘积之和超出 int16”的预测里（预测 2,098,302 个，89% 确实不同）。x86 的优化图里有 ONNX Runtime 自己加的 `qdq_s8_to_u8` 量化节点，说明有没有 VNNI 都改成 U8S8。误差逐层放大（logits 83% 不同），第一天全部窗口预测一致 84.8%；单层 7 位变体仍有 7%–15% 的预测不同（时间卷积改 7 位后第一个不同的层移到空间卷积），五层全 7 位才完全一致。分析脚本把 x86 加的三个量化张量标成“input / pool 1 / pool 2 (as uint8)”。课程报告第 6.5 节加一段（数字从分析文件读），POST_MIDTERM 1.7.8 B 写完，进度表更新。`tools/golden.py` 加上 int8_layers、int8_capture、int8_kernel_x86 三项分析的对照。18:39 结束的一次 golden 全部 SAME，只有 compute 和 course-report 不同（运行期间我改了课程报告，新研究也有了记录），最后再跑一次。
- 18:42 WP10 按 WP7 B 改第 21 页：“为什么”一行原来只说“没有 VNNI 的 x86 把两个 8 位乘积加在 16 位里、可能饱和”，现在写实测：第一个不同的层是时间卷积，100% 的不同输出是预测的 16 位溢出（构建脚本读 `int8_capture_study`，结果变了就报错）。讲稿第 21 页加一句逐层核对的结果，“If asked”补 x86 总用 U8S8、单层 7 位不够。重新算开始时刻：全文 17:52，去掉方括号 16:30（上限 18:30 / 17:00）。第 21 页在 Open Sans 和 Verdana 下渲染看过。问答预案 Q11 加逐层核对，Q3 的数字补“x86 上 37–38/38 种形状 INT8 更慢，调换后 INT8 比浮点快只在 M4 上成立”。重建 PPTX 和 Word 讲稿。
- 18:48 WP7 C 重做（最接近原意的版本）：x86 上 ONNX Runtime 把激活改成 uint8，锁定的两个 int8 内核装不进 x86 的图。新写 `kernels/ort_conv_u8.c`（两个算子的 uint8 版本，算法与锁定文件相同：精确 int32 累加、输入零点折进偏置、float32 缩放、截断到 uint8、四舍五入到偶数；只有纯 C 循环，用 `-march=native` 为每台机器编译）、`src/mibci/ortkernel_u8.py`（编译；借用锁定模块的换算子函数，再把节点改到 `mibci.u8` 域）和 `tests/test_ortkernel_u8.py` 2 项（需设 `MIBCI_ORT_HEADERS`，否则跳过）：把输入和零点都加 128 后，uint8 输出等于锁定 int8 算子的输出加 128，逐位相同，覆盖 EEGNet 的两种形状（int8 文件走 NEON）和另两种形状（走 C 循环）。在 WP7 B 取回的 Gold 6140 优化图上换算子成功。第一次计划没有记录，按先例归档为 `int8_kernel_x86_study_unexecuted_v1`（本机和 Longleaf），脚本改用 uint8 算子并只在 x86 上运行，18:46 重新写定计划（协议 SHA-256 8fe628764d…）。提交：VNNI 节点 10327956、Gold 6140 节点 10328059。README 的目录、模块表和脚本索引已改。
- 19:00 写计划的会话追加 WP1b（`residual_surrogate_study`，设计在计划第 5 节），并指出两处文字。已改：（1）WP1 写进去的“代价约 1.7 / 4.5、信息约 1.4 / 4.3”现在处处写明是线性插值的粗略估计，可能偏小（曲线在 0 附近很陡）也可能偏大（混合里的噪声在输入之间独立，真实残差相关）：课程报告 6.2 节、旧报告 6.12 节、第 12 页的说明文字、问答预案 Q1、进度表 WP1 一行；（2）讲稿第 13 页“在第一个数据集上约 1 个点”确实还是旧口径（那是噪声输入的代价，不是真实电极的），改为“和上一页说的一样，每多一个电极带来的信息和它的代价差不多”，等 WP1b 量出来再写数。重建两份报告、PPTX 和 Word 讲稿；讲稿时长不变（17:52 / 16:30），第 12 页在 Verdana 下仍排得下。另外，写计划的会话说 golden 里 compute 一项不同：那是新研究的计划和记录在汇总之后出现，每次拉回新记录后都会这样，收尾时重跑 `scripts/analyze_compute.py` 再对照。
- 19:09 WP1b `residual_surrogate_study` 写好并提交。新模块 `src/mibci/residual_surrogate.py`：最小二乘预测（带截距，每次试验的每个时间点算一个观测，只在训练试验上拟合，与 `input_kind.unexplained_fraction` 的拟合相同，测试核对过），把被去掉的电极分成能预测的部分和残差；重排用 SHA-256（数据集、人、划分、哪组试验、第几次）做种子，拒绝采样直到没有试验留在原位；三种额外输入 real、pred、surr。测试 `tests/test_residual_surrogate.py` 7 项全过。研究脚本 `scripts/residual_surrogate_study.py`：保留电极取 input_cost 计划的通道表；训练、当天留出组、第二天各自在组内重排；2a 第二天把无伪迹试验和有伪迹试验分开重排，保证主分析的试验不会拿到伪迹试验的残差（计划里写明）；训练时记下拟合的 SHA-256，打分时重新拟合并核对相同。分析脚本 `scripts/analyze_residual_surrogate_study.py`：分解表（代价、信息、两者之和、pred 的代价、real 对 full 的顺序对照、每次重排的代价），S1–S4（S4 用 WP1 的每人预测不了的比例，Spearman，参与者 bootstrap 10,000 次、种子 690），Lee 2019 当天留出组用 full-62 代替 real。本机冒烟：2a P1 种子 0 的 final 和 fold-0 各三种条件训练 2 轮并打分，流程通；real 的输入正好是换了顺序的 full-22；分析脚本在合成记录上跑通（含图）。guard 的第二天读取名单、README 索引和模块表、golden 已加。计划 19:07 写定（2,421 个任务；协议 SHA-256 b573de97cf…），推送后提交：训练 10332749（199 个任务，每个 4 CPU）、打分 10332751 排在训练之后，两者 nice 60，排在 WP2–WP5、WP8 之后。
- 19:10 WP7 C 完成（`int8_kernel_x86_study`，两台机器 18:51 起各跑 3–7 分钟，19:09 拉回分析）：AMD 9654（有 VNNI）上换上 uint8 分组卷积内核后，459/459 个模型的第二天 logits 与运行库逐位相同，预测与 M4 全同；Gold 6140（没有 VNNI）上 9/459 与运行库相同（运行库饱和），预测与 M4 不同的模型从 457 个减到 441 个，准确率比 M4 从 −3.97 变为 −1.88。时延（22 个电极，微秒）：AMD 9654 FP32 297、INT8 725、INT8 加内核 392、调换后 FP32 49、调换后 INT8 125、加第二个内核 175；Gold 6140 依次 561、1,823、952、115、324、431。读法：内核在 x86 上精确；只算准一层就把损失减半，其余来自其他层的饱和（与 B 一致）；x86 上 INT8 加内核仍比 FP32 慢，调换后第一层的纯 C 内核还不如运行库，最快的是调换后的 FP32（与 A 一致）。课程报告 6.5 节加一句（数字从结果文件读），POST_MIDTERM 1.7.8 C、进度表（WP7 完成）、问答 Q19 的数字已补。运行日志里 ONNX Runtime 设置线程亲和性的警告不影响结果（会话本身只用一个线程计时）。
- 19:22 WP8 完成（`pretrain_study`，打分 10296626 19:18 结束）：拉回记录（预训练 189、只预训练 621、微调 621 个训练和打分、从头训练 414，都等于计划数），模型也拉回本机备份。运行 `scripts/analyze_pretrain_study.py`：预训练轮数按预算规则定为 2a 125、Lee 2019 33。Lee 2019 上 T1、T2、T3 全部成立：全部 62 个电极预训练加微调比从头训练 +7.65 [+5.39, +9.94]（74.55% 对 66.90%）；46 个噪声输入的代价从 −12.08 降到 −2.17（T2 +9.91）；全部 62 个比共用的 16 个 +4.53 [+2.84, +6.44]（从头训练时 +1.19，不显著）。2a（探索性，只有 8 个别人）预训练有害：全部 22 个微调后比从头训练 −9.15，9 人全差；只预训练约 45%。这给“16 个就够”加了条件：每人只有一天数据、从头训练时成立；有几十个别人的数据时网络能用上更多电极（本人数据多 3 倍不够，C2）。课程报告：摘要加一句（压到 198 词：另删改几处措辞）、6.2 节“检验解释”里的预训练一句重写（标明共用的 16 个）、结论加“直到有足够的别人的数据”；PDF 正文仍到第 11 页。POST_MIDTERM 1.7.4 写完，进度表更新。
- 19:24 WP10 按 WP8 改第 13 页：“为什么”一行改为“只有本人一天的数据时，16 个以后每多一个电极带来的和它的代价差不多；用 45 个别人的数据预训练后，全部 62 个比共用的 16 个高 4.5 pp（事先登记）”，构建脚本读 `pretrain_study` 并在 T1–T3 任何一条不再成立时报错。讲稿第 13 页加一句（以及 If asked 里共用的 16 个是什么、噪声代价 12 → 2、2a 上预训练有害），重新算开始时刻：全文 18:00、去掉方括号 16:38（上限 18:30 / 17:00，余量 30 秒和 22 秒，之后的结果要替换而不是追加）。第 13 页在 Open Sans 和 Verdana 下渲染看过。问答预案 Q6 写完。第 26 页（要点）的第一条提到稀疏惩罚，等 WP3 的结果一起改。WP8 的模型已拉回本机（1,224 个文件：预训练 189、微调 621、从头训练 414）。重建 PPTX 和 Word 讲稿。
- 19:26 WP3 完成（`scalefree_penalty_study`，打分 10293433 19:24 结束）：拉回记录（fold 4,320、最终模型 846、参照 1,710），运行 `scripts/analyze_scalefree_penalty_study.py`。选中的 λ：2a 的 full-22 和 relevance-16 选 0（第二种读法 0.01、0.003），加噪声的两种选 0.01；Lee 2019 full-62 和 relevance-16 选 0.003，加 46 个噪声选 0.1。Lee 2019 上 N1、N2、N3 全部不成立：N1 +1.19 [−0.24, +2.63]，N2 −0.12 [−1.65, +1.38]，N3 没有一档 λ 让噪声的归一化范数中位数低于电极的三分之一（各档最低约 0.48–0.89；λ = 0.03 时反而 1.4–1.8，λ = 0.1 时 2a 留出组掉 7.6–9.2 pp）。读法（按计划）：这种惩罚也关不掉，试到 λ = 0.1；原因不只是 BatchNorm 的尺度不变，一天的数据里网络分不清哪些输入没用；配合 WP8，只有用很多别人的数据预训练才把代价降下来。另：本机 pilot 时 P1 在 0.1 降到 0.064，但 P2 在 0.03 仍 0.71，全体平均下没有出现。POST_MIDTERM 1.7.3 写完，进度表更新。WP3 的模型拉回在后台进行。
- 19:30 按 WP3 改报告和幻灯片。课程报告：6.2 节“检验解释”里尺度无关惩罚一句重写（也关不掉噪声，原因不只是尺度不变）；6.7 节“让网络自己选”改为两种惩罚都没关掉；结论的“下一步”原来是“一个不受尺度影响的输入门控”，这正是 WP3 试过且失败的，改为“用很多用户的数据预训练、在这样的数据上学怎么选电极、在真实单片机上计时”；贡献第 2 条改短并加上预训练。加这些后结论被挤到第 12 页、正文到了上限，所以压缩：6.2 节混合曲线的说明和机制段（group lasso 的权重范数移到新的附录 B）、6.5 节的快慢地图和逐层核对各留一句，数字移到附录 B（另有 x86 内核的结果）。附录不计页数；现在正文到第 11 页（结论从第 11 页开始），约有一页余量留给 WP2、WP4、WP5、WP1b。幻灯片：第 25 页“稀疏惩罚”一格改为两种惩罚 5 条登记预测全部不成立（构建脚本读 N1–N3，变了就报错）；第 26 页要点第一条改为“只有一天数据时……加稀疏惩罚也不行，用很多人的数据预训练能改变这一点”，三条的间距加大；在 Verdana 下第 25 页那一格原来溢出、第 26 页第一条压到第二条，都已改到排得下（Open Sans 同样看过）。讲稿第 25、26 页相应改，全文 18:05、去掉方括号 16:43（上限 18:30 / 17:00）。问答 Q16、Q23 改写。WP3 模型已拉回本机（5,166 个文件）。
- 19:44 WP3b `fresh_noise_study` 写好并提交（写计划的会话 19:35 追加）。新模块 `src/mibci/fresh_noise.py`：`draw0`（input_cost 的输入，逐位相同）、`epoch_noise`（每轮重抽的带限噪声，种子 [694, 数据集, 人, 划分, 模型种子, 轮]）、`unseen_noise`（draw 100，没有模型训练时见过）、`fit_epochwise`（每轮从函数取训练输入，其余和 `mibci.train.fit` 一样；测试核对每轮给同一份输入时逐位相同）。测试 `tests/test_fresh_noise.py` 5 项全过。研究脚本 `scripts/fresh_noise_study.py`：标准化用固定噪声对照的统计量（训练部分的 draw 0），每轮的噪声输入都按它标准化；打分输入和对照逐位相同；另加 `reference` 阶段，用同样口径测 input_cost 的固定噪声模型（噪声占空间滤波器输出的比例、噪声与电极的权重 RMS 之比、训练试验配上 draw 0 和没见过的噪声时的准确率；只读第一天），因为 F3 需要的权重比旧记录里没有。分析脚本 `scripts/analyze_fresh_noise_study.py`。本机冒烟（2a P1，2 轮）：流程通；reference 算出的训练集准确率与 input_cost 的记录逐位相同；固定噪声的旧模型（475 轮）换上没见过的噪声后训练集准确率 96.7% → 85.0%（final）、100% → 72.7%（fold-0），这是原因 A 的一个迹象，正式结果以 F1–F3 为准。guard 名单、README 索引和模块表、golden 已加。计划 19:43 写定（936 个任务；协议 SHA-256 3a6fd4d147…）。提交：训练 10340835、reference 10340836、打分 10340837（排在训练之后），都是 nice 70，排在 WP1b 之后。报告里“一天的数据里网络分不清哪些输入没用”保持现在的写法，不写成已证实的原因。
- 19:48 WP4 完成（`chdrop_study`，打分 10296902 19:37 结束）：拉回记录（新训练 1,881、其中最终模型的第二天打分 504；存档模型重新打分 1,008；不带 dropout 的 fold 参照 459），核对存档模型原样重打 1,008/1,008、参照 459/459 与记录相同，模型也拉回本机。运行 `scripts/analyze_chdrop_study.py`。Lee 2019：B1 不成立（重估 BatchNorm 后 p = 0.2 只好了 +0.38 [−0.04, +0.81]，救回 2.74 pp 损失的 14%；空间滤波器后那层存下的方差只有实际的 72%–76%，但这不是主要原因）；B2 成立（p = 0.05 比 0.2 少掉 +2.45 [+1.56, +3.37]）；B3 成立（坏一个电极时 p = 0.05 重估后的模型掉得少 +0.49 [+0.28, +0.72]）。2a 上坏一个电极：不加 dropout 平均掉 14.8 pp（最坏 31.7，同时坏 3 个掉 28.1），p = 0.05 掉 1.9（最坏 10.3）；Lee 2019 0.81 对 0.32。读法：损害在学到的滤波器，不在统计量对不上，1.6.3 的解释保留；dropout 换来对坏电极的耐受，是部署上有用的结论。已改：课程报告 6.7 节通道 dropout 一条、伦理一节加“部署时要考虑坏电极”；第 25 页的卡片（不是 BatchNorm 对不上；坏一个电极 1.9 pp 而不是 14.8，构建脚本在 B1/B3 结果变化时报错，Verdana 下排得下）；讲稿第 25 页加半句；问答 Q18 写完；POST_MIDTERM 1.7.5、进度表。讲稿时间只剩约 20 秒余量，所以删短第 25、26 页的开头，把第 19、22 页各一句次要的话放进方括号：现在全文 18:09、去掉方括号 16:38。
- 19:59 WP2 完成（`long_training_study`，打分 10287890 19:53 结束）：拉回记录（训练 1,485、最终模型第二天打分 756，都等于计划数），模型拉回本机。运行 `scripts/analyze_long_training_study.py`。Lee 2019 上 L1、L2、L3 全部不成立：L1 +0.52 [−0.40, +1.49]（噪声代价 −13.31 → −12.79），L2 +0.68 [−0.61, +1.94]，L3 +0.19 [−0.48, +0.85]。第 475 轮权重与旧研究逐位相同的有 495/1,485，正好是噪声条件（旧模型来自同一套代码）；另两种条件的旧模型来自别的研究、有浮点差别，按计划都用本研究自己的 475 保存点当对照。写计划的会话问“加噪声的模型留出组是不是先升后降”：用 `curves.csv` 看，没有。Lee 2019 加 46 个噪声的模型留出组第 100、475、1,900 轮为 50.3%、51.6%、51.7%，一直在随机水平附近，而它的训练损失下降得最快（第 475 轮 0.03，不加噪声 0.14）：一开始就学到推广不出去的东西，不是先学会后忘掉；这和 WP3b 的原因 A 相符，但只是读法。2a（探索性）4 倍长让三种条件第二天都多约 1.7–2.5 pp，比较不变。已改：课程报告 6.2 节的“训练 4 倍长”一句（数字从 `curves.csv` 和分析文件读）；这一句加上后结论又被挤到第 12 页，原因是第 8 页图 4 放不下留了大块空白、第 11 页差约十分之一页，所以把图 4 移到 6.4 节后面、六张图各矮 0.1–0.25 英寸（看过图 3、5、6 仍清楚），结论回到第 11 页。幻灯片第 25 页把“数据少、代价大”一格和训练 4 倍长合成“数据少、训练久”（构建脚本在 L1–L3 任何一条成立时报错，Verdana 下排得下）；讲稿第 25 页加半句，全文 18:13、去掉方括号 16:42；问答 Q5 写完。POST_MIDTERM 1.7.2、进度表已更新。
- 20:27 WP5 完成（`alignment_mechanism_study`，打分 10298351 20:24 结束）：拉回记录（训练 2,781、最终模型第二天打分 756、协方差谱 126，都等于计划数），模型拉回本机。运行 `scripts/analyze_alignment_mechanism_study.py`（两个数据集都是探索性的）。M1 成立：Lee 2019 的 62 个电极上同一天内对齐也降 −4.02 [−6.27, −2.05]；M3 成立：Ledoit–Wolf 系数平均 0.0061；M2 差一点不成立：ea-top 去掉 48% 的害处、ea-shrink10 88%（要求两者都 ≥ 50%），另加的 ea-floor 102%、ea-lw 23%。第一天协方差里 62 个方向平均 49 个的方差不到最大方向的 1%（2a 22 个里 17 个）。读法：害处来自白化把几十个几乎没有方差的方向放大成单位方差，相当于加了很多预测不了的输入（和 WP1 同一机制）；不放大或稍微收缩就几乎全部消失；置零只去掉一半，因为也丢了信号（2a 上 ea-top 掉 8.4 pp，正是计划写定时担心的）。已改：课程报告 6.7 节对齐一条（数字从结果文件读；结论仍在第 11 页）、旧报告 6.11/6.12 重建、第 25 页卡片（构建脚本在 M1、M3 或 ea-floor 的比例变化时报错，Verdana 下排得下）、讲稿第 25 页和 If asked、问答 Q17。讲稿全文 18:17、去掉方括号 16:47（余量各约 13 秒）。POST_MIDTERM 1.7.6、进度表已更新。
- 20:28 WP11 先做 README：第 1 节加两行（期中后在 Longleaf 上的研究和期末冲刺、课程报告与期末汇报）；第 2 节加“期末冲刺的结果”一小节（WP1、WP8、WP2/WP3、WP4、WP5、WP6/WP7 各一条，标明探索性和事先登记）；第 12 节更新版本管理状态（截至 11-22 期中后的工作都没有提交），并加三条：Longleaf 的 CPU 上限和用 nice 排顺序、手写 sbatch 的 constraint 没有生效、计划作废时归档为 `_unexecuted_v1` 的做法、渲染幻灯片要用 FONTCONFIG_FILE。第 13 节等最后的全部检查做完再写。
- 20:29 WP11 检查：`python3 -u scripts/audit_project.py --deep --require-artifacts` 通过（historical_evidence_consistent、replay_and_training_ready 都为真，缺失的模型产物 0；冻结方案对 4/9 人掉 2 pp 以上：P1 −4.27、P4 −2.04、P7 −2.88、P9 −3.41，和原来一致）。问答预案的两分钟概述加一句“只有能用很多别人的数据预训练时，全部电极才比 16 个好”；问答里已没有“待补”的题。
- 20:43 WP3b 完成（`fresh_noise_study`，训练 10340835、reference 10340836、打分 10340837 都是 COMPLETED，打分 20:33 结束）：拉回记录（训练 936、reference 936、打分 936、环境 199，和 Longleaf 上的 3,008 份对得上），936 个模型拉回本机。运行 `scripts/analyze_fresh_noise_study.py`。Lee 2019 上 F1 成立：重抽减固定 +12.18 [+8.77, +15.70]（47/54 人）；F2 不成立：重抽减只用 16 个 −1.15 [−2.12, −0.23]，区间下限低于 −2；F3 成立：噪声与电极权重比的差 −0.88 [−0.92, −0.85]，重抽时中位数 0.19（低于 1/3），固定时 1.07。2a（探索性）：6 个噪声的代价 −7.29 → −1.92，16 个 −12.51 → −3.70，9/9 人重抽更好。固定噪声的模型配原来的噪声训练试验 100%（Lee 2019），换一份没见过的噪声 61.5%；重抽的模型两种都是 92.3%；噪声占空间滤波器输出 86% 对 8%。按计划“F1 成立、F2 不成立：两种原因都有份”的读法：Lee 2019 上约 91% 的代价（2a 上 74% 和 70%）来自网络拿每轮重复的噪声记住训练试验，剩下约 1 分和记不记得住无关。写进 POST_MIDTERM 1.7.3b（并改 1.7.3 里“网络分不清哪些输入没用”一条）、README 第 2 节、课程报告 6.2 节（新的一段，接在尺度无关惩罚后面；那句“网络分不清哪些输入没用”删掉，开头改为“EEGNet does not learn to ignore these inputs”；附录 A 加 F1–F3，F3 和 H3 标出单位）、旧报告 6.12 节（LATER 加 fresh_noise_study 和 residual_surrogate_study，F3、S4 标出单位）、问答 Q15 重写，Q4、Q5、Q16 补追问时的说法；第 25 页稀疏惩罚卡片末句改为 “the noise helps it memorize the training trials”，第 26 页第一条和讲稿改为 “EEGNet uses useless inputs to memorize (its training) trials”。课程报告重建后正文仍到第 11 页，参考文献第 12 页。第 12、13 页和 Q1、总览等 WP1b 一起改
- 20:45 WP1b 打分 10332751 结束：53 个任务里 37 个完成、16 个失败（0–14 和 52 号）。失败原因是脚本自己的核对：打分时在训练部分上重新做最小二乘，系数的 SHA-256 要和训练时记下的 `fit_sha256` 相同，否则报错停下（第一个报错 `bnci2a_sub-01_seed-3_real-6_fold-0: the refitted prediction differs from the one used in training`）。用 `scontrol show node` 核对：训练的 199 个任务和打分成功的 37 个都在带 avx512 的 AMD 节点上（c151406–c151418、c151601），失败的 16 个都在不带 avx512 的 AMD 节点上（c130407、c130601、c151401–c151403）；打分任务提交时没有加 `--constraint`。两种 CPU 上 BLAS 走的指令路径不同，最小二乘的末位不同，核对把它拦住了，没有任何打分用到不同的系数。处理：不改核对，用 `python3 tools/longleaf.py submit residual_surrogate_study --stage evaluate --tasks 17 --cpus 2 --hours 2 --constraint 'amd&avx512'` 重新提交（10356893）；已经打完分的作业会被跳过。教训：一个阶段要逐位复现另一个阶段算过的东西时，两个阶段要用同一种节点
- 20:59 WP1b 完成（`residual_surrogate_study`，重交的打分 10356893 的 17 个任务全部 COMPLETED）：拉回记录（训练 2,421、打分 2,421、环境 199，都等于计划数），2,421 个模型拉回本机。运行 `scripts/analyze_residual_surrogate_study.py`。Lee 2019 上 S1 成立：real 减 surr +4.30 [+3.09, +5.50]；S2 成立：surr 减只用 16 个 −4.12 [−5.31, −3.02]；S3 成立：pred 减只用 16 个 +0.27 [−0.49, +1.02]；S4 不成立：54 人之间“预测不了的比例”和代价的 Spearman 相关 −0.079 [−0.340, +0.200]。按计划“S1、S2 成立：信息和代价都存在、互相抵消”的读法：Lee 2019 两者之和 +0.18 [−1.25, +1.56]；线性插值的粗估（代价 4.49）和实测很接近。2a（探索性）：代价 −1.50 [−2.29, −0.70]，信息第二天 +0.34 [−0.53, +1.18]、当天留出组 +2.20 [+1.14, +3.21]；只差输入顺序的对照 real 减 full-22 −0.84 [−2.32, +0.53]（种子误差 0.65），在噪声以内。训练集准确率（Lee 2019）：只用 16 个 96.1%、加预测部分 96.0%、加真实电极 98.9%、加 surr 99.0%。为弄清 S4 不成立是不是因为每人的代价量不准，在草稿里看了种子对半的可重复性（种子 0 对种子 1–2，Spearman 0.49），把它作为事后检查加进分析脚本（`analysis.json` 的 `seed_split_cost`，summary.md 里标明 post hoc、不是登记的检验）。又看了代价和基线准确率的关系：同一批种子算是 −0.44，换成不同种子的两半是 −0.20 到 −0.42，一部分是回归均值，没有写进报告，只记在 POST_MIDTERM 1.7.1b。写入：POST_MIDTERM 1.7.1b（并在 1.7.1 那条插值的读法前注明“粗估，实测见 1.7.1b”）；README 第 2 节；课程报告 6.2 节（插值的几句换成实测的分解、训练集准确率和 S4，附录 A 加 S1–S4，S4 标 rho；图 3 的星号改为实测代价并带区间，图例和图注随之改）；摘要的那句改为 “through the part the other inputs cannot predict, mostly because the network memorizes its training trials with it”（仍 198 词），贡献第 2 条和结论同样改；旧报告 6.12 节（有 WP1b 记录时用实测的数，附录列 S1–S4）；幻灯片第 12 页第二、三个数字改为实测的 −4.1 / +4.3（2a −1.5 / +0.3）和 WP3b 的 −13.3 → −1.1，“为什么”一行改为记答案的说法，标签上的模型数加上 WP1b、WP3b；第 8 页最后一行改为 31,572 个模型、“the network memorizes trials with it, even with a penalty”；讲稿第 12 页和 If asked 改写；问答总览和 Q1 重写。课程报告加了这些后，图 6 在第 10 页放不下，结论到第 12 页、参考文献第 13 页：正文 12 页，仍在 10–12 页以内，第 12 页剩下的空白正好留给作者要填的贡献和重叠说明，所以没有再压；把“训练 4 倍长”一段的曲线细节移到附录 B
- 20:59 WP11 重跑 `scripts/analyze_compute.py`（Longleaf 上已没有任务）：Longleaf 合计 58,626 个模型、7,452.3 CPU 小时（重跑前的上一版是 42,921 和 4,488.5，11-22 上午的版本是 40,338 和 3,961.5）。顺带改了一个错：没有环境记录、也不训练的研究，原来一律写 “M4 Mac mini (no training stage)”，可 WP7 的 int8_capture、int8_kernel_x86、int8_layers 是在 Longleaf 的 x86 节点上跑的，int8_paths 两边都有。现在按这些研究其他记录里的 `host`（`*.ll.unc.edu` 是 Longleaf）定地点，只有计划、从没执行的文件夹写 “plan only, not run”；模型数和 CPU 小时的合计不变。重建幻灯片时又发现读取函数要求每条 runs 记录都有 `train_time_s`，而 `chdrop_study` 的 runs 里有 459 条不训练的参照记录（把不带 dropout 的旧 fold 模型按同样口径重新测一遍，`*_reference.json`），算力汇总一直跳过它们；之前的汇总里还没有这项研究，所以没报错。改成和 `analyze_compute.py` 同一口径（只数有训练时间的记录）。讲稿第 9 页和问答 Q21 改为约 59,000 次训练、约 7,500 CPU 小时；课程报告引言里写死的 “40,000 trainings” 改为从汇总读（about 59,000）
- 21:12 WP9–WP11 收尾。`python3 tools/check.py` 先报 format 和 E501（`build_course_report.py`、`build_report.py` 新加的几行），用 `uvx ruff@0.17.6 format` 格式化并拆开三行超长的，之后四项全部 ok；unittest 112 项通过，2 项 uint8 内核测试要 `MIBCI_ORT_HEADERS`，用 MacDev 上的 `environments/onnxruntime-headers` 跑也通过。格式化后重建两份报告。`python3 tools/golden.py --all` 第一次 40 项全部 SAME，但最后报“运行期间仓库被改动”（我在它运行时改了 README 和文档），不改任何文件重跑一次：40 项全部 SAME，退出码 0（2 分 22 秒）。`python3 -u scripts/audit_project.py --deep --require-artifacts` 通过（历史证据一致、回放和训练就绪、缺失的模型产物 0）；这次不带 `--output` 的那一遍默认写了 `results/maintenance/audit.json`，单独跑 `golden.py audit` 核对仍是 SAME。`scripts/package_release.py` 原来不收 `kernels/`、`mcu/`（报告 Artifacts 一节说内核在这两个目录），加上后打包 `release/COMP690_final_2026-11-22.zip`：112,814 个文件、181 MB、SHA-256 79f0e957…；逐个文件按包内清单核对哈希全部一致，没有原始脑电、虚拟环境和模型文件，记录在 `release/package_verification_2026-11-22.json`（11-04 那份校验文件不动）。用网页读取助手读了课程页面（为 WP12 准备）：有单独的 “AI tools” 一节，允许用 AI 工具理解概念、找相关工作，不允许把课程材料和其他同学的作品交给 AI 工具，没有要求写使用说明；报告前面要有标题、成员姓名和邮箱、各人贡献、与其他课程或研究的重叠、少于 200 词的摘要；报告会公开给以后的学生，可以发邮件给老师要求不公开。README 第 1 节加交付包一行、第 11 节列出只有用户能做的三件事、第 13 节加本轮核验；POST_MIDTERM 的日程表把 11-22 的两行改为完成，1.6.6 小节标题下加指向 1.7.3b 的更正；进度表 WP9、WP10、WP11 改为完成，WP12 改为已问用户
- 21:37 WP12 第 1 问：用户回复“推送，排除副本”。提交前核对：只有本会话在这个文件夹里工作（父会话没有在运行）；本地 main 和 origin/main 都在 ffd32a1，远端没有新提交；仓库是公开的；在要提交的文件里查了私钥、令牌、密码和密钥赋值、私人邮箱、.env 和证书文件，都没有。课程报告 Artifacts 一节把“链接待定”换成仓库地址（第一次写法把网址包在加粗里，导出器把后面的 `</b>.` 一起当成链接，PDF 没生成；改成网址后面直接跟空格），重建后页码不变。按用户的选择，提交时排除 `docs/COURSE_REPORT 2.pdf`（iCloud 生成的旧版副本）、`presentation/midterm (iCloud version 2026-11-09 13h57 EST).pptx`（iCloud 冲突副本）、顶层的 `midterm_script.docx`（11-09 较早的版本）、`.claude/scheduled_tasks.lock`（本机锁文件）和用户的中文笔记；这几个文件留在本地没动。随后提交并推送到 main
