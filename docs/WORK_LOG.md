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
