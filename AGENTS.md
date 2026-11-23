# 代理工作规则

这是本项目给编码代理的最后一层约束。能用机器检查的规则都已经交给工具，见 [docs/GUARDRAILS.md](docs/GUARDRAILS.md)。这里只写工具判断不了、需要你自己判断的事。用户在任务里的明确要求优先于这里的规则。

项目是 COMP 690 的运动想象 EEG 研究（包名 `mibci`）：BCI IV 2a 数据，9 名被试，每人两天；研究电极选择、EEGNet、结构剪枝和量化部署。使用说明、命令和代码地图见 [README.md](README.md)，研究叙述见 [docs/REPORT.md](docs/REPORT.md)。

## 1. 交付前必须做的事

```bash
python3 tools/check.py          # 格式、lint 棘轮、项目规则、单元测试；全部 ok 才算完成
python3 tools/golden.py --all   # 改了分析、报告、幻灯片的代码时必跑；输出必须 SAME，或者能说明为什么变
```

编辑钩子会在你每次改 `.py` 文件后自动检查。它报的问题当场修，不要攒着。只声称你实际运行并通过的检查。

## 规则和执行方

每条规则都写明由谁执行。"人工判断"的规则没有任何东西会在被违反时报错，所以它们只能是机器真的判断不了的事。一条人工判断的规则如果又被违反了一次，就在同一个改动里把它升级成检查：在 `tools/guard.py` 加一项，并在 `tests/test_guard.py` 里复现这次错误。一条规则防的错误已经不可能再发生时，就把它删掉。

| 规则 | 执行方 |
| --- | --- |
| 不改被已完成实验锁定的源文件 | `guard` pinned-source；编辑钩子直接拒绝；ruff 跳过这些文件 |
| 只有白名单里的文件能读第二天数据 | `data.load_session` 默认报错；`guard` session2-access |
| 共享代码放 `src/mibci/`，脚本之间不互相导入 | `guard` script-imports |
| 不写死本机路径 | `guard` hardcoded-paths |
| 脚本要有 docstring、argparse 和 main 保护，`--help` 不执行任何操作 | `guard` script-contract |
| 读写记录用项目的函数（`load_session`、`load_bundle`、`write_json`、`mibci.paths`） | ruff 的禁用 API 规则（TID251），报错信息里写了该用什么 |
| lint 例外写在出问题的那一行，并写明原因 | `guard` noqa-reasons |
| 用裸 `python3` 运行的文件保持 3.9 可用 | ruff 按文件设 3.9 目标；`guard` py39-compat |
| 格式统一；旧的 lint 问题不增加 | `ruff format`；`tools/check.py` 的 lint 棘轮 |
| 历史结果一个字节都不改 | `guard` frozen-evidence |
| 文档链接有效，每个脚本都写进 README | `guard` doc-links、doc-index |
| 重构不改任何发布的数字 | `tools/golden.py`：CI 跑不需要 EEG 数据的生成器，`electrode` 和 `redundancy` 要在本地跑 |
| 新实验的计划写明是否在看过测试结果之后 | `guard` study-labels 只检查字段在不在；写得对不对是人工判断 |
| 第二天不参与任何选择 | 人工判断（第 2 节） |
| 结论措辞准确 | 人工判断（第 3 节） |
| 只用 Longleaf，不用 Sycamore；不替用户输密码 | 人工判断；`tools/longleaf.py` 只连 Longleaf |

## 2. 研究边界：工具看不出来，你要自己守住

- **第二天（session 2）是测试日。** 凡是"选择"都不能看它：选电极、选 epoch 预算、选位宽、选阈值、选种子、选被试。工具只检查谁调用了读取函数，管不到你拿结果去做决定。
- **测试后的分析要标明。** 11 月 3 日冻结之后追加的实验，都是在看过测试结果之后做的，只能写成探索性结果，不能用来替换冻结的最终方案（relevance-16-prune50 INT8），也不能写成事先计划的检验。每个新实验的 `protocol.json` 都要写 `historical_test_results_already_known` 和 `claim_boundary`，guard 只检查字段在不在，内容写得对不对要你负责。
- **改方法就开新实验。** 新的计算方式写进新模块，新实验用 `mibci.study.Study` 建自己的计划和目录。不要为了让旧计划继续跑去改锁定文件或改哈希字段。
- **历史结果一个字节都不改。** `results/` 里的历史记录和 `docs/evidence/` 是证据。派生的表格和报告由生成器重建，要改就改生成器（`scripts/build_report.py`、`presentation/deck_source/`、`presentation/pptx/*.js`），不要手改生成物。

## 3. 写结论时的措辞

- 均值接近不等于等效：65.01% 对 65.04% 不能写成"一样"，而且有 4/9 的人下降超过 2 个百分点。
- 区间包含 0 不等于证明了没有差别；只有 9 名被试，不能推广到新的人。
- 两种不确定性要分开：参与者 bootstrap 区间（换一批人会怎样）和种子误差（同一批人重新训练会怎样）。
- 压缩不自动等于加速。INT8 在 M4 上变慢是 ONNX Runtime 对分组空间卷积的通用慢路径造成的，不是机器不支持 8 位运算；调换前两层后 INT8 才变快。
- 参数个数、initializer 字节、ONNX 文件字节、理论激活内存、进程 RSS、单片机 SRAM 是不同的量，分别报告。低于 8 位的结果是模拟的整数网格，没有在真实的低位内核上计时。
- 回放是录制数据的重放，不是在线闭环；亚毫秒的时延不包含 2 秒的观察窗口。

## 4. 代码放哪里

- 共享代码放 `src/mibci/`，脚本之间不互相导入（guard 检查）。
- 一个实验分两个脚本：`scripts/<名字>_study.py` 负责计划、训练、打分，其源码会被计划锁定；`scripts/analyze_<名字>.py` 负责出表画图，可以随时改，不会让已经跑完的结果失效。
- 新脚本要有 docstring 和 argparse，写进 README 第 9 节的索引（guard 检查）。

## 5. 算力和环境

- 重的计算（批量训练、扫参、跨硬件计时）放到 UNC Longleaf，用 `python3 tools/longleaf.py`，或直接 `ssh longleaf`：这是 `~/.ssh/config` 里的别名，用专用钥匙 `~/.ssh/id_ed25519_longleaf` 登录，不需要密码和 Duo，这台 Mac 上的任何代理都能用，前提是 UNC VPN 连着。项目在 Longleaf 的 `~/comp690`（home 配额 50 GB，账户没有 `/work`）；大文件下载用 `datamover` 分区，缓存后删掉原文件。**绝对不要用 Sycamore**，那是用户做研究用的集群。永远不要替用户输入密码，即使用户把密码发给了你。
- 本地 M4 Mac mini 同时跑着用户的其他重任务。本地只做 lint、单元测试、小规模冒烟测试和生成报告；开多进程或计时之前先看 `uptime`。
- 环境以 `uv.lock` 为准，不为排版或检查工具升级依赖。ruff 用 `uvx ruff@0.17.6`。这台 Mac 的 `python3` 是 3.9，`tools/` 和 README 里用 `python3` 运行的脚本要保持 3.9 可用（`pyproject.toml` 已按 3.9 语法检查这些文件）。

## 6. Git

仓库是公开的。只在用户要求时提交或推送，不改写已推送的历史；`data.nosync/`、模型、虚拟环境、`release/` 和任何密钥都不进仓库。
