# 代码质量护栏：把纠正写成检查，文字规则放最后

这份说明讲本项目怎样防止代码变乱，不管是人写的还是 AI 代理写的；也说明每条规则由谁执行，以及这套做法出自哪里。

## 1. 出处和原则

Lauren Tan（GitHub 名 poteto）在 2026 年 11 月 11 日的 AI Engineer Code Summit 上做了题为 "All Code Is Slop (Except Mine)" 的演讲。她是 SpaceXSI（原 xAI）Grok Bot 的工程负责人，也是 React Compiler 核心成员，之前在 Cursor、Meta 和 Netflix 工作。

- 演讲视频：<https://www.youtube.com/watch?v=O_zcc6Xl12g>
- 演讲的幻灯片文字与文字稿：<https://ai.engineer/talks/O_zcc6Xl12g-all-code-is-slop-except-mine>（文字版由 AI Engineer 整理）
- 她 10 月 16 日介绍"倒金字塔"的长帖：<https://x.com/poteto/status/2111133080526495985>
- 她把这套方法写成代理技能，其中最明确的是 `/correct`：<https://github.com/cursor/plugins/blob/main/pstack/skills/correct/SKILL.md>

她的意思，用白话说：说别人的代码是烂代码很容易，难的是你敢不敢信任代理在你睡觉时合并进去的第 3000 个改动。代理会照抄代码库里已有的写法，包括坏写法，所以代码库本身就是代理最大的输入，要像打理花园一样保持干净。每当你纠正代理一次，不要在规则文档里多写一行，而要把这次纠正变成一个确定的检查，或者改接口、改结构，让这个错误根本写不出来。她幻灯片的结论是："lint against it or categorically eliminate the problem"。

## 2. 倒金字塔的五层

她的金字塔是倒着画的：最上面最宽，投入最多；越往下成本越高、越难扩大规模。下表是这五层在本项目里各对应什么。

| 层 | 作用 | 本项目里的实现 |
| --- | --- | --- |
| 1. 代码库本身 | 改接口、改结构，让错误写不出来 | `data.load_session(s, 2)` 默认报错；路径只在 `mibci.paths` 定义；参与者 bootstrap 只有 `mibci.stats.bootstrap_ci` 一份；主实验的目录和冻结计划集中在 `mibci.layout`，脚本不再互相导入；新实验经 `mibci.study.Study` 建计划，自动锁定源码的 SHA-256 |
| 2. 静态分析 | 确定性的检查，错误信息直接说该怎么改 | `ruff format`；`pyproject.toml` 里的 lint 规则，禁用的 API 各附一句"该用什么"（`numpy.load`、`torch.load`、`json.dump`、`pickle`、`os.environ`）；`tools/guard.py` 检查项目自己的规则；`tests/` 的单元测试；`.github/workflows/checks.yml` 在 CI 里跑全部检查 |
| 3. 验证 | 让代码真的跑起来，拿出证据 | `tools/golden.py` 在沙盒里重建报告、表、图、幻灯片、审计和清单，与仓库逐字节比较；`audit_project.py --deep` 从保存的预测重算全部指标；量化模拟与 ONNX Runtime INT8 的逐条一致率；每个新实验先在本地做冒烟测试 |
| 4. 文字规则 | 只放需要判断的事，越少越好 | `AGENTS.md`，里面有一张"规则 ↔ 由什么强制执行"的对照表；`.claude/skills/debt-cleanup/` |
| 5. 人工审查 | 最贵，目标是尽量不留东西给它 | 你看结论、数字和措辞；代码层面的问题应该在前四层就被拦下 |

前两层让错误写不出来或者一写就报错；第三层是她现在投入最多的一层；第四层她说"只留一点"。

## 3. 每天怎么用

```bash
python3 tools/check.py           # 格式、lint、项目规则、单元测试，全部 ok 才算完成
python3 tools/check.py --fast    # 不跑单元测试（提交前和代理钩子用）
python3 tools/golden.py --all    # 改了分析、报告、幻灯片代码时：重建并逐字节比较
python3 tools/check.py --todo    # 还允许存在的旧问题，也就是清理队列
python3 tools/check.py --tighten # 清理之后，把允许的数量降到现在的水平
```

这些检查会自动触发：代理每改一个 `.py` 文件，`.claude/settings.json` 里的钩子就调用 `tools/agent_hook.py` 检查这个文件，有问题当场退回；`git commit` 前跑 `--fast`（`.githooks/pre-commit`）；推到 GitHub 后 CI 跑全部检查。`tools/` 只用标准库，直接用 `python3` 运行；ruff 用 `uvx ruff@0.17.6` 调用，不改训练环境的 `uv.lock`。

## 4. 新问题挡住，旧问题慢慢清

她的规则是：一个写法已经很常见时，检查只在一次改动让它变多时才报错；例外写在出问题的那一行，附上原因。本项目这样实现：

- `tools/lint_baseline.json` 按"文件 × 规则"记下引入 lint 时已有的问题数。任何一项都不能变多，新文件必须是零。清理之后用 `--tighten` 把数量降下来；它只降不升，有一项变多就拒绝执行。
- 例外写成 `# noqa: 规则号  原因`，原因必须写在同一行，`tools/guard.py` 会检查。
- 让代理清理旧问题，借鉴了 OpenAI 在 "Harness engineering" 里写的"后台代理定期打扫"：<https://openai.com/index/harness-engineering/>。本项目的固定流程写在 `.claude/skills/debt-cleanup/SKILL.md`：选文件，先跑一遍证明确认是绿的，只改形式，再证明一次，最后 `--tighten`。

这次重构的数字：按项目规则集，起点 622 处问题；格式化后 370 处；三个代理分组清理后剩 4 处，都是三个历史脚本里过长的 `main` 函数。三组都用金标准对照证明输出逐字节不变；PDF 前后都是 18 页、每页文字相同。

## 5. 每个检查都要能抓住一次真实的错误

她要求新检查必须证明"在一次真实发生过的错误上会失败"。`tests/test_guard.py` 把这些真实的错误一一复现，确认 guard 会报错：

| 检查 | 它抓住的真实错误 |
| --- | --- |
| 被锁定的文件不能改 | 我用 `ruff --fix` 整理导入顺序时，顺手改了被 matched-budget 实验锁定的 `run_matched_control.py` |
| 脚本不能互相导入 | `analyze.py` 从 `summarize_validation.py` 导入函数，另有两处同类写法 |
| 脚本必须有 argparse | 8 个脚本没有参数解析，运行 `--help` 会直接执行并写文件 |
| 3.9 语法 | 这台 Mac 的 `python3` 是 3.9，一个清理代理加的 `zip(strict=True)` 让 `audit_project.py --deep` 崩溃 |
| README 列出每个脚本 | 这个检查本身原来用子串匹配，`check.py` 被当成已列出，因为 README 里有 `redundancy_check.py` |
| 例外必须写原因 | 只写规则号的 `# noqa` 读不出为什么可以例外 |

## 6. 被锁定的文件

三项已经完成的补充实验（matched budget、electrode count 两批）在 `protocol.json` 里记下了 9 个源文件的 SHA-256：`src/mibci/` 的 data、model、train、prune、experiment、deploy，以及三个 `run_*` 脚本。这些实验的结果就是这些字节算出来的；改了它们，就无法再证明结果对应哪份代码。所以 guard 发现它们变了就报错，ruff 的格式化和自动修复都跳过它们，代理钩子直接拒绝对它们的编辑。新功能写进新模块；新实验用 `Study` 建自己的计划，计划自动锁定它用到的文件。
