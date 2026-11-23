# 已合并到 AGENTS.md

2026-11-20 起，代理工作规则只保留一份：[AGENTS.md](AGENTS.md)。原来这里的内容分成了三类：

- 工具能检查的规则交给了 `tools/check.py`、`tools/guard.py`、`tools/golden.py` 和编辑钩子，说明见 [docs/GUARDRAILS.md](docs/GUARDRAILS.md)。
- 需要判断的规则（研究边界、措辞、代码放哪里、算力）留在 [AGENTS.md](AGENTS.md)。
- 数据格式、命令和代码地图在 [README.md](README.md)。

已经在代码里修好的"已知路径例外"删掉了。合并前最后一版的全文存档在 [docs/evidence/pre_refactor_2026-11-20/agent.md](docs/evidence/pre_refactor_2026-11-20/agent.md)，只作历史记录。
