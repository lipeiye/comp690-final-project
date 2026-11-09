# 相同微调预算的剪枝补充实验

## 目的和时间边界

原剪枝模型在 475 epoch 基础训练后增加了 100 epoch 微调；原未剪枝基线没有这段额外训练。补充对照用于拆分“继续训练的影响”和“删除结构的影响”。

本补充计划于 2026-11-04 在原 session-2 结果已知之后追加。必须标记 **supplementary / post-test**，不能称为原预注册分析，也不能据此重新选择原最终方案。

## 固定设计

| 项目 | 设置 |
| --- | --- |
| 被试与种子 | 全部 9 人，原种子 0、1、2 |
| 基础模型 | 原冻结方案的两个 pruning bases：full-22、relevance-16 |
| 输入通道 | 使用各人/种子已保存的 session-1 排名；与原训练记录通道列表校验 |
| 原权重存在 | 对其 SHA-256 固定后使用 |
| 原权重缺失 | 明确选择 `--rebuild-bases`，独立训练 475 epoch，不冒充恢复原权重 |
| 三条分支 | 不剪枝、剪枝 25%、剪枝 50% |
| 相同微调 | 每条 100 epoch、Adam 1e-4、batch 64、相同 seed/批次抽样；各自重建优化器 |
| PTQ | 每条 FP32/INT8，校准仅使用该被试 session-1 训练数据 |
| 主/次评价 | session-2 无 artifact / 全部试次；先种子内平均到被试，再比较 |
| 区间 | 被试配对差异的 10,000 次 percentile bootstrap，95%，seed 690 |
| 禁止 | 新的测试集调参、筛选 seed/被试、按补充结果重选原最终模型 |

`protocol.json` 在训练/评价前记录设计、基础来源、冻结配置哈希、通道排名哈希和训练/评价核心源码哈希。执行时检查这些哈希，修改协议须使用新的补充目录。

## 执行

先安装锁定依赖，恢复并校验数据。

有原权重时：

```bash
.venv.nosync/bin/python scripts/run_matched_control.py --stage plan
```

缺少原权重而执行独立重建时：

```bash
.venv.nosync/bin/python scripts/run_matched_control.py --stage plan --rebuild-bases
```

下面两组训练可同时运行，保持原 MPS/CPU 被试分配：

```bash
.venv.nosync/bin/python scripts/run_matched_control.py --stage train --subjects 1 2 3 4 5 6 --workers 3
.venv.nosync/bin/python scripts/run_matched_control.py --stage train --subjects 7 8 9 --workers 6
```

训练完成后：

```bash
.venv.nosync/bin/python scripts/run_matched_control.py --stage evaluate --workers 6
.venv.nosync/bin/python scripts/run_matched_control.py --stage analyze
.venv.nosync/bin/python scripts/build_report.py
```

实际执行的是哪些命令、是否并行和是否完成，以 [工作记录](WORK_LOG.md) 及 `docs/evidence/` 日志为准。协议命令不是已运行的证明。

## 结果位置与解释

- 结果：`results/supplementary/matched_budget/`。
- 权重和导出：`data.nosync/models/supplementary/matched_budget/`。
- `base`：475 epoch 基础模型。
- `0`：基础模型不剪枝，继续训练 100 epoch。
- `25` / `50`：基础模型剪枝后，继续训练相同的 100 epoch。
- `0 − base`：额外训练影响。
- `25/50 − 0`：相同微调预算下的剪枝影响。
- `25/50 − base`：与原比较类似的混合影响，不能作为单独的剪枝效应解释。

独立重建时保留与历史同条件预测的一致率。预测完全相同，也不能证明权重逐位一致。若不一致，仍完整报告新实验，不通过换种子寻找与旧分数更相似的运行。

这组实验改善了预算可比性，但不能消除小样本、跨天漂移、事后分析和训练后端差异等限制。它不增加独立被试数。
