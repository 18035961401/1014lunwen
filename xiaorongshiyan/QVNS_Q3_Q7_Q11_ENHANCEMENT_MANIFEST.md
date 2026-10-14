# QVNS-Q3/Q7-Q11 增强调优清单

本版本保留基础消融三组：

- NSGA-II
- VNS-only
- Q-learning-only

并将默认 QVNS 调优组设置为：

- QVNS-Q3：当前平衡基准组
- QVNS-Q7：稳定性 + archive 增强组
- QVNS-Q8：Spacing + 边界解保留组
- QVNS-Q9：自适应快速 VNS 组
- QVNS-Q10：稳定性 + Spacing 组
- QVNS-Q11：archive + 边界保留 + 自适应 VNS 综合增强组

## 主要代码变化

1. `code/main.py`
   - 增加 archive 边界解保留截断逻辑。
   - 增加 mixed VNS 候选选择：成本端边界、服务端边界、高 crowding distance 点共同参与局部搜索。
   - 增加自适应 VNS 触发：当最近若干代 archive HV 改进低于阈值时触发 VNS。
   - 增加运行统计：`actual_vns_calls`、`archive_final_size`、`archive_unique_solutions`。

2. `scripts/run_qvns_ablation_experiment.py`
   - 默认调优组从 Q1-Q6 改为 Q3/Q7/Q8/Q9/Q10/Q11。
   - 仍保留 Q1-Q6 复现入口。
   - 输出 `ablation_run_metrics.csv` 和 `ablation_summary.csv` 时记录 archive 与自适应 VNS 参数。

3. `data/补充约束数据/QVNS调优参数组_Q3_Q7_Q11.csv`
   - 新增 Q3/Q7-Q11 的参数表。

4. 文档同步更新
   - `README_独立运行说明.md`
   - `docs/消融实验说明.md`
   - `docs/QVNS-NSGA-II算法说明.md`
   - `data/补充约束数据/README_补充约束数据说明.md`
   - `data/模型参数总表_确认版.csv`

## 推荐运行命令

```bash
python scripts/run_qvns_ablation_experiment.py --comparison-mode all --pop-size 100 --max-gen 300 --runs 3 --quiet
```

只运行增强调优组：

```bash
python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q3 Q7 Q8 Q9 Q10 Q11 --pop-size 100 --max-gen 300 --runs 3 --quiet
```

## 验证

- `pytest -q`: 22 passed
- 小规模联合冒烟运行成功：基础消融三组 + Q3/Q7/Q8/Q9/Q10/Q11 共 9 组均可运行。
