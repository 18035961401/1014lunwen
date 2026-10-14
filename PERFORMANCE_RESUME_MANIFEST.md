# 长跑性能与断点续跑修复说明

本版本面向 `pop_size=100, max_gen=300` 的正式实验，对性能瓶颈和长跑可靠性做了专项修复。正式比较算法只有两类：`NSGA-II` 与 `QVNS-NSGA-II`。消融实验中的 `VNS-only` 和 `Q-learning-only` 仅用于解释 QVNS-NSGA-II 的模块贡献，不作为独立正式算法。

## 1. 性能修复

### 非支配排序向量化

`code/main.py` 中 `fast_non_dominated_sort()` 采用 NumPy 向量化支配矩阵，保持原有约束支配规则不变：

- 可行解优先支配不可行解；
- 不可行解之间按 `CV` 较小者优先；
- 可行解之间按双目标 Pareto 支配关系排序。

### 训练阶段快速评价

新增 `_evaluate_objective_cv_fast()`，用于算法迭代阶段只计算 `objs` 和 `CV`，避免每次评价都构造完整成本明细字典。完整明细仍由 `evaluate_solution(..., collect_parts=True)` 用于最终导出。

### 地铁线路缓存

`_station_lines()` 与 `_path_lines()` 已增加实例级缓存，避免在约束检查中反复访问 Pandas 行对象。

### 减少重复 repair

NSGA-II / QVNS-NSGA-II 生成 offspring 时不再在交叉后和变异后重复执行两次完整修复。现在交叉只做编码合法化，变异后统一执行一次修复。

### VNS 长跑限额

`_run_qvns_nsga2_core()` 支持 `vns_max_elites` 参数。消融脚本默认：

```bash
--vns-max-elites 20
--vns-trials 3
--archive-multiplier 2.0
```

这能防止 `p=200` 时 VNS 每两代对大量精英做多邻域深搜导致长跑不可控。若要增强 VNS，可提高：

```bash
--vns-max-elites 50 --vns-trials 4 --archive-multiplier 3.0
```

## 2. 断点续跑

`scripts/run_qvns_ablation_experiment.py` 默认启用断点续跑。脚本每完成一个 “情景-算法-运行编号” 组合，都会立即写出：

- `ablation_run_metrics.csv`
- `ablation_pareto_points.csv`
- `ablation_summary.csv`
- `fig_ablation_*.svg`

如果中断后再次运行同一命令，脚本会自动跳过已完成组合。

禁用断点续跑并重新开始：

```bash
python scripts/run_qvns_ablation_experiment.py --no-resume ...
```

## 3. 推荐正式实验命令

标准 NSGA-II：

```bash
python code/run_nsga2_for_thesis.py --pop-size 100 --max-gen 300 --runs 5 --quiet
```

QVNS-NSGA-II：

```bash
python code/run_qvns_nsga2_for_thesis.py --pop-size 100 --max-gen 300 --runs 5 --quiet
```

QVNS-NSGA-II 消融：

```bash
python scripts/run_qvns_ablation_experiment.py --pop-size 100 --max-gen 300 --runs 5 --quiet
```

默认情景唯一为：

- `no_budget_range`

默认消融组为：

- `NSGA-II`
- `VNS-only`
- `Q-learning-only`
- `QVNS-NSGA-II`
