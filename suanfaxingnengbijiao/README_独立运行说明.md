# QVNS-NSGA-II 地铁物流模型独立运行说明

本包为第五章地铁物流转运点选址—分配模型的可复现实验包。当前版本仅保留 **无预算约束情景 `no_budget_range`**：固定成本进入综合成本目标 `Z1_total_cost`，但不再设置固定成本预算上下限。

## 1. 当前模型口径

- 4 km 为地铁转运点至顺丰站点的服务边界；
- 覆盖率针对全局总需求计算，4 km 内需求尽可能由地铁服务；
- 4 km 外或受容量、班次、换乘等约束无法承接的货流由公路运输；
- 转运点容量为 SL/TR/HB = 5000/7000/10000；
- 转运点固定成本为 SL/TR/HB = 8000/12000/16000；
- 最低/最高负载率为 0.15/0.85；
- 每条线路最多设置 3 个转运点；
- 每条线路平峰期设置 5 趟可用货运班次；
- 删除目标函数中的时间折算成本和公路直送处理成本；
- 保留碳成本，并按“排放因子 × 碳价格”计算；
- 新增末端接驳碳成本和平峰期货运班次组织成本。

## 2. 运行 QVNS-NSGA-II

```bash
python code/run_qvns_nsga2_for_thesis.py --runs 1
```

快速测试运行：

```bash
python code/run_qvns_nsga2_for_thesis.py --runs 1 --pop-size 20 --max-gen 2 --skip-plots --quiet
```

输出目录默认为：

```text
results/qvns_nsga2_thesis/no_budget_range/
```

## 3. 运行基准 NSGA-II

```bash
python code/run_nsga2_for_thesis.py --runs 1
```

快速测试运行：

```bash
python code/run_nsga2_for_thesis.py --runs 1 --pop-size 20 --max-gen 2 --skip-plots --quiet
```

输出目录默认为：

```text
results/nsga2_no_budget_baseline/no_budget_range/
```

## 4. 运行测试

```bash
pytest -q
```

## 5. 关键数据表

```text
data/补充约束数据/模型全局参数_无预算.csv
data/补充约束数据/候选转运点最终属性表.csv
data/补充约束数据/平峰期列车班次与碳成本参数.csv
data/补充约束数据/物流园区货源比例与接入信息.csv
```

这些表是当前模型的正式参数来源。代码会优先读取这些表，以避免参数只硬写在程序中。

## 6. 运行消融与 QVNS 参数调优实验

当前脚本 `scripts/run_qvns_ablation_experiment.py` 已支持基础消融组与增强型 QVNS 调优组联合比较。默认 `--comparison-mode all` 会同时运行：

```text
NSGA-II、VNS-only、Q-learning-only、QVNS-Q3、QVNS-Q7、QVNS-Q8、QVNS-Q9、QVNS-Q10、QVNS-Q11
```

其中 Q3 是上一轮筛选出的平衡基准组；Q7–Q11 分别从 archive、边界解保留、降低后期随机性、自适应 VNS 触发和分布均衡角度增强，用于进一步改善 HV 标准差、Spacing、唯一解数和运行时间。

正式联合对比命令示例：

```bash
python scripts/run_qvns_ablation_experiment.py --comparison-mode all --pop-size 100 --max-gen 300 --runs 3 --quiet
```

若只想比较增强型 QVNS 参数组：

```bash
python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q3 Q7 Q8 Q9 Q10 Q11 --pop-size 100 --max-gen 300 --runs 3 --quiet
```

旧的 Q1–Q6 仍可通过 `--qvns-tuning-groups Q1 Q2 Q3 Q4 Q5 Q6` 复现实验，但不再作为默认运行组。

输出位于：

```text
results/qvns_ablation/
```
