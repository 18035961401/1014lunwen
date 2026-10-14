# 最终同步清单

本版本已统一为无预算模型口径和两类正式算法：

- `NSGA-II`
- `QVNS-NSGA-II`

## 已清理内容

- 删除/停用旧预算区间情景说明，唯一情景为 `no_budget_range`。
- 删除核心代码中未作为正式算法使用的旧辅助求解器。
- 将 `code/metro_data_plotting_tools.py` 重构为数据读取与绘图工具，不再作为旧多算法实验入口。
- 修正 `CHANGELOG_QVNS_NSGA2.md`、`PERFORMANCE_RESUME_MANIFEST.md`、`docs/消融实验说明.md` 等说明文件中的旧口径。
- 消融实验仅用于解释 QVNS-NSGA-II 模块贡献：NSGA-II、VNS-only、Q-learning-only、QVNS-NSGA-II。

## 已验证

- `pytest -q`：22 passed。
- NSGA-II 冒烟运行成功。
- QVNS-NSGA-II 冒烟运行成功。
- 已清理 `results`、`__pycache__`、`.pytest_cache`。
