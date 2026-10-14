# 3/4/5 km 服务半径敏感性分析运行说明

## 运行口径
- 脚本：`run_service_radius_sensitivity.py`
- 服务半径：3 km、4 km、5 km
- 当前运行参数：`runs=10`、`pop_size=100`、`max_gen=300`、`QVNS-Q11C` 配置。
- 说明：当前结果已按论文主表更稳口径重跑，可直接用于对比分析。

## 汇总结果
|   radius_km |   runs |   min_cost_mean |   max_service_ratio_mean |   topsis_cost_mean |   topsis_service_ratio_mean |   topsis_open_count_mean |   topsis_space_coverage_radius_ratio_mean |   topsis_fulfillment_ratio_within_radius_mean |   topsis_avg_feeder_distance_mean | representative_open_stations   |
|------------:|-------:|----------------:|-------------------------:|-------------------:|----------------------------:|-------------------------:|------------------------------------------:|----------------------------------------------:|----------------------------------:|:-------------------------------|
|           3 |     10 |          202480 |                 0.393381 |             210774 |                    0.303720 |                  3.7 |                                  0.382807 |                                      0.785420 |                           2.05925 | 高朋大道；高升桥；省体育馆；桐梓林          |
|           4 |     10 |          199007 |                 0.433141 |             206037 |                    0.304342 |                  3.1 |                                  0.470845 |                                      0.647153 |                           2.61382 | 火车南站；省体育馆；太平园                  |
|           5 |     10 |          193285 |                 0.472831 |             202784 |                    0.328861 |                  3.5 |                                  0.602237 |                                      0.542179 |                           2.93346 | 省体育馆；太平园；桐梓林                    |

## 初步解释
- 10 次重复后，服务半径增大总体会降低最低成本并提高空间覆盖能力，其中 5 km 的空间覆盖比例最高。
- TOPSIS 折中服务比例在 3/4/5 km 下分别约为 0.304、0.304 和 0.329，说明随机种子会影响单次折中站点组合，论文中不宜只依据某一次运行解释敏感性方向。
- 4 km 自动代表解与全文主方案仍不完全一致；为保持论文图文一致，4 km 地图采用全文代表性方案回放，统计表仍采用 10 次敏感性实验均值。

## 输出文件
- `service_radius_run_metrics.csv`
- `service_radius_summary.csv`
- `service_radius_pareto_points.csv`
- `fig_service_radius_sensitivity.png`
- `fig_service_radius_sensitivity.svg`
- `fig_service_radius_sensitivity.pdf`
- `../service_radius_maps/` 下的 3/4/5 km 武侯区地图效果图
