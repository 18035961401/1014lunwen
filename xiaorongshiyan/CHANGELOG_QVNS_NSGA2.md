# QVNS-NSGA-II 最终同步版变更说明

- 将最终改进算法统一命名为 `QVNS-NSGA-II`，基准算法统一为 `NSGA-II`。
- 模型、代码、运行脚本、参数表和文档均收敛到唯一情景：`no_budget_range`。
- 固定成本只进入综合运营成本目标，不再设置固定成本预算上下限。
- 新增并启用无预算全局参数表、候选转运点最终属性表、平峰期列车班次与碳成本参数表、物流园区货源比例与接入信息表。
- 成本函数删除时间折算成本和公路直送处理成本，保留道路/地铁/接驳碳成本，并新增平峰期货运班次组织成本。
- 线路容量由“线路总容量”改为“5 趟平峰列车的不可拆分装箱校验”。
- 消融实验仅围绕 `NSGA-II`、`VNS-only`、`Q-learning-only` 和 `QVNS-NSGA-II` 四组展开，其中后三组均服务于解释 QVNS-NSGA-II 组件贡献。

## 参数调优实验扩展

- `scripts/run_qvns_ablation_experiment.py` 支持 `--comparison-mode {ablation,tuning,all}`；
- 新增 QVNS-Q1 至 QVNS-Q6 六组参数配置，可与 NSGA-II、VNS-only、Q-learning-only 在同一汇总表中比较；
- `code/main.py` 新增 `vns_max_neighborhood` 与 `qvns_phase_schedule`，支持 Q4/Q5 的高覆盖强化邻域和 Q6 的分阶段 VNS 调度；
- 新增 `data/补充约束数据/QVNS调优参数组_Q1_Q6.csv`。
