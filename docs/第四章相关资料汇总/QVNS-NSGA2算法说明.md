# QVNS-NSGA-Ⅱ 算法说明

QVNS-NSGA-Ⅱ 是本包最终采用的调参版改进算法。它以 NSGA-Ⅱ 为全局多目标搜索框架，以 VNS 作为主要局部开发模块，以保守型 Q-learning 作为超启发式扰动选择器。

## 1. 核心思想

- **NSGA-Ⅱ**：负责非支配排序、拥挤距离和精英保留。
- **VNS**：负责对外部 archive / Pareto 精英进行局部邻域改进。
- **Q-learning**：不直接生成方案，而是在每代选择轻量扰动动作。
- **Safeguard 回滚**：若 Q-learning 动作导致 archive-level 奖励低于阈值，则回滚该代种群和 archive，防止负迁移。

## 2. 最终参数配置

当前 QVNS-NSGA-Ⅱ 默认采用 Q11C 快速综合增强配置，默认求解规模为 `pop_size=100, max_gen=300`。

```text
alpha = 0.15
gamma = 0.88
epsilon = 0.30
epsilon_min = 0.010
epsilon_decay = _epsilon_decay_for_target(0.30, 0.010, max_gen)
q_action_apply_rate = 0.25
q_action_profile = safe
safeguard = True
safeguard_threshold = -0.15
vns_interval = 0
vns_mode = adaptive
vns_elite_rate = 0.12
vns_trials_per_neighborhood = 2
vns_max_neighborhood = 3
archive_size = max(pop_size, int(round(2.5 * pop_size)))
archive_feedback_interval = 8
archive_feedback_rate = 0.15
archive_boundary_keep_rate = 0.20
vns_candidate_strategy = mixed
stagnation_window = 10
hv_improvement_threshold = 0.0025
min_vns_gap = 5
```

## 3. 与旧口径的区别

原先的 Q-learning 对整代 offspring 干预过强，可能使完整算法弱于 VNS-only。本版将 Q-learning 改为“局部、保守、可回滚”的策略，使 VNS 成为开发主力，Q-learning 作为辅助扰动选择机制。最终 Q11C 配置进一步采用自适应 VNS 触发、混合候选解选择和 archive 边界保留，以在降低运行时间的同时保持综合前沿质量。

## 4. TOPSIS 折中解

QVNS-NSGA-Ⅱ 与 NSGA-Ⅱ 统一使用 `choose_compromise()`，仅基于 `Z1_total_cost` 和 `Z2_metro_service_ratio` 两个目标选择折中解。

## 5. QVNS 参数调优组

为便于在同一个消融实验脚本中比较 QVNS-NSGA-Ⅱ 的不同参数组合，`scripts/run_qvns_ablation_experiment.py` 提供 `--comparison-mode` 参数：

- `ablation`：只运行 NSGA-Ⅱ、VNS-only、Q-learning-only；
- `tuning`：只运行指定 QVNS 调优组；
- `all`：同时运行基础消融组和 QVNS 调优组，默认值。

其中 Q11C 是当前 QVNS-NSGA-Ⅱ 的最终默认配置。

当 `vns_mode=adaptive` 时，算法根据最近 `stagnation_window` 代的 archive HV 改进量判断是否触发 VNS。若 HV 改进低于 `hv_improvement_threshold`，且距离上次 VNS 至少超过 `min_vns_gap` 代，则触发局部搜索。
