# Q11 分支与 13 组联合实验清单

本包在上一版 IGD/Spread + Q9 分支基础上新增 Q11A、Q11B、Q11C 三个综合增强分支。

## 默认 13 组

1. NSGA-II
2. VNS-only
3. Q-learning-only
4. QVNS-Q3
5. QVNS-Q9
6. QVNS-Q9A
7. QVNS-Q9B
8. QVNS-Q9C
9. QVNS-Q9D
10. QVNS-Q11
11. QVNS-Q11A
12. QVNS-Q11B
13. QVNS-Q11C

## Q11 分支定位

- Q11A：稳定收敛分支，降低后期随机性，目标是降低 HV 标准差和 IGD 波动。
- Q11B：分布多样性分支，扩大 archive 和边界保留比例，目标是降低 Spacing/Spread 并提高唯一解数。
- Q11C：快速综合分支，减少 VNS 作用对象和尝试次数，目标是降低运行时间并保持综合质量。

## 推荐命令

```bash
python scripts/run_qvns_ablation_experiment.py --comparison-mode all --pop-size 100 --max-gen 300 --runs 10 --quiet
```
