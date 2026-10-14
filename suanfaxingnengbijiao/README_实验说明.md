# 综合算法对比实验说明

## 实验目的

对比 **QVNS-NSGA2**、**NSGA-II**、**MOPSO**、**MOEA/D** 四种多目标优化算法在地铁物流选址问题上的性能，生成以下对比分析：

1. **总成本收敛曲线** - 展示各算法在迭代过程中总成本的下降趋势
2. **碳排放成本收敛曲线** - 展示各算法在碳排放成本方面的优化效果
3. **超体积指标收敛曲线** - 评估各算法的Pareto前沿质量
4. **收敛速度对比（成本改进率）** - 对比各算法的收敛速度

## 文件结构

```
suanfaxingnengbijiao/
├── scripts/
│   ├── run_comprehensive_algorithm_comparison.py  # 主实验脚本
│   ├── plot_convergence_curves.py                 # 绘图模块
│   ├── run_experiment.bat                         # Windows运行脚本
│   └── run_experiment.sh                          # Linux/Mac运行脚本
├── code/
│   └── main.py                                    # 核心算法实现
└── results/
    └── comprehensive_comparison/                  # 实验结果输出目录
```

## 快速开始

### 方法1：使用运行脚本（推荐）

**Windows:**
```bash
cd scripts
run_experiment.bat
```

**Linux/Mac:**
```bash
cd scripts
chmod +x run_experiment.sh
./run_experiment.sh
```

### 方法2：直接运行Python脚本

```bash
cd scripts
python run_comprehensive_algorithm_comparison.py \
    --algorithms QVNS-NSGA2 NSGA2 MOPSO MOEAD \
    --runs 10 \
    --pop-size 100 \
    --max-gen 300 \
    --seed 20260512
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithms` | 要运行的算法列表 | QVNS-NSGA2 NSGA2 MOPSO MOEAD |
| `--runs` | 每个算法的独立运行次数 | 10 |
| `--pop-size` | 种群大小 | 100 |
| `--max-gen` | 最大迭代代数 | 300 |
| `--seed` | 基础随机种子 | 20260512 |
| `--top-n` | 候选地铁站数量 | 17 |
| `--scenario` | 需求场景 | base |
| `--model-setting` | 模型设置 | no_budget_range |
| `--output-dir` | 输出目录 | ../results/comprehensive_comparison |
| `--quiet` | 静默模式（不显示详细日志） | False |
| `--skip-plots` | 跳过绘图 | False |

## 自定义实验

### 示例1：只运行部分算法

```bash
python run_comprehensive_algorithm_comparison.py \
    --algorithms QVNS-NSGA2 NSGA2 \
    --runs 5
```

### 示例2：快速测试（少量迭代）

```bash
python run_comprehensive_algorithm_comparison.py \
    --runs 3 \
    --max-gen 50 \
    --quiet
```

### 示例3：高精度实验（更多运行次数）

```bash
python run_comprehensive_algorithm_comparison.py \
    --runs 30 \
    --max-gen 500
```

## 输出结果

实验完成后，在 `results/comprehensive_comparison/` 目录下会生成以下文件：

### 数据文件

- `convergence_history_raw.csv` - 原始收敛历史数据（所有运行）
- `convergence_history_aggregated.csv` - 聚合收敛数据（均值和标准差）
- `hypervolume_history.csv` - 超体积指标历史数据

### 图表文件

- `convergence_total_cost.png/pdf` - 总成本收敛曲线
- `convergence_carbon_cost.png/pdf` - 碳排放成本收敛曲线
- `convergence_hypervolume.png/pdf` - 超体积指标收敛曲线
- `convergence_improvement_rate.png/pdf` - 成本改进率曲线
- `convergence_combined.png/pdf` - 综合收敛曲线（四合一）

## 结果解读

### 1. 总成本收敛曲线
- **纵轴**：总成本（元/天）
- **横轴**：迭代代数
- **解读**：曲线下降越快，算法收敛速度越快；最终值越低，算法找到的解越优

### 2. 碳排放成本收敛曲线
- **纵轴**：碳排放成本（元/天）
- **横轴**：迭代代数
- **解读**：评估算法在环境友好性目标上的优化能力

### 3. 超体积指标收敛曲线
- **纵轴**：超体积指标（HV）
- **横轴**：迭代代数
- **解读**：HV值越大，Pareto前沿质量越好；曲线上升越快，算法收敛越快

### 4. 成本改进率曲线
- **纵轴**：成本改进率（%）
- **横轴**：迭代代数
- **解读**：直观展示各算法相对于初始解的改进幅度

## 算法简介

### QVNS-NSGA2
- **全称**：Q-learning Variable Neighborhood Search NSGA-II
- **特点**：结合Q学习和变邻域搜索的改进NSGA-II算法
- **优势**：自适应搜索策略，局部搜索能力强

### NSGA-II
- **全称**：Non-dominated Sorting Genetic Algorithm II
- **特点**：经典多目标遗传算法
- **优势**：快速非支配排序，拥挤度距离保持多样性

### MOPSO
- **全称**：Multi-Objective Particle Swarm Optimization
- **特点**：基于粒子群的多目标优化算法
- **优势**：收敛速度快，参数少

### MOEA/D
- **全称**：Multi-Objective Evolutionary Algorithm based on Decomposition
- **特点**：基于分解的多目标进化算法
- **优势**：将多目标问题分解为多个单目标子问题

## 注意事项

1. **运行时间**：完整实验（4个算法 × 10次运行 × 300代）可能需要较长时间（数小时）
2. **内存占用**：确保有足够的内存（建议8GB以上）
3. **中断恢复**：实验会实时保存中间结果，可以查看 `convergence_history_raw.csv`
4. **随机性**：不同的随机种子会产生不同的结果，建议多次运行取平均

## 故障排除

### 问题1：导入错误
```
ModuleNotFoundError: No module named 'main'
```
**解决方案**：确保在 `scripts/` 目录下运行脚本，或检查 `sys.path` 设置

### 问题2：内存不足
```
MemoryError
```
**解决方案**：减少 `--runs` 或 `--pop-size` 参数

### 问题3：中文显示乱码
**解决方案**：检查 `chinese_font_utils.py` 中的字体配置，确保系统安装了中文字体

## 技术支持

如有问题，请检查：
1. Python版本（建议3.8+）
2. 依赖包是否完整安装（numpy, pandas, matplotlib等）
3. 数据文件是否存在

## 更新日志

- **2026-05-12**：初始版本，支持四种算法对比实验
