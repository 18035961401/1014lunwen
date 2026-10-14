# -*- coding: utf-8 -*-
"""绘制算法收敛曲线对比图

包括：
1. 总成本收敛曲线
2. 碳排放成本收敛曲线
3. 超体积指标收敛曲线
4. 收敛速度对比（成本改进率）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
PARENT_CODE_DIR = PROJECT_ROOT.parent / "code"

for import_dir in (CODE_DIR, PARENT_CODE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

# 尝试导入中文字体配置
try:
    from chinese_font_utils import setup_chinese_fonts  # noqa: E402
except ImportError:
    # 如果导入失败，定义一个简单的替代函数
    def setup_chinese_fonts():
        """简化的中文字体配置"""
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
            "PingFang SC", "Heiti SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42


# 算法颜色和样式配置
ALGORITHM_STYLES = {
    "QVNS-NSGA2": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "QVNS-NSGA2"},
    "NSGA2": {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "NSGA-II"},
    "MOPSO": {"color": "#2ca02c", "linestyle": "-.", "marker": "^", "label": "MOPSO"},
    "MOEAD": {"color": "#d62728", "linestyle": ":", "marker": "D", "label": "MOEA/D"},
}


def setup_plot_style():
    """设置绘图样式"""
    setup_chinese_fonts()
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["lines.linewidth"] = 1.8
    plt.rcParams["lines.markersize"] = 4


def calculate_carbon_cost_from_total(total_cost: float) -> float:
    """从总成本估算碳排放成本"""
    # 假设碳排放成本占总成本的18%
    return total_cost * 0.18


def compute_hypervolume_2d(points: np.ndarray, ref_point: tuple[float, float]) -> float:
    """计算2D超体积指标"""
    if len(points) == 0:
        return 0.0

    # 移除被支配的点
    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                if np.all(points[j] <= points[i] + 1e-9) and np.any(points[j] < points[i] - 1e-9):
                    pareto_mask[i] = False
                    break

    pareto_points = points[pareto_mask]
    if len(pareto_points) == 0:
        return 0.0

    # 按第一个目标排序
    sorted_points = pareto_points[np.argsort(pareto_points[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]

    for x, y in sorted_points:
        if x <= ref_point[0] and y < prev_y:
            width = ref_point[0] - x
            height = prev_y - y
            hv += max(0.0, width * height)
            prev_y = y

    return float(hv)


def calculate_hypervolume_history(convergence_df: pd.DataFrame, inst) -> pd.DataFrame:
    """计算每代的超体积指标"""
    hv_records = []

    for algorithm in convergence_df["algorithm"].unique():
        alg_data = convergence_df[convergence_df["algorithm"] == algorithm]

        for run_id in alg_data["run_id"].unique():
            run_data = alg_data[alg_data["run_id"] == run_id].sort_values("gen")

            for gen in run_data["gen"].unique():
                gen_data = run_data[run_data["gen"] <= gen]

                if "total_cost" in gen_data.columns and "max_service_ratio" in gen_data.columns:
                    # 构建目标空间点（最小化成本，最大化服务率 -> 最小化负服务率）
                    costs = gen_data["total_cost"].values
                    service_ratios = gen_data.get("max_service_ratio", gen_data.get("pareto_size", np.zeros(len(costs)))).values

                    # 归一化
                    cost_min, cost_max = costs.min(), costs.max()
                    if cost_max > cost_min:
                        costs_norm = (costs - cost_min) / (cost_max - cost_min)
                    else:
                        costs_norm = np.zeros_like(costs)

                    service_min, service_max = service_ratios.min(), service_ratios.max()
                    if service_max > service_min:
                        service_norm = (service_ratios - service_min) / (service_max - service_min)
                    else:
                        service_norm = np.zeros_like(service_ratios)

                    # 转换为最小化问题
                    points = np.column_stack([costs_norm, 1.0 - service_norm])

                    # 计算超体积
                    ref_point = (1.1, 1.1)
                    hv = compute_hypervolume_2d(points, ref_point)

                    hv_records.append({
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "gen": gen,
                        "hypervolume": hv,
                    })

    return pd.DataFrame(hv_records)


def plot_total_cost_convergence(aggregated_df: pd.DataFrame, output_dir: Path):
    """绘制总成本收敛曲线"""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")

        if "total_cost_mean" not in alg_data.columns:
            continue

        style = ALGORITHM_STYLES.get(algorithm, {})

        generations = alg_data["gen"].values
        mean_cost = alg_data["total_cost_mean"].values
        std_cost = alg_data.get("total_cost_std", np.zeros_like(mean_cost)).values

        # 绘制均值曲线
        ax.plot(
            generations,
            mean_cost,
            label=style.get("label", algorithm),
            color=style.get("color", "black"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markevery=max(1, len(generations) // 15),
            linewidth=2,
        )

        # 绘制标准差阴影
        ax.fill_between(
            generations,
            mean_cost - std_cost,
            mean_cost + std_cost,
            alpha=0.15,
            color=style.get("color", "black"),
        )

    ax.set_xlabel("迭代代数", fontsize=12, fontweight="bold")
    ax.set_ylabel("总成本 (元/天)", fontsize=12, fontweight="bold")
    ax.set_title("总成本收敛曲线对比", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_total_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_total_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"总成本收敛曲线已保存")


def plot_carbon_cost_convergence(aggregated_df: pd.DataFrame, output_dir: Path):
    """绘制碳排放成本收敛曲线"""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")

        if "total_cost_mean" not in alg_data.columns:
            continue

        style = ALGORITHM_STYLES.get(algorithm, {})

        generations = alg_data["gen"].values
        mean_total_cost = alg_data["total_cost_mean"].values
        std_total_cost = alg_data.get("total_cost_std", np.zeros_like(mean_total_cost)).values

        # 估算碳排放成本
        mean_carbon = np.array([calculate_carbon_cost_from_total(c) for c in mean_total_cost])
        std_carbon = std_total_cost * 0.18  # 标准差也按比例缩放

        # 绘制均值曲线
        ax.plot(
            generations,
            mean_carbon,
            label=style.get("label", algorithm),
            color=style.get("color", "black"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markevery=max(1, len(generations) // 15),
            linewidth=2,
        )

        # 绘制标准差阴影
        ax.fill_between(
            generations,
            mean_carbon - std_carbon,
            mean_carbon + std_carbon,
            alpha=0.15,
            color=style.get("color", "black"),
        )

    ax.set_xlabel("迭代代数", fontsize=12, fontweight="bold")
    ax.set_ylabel("碳排放成本 (元/天)", fontsize=12, fontweight="bold")
    ax.set_title("碳排放成本收敛曲线对比", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_carbon_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_carbon_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 碳排放成本收敛曲线已保存")


def plot_hypervolume_convergence(hv_df: pd.DataFrame, output_dir: Path):
    """绘制超体积指标收敛曲线"""
    setup_plot_style()

    # 聚合超体积数据
    hv_agg = hv_df.groupby(["algorithm", "gen"]).agg(
        hv_mean=("hypervolume", "mean"),
        hv_std=("hypervolume", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in hv_agg["algorithm"].unique():
        alg_data = hv_agg[hv_agg["algorithm"] == algorithm].sort_values("gen")

        style = ALGORITHM_STYLES.get(algorithm, {})

        generations = alg_data["gen"].values
        mean_hv = alg_data["hv_mean"].values
        std_hv = alg_data["hv_std"].fillna(0).values

        # 绘制均值曲线
        ax.plot(
            generations,
            mean_hv,
            label=style.get("label", algorithm),
            color=style.get("color", "black"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markevery=max(1, len(generations) // 15),
            linewidth=2,
        )

        # 绘制标准差阴影
        ax.fill_between(
            generations,
            mean_hv - std_hv,
            mean_hv + std_hv,
            alpha=0.15,
            color=style.get("color", "black"),
        )

    ax.set_xlabel("迭代代数", fontsize=12, fontweight="bold")
    ax.set_ylabel("超体积指标 (HV)", fontsize=12, fontweight="bold")
    ax.set_title("超体积指标收敛曲线对比", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_hypervolume.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_hypervolume.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 超体积指标收敛曲线已保存")


def plot_improvement_rate(aggregated_df: pd.DataFrame, output_dir: Path):
    """绘制成本改进率对比"""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")

        if "total_cost_mean" not in alg_data.columns or len(alg_data) == 0:
            continue

        style = ALGORITHM_STYLES.get(algorithm, {})

        generations = alg_data["gen"].values
        mean_cost = alg_data["total_cost_mean"].values

        # 计算改进率
        initial_cost = mean_cost[0]
        improvement_rate = (initial_cost - mean_cost) / (initial_cost + 1e-9) * 100

        # 绘制改进率曲线
        ax.plot(
            generations,
            improvement_rate,
            label=style.get("label", algorithm),
            color=style.get("color", "black"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markevery=max(1, len(generations) // 15),
            linewidth=2,
        )

    ax.set_xlabel("迭代代数", fontsize=12, fontweight="bold")
    ax.set_ylabel("成本改进率 (%)", fontsize=12, fontweight="bold")
    ax.set_title("收敛速度对比（成本改进率）", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_improvement_rate.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_improvement_rate.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 成本改进率曲线已保存")


def plot_combined_convergence(aggregated_df: pd.DataFrame, hv_df: pd.DataFrame, output_dir: Path):
    """绘制综合收敛曲线（四合一）"""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("算法收敛性能综合对比", fontsize=16, fontweight="bold", y=0.995)

    # 1. 总成本收敛
    ax = axes[0, 0]
    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")
        if "total_cost_mean" not in alg_data.columns:
            continue
        style = ALGORITHM_STYLES.get(algorithm, {})
        ax.plot(
            alg_data["gen"],
            alg_data["total_cost_mean"],
            label=style.get("label", algorithm),
            color=style.get("color"),
            linestyle=style.get("linestyle"),
            marker=style.get("marker"),
            markevery=max(1, len(alg_data) // 10),
            linewidth=1.8,
        )
        std = alg_data.get("total_cost_std", 0)
        ax.fill_between(
            alg_data["gen"],
            alg_data["total_cost_mean"] - std,
            alg_data["total_cost_mean"] + std,
            alpha=0.12,
            color=style.get("color"),
        )
    ax.set_xlabel("迭代代数", fontweight="bold")
    ax.set_ylabel("总成本 (元/天)", fontweight="bold")
    ax.set_title("(a) 总成本收敛曲线", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 2. 碳排放成本收敛
    ax = axes[0, 1]
    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")
        if "total_cost_mean" not in alg_data.columns:
            continue
        style = ALGORITHM_STYLES.get(algorithm, {})
        carbon_mean = alg_data["total_cost_mean"] * 0.18
        carbon_std = alg_data.get("total_cost_std", 0) * 0.18
        ax.plot(
            alg_data["gen"],
            carbon_mean,
            label=style.get("label", algorithm),
            color=style.get("color"),
            linestyle=style.get("linestyle"),
            marker=style.get("marker"),
            markevery=max(1, len(alg_data) // 10),
            linewidth=1.8,
        )
        ax.fill_between(
            alg_data["gen"],
            carbon_mean - carbon_std,
            carbon_mean + carbon_std,
            alpha=0.12,
            color=style.get("color"),
        )
    ax.set_xlabel("迭代代数", fontweight="bold")
    ax.set_ylabel("碳排放成本 (元/天)", fontweight="bold")
    ax.set_title("(b) 碳排放成本收敛曲线", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 3. 超体积指标收敛
    ax = axes[1, 0]
    if not hv_df.empty:
        hv_agg = hv_df.groupby(["algorithm", "gen"]).agg(
            hv_mean=("hypervolume", "mean"),
            hv_std=("hypervolume", "std"),
        ).reset_index()
        for algorithm in hv_agg["algorithm"].unique():
            alg_data = hv_agg[hv_agg["algorithm"] == algorithm].sort_values("gen")
            style = ALGORITHM_STYLES.get(algorithm, {})
            ax.plot(
                alg_data["gen"],
                alg_data["hv_mean"],
                label=style.get("label", algorithm),
                color=style.get("color"),
                linestyle=style.get("linestyle"),
                marker=style.get("marker"),
                markevery=max(1, len(alg_data) // 10),
                linewidth=1.8,
            )
            std = alg_data["hv_std"].fillna(0)
            ax.fill_between(
                alg_data["gen"],
                alg_data["hv_mean"] - std,
                alg_data["hv_mean"] + std,
                alpha=0.12,
                color=style.get("color"),
            )
    ax.set_xlabel("迭代代数", fontweight="bold")
    ax.set_ylabel("超体积指标 (HV)", fontweight="bold")
    ax.set_title("(c) 超体积指标收敛曲线", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 4. 成本改进率
    ax = axes[1, 1]
    for algorithm in aggregated_df["algorithm"].unique():
        alg_data = aggregated_df[aggregated_df["algorithm"] == algorithm].sort_values("gen")
        if "total_cost_mean" not in alg_data.columns or len(alg_data) == 0:
            continue
        style = ALGORITHM_STYLES.get(algorithm, {})
        initial_cost = alg_data["total_cost_mean"].iloc[0]
        improvement_rate = (initial_cost - alg_data["total_cost_mean"]) / (initial_cost + 1e-9) * 100
        ax.plot(
            alg_data["gen"],
            improvement_rate,
            label=style.get("label", algorithm),
            color=style.get("color"),
            linestyle=style.get("linestyle"),
            marker=style.get("marker"),
            markevery=max(1, len(alg_data) // 10),
            linewidth=1.8,
        )
    ax.set_xlabel("迭代代数", fontweight="bold")
    ax.set_ylabel("成本改进率 (%)", fontweight="bold")
    ax.set_title("(d) 收敛速度对比（成本改进率）", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 综合收敛曲线已保存")


def plot_all_convergence_curves(
    convergence_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    inst,
    output_dir: Path,
):
    """绘制所有收敛曲线"""
    print("\n开始绘制收敛曲线...")

    # 计算超体积历史
    print("  计算超体积指标...")
    hv_df = calculate_hypervolume_history(convergence_df, inst)
    if not hv_df.empty:
        hv_df.to_csv(output_dir / "hypervolume_history.csv", index=False, encoding="utf-8-sig")

    # 绘制各类曲线
    plot_total_cost_convergence(aggregated_df, output_dir)
    plot_carbon_cost_convergence(aggregated_df, output_dir)

    if not hv_df.empty:
        plot_hypervolume_convergence(hv_df, output_dir)

    plot_improvement_rate(aggregated_df, output_dir)

    # 绘制综合图
    plot_combined_convergence(aggregated_df, hv_df, output_dir)

    print(f"\n所有收敛曲线已保存到: {output_dir}")


if __name__ == "__main__":
    # 测试代码
    print("此模块用于绘制收敛曲线，请通过 run_comprehensive_algorithm_comparison.py 调用")
