# -*- coding: utf-8 -*-
"""Regenerate Chapter 5 algorithm-comparison figures with Chinese labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "chapter5_algorithm_comparison"
FIG_DIR = RESULT_DIR / "figures"


PALETTE = {
    "QVNS-NSGA-II": "#c0392b",
    "NSGA-II": "#2c7fb8",
    "MOPSO": "#31a354",
    "MOEA/D": "#756bb1",
}
ALGORITHM_ORDER = ["QVNS-NSGA-II", "NSGA-II", "MOPSO", "MOEA/D"]


def setup_chinese_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 420,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULT_DIR / "algorithm_comparison_summary.csv", encoding="utf-8-sig")
    runs = pd.read_csv(RESULT_DIR / "algorithm_comparison_run_metrics.csv", encoding="utf-8-sig")
    points = pd.read_csv(RESULT_DIR / "algorithm_comparison_pareto_points.csv", encoding="utf-8-sig")
    summary["algorithm"] = pd.Categorical(summary["algorithm"], ALGORITHM_ORDER, ordered=True)
    runs["algorithm"] = pd.Categorical(runs["algorithm"], ALGORITHM_ORDER, ordered=True)
    points["algorithm"] = pd.Categorical(points["algorithm"], ALGORITHM_ORDER, ordered=True)
    return summary.sort_values("algorithm"), runs.sort_values("algorithm"), points.sort_values("algorithm")


def plot_pareto_scatter_original(points: pd.DataFrame) -> None:
    """Keep the original all-points scatter style, with fixed Chinese labels."""
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for algorithm in ALGORITHM_ORDER:
        subset = points[points["algorithm"] == algorithm]
        if subset.empty:
            continue
        ax.scatter(
            subset["Z1_total_cost"],
            subset["Z2_metro_service_ratio"] * 100,
            s=14,
            alpha=0.35,
            color=PALETTE[algorithm],
            edgecolors="none",
            label=algorithm,
        )
    ax.set_xlabel("综合成本（元/日）")
    ax.set_ylabel("地铁服务比例（%）")
    ax.set_title("典型多目标算法 Pareto 解集对比")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save_figure(fig, "fig5_algorithm_pareto_scatter")


def plot_pareto_scatter_frontier(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 6.1))
    for algorithm in ALGORITHM_ORDER:
        subset = points[points["algorithm"] == algorithm].copy()
        if subset.empty:
            continue
        subset["cost_wan"] = subset["Z1_total_cost"] / 10000.0
        subset["service_pct"] = subset["Z2_metro_service_ratio"] * 100.0
        unique_points = (
            subset[["cost_wan", "service_pct"]]
            .round({"cost_wan": 3, "service_pct": 3})
            .drop_duplicates()
            .sort_values(["cost_wan", "service_pct"])
        )
        ax.scatter(
            unique_points["cost_wan"],
            unique_points["service_pct"],
            s=13 if algorithm != "QVNS-NSGA-II" else 17,
            alpha=0.22 if algorithm != "QVNS-NSGA-II" else 0.42,
            color=PALETTE[algorithm],
            edgecolors="none",
            label=algorithm,
        )
        front = unique_points.sort_values("cost_wan")
        front = front[front["service_pct"].cummax() <= front["service_pct"] + 1e-9]
        if len(front) > 1:
            ax.plot(
                front["cost_wan"],
                front["service_pct"],
                color=PALETTE[algorithm],
                linewidth=2.4 if algorithm == "QVNS-NSGA-II" else 1.5,
                alpha=0.95 if algorithm == "QVNS-NSGA-II" else 0.72,
            )
            ax.scatter(
                front["cost_wan"],
                front["service_pct"],
                s=34 if algorithm == "QVNS-NSGA-II" else 22,
                color=PALETTE[algorithm],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.96,
            )
        best_service = unique_points.loc[unique_points["service_pct"].idxmax()]
        ax.text(
            best_service["cost_wan"] + 0.06,
            best_service["service_pct"] + 0.45,
            algorithm,
            color=PALETTE[algorithm],
            fontsize=9.5,
            weight="bold" if algorithm == "QVNS-NSGA-II" else "normal",
        )
    ax.annotate("成本更低", xy=(19.65, 52.8), xytext=(21.1, 52.8),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                color="#555555", ha="center", va="center", fontsize=10)
    ax.annotate("服务比例更高", xy=(19.55, 53.2), xytext=(19.55, 45.0),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                color="#555555", ha="center", va="center", rotation=90, fontsize=10)
    ax.set_xlabel("综合成本（万元/日）")
    ax.set_ylabel("地铁服务比例（%）")
    ax.set_title("典型多目标算法 Pareto 前沿对比", fontsize=14, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(19.3, 25.3)
    ax.set_ylim(-1.5, 55.5)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.20)
    save_figure(fig, "fig5_algorithm_pareto_scatter_frontier")


def plot_metric_bars(summary: pd.DataFrame) -> None:
    metrics = [
        ("HV_mean", "HV 指标（越大越好）", "HV"),
        ("IGD_mean", "IGD 指标（越小越好）", "IGD"),
        ("Spacing_mean", "Spacing 指标（越小越好）", "Spacing"),
        ("Spread_mean", "Spread 指标（越小越好）", "Spread"),
        ("unique_points_mean", "唯一 Pareto 解数（越大越好）", "唯一解数"),
        ("runtime_seconds_mean", "平均运行时间（s）", "运行时间（s）"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
    axes = axes.ravel()
    for ax, (column, title, ylabel) in zip(axes, metrics):
        values = summary[column].astype(float).to_numpy()
        labels = summary["algorithm"].astype(str).to_list()
        bars = ax.bar(labels, values, color=[PALETTE[label] for label in labels], alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
        for bar, value in zip(bars, values):
            fmt = "{:.3f}" if abs(value) < 10 else "{:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha="center", va="bottom", fontsize=9)
    fig.suptitle("典型多目标算法质量与效率指标对比", fontsize=15, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig5_algorithm_metric_bars")


def plot_metric_boxplots(runs: pd.DataFrame) -> None:
    metrics = [
        ("HV", "HV 分布"),
        ("IGD", "IGD 分布"),
        ("Spacing", "Spacing 分布"),
        ("Spread", "Spread 分布"),
        ("unique_points", "唯一 Pareto 解数分布"),
        ("runtime_seconds", "运行时间分布（s）"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
    axes = axes.ravel()
    for ax, (column, title) in zip(axes, metrics):
        data = [runs.loc[runs["algorithm"] == algorithm, column].dropna().to_numpy() for algorithm in ALGORITHM_ORDER]
        try:
            box = ax.boxplot(data, tick_labels=ALGORITHM_ORDER, patch_artist=True, showmeans=True)
        except TypeError:
            box = ax.boxplot(data, labels=ALGORITHM_ORDER, patch_artist=True, showmeans=True)
        for patch, algorithm in zip(box["boxes"], ALGORITHM_ORDER):
            patch.set_facecolor(PALETTE[algorithm])
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    fig.suptitle("典型多目标算法多次运行指标分布", fontsize=15, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig5_algorithm_metric_boxplots")


def plot_business_objectives(summary: pd.DataFrame) -> None:
    labels = summary["algorithm"].astype(str).to_list()
    colors = [PALETTE[label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    cost_values = summary["min_cost_mean"].astype(float).to_numpy() / 10000
    service_values = summary["max_service_ratio_mean"].astype(float).to_numpy() * 100

    bars = axes[0].bar(labels, cost_values, color=colors, alpha=0.88)
    axes[0].set_title("最低综合成本对比")
    axes[0].set_ylabel("最低成本（万元/日）")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    for bar, value in zip(bars, cost_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(labels, service_values, color=colors, alpha=0.88)
    axes[1].set_title("最大地铁服务比例对比")
    axes[1].set_ylabel("最大服务比例（%）")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    for bar, value in zip(bars, service_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("算法业务目标表现对比", fontsize=15, y=1.03)
    fig.tight_layout()
    save_figure(fig, "fig5_business_objective_bars")


def main() -> int:
    setup_chinese_font()
    summary, runs, points = read_data()
    plot_pareto_scatter_original(points)
    plot_pareto_scatter_frontier(points)
    plot_metric_bars(summary)
    plot_metric_boxplots(runs)
    plot_business_objectives(summary)
    print(f"已重新生成中文图件：{FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
