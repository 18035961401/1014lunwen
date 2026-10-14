# -*- coding: utf-8 -*-
"""Generate publication-ready Chapter 5 algorithm-comparison figures.

This script creates a separate ``figures_publication`` folder so the original
figures remain untouched.  It uses editable SVG text and Chinese font fallbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from chinese_font_utils import configure_matplotlib_cjk  # noqa: E402

RESULT_DIR = PROJECT_ROOT / "results" / "chapter5_algorithm_comparison"
PUB_DIR = RESULT_DIR / "figures_publication"

ALGORITHM_ORDER = ["QVNS-NSGA-II", "NSGA-II", "MOPSO", "MOEA/D"]
PALETTE = {
    "QVNS-NSGA-II": "#B23A2F",
    "NSGA-II": "#3E7CB1",
    "MOPSO": "#4C956C",
    "MOEA/D": "#7B6BA8",
}


def setup_style() -> None:
    configure_matplotlib_cjk(plt)
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PUB_DIR / f"{filename}.png", bbox_inches="tight")
    fig.savefig(PUB_DIR / f"{filename}.svg", bbox_inches="tight")
    fig.savefig(PUB_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULT_DIR / "algorithm_comparison_summary.csv", encoding="utf-8-sig")
    runs = pd.read_csv(RESULT_DIR / "algorithm_comparison_run_metrics.csv", encoding="utf-8-sig")
    points = pd.read_csv(RESULT_DIR / "algorithm_comparison_pareto_points.csv", encoding="utf-8-sig")
    for df in (summary, runs, points):
        df["algorithm"] = pd.Categorical(df["algorithm"], ALGORITHM_ORDER, ordered=True)
    return summary.sort_values("algorithm"), runs.sort_values("algorithm"), points.sort_values("algorithm")


def nondominated_cost_service(df: pd.DataFrame) -> pd.DataFrame:
    front = (
        df[["cost_wan", "service_pct"]]
        .round({"cost_wan": 3, "service_pct": 3})
        .drop_duplicates()
        .sort_values(["cost_wan", "service_pct"])
        .reset_index(drop=True)
    )
    keep = []
    best_service = -np.inf
    for idx, row in front.iterrows():
        service = float(row["service_pct"])
        if service >= best_service - 1e-9:
            keep.append(idx)
            best_service = max(best_service, service)
    return front.loc[keep].reset_index(drop=True)


def figure_pareto_front(summary: pd.DataFrame, points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for algorithm in ALGORITHM_ORDER:
        subset = points[points["algorithm"] == algorithm].copy()
        subset["cost_wan"] = subset["Z1_total_cost"] / 10000.0
        subset["service_pct"] = subset["Z2_metro_service_ratio"] * 100.0
        unique = subset[["cost_wan", "service_pct"]].round(3).drop_duplicates()
        ax.scatter(
            unique["cost_wan"], unique["service_pct"],
            s=9 if algorithm != "QVNS-NSGA-II" else 12,
            color=PALETTE[algorithm], alpha=0.16 if algorithm != "QVNS-NSGA-II" else 0.28,
            edgecolors="none",
        )
        front = nondominated_cost_service(subset)
        ax.plot(
            front["cost_wan"], front["service_pct"],
            color=PALETTE[algorithm], lw=2.2 if algorithm == "QVNS-NSGA-II" else 1.45,
            alpha=0.95 if algorithm == "QVNS-NSGA-II" else 0.74,
        )
        ax.scatter(
            front["cost_wan"], front["service_pct"],
            s=24 if algorithm == "QVNS-NSGA-II" else 16,
            color=PALETTE[algorithm], edgecolors="white", linewidths=0.45, zorder=3,
        )
    handles = [Line2D([0], [0], color=PALETTE[a], lw=2.2 if a == "QVNS-NSGA-II" else 1.5, marker="o", markersize=4, label=a) for a in ALGORITHM_ORDER]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4, handlelength=1.6, columnspacing=1.1)
    ax.set_xlabel("综合成本（万元/日）")
    ax.set_ylabel("地铁服务比例（%）")
    ax.set_title("a  算法 Pareto 前沿对比", loc="left", fontweight="bold")
    ax.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.55, alpha=0.55)
    ax.set_xlim(19.3, 25.2)
    ax.set_ylim(-1.0, 55.0)
    ax.annotate("低成本", xy=(19.65, 52.0), xytext=(21.0, 52.0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555"), color="#555555")
    ax.annotate("高服务", xy=(19.55, 52.5), xytext=(19.55, 45.0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555"), color="#555555", rotation=90, va="center")
    fig.subplots_adjust(bottom=0.24)
    save(fig, "fig5_pub_a_pareto_front")


def figure_quality_metrics(summary: pd.DataFrame) -> None:
    metrics = [
        ("HV_mean", "HV", "越大越好", False),
        ("IGD_mean", "IGD", "越小越好", False),
        ("Spacing_mean", "Spacing", "越小越好", False),
        ("Spread_mean", "Spread", "越小越好", False),
        ("unique_points_mean", "唯一解数", "越大越好", False),
        ("runtime_seconds_mean", "运行时间/s", "越小越好", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.3))
    axes = axes.ravel()
    for ax, (column, title, note, _) in zip(axes, metrics):
        labels = summary["algorithm"].astype(str).tolist()
        values = summary[column].astype(float).to_numpy()
        colors = [PALETTE[label] for label in labels]
        bars = ax.bar(labels, values, color=colors, alpha=0.86, width=0.68)
        ax.set_title(f"{title}（{note}）", pad=5)
        ax.tick_params(axis="x", rotation=24)
        ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.5, alpha=0.45)
        ymax = values.max() if len(values) else 1
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
        for bar, value in zip(bars, values):
            if column in {"HV_mean", "IGD_mean", "Spacing_mean", "Spread_mean"}:
                text = f"{value:.3f}" if value >= 0.01 else f"{value:.4f}"
            elif column == "runtime_seconds_mean":
                text = f"{value:.0f}"
            else:
                text = f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025, text, ha="center", va="bottom", fontsize=7.6)
    fig.suptitle("b  算法质量与效率指标对比", x=0.02, y=1.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_pub_b_quality_metrics")


def figure_quality_boxplots(runs: pd.DataFrame) -> None:
    metrics = [
        ("HV", "HV"),
        ("IGD", "IGD"),
        ("Spacing", "Spacing"),
        ("Spread", "Spread"),
        ("unique_points", "唯一解数"),
        ("runtime_seconds", "运行时间/s"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.3))
    axes = axes.ravel()
    for ax, (column, title) in zip(axes, metrics):
        data = [runs.loc[runs["algorithm"] == a, column].dropna().to_numpy() for a in ALGORITHM_ORDER]
        box = ax.boxplot(data, tick_labels=ALGORITHM_ORDER, patch_artist=True, showmeans=True, widths=0.55)
        for patch, algorithm in zip(box["boxes"], ALGORITHM_ORDER):
            patch.set_facecolor(PALETTE[algorithm])
            patch.set_alpha(0.52)
            patch.set_linewidth(0.8)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.0)
        ax.set_title(title, pad=5)
        ax.tick_params(axis="x", rotation=24)
        ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.suptitle("c  多次运行稳定性分布", x=0.02, y=1.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_pub_c_metric_boxplots")


def figure_business_objectives(summary: pd.DataFrame) -> None:
    labels = summary["algorithm"].astype(str).tolist()
    x = np.arange(len(labels))
    cost = summary["min_cost_mean"].astype(float).to_numpy() / 10000.0
    service = summary["max_service_ratio_mean"].astype(float).to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    bars = axes[0].bar(x, cost, color=[PALETTE[label] for label in labels], alpha=0.86, width=0.65)
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("最低成本（万元/日）")
    axes[0].set_title("d  最低综合成本", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.5, alpha=0.45)
    for bar, value in zip(bars, cost):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", fontsize=8)

    bars = axes[1].bar(x, service, color=[PALETTE[label] for label in labels], alpha=0.86, width=0.65)
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("最大地铁服务比例（%）")
    axes[1].set_title("e  最大服务比例", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.5, alpha=0.45)
    for bar, value in zip(bars, service):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_pub_d_business_objectives")


def write_readme() -> None:
    text = """# Chapter 5 publication-ready figures

This folder contains optimized figures generated from `chapter5_algorithm_comparison` CSV files.

- `fig5_pub_a_pareto_front.*`: Pareto front comparison with background feasible points.
- `fig5_pub_b_quality_metrics.*`: HV, IGD, Spacing, Spread, unique Pareto points, and runtime.
- `fig5_pub_c_metric_boxplots.*`: multi-run stability distributions.
- `fig5_pub_d_business_objectives.*`: minimum cost and maximum metro service ratio.

Design notes: Chinese labels use Microsoft YaHei/SimHei fallbacks; SVG text remains editable; PNG is exported at 600 dpi.
"""
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    (PUB_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    setup_style()
    summary, runs, points = read_data()
    figure_pareto_front(summary, points)
    figure_quality_metrics(summary)
    figure_quality_boxplots(runs)
    figure_business_objectives(summary)
    write_readme()
    print(f"论文版图件已生成：{PUB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
