
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from chinese_font_utils import configure_matplotlib_cjk, get_cjk_font_properties  # noqa: E402
configure_matplotlib_cjk(plt)
CJK_FONT = get_cjk_font_properties()

ALGORITHM_ORDER = ["QVNS-NSGA2", "NSGA2", "MOPSO", "SPEA2"]
LABELS = {"QVNS-NSGA2": "QVNS-NSGA-II", "NSGA2": "NSGA-II", "MOPSO": "MOPSO", "SPEA2": "SPEA2"}
SLUGS = {"QVNS-NSGA2": "qvns_nsga2", "NSGA2": "nsga2", "MOPSO": "mopso", "SPEA2": "spea2"}
PALETTE = {"QVNS-NSGA2": "#B23A48", "NSGA2": "#3B73B9", "MOPSO": "#4C956C", "SPEA2": "#7E6AAD"}
LINESTYLES = {"QVNS-NSGA2": "-", "NSGA2": "--", "MOPSO": "-.", "SPEA2": ":"}
MARKERS = {"QVNS-NSGA2": "o", "NSGA2": "s", "MOPSO": "^", "SPEA2": "D"}
METRIC_DIRECTIONS = {"HV": "max", "IGD": "min", "Spacing": "min", "Spread": "min", "unique_solutions": "max", "runtime_seconds": "min"}
METRIC_CN = {
    "HV": "HV",
    "IGD": "IGD",
    "Spacing": "Spacing",
    "Spread": "Spread",
    "unique_solutions": "非重复解数",
    "runtime_seconds": "运行时间 (s)",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 600,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save_figure(fig, output_dir: Path, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")


def set_cjk_text(ax) -> None:
    ax.xaxis.label.set_fontproperties(CJK_FONT)
    ax.yaxis.label.set_fontproperties(CJK_FONT)
    ax.title.set_fontproperties(CJK_FONT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CJK_FONT)


def sorted_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].unique())
    return [alg for alg in ALGORITHM_ORDER if alg in present]


def plot_convergence_grid(agg: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("pareto_mean_total_cost", "Pareto前沿平均总成本 (万元/日)", "a", "前沿平均总成本"),
        ("pareto_mean_service_ratio", "Pareto前沿平均地铁服务比例 (%)", "b", "前沿平均地铁服务比例"),
        ("hypervolume", "超体积指标 HV", "c", "超体积收敛"),
        ("cost_improvement_rate", "成本改进率 (%)", "d", "成本改进率"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, (metric, ylabel, panel, title) in zip(axes.ravel(), metrics):
        for alg in sorted_algorithms(agg):
            data = agg[agg["algorithm"] == alg].sort_values("gen")
            y = data[f"{metric}_mean"].astype(float).to_numpy()
            if metric == "pareto_mean_total_cost": y = y / 10000.0
            if metric == "pareto_mean_service_ratio": y = y * 100.0
            x = data["gen"].astype(float).to_numpy()
            std_col = f"{metric}_std"
            y_std = data[std_col].fillna(0.0).astype(float).to_numpy() if std_col in data else np.zeros_like(y)
            if metric == "pareto_mean_total_cost": y_std = y_std / 10000.0
            if metric == "pareto_mean_service_ratio": y_std = y_std * 100.0
            ax.plot(x, y, color=PALETTE[alg], linestyle=LINESTYLES[alg], linewidth=1.7,
                    marker=MARKERS[alg], markersize=3.2, markevery=max(1, len(x)//8), label=LABELS[alg])
            ax.fill_between(x, y-y_std, y+y_std, color=PALETTE[alg], alpha=0.10, linewidth=0)
        ax.set_title(f"({panel}) {title}", loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("迭代代数")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, color="#D9D9D9", alpha=0.65)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        set_cjk_text(ax)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), prop=CJK_FONT)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, output_dir, "paper_fig_algorithm_convergence_grid")
    plt.close(fig)


def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    data = points[points["CV"].astype(float) <= 1e-9].copy()
    if data.empty: data = points.copy()
    data = data.sort_values(["Z1_total_cost_10k", "Z2_metro_service_percent"])
    kept, best_service = [], -np.inf
    for _, row in data.iterrows():
        service = float(row["Z2_metro_service_percent"])
        if service >= best_service - 1e-12:
            kept.append(row); best_service = service
    return pd.DataFrame(kept)


def plot_pareto(points: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for alg in sorted_algorithms(points):
        data = points[points["algorithm"] == alg].copy()
        ax.scatter(data["Z1_total_cost_10k"], data["Z2_metro_service_percent"], s=10, color=PALETTE[alg], alpha=0.14, edgecolors="none")
        front = pareto_front(data)
        ax.plot(front["Z1_total_cost_10k"], front["Z2_metro_service_percent"], color=PALETTE[alg], linewidth=1.9, marker=MARKERS[alg], markersize=3.5, label=LABELS[alg])
    ax.annotate("优化方向", xy=(0.08,0.90), xytext=(0.25,0.76), xycoords="axes fraction", arrowprops={"arrowstyle":"->","lw":1.2,"color":"#555555"}, fontproperties=CJK_FONT, fontsize=9)
    ax.set_title("算法 Pareto 前沿对比", fontweight="bold", pad=6)
    ax.set_xlabel("综合成本 (万元/日)")
    ax.set_ylabel("地铁服务比例 (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#D9D9D9", alpha=0.65)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False, prop=CJK_FONT)
    set_cjk_text(ax)
    fig.tight_layout(rect=[0,0.08,1,1])
    save_figure(fig, output_dir, "paper_fig_pareto_front_cost_service")
    plt.close(fig)


def plot_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["HV", "IGD", "Spacing", "Spread", "unique_solutions", "runtime_seconds"]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.8))
    ordered = summary.copy(); ordered["order"] = ordered["algorithm"].map({alg:i for i,alg in enumerate(ALGORITHM_ORDER)}); ordered = ordered.sort_values("order")
    labels = [LABELS.get(alg, alg) for alg in ordered["algorithm"]]
    colors = [PALETTE.get(alg, "#999") for alg in ordered["algorithm"]]
    for ax, metric in zip(axes.ravel(), metrics):
        means = ordered[f"{metric}_mean"].astype(float).to_numpy(); stds = ordered[f"{metric}_std"].fillna(0.0).astype(float).to_numpy()
        bars = ax.bar(np.arange(len(labels)), means, yerr=stds, color=colors, alpha=0.9, edgecolor="white", linewidth=0.7, capsize=2.5)
        direction = "↑" if METRIC_DIRECTIONS[metric] == "max" else "↓"
        ax.set_title(f"{METRIC_CN[metric]} {direction}", loc="left", fontweight="bold", pad=4)
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#D9D9D9", alpha=0.65)
        for bar, value in zip(bars, means): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=6.8)
        set_cjk_text(ax)
    fig.tight_layout(); save_figure(fig, output_dir, "paper_fig_algorithm_performance_metrics"); plt.close(fig)


def plot_dual_axis_individual(agg: pd.DataFrame, output_dir: Path) -> None:
    for alg in sorted_algorithms(agg):
        data = agg[agg["algorithm"] == alg].sort_values("gen")
        x = data["gen"].astype(float).to_numpy(); cost = data["pareto_mean_total_cost_mean"].astype(float).to_numpy()/10000.0; service = data["pareto_mean_service_ratio_mean"].astype(float).to_numpy()*100.0
        fig, ax = plt.subplots(figsize=(5.8,3.6)); ax2 = ax.twinx()
        l1 = ax.step(x, cost, where="post", color="#2F6B9A", linewidth=2.0, label="前沿平均总成本")[0]
        l2 = ax2.step(x, service, where="post", color="#C73E3A", linestyle="--", linewidth=2.0, label="前沿平均服务比例")[0]
        ax.set_title(f"{LABELS[alg]} 迭代效果", fontweight="bold", pad=6)
        ax.set_xlabel("迭代代数"); ax.set_ylabel("总成本 (万元/日)"); ax2.set_ylabel("地铁服务比例 (%)", fontproperties=CJK_FONT)
        ax.grid(True, linestyle="--", linewidth=0.5, color="#D9D9D9", alpha=0.65)
        ax.legend([l1,l2], ["前沿平均总成本", "前沿平均服务比例"], loc="best", frameon=False, prop=CJK_FONT)
        set_cjk_text(ax)
        for label in ax2.get_yticklabels(): label.set_fontproperties(CJK_FONT)
        fig.tight_layout(); save_figure(fig, output_dir, f"paper_fig_iteration_effect_{SLUGS[alg]}"); plt.close(fig)


def make_paper_tables(perf: pd.DataFrame, final_summary: pd.DataFrame, output_dir: Path) -> None:
    table = perf.copy(); table["算法"] = table["algorithm"].map(LABELS)
    paper = pd.DataFrame({
        "算法": table["算法"], "HV": table["HV_mean"].astype(float).map(lambda x:f"{x:.4f}"), "IGD": table["IGD_mean"].astype(float).map(lambda x:f"{x:.4f}"),
        "Spacing": table["Spacing_mean"].astype(float).map(lambda x:f"{x:.4f}"), "Spread": table["Spread_mean"].astype(float).map(lambda x:f"{x:.4f}"),
        "非重复解数": table["unique_solutions_mean"].astype(float).map(lambda x:f"{x:.1f}"), "运行时间(s)": table["runtime_seconds_mean"].astype(float).map(lambda x:f"{x:.2f}"),
    })
    paper.to_csv(output_dir/"paper_table_algorithm_performance.csv", index=False, encoding="utf-8-sig")
    compact = final_summary.copy(); compact["算法"] = compact["algorithm"].map(LABELS)
    cols = ["算法", "run_id", "runtime_seconds", "final_pareto_size", "pareto_mean_total_cost", "pareto_mean_service_ratio"]
    compact[cols].to_csv(output_dir/"paper_table_run_summary.csv", index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--result-dir", type=Path, required=True); args = parser.parse_args(argv)
    result_dir = args.result_dir; apply_paper_style()
    agg = pd.read_csv(result_dir/"convergence_aggregated.csv", encoding="utf-8-sig")
    points = pd.read_csv(result_dir/"pareto_cost_service_points.csv", encoding="utf-8-sig")
    perf = pd.read_csv(result_dir/"algorithm_performance_summary.csv", encoding="utf-8-sig")
    run_metrics = pd.read_csv(result_dir/"algorithm_performance_run_metrics.csv", encoding="utf-8-sig") if (result_dir/"algorithm_performance_run_metrics.csv").exists() else None
    final_summary = pd.read_csv(result_dir/"final_summary.csv", encoding="utf-8-sig")
    plot_convergence_grid(agg, result_dir); plot_pareto(points, result_dir); plot_performance(perf, result_dir); plot_dual_axis_individual(agg, result_dir); make_paper_tables(perf, final_summary, result_dir)
    if run_metrics is not None:
        perf_script = PROJECT_ROOT / "scripts" / "plot_algorithm_performance_metrics.py"
        spec = importlib.util.spec_from_file_location("plot_algorithm_performance_metrics", perf_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.plot_stability_distribution(run_metrics, result_dir, prefix="paper_fig_algorithm")
    print(f"Paper-ready figures and tables saved to: {result_dir}"); return 0
if __name__ == "__main__": raise SystemExit(main())
