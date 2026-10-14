from __future__ import annotations

from pathlib import Path
import json
import shutil
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, PercentFormatter


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
PROCESS_DIR = OUTPUT_DIR / "process_data"
FIG_DIR = OUTPUT_DIR

DATA_PATHS = {
    "weight_scenarios": PROJECT_ROOT / "results" / "5.20" / "weight_scenario_topsis_final_solutions.csv",
    "pareto": PROJECT_ROOT / "results" / "chapter5_q11_results" / "representative_solution" / "representative_pareto_points.csv",
    "baseline": PROJECT_ROOT / "results" / "chapter5_q11_results" / "road_baseline_solution_comparison.csv",
    "algorithm_summary": PROJECT_ROOT / "results" / "chapter5_algorithm_comparison" / "algorithm_performance_summary.csv",
    "ablation_summary": PROJECT_ROOT / "results" / "chapter5_ablation_experiment_5.16" / "results" / "ablation_summary.csv",
    "sensitivity": PROJECT_ROOT / "results" / "chapter5_q11_results" / "q11_sensitivity_summary.csv",
    "station_load": PROJECT_ROOT / "results" / "chapter5_q11_results" / "representative_solution" / "qvns_nsga2_station_load.csv",
    "line_load": PROJECT_ROOT / "results" / "chapter5_q11_results" / "representative_solution" / "qvns_nsga2_line_load.csv",
    "history": PROJECT_ROOT / "results" / "chapter5_q11_results" / "representative_solution" / "qvns_nsga2_final_history.csv",
    "compromise_map": PROJECT_ROOT / "results" / "chapter5_q11_results" / "representative_solution_figures" / "fig_compromise_map.png",
}

ECHARTS_REFERENCE = {
    "source": "https://echarts.org.cn/examples/en/index.html",
    "screenshot": str(OUTPUT_DIR / "echarts_examples_reference.png"),
    "ocr_keywords": [
        "Line", "Bar", "Pie", "Scatter", "Graph", "Lines", "Heatmap",
        "Stacked Line", "Basic Line Chart", "Gradient Stacked Area Chart",
    ],
    "style_observation": {
        "font_family": '"Open Sans", "PingFang SC", Helvetica, Arial, sans-serif',
        "background": "#FFFFFF",
        "text": "#333333",
        "axis_label": "#6E7079",
        "deep_text": "#081642",
        "grid": "#E7E7E7",
        "accent": "#F72C5B",
    },
}

# ECharts default-like palette, slightly muted for thesis printing.
PALETTE = {
    "blue": "#5470C6",
    "green": "#91CC75",
    "yellow": "#FAC858",
    "red": "#EE6666",
    "cyan": "#73C0DE",
    "purple": "#9A60B4",
    "orange": "#FC8452",
    "gray": "#6E7079",
    "deep": "#081642",
}


def pick_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    return "DejaVu Sans"


FONT = pick_font()
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D8DCE6",
    "axes.labelcolor": "#333333",
    "xtick.color": "#6E7079",
    "ytick.color": "#6E7079",
    "text.color": "#333333",
    "legend.frameon": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing input for {name}: {path}")
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(PROCESS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_excel(PROCESS_DIR / f"{name}.xlsx", index=False)
    except Exception:
        pass


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="#E7E7E7", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#D8DCE6")
        ax.spines[side].set_linewidth(0.8)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def data_health_report(datasets: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# 第五章可视化数据健康检查",
        "",
        f"- 输出目录：`{OUTPUT_DIR}`",
        f"- 绘图字体：`{FONT}`",
        f"- ECharts 参考页：`{ECHARTS_REFERENCE['source']}`",
        "- 参考风格：白底、浅灰网格、顶部图例、蓝/绿/黄/红/青色分类配色、坐标轴文字弱化。",
        "",
    ]
    for name, df in datasets.items():
        lines.append(f"## {name}")
        lines.append(f"- 维度：{df.shape[0]} 行 × {df.shape[1]} 列")
        lines.append(f"- 重复行：{int(df.duplicated().sum())}")
        missing = df.isna().sum()
        miss = missing[missing > 0]
        if len(miss):
            lines.append("- 缺失列：" + "；".join([f"{k}={int(v)}" for k, v in miss.items()]))
        else:
            lines.append("- 缺失列：无")
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            desc = num.describe().T[["min", "max", "mean"]].round(6)
            save_table(desc.reset_index().rename(columns={"index": "column"}), f"health_{name}_numeric_summary")
            lines.append(f"- 数值字段描述已输出：`process_data/health_{name}_numeric_summary.csv`")
        lines.append("")
    (OUTPUT_DIR / "00_data_health.md").write_text("\n".join(lines), encoding="utf-8")


def fig_01_pareto_preference(pareto: pd.DataFrame, weights: pd.DataFrame) -> None:
    df = pareto.drop_duplicates(subset=["Z1_total_cost", "Z2_metro_service_ratio"]).copy()
    w = weights.copy()
    save_table(df[["solution_id", "Z1_total_cost", "Z2_metro_service_ratio", "num_open", "metro_volume", "direct_volume"]], "fig01_pareto_points")
    save_table(w[["权重情景", "solution_id", "TOPSIS贴近度", "Z1_total_cost", "Z2_metro_service_ratio", "num_open", "metro_volume", "direct_volume"]], "fig01_weight_scenario_points")

    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    ax.scatter(df["Z1_total_cost"] / 10000, df["Z2_metro_service_ratio"] * 100,
               s=28, color=PALETTE["blue"], alpha=0.28, edgecolor="none", label="Pareto 解")
    colors = [PALETTE["green"], PALETTE["yellow"], PALETTE["red"]]
    markers = ["o", "D", "s"]
    for (_, row), c, m in zip(w.iterrows(), colors, markers):
        ax.scatter(row["Z1_total_cost"] / 10000, row["Z2_metro_service_ratio"] * 100,
                   s=120, color=c, marker=m, edgecolor="#333333", linewidth=0.8, zorder=5,
                   label=row["权重情景"])
        ax.annotate(f"{row['权重情景']}\n解{int(row['solution_id'])}",
                    (row["Z1_total_cost"] / 10000, row["Z2_metro_service_ratio"] * 100),
                    xytext=(8, 8), textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D8DCE6", alpha=0.92))
    ax.set_xlabel("综合成本 Z1（万元/日）")
    ax.set_ylabel("地铁服务比例 Z2（%）")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
    style_axes(ax)
    ax.legend(loc="lower right", ncol=1)
    save_fig(fig, "fig_01_pareto_weight_scenario_selection")


def fig_02_weight_metrics(weights: pd.DataFrame) -> None:
    df = weights.copy()
    order = ["成本偏好", "均衡偏好/默认方案", "服务偏好"]
    df["权重情景"] = pd.Categorical(df["权重情景"], categories=order, ordered=True)
    df = df.sort_values("权重情景")
    save_table(df[["权重情景", "Z1_total_cost", "Z2_metro_service_ratio", "metro_volume", "direct_volume", "num_open", "used_train_trips_total"]], "fig02_weight_metrics")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9))
    labels = [str(x) for x in df["权重情景"]]
    x = np.arange(len(df))
    bar_colors = [PALETTE["green"], PALETTE["yellow"], PALETTE["red"]]

    panels = [
        ("Z1_total_cost", "综合成本（万元/日）", lambda s: s / 10000, "{:.2f}"),
        ("Z2_metro_service_ratio", "地铁服务比例（%）", lambda s: s * 100, "{:.1f}"),
        ("direct_volume", "公路直送量（万 SLU/日）", lambda s: s / 10000, "{:.2f}"),
        ("metro_volume", "地铁联运量（万 SLU/日）", lambda s: s / 10000, "{:.2f}"),
    ]
    for ax, (col, ylabel, trans, fmt) in zip(axes.ravel(), panels):
        vals = trans(df[col].astype(float))
        ax.bar(x, vals, color=bar_colors, width=0.58, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(ylabel)
        style_axes(ax)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.025, fmt.format(v), ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.subplots_adjust(wspace=0.30, hspace=0.36)
    save_fig(fig, "fig_02_weight_scenario_key_metrics")


def fig_03_cost_composition(weights: pd.DataFrame) -> None:
    df = weights.copy()
    components = [
        ("fixed_cost_total", "固定启用成本", PALETTE["gray"]),
        ("road_cost", "公路运输成本", PALETTE["blue"]),
        ("metro_cost", "地铁干线成本", PALETTE["green"]),
        ("feeder_cost", "末端接驳成本", PALETTE["cyan"]),
        ("transfer_cost", "换乘作业成本", PALETTE["yellow"]),
        ("inout_cost", "进出站作业成本", PALETTE["orange"]),
        ("train_organization_cost", "班次组织成本", PALETTE["red"]),
    ]
    cols = ["权重情景"] + [c for c, _, _ in components]
    save_table(df[cols], "fig03_cost_composition")
    labels = df["权重情景"].tolist()
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for col, lab, color in components:
        vals = df[col].fillna(0).astype(float).to_numpy() / 10000
        ax.bar(x, vals, bottom=bottom, label=lab, color=color, width=0.62, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("成本构成（万元/日）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    style_axes(ax)
    save_fig(fig, "fig_03_weight_scenario_cost_composition")


def fig_04_baseline_comparison(baseline: pd.DataFrame) -> None:
    df = baseline.copy()
    label_map = {
        "all_road": "全公路直送",
        "min_cost": "最小成本方案",
        "topsis_compromise": "TOPSIS折中方案",
        "max_service": "最大服务方案",
    }
    df["方案"] = df["scenario_en"].map(label_map).fillna(df["scenario_en"])
    save_table(df[["方案", "total_cost_yuan_day", "metro_service_ratio", "direct_volume", "total_carbon_tCO2", "total_cost_change_pct"]], "fig04_baseline_comparison")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    metrics = [
        ("total_cost_yuan_day", "综合成本（万元/日）", lambda s: s / 10000, "{:.2f}"),
        ("metro_service_ratio", "地铁服务比例（%）", lambda s: s * 100, "{:.1f}"),
        ("direct_volume", "公路直送量（万 SLU/日）", lambda s: s / 10000, "{:.2f}"),
        ("total_carbon_tCO2", "碳排放（tCO2/日）", lambda s: s, "{:.3f}"),
    ]
    colors = [PALETTE["gray"], PALETTE["green"], PALETTE["yellow"], PALETTE["red"]]
    x = np.arange(len(df))
    for ax, (col, ylabel, trans, fmt) in zip(axes.ravel(), metrics):
        vals = trans(df[col].astype(float))
        ax.bar(x, vals, width=0.58, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["方案"], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.025, fmt.format(v), ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(wspace=0.28, hspace=0.46)
    save_fig(fig, "fig_04_road_baseline_scheme_comparison")


def fig_05_station_load(station_load: pd.DataFrame) -> None:
    df = station_load.copy()
    df = df[df["是否启用"] == 1].sort_values("负载率", ascending=True)
    save_table(df[["地铁站名称", "站点类型集合", "分配量_件日", "容量_件日", "负载率"]], "fig05_station_load")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = [PALETTE["green"] if v < 0.83 else PALETTE["yellow"] if v < 0.85 else PALETTE["red"] for v in df["负载率"]]
    ax.barh(df["地铁站名称"], df["负载率"] * 100, color=colors, edgecolor="white", height=0.56)
    ax.axvline(85, color=PALETTE["red"], linestyle="--", linewidth=1.2, label="最高有效负载率 85%")
    ax.set_xlabel("负载率（%）")
    ax.set_ylabel("启用地铁物流配送转运点")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    style_axes(ax, grid_axis="x")
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(row["负载率"] * 100 + 0.5, y, f"{row['分配量_件日']:.0f}/{row['容量_件日']:.0f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 92)
    ax.legend(loc="lower right")
    save_fig(fig, "fig_05_compromise_station_load")


def fig_06_line_trip_heatmap(line_load: pd.DataFrame) -> None:
    df = line_load.copy()
    pivot = df.pivot_table(index="线路", columns="班次", values="班次负载率", aggfunc="mean").fillna(0)
    # Natural line order by leading number.
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: int(str(x).replace("号线", "")))]
    save_table(pivot.reset_index(), "fig06_line_trip_load_matrix")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(pivot.values * 100, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "echarts_load", ["#F5F7FA", "#73C0DE", "#FAC858", "#EE6666"]
    ), vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"班次{c}" for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] * 100
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 65 else "#333333")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#D8DCE6")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("班次负载率（%）")
    ax.set_xlabel("可用平峰期班次")
    ax.set_ylabel("地铁线路")
    save_fig(fig, "fig_06_line_trip_capacity_heatmap")


def fig_07_algorithm_performance(alg: pd.DataFrame) -> None:
    df = alg.copy()
    save_table(df, "fig07_algorithm_performance")
    metrics = [
        ("HV_mean", "HV（越大越优）"),
        ("IGD_mean", "IGD（越小越优）"),
        ("Spacing_mean", "Spacing（越小越优）"),
        ("runtime_seconds_mean", "运行时间（s）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    x = np.arange(len(df))
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["red"]]
    for ax, (col, ylabel) in zip(axes.ravel(), metrics):
        err_col = col.replace("_mean", "_std")
        err = df[err_col] if err_col in df.columns else None
        ax.bar(x, df[col], yerr=err, capsize=3, color=colors, width=0.58, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(df["algorithm_label"], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ymax = (df[col] + (err if err is not None else 0)).max()
        ax.set_ylim(0, ymax * 1.18)
    fig.subplots_adjust(wspace=0.32, hspace=0.46)
    save_fig(fig, "fig_07_algorithm_performance_metrics")


def fig_08_ablation(ablation: pd.DataFrame) -> None:
    df = ablation.copy()
    df["algorithm_display"] = df["algorithm"].replace({"QVNS-Q11C": "QVNS-NSGA-II"})
    save_table(df, "fig08_ablation_metrics")
    metrics = [
        ("HV_mean", "HV（越大越优）"),
        ("IGD_mean", "IGD（越小越优）"),
        ("Spread_mean", "Spread"),
        ("runtime_seconds_mean", "运行时间（s）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    x = np.arange(len(df))
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["gray"]]
    for ax, (col, ylabel) in zip(axes.ravel(), metrics):
        err_col = col.replace("_mean", "_std")
        err = df[err_col] if err_col in df.columns else None
        ax.bar(x, df[col], yerr=err, capsize=3, color=colors, width=0.58, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["algorithm_display"], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ymax = (df[col] + (err if err is not None else 0)).max()
        ax.set_ylim(0, ymax * 1.18)
    fig.subplots_adjust(wspace=0.32, hspace=0.46)
    save_fig(fig, "fig_08_ablation_mechanism_comparison")


def fig_09_sensitivity(sens: pd.DataFrame) -> None:
    df = sens.copy()
    groups = [
        ("demand_scale", "需求规模情景", ["low", "base", "high"], ["低需求", "基准", "高需求"]),
        ("service_radius", "末端服务半径", ["r3", "r4", "r5"], ["3 km", "4 km", "5 km"]),
        ("source_share", "货源比例 L1:L2", ["s5_5", "s6_4", "s7_3", "s8_2", "s9_1"], ["5:5", "6:4", "7:3", "8:2", "9:1"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.2))
    export_rows = []
    for r, (gtype, title, order, labels) in enumerate(groups):
        sub = df[df["group_type"] == gtype].copy()
        sub["case_key"] = pd.Categorical(sub["case_key"], categories=order, ordered=True)
        sub = sub.sort_values("case_key")
        export_rows.append(sub[["group_type", "case_key", "topsis_cost", "topsis_service_ratio", "topsis_open_count", "topsis_avg_feeder_distance", "topsis_open_stations"]])
        x = np.arange(len(sub))
        ax1, ax2 = axes[r, 0], axes[r, 1]
        ax1.plot(x, sub["topsis_cost"] / 10000, marker="o", color=PALETTE["blue"], linewidth=2)
        ax1.set_title(title, loc="left", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("TOPSIS 成本（万元/日）")
        style_axes(ax1)
        ax2.plot(x, sub["topsis_service_ratio"] * 100, marker="D", color=PALETTE["green"], linewidth=2, label="服务比例")
        ax2b = ax2.twinx()
        ax2b.bar(x, sub["topsis_open_count"], color=PALETTE["yellow"], alpha=0.45, width=0.5, label="启用转运点数")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("地铁服务比例（%）")
        ax2b.set_ylabel("启用转运点数（个）")
        style_axes(ax2)
        ax2b.grid(False)
        for side in ["left", "right", "top", "bottom"]:
            ax2b.spines[side].set_visible(True)
            ax2b.spines[side].set_color("#D8DCE6")
    fig.subplots_adjust(wspace=0.36, hspace=0.42)
    save_table(pd.concat(export_rows, ignore_index=True), "fig09_sensitivity_summary")
    save_fig(fig, "fig_09_sensitivity_cost_service_response")


def fig_10_convergence(history: pd.DataFrame) -> None:
    df = history.copy()
    save_table(df[["gen", "best_cost", "best_service_ratio", "pareto_size", "archive_size", "epsilon"]], "fig10_qvns_training_history")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    axes[0, 0].plot(df["gen"], df["best_cost"] / 10000, color=PALETTE["blue"], linewidth=1.8)
    axes[0, 0].set_ylabel("最优成本（万元/日）")
    axes[0, 0].set_xlabel("迭代代数")
    axes[0, 1].plot(df["gen"], df["best_service_ratio"] * 100, color=PALETTE["green"], linewidth=1.8)
    axes[0, 1].set_ylabel("最优服务比例（%）")
    axes[0, 1].set_xlabel("迭代代数")
    axes[1, 0].plot(df["gen"], df["pareto_size"], color=PALETTE["yellow"], linewidth=1.8, label="Pareto规模")
    axes[1, 0].plot(df["gen"], df["archive_size"], color=PALETTE["red"], linewidth=1.5, linestyle="--", label="档案规模")
    axes[1, 0].set_ylabel("解集规模")
    axes[1, 0].set_xlabel("迭代代数")
    axes[1, 0].legend(loc="lower right")
    axes[1, 1].plot(df["gen"], df["epsilon"], color=PALETTE["purple"], linewidth=1.8)
    axes[1, 1].set_ylabel("探索率 epsilon")
    axes[1, 1].set_xlabel("迭代代数")
    for ax in axes.ravel():
        style_axes(ax)
    fig.subplots_adjust(wspace=0.32, hspace=0.38)
    save_fig(fig, "fig_10_qvns_training_process")


def copy_compromise_map() -> None:
    src = DATA_PATHS["compromise_map"]
    if src.exists():
        dst = OUTPUT_DIR / "fig_11_compromise_spatial_layout_reference.png"
        shutil.copyfile(src, dst)


def write_style_notes() -> None:
    notes = f"""# ECharts 参考风格与本次图件规范

## 参考来源

- ECharts 示例页：{ECHARTS_REFERENCE['source']}
- 页面截图：`echarts_examples_reference.png`
- OCR 识别到的图表类型关键词：{', '.join(ECHARTS_REFERENCE['ocr_keywords'])}

## 提取到的视觉特征

- 页面整体采用白色背景、弱灰坐标轴和网格线，适合论文插图打印。
- 字体层级接近 `"Open Sans", "PingFang SC", Helvetica, Arial, sans-serif`；本机绘图字体选择为 `{FONT}`。
- 分类配色参考 ECharts 常用蓝、绿、黄、红、青、紫色系，并降低装饰性饱和度。
- 图例统一靠上或靠右，坐标轴保留四边框，避免网页仪表盘式重装饰。

## 论文图件使用建议

- `fig_01` 和 `fig_02` 用于 5.3.3 权重情景和 Pareto 方案选择。
- `fig_03` 用于解释三类权重方案的综合成本构成。
- `fig_04` 用于全公路基准对比。
- `fig_05` 与 `fig_06` 用于折中方案运营可行性诊断。
- `fig_07` 与 `fig_08` 用于算法对比和消融实验。
- `fig_09` 用于 5.4 敏感性分析。
- `fig_10` 用于算法迭代过程说明，可根据篇幅放正文或附录。
"""
    (OUTPUT_DIR / "echarts_style_notes.md").write_text(notes, encoding="utf-8")
    (OUTPUT_DIR / "echarts_style_reference.json").write_text(
        json.dumps(ECHARTS_REFERENCE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    datasets = {name: read_csv(name) for name in [
        "weight_scenarios", "pareto", "baseline", "algorithm_summary",
        "ablation_summary", "sensitivity", "station_load", "line_load", "history"
    ]}
    data_health_report(datasets)
    fig_01_pareto_preference(datasets["pareto"], datasets["weight_scenarios"])
    fig_02_weight_metrics(datasets["weight_scenarios"])
    fig_03_cost_composition(datasets["weight_scenarios"])
    fig_04_baseline_comparison(datasets["baseline"])
    fig_05_station_load(datasets["station_load"])
    fig_06_line_trip_heatmap(datasets["line_load"])
    fig_07_algorithm_performance(datasets["algorithm_summary"])
    fig_08_ablation(datasets["ablation_summary"])
    fig_09_sensitivity(datasets["sensitivity"])
    fig_10_convergence(datasets["history"])
    copy_compromise_map()
    write_style_notes()


if __name__ == "__main__":
    main()
