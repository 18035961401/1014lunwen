from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from matplotlib.lines import Line2D
from shapely.geometry import MultiPoint


ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "docs" / "同步更新输出_5.17"
RESULT_DIR = ROOT / "results" / "chapter5_q11_results"
ABLATION_DIR = ROOT / "results" / "chapter5_ablation_experiment_5.16" / "results"
OUT_DIR = DOC_DIR / "nature_figures_q11_runs10_v2"


PALETTE = {
    "navy": "#2F4A6D",
    "blue": "#5B8DB8",
    "sky": "#9CCAE2",
    "teal": "#3E8F8C",
    "green": "#6DAA6B",
    "orange": "#D99B5A",
    "red": "#C86B5F",
    "purple": "#7E77A8",
    "grey": "#7D8790",
    "light_grey": "#E8ECEF",
    "dark": "#2B2B2B",
}

ACTION_CN = {
    "standard-conservative": "保守搜索",
    "standard-baseline": "基准搜索",
    "standard-diversify": "多样化搜索",
    "service-ratio-oriented": "服务导向",
    "cost-oriented": "成本导向",
}

ALG_CN = {
    "NSGA-II": "NSGA-II",
    "Q-learning-only": "仅Q学习",
    "VNS-only": "仅VNS",
    "QVNS-Q11C": "QVNS-Q11C",
}

FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def setup_style() -> None:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["dark"],
    )


def extract_critic_weights() -> pd.DataFrame:
    doc = Document(DOC_DIR / "第五章_同步更新版.docx")
    table = doc.tables[2]
    rows = []
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        if len(cells) >= 2 and cells[0]:
            rows.append({"指标": cells[0], "权重": float(cells[1])})
    return pd.DataFrame(rows)


def plot_fig54_pie() -> None:
    df = extract_critic_weights()
    colors = [
        PALETTE["navy"],
        PALETTE["blue"],
        PALETTE["sky"],
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["purple"],
    ]
    explode = [0.035 if i < 2 else 0.0 for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    wedges, _ = ax.pie(
        df["权重"],
        startangle=100,
        counterclock=False,
        colors=colors[: len(df)],
        explode=explode,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
    )
    total = df["权重"].sum()
    for wedge, (_, row) in zip(wedges, df.iterrows(), strict=False):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        ha = "left" if x > 0 else "right"
        ax.annotate(
            f"{row['指标']}\n{row['权重'] / total:.1%}",
            xy=(0.79 * x, 0.79 * y),
            xytext=(1.20 * x, 1.18 * y),
            ha=ha,
            va="center",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
        )
    ax.text(0, 0, "CRITIC\n权重结构", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["dark"])
    ax.set_title("CRITIC 指标权重分布", pad=12, fontweight="bold")
    ax.set_aspect("equal")
    save_fig(fig, "fig5_4_critic_weight_pie")


def topsis_solution_id_from_summary() -> int:
    summary = pd.read_csv(RESULT_DIR / "representative_solution" / "representative_summary.csv")
    for col in ["topsis_solution_id", "solution_id"]:
        if col in summary.columns:
            return int(summary[col].dropna().iloc[0])
    sens = pd.read_csv(RESULT_DIR / "q11_sensitivity_summary.csv")
    return int(sens.loc[(sens["group_type"] == "demand_scale") & (sens["case_key"] == "base"), "topsis_solution_id"].iloc[0])


def plot_fig56_pareto() -> None:
    pareto = pd.read_csv(RESULT_DIR / "representative_solution" / "qvns_nsga2_pareto.csv")
    solution_id = topsis_solution_id_from_summary()
    comp = pareto.loc[pareto["solution_id"] == solution_id]
    min_cost = pareto.loc[pareto["Z1_total_cost"].idxmin()]
    max_service = pareto.loc[pareto["Z2_metro_service_ratio"].idxmax()]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = pareto["Z1_total_cost"] / 10000
    y = pareto["Z2_metro_service_ratio"] * 100
    sc = ax.scatter(
        x,
        y,
        c=pareto["num_open"],
        cmap="viridis",
        s=30 + 8 * pareto["num_open"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    ordered = pareto.sort_values("Z1_total_cost")
    ax.plot(ordered["Z1_total_cost"] / 10000, ordered["Z2_metro_service_ratio"] * 100, color=PALETTE["navy"], lw=1.1, alpha=0.55)
    special = [
        (min_cost, "最低成本端", PALETTE["green"], "s"),
        (max_service, "最高服务端", PALETTE["red"], "^"),
    ]
    if not comp.empty:
        special.append((comp.iloc[0], "TOPSIS 折中解", PALETTE["orange"], "*"))
    for row, label, color, marker in special:
        ax.scatter(row["Z1_total_cost"] / 10000, row["Z2_metro_service_ratio"] * 100, s=130 if marker != "*" else 210, marker=marker, color=color, edgecolor="black", linewidth=0.7, zorder=5, label=label)
    ax.set_xlabel("综合成本（万元/日）")
    ax.set_ylabel("地铁服务比例（%）")
    ax.set_title("QVNS-NSGA-II 求解得到的 Pareto 前沿（p100-g300-runs10）", fontweight="bold")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("启用转运点数")
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, "fig5_6_qvns_pareto_front_updated")


def plot_fig510_qlearning() -> None:
    training = pd.read_csv(RESULT_DIR / "representative_solution" / "qvns_nsga2_training_history.csv")
    final = pd.read_csv(RESULT_DIR / "representative_solution" / "qvns_nsga2_final_history.csv")
    q_table = pd.read_csv(RESULT_DIR / "representative_solution" / "qvns_nsga2_q_table.csv")
    gen_col = "gen" if "gen" in final.columns else final.columns[0]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True)
    ax = axes[0, 0]
    if {"gen", "action", "reward"}.issubset(training.columns):
        reward = training.groupby("gen", as_index=False)["reward"].mean()
        ax.plot(reward["gen"], reward["reward"], color=PALETTE["blue"], lw=1.2, label="平均奖励")
    if {"gen", "epsilon"}.issubset(training.columns):
        eps = training.groupby("gen", as_index=False)["epsilon"].mean()
        ax2 = ax.twinx()
        ax2.plot(eps["gen"], eps["epsilon"], color=PALETTE["green"], lw=1.2, label="ε 探索率")
        ax2.set_ylabel("ε")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    else:
        ax.legend(loc="best", frameon=False)
    ax.set_title("Q-learning 奖励与探索率")
    ax.set_xlabel("迭代代数")
    ax.set_ylabel("奖励")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    if {"gen", "action"}.issubset(training.columns):
        actions = training["action"].map(ACTION_CN).fillna(training["action"])
        action_counts = pd.crosstab(training["gen"], actions).sort_index()
        if len(action_counts) > 80:
            bins = pd.cut(action_counts.index, bins=20)
            action_counts = action_counts.groupby(bins, observed=True).sum()
            xvals = np.arange(len(action_counts))
            labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in action_counts.index]
        else:
            xvals = action_counts.index.to_numpy()
            labels = None
        bottom = np.zeros(len(action_counts))
        colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
        for i, col in enumerate(action_counts.columns):
            ax.bar(xvals, action_counts[col].to_numpy(), bottom=bottom, width=0.82, color=colors[i % len(colors)], label=str(col), linewidth=0)
            bottom += action_counts[col].to_numpy()
        if labels:
            ax.set_xticks(xvals[::3], labels[::3], rotation=35, ha="right")
        ax.legend(ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_title("动作选择频次")
    ax.set_xlabel("迭代区间")
    ax.set_ylabel("触发次数")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    if {"best_service_ratio", "best_cost"}.issubset(final.columns):
        ax.plot(final[gen_col], final["best_service_ratio"] * 100, color=PALETTE["teal"], lw=1.35, label="最佳服务比例")
        ax.set_ylabel("地铁服务比例（%）")
        ax2 = ax.twinx()
        ax2.plot(final[gen_col], final["best_cost"] / 10000, color=PALETTE["orange"], lw=1.2, label="最低成本")
        ax2.set_ylabel("综合成本（万元/日）")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc="best", frameon=False)
    ax.set_title("搜索过程的目标改善")
    ax.set_xlabel("迭代代数")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    if not q_table.empty:
        q_best = q_table.sort_values("q_value").tail(10)
        colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in q_best["q_value"]]
        action_labels = q_best["action"].map(ACTION_CN).fillna(q_best["action"])
        labels = q_best["state"].astype(str) + "\n" + action_labels.astype(str)
        ax.barh(np.arange(len(q_best)), q_best["q_value"], color=colors)
        ax.set_yticks(np.arange(len(q_best)), labels, fontsize=6.5)
        ax.axvline(0, color="#333333", lw=0.8)
    if "safeguard_reject" in training.columns:
        total_reject = int(training["safeguard_reject"].sum())
    elif "safeguard_rejected" in training.columns:
        total_reject = int(training["safeguard_rejected"].sum())
    else:
        total_reject = int(pd.read_csv(RESULT_DIR / "representative_solution" / "representative_run_metrics.csv")["safeguard_rejects"].iloc[0])
    ax.text(0.98, 0.08, f"Safeguard 回滚：{total_reject} 次", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F5F7F9", "edgecolor": "#B8C1CC"})
    ax.set_title("Q 表优势动作与回滚记录")
    ax.set_xlabel("Q 值")
    ax.grid(axis="x", alpha=0.18)
    add_panel_label(ax, "d")
    save_fig(fig, "fig5_10_qlearning_safeguard_updated")


def plot_fig512_ablation() -> None:
    summary = pd.read_csv(ABLATION_DIR / "ablation_summary.csv")
    run = pd.read_csv(ABLATION_DIR / "ablation_run_metrics.csv")
    order = ["NSGA-II", "Q-learning-only", "VNS-only", "QVNS-Q11C"]
    summary["algorithm"] = pd.Categorical(summary["algorithm"], categories=order, ordered=True)
    summary = summary.sort_values("algorithm")
    summary["algorithm_label"] = summary["algorithm"].astype(str).map(ALG_CN).fillna(summary["algorithm"].astype(str))
    run["algorithm_label"] = run["algorithm"].map(ALG_CN).fillna(run["algorithm"])
    colors = {"NSGA-II": "#8A95A5", "Q-learning-only": PALETTE["blue"], "VNS-only": PALETTE["teal"], "QVNS-Q11C": PALETTE["orange"]}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(summary))
    ax.bar(x, summary["HV_mean"], yerr=summary["HV_std"], color=[colors[a] for a in summary["algorithm"].astype(str)], capsize=3, width=0.62)
    ax.set_xticks(x, summary["algorithm_label"], rotation=18, ha="right")
    ax.set_ylabel("HV（越高越好）")
    ax.set_title("前沿综合质量")
    ax.set_ylim(max(0, summary["HV_mean"].min() - 0.12), min(1.05, summary["HV_mean"].max() + summary["HV_std"].max() + 0.08))
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    width = 0.36
    ax.bar(x - width / 2, summary["IGD_mean"], yerr=summary["IGD_std"], width=width, color=PALETTE["red"], alpha=0.78, capsize=3, label="IGD↓")
    ax.bar(x + width / 2, summary["Spacing_mean"], width=width, color=PALETTE["purple"], alpha=0.78, label="Spacing↓")
    ax.set_xticks(x, summary["algorithm_label"], rotation=18, ha="right")
    ax.set_ylabel("指标值")
    ax.set_title("收敛误差与分布均匀性")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    ax.scatter(run["topsis_cost"] / 10000, run["topsis_service_ratio"] * 100, s=48, c=[colors.get(a, "#999999") for a in run["algorithm"]], edgecolor="white", linewidth=0.4, alpha=0.82)
    for _, row in summary.iterrows():
        ax.scatter(row["topsis_cost_mean"] / 10000, row["topsis_service_ratio_mean"] * 100, s=130, marker="D", color=colors[str(row["algorithm"])], edgecolor="black", linewidth=0.7)
        ax.text(row["topsis_cost_mean"] / 10000 + 0.05, row["topsis_service_ratio_mean"] * 100 + 0.15, str(row["algorithm_label"]), fontsize=7.2)
    ax.set_xlabel("TOPSIS 成本（万元/日）")
    ax.set_ylabel("TOPSIS 服务比例（%）")
    ax.set_title("10 次运行折中解分布")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    mech = summary[["algorithm", "actual_vns_calls_mean", "safeguard_rejects_mean", "archive_unique_solutions_mean"]].copy()
    mech["algorithm"] = mech["algorithm"].astype(str).map(ALG_CN).fillna(mech["algorithm"].astype(str))
    mech = mech.set_index("algorithm")
    mech_norm = mech.copy()
    for col in mech_norm.columns:
        maxv = mech_norm[col].max()
        mech_norm[col] = mech_norm[col] / maxv if maxv else 0
    im = ax.imshow(mech_norm.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(3), ["VNS触发", "Safeguard回滚", "档案唯一解"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(mech_norm)), mech_norm.index.astype(str))
    for i in range(mech.shape[0]):
        for j in range(mech.shape[1]):
            ax.text(j, i, f"{mech.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color="#1A1A1A")
    ax.set_title("机制触发诊断")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="归一化强度")
    add_panel_label(ax, "d")
    save_fig(fig, "fig5_12_ablation_mechanism_updated")


def load_geo_layers():
    metro_stations = gpd.read_file(ROOT / "data" / "武侯区地铁站点_2024年10月_WGS84.geojson")
    metro_lines = gpd.read_file(ROOT / "data" / "武侯区地铁线路折线_2024年10月_WGS84.geojson")
    sf_sites = gpd.read_file(ROOT / "data" / "顺丰站点（78）.geojson")
    boundary_path = ROOT / "data" / "510107.json"
    if boundary_path.exists():
        boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        boundary_geom = boundary.geometry.union_all()
    else:
        all_geom = pd.concat([metro_stations.geometry, sf_sites.geometry], ignore_index=True)
        boundary_geom = MultiPoint([g for g in all_geom if g is not None]).convex_hull.buffer(0.018)
    return metro_stations, metro_lines, sf_sites, boundary_geom


def selected_case_path(group_type: str, case_key: str) -> Path:
    return RESULT_DIR / "sensitivity_current_q11c" / group_type / case_key


def draw_case_map(ax: plt.Axes, group_type: str, case_key: str, title: str, metro_stations, metro_lines, sf_sites, boundary_geom) -> None:
    case_dir = selected_case_path(group_type, case_key)
    selected = pd.read_csv(case_dir / "map_selected_station_coordinates.csv")
    assign = pd.read_csv(case_dir / "compromise_assignments.csv")
    gpd.GeoSeries([boundary_geom], crs="EPSG:4326").plot(ax=ax, facecolor="#FBFCFD", edgecolor="#AAB2BD", linewidth=0.9, zorder=0)
    metro_lines.plot(ax=ax, color="#B6BEC7", linewidth=1.2, alpha=0.7, zorder=1)
    sf_sites.plot(ax=ax, marker="o", markersize=9, facecolor="none", edgecolor="#D0A46A", linewidth=0.55, alpha=0.72, zorder=2)
    metro_stations.plot(ax=ax, marker="o", markersize=11, color="#FFFFFF", edgecolor="#596676", linewidth=0.55, zorder=3)
    metro_assign = assign.loc[assign["服务方式"].astype(str).str.contains("地铁", na=False)].copy()
    if not metro_assign.empty:
        sample = metro_assign.sample(min(len(metro_assign), 55), random_state=17)
        station_lookup = selected.set_index("地铁站名称")[["绘图经度", "绘图纬度"]].to_dict("index")
        for _, row in sample.iterrows():
            station = str(row.get("中转站", ""))
            if station in station_lookup:
                ax.plot([row["经度"], station_lookup[station]["绘图经度"]], [row["纬度"], station_lookup[station]["绘图纬度"]], color="#7AA6C2", lw=0.45, alpha=0.26, zorder=2)
    sizes = 80 + 210 * selected["负载率"].clip(0, 1)
    ax.scatter(selected["绘图经度"], selected["绘图纬度"], s=sizes, marker="^", color=PALETTE["orange"], edgecolor="#2F2F2F", linewidth=0.75, zorder=5)
    for _, row in selected.iterrows():
        ax.text(row["绘图经度"] + 0.002, row["绘图纬度"] + 0.002, str(row["地铁站名称"]), fontsize=6.6, color="#202020", zorder=6)
    minx, miny, maxx, maxy = boundary_geom.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=9.2, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def sensitivity_label(case_key: str) -> str:
    labels = {
        "low": "低需求",
        "base": "基准",
        "high": "高需求",
        "r3": "3 km",
        "r4": "4 km",
        "r5": "5 km",
        "s5_5": "5:5",
        "s6_4": "6:4",
        "s7_3": "7:3",
        "s8_2": "8:2",
        "s9_1": "9:1",
    }
    return labels.get(case_key, case_key)


def plot_sensitivity_group(group_type: str, cases: list[str], name: str, title: str) -> None:
    summary = pd.read_csv(RESULT_DIR / "q11_sensitivity_summary.csv")
    subset = summary[(summary["group_type"] == group_type) & (summary["case_key"].isin(cases))].copy()
    subset["case_key"] = pd.Categorical(subset["case_key"], categories=cases, ordered=True)
    subset = subset.sort_values("case_key")
    metro_stations, metro_lines, sf_sites, boundary_geom = load_geo_layers()
    n = len(cases)
    fig = plt.figure(figsize=(8.6, 5.1 if n == 3 else 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 0.75])
    for i, case in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        row = subset.loc[subset["case_key"] == case].iloc[0]
        draw_case_map(
            ax,
            group_type,
            case,
            f"{sensitivity_label(case)}：{row['topsis_open_count']:.0f} 站 / {row['topsis_service_ratio'] * 100:.1f}%",
            metro_stations,
            metro_lines,
            sf_sites,
            boundary_geom,
        )
        add_panel_label(ax, chr(ord("a") + i))
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(subset))
    ax.bar(x - 0.18, subset["topsis_cost"] / 10000, width=0.36, color=PALETTE["navy"], label="综合成本")
    ax.set_ylabel("综合成本（万元/日）")
    ax.set_xticks(x, [sensitivity_label(c) for c in subset["case_key"].astype(str)])
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, subset["topsis_service_ratio"] * 100, color=PALETTE["green"], marker="o", lw=1.8, label="地铁服务比例")
    ax2.set_ylabel("地铁服务比例（%）")
    for xi, row in zip(x, subset.itertuples(), strict=False):
        ax.text(xi - 0.18, row.topsis_cost / 10000 + 0.35, f"{row.topsis_cost / 10000:.1f}", ha="center", va="bottom", fontsize=7)
        ax2.text(xi + 0.18, row.topsis_service_ratio * 100 + 0.9, f"{row.topsis_service_ratio * 100:.1f}%", ha="center", va="bottom", fontsize=7, color=PALETTE["green"])
    ax.grid(axis="y", alpha=0.18)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=2, frameon=False)
    add_panel_label(ax, chr(ord("a") + n))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color="#B6BEC7", lw=1.4, label="地铁线路"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#D0A46A", markersize=5, label="顺丰站点"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=PALETTE["orange"], markeredgecolor="#2F2F2F", markersize=7, label="启用转运点"),
        Line2D([0], [0], color="#7AA6C2", lw=1.0, alpha=0.45, label="地铁服务连接"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    save_fig(fig, name)


def plot_all_sensitivity() -> None:
    plot_sensitivity_group("demand_scale", ["low", "base", "high"], "fig5_13_demand_scale_sensitivity_map", "需求规模变化下的布局结果对比")
    plot_sensitivity_group("service_radius", ["r3", "r4", "r5"], "fig5_14_service_radius_sensitivity_map", "服务半径变化下的布局结果对比")
    plot_sensitivity_group("source_share", ["s5_5", "s6_4", "s7_3", "s8_2", "s9_1"], "fig5_15_source_share_sensitivity_map", "物流货源比例变化下的布局结果对比")


def main() -> None:
    setup_style()
    plot_fig54_pie()
    plot_fig56_pareto()
    plot_fig510_qlearning()
    plot_fig512_ablation()
    plot_all_sensitivity()
    print(f"Figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
