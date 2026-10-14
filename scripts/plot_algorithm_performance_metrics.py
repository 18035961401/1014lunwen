# -*- coding: utf-8 -*-
"""Compute and plot algorithm performance metrics from saved Pareto results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from chinese_font_utils import configure_matplotlib_cjk, get_cjk_font_properties  # noqa: E402

configure_matplotlib_cjk(plt)
CJK_FONT = get_cjk_font_properties()

ALGORITHM_ORDER = ["QVNS-NSGA2", "NSGA2", "MOPSO", "SPEA2"]
ALGORITHM_LABELS = {
    "QVNS-NSGA2": "QVNS-NSGA-II",
    "NSGA2": "NSGA-II",
    "MOPSO": "MOPSO",
    "SPEA2": "SPEA2",
}
ALGORITHM_COLORS = {
    "QVNS-NSGA2": "#1f77b4",
    "NSGA2": "#ff7f0e",
    "MOPSO": "#2ca02c",
    "SPEA2": "#9467bd",
}
METRICS = ["HV", "IGD", "Spacing", "Spread", "unique_solutions", "runtime_seconds"]
METRIC_LABELS = {
    "HV": "HV（越大越好）",
    "IGD": "IGD（越小越好）",
    "Spacing": "Spacing（越小越好）",
    "Spread": "Spread（越小越好）",
    "unique_solutions": "非重复解数目（越大越好）",
    "runtime_seconds": "运行时间（s，越小越好）",
}
STABILITY_METRICS = [
    ("HV", "HV"),
    ("IGD", "IGD"),
    ("Spacing", "Spacing"),
    ("Spread", "Spread"),
    ("min_total_cost_10k", "最低综合成本（万元/日）"),
    ("max_service_percent", "最大地铁服务比例（%）"),
]
def normalize_objectives(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.maximum(upper - lower, 1e-12)
    return (points - lower) / span


def compute_hypervolume_2d(points: np.ndarray, ref_point: tuple[float, float]) -> float:
    if len(points) == 0:
        return 0.0
    clean = points[np.all(np.isfinite(points), axis=1)]
    if len(clean) == 0:
        return 0.0
    front = _nondominated_min_points(clean)
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    previous_y = float(ref_point[1])
    for x_value, y_value in front:
        if x_value <= ref_point[0] and y_value < previous_y:
            hv += max(0.0, ref_point[0] - x_value) * max(0.0, previous_y - y_value)
            previous_y = y_value
    return float(hv)


def _nondominated_min_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if np.all(other <= point + 1e-12) and np.any(other < point - 1e-12):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points[np.array(keep, dtype=int)] if keep else points[:0]


def compute_igd(points: np.ndarray, reference: np.ndarray) -> float:
    if len(points) == 0 or len(reference) == 0:
        return float("nan")
    distances = np.linalg.norm(reference[:, None, :] - points[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def compute_spacing(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.sum((nearest - np.mean(nearest)) ** 2) / max(1, len(points) - 1)))


def compute_spread(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    front = points[np.argsort(points[:, 0])]
    consecutive = np.linalg.norm(np.diff(front, axis=0), axis=1)
    mean_distance = float(np.mean(consecutive)) if len(consecutive) else 0.0
    if mean_distance <= 1e-12:
        return 0.0
    extent = float(np.linalg.norm(front[0] - front[-1]))
    numerator = extent + float(np.sum(np.abs(consecutive - mean_distance)))
    denominator = extent + len(consecutive) * mean_distance
    return float(numerator / max(denominator, 1e-12))


def _objective_points(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["Z1_total_cost"].astype(float).to_numpy(),
        -df["Z2_metro_service_ratio"].astype(float).to_numpy(),
    ])


def _unique_solution_count(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    rounded = np.round(points, 8)
    return int(len(np.unique(rounded, axis=0)))


def build_performance_tables(pareto_df: pd.DataFrame, final_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feasible = pareto_df[pareto_df["CV"].astype(float) <= 1e-9].copy()
    if feasible.empty:
        feasible = pareto_df.copy()
    all_points = _objective_points(feasible)
    lower = np.nanmin(all_points, axis=0)
    upper = np.nanmax(all_points, axis=0)
    reference_front = _nondominated_min_points(normalize_objectives(all_points, lower, upper))
    rows = []
    runtime_lookup = {
        (str(row.algorithm), int(row.run_id)): float(row.runtime_seconds)
        for row in final_summary_df.itertuples(index=False)
        if hasattr(row, "runtime_seconds")
    }
    for (algorithm, run_id), group in feasible.groupby(["algorithm", "run_id"]):
        raw_points = _objective_points(group)
        norm_points = normalize_objectives(raw_points, lower, upper)
        norm_front = _nondominated_min_points(norm_points)
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "run_id": int(run_id),
                "HV": compute_hypervolume_2d(norm_front, (1.1, 1.1)),
                "IGD": compute_igd(norm_front, reference_front),
                "Spacing": compute_spacing(norm_front),
                "Spread": compute_spread(norm_front),
                "unique_solutions": _unique_solution_count(raw_points),
                "min_total_cost_10k": float(np.min(raw_points[:, 0]) / 10000.0) if len(raw_points) else float("nan"),
                "max_service_percent": float(np.max(-raw_points[:, 1]) * 100.0) if len(raw_points) else float("nan"),
                "runtime_seconds": runtime_lookup.get((str(algorithm), int(run_id)), float("nan")),
            }
        )
    run_metrics = pd.DataFrame(rows)
    agg_spec = {}
    for metric in METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
    summary = run_metrics.groupby(["algorithm", "algorithm_label"], as_index=False).agg(**agg_spec)
    summary["algorithm_order"] = summary["algorithm"].map({name: idx for idx, name in enumerate(ALGORITHM_ORDER)}).fillna(999)
    summary = summary.sort_values("algorithm_order").drop(columns=["algorithm_order"])
    return run_metrics, summary


def plot_stability_distribution(run_metrics: pd.DataFrame, output_dir: Path, prefix: str = "paper") -> None:
    if run_metrics.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    algorithms = [alg for alg in ALGORITHM_ORDER if alg in set(run_metrics["algorithm"])]
    labels = [ALGORITHM_LABELS.get(alg, alg) for alg in algorithms]
    colors = [ALGORITHM_COLORS.get(alg, "#999999") for alg in algorithms]
    for ax, (metric, title) in zip(axes.ravel(), STABILITY_METRICS):
        values = [run_metrics.loc[run_metrics["algorithm"] == alg, metric].astype(float).dropna().to_numpy() for alg in algorithms]
        box = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.2},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"color": "#666666", "linewidth": 0.9},
            boxprops={"linewidth": 0.9, "color": "#555555"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.48)
        for index, series in enumerate(values, start=1):
            if len(series):
                jitter = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else np.array([0.0])
                ax.scatter(np.full(len(series), index) + jitter, series, s=16, color=colors[index - 1], alpha=0.75, zorder=3, edgecolors="white", linewidths=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold", fontproperties=CJK_FONT)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(CJK_FONT)
    fig.suptitle("多次运行稳定性分布", fontsize=16, fontweight="bold", y=0.99, fontproperties=CJK_FONT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{prefix}_stability_distribution.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_performance_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    ordered = summary.copy()
    ordered["algorithm_order"] = ordered["algorithm"].map({name: idx for idx, name in enumerate(ALGORITHM_ORDER)}).fillna(999)
    ordered = ordered.sort_values("algorithm_order")
    labels = [ALGORITHM_LABELS.get(alg, alg) for alg in ordered["algorithm"]]
    colors = [ALGORITHM_COLORS.get(alg, "#777777") for alg in ordered["algorithm"]]
    for ax, metric in zip(axes, METRICS):
        means = ordered[f"{metric}_mean"].astype(float).to_numpy()
        stds = ordered[f"{metric}_std"].fillna(0.0).astype(float).to_numpy()
        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.5)
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold", fontproperties=CJK_FONT)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        for label in ax.get_xticklabels():
            label.set_fontproperties(CJK_FONT)
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("四种算法性能指标对比", fontsize=16, fontweight="bold", y=0.99, fontproperties=CJK_FONT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig_algorithm_performance_metrics.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / "fig_algorithm_performance_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HV, IGD, Spacing, Spread, unique solutions, and runtime.")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.result_dir
    pareto_df = pd.read_csv(result_dir / "pareto_cost_service_points.csv", encoding="utf-8-sig")
    final_summary_df = pd.read_csv(result_dir / "final_summary.csv", encoding="utf-8-sig")
    run_metrics, summary = build_performance_tables(pareto_df, final_summary_df)
    run_metrics.to_csv(result_dir / "algorithm_performance_run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(result_dir / "algorithm_performance_summary.csv", index=False, encoding="utf-8-sig")
    plot_performance_metrics(summary, result_dir)
    plot_stability_distribution(run_metrics, result_dir, prefix="paper_fig_algorithm")
    print(f"Performance metrics saved to: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
