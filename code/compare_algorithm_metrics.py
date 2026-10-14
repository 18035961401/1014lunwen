# -*- coding: utf-8 -*-
"""Compare QVNS-NSGA-II and traditional NSGA-II outputs.

The script reads Pareto CSV files from two output folders, removes duplicated
objective points, normalizes objectives to a two-objective minimization space,
then calculates HV, GD, IGD, Spacing and plots comparison figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _get_pyplot():
    import matplotlib.pyplot as plt
    return plt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare multi-objective performance of QVNS-NSGA-II and NSGA-II.")
    parser.add_argument("--qvns-dir", type=Path, required=True, help="QVNS-NSGA-II scenario output directory, e.g. results/qvns_nsga2_thesis/no_budget_range")
    parser.add_argument("--nsga-dir", type=Path, required=True, help="Traditional NSGA-II scenario output directory, e.g. results/nsga2_no_budget_baseline/no_budget_range")
    parser.add_argument("--output-dir", type=Path, default=Path("results/algorithm_compare"), help="Directory for metrics and comparison figures.")
    parser.add_argument("--label", default="scenario", help="Figure label, e.g. no_budget.")
    parser.add_argument("--reference", type=float, nargs=2, default=[1.1, 1.1], help="HV reference point in normalized minimization space.")
    return parser.parse_args(argv)


def read_pareto_csv(folder: Path, kind: str) -> pd.DataFrame:
    candidates = []
    if kind == "qvns":
        candidates = [
            folder / "qvns_nsga2_pareto.csv",
            folder / "qvns_nsga2_pareto_unique_objectives.csv",
        ]
    else:
        candidates = [folder / "nsga2_pareto.csv", folder / "nsga2_pareto_unique_objectives.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Cannot find Pareto CSV in {folder}")


def unique_objective_points(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_cost_key"] = out["Z1_total_cost"].round(6)
    out["_service_ratio_key"] = out["Z2_metro_service_ratio"].round(12)
    out = out.drop_duplicates(["_cost_key", "_service_ratio_key"]).sort_values(["Z1_total_cost", "Z2_metro_service_ratio"])
    return out.drop(columns=["_cost_key", "_service_ratio_key"])


def nondominated(points: np.ndarray) -> np.ndarray:
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if np.all(q <= p + 1e-12) and np.any(q < p - 1e-12):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return np.array(keep, dtype=int)


def hv_2d(points: np.ndarray, ref=(1.1, 1.1)) -> float:
    if len(points) == 0:
        return 0.0
    pts = points[nondominated(points)]
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if x <= ref[0] and y < prev_y:
            hv += max(0.0, ref[0] - x) * max(0.0, prev_y - y)
            prev_y = y
    return float(hv)


def min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2)).min(axis=1)


def spacing(points: np.ndarray) -> float:
    if len(points) <= 2:
        return 0.0
    d = []
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        d.append(np.abs(others - p).sum(axis=1).min())
    d = np.array(d)
    return float(np.sqrt(((d - d.mean()) ** 2).sum() / (len(d) - 1)))


def extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.sqrt(((points.max(axis=0) - points.min(axis=0)) ** 2).sum()))


def add_normalized_objectives(df: pd.DataFrame, cost_min, cost_max, cov_min, cov_max) -> pd.DataFrame:
    df = df.copy()
    df["f1_cost_norm"] = (df["Z1_total_cost"] - cost_min) / (cost_max - cost_min + 1e-12)
    df["f2_service_ratio_loss_norm"] = (cov_max - df["Z2_metro_service_ratio"]) / (cov_max - cov_min + 1e-12)
    return df


def build_metrics(name: str, df: pd.DataFrame, reference_front: np.ndarray, hv_ref) -> dict:
    points = df[["f1_cost_norm", "f2_service_ratio_loss_norm"]].to_numpy()
    row = {
        "algorithm": name,
        "unique_points": len(df),
        "HV": hv_2d(points, ref=hv_ref),
        "GD": float(min_dist(points, reference_front).mean()) if len(reference_front) else np.nan,
        "IGD": float(min_dist(reference_front, points).mean()) if len(reference_front) else np.nan,
        "Spacing": spacing(points),
        "Extent": extent(points),
        "min_cost": float(df["Z1_total_cost"].min()),
        "max_metro_service_ratio": float(df["Z2_metro_service_ratio"].max()),
    }
    if "carbon_cost" in df.columns:
        row["min_carbon_cost"] = float(df["carbon_cost"].min())
    for col, agg, out_col in [
        ("line_capacity_violation", "max", "max_line_capacity_violation"),
        ("max_line_load_rate", "max", "max_line_load_rate"),
        ("max_line_open_count", "max", "max_line_open_count"),
    ]:
        if col in df.columns:
            row[out_col] = float(df[col].max())
    return row


def plot_outputs(qvns: pd.DataFrame, nsga: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path, label: str):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "PingFang SC", "Heiti SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.scatter(nsga["Z1_total_cost"], nsga["Z2_metro_service_ratio"], s=46, color="#8c8c8c", alpha=0.75, edgecolor="white", linewidth=0.5, label="NSGA-II")
    ax.scatter(qvns["Z1_total_cost"], qvns["Z2_metro_service_ratio"], s=52, color="#1f77b4", alpha=0.88, edgecolor="white", linewidth=0.6, label="QVNS-NSGA-II")
    ax.set_xlabel("Total cost (yuan/day)")
    ax.set_ylabel("Metro service ratio")
    ax.set_title(f"Pareto front comparison, {label}")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_pareto_compare.png", dpi=420, bbox_inches="tight")
    fig.savefig(output_dir / "fig_pareto_compare.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(metrics["algorithm"], metrics["HV"], color=["#1f77b4", "#8c8c8c"])
    ax.set_ylabel("Hypervolume, normalized, higher is better")
    ax.set_title(f"HV comparison, {label}")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for i, value in enumerate(metrics["HV"]):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_hv_compare.png", dpi=420, bbox_inches="tight")
    fig.savefig(output_dir / "fig_hv_compare.pdf", bbox_inches="tight")
    plt.close(fig)

    cols = ["GD", "IGD", "Spacing"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, col in zip(axes, cols):
        ax.bar(metrics["algorithm"], metrics[col], color=["#1f77b4", "#8c8c8c"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for i, value in enumerate(metrics[col]):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Convergence and diversity indicators, {label} (lower is better)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_gd_igd_spacing_compare.png", dpi=420, bbox_inches="tight")
    fig.savefig(output_dir / "fig_gd_igd_spacing_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qvns = unique_objective_points(read_pareto_csv(args.qvns_dir, "qvns"))
    nsga = unique_objective_points(read_pareto_csv(args.nsga_dir, "nsga"))
    qvns["algorithm"] = "QVNS-NSGA-II"
    nsga["algorithm"] = "NSGA-II"
    combined = pd.concat([qvns, nsga], ignore_index=True)
    cost_min, cost_max = combined["Z1_total_cost"].min(), combined["Z1_total_cost"].max()
    cov_min, cov_max = combined["Z2_metro_service_ratio"].min(), combined["Z2_metro_service_ratio"].max()
    qvns = add_normalized_objectives(qvns, cost_min, cost_max, cov_min, cov_max)
    nsga = add_normalized_objectives(nsga, cost_min, cost_max, cov_min, cov_max)
    combined_norm = pd.concat([qvns, nsga], ignore_index=True)
    all_points = combined_norm[["f1_cost_norm", "f2_service_ratio_loss_norm"]].to_numpy()
    reference_front = all_points[nondominated(all_points)]
    metrics = pd.DataFrame([
        build_metrics("QVNS-NSGA-II", qvns, reference_front, tuple(args.reference)),
        build_metrics("NSGA-II", nsga, reference_front, tuple(args.reference)),
    ])
    qvns.to_csv(args.output_dir / "qvns_unique_objectives.csv", index=False, encoding="utf-8-sig")
    nsga.to_csv(args.output_dir / "nsga_unique_objectives.csv", index=False, encoding="utf-8-sig")
    combined_norm.to_csv(args.output_dir / "combined_normalized_points.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "algorithm_performance_metrics.csv", index=False, encoding="utf-8-sig")
    plot_outputs(qvns, nsga, metrics, args.output_dir, args.label)
    print(metrics.to_string(index=False))
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
