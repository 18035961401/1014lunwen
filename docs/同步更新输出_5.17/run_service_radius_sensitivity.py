# -*- coding: utf-8 -*-
"""Run 3/4/5 km terminal service-radius sensitivity analysis.

The script keeps the same case setting and QVNS-Q11C solver configuration used
in the final paper experiments, and changes only ``D_k_max`` (the terminal
service radius constraint) to 3, 4 and 5 km.

Default settings are intentionally moderate so the script can be checked quickly.
For thesis-final numbers, use for example::

    python run_service_radius_sensitivity.py --runs 10 --pop-size 100 --max-gen 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "code" / "main.py").exists():
            return path
    raise RuntimeError("Cannot locate project root containing code/main.py")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import (  # noqa: E402
    build_metro_logistics_instance,
    choose_compromise,
    evaluate_solution,
    run_qvns_nsga2,
    solution_to_tables,
)


Q_TO_RADIUS_COLUMNS = {
    "space_coverage_4km_volume": "space_coverage_radius_volume",
    "space_coverage_4km_ratio": "space_coverage_radius_ratio",
    "fulfilled_4km_volume": "fulfilled_radius_volume",
    "fulfillment_ratio_within_4km": "fulfillment_ratio_within_radius",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", nargs="+", type=float, default=[3.0, 4.0, 5.0], help="Service radii in km.")
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per radius.")
    parser.add_argument("--pop-size", type=int, default=80, help="Population size.")
    parser.add_argument("--max-gen", type=int, default=200, help="Maximum generations.")
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--demand-scenario", default="base")
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "service_radius_sensitivity_results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def qvns_q11c_kwargs(pop_size: int, max_gen: int) -> dict[str, Any]:
    epsilon_decay = (0.010 / 0.30) ** (1.0 / max(1, int(max_gen)))
    return dict(
        alpha=0.15,
        gamma=0.88,
        epsilon=0.30,
        epsilon_min=0.010,
        epsilon_decay=epsilon_decay,
        q_learning_enabled=True,
        q_action_apply_rate=0.25,
        q_action_profile="safe",
        safeguard=True,
        safeguard_threshold=-0.15,
        archive_feedback_interval=8,
        archive_feedback_rate=0.15,
        vns_mode="adaptive",
        vns_interval=0,
        vns_elite_rate=0.12,
        vns_trials_per_neighborhood=2,
        vns_max_neighborhood=3,
        archive_size=max(pop_size, int(round(2.5 * pop_size))),
        archive_boundary_keep_rate=0.20,
        vns_candidate_strategy="mixed",
        stagnation_window=10,
        hv_improvement_threshold=0.0025,
        min_vns_gap=5,
    )


def build_instance_for_radius(args: argparse.Namespace, radius_km: float):
    inst = build_metro_logistics_instance(
        args.demand_scenario,
        top_n=args.top_n,
        model_setting=args.model_setting,
    )
    inst.D_k_max = np.full(inst.K, float(radius_km), dtype=float)
    return inst


def safe_station_names(station_df: pd.DataFrame) -> str:
    if station_df.empty:
        return ""
    open_df = station_df[station_df.iloc[:, :].apply(lambda row: True, axis=1)].copy()
    enable_cols = [c for c in station_df.columns if "启用" in c or "鍚敤" in c]
    name_cols = [c for c in station_df.columns if "站" in c or "绔" in c]
    if enable_cols:
        open_df = station_df[station_df[enable_cols[0]].astype(float) > 0.5]
    if name_cols and len(open_df):
        return "；".join(str(v) for v in open_df[name_cols[0]].tolist())
    return ""


def radius_folder_name(radius_km: float) -> str:
    return f"radius_{radius_km:.1f}km".replace(".", "p")


def save_compromise_tables(
    output_dir: Path,
    radius_km: float,
    run_id: int,
    assign_df: pd.DataFrame,
    station_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> Path:
    radius_dir = output_dir / "compromise_tables" / radius_folder_name(radius_km) / f"run_{run_id:02d}"
    radius_dir.mkdir(parents=True, exist_ok=True)
    assign_df.to_csv(radius_dir / "service_radius_compromise_assignments.csv", index=False, encoding="utf-8-sig")
    station_df.to_csv(radius_dir / "service_radius_compromise_station_load.csv", index=False, encoding="utf-8-sig")
    line_df.to_csv(radius_dir / "service_radius_compromise_line_load.csv", index=False, encoding="utf-8-sig")
    return radius_dir


def summarize_result(
    result: dict[str, Any],
    inst,
    radius_km: float,
    run_id: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for solution_id, dec in enumerate(result["pareto_dec"]):
        objs, cv, parts = evaluate_solution(dec, inst, use_repair=True, collect_parts=True)
        row = {
            "radius_km": float(radius_km),
            "run_id": int(run_id),
            "seed": int(seed),
            "solution_id": int(solution_id),
            "Z1_total_cost": float(objs[0]),
            "Z2_metro_service_ratio": float(-objs[1]),
            "CV": float(cv),
        }
        row.update(parts)
        rows.append(row)
    points = pd.DataFrame(rows)
    for old, new in Q_TO_RADIUS_COLUMNS.items():
        if old in points.columns:
            points[new] = points[old]
    if points.empty:
        return {
            "radius_km": float(radius_km),
            "run_id": int(run_id),
            "seed": int(seed),
            "runtime_seconds": float(result.get("runtime", np.nan)),
            "pareto_size": 0,
        }, points

    feasible = points[points["CV"] <= 1e-9].copy()
    view = feasible if len(feasible) else points
    best_cost = view.loc[view["Z1_total_cost"].idxmin()]
    best_service = view.loc[view["Z2_metro_service_ratio"].idxmax()]
    comp_idx = choose_compromise(result["pareto_objs"]) if len(result.get("pareto_objs", [])) else 0
    comp_dec = result["pareto_dec"][comp_idx]
    comp_objs, comp_cv, comp_parts = evaluate_solution(comp_dec, inst, use_repair=True, collect_parts=True)
    comp_assign_df, comp_station_df, comp_line_df = solution_to_tables(comp_dec, inst, use_repair=True)
    open_station_count = int(comp_parts.get("num_open", np.nan)) if "num_open" in comp_parts else np.nan
    used_train_trips = float(comp_parts.get("used_train_trips_total", np.nan))
    row = {
        "radius_km": float(radius_km),
        "run_id": int(run_id),
        "seed": int(seed),
        "runtime_seconds": float(result.get("runtime", np.nan)),
        "pareto_size": int(len(points)),
        "feasible_pareto_size": int(len(feasible)),
        "unique_points": int(len(view[["Z1_total_cost", "Z2_metro_service_ratio"]].round(8).drop_duplicates())),
        "min_cost": float(best_cost["Z1_total_cost"]),
        "max_service_ratio": float(best_service["Z2_metro_service_ratio"]),
        "topsis_cost": float(comp_objs[0]),
        "topsis_service_ratio": float(-comp_objs[1]),
        "topsis_cv": float(comp_cv),
        "topsis_open_count": open_station_count,
        "topsis_metro_volume": float(comp_parts.get("metro_volume", np.nan)),
        "topsis_direct_volume": float(comp_parts.get("direct_volume", np.nan)),
        "topsis_space_coverage_radius_ratio": float(comp_parts.get("space_coverage_4km_ratio", np.nan)),
        "topsis_fulfillment_ratio_within_radius": float(comp_parts.get("fulfillment_ratio_within_4km", np.nan)),
        "topsis_avg_feeder_distance": float(comp_parts.get("avg_feeder_distance", np.nan)),
        "topsis_used_train_trips": used_train_trips,
        "topsis_total_carbon_tCO2": float(comp_parts.get("total_carbon_tCO2", np.nan)),
        "actual_vns_calls": int(result.get("actual_vns_calls", 0)),
        "archive_final_size": int(result.get("archive_final_size", 0)),
        "archive_unique_solutions": int(result.get("archive_unique_solutions", 0)),
        "topsis_open_stations": safe_station_names(comp_station_df),
    }
    return row, points, comp_assign_df, comp_station_df, comp_line_df


def aggregate_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "runtime_seconds",
        "pareto_size",
        "unique_points",
        "min_cost",
        "max_service_ratio",
        "topsis_cost",
        "topsis_service_ratio",
        "topsis_open_count",
        "topsis_metro_volume",
        "topsis_direct_volume",
        "topsis_space_coverage_radius_ratio",
        "topsis_fulfillment_ratio_within_radius",
        "topsis_avg_feeder_distance",
        "topsis_used_train_trips",
        "topsis_total_carbon_tCO2",
    ]
    rows = []
    for radius, group in run_df.groupby("radius_km", sort=True):
        row = {"radius_km": radius, "runs": int(len(group))}
        for col in metric_cols:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        mode_station = group["topsis_open_stations"].mode(dropna=True) if "topsis_open_stations" in group else []
        row["representative_open_stations"] = str(mode_station.iloc[0]) if len(mode_station) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped: {exc}")
        return

    mpl.rcParams.update({
        "font.family": "Microsoft YaHei",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    x = summary["radius_km"].astype(float).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=200)
    panels = [
        ("topsis_cost_mean", "TOPSIS成本（万元/日）", 10000.0, "a  折中方案成本"),
        ("topsis_service_ratio_mean", "TOPSIS服务比例（%）", 0.01, "b  折中方案服务比例"),
        ("topsis_space_coverage_radius_ratio_mean", "服务半径覆盖需求比例（%）", 0.01, "c  空间覆盖能力"),
        ("topsis_avg_feeder_distance_mean", "平均末端接驳距离（km）", 1.0, "d  接驳距离"),
    ]
    for ax, (col, ylabel, scale, title) in zip(axes.ravel(), panels):
        if col not in summary.columns:
            ax.axis("off")
            continue
        y = summary[col].astype(float).to_numpy() / scale
        ax.plot(x, y, marker="o", linewidth=1.8, color="#2F6F73")
        ax.set_xlabel("末端服务半径（km）")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.grid(axis="y", color="#E8ECEF", linewidth=0.6)
    fig.suptitle("末端服务半径敏感性分析", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    for ext in ("png", "svg", "pdf"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"fig_service_radius_sensitivity.{ext}", **kwargs)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    for radius in args.radii:
        for run_id in range(1, args.runs + 1):
            seed = int(args.seed + 1000 * int(round(radius * 10)) + run_id - 1)
            print(f"[radius={radius:.1f} km] run={run_id}/{args.runs}, seed={seed}")
            inst = build_instance_for_radius(args, radius)
            started = time.time()
            result = run_qvns_nsga2(
                inst,
                pop_size=args.pop_size,
                max_gen=args.max_gen,
                seed=seed,
                verbose=not args.quiet,
                **qvns_q11c_kwargs(args.pop_size, args.max_gen),
            )
            result["runtime"] = float(time.time() - started)
            row, points, comp_assign_df, comp_station_df, comp_line_df = summarize_result(result, inst, radius, run_id, seed)
            run_rows.append(row)
            if len(points):
                point_frames.append(points)
            save_compromise_tables(args.output_dir, radius, run_id, comp_assign_df, comp_station_df, comp_line_df)
            pd.DataFrame(run_rows).to_csv(args.output_dir / "service_radius_run_metrics.csv", index=False, encoding="utf-8-sig")
            if point_frames:
                pd.concat(point_frames, ignore_index=True).to_csv(args.output_dir / "service_radius_pareto_points.csv", index=False, encoding="utf-8-sig")
            aggregate_summary(pd.DataFrame(run_rows)).to_csv(args.output_dir / "service_radius_summary.csv", index=False, encoding="utf-8-sig")
    summary = aggregate_summary(pd.DataFrame(run_rows))
    summary.to_csv(args.output_dir / "service_radius_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, args.output_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nOutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
