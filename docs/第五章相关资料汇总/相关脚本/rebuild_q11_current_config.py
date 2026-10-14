# -*- coding: utf-8 -*-
"""Rebuild Chapter 5 Q11 results with the current default QVNS-NSGA-II config.

This script is intentionally placed next to the result files it regenerates.
It overwrites old-Q11 result files in this folder and writes sensitivity tables
and maps for demand scale, service radius and source-share scenarios.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "code" / "main.py").exists():
            return path
    raise RuntimeError("Cannot locate project root containing code/main.py")


RESULT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(RESULT_DIR)
RESULTS_ROOT = PROJECT_ROOT / "results"
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (CODE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import (  # noqa: E402
    build_metro_logistics_instance,
    choose_compromise,
    evaluate_solution,
    export_qvns_nsga2_results,
    plot_compromise_map,
    run_qvns_nsga2,
    solution_to_tables,
)
from run_qvns_ablation_experiment import (  # noqa: E402
    _epsilon_decay_for_target,
    hv_2d,
    igd,
    nondominated,
    spacing,
    spread_delta,
    summarize_result,
)


CURRENT_Q11C_CONFIG: dict[str, Any] = {
    "alpha": 0.15,
    "gamma": 0.88,
    "epsilon": 0.30,
    "epsilon_min": 0.010,
    "epsilon_decay": (0.010 / 0.30) ** (1.0 / 300),
    "q_action_apply_rate": 0.25,
    "vns_interval": 0,
    "vns_mode": "adaptive",
    "vns_elite_rate": 0.12,
    "vns_trials_per_neighborhood": 2,
    "vns_max_neighborhood": 3,
    "archive_size": 250,
    "archive_boundary_keep_rate": 0.20,
    "stagnation_window": 10,
    "hv_improvement_threshold": 0.0025,
    "min_vns_gap": 5,
    "archive_feedback_interval": 8,
    "archive_feedback_rate": 0.15,
    "q_learning_enabled": True,
    "q_action_profile": "safe",
    "safeguard": True,
    "safeguard_threshold": -0.15,
    "vns_candidate_strategy": "mixed",
}


DEMAND_CASES = [
    {"case_key": "low", "case_name": "保守中心城区需求情景", "scenario": "low", "lambda": 1.20},
    {"case_key": "base", "case_name": "基准中心城区需求情景", "scenario": "base", "lambda": 1.35},
    {"case_key": "high", "case_name": "高强度中心城区需求情景", "scenario": "high", "lambda": 1.50},
]

RADIUS_CASES = [
    {"case_key": "r3", "case_name": "服务半径3km", "radius_km": 3.0},
    {"case_key": "r4", "case_name": "服务半径4km", "radius_km": 4.0},
    {"case_key": "r5", "case_name": "服务半径5km", "radius_km": 5.0},
]

SOURCE_SHARE_CASES = [
    {"case_key": "s5_5", "case_name": "货源比例5:5", "source_share": (0.5, 0.5)},
    {"case_key": "s6_4", "case_name": "货源比例6:4", "source_share": (0.6, 0.4)},
    {"case_key": "s7_3", "case_name": "货源比例7:3", "source_share": (0.7, 0.3)},
    {"case_key": "s8_2", "case_name": "货源比例8:2（基准）", "source_share": (0.8, 0.2)},
    {"case_key": "s9_1", "case_name": "货源比例9:1", "source_share": (0.9, 0.1)},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_case_instance(args: argparse.Namespace, case: dict[str, Any]):
    inst = build_metro_logistics_instance(
        scenario=case.get("scenario", "base"),
        top_n=args.top_n,
        model_setting=args.model_setting,
    )
    if "radius_km" in case:
        inst.D_k_max[:] = float(case["radius_km"])
    if "source_share" in case:
        inst.source_share = np.asarray(case["source_share"], dtype=float)
    return inst


def qvns_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(CURRENT_Q11C_CONFIG)
    cfg["archive_size"] = max(args.pop_size, int(round(2.5 * args.pop_size)))
    cfg["epsilon_decay"] = _epsilon_decay_for_target(cfg["epsilon"], cfg["epsilon_min"], args.max_gen)
    return cfg


def run_case(args: argparse.Namespace, case: dict[str, Any], seed: int):
    inst = build_case_instance(args, case)
    start = time.time()
    result = run_qvns_nsga2(
        inst,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        seed=seed,
        verbose=not args.quiet,
        use_repair=True,
        **qvns_kwargs(args),
    )
    result["runtime"] = float(result.get("runtime", time.time() - start))
    return inst, result


def pareto_points(result: dict[str, Any], inst, case: dict[str, Any], group_type: str, seed: int) -> pd.DataFrame:
    rows = []
    for solution_id, dec in enumerate(result.get("pareto_dec", [])):
        objs, cv, parts = evaluate_solution(dec, inst, use_repair=True, collect_parts=True)
        row = {
            "group_type": group_type,
            "case_key": case["case_key"],
            "case_name": case["case_name"],
            "seed": int(seed),
            "solution_id": int(solution_id),
            "Z1_total_cost": float(objs[0]),
            "Z2_metro_service_ratio": float(-objs[1]),
            "CV": float(cv),
        }
        row.update(parts)
        rows.append(row)
    return pd.DataFrame(rows)


def compromise_index(result: dict[str, Any]) -> int:
    objs = np.asarray(result.get("pareto_objs", []), dtype=float)
    return choose_compromise(objs) if len(objs) else 0


def open_station_names(station_df: pd.DataFrame) -> str:
    if station_df.empty or "是否启用" not in station_df.columns:
        return ""
    open_df = station_df[station_df["是否启用"].astype(float) > 0.5]
    return "；".join(open_df["地铁站名称"].astype(str).tolist())


def summarize_case(result: dict[str, Any], inst, case: dict[str, Any], group_type: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = pareto_points(result, inst, case, group_type, seed)
    feasible = points[points["CV"] <= 1e-9].copy()
    view = feasible if len(feasible) else points
    comp_idx = compromise_index(result)
    comp_dec = result["pareto_dec"][comp_idx]
    comp_objs, comp_cv, comp_parts = evaluate_solution(comp_dec, inst, use_repair=True, collect_parts=True)
    comp_assign, comp_station, comp_line = solution_to_tables(comp_dec, inst, use_repair=True)
    best_cost = view.loc[view["Z1_total_cost"].idxmin()] if len(view) else None
    best_service = view.loc[view["Z2_metro_service_ratio"].idxmax()] if len(view) else None
    summary = {
        "group_type": group_type,
        "case_key": case["case_key"],
        "case_name": case["case_name"],
        "seed": int(seed),
        "runtime_seconds": float(result.get("runtime", np.nan)),
        "pareto_size": int(len(points)),
        "feasible_pareto_size": int(len(feasible)),
        "unique_points": int(len(view[["Z1_total_cost", "Z2_metro_service_ratio"]].round(8).drop_duplicates())) if len(view) else 0,
        "min_cost": float(best_cost["Z1_total_cost"]) if best_cost is not None else np.nan,
        "max_service_ratio": float(best_service["Z2_metro_service_ratio"]) if best_service is not None else np.nan,
        "topsis_solution_id": int(comp_idx),
        "topsis_cost": float(comp_objs[0]),
        "topsis_service_ratio": float(-comp_objs[1]),
        "topsis_cv": float(comp_cv),
        "topsis_open_count": float(comp_parts.get("num_open", np.nan)),
        "topsis_metro_volume": float(comp_parts.get("metro_volume", np.nan)),
        "topsis_direct_volume": float(comp_parts.get("direct_volume", np.nan)),
        "topsis_total_carbon_tCO2": float(comp_parts.get("total_carbon_tCO2", np.nan)),
        "topsis_avg_feeder_distance": float(comp_parts.get("avg_feeder_distance", np.nan)),
        "topsis_space_coverage_4km_ratio": float(comp_parts.get("space_coverage_4km_ratio", np.nan)),
        "topsis_fulfillment_ratio_within_4km": float(comp_parts.get("fulfillment_ratio_within_4km", np.nan)),
        "topsis_used_train_trips": float(comp_parts.get("used_train_trips_total", np.nan)),
        "topsis_open_stations": open_station_names(comp_station),
    }
    for key in ("lambda", "radius_km", "source_share"):
        if key in case:
            summary[key] = str(case[key]) if key == "source_share" else case[key]
    return summary, points, comp_assign, comp_station, comp_line


def zero_direct_dec(inst) -> np.ndarray:
    return np.zeros(inst.K + inst.I * inst.J, dtype=float)


def case_parameter_fields(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "lambda": case.get("lambda", np.nan),
        "radius_km": case.get("radius_km", np.nan),
        "source_share": str(case["source_share"]) if "source_share" in case else "",
    }


def build_display_cases() -> list[dict[str, Any]]:
    low, base, high = [dict(case) for case in DEMAND_CASES]
    r3, r4, r5 = [dict(case) for case in RADIUS_CASES]
    s5_5, s6_4, s7_3, s8_2, s9_1 = [dict(case) for case in SOURCE_SHARE_CASES]
    return [
        {**low, "group_type": "demand_scale", "canonical_key": "low", "seed_offset": 1000},
        {
            **base,
            "group_type": "demand_scale",
            "canonical_key": "base",
            "seed_offset": 1100,
            "sync_note_name": "BASE_REPLACED_BY_REPRESENTATIVE_SOLUTION.md",
            "sync_note": "本目录与 representative_solution/ 使用同一组 p100-g300-runs10 结果，按 HV 最大的代表运行导出，因此与主方案完全一致。",
        },
        {**high, "group_type": "demand_scale", "canonical_key": "high", "seed_offset": 1200},
        {**r3, "group_type": "service_radius", "canonical_key": "r3", "seed_offset": 2000},
        {
            **r4,
            "group_type": "service_radius",
            "canonical_key": "base",
            "is_alias_case": True,
            "sync_note_name": "SYNCED_WITH_REPRESENTATIVE_SOLUTION.md",
            "sync_note": "本目录与 representative_solution/、demand_scale/base/ 共用同一组 p100-g300-runs10 代表运行结果，以保证基准情景口径一致。",
        },
        {**r5, "group_type": "service_radius", "canonical_key": "r5", "seed_offset": 2200},
        {**s5_5, "group_type": "source_share", "canonical_key": "s5_5", "seed_offset": 3000},
        {**s6_4, "group_type": "source_share", "canonical_key": "s6_4", "seed_offset": 3100},
        {**s7_3, "group_type": "source_share", "canonical_key": "s7_3", "seed_offset": 3200},
        {
            **s8_2,
            "group_type": "source_share",
            "canonical_key": "base",
            "is_alias_case": True,
            "sync_note_name": "SYNCED_WITH_REPRESENTATIVE_SOLUTION.md",
            "sync_note": "本目录与 representative_solution/、demand_scale/base/ 共用同一组 p100-g300-runs10 代表运行结果，以保证基准情景口径一致。",
        },
        {**s9_1, "group_type": "source_share", "canonical_key": "s9_1", "seed_offset": 3400},
    ]


def build_canonical_cases(display_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered: dict[str, dict[str, Any]] = {}
    for case in display_cases:
        ordered.setdefault(str(case["canonical_key"]), dict(case))
    return ordered


def scenario_metrics(label: str, dec: np.ndarray, inst) -> dict[str, Any]:
    objs, cv, parts = evaluate_solution(dec, inst, use_repair=False, collect_parts=True)
    row = {"scenario_en": label, "total_cost_yuan_day": float(objs[0]), "metro_service_ratio": float(-objs[1]), "CV": float(cv)}
    for key in [
        "Z1_unit_cost", "metro_volume", "direct_volume", "road_cost", "fixed_cost", "metro_cost",
        "feeder_cost", "transfer_cost", "inout_cost", "train_organization_cost", "carbon_cost",
        "total_carbon_tCO2", "unit_carbon_kg_per_piece", "num_open", "used_train_trips_total",
        "avg_feeder_distance",
    ]:
        row[key] = float(parts.get(key, np.nan))
    return row


def add_quality_indicators_for_case(run_df: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()
    if points_df.empty:
        out = run_df.copy()
        out["HV"] = 0.0
        out["Spacing"] = 0.0
        out["IGD"] = np.nan
        out["Spread"] = np.nan
        return out
    cost_min, cost_max = points_df["Z1_total_cost"].min(), points_df["Z1_total_cost"].max()
    srv_min, srv_max = points_df["Z2_metro_service_ratio"].min(), points_df["Z2_metro_service_ratio"].max()
    all_normalized = np.column_stack([
        (points_df["Z1_total_cost"].to_numpy() - cost_min) / (cost_max - cost_min + 1e-12),
        (srv_max - points_df["Z2_metro_service_ratio"].to_numpy()) / (srv_max - srv_min + 1e-12),
    ])
    reference_front = all_normalized[nondominated(all_normalized)] if len(all_normalized) else np.empty((0, 2))
    rows: list[dict[str, Any]] = []
    for _, row in run_df.iterrows():
        pts = points_df[points_df["run_id"].astype(int) == int(row["run_id"])]
        out_row = row.to_dict()
        if pts.empty:
            out_row["HV"] = 0.0
            out_row["Spacing"] = 0.0
            out_row["IGD"] = np.nan
            out_row["Spread"] = np.nan
        else:
            normalized = np.column_stack([
                (pts["Z1_total_cost"].to_numpy() - cost_min) / (cost_max - cost_min + 1e-12),
                (srv_max - pts["Z2_metro_service_ratio"].to_numpy()) / (srv_max - srv_min + 1e-12),
            ])
            out_row["HV"] = hv_2d(normalized)
            out_row["Spacing"] = spacing(normalized)
            out_row["IGD"] = igd(normalized, reference_front)
            out_row["Spread"] = spread_delta(normalized, reference_front)
        rows.append(out_row)
    return pd.DataFrame(rows)


def select_representative_run(run_df: pd.DataFrame) -> dict[str, Any]:
    if run_df.empty:
        raise RuntimeError("No run metrics available for representative-run selection")
    ranking = run_df.copy()
    ranking["HV"] = ranking["HV"].fillna(-np.inf)
    ranking["IGD"] = ranking["IGD"].fillna(np.inf)
    ranking["Spread"] = ranking["Spread"].fillna(np.inf)
    ranking["unique_points"] = ranking["unique_points"].fillna(-np.inf)
    ranking["topsis_cost"] = ranking["topsis_cost"].fillna(np.inf)
    ranking["topsis_service_ratio"] = ranking["topsis_service_ratio"].fillna(-np.inf)
    ranking["runtime_seconds"] = ranking["runtime_seconds"].fillna(np.inf)
    ranking = ranking.sort_values(
        ["HV", "IGD", "Spread", "unique_points", "topsis_cost", "topsis_service_ratio", "runtime_seconds", "run_id"],
        ascending=[False, True, True, False, True, False, True, True],
    )
    return ranking.iloc[0].to_dict()


def aggregate_run_metrics(run_df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"runs": int(len(run_df))}
    for col, with_std in [
        ("runtime_seconds", True),
        ("pareto_size", True),
        ("unique_points", True),
        ("min_cost", True),
        ("max_service_ratio", True),
        ("topsis_cost", True),
        ("topsis_service_ratio", True),
        ("HV", True),
        ("Spacing", False),
        ("IGD", True),
        ("Spread", True),
    ]:
        stats[f"{col}_mean"] = float(run_df[col].mean()) if col in run_df else np.nan
        if with_std:
            stats[f"{col}_std"] = float(run_df[col].std(ddof=1)) if len(run_df) > 1 and col in run_df else 0.0
    return stats


def decorate_run_metrics(run_df: pd.DataFrame, case: dict[str, Any]) -> pd.DataFrame:
    out = run_df.copy()
    out["group_type"] = case["group_type"]
    out["case_key"] = case["case_key"]
    out["case_name"] = case["case_name"]
    out["case_origin_key"] = case["canonical_key"]
    out["is_alias_case"] = bool(case.get("is_alias_case", False))
    for key, value in case_parameter_fields(case).items():
        out[key] = value
    return out


def decorate_points(points_df: pd.DataFrame, case: dict[str, Any]) -> pd.DataFrame:
    out = points_df.copy()
    out["group_type"] = case["group_type"]
    out["case_key"] = case["case_key"]
    out["case_name"] = case["case_name"]
    out["case_origin_key"] = case["canonical_key"]
    out["is_alias_case"] = bool(case.get("is_alias_case", False))
    for key, value in case_parameter_fields(case).items():
        out[key] = value
    return out


def sync_root_shortcuts(rep_dir: Path) -> None:
    mapping = {
        rep_dir / "qvns_nsga2_station_load.csv": RESULTS_ROOT / "qvns_nsga2_station_load.csv",
        rep_dir / "qvns_nsga2_compromise_assignments.csv": RESULTS_ROOT / "qvns_nsga2_compromise_assignments.csv",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def _run_case_once(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    args = argparse.Namespace(**task["args"])
    case = dict(task["case"])
    run_id = int(task["run_id"])
    seed = int(task["seed"])
    inst, result = run_case(args, case, seed)
    row, points = summarize_result(
        algorithm="QVNS-NSGA-II-Q11C",
        model_setting=str(args.model_setting),
        run_id=run_id,
        seed=seed,
        result=result,
        runtime=float(result.get("runtime", np.nan)),
        group_type=str(case["group_type"]),
        config=qvns_kwargs(args),
    )
    row.update({"group_type": case["group_type"], "case_key": case["case_key"], "case_name": case["case_name"]})
    for key, value in case_parameter_fields(case).items():
        row[key] = value
    if len(points):
        points = points.copy()
        points["group_type"] = case["group_type"]
        points["case_key"] = case["case_key"]
        points["case_name"] = case["case_name"]
        for key, value in case_parameter_fields(case).items():
            points[key] = value
    return row, points


def run_one_canonical_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    task_args = {
        "pop_size": args.pop_size,
        "max_gen": args.max_gen,
        "runs": args.runs,
        "workers": args.workers,
        "top_n": args.top_n,
        "model_setting": args.model_setting,
        "seed": args.seed,
        "quiet": args.quiet,
    }
    tasks = [
        {
            "args": task_args,
            "case": case,
            "run_id": run_id,
            "seed": int(args.seed + int(case.get("seed_offset", 0)) + run_id),
        }
        for run_id in range(1, args.runs + 1)
    ]
    rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    workers = max(1, min(int(args.workers), len(tasks)))
    print(f"[RUNS] canonical={case['canonical_key']} case={case['case_key']} runs={args.runs} workers={workers}", flush=True)
    if workers == 1:
        for task in tasks:
            row, points = _run_case_once(task)
            rows.append(row)
            if not points.empty:
                point_frames.append(points)
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_case_once, task): task for task in tasks}
            for future in futures.as_completed(future_map):
                task = future_map[future]
                row, points = future.result()
                rows.append(row)
                if not points.empty:
                    point_frames.append(points)
                print(f"  done run_id={task['run_id']} seed={task['seed']} runtime={row['runtime_seconds']:.2f}s", flush=True)
    run_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    if not points_df.empty:
        sort_cols = [col for col in ["run_id", "Z1_total_cost", "Z2_metro_service_ratio"] if col in points_df.columns]
        points_df = points_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else points_df.reset_index(drop=True)
    run_df = add_quality_indicators_for_case(run_df, points_df)
    selected = select_representative_run(run_df)
    rep_seed = int(selected["seed"])
    rep_run_id = int(selected["run_id"])
    inst, result = run_case(args, case, rep_seed)
    summary, rep_points, assign, station, line = summarize_case(result, inst, case, case["group_type"], rep_seed)
    summary["runtime_seconds"] = float(selected["runtime_seconds"])
    summary["selection_rule"] = "max_HV_then_TOPSIS"
    summary["representative_run_id"] = rep_run_id
    summary["representative_seed"] = rep_seed
    summary["representative_HV"] = float(selected.get("HV", np.nan))
    summary["representative_IGD"] = float(selected.get("IGD", np.nan))
    summary["representative_Spread"] = float(selected.get("Spread", np.nan))
    summary["representative_Spacing"] = float(selected.get("Spacing", np.nan))
    summary["representative_unique_points"] = int(selected.get("unique_points", summary.get("unique_points", 0)))
    summary["case_origin_key"] = case["canonical_key"]
    summary.update(aggregate_run_metrics(run_df))
    print(f"[SELECT] canonical={case['canonical_key']} rep_run={rep_run_id} seed={rep_seed} HV={summary['representative_HV']:.6f}", flush=True)
    return {
        "case": case,
        "run_metrics_df": run_df,
        "all_points_df": points_df,
        "summary": summary,
        "representative_points_df": rep_points,
        "assign_df": assign,
        "station_df": station,
        "line_df": line,
        "inst": inst,
        "result": result,
    }


def rebuild_representative(args: argparse.Namespace, bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rep_dir = RESULT_DIR / "representative_solution"
    fig_dir = RESULT_DIR / "representative_solution_figures"
    clean_dir(rep_dir)
    clean_dir(fig_dir)

    inst = bundle["inst"]
    result = bundle["result"]
    summary = bundle["summary"]
    run_df = bundle["run_metrics_df"]
    all_points_df = bundle["all_points_df"]
    representative_points_df = bundle["representative_points_df"]
    comp_assign = bundle["assign_df"]
    comp_station = bundle["station_df"]
    comp_line = bundle["line_df"]

    export_qvns_nsga2_results(result, inst, str(rep_dir), use_repair=True, skip_plots=True)
    plot_compromise_map(comp_assign, comp_station, inst, str(fig_dir))

    aliases = {
        "qvns_nsga2_pareto.csv": "q11_representative_pareto.csv",
        "qvns_nsga2_compromise_assignments.csv": "q11_representative_assignments.csv",
        "qvns_nsga2_station_load.csv": "q11_representative_station_load.csv",
        "qvns_nsga2_line_load.csv": "q11_representative_line_load.csv",
    }
    for src_name, dst_name in aliases.items():
        src = rep_dir / src_name
        if src.exists():
            shutil.copy2(src, rep_dir / dst_name)

    pd.DataFrame([summary]).to_csv(rep_dir / "representative_summary.csv", index=False, encoding="utf-8-sig")
    representative_points_df.to_csv(rep_dir / "representative_pareto_points.csv", index=False, encoding="utf-8-sig")
    all_points_df.to_csv(rep_dir / "representative_all_runs_pareto_points.csv", index=False, encoding="utf-8-sig")
    run_df.to_csv(rep_dir / "representative_run_metrics.csv", index=False, encoding="utf-8-sig")

    feasible = representative_points_df[representative_points_df["CV"] <= 1e-9]
    view = feasible if len(feasible) else representative_points_df
    min_idx = int(view.loc[view["Z1_total_cost"].idxmin(), "solution_id"]) if len(view) else 0
    max_idx = int(view.loc[view["Z2_metro_service_ratio"].idxmax(), "solution_id"]) if len(view) else 0
    comp_idx = int(summary["topsis_solution_id"])
    selected = {
        "all_road": zero_direct_dec(inst),
        "min_cost": result["pareto_dec"][min_idx],
        "topsis_compromise": result["pareto_dec"][comp_idx],
        "max_service": result["pareto_dec"][max_idx],
    }
    comparison_rows = [scenario_metrics(label, dec, inst) for label, dec in selected.items()]
    comp_df = pd.DataFrame(comparison_rows)
    baseline = comp_df.iloc[0]
    for col, base_col in [
        ("total_cost_change_pct", "total_cost_yuan_day"),
        ("road_cost_change_pct", "road_cost"),
        ("carbon_change_pct", "total_carbon_tCO2"),
        ("direct_volume_change_pct", "direct_volume"),
    ]:
        denom = float(baseline[base_col]) if abs(float(baseline[base_col])) > 1e-12 else np.nan
        comp_df[col] = (comp_df[base_col].astype(float) - float(baseline[base_col])) / denom * 100.0
    comp_df.to_csv(RESULT_DIR / "road_baseline_solution_comparison_en.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(RESULT_DIR / "road_baseline_solution_comparison.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(RESULT_DIR / "cost_diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(RESULT_DIR / "cost_increment_vs_road.csv", index=False, encoding="utf-8-sig")
    comp_df[comp_df["scenario_en"].isin(["all_road", "topsis_compromise"])].to_csv(
        RESULT_DIR / "all_road_vs_qvns_compromise.csv", index=False, encoding="utf-8-sig"
    )
    write_road_baseline_figure(comp_df)

    best_params = pd.read_csv(rep_dir / "qvns_nsga2_best_params.csv", encoding="utf-8-sig").iloc[0].to_dict()
    metadata = {
        "representative_run_id": int(summary["representative_run_id"]),
        "representative_seed": int(summary["representative_seed"]),
        "selection_rule": summary["selection_rule"],
        "runs": int(summary["runs"]),
        "compromise_solution_id": int(comp_idx),
        "compromise_open_stations": open_station_names(comp_station),
        "compromise_cost": float(comp_df.loc[comp_df["scenario_en"] == "topsis_compromise", "total_cost_yuan_day"].iloc[0]),
        "compromise_service_ratio": float(comp_df.loc[comp_df["scenario_en"] == "topsis_compromise", "metro_service_ratio"].iloc[0]),
        "representative_HV": float(summary["representative_HV"]),
        "representative_IGD": float(summary["representative_IGD"]),
        "representative_Spread": float(summary["representative_Spread"]),
        "representative_Spacing": float(summary["representative_Spacing"]),
        "HV_mean": float(summary["HV_mean"]),
        "HV_std": float(summary["HV_std"]),
        "runtime_seconds_mean": float(summary["runtime_seconds_mean"]),
        "runtime_seconds_std": float(summary["runtime_seconds_std"]),
        "pop_size": int(args.pop_size),
        "max_gen": int(args.max_gen),
    }
    pd.DataFrame([{**metadata, **best_params}]).to_csv(RESULT_DIR / "current_q11_config_metadata.csv", index=False, encoding="utf-8-sig")
    sync_root_shortcuts(rep_dir)
    return comp_df, metadata


def write_road_baseline_figure(comp_df: pd.DataFrame) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    out_dir = RESULT_DIR / "figures_publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in fm.fontManager.ttflist}
    font_name = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun", "DejaVu Sans"] if f in fonts), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
    })
    labels = {
        "all_road": "全公路基准",
        "min_cost": "最低成本方案",
        "topsis_compromise": "TOPSIS折中方案",
        "max_service": "最高覆盖率方案",
    }
    df = comp_df.copy()
    df["label"] = df["scenario_en"].map(labels)
    colors = ["#8B8B8B", "#4C78A8", "#F58518", "#54A24B"]
    panels = [
        ("total_cost_yuan_day", "综合成本（万元/日）", 10000.0, "a  综合成本对比", "{:.2f}"),
        ("metro_service_ratio", "地铁服务比例（%）", 0.01, "b  地铁服务比例", "{:.1f}"),
        ("total_carbon_tCO2", "碳排放（tCO2/日）", 1.0, "c  碳排放水平", "{:.3f}"),
        ("direct_volume", "公路直送量（万件/日）", 10000.0, "d  公路直送量", "{:.2f}"),
    ]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), dpi=220)
    for ax, (col, ylabel, scale, title, fmt) in zip(axes.ravel(), panels):
        vals = df[col].astype(float).to_numpy() / scale
        bars = ax.bar(x, vals, color=colors, edgecolor="#3A3A3A", linewidth=0.55, width=0.62)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=18, ha="right")
        ax.grid(axis="y", color="#E7E9ED", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(vals) if len(vals) else 1
        ax.set_ylim(0, ymax * 1.22 + (0.02 if ymax == 0 else 0))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025, fmt.format(val), ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("全公路基准与地铁联运方案效果对比", fontsize=12, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.015, "注：综合成本、公路直送量分别按万元/日、万件/日展示；TOPSIS折中方案为文章主方案。", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95], h_pad=1.0, w_pad=1.0)
    base_name = out_dir / "fig5_pub_e_road_baseline_comparison"
    fig.savefig(base_name.with_suffix(".png"), dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(base_name.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base_name.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_case_outputs(root: Path, case: dict[str, Any], summary: dict[str, Any], representative_points: pd.DataFrame, all_points: pd.DataFrame, run_metrics: pd.DataFrame, assign: pd.DataFrame, station: pd.DataFrame, line: pd.DataFrame, inst) -> None:
    case_dir = root / case["group_type"] / case["case_key"]
    clean_dir(case_dir)
    pd.DataFrame([summary]).to_csv(case_dir / "summary.csv", index=False, encoding="utf-8-sig")
    representative_points.to_csv(case_dir / "pareto_points.csv", index=False, encoding="utf-8-sig")
    all_points.to_csv(case_dir / "all_runs_pareto_points.csv", index=False, encoding="utf-8-sig")
    run_metrics.to_csv(case_dir / "run_metrics.csv", index=False, encoding="utf-8-sig")
    assign.to_csv(case_dir / "compromise_assignments.csv", index=False, encoding="utf-8-sig")
    station.to_csv(case_dir / "compromise_station_load.csv", index=False, encoding="utf-8-sig")
    line.to_csv(case_dir / "compromise_line_load.csv", index=False, encoding="utf-8-sig")
    plot_compromise_map(assign, station, inst, str(case_dir))
    if case.get("sync_note"):
        (case_dir / case["sync_note_name"]).write_text(str(case["sync_note"]) + "\n", encoding="utf-8")


def rebuild_sensitivity(args: argparse.Namespace, bundles: dict[str, dict[str, Any]]) -> None:
    sens_root = RESULT_DIR / "sensitivity_current_q11c"
    clean_dir(sens_root)
    all_summaries: list[dict[str, Any]] = []
    all_points: list[pd.DataFrame] = []
    all_run_metrics: list[pd.DataFrame] = []
    for case in build_display_cases():
        bundle = bundles[str(case["canonical_key"])]
        summary = dict(bundle["summary"])
        summary["group_type"] = case["group_type"]
        summary["case_key"] = case["case_key"]
        summary["case_name"] = case["case_name"]
        summary["case_origin_key"] = case["canonical_key"]
        summary["is_alias_case"] = bool(case.get("is_alias_case", False))
        for key, value in case_parameter_fields(case).items():
            summary[key] = value
        representative_points = decorate_points(bundle["representative_points_df"], case)
        all_case_points = decorate_points(bundle["all_points_df"], case)
        run_metrics = decorate_run_metrics(bundle["run_metrics_df"], case)
        save_case_outputs(
            sens_root,
            case,
            summary,
            representative_points,
            all_case_points,
            run_metrics,
            bundle["assign_df"],
            bundle["station_df"],
            bundle["line_df"],
            bundle["inst"],
        )
        all_summaries.append(summary)
        all_points.append(all_case_points)
        all_run_metrics.append(run_metrics)
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULT_DIR / "q11_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(sens_root / "q11_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    if all_points:
        point_df = pd.concat(all_points, ignore_index=True)
        point_df.to_csv(RESULT_DIR / "q11_sensitivity_pareto_points.csv", index=False, encoding="utf-8-sig")
        point_df.to_csv(sens_root / "q11_sensitivity_pareto_points.csv", index=False, encoding="utf-8-sig")
    if all_run_metrics:
        run_df = pd.concat(all_run_metrics, ignore_index=True)
        run_df.to_csv(RESULT_DIR / "q11_sensitivity_run_metrics.csv", index=False, encoding="utf-8-sig")
        run_df.to_csv(sens_root / "q11_sensitivity_run_metrics.csv", index=False, encoding="utf-8-sig")


def write_run_note(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    note = RESULT_DIR / "当前Q11C默认配置重建说明.md"
    note.write_text(
        "# 当前 QVNS-NSGA-II 默认配置重建说明\n\n"
        f"- 运行时间：{pd.Timestamp.now()}\n"
        f"- 种群规模：{args.pop_size}\n"
        f"- 最大迭代次数：{args.max_gen}\n"
        f"- 每个情景独立运行次数：{args.runs}\n"
        f"- 基准随机种子：{args.seed}\n"
        "- 统一口径：每个情景按 HV 最大的代表运行导出；折中解继续使用 Pareto 前沿上的 TOPSIS 规则。\n"
        f"- 代表运行 run_id：{metadata.get('representative_run_id')}\n"
        f"- 代表运行 seed：{metadata.get('representative_seed')}\n"
        f"- 代表性折中解 solution_id：{metadata.get('compromise_solution_id')}\n"
        f"- 代表性折中解启用站点：{metadata.get('compromise_open_stations')}\n"
        f"- 代表性折中解综合成本：{metadata.get('compromise_cost'):.6f}\n"
        f"- 代表性折中解地铁服务比例：{metadata.get('compromise_service_ratio'):.6f}\n"
        f"- 代表运行 HV：{metadata.get('representative_HV'):.6f}\n"
        f"- 10 次运行 HV 均值：{metadata.get('HV_mean'):.6f}\n"
        f"- 10 次运行平均耗时：{metadata.get('runtime_seconds_mean'):.6f} s\n\n"
        "## 当前默认参数\n\n"
        "```text\n"
        + "\n".join(f"{k}: {v}" for k, v in qvns_kwargs(args).items())
        + "\n```\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundles: dict[str, dict[str, Any]] = {}
    for canonical_key, case in build_canonical_cases(build_display_cases()).items():
        bundles[canonical_key] = run_one_canonical_case(args, case)
    print("[REBUILD] representative solution")
    _, metadata = rebuild_representative(args, bundles["base"])
    print("[REBUILD] sensitivity scenarios")
    rebuild_sensitivity(args, bundles)
    write_run_note(args, metadata)
    print(f"\nDone. Updated: {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
