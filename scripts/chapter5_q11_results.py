# -*- coding: utf-8 -*-
"""Generate Chapter 5 Q11 tables and sensitivity results.

The script reuses the thesis model and QVNS-NSGA-II implementation.  It is meant
for reproducible chapter writing rather than algorithm development.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (CODE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import (  # noqa: E402
    build_metro_logistics_instance,
    choose_compromise,
    export_qvns_nsga2_results,
    run_qvns_nsga2,
    solution_to_tables,
)
from run_qvns_ablation_experiment import (  # noqa: E402
    _epsilon_decay_for_target,
    summarize_result,
)


def q11_config(pop_size: int, max_gen: int) -> dict:
    return {
        "alpha": 0.15,
        "gamma": 0.88,
        "epsilon": 0.30,
        "epsilon_min": 0.010,
        "epsilon_decay": _epsilon_decay_for_target(0.30, 0.010, max_gen),
        "vns_interval": 0,
        "vns_mode": "adaptive",
        "vns_elite_rate": 0.12,
        "vns_trials_per_neighborhood": 2,
        "vns_max_neighborhood": 3,
        "archive_size": max(pop_size, int(round(2.5 * pop_size))),
        "archive_boundary_keep_rate": 0.20,
        "vns_candidate_strategy": "mixed",
        "stagnation_window": 10,
        "hv_improvement_threshold": 0.0025,
        "min_vns_gap": 5,
        "q_learning_enabled": True,
        "q_action_apply_rate": 0.25,
        "q_action_profile": "safe",
        "safeguard": True,
        "safeguard_threshold": -0.15,
        "archive_feedback_interval": 8,
        "archive_feedback_rate": 0.15,
        "tuning_group": "Q11C",
        "tuning_goal": "q11_branch_fast_integrated",
    }


def export_base_tables(output_dir: Path) -> None:
    inst = build_metro_logistics_instance("base", top_n=17, model_setting="no_budget_range")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = pd.DataFrame(
        [
            ("候选转运点数量", inst.K, "个"),
            ("顺丰需求点数量", inst.J, "个"),
            ("物流园区数量", inst.I, "个"),
            ("基准需求总量", float(inst.Q_j.sum()), "件/日"),
            ("需求点平均需求", float(inst.Q_j.mean()), "件/日"),
            ("需求点需求标准差", float(inst.Q_j.std()), "件/日"),
            ("需求点最小需求", float(inst.Q_j.min()), "件/日"),
            ("需求点最大需求", float(inst.Q_j.max()), "件/日"),
            ("候选点总处理能力", float(inst.station_capacity.sum()), "件/日"),
            ("候选点平均固定成本", float(inst.fixed_cost.mean()), "元/日"),
            ("候选点至需求点平均道路距离", float(inst.d_kj_road.mean()), "km"),
            ("候选点至需求点4km可达关系占比", float((inst.d_kj_road <= inst.D_k_max[:, None]).mean() * 100), "%"),
            ("园区至需求点平均道路距离", float(inst.d_ij_road.mean()), "km"),
            ("园区接入地铁平均道路距离", float(inst.d_is_road.mean()), "km"),
            ("园区至候选点平均地铁距离", float(inst.d_metro.mean()), "km"),
            ("平均换乘次数", float(inst.transfer_count.mean()), "次"),
        ],
        columns=["指标", "数值", "单位"],
    )
    base_summary.to_csv(output_dir / "table_5_base_data_summary.csv", index=False, encoding="utf-8-sig")

    candidates = inst.candidate_df.copy()
    candidates["站点类型集合"] = inst.station_type_labels
    candidates["处理能力_件日"] = inst.station_capacity
    candidates["固定运营成本_元日"] = inst.fixed_cost
    candidates["重要度得分"] = inst.station_importance
    candidates.to_csv(output_dir / "table_5_candidate_transfer_points.csv", index=False, encoding="utf-8-sig")

    demand = inst.demand_df.copy()
    demand["需求量_件日"] = inst.Q_j
    demand.sort_values("需求量_件日", ascending=False).to_csv(
        output_dir / "table_5_demand_points_ranked.csv", index=False, encoding="utf-8-sig"
    )

    line_capacity = pd.DataFrame({"线路": list(inst.line_capacity.keys()), "单趟容量_件": list(inst.line_capacity.values())})
    line_capacity["平峰可用班次"] = inst.available_train_trips
    line_capacity["平峰总容量_件日"] = line_capacity["单趟容量_件"] * inst.available_train_trips
    line_capacity["班次组织成本_元趟"] = line_capacity["线路"].map(inst.train_organization_cost)
    line_capacity.to_csv(output_dir / "table_5_line_capacity.csv", index=False, encoding="utf-8-sig")

    source = pd.DataFrame(
        {
            "物流园区": list(inst.logistics_df.index)[: inst.I],
            "货源比例": inst.source_share,
            "接入道路距离_km": inst.d_is_road,
        }
    )
    source.to_csv(output_dir / "table_5_source_info.csv", index=False, encoding="utf-8-sig")


def apply_case(inst, case_type: str, factor: float | None, source_share: tuple[float, float] | None) -> None:
    if case_type == "cost" and factor is not None:
        inst.fixed_cost = inst.fixed_cost * factor
        inst.road_operating_cost *= factor
        inst.metro_electric_cost *= factor
        inst.feeder_cost *= factor
        inst.transfer_cost *= factor
        inst.inout_cost *= factor
        if inst.train_organization_cost:
            inst.train_organization_cost = {line: value * factor for line, value in inst.train_organization_cost.items()}
    if case_type == "source" and source_share is not None:
        inst.source_share = np.array(source_share, dtype=float)


def run_q11_case(case_name: str, scenario: str, case_type: str, factor: float | None, source_share, args) -> tuple[dict, pd.DataFrame, dict]:
    inst = build_metro_logistics_instance(scenario=scenario, top_n=17, model_setting="no_budget_range")
    apply_case(inst, case_type, factor, source_share)
    config = q11_config(args.pop_size, args.max_gen)
    started = time.time()
    result = run_qvns_nsga2(
        inst,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        seed=args.seed,
        verbose=not args.quiet,
        use_repair=True,
        pc=0.90,
        **{key: value for key, value in config.items() if key not in {"tuning_group", "tuning_goal"}},
    )
    runtime = float(result.get("runtime", time.time() - started))
    row, points = summarize_result(
        "QVNS-NSGA-II-Q11C",
        "no_budget_range",
        1,
        args.seed,
        result,
        runtime,
        group_type=case_type,
        config=config,
    )
    row.update({"case_name": case_name, "demand_scenario": scenario, "factor": factor, "source_share": str(source_share)})
    return row, points, {"inst": inst, "result": result, "config": config}


def export_representative_solution(case_bundle: dict, output_dir: Path) -> None:
    inst = case_bundle["inst"]
    result = case_bundle["result"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pareto_df = export_qvns_nsga2_results(result, inst, str(output_dir), use_repair=True, skip_plots=True)
    compromise_idx = choose_compromise(result["pareto_objs"])
    assignments, stations, lines = solution_to_tables(result["pareto_dec"][compromise_idx], inst)
    pareto_df.to_csv(output_dir / "q11_representative_pareto.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(output_dir / "q11_representative_assignments.csv", index=False, encoding="utf-8-sig")
    stations.to_csv(output_dir / "q11_representative_station_load.csv", index=False, encoding="utf-8-sig")
    lines.to_csv(output_dir / "q11_representative_line_load.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Chapter 5 Q11 result tables.")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chapter5_q11_results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_base_tables(args.output_dir / "base_tables")

    cases = [
        ("需求规模-低需求", "low", "demand", None, None),
        ("需求规模-基准需求", "base", "demand", None, None),
        ("需求规模-高需求", "high", "demand", None, None),
        ("成本参数-降低20%", "base", "cost", 0.80, None),
        ("成本参数-基准", "base", "cost", 1.00, None),
        ("成本参数-提高20%", "base", "cost", 1.20, None),
        ("货源比例-双流西60%", "base", "source", None, (0.60, 0.40)),
        ("货源比例-基准80%", "base", "source", None, (0.80, 0.20)),
        ("货源比例-双流西90%", "base", "source", None, (0.90, 0.10)),
    ]

    rows = []
    point_frames = []
    representative = None
    for case_name, scenario, case_type, factor, source_share in cases:
        print(f"[Q11] Running {case_name}", flush=True)
        row, points, bundle = run_q11_case(case_name, scenario, case_type, factor, source_share, args)
        rows.append(row)
        if len(points):
            points = points.copy()
            points["case_name"] = case_name
            point_frames.append(points)
        if case_name == "需求规模-基准需求":
            representative = bundle

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "q11_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_csv(
            args.output_dir / "q11_sensitivity_pareto_points.csv", index=False, encoding="utf-8-sig"
        )
    if representative is not None:
        export_representative_solution(representative, args.output_dir / "representative_solution")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
