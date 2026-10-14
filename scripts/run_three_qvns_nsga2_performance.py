"""Compare NSGA-II, MOPSO, MOEA/D and unified Q11C QVNS-NSGA-II.

The three algorithms are run in parallel on the Chapter 5 model with
``pop_size=100`` and ``max_gen=300`` by default:

- NSGA-II baseline
- MOPSO
- MOEA/D
- QVNS-NSGA-II Q11C Chapter-5 reproduction settings, now also the main default
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (CODE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import build_metro_logistics_instance, run_nsga2, run_qvns_nsga2  # noqa: E402
from run_moo_comparison_experiment import run_moead, run_mopso  # noqa: E402
from run_qvns_ablation_experiment import (  # noqa: E402
    _epsilon_decay_for_target,
    add_quality_indicators,
    make_summary,
    summarize_result,
)


def q11c_config(pop_size: int, max_gen: int) -> dict[str, Any]:
    return {
        "pop_size": pop_size,
        "max_gen": max_gen,
        "pc": 0.90,
        "alpha": 0.15,
        "gamma": 0.88,
        "epsilon": 0.30,
        "epsilon_min": 0.010,
        "epsilon_decay": _epsilon_decay_for_target(0.30, 0.010, max_gen),
        "archive_size": max(pop_size, int(round(2.5 * pop_size))),
        "vns_interval": 0,
        "vns_mode": "adaptive",
        "vns_elite_rate": 0.12,
        "vns_trials_per_neighborhood": 2,
        "vns_max_neighborhood": 3,
        "archive_feedback_interval": 8,
        "archive_feedback_rate": 0.15,
        "q_action_apply_rate": 0.25,
        "q_action_profile": "safe",
        "q_learning_enabled": True,
        "safeguard": True,
        "safeguard_threshold": -0.15,
        "stagnation_window": 10,
        "hv_improvement_threshold": 0.0025,
        "min_vns_gap": 5,
        "archive_boundary_keep_rate": 0.20,
        "vns_candidate_strategy": "mixed",
        "tuning_group": "Q11C",
        "tuning_goal": "chapter5_reproduction_q11c",
    }


def run_algorithm(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    algorithm = task["algorithm"]
    seed = int(task["seed"])
    pop_size = int(task["pop_size"])
    max_gen = int(task["max_gen"])
    inst = build_metro_logistics_instance("base", top_n=17, model_setting="no_budget_range")
    started = time.time()
    if algorithm == "NSGA-II":
        result = run_nsga2(
            inst,
            pop_size=pop_size,
            max_gen=max_gen,
            seed=seed,
            verbose=False,
            use_repair=True,
            archive_size=pop_size,
        )
        config = {"pop_size": pop_size, "max_gen": max_gen, "archive_size": pop_size}
        group_type = "baseline"
    elif algorithm == "MOPSO":
        result = run_mopso(inst, pop_size=pop_size, max_gen=max_gen, seed=seed, verbose=False)
        config = {"pop_size": pop_size, "max_gen": max_gen}
        group_type = "algorithm_comparison"
    elif algorithm == "MOEA/D":
        result = run_moead(inst, pop_size=pop_size, max_gen=max_gen, seed=seed, verbose=False)
        config = {"pop_size": pop_size, "max_gen": max_gen}
        group_type = "algorithm_comparison"
    else:
        config = task["config"]
        kwargs = {key: value for key, value in config.items() if key not in {"tuning_group", "tuning_goal", "pop_size", "max_gen"}}
        result = run_qvns_nsga2(
            inst,
            pop_size=pop_size,
            max_gen=max_gen,
            seed=seed,
            verbose=False,
            use_repair=True,
            **kwargs,
        )
        group_type = "qvns_q11c_variant"
    runtime = float(result.get("runtime", time.time() - started))
    row, points = summarize_result(
        algorithm=algorithm,
        model_setting="no_budget_range",
        run_id=int(task["run_id"]),
        seed=seed,
        result=result,
        runtime=runtime,
        group_type=group_type,
        config=config,
    )
    row["pop_size"] = pop_size
    row["max_gen"] = max_gen
    return row, points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five algorithm variants in parallel.")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows in the output directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "five_algorithm_q11c_performance",
    )
    return parser.parse_args()


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], point_frames: list[pd.DataFrame]) -> pd.DataFrame:
    run_df = pd.DataFrame(rows).sort_values(["run_id", "algorithm"]).reset_index(drop=True)
    points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    run_df = add_quality_indicators(run_df, points_df) if len(points_df) else run_df
    summary_df = make_summary(run_df) if len(run_df) else pd.DataFrame()
    run_df.to_csv(output_dir / "five_algorithm_run_metrics.csv", index=False, encoding="utf-8-sig")
    points_df.to_csv(output_dir / "five_algorithm_pareto_points.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "five_algorithm_summary.csv", index=False, encoding="utf-8-sig")
    if len(summary_df):
        table = summary_df[[
            "algorithm",
            "HV_mean",
            "HV_std",
            "IGD_mean",
            "Spread_mean",
            "unique_points_mean",
            "min_cost_mean",
            "max_service_ratio_mean",
            "runtime_seconds_mean",
        ]].copy()
        table = table.rename(columns={
            "algorithm": "算法",
            "HV_mean": "HV均值",
            "HV_std": "HV标准差",
            "IGD_mean": "IGD均值",
            "Spread_mean": "Spread均值",
            "unique_points_mean": "唯一解数均值",
            "min_cost_mean": "最小成本均值/元日",
            "max_service_ratio_mean": "最大服务比例均值",
            "runtime_seconds_mean": "平均运行时间/s",
        })
        table.to_csv(output_dir / "five_algorithm_selected_metrics_table.csv", index=False, encoding="utf-8-sig")
        (output_dir / "five_algorithm_selected_metrics_table.md").write_text(
            table.to_markdown(index=False) + "\n", encoding="utf-8-sig"
        )
    return summary_df


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "five_algorithm_run_metrics.csv"
    points_path = args.output_dir / "five_algorithm_pareto_points.csv"
    rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    if args.resume and run_path.exists():
        existing = pd.read_csv(run_path, encoding="utf-8-sig")
        rows = existing.to_dict("records")
        completed = set(zip(existing["algorithm"].astype(str), existing["run_id"].astype(int)))
        if points_path.exists():
            point_frames.append(pd.read_csv(points_path, encoding="utf-8-sig"))
    tasks = []
    for run_id in range(1, args.runs + 1):
        base_seed = args.seed + run_id - 1
        candidates = [
            {"algorithm": "NSGA-II", "run_id": run_id, "seed": base_seed, "pop_size": args.pop_size, "max_gen": args.max_gen},
            {"algorithm": "MOPSO", "run_id": run_id, "seed": base_seed, "pop_size": args.pop_size, "max_gen": args.max_gen},
            {"algorithm": "MOEA/D", "run_id": run_id, "seed": base_seed, "pop_size": args.pop_size, "max_gen": args.max_gen},
            {
                "algorithm": "QVNS-NSGA-II-Q11C",
                "run_id": run_id,
                "seed": base_seed,
                "pop_size": args.pop_size,
                "max_gen": args.max_gen,
                "config": q11c_config(args.pop_size, args.max_gen),
            },
        ]
        tasks.extend(task for task in candidates if (task["algorithm"], int(task["run_id"])) not in completed)
    print(f"scheduled_tasks={len(tasks)} completed={len(completed)} workers={args.workers}", flush=True)
    with futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(run_algorithm, task): task for task in tasks}
        for future in futures.as_completed(future_map):
            task = future_map[future]
            row, points = future.result()
            print(f"done {task['algorithm']} run={task['run_id']} seed={task['seed']} runtime={row['runtime_seconds']:.2f}s", flush=True)
            rows.append(row)
            if len(points):
                point_frames.append(points)
            write_outputs(args.output_dir, rows, point_frames)
    summary_df = write_outputs(args.output_dir, rows, point_frames)
    print(f"output_dir={args.output_dir}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
