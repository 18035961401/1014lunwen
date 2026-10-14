# -*- coding: utf-8 -*-
"""Run MOPSO and MOEA/D comparison experiments for Chapter 5.

The implementation reuses the thesis model encoding, repair, objective
evaluation, constrained non-dominated sorting, and metric calculation already
used by NSGA-II and QVNS-NSGA-II.  The goal is a fair algorithm-level comparison
under the same metro-logistics location-allocation model rather than a generic
benchmark implementation.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
# 添加上级scripts目录到路径
PARENT_SCRIPTS_DIR = PROJECT_ROOT.parent / "scripts"
for import_dir in (CODE_DIR, SCRIPTS_DIR, PARENT_SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from main import (  # noqa: E402
    _normalize_model_setting,
    _select_by_fronts,
    build_metro_logistics_instance,
    crossover,
    crowding_distance,
    evaluate_population,
    fast_non_dominated_sort,
    initialize_population,
    mutate,
    normalize_solution,
    repair_solution,
    run_nsga2,
)

# 尝试导入ablation实验的函数，如果失败则使用简化版本
try:
    from run_qvns_ablation_experiment import (  # noqa: E402
        add_quality_indicators,
        make_summary,
        plot_metrics,
        plot_pareto,
        summarize_result,
    )
except ImportError:
    # 如果导入失败，定义简化版本
    def summarize_result(algorithm, model_setting, run_id, seed, result, runtime, group_type="algorithm_comparison", config=None):
        """简化的结果汇总函数"""
        pareto_objs = result.get("pareto_objs", np.array([]))
        row = {
            "algorithm": algorithm,
            "group_type": group_type,
            "model_setting": model_setting,
            "run_id": run_id,
            "seed": seed,
            "runtime_seconds": runtime,
            "pareto_size": len(pareto_objs),
        }
        pts = pd.DataFrame()
        if len(pareto_objs) > 0:
            pts = pd.DataFrame({
                "algorithm": algorithm,
                "run_id": run_id,
                "Z1_total_cost": pareto_objs[:, 0],
                "Z2_metro_service_ratio": -pareto_objs[:, 1],
            })
        return row, pts

    def add_quality_indicators(run_df, points_df):
        """简化的质量指标计算"""
        return run_df

    def make_summary(run_df):
        """简化的汇总统计"""
        return run_df.groupby("algorithm").agg({
            "runtime_seconds": ["mean", "std"],
            "pareto_size": ["mean", "std"],
        }).reset_index()

    def plot_pareto(points_df, output_dir):
        """简化的Pareto前沿绘图"""
        pass

    def plot_metrics(summary_df, output_dir):
        """简化的指标绘图"""
        pass


COMPARISON_ALGORITHMS = ("NSGA-II", "MOPSO", "MOEA/D", "QVNS-NSGA-II")
RUN_METRIC_COLUMNS = [
    "algorithm",
    "group_type",
    "model_setting",
    "run_id",
    "seed",
    "runtime_seconds",
    "pareto_size",
    "unique_points",
    "min_cost",
    "max_service_ratio",
    "topsis_cost",
    "topsis_service_ratio",
    "safeguard_rejects",
    "actual_vns_calls",
    "archive_final_size",
    "archive_unique_solutions",
    "HV",
    "Spacing",
    "IGD",
    "Spread",
]


def empty_run_metric_row(algorithm: str, run_id: int, seed: int, runtime_seconds: float = 0.0) -> dict:
    row = {column: np.nan for column in RUN_METRIC_COLUMNS}
    row.update(
        {
            "algorithm": algorithm,
            "group_type": "algorithm_comparison",
            "model_setting": "no_budget_range",
            "run_id": int(run_id),
            "seed": int(seed),
            "runtime_seconds": float(runtime_seconds),
            "safeguard_rejects": 0,
            "actual_vns_calls": 0,
            "archive_final_size": 0,
            "archive_unique_solutions": 0,
        }
    )
    return row


def _seed_everything(seed: int | None):
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))


def _archive_from_population(pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, archive_size: int):
    fronts, _ = fast_non_dominated_sort(objs, cvs)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= archive_size:
            selected.extend(front)
        else:
            cd = crowding_distance(objs, front)
            order = np.argsort(-cd)
            selected.extend([front[i] for i in order[: archive_size - len(selected)]])
            break
    if not selected:
        selected = np.argsort(cvs)[: max(1, archive_size)].tolist()
    selected_arr = np.array(selected, dtype=int)
    return pop[selected_arr].copy(), objs[selected_arr].copy(), cvs[selected_arr].copy()


def _pareto_result(name: str, pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, history: list[dict], runtime: float) -> dict:
    fronts, _ = fast_non_dominated_sort(objs, cvs)
    first = np.array(fronts[0] if fronts else np.arange(len(pop)), dtype=int)
    feasible = first[cvs[first] <= 1e-9]
    if len(feasible):
        first = feasible
    return {
        "algorithm": name,
        "pareto_dec": pop[first].copy(),
        "pareto_objs": objs[first].copy(),
        "pareto_cvs": cvs[first].copy(),
        "history": pd.DataFrame(history),
        "actual_vns_calls": 0,
        "archive_final_size": int(len(first)),
        "archive_unique_solutions": int(len(np.unique(np.round(objs[first], 8), axis=0))),
        "runtime": float(runtime),
    }


def _personal_best_update(current_objs, current_cvs, best_objs, best_cvs):
    cur_feas = current_cvs <= 1e-9
    best_feas = best_cvs <= 1e-9
    cur_better = np.zeros(len(current_objs), dtype=bool)
    cur_better |= cur_feas & (~best_feas)
    cur_better |= (~cur_feas) & (~best_feas) & (current_cvs < best_cvs)
    both = cur_feas & best_feas
    cur_better |= both & np.all(current_objs <= best_objs + 1e-12, axis=1) & np.any(current_objs < best_objs - 1e-12, axis=1)
    tie = both & (~cur_better) & (~(np.all(best_objs <= current_objs + 1e-12, axis=1) & np.any(best_objs < current_objs - 1e-12, axis=1)))
    cur_better |= tie & (np.random.rand(len(current_objs)) < 0.5)
    return cur_better


def _mix_discrete_particle(position: np.ndarray, pbest: np.ndarray, leader: np.ndarray, inst, inertia_keep: float, c1: float, c2: float):
    child = position.copy()
    n = len(child)
    pbest_mask = np.random.rand(n) < min(0.95, max(0.0, c1))
    leader_mask = np.random.rand(n) < min(0.95, max(0.0, c2))
    keep_mask = np.random.rand(n) < min(0.95, max(0.0, inertia_keep))
    child[~keep_mask & pbest_mask] = pbest[~keep_mask & pbest_mask]
    child[~keep_mask & leader_mask] = leader[~keep_mask & leader_mask]
    child = mutate(child, inst)
    return repair_solution(child, inst)


def run_mopso(
    inst,
    pop_size: int = 100,
    max_gen: int = 300,
    seed: int | None = None,
    archive_size: int | None = None,
    verbose: bool = False,
):
    """Discrete MOPSO adapted to the thesis chromosome encoding."""
    _seed_everything(seed)
    archive_size = int(archive_size or 2 * pop_size)
    started = time.time()
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    pbest = pop.copy()
    pbest_objs = objs.copy()
    pbest_cvs = cvs.copy()
    archive_pop, archive_objs, archive_cvs = _archive_from_population(pop, objs, cvs, archive_size)
    history = []
    for gen in range(max_gen):
        inertia = 0.70 - 0.35 * (gen / max(1, max_gen - 1))
        c1 = 0.35 + 0.15 * (1 - gen / max(1, max_gen - 1))
        c2 = 0.35 + 0.15 * (gen / max(1, max_gen - 1))
        fronts, _ = fast_non_dominated_sort(archive_objs, archive_cvs)
        leader_front = np.array(fronts[0] if fronts else np.arange(len(archive_pop)), dtype=int)
        if len(leader_front) == 0:
            leader_front = np.arange(len(archive_pop))
        offspring = []
        for idx in range(pop_size):
            leader = archive_pop[int(np.random.choice(leader_front))]
            offspring.append(_mix_discrete_particle(pop[idx], pbest[idx], leader, inst, inertia, c1, c2))
        pop = np.asarray(offspring, dtype=int)
        objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
        update_mask = _personal_best_update(objs, cvs, pbest_objs, pbest_cvs)
        pbest[update_mask] = pop[update_mask]
        pbest_objs[update_mask] = objs[update_mask]
        pbest_cvs[update_mask] = cvs[update_mask]
        combined_pop = np.vstack([archive_pop, pop, pbest])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        archive_pop, archive_objs, archive_cvs = _archive_from_population(combined_pop, combined_objs, combined_cvs, archive_size)
        if verbose and (gen % 20 == 0 or gen == max_gen - 1):
            print(f"[MOPSO] Gen {gen + 1}/{max_gen}: archive={len(archive_pop)}")
        history.append({"gen": gen + 1, "pareto_size": len(archive_pop), "min_cost": float(archive_objs[:, 0].min()), "max_service_ratio": float((-archive_objs[:, 1]).max())})
    return _pareto_result("MOPSO", archive_pop, archive_objs, archive_cvs, history, time.time() - started)


def _uniform_weights(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([[0.5, 0.5]], dtype=float)
    values = np.linspace(0.0, 1.0, n)
    return np.column_stack([values, 1.0 - values])


def _tchebycheff(objs: np.ndarray, weights: np.ndarray, ideal: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    penalty = np.maximum(cvs, 0.0) * 1e9
    return np.max(weights * np.abs(objs - ideal), axis=1) + penalty


def run_moead(
    inst,
    pop_size: int = 100,
    max_gen: int = 300,
    seed: int | None = None,
    n_neighbors: int | None = None,
    verbose: bool = False,
):
    """Two-objective MOEA/D with Tchebycheff decomposition."""
    _seed_everything(seed)
    started = time.time()
    n_neighbors = int(n_neighbors or max(5, min(20, pop_size // 5)))
    weights = _uniform_weights(pop_size)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :n_neighbors]
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    ideal = objs[cvs <= 1e-9].min(axis=0) if np.any(cvs <= 1e-9) else objs[np.argmin(cvs)].copy()
    archive_pop, archive_objs, archive_cvs = _archive_from_population(pop, objs, cvs, 2 * pop_size)
    history = []
    for gen in range(max_gen):
        for i in range(pop_size):
            pool = neighbors[i]
            p1, p2 = np.random.choice(pool, size=2, replace=True)
            child1, child2 = crossover(pop[p1], pop[p2], inst, pc=0.9)
            child = child1 if np.random.rand() < 0.5 else child2
            child = mutate(child, inst)
            child = normalize_solution(child, inst)
            child_obj, child_cv, _ = evaluate_population(np.asarray([child]), inst, use_repair=True, collect_parts=False)
            child_obj = child_obj[0]
            child_cv = float(child_cv[0])
            if child_cv <= 1e-9:
                ideal = np.minimum(ideal, child_obj)
            current_fit = _tchebycheff(objs[pool], weights[pool], ideal, cvs[pool])
            child_fit = _tchebycheff(np.repeat(child_obj[None, :], len(pool), axis=0), weights[pool], ideal, np.repeat(child_cv, len(pool)))
            replace = pool[child_fit <= current_fit]
            if len(replace):
                pop[replace] = child
                objs[replace] = child_obj
                cvs[replace] = child_cv
        combined_pop = np.vstack([archive_pop, pop])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        archive_pop, archive_objs, archive_cvs = _archive_from_population(combined_pop, combined_objs, combined_cvs, 2 * pop_size)
        if verbose and (gen % 20 == 0 or gen == max_gen - 1):
            print(f"[MOEA/D] Gen {gen + 1}/{max_gen}: archive={len(archive_pop)}")
        history.append({"gen": gen + 1, "pareto_size": len(archive_pop), "min_cost": float(archive_objs[:, 0].min()), "max_service_ratio": float((-archive_objs[:, 1]).max())})
    return _pareto_result("MOEA/D", archive_pop, archive_objs, archive_cvs, history, time.time() - started)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run NSGA-II, MOPSO, MOEA/D and QVNS-NSGA-II comparison experiments.")
    parser.add_argument("--algorithms", nargs="+", default=["MOPSO", "MOEA/D"], choices=list(COMPARISON_ALGORITHMS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "moo_comparison")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def _run_algorithm(name: str, inst, pop_size: int, max_gen: int, seed: int, quiet: bool):
    if name == "NSGA-II":
        return run_nsga2(inst, pop_size=pop_size, max_gen=max_gen, seed=seed, verbose=not quiet, archive_size=pop_size)
    if name == "MOPSO":
        return run_mopso(inst, pop_size=pop_size, max_gen=max_gen, seed=seed, verbose=not quiet)
    if name == "MOEA/D":
        return run_moead(inst, pop_size=pop_size, max_gen=max_gen, seed=seed, verbose=not quiet)
    raise ValueError(f"Unsupported algorithm in this script: {name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_setting = _normalize_model_setting(args.model_setting)
    run_path = args.output_dir / "ablation_run_metrics.csv"
    points_path = args.output_dir / "ablation_pareto_points.csv"
    run_rows: list[dict] = []
    point_frames: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    if args.resume and run_path.exists():
        existing = pd.read_csv(run_path, encoding="utf-8-sig")
        run_rows.extend(existing.to_dict("records"))
        completed = set(zip(existing["algorithm"].astype(str), existing["run_id"].astype(int)))
        if points_path.exists():
            point_frames.append(pd.read_csv(points_path, encoding="utf-8-sig"))
    inst = build_metro_logistics_instance("base", top_n=args.top_n, model_setting=model_setting)
    for algorithm in args.algorithms:
        for run_id in range(1, args.runs + 1):
            if (algorithm, run_id) in completed:
                print(f"[{algorithm}] run {run_id}/{args.runs} skipped")
                continue
            seed = int(args.seed + run_id - 1)
            print(f"[{algorithm}] run {run_id}/{args.runs}, seed={seed}", flush=True)
            started = time.time()
            result = _run_algorithm(algorithm, inst, args.pop_size, args.max_gen, seed, args.quiet)
            runtime = float(result.get("runtime", time.time() - started))
            display_algorithm = algorithm
            row, pts = summarize_result(display_algorithm, model_setting, run_id, seed, result, runtime, group_type="algorithm_comparison", config=None)
            run_rows.append(row)
            if len(pts):
                point_frames.append(pts)
            run_df = pd.DataFrame(run_rows)
            points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
            if len(points_df):
                run_df = add_quality_indicators(run_df, points_df)
            summary_df = make_summary(run_df) if len(run_df) else pd.DataFrame()
            run_df.to_csv(run_path, index=False, encoding="utf-8-sig")
            points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
            summary_df.to_csv(args.output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    if point_frames and not args.skip_plots:
        points_df = pd.concat(point_frames, ignore_index=True)
        run_df = add_quality_indicators(pd.DataFrame(run_rows), points_df)
        summary_df = make_summary(run_df)
        plot_pareto(points_df, args.output_dir)
        plot_metrics(summary_df, args.output_dir)
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
