# -*- coding: utf-8 -*-
"""Run the four-algorithm performance comparison with solver-only runtime.

This runner keeps the algorithm settings used by the original chapter-5
comparison, but it does not include per-generation convergence logging in the
runtime of NSGA-II, MOPSO, or SPEA2. The exported runtime is therefore the
solver execution time for every algorithm.
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
        if (path / "code" / "main.py").exists() and (path / "scripts" / "run_four_algorithm_convergence.py").exists():
            return path
    raise RuntimeError("Cannot locate project root containing code/main.py")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (CODE_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import run_four_algorithm_convergence as conv  # noqa: E402
from main import (  # noqa: E402
    _normalize_model_setting,
    build_metro_logistics_instance,
    crossover,
    crowding_distance,
    evaluate_population,
    fast_non_dominated_sort,
    initialize_population,
    mutate,
    rank_and_crowding,
    repair_solution,
    run_nsga2,
    tournament_select,
)
from plot_algorithm_performance_metrics import build_performance_tables, plot_performance_metrics, plot_stability_distribution  # noqa: E402


ALGORITHMS = ("QVNS-NSGA2", "NSGA2", "MOPSO", "SPEA2")


def _retag_result(result: dict[str, Any], algorithm: str) -> dict[str, Any]:
    result = dict(result)
    result["algorithm"] = algorithm
    return result


def run_nsga2_solver_only(inst, pop_size: int, max_gen: int, seed: int) -> dict[str, Any]:
    result = run_nsga2(
        inst,
        pop_size=pop_size,
        max_gen=max_gen,
        seed=seed,
        verbose=False,
        archive_size=pop_size,
    )
    return _retag_result(result, "NSGA2")


def run_mopso_solver_only(inst, pop_size: int, max_gen: int, seed: int) -> dict[str, Any]:
    conv.seed_everything(seed)
    started = time.perf_counter()
    archive_size = 2 * pop_size
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    pbest = pop.copy()
    pbest_objs = objs.copy()
    pbest_cvs = cvs.copy()
    archive_pop, archive_objs, archive_cvs = conv._archive_from_population(pop, objs, cvs, archive_size)
    for gen in range(max_gen):
        progress = gen / max(1, max_gen - 1)
        inertia = 0.70 - 0.35 * progress
        c1 = 0.35 + 0.15 * (1.0 - progress)
        c2 = 0.35 + 0.15 * progress
        fronts, _ = fast_non_dominated_sort(archive_objs, archive_cvs)
        leader_front = np.array(fronts[0] if fronts else np.arange(len(archive_pop)), dtype=int)
        if len(leader_front) == 0:
            leader_front = np.arange(len(archive_pop))
        offspring = []
        for idx in range(pop_size):
            leader = archive_pop[int(np.random.choice(leader_front))]
            offspring.append(conv._mix_particle(pop[idx], pbest[idx], leader, inst, inertia, c1, c2))
        pop = np.asarray(offspring, dtype=int)
        objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
        update = conv._personal_best_update(objs, cvs, pbest_objs, pbest_cvs)
        pbest[update] = pop[update]
        pbest_objs[update] = objs[update]
        pbest_cvs[update] = cvs[update]
        combined_pop = np.vstack([archive_pop, pop, pbest])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        archive_pop, archive_objs, archive_cvs = conv._archive_from_population(combined_pop, combined_objs, combined_cvs, archive_size)
    result = conv._make_result("MOPSO", archive_pop, archive_objs, archive_cvs, started)
    result["runtime"] = float(time.perf_counter() - started)
    return result


def run_spea2_solver_only(inst, pop_size: int, max_gen: int, seed: int) -> dict[str, Any]:
    conv.seed_everything(seed)
    started = time.perf_counter()
    archive_size = pop_size
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    archive_pop, archive_objs, archive_cvs = conv._spea2_environmental_selection(pop, objs, cvs, archive_size)
    for _ in range(max_gen):
        combined_pop = np.vstack([pop, archive_pop])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        fitness = conv._spea2_fitness(combined_objs, combined_cvs)
        offspring = []
        while len(offspring) < pop_size:
            contenders = np.random.choice(len(combined_pop), size=2, replace=True)
            p1 = combined_pop[contenders[int(np.argmin(fitness[contenders]))]]
            contenders = np.random.choice(len(combined_pop), size=2, replace=True)
            p2 = combined_pop[contenders[int(np.argmin(fitness[contenders]))]]
            child1, child2 = crossover(p1, p2, inst, pc=0.9)
            offspring.append(mutate(child1, inst))
            if len(offspring) < pop_size:
                offspring.append(mutate(child2, inst))
        pop = np.asarray(offspring[:pop_size], dtype=int)
        objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
        env_pop = np.vstack([pop, archive_pop])
        env_objs, env_cvs, _ = evaluate_population(env_pop, inst, use_repair=True, collect_parts=False)
        archive_pop, archive_objs, archive_cvs = conv._spea2_environmental_selection(env_pop, env_objs, env_cvs, archive_size)
    result = conv._make_result("SPEA2", archive_pop, archive_objs, archive_cvs, started)
    result["runtime"] = float(time.perf_counter() - started)
    return result


def run_qvns_solver_only(inst, pop_size: int, max_gen: int, seed: int) -> dict[str, Any]:
    result = conv._run_qvns(inst, pop_size, max_gen, seed, verbose=False)
    return _retag_result(result, "QVNS-NSGA2")


def run_algorithm(algorithm: str, inst, pop_size: int, max_gen: int, seed: int) -> dict[str, Any]:
    if algorithm == "QVNS-NSGA2":
        return run_qvns_solver_only(inst, pop_size, max_gen, seed)
    if algorithm == "NSGA2":
        return run_nsga2_solver_only(inst, pop_size, max_gen, seed)
    if algorithm == "MOPSO":
        return run_mopso_solver_only(inst, pop_size, max_gen, seed)
    if algorithm == "SPEA2":
        return run_spea2_solver_only(inst, pop_size, max_gen, seed)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def result_final_row(result: dict[str, Any], algorithm: str, run_id: int, seed: int, max_gen: int, inst) -> dict[str, Any]:
    pop = np.asarray(result.get("pareto_dec", result.get("pop")))
    objs = np.asarray(result.get("pareto_objs", result.get("objs")))
    cvs = np.asarray(result.get("pareto_cvs", result.get("cvs", np.zeros(len(objs)))))
    row = conv.extract_generation_metrics(algorithm, run_id, seed, max_gen, pop, objs, cvs, inst)
    row["runtime_seconds"] = float(result.get("runtime", np.nan))
    row["final_pareto_size"] = int(len(result.get("pareto_objs", [])))
    row["runtime_scope"] = "solver_only"
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solver-only four-algorithm performance comparison.")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_setting = _normalize_model_setting(args.model_setting)
    inst = build_metro_logistics_instance(args.scenario, top_n=args.top_n, model_setting=model_setting)

    final_path = args.output_dir / "final_summary.csv"
    pareto_path = args.output_dir / "pareto_cost_service_points.csv"
    if args.resume and final_path.exists() and pareto_path.exists():
        final_summary = pd.read_csv(final_path, encoding="utf-8-sig")
        pareto_points = pd.read_csv(pareto_path, encoding="utf-8-sig")
        completed = set(zip(final_summary["algorithm"].astype(str), final_summary["run_id"].astype(int)))
        summary_rows = final_summary.to_dict("records")
        point_frames = [pareto_points]
    else:
        completed = set()
        summary_rows: list[dict[str, Any]] = []
        point_frames: list[pd.DataFrame] = []

    for algorithm in args.algorithms:
        for run_id in range(1, args.runs + 1):
            key = (algorithm, run_id)
            if key in completed:
                print(f"[{algorithm}] run {run_id}/{args.runs} skipped", flush=True)
                continue
            seed = int(args.seed + (run_id - 1) * 100 + list(ALGORITHMS).index(algorithm) * 10000)
            print(f"[{algorithm}] run {run_id}/{args.runs}, seed={seed}", flush=True)
            result = run_algorithm(algorithm, inst, args.pop_size, args.max_gen, seed)
            row = result_final_row(result, algorithm, run_id, seed, args.max_gen, inst)
            summary_rows.append(row)
            point_frames.append(conv.extract_pareto_points(result, algorithm, run_id, seed))
            pd.DataFrame(summary_rows).to_csv(final_path, index=False, encoding="utf-8-sig")
            pd.concat(point_frames, ignore_index=True).to_csv(pareto_path, index=False, encoding="utf-8-sig")
            print(f"[{algorithm}] run {run_id} done, solver_runtime={row['runtime_seconds']:.2f}s", flush=True)

    final_summary = pd.DataFrame(summary_rows)
    pareto_points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    final_summary.to_csv(final_path, index=False, encoding="utf-8-sig")
    pareto_points.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    run_metrics, summary = build_performance_tables(pareto_points, final_summary)
    run_metrics.to_csv(args.output_dir / "algorithm_performance_run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "algorithm_performance_summary.csv", index=False, encoding="utf-8-sig")
    plot_performance_metrics(summary, args.output_dir)
    plot_stability_distribution(run_metrics, args.output_dir, prefix="paper_fig_algorithm")

    note = args.output_dir / "solver_only_runtime_note.md"
    note.write_text(
        "# Solver-only runtime note\n\n"
        "This experiment reports solver-only runtime for all algorithms. "
        "Per-generation convergence logging is excluded from NSGA-II, MOPSO and SPEA2 runtime. "
        "QVNS-NSGA-II uses the runtime returned by its solver implementation.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
