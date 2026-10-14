# -*- coding: utf-8 -*-
"""Run four metro-logistics algorithms and plot convergence curves.

Outputs per-generation total cost, real carbon cost, hypervolume, and cost
improvement rate for QVNS-NSGA2, NSGA2, MOPSO, and MOEA/D on the thesis case.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (CODE_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from main import (  # noqa: E402
    _normalize_model_setting,
    build_metro_logistics_instance,
    crossover,
    crowding_distance,
    evaluate_population,
    evaluate_solution,
    fast_non_dominated_sort,
    initialize_population,
    mutate,
    normalize_solution,
    rank_and_crowding,
    repair_solution,
    tournament_select,
)
from chinese_font_utils import configure_matplotlib_cjk  # noqa: E402

configure_matplotlib_cjk(plt)

ALGORITHMS = ("QVNS-NSGA2", "NSGA2", "MOPSO", "SPEA2")
OPTIONAL_ALGORITHMS = ALGORITHMS + ("MOEAD",)
ALGORITHM_LABELS = {
    "QVNS-NSGA2": "QVNS-NSGA-II",
    "NSGA2": "NSGA-II",
    "MOPSO": "MOPSO",
    "SPEA2": "SPEA2",
    "MOEAD": "MOEA/D",
}
ALGORITHM_STYLES = {
    "QVNS-NSGA2": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "NSGA2": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "MOPSO": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    "SPEA2": {"color": "#9467bd", "linestyle": ":", "marker": "D"},
    "MOEAD": {"color": "#d62728", "linestyle": ":", "marker": "D"},
}
ALGORITHM_SLUGS = {
    "QVNS-NSGA2": "qvns_nsga2",
    "NSGA2": "nsga2",
    "MOPSO": "mopso",
    "SPEA2": "spea2",
    "MOEAD": "moead",
}


def seed_everything(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))


def compute_cost_improvement_rate(initial_cost: float, current_cost: float) -> float:
    if initial_cost <= 0 or not np.isfinite(initial_cost) or not np.isfinite(current_cost):
        return 0.0
    return float((initial_cost - current_cost) / initial_cost * 100.0)


def _select_best_index(objectives: np.ndarray, cvs: np.ndarray) -> int:
    feasible = np.where(cvs <= 1e-9)[0]
    candidates = feasible if len(feasible) else np.arange(len(objectives))
    if len(feasible):
        return int(candidates[np.argmin(objectives[candidates, 0])])
    return int(candidates[np.lexsort((objectives[candidates, 0], cvs[candidates]))[0]])


def _candidate_indices(objectives: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    feasible = np.where(cvs <= 1e-9)[0]
    return feasible if len(feasible) else np.arange(len(objectives))


def _evaluate_candidate_parts(population: np.ndarray, indices: np.ndarray, inst) -> list[tuple[int, np.ndarray, float, dict[str, float]]]:
    evaluated = []
    for idx in indices:
        objective, cv, parts = evaluate_solution(population[int(idx)], inst, use_repair=True, collect_parts=True)
        evaluated.append((int(idx), objective, float(cv), parts))
    return evaluated


def extract_generation_metrics(
    algorithm: str,
    run_id: int,
    seed: int,
    gen: int,
    population: np.ndarray,
    objectives: np.ndarray,
    cvs: np.ndarray,
    inst,
) -> dict[str, Any]:
    best_idx = _select_best_index(objectives, cvs)
    candidate_indices = _candidate_indices(objectives, cvs)
    evaluated = _evaluate_candidate_parts(population, candidate_indices, inst)
    evaluated_by_idx = {idx: (objective, cv, parts) for idx, objective, cv, parts in evaluated}
    fronts, _ = fast_non_dominated_sort(objectives, cvs)
    pareto_front = np.array(fronts[0] if fronts else np.arange(len(objectives)), dtype=int)
    pareto_feasible = pareto_front[cvs[pareto_front] <= 1e-9]
    pareto_indices = pareto_feasible if len(pareto_feasible) else pareto_front
    pareto_costs = objectives[pareto_indices, 0] if len(pareto_indices) else np.array([], dtype=float)
    pareto_services = -objectives[pareto_indices, 1] if len(pareto_indices) else np.array([], dtype=float)
    best_objective, best_cv, parts = evaluated_by_idx.get(
        best_idx,
        evaluate_solution(population[best_idx], inst, use_repair=True, collect_parts=True),
    )
    carbon_values = [float(item_parts.get("carbon_cost", np.nan)) for _, _, _, item_parts in evaluated]
    min_carbon_cost = float(np.nanmin(carbon_values)) if carbon_values else float("nan")
    total_cost = float(best_objective[0])
    service_ratio = float(-best_objective[1])
    carbon_cost_at_min_total_cost = float(parts.get("carbon_cost", np.nan))
    return {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
        "run_id": int(run_id),
        "seed": int(seed),
        "gen": int(gen),
        "total_cost": total_cost,
        "carbon_cost": min_carbon_cost,
        "carbon_cost_at_min_total_cost": carbon_cost_at_min_total_cost,
        "service_ratio": service_ratio,
        "pareto_mean_total_cost": float(np.mean(pareto_costs)) if len(pareto_costs) else float("nan"),
        "pareto_mean_service_ratio": float(np.mean(pareto_services)) if len(pareto_services) else float("nan"),
        "cv": float(best_cv),
        "pareto_size": int(len(fast_non_dominated_sort(objectives, cvs)[0][0])) if len(objectives) else 0,
        "feasible_count": int(np.sum(cvs <= 1e-9)),
        "population_size": int(len(population)),
    }


def _archive_from_population(pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, archive_size: int):
    fronts, _ = fast_non_dominated_sort(objs, cvs)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= archive_size:
            selected.extend(front)
        else:
            distances = crowding_distance(objs, front)
            order = np.argsort(-distances)
            selected.extend([front[i] for i in order[: archive_size - len(selected)]])
            break
    if not selected:
        selected = np.argsort(cvs)[: max(1, archive_size)].tolist()
    selected_arr = np.array(selected, dtype=int)
    return pop[selected_arr].copy(), objs[selected_arr].copy(), cvs[selected_arr].copy()


def _append_metrics(history: list[dict[str, Any]], algorithm: str, run_id: int, seed: int, gen: int, pop, objs, cvs, inst) -> None:
    row = extract_generation_metrics(algorithm, run_id, seed, gen, pop, objs, cvs, inst)
    history.append(row)


def run_nsga2_tracked(inst, pop_size: int, max_gen: int, seed: int, run_id: int, verbose: bool) -> tuple[dict, pd.DataFrame]:
    seed_everything(seed)
    started = time.time()
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    history: list[dict[str, Any]] = []
    for gen in range(max_gen + 1):
        _append_metrics(history, "NSGA2", run_id, seed, gen, pop, objs, cvs, inst)
        if verbose and gen % 20 == 0:
            print(f"[NSGA2] run={run_id} gen={gen}/{max_gen}", flush=True)
        if gen == max_gen:
            break
        fronts, rank, crowd = rank_and_crowding(objs, cvs)
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, objs, cvs, rank, crowd)
            p2 = tournament_select(pop, objs, cvs, rank, crowd)
            c1, c2 = crossover(p1, p2, inst, pc=0.9)
            offspring.append(mutate(c1, inst))
            if len(offspring) < pop_size:
                offspring.append(mutate(c2, inst))
        offspring = np.asarray(offspring[:pop_size], dtype=int)
        off_objs, off_cvs, _ = evaluate_population(offspring, inst, use_repair=True, collect_parts=False)
        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, off_objs])
        combined_cvs = np.concatenate([cvs, off_cvs])
        fronts, _, _ = rank_and_crowding(combined_objs, combined_cvs)
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                distances = crowding_distance(combined_objs, front)
                order = np.argsort(-distances)
                selected.extend([front[i] for i in order[: pop_size - len(selected)]])
                break
        selected_arr = np.array(selected, dtype=int)
        pop = combined_pop[selected_arr]
        objs = combined_objs[selected_arr]
        cvs = combined_cvs[selected_arr]
    result = _make_result("NSGA2", pop, objs, cvs, started)
    return result, pd.DataFrame(history)


def _personal_best_update(current_objs, current_cvs, best_objs, best_cvs):
    current_feasible = current_cvs <= 1e-9
    best_feasible = best_cvs <= 1e-9
    better = current_feasible & (~best_feasible)
    better |= (~current_feasible) & (~best_feasible) & (current_cvs < best_cvs)
    both = current_feasible & best_feasible
    dominates = np.all(current_objs <= best_objs + 1e-12, axis=1) & np.any(current_objs < best_objs - 1e-12, axis=1)
    reverse = np.all(best_objs <= current_objs + 1e-12, axis=1) & np.any(best_objs < current_objs - 1e-12, axis=1)
    ties = both & (~dominates) & (~reverse)
    better |= both & dominates
    better |= ties & (np.random.rand(len(current_objs)) < 0.5)
    return better


def _mix_particle(position: np.ndarray, pbest: np.ndarray, leader: np.ndarray, inst, inertia_keep: float, c1: float, c2: float):
    child = position.copy()
    n_vars = len(child)
    pbest_mask = np.random.rand(n_vars) < min(0.95, max(0.0, c1))
    leader_mask = np.random.rand(n_vars) < min(0.95, max(0.0, c2))
    keep_mask = np.random.rand(n_vars) < min(0.95, max(0.0, inertia_keep))
    child[~keep_mask & pbest_mask] = pbest[~keep_mask & pbest_mask]
    child[~keep_mask & leader_mask] = leader[~keep_mask & leader_mask]
    return repair_solution(mutate(child, inst), inst)


def run_mopso_tracked(inst, pop_size: int, max_gen: int, seed: int, run_id: int, verbose: bool) -> tuple[dict, pd.DataFrame]:
    seed_everything(seed)
    started = time.time()
    archive_size = 2 * pop_size
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    pbest = pop.copy()
    pbest_objs = objs.copy()
    pbest_cvs = cvs.copy()
    archive_pop, archive_objs, archive_cvs = _archive_from_population(pop, objs, cvs, archive_size)
    history: list[dict[str, Any]] = []
    for gen in range(max_gen + 1):
        _append_metrics(history, "MOPSO", run_id, seed, gen, archive_pop, archive_objs, archive_cvs, inst)
        if verbose and gen % 20 == 0:
            print(f"[MOPSO] run={run_id} gen={gen}/{max_gen}", flush=True)
        if gen == max_gen:
            break
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
            offspring.append(_mix_particle(pop[idx], pbest[idx], leader, inst, inertia, c1, c2))
        pop = np.asarray(offspring, dtype=int)
        objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
        update = _personal_best_update(objs, cvs, pbest_objs, pbest_cvs)
        pbest[update] = pop[update]
        pbest_objs[update] = objs[update]
        pbest_cvs[update] = cvs[update]
        combined_pop = np.vstack([archive_pop, pop, pbest])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        archive_pop, archive_objs, archive_cvs = _archive_from_population(combined_pop, combined_objs, combined_cvs, archive_size)
    result = _make_result("MOPSO", archive_pop, archive_objs, archive_cvs, started)
    return result, pd.DataFrame(history)


def _spea2_fitness(objs: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    n = len(objs)
    feasible = cvs <= 1e-9
    dominates = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if feasible[i] and not feasible[j]:
                dominates[i, j] = True
            elif feasible[i] and feasible[j]:
                dominates[i, j] = bool(np.all(objs[i] <= objs[j] + 1e-12) and np.any(objs[i] < objs[j] - 1e-12))
            elif (not feasible[i]) and (not feasible[j]) and cvs[i] < cvs[j] - 1e-12:
                dominates[i, j] = True
    strength = dominates.sum(axis=1).astype(float)
    raw = dominates.T @ strength
    if n <= 1:
        density = np.zeros(n, dtype=float)
    else:
        distances = np.linalg.norm(objs[:, None, :] - objs[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        kth = max(1, int(np.sqrt(n)))
        kth_dist = np.partition(distances, kth - 1, axis=1)[:, kth - 1]
        density = 1.0 / (kth_dist + 2.0)
    return raw + density + np.maximum(cvs, 0.0) * 1e6


def _spea2_environmental_selection(pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, archive_size: int):
    fitness = _spea2_fitness(objs, cvs)
    selected = np.where(fitness < 1.0)[0].tolist()
    if len(selected) < archive_size:
        remaining = [idx for idx in np.argsort(fitness) if idx not in selected]
        selected.extend(remaining[: archive_size - len(selected)])
    elif len(selected) > archive_size:
        selected = selected.copy()
        while len(selected) > archive_size:
            selected_objs = objs[selected]
            distances = np.linalg.norm(selected_objs[:, None, :] - selected_objs[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            nearest = np.min(distances, axis=1)
            remove_pos = int(np.argmin(nearest))
            selected.pop(remove_pos)
    selected_arr = np.array(selected[:archive_size], dtype=int)
    return pop[selected_arr].copy(), objs[selected_arr].copy(), cvs[selected_arr].copy()


def run_spea2_tracked(inst, pop_size: int, max_gen: int, seed: int, run_id: int, verbose: bool) -> tuple[dict, pd.DataFrame]:
    seed_everything(seed)
    started = time.time()
    archive_size = pop_size
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    archive_pop, archive_objs, archive_cvs = _spea2_environmental_selection(pop, objs, cvs, archive_size)
    history: list[dict[str, Any]] = []
    for gen in range(max_gen + 1):
        _append_metrics(history, "SPEA2", run_id, seed, gen, archive_pop, archive_objs, archive_cvs, inst)
        if verbose and gen % 20 == 0:
            print(f"[SPEA2] run={run_id} gen={gen}/{max_gen}", flush=True)
        if gen == max_gen:
            break
        combined_pop = np.vstack([pop, archive_pop])
        combined_objs, combined_cvs, _ = evaluate_population(combined_pop, inst, use_repair=True, collect_parts=False)
        fitness = _spea2_fitness(combined_objs, combined_cvs)
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
        archive_pop, archive_objs, archive_cvs = _spea2_environmental_selection(env_pop, env_objs, env_cvs, archive_size)
    result = _make_result("SPEA2", archive_pop, archive_objs, archive_cvs, started)
    return result, pd.DataFrame(history)


def _uniform_weights(n_weights: int) -> np.ndarray:
    if n_weights <= 1:
        return np.array([[0.5, 0.5]], dtype=float)
    values = np.linspace(0.0, 1.0, n_weights)
    return np.column_stack([values, 1.0 - values])


def _tchebycheff(objs: np.ndarray, weights: np.ndarray, ideal: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    penalty = np.maximum(cvs, 0.0) * 1e9
    return np.max(weights * np.abs(objs - ideal), axis=1) + penalty


def run_moead_tracked(inst, pop_size: int, max_gen: int, seed: int, run_id: int, verbose: bool) -> tuple[dict, pd.DataFrame]:
    seed_everything(seed)
    started = time.time()
    n_neighbors = max(5, min(20, pop_size // 5))
    weights = _uniform_weights(pop_size)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :n_neighbors]
    pop = initialize_population(inst, pop_size, use_repair=True)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=True, collect_parts=False)
    ideal = objs[cvs <= 1e-9].min(axis=0) if np.any(cvs <= 1e-9) else objs[np.argmin(cvs)].copy()
    archive_pop, archive_objs, archive_cvs = _archive_from_population(pop, objs, cvs, 2 * pop_size)
    history: list[dict[str, Any]] = []
    for gen in range(max_gen + 1):
        _append_metrics(history, "MOEAD", run_id, seed, gen, archive_pop, archive_objs, archive_cvs, inst)
        if verbose and gen % 20 == 0:
            print(f"[MOEA/D] run={run_id} gen={gen}/{max_gen}", flush=True)
        if gen == max_gen:
            break
        for idx in range(pop_size):
            pool = neighbors[idx]
            parent_a, parent_b = np.random.choice(pool, size=2, replace=True)
            child_a, child_b = crossover(pop[parent_a], pop[parent_b], inst, pc=0.9)
            child = child_a if np.random.rand() < 0.5 else child_b
            child = normalize_solution(mutate(child, inst), inst)
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
    result = _make_result("MOEAD", archive_pop, archive_objs, archive_cvs, started)
    return result, pd.DataFrame(history)


def _run_qvns(inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    from main import run_qvns_nsga2

    return run_qvns_nsga2(
        inst,
        pop_size=pop_size,
        max_gen=max_gen,
        seed=seed,
        verbose=verbose,
        use_repair=True,
        alpha=0.15,
        gamma=0.88,
        epsilon=0.30,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        archive_size=None,
        preserve_baseline=False,
        vns_interval=0,
        vns_elite_rate=0.12,
        vns_trials_per_neighborhood=2,
        vns_max_neighborhood=3,
        archive_feedback_interval=8,
        archive_feedback_rate=0.15,
        q_action_apply_rate=0.25,
        q_action_profile="safe",
        q_learning_enabled=True,
        safeguard=True,
        safeguard_threshold=-0.15,
        pc=0.90,
    )


def run_qvns_tracked(inst, pop_size: int, max_gen: int, seed: int, run_id: int, verbose: bool) -> tuple[dict, pd.DataFrame]:
    result = _run_qvns(inst, pop_size, max_gen, seed, verbose)
    final_decisions = np.asarray(result.get("pop", result.get("pareto_dec")))
    final_objs = np.asarray(result.get("objs", result.get("pareto_objs")))
    final_cvs = np.asarray(result.get("cvs", result.get("pareto_cvs")))
    source_history = result.get("history", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if isinstance(source_history, pd.DataFrame) and not source_history.empty:
        for _, record in source_history.iterrows():
            gen = int(record.get("gen", len(rows)))
            row = extract_generation_metrics("QVNS-NSGA2", run_id, seed, gen, final_decisions, final_objs, final_cvs, inst)
            if "best_cost" in record and pd.notna(record["best_cost"]):
                row["total_cost"] = float(record["best_cost"])
            if "best_cost_carbon_cost" in record and pd.notna(record["best_cost_carbon_cost"]):
                row["carbon_cost"] = float(record["best_cost_carbon_cost"])
            if "best_cost_cv" in record and pd.notna(record["best_cost_cv"]):
                row["cv"] = float(record["best_cost_cv"])
            if "best_service_ratio" in record and pd.notna(record["best_service_ratio"]):
                row["service_ratio"] = float(record["best_service_ratio"])
            if "pareto_mean_total_cost" in record and pd.notna(record["pareto_mean_total_cost"]):
                row["pareto_mean_total_cost"] = float(record["pareto_mean_total_cost"])
            if "pareto_mean_service_ratio" in record and pd.notna(record["pareto_mean_service_ratio"]):
                row["pareto_mean_service_ratio"] = float(record["pareto_mean_service_ratio"])
            if "population_pareto_mean_total_cost" in record and pd.notna(record["population_pareto_mean_total_cost"]):
                row["population_pareto_mean_total_cost"] = float(record["population_pareto_mean_total_cost"])
            if "population_pareto_mean_service_ratio" in record and pd.notna(record["population_pareto_mean_service_ratio"]):
                row["population_pareto_mean_service_ratio"] = float(record["population_pareto_mean_service_ratio"])
            if "population_pareto_size" in record and pd.notna(record["population_pareto_size"]):
                row["population_pareto_size"] = int(record["population_pareto_size"])
            if "pareto_size" in record and pd.notna(record["pareto_size"]):
                row["pareto_size"] = int(record["pareto_size"])
            rows.append(row)
    else:
        for gen in range(max_gen + 1):
            rows.append(extract_generation_metrics("QVNS-NSGA2", run_id, seed, gen, final_decisions, final_objs, final_cvs, inst))
    return result, pd.DataFrame(rows)


def _make_result(algorithm: str, pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, started: float) -> dict[str, Any]:
    fronts, _ = fast_non_dominated_sort(objs, cvs)
    first = np.array(fronts[0] if fronts else np.arange(len(pop)), dtype=int)
    feasible = first[cvs[first] <= 1e-9]
    if len(feasible):
        first = feasible
    return {
        "algorithm": algorithm,
        "pop": pop,
        "objs": objs,
        "cvs": cvs,
        "pareto_dec": pop[first].copy(),
        "pareto_objs": objs[first].copy(),
        "pareto_cvs": cvs[first].copy(),
        "runtime": float(time.time() - started),
    }


def compute_hypervolume_2d(points: np.ndarray, ref_point: tuple[float, float]) -> float:
    if len(points) == 0:
        return 0.0
    clean = points[np.all(np.isfinite(points), axis=1)]
    if len(clean) == 0:
        return 0.0
    clean = clean[np.argsort(clean[:, 0])]
    hv = 0.0
    previous_y = float(ref_point[1])
    for x_value, y_value in clean:
        if x_value <= ref_point[0] and y_value < previous_y:
            hv += max(0.0, ref_point[0] - x_value) * max(0.0, previous_y - y_value)
            previous_y = y_value
    return float(hv)


def add_derived_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()
    df = raw_df.copy()
    df["cost_improvement_rate"] = 0.0
    for (algorithm, run_id), group in df.groupby(["algorithm", "run_id"]):
        ordered = group.sort_values("gen")
        initial_cost = float(ordered.iloc[0]["total_cost"])
        rates = [compute_cost_improvement_rate(initial_cost, float(cost)) for cost in ordered["total_cost"]]
        df.loc[ordered.index, "cost_improvement_rate"] = rates
    obj_points = np.column_stack([df["total_cost"].to_numpy(dtype=float), -df["service_ratio"].to_numpy(dtype=float)])
    ref = (float(np.nanmax(obj_points[:, 0]) * 1.05), float(np.nanmax(obj_points[:, 1]) * 1.05))
    df["hypervolume"] = 0.0
    for (algorithm, run_id), group in df.groupby(["algorithm", "run_id"]):
        cumulative: list[list[float]] = []
        for idx, row in group.sort_values("gen").iterrows():
            cumulative.append([float(row["total_cost"]), -float(row["service_ratio"])])
            df.loc[idx, "hypervolume"] = compute_hypervolume_2d(np.asarray(cumulative, dtype=float), ref)
    return df


def aggregate_convergence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "total_cost",
        "carbon_cost",
        "service_ratio",
        "pareto_mean_total_cost",
        "pareto_mean_service_ratio",
        "hypervolume",
        "cost_improvement_rate",
        "feasible_count",
        "pareto_size",
    ]
    agg = {}
    for metric in metrics:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_std"] = (metric, "std")
    return df.groupby(["algorithm", "algorithm_label", "gen"], as_index=False).agg(**agg)


def plot_metric(aggregated: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm in ALGORITHMS:
        data = aggregated[aggregated["algorithm"] == algorithm].sort_values("gen")
        if data.empty:
            continue
        style = ALGORITHM_STYLES[algorithm]
        label = ALGORITHM_LABELS[algorithm]
        y_mean = data[f"{metric}_mean"].to_numpy(dtype=float)
        y_std = data[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        x_values = data["gen"].to_numpy(dtype=float)
        ax.plot(x_values, y_mean, label=label, color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markevery=max(1, len(x_values) // 10))
        if len(data["gen"].unique()) > 1:
            ax.fill_between(x_values, y_mean - y_std, y_mean + y_std, color=style["color"], alpha=0.12)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(aggregated: pd.DataFrame, output_dir: Path) -> None:
    plot_metric(aggregated, output_dir, "total_cost", "Total Cost (yuan/day)", "convergence_total_cost")
    plot_metric(aggregated, output_dir, "carbon_cost", "Carbon Cost (yuan/day)", "convergence_carbon_cost")
    plot_metric(aggregated, output_dir, "hypervolume", "Hypervolume", "convergence_hypervolume")
    plot_metric(aggregated, output_dir, "cost_improvement_rate", "Cost Improvement Rate (%)", "convergence_cost_improvement_rate")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("total_cost", "Total Cost (yuan/day)", axes[0, 0]),
        ("carbon_cost", "Carbon Cost (yuan/day)", axes[0, 1]),
        ("hypervolume", "Hypervolume", axes[1, 0]),
        ("cost_improvement_rate", "Cost Improvement Rate (%)", axes[1, 1]),
    ]
    for metric, ylabel, ax in specs:
        for algorithm in ALGORITHMS:
            data = aggregated[aggregated["algorithm"] == algorithm].sort_values("gen")
            if data.empty:
                continue
            style = ALGORITHM_STYLES[algorithm]
            ax.plot(data["gen"], data[f"{metric}_mean"], label=ALGORITHM_LABELS[algorithm], color=style["color"], linestyle=style["linestyle"])
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0, 0].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_thesis_panel(ax, aggregated: pd.DataFrame, metric: str, ylabel: str, title: str, lower_is_better: bool = False) -> None:
    for algorithm in ALGORITHMS:
        data = aggregated[aggregated["algorithm"] == algorithm].sort_values("gen")
        if data.empty:
            continue
        style = ALGORITHM_STYLES[algorithm]
        ax.plot(
            data["gen"],
            data[f"{metric}_mean"],
            label=ALGORITHM_LABELS[algorithm],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            linewidth=1.8,
            markevery=max(1, len(data) // 20),
        )
    values = aggregated[f"{metric}_mean"].dropna()
    if len(values):
        reference = float(values.min() if lower_is_better else values.max())
        ax.axhline(reference, color="#ff9999", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("迭代次数", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(labelsize=9)


def _final_metric_table(aggregated: pd.DataFrame) -> list[list[str]]:
    rows = []
    for algorithm in ALGORITHMS:
        data = aggregated[aggregated["algorithm"] == algorithm].sort_values("gen")
        if data.empty:
            continue
        last = data.iloc[-1]
        rows.append(
            [
                ALGORITHM_LABELS[algorithm],
                f"{float(last['total_cost_mean']) / 10000:.1f}",
                f"{float(last['carbon_cost_mean']):.2f}",
                f"{float(last['hypervolume_mean']):.3f}",
                f"{float(last['cost_improvement_rate_mean']):.1f}%",
            ]
        )
    return rows


def plot_thesis_combined_convergence(aggregated: pd.DataFrame, output_dir: Path) -> None:
    if aggregated.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.suptitle("图4-3  算法收敛曲线对比图", fontsize=15, fontweight="bold", y=0.985)
    _plot_thesis_panel(axes[0, 0], aggregated, "total_cost", "总成本（元/日）", "(a)  总成本收敛曲线", lower_is_better=True)
    _plot_thesis_panel(axes[0, 1], aggregated, "carbon_cost", "碳排放成本（元/日）", "(b)  碳排放收敛曲线", lower_is_better=True)
    _plot_thesis_panel(axes[1, 0], aggregated, "hypervolume", "超体积指标 HV", "(c)  超体积指标收敛曲线")
    _plot_thesis_panel(axes[1, 1], aggregated, "cost_improvement_rate", "成本改进率（%）", "(d)  收敛速度对比（成本改进率）")
    axes[0, 1].legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    note = (
        "说明：曲线基于真实案例数据与模型参数求解；总成本与碳排放成本为每代候选解中的最优值，"
        "HV为累计非支配表现的二维超体积指标，成本改进率以第0代为基准。"
    )
    fig.text(0.03, 0.025, note, fontsize=8, va="bottom", bbox={"boxstyle": "round", "facecolor": "#eaf4f8", "edgecolor": "#8aa7b0", "alpha": 0.95})
    table_rows = _final_metric_table(aggregated)
    if table_rows:
        table = axes[1, 1].table(
            cellText=table_rows,
            colLabels=["算法", "总成本(万元)", "碳成本", "HV", "改进率"],
            loc="lower right",
            bbox=[0.42, -0.02, 0.56, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for _, cell in table.get_celld().items():
            cell.set_linewidth(0.4)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_dir / "fig_algorithm_convergence_comparison.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / "fig_algorithm_convergence_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_one_dual_axis_iteration_effect(ax, data: pd.DataFrame, algorithm: str, show_legend: bool = True):
    ordered = data.sort_values("gen")
    x_values = ordered["gen"].to_numpy(dtype=float)
    cost_col = "pareto_mean_total_cost_mean" if "pareto_mean_total_cost_mean" in ordered.columns else "total_cost_mean"
    service_col = "pareto_mean_service_ratio_mean" if "pareto_mean_service_ratio_mean" in ordered.columns else "service_ratio_mean"
    cost_values = ordered[cost_col].to_numpy(dtype=float)
    service_values = ordered[service_col].to_numpy(dtype=float) * 100.0
    ax2 = ax.twinx()
    cost_line = ax.step(
        x_values,
        cost_values,
        where="post",
        color="#1f77b4",
        linewidth=2.6,
        label="最优成本",
    )[0]
    service_line = ax2.step(
        x_values,
        service_values,
        where="post",
        color="#d62728",
        linestyle="--",
        linewidth=2.6,
        label="最优地铁服务比例",
    )[0]
    ax.set_title(f"{ALGORITHM_LABELS.get(algorithm, algorithm)} 迭代效果", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("迭代代数", fontsize=13)
    ax.set_ylabel("Pareto前沿平均总成本（元/日）", fontsize=13)
    ax2.set_ylabel("Pareto前沿平均地铁服务比例（%）", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(axis="both", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    if show_legend:
        ax.legend([cost_line, service_line], ["最优成本", "最优地铁服务比例"], loc="upper right", fontsize=12, frameon=True)
    return ax2


def plot_dual_axis_iteration_effects(aggregated: pd.DataFrame, output_dir: Path) -> None:
    if aggregated.empty:
        return
    for algorithm in ALGORITHMS:
        data = aggregated[aggregated["algorithm"] == algorithm]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 7))
        _plot_one_dual_axis_iteration_effect(ax, data, algorithm, show_legend=True)
        fig.tight_layout()
        slug = ALGORITHM_SLUGS[algorithm]
        fig.savefig(output_dir / f"fig_iteration_effect_{slug}.png", dpi=350, bbox_inches="tight")
        fig.savefig(output_dir / f"fig_iteration_effect_{slug}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.suptitle("四种算法迭代效果对比（P=100, G=300）", fontsize=18, fontweight="bold", y=0.985)
    for ax, algorithm in zip(axes.ravel(), ALGORITHMS):
        data = aggregated[aggregated["algorithm"] == algorithm]
        if data.empty:
            ax.axis("off")
            continue
        _plot_one_dual_axis_iteration_effect(ax, data, algorithm, show_legend=False)
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2.6),
        plt.Line2D([0], [0], color="#d62728", linestyle="--", linewidth=2.6),
    ]
    fig.legend(handles, ["最优成本", "最优地铁服务比例"], loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, fontsize=12, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / "fig_iteration_effect_four_algorithms.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / "fig_iteration_effect_four_algorithms.pdf", bbox_inches="tight")
    plt.close(fig)


def extract_pareto_points(result: dict[str, Any], algorithm: str, run_id: int, seed: int) -> pd.DataFrame:
    objs = np.asarray(result.get("pareto_objs", []), dtype=float)
    cvs = np.asarray(result.get("pareto_cvs", np.zeros(len(objs))), dtype=float)
    rows = []
    for solution_id, obj in enumerate(objs):
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "run_id": int(run_id),
                "seed": int(seed),
                "solution_id": int(solution_id),
                "Z1_total_cost": float(obj[0]),
                "Z1_total_cost_10k": float(obj[0]) / 10000.0,
                "Z2_metro_service_ratio": float(-obj[1]),
                "Z2_metro_service_percent": float(-obj[1]) * 100.0,
                "CV": float(cvs[solution_id]) if solution_id < len(cvs) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _pareto_front_for_plot(points: pd.DataFrame) -> pd.DataFrame:
    feasible = points[points["CV"].fillna(0.0) <= 1e-9].copy()
    if feasible.empty:
        feasible = points.copy()
    feasible = feasible.sort_values(["Z1_total_cost_10k", "Z2_metro_service_percent"])
    front_rows = []
    best_service = -np.inf
    for _, row in feasible.iterrows():
        service = float(row["Z2_metro_service_percent"])
        if service >= best_service - 1e-12:
            front_rows.append(row)
            best_service = service
    return pd.DataFrame(front_rows)


def plot_pareto_front_comparison(points_df: pd.DataFrame, output_dir: Path) -> None:
    if points_df.empty:
        return
    points_df = points_df.copy()
    if "Z1_total_cost_10k" not in points_df.columns and "Z1_total_cost" in points_df.columns:
        points_df["Z1_total_cost_10k"] = points_df["Z1_total_cost"].astype(float) / 10000.0
    if "Z2_metro_service_percent" not in points_df.columns and "Z2_metro_service_ratio" in points_df.columns:
        points_df["Z2_metro_service_percent"] = points_df["Z2_metro_service_ratio"].astype(float) * 100.0
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    colors = {
        "QVNS-NSGA2": "#c8524a",
        "NSGA2": "#4f8cc9",
        "MOPSO": "#5aa06f",
        "MOEAD": "#8172b3",
    }
    for algorithm in ALGORITHMS:
        alg_points = points_df[points_df["algorithm"] == algorithm]
        if alg_points.empty:
            continue
        color = colors.get(algorithm, ALGORITHM_STYLES[algorithm]["color"])
        ax.scatter(
            alg_points["Z1_total_cost_10k"],
            alg_points["Z2_metro_service_percent"],
            s=14,
            alpha=0.16,
            color=color,
            edgecolors="none",
        )
        front = _pareto_front_for_plot(alg_points)
        if not front.empty:
            ax.plot(
                front["Z1_total_cost_10k"],
                front["Z2_metro_service_percent"],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                label=ALGORITHM_LABELS[algorithm],
            )
    ax.set_title("算法 Pareto 前沿对比", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("综合成本（万元/日）", fontsize=13)
    ax.set_ylabel("地铁服务比例（%）", fontsize=13)
    ax.grid(True, linestyle="-", alpha=0.22)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "fig_pareto_front_cost_service.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / "fig_pareto_front_cost_service.pdf", bbox_inches="tight")
    plt.close(fig)


def run_algorithm(algorithm: str, inst, pop_size: int, max_gen: int, seed: int, run_id: int, quiet: bool) -> tuple[dict, pd.DataFrame]:
    verbose = not quiet
    if algorithm == "QVNS-NSGA2":
        return run_qvns_tracked(inst, pop_size, max_gen, seed, run_id, verbose)
    if algorithm == "NSGA2":
        return run_nsga2_tracked(inst, pop_size, max_gen, seed, run_id, verbose)
    if algorithm == "MOPSO":
        return run_mopso_tracked(inst, pop_size, max_gen, seed, run_id, verbose)
    if algorithm == "SPEA2":
        return run_spea2_tracked(inst, pop_size, max_gen, seed, run_id, verbose)
    if algorithm == "MOEAD":
        return run_moead_tracked(inst, pop_size, max_gen, seed, run_id, verbose)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-algorithm convergence comparison with real carbon_cost.")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(OPTIONAL_ALGORITHMS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "four_algorithm_convergence")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_setting = _normalize_model_setting(args.model_setting)
    inst = build_metro_logistics_instance(args.scenario, top_n=args.top_n, model_setting=model_setting)
    all_histories: list[pd.DataFrame] = []
    all_pareto_points: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        for run_id in range(1, args.runs + 1):
            seed = int(args.seed + (run_id - 1) * 100 + list(ALGORITHMS).index(algorithm) * 10000)
            print(f"[{algorithm}] run {run_id}/{args.runs}, seed={seed}", flush=True)
            result, history = run_algorithm(algorithm, inst, args.pop_size, args.max_gen, seed, run_id, args.quiet)
            all_histories.append(history)
            all_pareto_points.append(extract_pareto_points(result, algorithm, run_id, seed))
            final_row = history.sort_values("gen").iloc[-1].to_dict()
            final_row["runtime_seconds"] = float(result.get("runtime", np.nan))
            final_row["final_pareto_size"] = int(len(result.get("pareto_objs", [])))
            summary_rows.append(final_row)
            raw_partial = add_derived_metrics(pd.concat(all_histories, ignore_index=True))
            raw_partial.to_csv(args.output_dir / "convergence_raw.csv", index=False, encoding="utf-8-sig")
    raw_df = add_derived_metrics(pd.concat(all_histories, ignore_index=True)) if all_histories else pd.DataFrame()
    aggregated = aggregate_convergence(raw_df)
    summary = pd.DataFrame(summary_rows)
    pareto_points = pd.concat(all_pareto_points, ignore_index=True) if all_pareto_points else pd.DataFrame()
    raw_df.to_csv(args.output_dir / "convergence_raw.csv", index=False, encoding="utf-8-sig")
    aggregated.to_csv(args.output_dir / "convergence_aggregated.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "final_summary.csv", index=False, encoding="utf-8-sig")
    pareto_points.to_csv(args.output_dir / "pareto_cost_service_points.csv", index=False, encoding="utf-8-sig")
    if not args.skip_plots and not aggregated.empty:
        plot_all(aggregated, args.output_dir)
        plot_thesis_combined_convergence(aggregated, args.output_dir)
        plot_dual_axis_iteration_effects(aggregated, args.output_dir)
        plot_pareto_front_comparison(pareto_points, args.output_dir)
    print(f"Done. Results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
