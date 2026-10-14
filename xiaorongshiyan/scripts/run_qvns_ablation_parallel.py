# -*- coding: utf-8 -*-
"""Parallel QVNS-NSGA-II ablation experiment runner.

Runs each algorithm group in a separate process for maximum speedup.
Results are merged into a single output directory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import (
    _normalize_model_setting,
    build_metro_logistics_instance,
    choose_compromise,
    run_nsga2,
    run_qvns_nsga2,
)

# Import from original script
from run_qvns_ablation_experiment import (
    parse_args as _orig_parse_args,
    build_experiment_groups,
    _load_checkpoint,
    _completed_keys,
    _save_checkpoint,
    summarize_result,
    add_quality_indicators,
    make_summary,
    plot_pareto,
    plot_metrics,
)


def run_single_algorithm(args_tuple):
    """Run all runs for a single algorithm group in one process.
    
    Returns: (algorithm_name, run_rows, point_frames)
    """
    (algorithm, kind, group_type, kwargs, args_dict, scenario_idx, model_setting) = args_tuple
    
    # Reconstruct args
    args = argparse.Namespace(**args_dict)
    
    # Build instance
    inst = build_metro_logistics_instance(
        args.demand_scenario, top_n=args.top_n, model_setting=model_setting
    )
    
    archive_size = max(1, int(round(args.archive_multiplier * args.pop_size)))
    
    run_rows = []
    point_frames = []
    
    print(f"[PID {multiprocessing.current_process().pid}] Starting {algorithm} - {args.runs} runs")
    
    for run_id in range(1, args.runs + 1):
        seed = int(args.seed + 1000 * scenario_idx + run_id - 1)
        print(f"[PID {multiprocessing.current_process().pid}][{algorithm}] run={run_id}/{args.runs}, seed={seed}")
        
        started = time.time()
        if kind == "nsga2":
            result = run_nsga2(
                inst, pop_size=args.pop_size, max_gen=args.max_gen, seed=seed,
                verbose=False, archive_size=args.pop_size
            )
        else:
            run_kwargs = dict(kwargs)
            cfg_archive_size = int(run_kwargs.pop("archive_size", archive_size))
            result = run_qvns_nsga2(
                inst, pop_size=args.pop_size, max_gen=args.max_gen, seed=seed,
                verbose=False, archive_size=cfg_archive_size, **run_kwargs
            )
        
        row, pts = summarize_result(
            algorithm, model_setting, run_id, seed, result,
            time.time() - started, group_type=group_type, config=kwargs
        )
        run_rows.append(row)
        if len(pts):
            point_frames.append(pts)
    
    print(f"[PID {multiprocessing.current_process().pid}] Finished {algorithm} - {len(run_rows)} runs")
    return (algorithm, run_rows, point_frames)


def main_parallel(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run QVNS-NSGA-II ablation experiments in parallel.")
    parser.add_argument("--demand-scenario", default="base")
    parser.add_argument("--model-settings", nargs="+", default=["no_budget_range"])
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "bingxianjieguo")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--archive-multiplier", type=float, default=2.0)
    parser.add_argument("--vns-max-elites", type=int, default=20)
    parser.add_argument("--vns-trials", type=int, default=3)
    parser.add_argument("--comparison-mode", choices=["ablation", "tuning", "all"], default="all")
    parser.add_argument("--qvns-tuning-groups", nargs="+", 
                        default=["Q3", "Q9", "Q9A", "Q9B", "Q9C", "Q9D", "Q11", "Q11A", "Q11B", "Q11C"])
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of parallel processes. Default = min(CPU count, number of groups)")
    
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = []
    for value in args.model_settings:
        scenario = _normalize_model_setting(value)
        if scenario not in scenarios:
            scenarios.append(scenario)
    
    # Build all groups
    groups = build_experiment_groups(args)
    
    # Determine workers
    num_groups = len(groups) * len(scenarios)
    max_workers = args.workers or min(multiprocessing.cpu_count(), num_groups)
    max_workers = max(1, max_workers)
    
    print(f"=" * 60)
    print(f"Parallel QVNS-NSGA-II Ablation Experiment")
    print(f"=" * 60)
    print(f"Scenarios: {scenarios}")
    print(f"Groups: {len(groups)}")
    print(f"Runs per group: {args.runs}")
    print(f"Total runs: {len(groups) * args.runs * len(scenarios)}")
    print(f"Parallel workers: {max_workers}")
    print(f"Output: {args.output_dir}")
    print(f"=" * 60)
    
    # Prepare tasks
    tasks = []
    args_dict = vars(args)
    for scenario_idx, model_setting in enumerate(scenarios):
        for algorithm, kind, group_type, kwargs in groups:
            tasks.append((algorithm, kind, group_type, kwargs, args_dict, scenario_idx, model_setting))
    
    all_run_rows = []
    all_point_frames = []
    
    start_time = time.time()
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_algorithm, task): task for task in tasks}
        
        for future in as_completed(futures):
            algorithm, run_rows, point_frames = future.result()
            all_run_rows.extend(run_rows)
            all_point_frames.extend(point_frames)
            print(f"[Main] Collected {algorithm}: {len(run_rows)} runs, {len(point_frames)} point frames")
    
    elapsed = time.time() - start_time
    print(f"\nAll groups completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Save final results
    run_df, points_df, summary_df = _save_checkpoint(
        all_run_rows, all_point_frames, args.output_dir, make_plots=not args.skip_plots
    )
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - ablation_run_metrics.csv: {len(run_df)} runs")
    print(f"  - ablation_pareto_points.csv: {len(points_df)} points")
    print(f"  - ablation_summary.csv: {len(summary_df)} groups")
    
    if len(summary_df):
        print("\n" + summary_df.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main_parallel())
