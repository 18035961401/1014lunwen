# -*- coding: utf-8 -*-
"""Thesis runner for QVNS-NSGA-II.

This script is a thin, reproducible entry point around the fifth-chapter model
and solver. It builds the real-data metro-logistics instance under the retained
the no-budget setting, runs QVNS-NSGA-II, and exports
Pareto/assignment/Q-table outputs.

Example
-------
Run the retained no-model setting::

    python code/run_qvns_nsga2_for_thesis.py --runs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from main import (
    _normalize_model_setting,
    base,
    build_metro_logistics_instance,
    export_qvns_nsga2_results,
    run_qvns_nsga2,
)

DEFAULT_OUTPUT_DIR = Path(base.OUTPUT_DIR) / "qvns_nsga2_thesis"
DEFAULT_MODEL_SETTINGS = ["no_budget_range"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QVNS-NSGA-II solver for the thesis metro-logistics model."
    )
    parser.add_argument("--scenario", default="base", help="Demand scenario passed to build_metro_logistics_instance.")
    parser.add_argument("--top-n", type=int, default=17, help="Number of CRITIC-TOPSIS candidate metro stations.")
    parser.add_argument("--candidate-mode", default="critic_topsis_top", choices=["critic_topsis_top", "transfer_all"], help="Candidate station input mode.")
    parser.add_argument("--model-setting", default="no_budget_range", help="Model setting. The rebuilt model supports only no_budget_range.")
    parser.add_argument("--model-settings", nargs="+", default=DEFAULT_MODEL_SETTINGS, help="Model settings to run; only no_budget_range is supported.")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size.")
    parser.add_argument("--max-gen", type=int, default=300, help="Maximum generation count.")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs per model setting.")
    parser.add_argument("--seed", type=int, default=20260429, help="Base random seed.")
    parser.add_argument("--pc", type=float, default=0.90, help="Base crossover probability.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Q-learning learning rate.")
    parser.add_argument("--gamma", type=float, default=0.88, help="Q-learning discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.30, help="Initial epsilon for epsilon-greedy action selection.")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Epsilon decay factor per generation; defaults to Q11C target decay.")
    parser.add_argument("--epsilon-min", type=float, default=0.010, help="Minimum epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Optional target final epsilon; if set, epsilon_decay is computed automatically.")
    parser.add_argument("--archive-size", type=int, default=None, help="External archive size.")
    parser.add_argument("--vns-interval", type=int, default=0, help="Run VNS local search every N generations; 0 uses adaptive VNS.")
    parser.add_argument("--vns-elite-rate", type=float, default=0.12, help="Elite fraction selected for VNS.")
    parser.add_argument("--vns-trials-per-neighborhood", type=int, default=2, help="VNS trials for each neighborhood.")
    parser.add_argument("--vns-max-neighborhood", type=int, default=3, help="Number of VNS neighborhoods to use; 5 enables the high-coverage reinforcement neighborhood.")
    parser.add_argument("--archive-feedback-interval", type=int, default=8, help="Inject archive elites every N generations; 0 disables feedback.")
    parser.add_argument("--archive-feedback-rate", type=float, default=0.15, help="Archive feedback replacement fraction.")
    parser.add_argument("--q-action-apply-rate", type=float, default=0.25, help="Fraction of offspring directly using the selected Q-learning action.")
    parser.add_argument("--q-action-profile", choices=["safe", "legacy"], default="safe", help="Q-learning action profile.")
    parser.add_argument("--disable-q-learning", action="store_true", help="Disable Q-learning and run the VNS-only ablation variant.")
    parser.add_argument("--disable-safeguard", action="store_true", help="Disable the archive-level Q-learning rollback safeguard.")
    parser.add_argument("--safeguard-threshold", type=float, default=-0.15, help="Reject Q-learning generation if archive reward is below this threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for all scenarios.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress generation-level solver logs.")
    parser.add_argument("--skip-plots", action="store_true", help="Export CSV files only and skip PNG/PDF plotting.")
    parser.add_argument(
        "--preserve-baseline",
        action="store_true",
        help="Append a standard NSGA-II preservation branch after Q-learning evolution. Default is off for fair comparison.",
    )
    return parser.parse_args(argv)


def resolve_model_settings(args: argparse.Namespace) -> list[str]:
    raw_values: Iterable[str] = args.model_settings if args.model_settings is not None else [args.model_setting]
    values: list[str] = []
    for value in raw_values:
        scenario = _normalize_model_setting(value)
        if scenario not in values:
            values.append(scenario)
    return values


def build_run_output_dir(root: Path, model_setting: str, run_id: int, total_runs: int) -> Path:
    if total_runs == 1:
        return root / model_setting
    return root / f"{model_setting}_run{run_id:02d}"


def _model_setting_seed_offset(model_setting: str) -> int:
    return 0


def summarize_run(model_setting: str, run_id: int, seed: int, output_dir: Path, pareto_df: pd.DataFrame, result: dict, inst) -> dict:
    row = {
        "algorithm": "QVNS-NSGA-II",
        "model_setting": model_setting,
        "fixed_cost_reference_lower": 0.0,
        "fixed_cost_reference_upper_all_stations": float(inst.fixed_cost.sum()),
        "run_id": run_id,
        "seed": seed,
        "pareto_size": int(len(pareto_df)),
        "runtime_seconds": float(result.get("runtime", 0.0)),
        "baseline_preservation": bool(result.get("baseline_preservation", False)),
        "output_dir": str(output_dir),
    }
    if len(pareto_df) > 0:
        row.update(
            {
                "min_total_cost": float(pareto_df["Z1_total_cost"].min()),
                "max_metro_service_ratio": float(pareto_df["Z2_metro_service_ratio"].max()),
                "min_cv": float(pareto_df["CV"].min()) if "CV" in pareto_df.columns else None,
                "min_fixed_cost": float(pareto_df["fixed_cost_total"].min()) if "fixed_cost_total" in pareto_df.columns else None,
                "max_fixed_cost": float(pareto_df["fixed_cost_total"].max()) if "fixed_cost_total" in pareto_df.columns else None,
                "min_num_open": int(pareto_df["num_open"].min()) if "num_open" in pareto_df.columns else None,
                "max_num_open": int(pareto_df["num_open"].max()) if "num_open" in pareto_df.columns else None,
                "max_line_load_rate": float(pareto_df["max_line_load_rate"].max()) if "max_line_load_rate" in pareto_df.columns else None,
                "max_line_flow": float(pareto_df["max_line_flow"].max()) if "max_line_flow" in pareto_df.columns else None,
                "max_line_capacity_violation": float(pareto_df["line_capacity_violation"].max()) if "line_capacity_violation" in pareto_df.columns else None,
                "max_line_open_count": int(pareto_df["max_line_open_count"].max()) if "max_line_open_count" in pareto_df.columns else None,
            }
        )
    else:
        row.update(
            {
                "min_total_cost": None,
                "max_metro_service_ratio": None,
                "min_cv": None,
                "min_fixed_cost": None,
                "max_fixed_cost": None,
                "min_num_open": None,
                "max_num_open": None,
                "max_line_load_rate": None,
                "max_line_flow": None,
                "max_line_capacity_violation": None,
                "max_line_open_count": None,
            }
        )
    return row



def _epsilon_decay_for_target(epsilon_start: float, epsilon_end: float, max_gen: int) -> float:
    epsilon_start = max(float(epsilon_start), 1e-12)
    epsilon_end = max(float(epsilon_end), 1e-12)
    return float((epsilon_end / epsilon_start) ** (1.0 / max(1, int(max_gen))))

def run_one(args: argparse.Namespace, model_setting: str, run_id: int, total_runs: int) -> dict:
    output_dir = build_run_output_dir(args.output_dir, model_setting, run_id, total_runs)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed + _model_setting_seed_offset(model_setting) + run_id)
    print(f"\n[QVNS-NSGA-II] Start setting={model_setting}, run={run_id}/{total_runs}, seed={seed}")
    print(f"[QVNS-NSGA-II] Output: {output_dir}")

    inst = build_metro_logistics_instance(scenario=args.scenario, top_n=args.top_n, model_setting=model_setting, candidate_mode=args.candidate_mode)

    start = time.time()
    epsilon_decay = args.epsilon_decay
    if args.epsilon_end is not None:
        epsilon_decay = _epsilon_decay_for_target(args.epsilon, args.epsilon_end, args.max_gen)
    elif epsilon_decay is None:
        epsilon_decay = _epsilon_decay_for_target(args.epsilon, args.epsilon_min, args.max_gen)
    result = run_qvns_nsga2(
        inst,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        seed=seed,
        verbose=not args.quiet,
        use_repair=True,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=(args.epsilon_end if args.epsilon_end is not None else args.epsilon_min),
        archive_size=args.archive_size,
        preserve_baseline=args.preserve_baseline,
        vns_interval=args.vns_interval,
        vns_elite_rate=args.vns_elite_rate,
        vns_trials_per_neighborhood=args.vns_trials_per_neighborhood,
        vns_max_neighborhood=args.vns_max_neighborhood,
        archive_feedback_interval=args.archive_feedback_interval,
        archive_feedback_rate=args.archive_feedback_rate,
        q_action_apply_rate=args.q_action_apply_rate,
        q_action_profile=args.q_action_profile,
        q_learning_enabled=not args.disable_q_learning,
        safeguard=not args.disable_safeguard,
        safeguard_threshold=args.safeguard_threshold,
        pc=args.pc,
    )
    result["runtime"] = float(result.get("runtime", time.time() - start))
    pareto_df = export_qvns_nsga2_results(result, inst, str(output_dir), use_repair=True, skip_plots=args.skip_plots)

    summary = summarize_run(model_setting, run_id, seed, output_dir, pareto_df, result, inst)
    print(
        "[QVNS-NSGA-II] Done "
        f"setting={model_setting}, run={run_id}, pareto_size={summary['pareto_size']}, "
        f"runtime={summary['runtime_seconds']:.2f}s"
    )
    return summary


def write_summary(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_settings = resolve_model_settings(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for model_setting in model_settings:
        for run_id in range(1, args.runs + 1):
            rows.append(run_one(args, model_setting, run_id, args.runs))

    summary_path = write_summary(rows, args.output_dir)
    print(f"\n[QVNS-NSGA-II] Summary saved: {summary_path}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
