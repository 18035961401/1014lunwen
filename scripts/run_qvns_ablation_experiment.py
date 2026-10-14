# -*- coding: utf-8 -*-
"""Run ablation and QVNS parameter-tuning experiments.

The script can compare the original ablation groups and enhanced QVNS tuning
configurations in one run:

Ablation groups:
  1) NSGA-II baseline;
  2) VNS-only: VNS enabled, Q-learning disabled;
  3) Q-learning-only: Q-learning enabled, VNS disabled.

QVNS tuning groups:
  Q3 balanced reference setting;
  Q3 balanced reference setting;
  Q9 adaptive fast VNS baseline;
  Q9A/Q9B/Q9C/Q9D adaptive-VNS branches for speed, stability, diversity, and balanced performance;
  Q11 integrated archive/adaptive VNS setting;
  Q11A/Q11B/Q11C integrated branches for stability, distribution diversity, and fast integrated search.
Legacy Q1--Q8/Q10 groups are still accepted for reproduction.

Default mode is ``all`` so the output CSV can be used to compare ablation and
QVNS parameter tuning directly. Use ``--comparison-mode ablation`` or
``--comparison-mode tuning`` to run only one part.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import (  # noqa: E402
    _normalize_model_setting,
    build_metro_logistics_instance,
    choose_compromise,
    run_nsga2,
    run_qvns_nsga2,
)

DEFAULT_SCENARIOS = ["no_budget_range"]

# SVG font fallback: do not bundle fonts; rely on the user's OS.
SVG_FONT_FAMILY = (
    '"' + __import__("os").environ.get("QVNS_SVG_FONT", "Microsoft YaHei") + '",'
    '"SimHei","Noto Sans CJK SC","Source Han Sans SC",'
    '"PingFang SC","Heiti SC","WenQuanYi Micro Hei",'
    '"Arial Unicode MS","DejaVu Sans",sans-serif'
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run QVNS-NSGA-II ablation and tuning experiments.")
    parser.add_argument("--demand-scenario", default="base", help="Demand scenario: low, base, or high.")
    parser.add_argument("--model-settings", "--settings", dest="model_settings", nargs="+", default=DEFAULT_SCENARIOS, help="Model settings to run; only no_budget_range is supported.")
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--pop-size", type=int, default=100, help="Use 100 for thesis-grade runs.")
    parser.add_argument("--max-gen", type=int, default=300, help="Use 300 for thesis-grade runs.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "qvns_ablation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Only export CSV metrics and skip SVG figures.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume; overwrite and rerun all combinations.")
    parser.add_argument("--archive-multiplier", type=float, default=2.0, help="QVNS archive size = multiplier * pop_size; Q7-Q11 may override this per group.")
    parser.add_argument("--vns-max-elites", type=int, default=20, help="Cap the number of VNS elites per VNS generation for long runs.")
    parser.add_argument("--vns-trials", type=int, default=3, help="VNS trials per neighborhood used by VNS-only and Q1 baseline.")
    parser.add_argument(
        "--comparison-mode",
        choices=["ablation", "tuning", "all"],
        default="all",
        help="ablation=NSGA-II/VNS-only/Q-learning-only; tuning=QVNS Q1-Q6; all=both.",
    )
    parser.add_argument(
        "--qvns-tuning-groups",
        nargs="+",
        default=["Q3", "Q9", "Q9A", "Q9B", "Q9C", "Q9D", "Q11", "Q11A", "Q11B", "Q11C"],
        help="QVNS tuning groups to include when comparison-mode is tuning/all. Choices: Q1..Q11, Q9A..Q9D, and Q11A..Q11C; default is Q3 Q9 Q9A Q9B Q9C Q9D Q11 Q11A Q11B Q11C.",
    )
    return parser.parse_args(argv)


def _epsilon_decay_for_target(epsilon_start: float, epsilon_end: float, max_gen: int) -> float:
    epsilon_start = max(float(epsilon_start), 1e-12)
    epsilon_end = max(float(epsilon_end), 1e-12)
    return float((epsilon_end / epsilon_start) ** (1.0 / max(1, int(max_gen))))


def _qvns_config(name: str, args) -> dict[str, Any]:
    """Return one QVNS tuning configuration corresponding to Q1--Q11 and branch groups."""
    key = str(name).upper().strip()
    common = dict(
        q_learning_enabled=True,
        q_action_apply_rate=0.25,
        q_action_profile="safe",
        safeguard=True,
        safeguard_threshold=-0.15,
        archive_feedback_interval=8,
        archive_feedback_rate=0.15,
        vns_max_elites=args.vns_max_elites,
    )
    if key == "Q1":
        cfg = dict(
            alpha=0.06, gamma=0.75, epsilon=0.10, epsilon_min=0.005,
            epsilon_decay=0.93, vns_interval=2, vns_elite_rate=0.25,
            vns_trials_per_neighborhood=args.vns_trials, vns_max_neighborhood=4,
            tuning_goal="control_current_baseline",
        )
    elif key == "Q2":
        cfg = dict(
            alpha=0.20, gamma=0.85, epsilon=0.30, epsilon_min=0.03,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.03, args.max_gen),
            vns_interval=8, vns_elite_rate=0.15, vns_trials_per_neighborhood=3,
            vns_max_neighborhood=4, tuning_goal="stable_improvement",
        )
    elif key == "Q3":
        cfg = dict(
            alpha=0.20, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=5, vns_elite_rate=0.18, vns_trials_per_neighborhood=4,
            vns_max_neighborhood=4, tuning_goal="balanced_hv_runtime",
        )
    elif key == "Q4":
        cfg = dict(
            alpha=0.20, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=3, vns_elite_rate=0.20, vns_trials_per_neighborhood=5,
            vns_max_neighborhood=5, tuning_goal="strong_vns_hv",
        )
    elif key == "Q5":
        cfg = dict(
            alpha=0.20, gamma=0.85, epsilon=0.30, epsilon_min=0.03,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.03, args.max_gen),
            vns_interval=5, vns_elite_rate=0.20, vns_trials_per_neighborhood=4,
            vns_max_neighborhood=5, tuning_goal="high_coverage",
        )
    elif key == "Q6":
        cfg = dict(
            alpha=0.20, gamma=0.85, epsilon=0.30, epsilon_min=0.01,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.01, args.max_gen),
            vns_interval=10, vns_elite_rate=0.15, vns_trials_per_neighborhood=3,
            vns_max_neighborhood=4,
            qvns_phase_schedule=[
                {"until_ratio": 1/3, "vns_interval": 10, "vns_elite_rate": 0.15, "vns_trials_per_neighborhood": 3, "vns_max_neighborhood": 4},
                {"until_ratio": 2/3, "vns_interval": 5, "vns_elite_rate": 0.18, "vns_trials_per_neighborhood": 4, "vns_max_neighborhood": 4},
                {"until_ratio": 1.0, "vns_interval": 3, "vns_elite_rate": 0.22, "vns_trials_per_neighborhood": 5, "vns_max_neighborhood": 4},
            ],
            tuning_goal="late_stage_reinforcement",
        )
    elif key == "Q7":
        cfg = dict(
            alpha=0.15, gamma=0.90, epsilon=0.30, epsilon_min=0.005,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.005, args.max_gen),
            vns_interval=5, vns_mode="fixed", vns_elite_rate=0.16,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, 3 * args.pop_size),
            archive_boundary_keep_rate=0.15, vns_candidate_strategy="mixed",
            tuning_goal="stability_archive_unique_points",
        )
    elif key == "Q8":
        cfg = dict(
            alpha=0.18, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=6, vns_mode="fixed", vns_elite_rate=0.15,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, 4 * args.pop_size),
            archive_boundary_keep_rate=0.25, vns_candidate_strategy="mixed",
            tuning_goal="spacing_boundary_diversity",
        )
    elif key == "Q9":
        cfg = dict(
            alpha=0.18, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.12,
            vns_trials_per_neighborhood=2, vns_max_neighborhood=3,
            archive_size=max(args.pop_size, int(round(2.5 * args.pop_size))),
            archive_boundary_keep_rate=0.20, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=3,
            tuning_goal="adaptive_fast_runtime",
        )
    elif key == "Q9A":
        # 更快分支：更严格地限制 VNS 调用，目标是降低运行时间并保持 Q9 的基本前沿质量。
        cfg = dict(
            alpha=0.18, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.10,
            vns_trials_per_neighborhood=2, vns_max_neighborhood=3,
            archive_size=max(args.pop_size, int(round(2.0 * args.pop_size))),
            archive_boundary_keep_rate=0.18, vns_candidate_strategy="mixed",
            stagnation_window=10, hv_improvement_threshold=0.0025, min_vns_gap=5,
            tuning_goal="q9_branch_fast_adaptive_vns",
        )
    elif key == "Q9B":
        # 稳定分支：降低后期随机性并提高折扣因子，目标是降低 HV 标准差。
        cfg = dict(
            alpha=0.15, gamma=0.90, epsilon=0.30, epsilon_min=0.008,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.008, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.12,
            vns_trials_per_neighborhood=2, vns_max_neighborhood=3,
            archive_size=max(args.pop_size, int(round(2.5 * args.pop_size))),
            archive_boundary_keep_rate=0.20, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=4,
            tuning_goal="q9_branch_stability",
        )
    elif key == "Q9C":
        # 多样性分支：扩大 archive 与边界解保留，目标是提高唯一解数并降低 Spread/Spacing。
        cfg = dict(
            alpha=0.18, gamma=0.85, epsilon=0.30, epsilon_min=0.02,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.02, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.14,
            vns_trials_per_neighborhood=2, vns_max_neighborhood=3,
            archive_size=max(args.pop_size, int(round(4.0 * args.pop_size))),
            archive_boundary_keep_rate=0.30, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=3,
            tuning_goal="q9_branch_diversity_archive",
        )
    elif key == "Q9D":
        # 平衡增强分支：比 Q9 略强的 VNS 与 archive，目标是兼顾 HV、IGD、Spread 和时间。
        cfg = dict(
            alpha=0.16, gamma=0.88, epsilon=0.30, epsilon_min=0.012,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.012, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.15,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, int(round(3.0 * args.pop_size))),
            archive_boundary_keep_rate=0.25, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=4,
            tuning_goal="q9_branch_balanced_quality",
        )
    elif key == "Q10":
        cfg = dict(
            alpha=0.15, gamma=0.90, epsilon=0.30, epsilon_min=0.008,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.008, args.max_gen),
            vns_interval=5, vns_mode="fixed", vns_elite_rate=0.16,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, int(round(3.5 * args.pop_size))),
            archive_boundary_keep_rate=0.25, vns_candidate_strategy="mixed",
            tuning_goal="stability_spacing",
        )
    elif key == "Q11":
        cfg = dict(
            alpha=0.15, gamma=0.90, epsilon=0.30, epsilon_min=0.008,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.008, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.18,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, int(round(3.5 * args.pop_size))),
            archive_boundary_keep_rate=0.25, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=3,
            tuning_goal="integrated_archive_adaptive_vns",
        )

    elif key == "Q11A":
        # 稳定收敛分支：比 Q11 更低后期随机性、更平滑的 Q-learning 更新，目标是降低 HV 标准差和 IGD 波动。
        cfg = dict(
            alpha=0.12, gamma=0.92, epsilon=0.30, epsilon_min=0.005,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.005, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.16,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, int(round(3.0 * args.pop_size))),
            archive_boundary_keep_rate=0.18, vns_candidate_strategy="mixed",
            stagnation_window=10, hv_improvement_threshold=0.002, min_vns_gap=4,
            tuning_goal="q11_branch_stable_convergence",
        )
    elif key == "Q11B":
        # 分布多样性分支：扩大 archive 和边界保留，目标是降低 Spacing/Spread 并提高唯一解数。
        cfg = dict(
            alpha=0.16, gamma=0.88, epsilon=0.30, epsilon_min=0.012,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.012, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.18,
            vns_trials_per_neighborhood=3, vns_max_neighborhood=4,
            archive_size=max(args.pop_size, int(round(5.0 * args.pop_size))),
            archive_boundary_keep_rate=0.35, vns_candidate_strategy="mixed",
            stagnation_window=8, hv_improvement_threshold=0.003, min_vns_gap=3,
            tuning_goal="q11_branch_distribution_diversity",
        )
    elif key == "Q11C":
        # 快速综合分支：减少 VNS 个体比例和尝试次数，目标是降低运行时间并保持综合前沿质量。
        cfg = dict(
            alpha=0.15, gamma=0.88, epsilon=0.30, epsilon_min=0.010,
            epsilon_decay=_epsilon_decay_for_target(0.30, 0.010, args.max_gen),
            vns_interval=0, vns_mode="adaptive", vns_elite_rate=0.12,
            vns_trials_per_neighborhood=2, vns_max_neighborhood=3,
            archive_size=max(args.pop_size, int(round(2.5 * args.pop_size))),
            archive_boundary_keep_rate=0.20, vns_candidate_strategy="mixed",
            stagnation_window=10, hv_improvement_threshold=0.0025, min_vns_gap=5,
            tuning_goal="q11_branch_fast_integrated",
        )
    else:
        raise ValueError(f"Unknown QVNS tuning group: {name!r}; choose from Q1..Q11, Q9A..Q9D, or Q11A..Q11C")
    cfg.update(common)
    cfg["tuning_group"] = key
    return cfg


def build_experiment_groups(args):
    groups: list[tuple[str, str, str, dict[str, Any]]] = []
    if args.comparison_mode in {"ablation", "all"}:
        groups.extend([
            ("NSGA-II", "nsga2", "ablation", {}),
            ("VNS-only", "qvns", "ablation", dict(
                q_learning_enabled=False, q_action_apply_rate=0.0,
                epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0,
                vns_max_elites=args.vns_max_elites,
                vns_trials_per_neighborhood=args.vns_trials,
                vns_max_neighborhood=4,
            )),
            ("Q-learning-only", "qvns", "ablation", dict(
                vns_interval=0, archive_feedback_interval=0,
                q_action_apply_rate=0.25, q_action_profile="safe",
                safeguard=True, safeguard_threshold=-0.15,
            )),
        ])
    if args.comparison_mode in {"tuning", "all"}:
        seen: list[str] = []
        for name in args.qvns_tuning_groups:
            key = str(name).upper().strip()
            if key not in seen:
                seen.append(key)
        for key in seen:
            groups.append((f"QVNS-{key}", "qvns", "qvns_tuning", _qvns_config(key, args)))
    return groups


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
    prev_y = float(ref[1])
    for x, y in pts:
        if x <= ref[0] and y < prev_y:
            hv += max(0.0, ref[0] - x) * max(0.0, prev_y - y)
            prev_y = y
    return float(hv)


def spacing(points: np.ndarray) -> float:
    if len(points) <= 2:
        return 0.0
    nearest = []
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        nearest.append(np.abs(others - p).sum(axis=1).min())
    nearest = np.array(nearest)
    return float(np.sqrt(((nearest - nearest.mean()) ** 2).sum() / (len(nearest) - 1)))


def igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted Generational Distance on normalized minimization objectives.

    Lower is better.  The reference front is built from the nondominated union
    of all algorithms/runs in the same model setting.
    """
    points = np.asarray(points, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(points) == 0 or len(reference_front) == 0:
        return float("nan")
    dists = []
    for ref in reference_front:
        dists.append(np.linalg.norm(points - ref, axis=1).min())
    return float(np.mean(dists))


def spread_delta(points: np.ndarray, reference_front: np.ndarray) -> float:
    """Deb-style spread/diversity metric for two normalized objectives.

    Lower is better.  It penalizes poor coverage of the two extreme reference
    points and uneven spacing along the approximation front.
    """
    points = np.asarray(points, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(points) <= 1 or len(reference_front) <= 1:
        return 1.0
    pts = points[nondominated(points)]
    if len(pts) <= 1:
        return 1.0
    pts = pts[np.argsort(pts[:, 0])]
    ref = reference_front[np.argsort(reference_front[:, 0])]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d_bar = float(np.mean(d)) if len(d) else 0.0
    d_f = float(np.linalg.norm(pts[0] - ref[0]))
    d_l = float(np.linalg.norm(pts[-1] - ref[-1]))
    denom = d_f + d_l + max(0, len(pts) - 1) * d_bar
    if denom <= 1e-12:
        return 0.0
    return float((d_f + d_l + np.sum(np.abs(d - d_bar))) / denom)


def summarize_result(
    algorithm: str,
    model_setting: str,
    run_id: int,
    seed: int,
    result: dict,
    runtime: float,
    group_type: str = "",
    config: dict[str, Any] | None = None,
) -> tuple[dict, pd.DataFrame]:
    objs = np.asarray(result["pareto_objs"], dtype=float)
    cvs = np.asarray(result.get("pareto_cvs", np.zeros(len(objs))), dtype=float)
    feasible = cvs <= 1e-9
    if np.any(feasible):
        objs = objs[feasible]
        cvs = cvs[feasible]
    base_row = {
        "algorithm": algorithm,
        "group_type": group_type,
        "model_setting": model_setting,
        "run_id": run_id,
        "seed": seed,
        "runtime_seconds": float(runtime),
    }
    if config:
        for key in [
            "tuning_group", "tuning_goal", "vns_interval", "vns_mode", "vns_elite_rate",
            "vns_trials_per_neighborhood", "vns_max_neighborhood", "epsilon",
            "epsilon_min", "epsilon_decay", "alpha", "gamma", "q_action_apply_rate",
            "archive_size", "archive_boundary_keep_rate", "vns_candidate_strategy",
            "stagnation_window", "hv_improvement_threshold", "min_vns_gap",
        ]:
            if key in config:
                base_row[key] = config[key]
        if "qvns_phase_schedule" in config:
            base_row["qvns_phase_schedule"] = str(config["qvns_phase_schedule"])
    if len(objs) == 0:
        row = dict(base_row)
        row.update({
            "pareto_size": 0,
            "unique_points": 0,
            "min_cost": np.nan,
            "max_service_ratio": np.nan,
            "topsis_cost": np.nan,
            "topsis_service_ratio": np.nan,
            "safeguard_rejects": 0,
        })
        return row, pd.DataFrame()
    compromise_idx = choose_compromise(objs)
    points_df = pd.DataFrame({
        "algorithm": algorithm,
        "group_type": group_type,
        "model_setting": model_setting,
        "run_id": run_id,
        "seed": seed,
        "Z1_total_cost": objs[:, 0],
        "Z2_metro_service_ratio": -objs[:, 1],
        "CV": cvs,
    })
    action_history = result.get("action_history")
    rejects = 0
    if isinstance(action_history, pd.DataFrame) and "rejected_by_safeguard" in action_history:
        rejects = int(action_history["rejected_by_safeguard"].sum())
    row = dict(base_row)
    row.update({
        "pareto_size": int(len(objs)),
        "unique_points": int(len(np.unique(np.round(points_df[["Z1_total_cost", "Z2_metro_service_ratio"]].to_numpy(), 8), axis=0))),
        "min_cost": float(objs[:, 0].min()),
        "max_service_ratio": float((-objs[:, 1]).max()),
        "topsis_cost": float(objs[compromise_idx, 0]),
        "topsis_service_ratio": float(-objs[compromise_idx, 1]),
        "safeguard_rejects": rejects,
        "actual_vns_calls": int(result.get("actual_vns_calls", 0)),
        "archive_final_size": int(result.get("archive_final_size", 0)),
        "archive_unique_solutions": int(result.get("archive_unique_solutions", 0)),
    })
    return row, points_df


def add_quality_indicators(run_df: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    """Add HV, Spacing, IGD and Spread for every algorithm/run.

    All indicators are computed in normalized minimization space:
    cost is minimized, and service ratio is converted to ``1 - normalized service``.
    For IGD and Spread, the reference front is the nondominated union of all
    algorithm/run points within the same model setting.
    """
    rows = []
    for scenario, sdf in points_df.groupby("model_setting"):
        cost_min, cost_max = sdf["Z1_total_cost"].min(), sdf["Z1_total_cost"].max()
        srv_min, srv_max = sdf["Z2_metro_service_ratio"].min(), sdf["Z2_metro_service_ratio"].max()
        all_normalized = np.column_stack([
            (sdf["Z1_total_cost"].to_numpy() - cost_min) / (cost_max - cost_min + 1e-12),
            (srv_max - sdf["Z2_metro_service_ratio"].to_numpy()) / (srv_max - srv_min + 1e-12),
        ])
        reference_front = all_normalized[nondominated(all_normalized)] if len(all_normalized) else np.empty((0, 2))
        for _, row in run_df[run_df["model_setting"] == scenario].iterrows():
            pts = sdf[(sdf["algorithm"] == row["algorithm"]) & (sdf["run_id"] == row["run_id"])]
            row = row.copy()
            if len(pts) == 0:
                row["HV"] = 0.0
                row["Spacing"] = 0.0
                row["IGD"] = np.nan
                row["Spread"] = np.nan
            else:
                normalized = np.column_stack([
                    (pts["Z1_total_cost"].to_numpy() - cost_min) / (cost_max - cost_min + 1e-12),
                    (srv_max - pts["Z2_metro_service_ratio"].to_numpy()) / (srv_max - srv_min + 1e-12),
                ])
                row["HV"] = hv_2d(normalized)
                row["Spacing"] = spacing(normalized)
                row["IGD"] = igd(normalized, reference_front)
                row["Spread"] = spread_delta(normalized, reference_front)
            rows.append(row)
    return pd.DataFrame(rows)


def make_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_setting", "group_type", "algorithm"]
    return (
        run_df.groupby(group_cols, dropna=False)
        .agg(
            runs=("run_id", "count"),
            HV_mean=("HV", "mean"),
            HV_std=("HV", "std"),
            Spacing_mean=("Spacing", "mean"),
            IGD_mean=("IGD", "mean"),
            IGD_std=("IGD", "std"),
            Spread_mean=("Spread", "mean"),
            Spread_std=("Spread", "std"),
            unique_points_mean=("unique_points", "mean"),
            min_cost_mean=("min_cost", "mean"),
            max_service_ratio_mean=("max_service_ratio", "mean"),
            topsis_cost_mean=("topsis_cost", "mean"),
            topsis_service_ratio_mean=("topsis_service_ratio", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            safeguard_rejects_mean=("safeguard_rejects", "mean"),
            actual_vns_calls_mean=("actual_vns_calls", "mean"),
            archive_final_size_mean=("archive_final_size", "mean"),
            archive_unique_solutions_mean=("archive_unique_solutions", "mean"),
        )
        .reset_index()
        .sort_values(["model_setting", "HV_mean"], ascending=[True, False])
    )


def _svg_escape(value) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_svg(path: Path, elements: list[str], width: int = 900, height: int = 620) -> None:
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append(f'<style>text{{font-family:{SVG_FONT_FAMILY};}} .title{{font-weight:600;}}</style>')
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg.extend(elements)
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def plot_pareto(points_df: pd.DataFrame, output_dir: Path) -> None:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]
    for scenario, sdf in points_df.groupby("model_setting"):
        width, height = 960, 650
        left, right, top, bottom = 90, 260, 70, 80
        plot_w = width - left - right
        plot_h = height - top - bottom
        x_min, x_max = float(sdf["Z1_total_cost"].min()), float(sdf["Z1_total_cost"].max())
        y_min, y_max = float(sdf["Z2_metro_service_ratio"].min()), float(sdf["Z2_metro_service_ratio"].max())
        if abs(x_max - x_min) < 1e-12:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-12:
            y_max = y_min + 0.01
        def sx(x): return left + (float(x) - x_min) / (x_max - x_min) * plot_w
        def sy(y): return top + (y_max - float(y)) / (y_max - y_min) * plot_h
        elements = [
            f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20">消融与QVNS调优 Pareto 前沿：{_svg_escape(scenario)}</text>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
            f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="14">Z1 总成本（元/日）</text>',
            f'<text x="25" y="{top+plot_h/2}" transform="rotate(-90 25 {top+plot_h/2})" text-anchor="middle" font-size="14">Z2 地铁服务占比</text>',
        ]
        for tick in range(5):
            x = left + tick * plot_w / 4
            y = top + tick * plot_h / 4
            xv = x_min + tick * (x_max - x_min) / 4
            yv = y_max - tick * (y_max - y_min) / 4
            elements.append(f'<line x1="{x}" y1="{top+plot_h}" x2="{x}" y2="{top+plot_h+5}" stroke="#333"/>')
            elements.append(f'<text x="{x}" y="{top+plot_h+22}" text-anchor="middle" font-size="11">{xv:,.0f}</text>')
            elements.append(f'<line x1="{left-5}" y1="{y}" x2="{left}" y2="{y}" stroke="#333"/>')
            elements.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{yv:.3f}</text>')
        for idx, (algorithm, adf) in enumerate(sdf.groupby("algorithm", sort=False)):
            color = palette[idx % len(palette)]
            for _, row in adf.iterrows():
                elements.append(f'<circle cx="{sx(row["Z1_total_cost"]):.2f}" cy="{sy(row["Z2_metro_service_ratio"]):.2f}" r="4.5" fill="{color}" fill-opacity="0.78"/>')
            ly = top + 25 + idx * 22
            elements.append(f'<rect x="{left+plot_w+35}" y="{ly-11}" width="14" height="14" fill="{color}"/>')
            elements.append(f'<text x="{left+plot_w+56}" y="{ly}" font-size="12">{_svg_escape(algorithm)}</text>')
        _write_svg(output_dir / f"fig_ablation_pareto_{scenario}.svg", elements, width, height)


def plot_metrics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("HV_mean", "HV 均值", "hv"),
        ("IGD_mean", "IGD 均值（越小越好）", "igd"),
        ("Spread_mean", "Spread 均值（越小越好）", "spread"),
        ("Spacing_mean", "Spacing 均值（越小越好）", "spacing"),
        ("min_cost_mean", "最小成本", "min_cost"),
    ]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]
    for scenario, sdf in summary_df.groupby("model_setting"):
        sdf = sdf.sort_values("HV_mean", ascending=False)
        for metric, title, fname in metrics:
            width, height = 980, 580
            left, right, top, bottom = 90, 40, 70, 150
            plot_w = width - left - right
            plot_h = height - top - bottom
            vals = sdf[metric].fillna(0).astype(float).to_numpy()
            vmax = max(float(vals.max()) if len(vals) else 1.0, 1e-9)
            elements = [
                f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20">{_svg_escape(title)}: {_svg_escape(scenario)}</text>',
                f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
            ]
            n = max(1, len(sdf))
            bar_w = plot_w / n * 0.62
            for idx, (_, row) in enumerate(sdf.iterrows()):
                val = float(row[metric]) if pd.notna(row[metric]) else 0.0
                h = (val / vmax) * plot_h if vmax else 0.0
                x = left + (idx + 0.5) * plot_w / n - bar_w / 2
                y = top + plot_h - h
                color = palette[idx % len(palette)]
                elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" fill-opacity="0.85"/>')
                elements.append(f'<text x="{x+bar_w/2:.2f}" y="{y-6:.2f}" text-anchor="middle" font-size="11">{val:.4g}</text>')
                label = _svg_escape(row["algorithm"])
                elements.append(f'<text x="{x+bar_w/2:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-size="10" transform="rotate(25 {x+bar_w/2:.2f} {top+plot_h+24})">{label}</text>')
            _write_svg(output_dir / f"fig_ablation_{fname}_{scenario}.svg", elements, width, height)


def _load_checkpoint(output_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not resume:
        return pd.DataFrame(), pd.DataFrame()
    run_path = output_dir / "ablation_run_metrics.csv"
    points_path = output_dir / "ablation_pareto_points.csv"
    run_df = pd.read_csv(run_path, encoding="utf-8-sig") if run_path.exists() else pd.DataFrame()
    points_df = pd.read_csv(points_path, encoding="utf-8-sig") if points_path.exists() else pd.DataFrame()
    return run_df, points_df


def _completed_keys(run_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    if run_df.empty:
        return set()
    return set(zip(run_df["model_setting"].astype(str), run_df["algorithm"].astype(str), run_df["run_id"].astype(int)))


def _save_checkpoint(run_rows: list[dict], point_frames: list[pd.DataFrame], output_dir: Path, make_plots: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_df = pd.DataFrame(run_rows)
    points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    if len(points_df):
        run_df = add_quality_indicators(run_df, points_df)
    elif len(run_df):
        run_df = run_df.assign(HV=0.0, Spacing=0.0)
    summary_df = make_summary(run_df) if len(run_df) else pd.DataFrame()
    run_df.to_csv(output_dir / "ablation_run_metrics.csv", index=False, encoding="utf-8-sig")
    points_df.to_csv(output_dir / "ablation_pareto_points.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    if len(points_df) and make_plots:
        plot_pareto(points_df, output_dir)
        plot_metrics(summary_df, output_dir)
    return run_df, points_df, summary_df


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: list[str] = []
    for value in args.model_settings:
        scenario = _normalize_model_setting(value)
        if scenario not in scenarios:
            scenarios.append(scenario)

    archive_size = max(1, int(round(args.archive_multiplier * args.pop_size)))
    groups = build_experiment_groups(args)

    existing_run_df, existing_points_df = _load_checkpoint(args.output_dir, resume=not args.no_resume)
    completed = _completed_keys(existing_run_df)
    run_rows = existing_run_df.to_dict("records") if len(existing_run_df) else []
    point_frames = [existing_points_df] if len(existing_points_df) else []
    if completed:
        print(f"Resume enabled: {len(completed)} completed run(s) loaded from {args.output_dir}")
    for model_setting in scenarios:
        inst = build_metro_logistics_instance(args.demand_scenario, top_n=args.top_n, model_setting=model_setting)
        for algorithm, kind, group_type, kwargs in groups:
            for run_id in range(1, args.runs + 1):
                seed = int(args.seed + 1000 * scenarios.index(model_setting) + run_id - 1)
                key = (model_setting, algorithm, run_id)
                if key in completed:
                    print(f"[{model_setting}][{algorithm}] run={run_id}/{args.runs} skipped (checkpoint exists)")
                    continue
                print(f"[{model_setting}][{algorithm}] run={run_id}/{args.runs}, seed={seed}")
                started = time.time()
                if kind == "nsga2":
                    result = run_nsga2(inst, pop_size=args.pop_size, max_gen=args.max_gen, seed=seed, verbose=not args.quiet, archive_size=args.pop_size)
                else:
                    run_kwargs = dict(kwargs)
                    cfg_archive_size = int(run_kwargs.pop("archive_size", archive_size))
                    result = run_qvns_nsga2(
                        inst, pop_size=args.pop_size, max_gen=args.max_gen, seed=seed,
                        verbose=not args.quiet, archive_size=cfg_archive_size, **run_kwargs
                    )
                row, pts = summarize_result(algorithm, model_setting, run_id, seed, result, time.time() - started, group_type=group_type, config=kwargs)
                run_rows.append(row)
                if len(pts):
                    point_frames.append(pts)
                completed.add(key)
                _save_checkpoint(run_rows, point_frames, args.output_dir, make_plots=not args.skip_plots)

    _, _, summary_df = _save_checkpoint(run_rows, point_frames, args.output_dir, make_plots=not args.skip_plots)
    print(summary_df.to_string(index=False) if len(summary_df) else "No results generated.")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
