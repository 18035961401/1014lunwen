# -*- coding: utf-8 -*-
"""Plot representative Wuhou maps for 3/4/5 km service-radius sensitivity.

The script reads the sensitivity run metrics and the saved compromise tables,
selects one representative run per radius, rebuilds the corresponding instance,
and exports a map in a radius-specific folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "code" / "main.py").exists():
            return path
    raise RuntimeError("Cannot locate project root containing code/main.py")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import build_metro_logistics_instance, plot_compromise_map  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "service_radius_sensitivity_results")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "service_radius_maps")
    parser.add_argument("--radii", nargs="+", type=float, default=[3.0, 4.0, 5.0])
    parser.add_argument(
        "--selection",
        choices=["median", "best_service", "best_cost"],
        default="median",
        help="How to choose a representative run for each radius.",
    )
    parser.add_argument("--demand-scenario", default="base")
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument(
        "--use-manuscript-r4",
        action="store_true",
        help="Use the thesis manuscript representative solution for radius 4 km.",
    )
    parser.add_argument(
        "--manuscript-r4-dir",
        type=Path,
        default=PROJECT_ROOT / "xiaorongshiyan" / "results" / "chapter5_q11_results" / "representative_solution",
        help="Folder containing qvns_nsga2_compromise_assignments.csv and qvns_nsga2_station_load.csv.",
    )
    return parser.parse_args(argv)


def radius_folder_name(radius_km: float) -> str:
    return f"radius_{radius_km:.1f}km".replace(".", "p")


def select_run(run_metrics: pd.DataFrame, radius_km: float, selection: str) -> pd.Series:
    subset = run_metrics[run_metrics["radius_km"].astype(float) == float(radius_km)].copy()
    if subset.empty:
        raise ValueError(f"No run metrics found for radius {radius_km}")
    subset = subset.sort_values(["topsis_service_ratio", "topsis_cost", "run_id"], ascending=[True, True, True])
    if selection == "best_service":
        return subset.sort_values(["topsis_service_ratio", "topsis_cost"], ascending=[False, True]).iloc[0]
    if selection == "best_cost":
        return subset.sort_values(["topsis_cost", "topsis_service_ratio"], ascending=[True, False]).iloc[0]
    median_value = subset["topsis_service_ratio"].median()
    subset["distance_to_median"] = (subset["topsis_service_ratio"] - median_value).abs()
    return subset.sort_values(["distance_to_median", "topsis_cost", "run_id"], ascending=[True, True, True]).iloc[0]


def load_tables(base_dir: Path, radius_km: float, run_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    radius_dir = base_dir / "compromise_tables" / radius_folder_name(radius_km) / f"run_{run_id:02d}"
    assign_path = radius_dir / "service_radius_compromise_assignments.csv"
    station_path = radius_dir / "service_radius_compromise_station_load.csv"
    if not assign_path.exists() or not station_path.exists():
        raise FileNotFoundError(f"Missing compromise tables under {radius_dir}")
    assign_df = pd.read_csv(assign_path, encoding="utf-8-sig")
    station_df = pd.read_csv(station_path, encoding="utf-8-sig")
    return assign_df, station_df


def load_tables_from_manuscript(solution_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assign_path = solution_dir / "qvns_nsga2_compromise_assignments.csv"
    station_path = solution_dir / "qvns_nsga2_station_load.csv"
    if not assign_path.exists() or not station_path.exists():
        raise FileNotFoundError(f"Missing manuscript solution tables under {solution_dir}")
    assign_df = pd.read_csv(assign_path, encoding="utf-8-sig")
    station_df = pd.read_csv(station_path, encoding="utf-8-sig")
    return assign_df, station_df


def build_instance(args: argparse.Namespace, radius_km: float):
    inst = build_metro_logistics_instance(
        args.demand_scenario,
        top_n=args.top_n,
        model_setting=args.model_setting,
    )
    inst.D_k_max[:] = float(radius_km)
    return inst


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_path = args.input_dir / "service_radius_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run metrics not found: {metrics_path}")
    run_metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for radius_km in args.radii:
        inst = build_instance(args, radius_km)
        if float(radius_km) == 4.0 and args.use_manuscript_r4:
            assign_df, station_df = load_tables_from_manuscript(args.manuscript_r4_dir)
            chosen = None
            run_id = 0
            out_dir = args.output_dir / radius_folder_name(radius_km) / "manuscript_version"
        else:
            chosen = select_run(run_metrics, radius_km, args.selection)
            run_id = int(chosen["run_id"])
            assign_df, station_df = load_tables(args.input_dir, radius_km, run_id)
            out_dir = args.output_dir / radius_folder_name(radius_km) / f"run_{run_id:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_compromise_map(assign_df, station_df, inst, str(out_dir))
        if chosen is not None:
            summary_rows.append(
                {
                    "radius_km": float(radius_km),
                    "run_id": run_id,
                    "seed": int(chosen["seed"]),
                    "selection": args.selection,
                    "topsis_cost": float(chosen["topsis_cost"]),
                    "topsis_service_ratio": float(chosen["topsis_service_ratio"]),
                    "output_dir": str(out_dir),
                }
            )
            print(f"[radius={radius_km:.1f} km] run={run_id:02d} -> {out_dir}")
        else:
            summary_rows.append(
                {
                    "radius_km": float(radius_km),
                    "run_id": run_id,
                    "seed": None,
                    "selection": "manuscript_r4",
                    "topsis_cost": None,
                    "topsis_service_ratio": None,
                    "output_dir": str(out_dir),
                }
            )
            print(f"[radius={radius_km:.1f} km] manuscript version -> {out_dir}")

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "service_radius_map_selection.csv", index=False, encoding="utf-8-sig")
    print(f"\nMaps exported to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
