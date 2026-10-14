"""Select final layout schemes from Pareto solutions under TOPSIS weights.

The post-evaluation uses only the two model objectives:
Z1_total_cost is a cost criterion and Z2_metro_service_ratio is a benefit
criterion. To remain consistent with code.main.choose_compromise, Z2 is first
converted to a loss criterion (-Z2), and both criteria are min-max normalized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = [
    ("成本偏好", 0.70, 0.30, "更关注企业运营成本控制"),
    ("均衡偏好/默认方案", 0.50, 0.50, "无明确偏好时的基准折中口径"),
    ("服务偏好", 0.30, 0.70, "更关注地铁联运服务覆盖和道路减负"),
]


SUMMARY_COLUMNS = [
    "权重情景",
    "Z1综合成本权重",
    "Z2地铁服务比例权重",
    "适用解释",
    "solution_id",
    "TOPSIS贴近度",
    "Z1_total_cost",
    "Z2_metro_service_ratio",
    "num_open",
    "metro_volume",
    "direct_volume",
    "fixed_cost_total",
    "road_cost",
    "metro_cost",
    "feeder_cost",
    "transfer_cost",
    "inout_cost",
    "train_organization_cost",
    "total_carbon_tCO2",
    "avg_feeder_distance",
    "used_train_trips_total",
    "max_load_rate",
    "max_train_load_rate",
    "CV",
]


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalization with zero-span protection."""
    v_min = np.min(values, axis=0)
    v_max = np.max(values, axis=0)
    denom = v_max - v_min
    return np.divide(values - v_min, denom, out=np.zeros_like(values), where=denom > 1e-12)


def topsis_scores(
    pareto_df: pd.DataFrame,
    z1_weight: float,
    z2_weight: float,
) -> pd.Series:
    """Calculate TOPSIS closeness for cost-minimization and service-maximization."""
    criteria = np.column_stack(
        [
            pareto_df["Z1_total_cost"].to_numpy(dtype=float),
            -pareto_df["Z2_metro_service_ratio"].to_numpy(dtype=float),
        ]
    )
    norm = minmax_normalize(criteria)
    weights = np.array([z1_weight, z2_weight], dtype=float)
    weights = weights / weights.sum()

    ideal = np.zeros(2, dtype=float)
    nadir = np.ones(2, dtype=float)
    d_pos = np.sqrt(np.sum(weights * (norm - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum(weights * (norm - nadir) ** 2, axis=1))
    return pd.Series(d_neg / (d_pos + d_neg + 1e-12), index=pareto_df.index)


def select_scenarios(pareto_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Z1_total_cost", "Z2_metro_service_ratio", "solution_id"}
    missing = required.difference(pareto_df.columns)
    if missing:
        raise ValueError(f"Pareto file is missing required columns: {sorted(missing)}")

    feasible = pareto_df.copy()
    if "CV" in feasible.columns:
        feasible = feasible[pd.to_numeric(feasible["CV"], errors="coerce").fillna(np.inf) <= 1e-9]
    if feasible.empty:
        raise ValueError("No feasible Pareto solution is available after CV filtering.")

    feasible["_z1_key"] = feasible["Z1_total_cost"].round(6)
    feasible["_z2_key"] = feasible["Z2_metro_service_ratio"].round(12)
    unique = (
        feasible.sort_values(["Z1_total_cost", "Z2_metro_service_ratio"], ascending=[True, False])
        .drop_duplicates(["_z1_key", "_z2_key"], keep="first")
        .drop(columns=["_z1_key", "_z2_key"])
        .reset_index(drop=True)
    )

    selected_rows = []
    for scenario, z1_weight, z2_weight, explanation in DEFAULT_SCENARIOS:
        scores = topsis_scores(unique, z1_weight, z2_weight)
        selected = unique.loc[int(scores.idxmax())].copy()
        selected["权重情景"] = scenario
        selected["Z1综合成本权重"] = z1_weight
        selected["Z2地铁服务比例权重"] = z2_weight
        selected["适用解释"] = explanation
        selected["TOPSIS贴近度"] = float(scores.max())
        selected_rows.append(selected)

    result = pd.DataFrame(selected_rows)
    ordered = [c for c in SUMMARY_COLUMNS if c in result.columns]
    extras = [c for c in result.columns if c not in ordered]
    return result[ordered + extras]


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    display = df.copy()
    rename_map = {
        "solution_id": "解编号",
        "TOPSIS贴近度": "贴近度",
        "Z1_total_cost": "Z1综合成本(元/日)",
        "Z2_metro_service_ratio": "Z2地铁服务比例",
        "num_open": "启用转运点数",
        "metro_volume": "地铁联运量(SLU/日)",
        "direct_volume": "公路直送量(SLU/日)",
        "total_carbon_tCO2": "总碳排放(tCO2/日)",
        "avg_feeder_distance": "平均末端接驳距离(km)",
        "used_train_trips_total": "占用班次数",
        "max_load_rate": "最大站点负载率",
        "max_train_load_rate": "最大列车装载率",
        "CV": "约束违反度",
    }
    keep = list(rename_map.keys())
    front = ["权重情景", "Z1综合成本权重", "Z2地铁服务比例权重", "适用解释"]
    display = display[[c for c in front + keep if c in display.columns]].rename(columns=rename_map)

    formats = {
        "Z1综合成本权重": "{:.2f}",
        "Z2地铁服务比例权重": "{:.2f}",
        "贴近度": "{:.6f}",
        "Z1综合成本(元/日)": "{:.2f}",
        "Z2地铁服务比例": "{:.6f}",
        "启用转运点数": "{:.0f}",
        "地铁联运量(SLU/日)": "{:.2f}",
        "公路直送量(SLU/日)": "{:.2f}",
        "总碳排放(tCO2/日)": "{:.6f}",
        "平均末端接驳距离(km)": "{:.4f}",
        "占用班次数": "{:.0f}",
        "最大站点负载率": "{:.6f}",
        "最大列车装载率": "{:.6f}",
        "约束违反度": "{:.0f}",
    }
    for col, fmt in formats.items():
        if col in display.columns:
            display[col] = display[col].map(lambda x, f=fmt: f.format(float(x)))

    lines = [
        "# 权重情景 TOPSIS 最终解选择结果",
        "",
        "本表基于代表性运行得到的可行帕累托最优解，采用两目标 TOPSIS 后评价生成。评价指标仅包括 $Z_1$ 综合成本和 $Z_2$ 地铁服务比例；其中 $Z_1$ 为成本型指标，$Z_2$ 为效益型指标。无明确偏好时，采用 0.50/0.50 等权作为默认基准折中口径。",
        "",
        display.to_markdown(index=False, disable_numparse=True),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_input = root / "results" / "chapter5_q11_results" / "representative_solution" / "representative_pareto_points.csv"
    default_output = root / "results" / "5.20" / "weight_scenario_topsis_final_solutions.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input, help="Pareto solution CSV.")
    parser.add_argument("--output", type=Path, default=default_output, help="Selected scenario CSV.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=default_output.with_suffix(".md"),
        help="Markdown table output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pareto_df = pd.read_csv(args.input)
    result = select_scenarios(pareto_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_markdown(result, args.markdown_output)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown_output}")
    print(result[SUMMARY_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
