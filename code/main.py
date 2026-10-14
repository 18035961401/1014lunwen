# -*- coding: utf-8 -*-
"""NSGA-II and QVNS-NSGA-II solvers for the metro logistics location-allocation model."""

from dataclasses import dataclass
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
plt = None

try:
    from chinese_font_utils import configure_matplotlib_cjk
except ImportError:  # pragma: no cover - fallback for unusual direct imports
    configure_matplotlib_cjk = None

def _setup_matplotlib_chinese(_plt):
    """Configure Chinese-capable font fallbacks for generated figures.

    No font files are bundled in the package; this only selects fonts that are
    already installed on the user's system.
    """
    if configure_matplotlib_cjk is not None:
        configure_matplotlib_cjk(_plt)
        return
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Noto Sans CJK KR",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    try:
        from matplotlib import font_manager as _fm
        available = {font.name for font in _fm.fontManager.ttflist}
        ordered = [font for font in candidates if font in available]
        ordered += [font for font in candidates if font not in ordered]
    except Exception:
        ordered = candidates
    _plt.rcParams["font.family"] = "sans-serif"
    _plt.rcParams["font.sans-serif"] = ordered
    _plt.rcParams["axes.unicode_minus"] = False
    _plt.rcParams["svg.fonttype"] = "none"
    _plt.rcParams["pdf.fonttype"] = 42
    _plt.rcParams["ps.fonttype"] = 42

def _get_pyplot():
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        _setup_matplotlib_chinese(_plt)
        plt = _plt
    return plt

import metro_data_plotting_tools as base
from station_importance import extract_station_importance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SMALL_PAPER_TRANSFER_DATA_DIR = DATA_DIR / "小论文换乘站输入"


def _find_data_file(predicate, fallback_name="") -> Path:
    if DATA_DIR.exists():
        matches = [path for path in DATA_DIR.iterdir() if path.is_file() and predicate(path)]
        if matches:
            return matches[0]
    return DATA_DIR / fallback_name


def _find_geojson_file(keyword: str, fallback_name: str = "") -> Path:
    """Find a bundled GeoJSON by semantic filename keyword, not by file size."""
    if DATA_DIR.exists():
        matches = [
            path for path in DATA_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".geojson" and keyword in path.name
        ]
        if matches:
            matches.sort(key=lambda path: ("2024" not in path.name, "WGS84" not in path.name, len(path.name)))
            return matches[0]
    return DATA_DIR / fallback_name


def _name_key(name) -> str:
    return str(name).strip().replace(" ", "").replace("（", "(").replace("）", ")")


def _read_matrix_csv(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / "矩阵数据" / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_candidate_mode_csv(candidate_mode: str, filename: str, subdir: str = "") -> pd.DataFrame | None:
    mode = str(candidate_mode or "critic_topsis_top").strip().lower()
    if mode != "transfer_all":
        return None
    path = SMALL_PAPER_TRANSFER_DATA_DIR / subdir / filename if subdir else SMALL_PAPER_TRANSFER_DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_real_station_to_demand(candidate_df: pd.DataFrame, fallback: np.ndarray, candidate_mode="critic_topsis_top") -> np.ndarray:
    matrix = _read_candidate_mode_csv(candidate_mode, "地铁站点到顺丰站点_换乘站.csv")
    if matrix is None:
        matrix = _read_matrix_csv("地铁站点到顺丰站点.csv")
    if matrix is None:
        return fallback
    by_name = {_name_key(row["origin_name"]): row for _, row in matrix.iterrows()}
    values = np.zeros_like(fallback, dtype=float)
    demand_cols = [str(i) for i in range(1, fallback.shape[1] + 1)]
    for k, row in candidate_df.iterrows():
        source = by_name.get(_name_key(row["地铁站名称"]))
        if source is None:
            values[k] = fallback[k]
        else:
            values[k] = source[demand_cols].astype(float).to_numpy()
    return values


def _load_real_logistics_to_demand(fallback: np.ndarray) -> np.ndarray:
    matrix = _read_matrix_csv("顺丰园区到顺丰站点.csv")
    if matrix is None:
        return fallback
    values = np.zeros_like(fallback, dtype=float)
    demand_cols = [str(i) for i in range(1, fallback.shape[1] + 1)]
    for i in range(min(fallback.shape[0], len(matrix))):
        values[i] = matrix.iloc[i][demand_cols].astype(float).to_numpy()
    return values


def _load_real_logistics_to_access(fallback: np.ndarray) -> np.ndarray:
    matrix = _read_matrix_csv("顺丰园区到接驳地铁站.csv")
    if matrix is None or "distance_km" not in matrix.columns:
        return fallback
    values = fallback.copy()
    for i in range(min(len(values), len(matrix))):
        values[i] = float(matrix.iloc[i]["distance_km"])
    return values


def _load_real_metro_matrices(candidate_df: pd.DataFrame, fallback_distance: np.ndarray, fallback_transfer: np.ndarray, candidate_mode="critic_topsis_top"):
    summary = _read_candidate_mode_csv(candidate_mode, "S1_S2_metro_distance_segmented_speed_summary_换乘站.csv", subdir="矩阵数据")
    if summary is None:
        summary = _read_matrix_csv("S1_S2_metro_distance_segmented_speed_summary.csv")
    distance = fallback_distance.copy().astype(float)
    ordinary = np.zeros_like(distance)
    express = np.zeros_like(distance)
    metro_time = np.divide(distance, 35.0, out=np.zeros_like(distance), where=distance > 0)
    transfer = fallback_transfer.copy().astype(float)
    if summary is None:
        return distance, ordinary, express, metro_time, transfer

    source_keys = ["S1", "S2"]
    for i, source_key in enumerate(source_keys[:distance.shape[0]]):
        source_rows = summary[summary["source_file"].astype(str).str.contains(source_key, na=False)]
        by_station = {_name_key(row["目标站点"]): row for _, row in source_rows.iterrows()}
        for k, row in candidate_df.iterrows():
            source = by_station.get(_name_key(row["地铁站名称"]))
            if source is None:
                continue
            distance[i, k] = float(source["地铁线网距离/km"])
            ordinary[i, k] = float(source["ordinary_line_distance_km"])
            express[i, k] = float(source["express_line_distance_km"])
            metro_time[i, k] = float(source["segmented_metro_time_h"])
            transfer[i, k] = float(source["换乘次数"])
    return distance, ordinary, express, metro_time, transfer


def _load_station_importance(candidate_df: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    path = DATA_DIR / "第三章评价结果" / "critic_topsis_top17.csv"
    if not path.exists():
        return fallback / (np.max(fallback) + 1e-12)
    ranking = pd.read_csv(path, encoding="utf-8-sig")
    if "alternative" not in ranking.columns or "ict_score" not in ranking.columns:
        return fallback / (np.max(fallback) + 1e-12)
    score_by_name = {_name_key(row["alternative"]): float(row["ict_score"]) for _, row in ranking.iterrows()}
    values = np.array([score_by_name.get(_name_key(row["地铁站名称"]), np.nan) for _, row in candidate_df.iterrows()], dtype=float)
    if np.any(~np.isfinite(values)):
        fallback_scaled = fallback / (np.max(fallback) + 1e-12)
        values = np.where(np.isfinite(values), values, fallback_scaled)
    return values / (np.max(values) + 1e-12)


def _read_supplementary_csv(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / "补充约束数据" / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_global_parameters() -> Dict[str, float | str]:
    """Read confirmed no-budget global parameters from the supplementary CSV."""
    table = _read_supplementary_csv("模型全局参数_无预算.csv")
    values: Dict[str, float | str] = {}
    if table is None or not {"代码变量", "参数值"}.issubset(table.columns):
        return values
    for _, row in table.iterrows():
        key = str(row.get("代码变量", "")).strip()
        if not key or key.lower() == "nan":
            continue
        raw = row.get("参数值", "")
        try:
            val = float(raw)
            if val.is_integer():
                val = int(val)
        except Exception:
            val = str(raw)
        values[key] = val
    return values


def _load_candidate_final_attributes(candidate_df: pd.DataFrame, candidate_mode="critic_topsis_top") -> pd.DataFrame | None:
    table = _read_candidate_mode_csv(candidate_mode, "候选转运点最终属性表_换乘站.csv")
    if table is None:
        table = _read_supplementary_csv("候选转运点最终属性表.csv")
    if table is None or "station_name" not in table.columns:
        return None
    return table.copy()


def _load_source_info(default_share: np.ndarray, default_access: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = _read_supplementary_csv("物流园区货源比例与接入信息.csv")
    shares = np.asarray(default_share, dtype=float).copy()
    access = np.asarray(default_access, dtype=float).copy()
    if table is None:
        return shares, access
    if "source_share" in table.columns:
        vals = pd.to_numeric(table["source_share"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) >= len(shares) and np.sum(vals[:len(shares)]) > 0:
            shares = vals[:len(shares)]
    if "access_distance_km" in table.columns:
        vals = pd.to_numeric(table["access_distance_km"], errors="coerce").to_numpy(dtype=float)
        if len(vals) >= len(access):
            access = np.where(np.isfinite(vals[:len(access)]), vals[:len(access)], access)
    return shares, access


def _load_train_parameter_table() -> pd.DataFrame | None:
    table = _read_supplementary_csv("平峰期列车班次与碳成本参数.csv")
    if table is not None and {"line_name", "single_train_capacity_pieces"}.issubset(table.columns):
        return table
    return None


def _load_line_capacity() -> Tuple[List[str], Dict[str, float], float]:
    """Load single-train capacity for each line.

    The confirmed model treats ``整列车可装箱数_件`` as the capacity of one
    usable off-peak train, not as an all-day line capacity.
    """
    train_table = _load_train_parameter_table()
    if train_table is not None:
        capacity = {
            str(row["line_name"]).strip(): float(row["single_train_capacity_pieces"])
            for _, row in train_table.iterrows()
            if pd.notna(row["line_name"]) and pd.notna(row["single_train_capacity_pieces"])
        }
        line_names = list(capacity.keys())
        line_capacity_limit = min(capacity.values()) if capacity else 0.0
        return line_names, capacity, line_capacity_limit

    table = _read_supplementary_csv("平峰期地铁货运体积与装箱能力参数.csv")
    if table is None or not {"线路", "整列车可装箱数_件"}.issubset(table.columns):
        return [], {}, 0.0
    capacity = {
        str(row["线路"]).strip(): float(row["整列车可装箱数_件"])
        for _, row in table.iterrows()
        if pd.notna(row["线路"]) and pd.notna(row["整列车可装箱数_件"])
    }
    line_names = list(capacity.keys())
    line_capacity_limit = min(capacity.values()) if capacity else 0.0
    return line_names, capacity, line_capacity_limit


def _load_train_organization_cost(line_capacity: Dict[str, float]) -> tuple[int, float, Dict[str, float], float, float, float, float]:
    """Return train trips, organization cost, carbon price and emission factors."""
    table = _load_train_parameter_table()
    available_trips = 5
    rate = 0.06
    costs = {line: cap * rate for line, cap in line_capacity.items()}
    carbon_price = 0.10
    road_ef = 3.0e-4
    metro_ef = 3.0e-5
    feeder_ef = 1.0e-4
    if table is None or table.empty:
        return available_trips, rate, costs, carbon_price, road_ef, metro_ef, feeder_ef
    row0 = table.iloc[0]
    if "available_train_trips" in table.columns and pd.notna(row0.get("available_train_trips")):
        available_trips = int(float(row0["available_train_trips"]))
    if "train_organization_cost_rate_yuan_per_piece" in table.columns and pd.notna(row0.get("train_organization_cost_rate_yuan_per_piece")):
        rate = float(row0["train_organization_cost_rate_yuan_per_piece"])
    if "carbon_price_yuan_per_kg" in table.columns and pd.notna(row0.get("carbon_price_yuan_per_kg")):
        carbon_price = float(row0["carbon_price_yuan_per_kg"])
    if "road_emission_factor_kg_per_piece_km" in table.columns and pd.notna(row0.get("road_emission_factor_kg_per_piece_km")):
        road_ef = float(row0["road_emission_factor_kg_per_piece_km"])
    if "metro_emission_factor_kg_per_piece_km" in table.columns and pd.notna(row0.get("metro_emission_factor_kg_per_piece_km")):
        metro_ef = float(row0["metro_emission_factor_kg_per_piece_km"])
    if "feeder_emission_factor_kg_per_piece_km" in table.columns and pd.notna(row0.get("feeder_emission_factor_kg_per_piece_km")):
        feeder_ef = float(row0["feeder_emission_factor_kg_per_piece_km"])
    if "train_organization_cost_per_trip_yuan" in table.columns:
        costs = {}
        for _, row in table.iterrows():
            line = str(row["line_name"]).strip()
            val = row.get("train_organization_cost_per_trip_yuan")
            if pd.notna(val):
                costs[line] = float(val)
            else:
                costs[line] = float(line_capacity.get(line, 0.0)) * rate
    else:
        costs = {line: cap * rate for line, cap in line_capacity.items()}
    return available_trips, rate, costs, carbon_price, road_ef, metro_ef, feeder_ef



def _load_path_line_incidence(candidate_df: pd.DataFrame, line_names: List[str], I: int, K: int, candidate_mode="critic_topsis_top") -> np.ndarray:
    incidence = np.zeros((I, K, len(line_names)), dtype=float)
    matrix = _read_candidate_mode_csv(candidate_mode, "路径线路关联矩阵_a_ik_line_换乘站.csv")
    if matrix is None:
        matrix = _read_supplementary_csv("路径线路关联矩阵_a_ik_line.csv")
    if matrix is None or not line_names:
        return incidence
    required = {"source_id", "target_station", "line_name", "a_ik_line"}
    if not required.issubset(matrix.columns):
        return incidence

    source_index = {f"S{i + 1}": i for i in range(I)}
    station_index = {_name_key(row["地铁站名称"]): k for k, row in candidate_df.iterrows()}
    line_index = {line: idx for idx, line in enumerate(line_names)}

    for _, row in matrix.iterrows():
        i = source_index.get(str(row["source_id"]).strip())
        k = station_index.get(_name_key(row["target_station"]))
        line_pos = line_index.get(str(row["line_name"]).strip())
        if i is None or k is None or line_pos is None:
            continue
        incidence[i, k, line_pos] = float(row["a_ik_line"])
    return incidence


def _chromosome_length(inst: "MetroLogisticsInstance") -> int:
    # Encoding: first K genes are y_k. Remaining I*J genes encode route choice:
    # 0 -> z_ij = 1 (D, direct road delivery); 1..K -> x_ijk = 1 (M, via transfer point k).
    return inst.K + inst.I * inst.J


def _assignment_start(inst: "MetroLogisticsInstance") -> int:
    return inst.K


def _assignment_end(inst: "MetroLogisticsInstance") -> int:
    return inst.K + inst.I * inst.J


def _flat_assignment_index(inst: "MetroLogisticsInstance", i: int, j: int) -> int:
    return inst.K + i * inst.J + j


def _coerce_chromosome(dec: np.ndarray, inst: "MetroLogisticsInstance") -> np.ndarray:
    dec = np.asarray(dec, dtype=float).copy()
    expected = _chromosome_length(inst)
    if len(dec) == expected:
        return dec
    resized = np.zeros(expected, dtype=float)
    resized[:min(expected, len(dec))] = dec[:min(expected, len(dec))]
    return resized


if DATA_DIR.exists():
    base.BASE_DIR = str(DATA_DIR)
    base.METRO_STATION_FILE = str(_find_data_file(lambda path: path.suffix.lower() == ".csv" and "2024" in path.name and "10" in path.name))
    base.SF_SITE_FILE = str(_find_data_file(lambda path: path.suffix.lower() == ".csv" and "20251218" in path.name))
    base.TOPSIS_FILE = str((DATA_DIR / "第三章评价结果" / "critic_topsis_top17.csv") if (DATA_DIR / "第三章评价结果" / "critic_topsis_top17.csv").exists() else _find_data_file(lambda path: path.is_file() and path.suffix.lower() == ".csv" and path.name.lower().startswith("critic_topsis")))
    base.BOUNDARY_REMOTE_CACHE_FILE = str(DATA_DIR / "510107.json")
    base.METRO_POINT_GEOJSON_FILE = str(_find_geojson_file("地铁站点", "武侯区地铁站点_2024年10月_WGS84.geojson"))
    base.METRO_POINT_GEOJSON_FILE_ALT = base.METRO_POINT_GEOJSON_FILE
    base.METRO_LINE_GEOJSON_FILE = str(_find_geojson_file("地铁线路", "武侯区地铁线路折线_2024年10月_WGS84.geojson"))
    base.OUTPUT_DIR = str(RESULTS_DIR)
    os.makedirs(base.OUTPUT_DIR, exist_ok=True)

NSGA2_OUTPUT_DIR = os.path.join(base.OUTPUT_DIR, "nsga2_no_budget_baseline")

MODEL_SETTINGS: Dict[str, Tuple[float, float | None]] = {
    # 正式模型为无预算约束：固定成本进入 Z1，但不设置硬性预算区间；
    # 上界仅作为结果报告中的“17个候选站点固定成本总和”参考值。
    "no_budget_range": (0.0, None),
}
MODEL_SETTING_DELTA = 0.0
STATION_TYPE_CAPACITY = {"SL": 5000.0, "TR": 7000.0, "HB": 10000.0}
STATION_TYPE_FIXED_COST = {"SL": 8000.0, "TR": 12000.0, "HB": 16000.0}
NSGA2_OUTPUT_FILES = (
    "nsga2_pareto.csv",
    "nsga2_history.csv",
    "nsga2_compromise_assignments.csv",
    "nsga2_station_load.csv",
    "nsga2_line_load.csv",
)
QVNS_NSGA2_OUTPUT_FILES = (
    "qvns_nsga2_pareto.csv",
    "qvns_nsga2_final_history.csv",
    "qvns_nsga2_training_history.csv",
    "qvns_nsga2_q_table.csv",
    "qvns_nsga2_action_space.csv",
    "qvns_nsga2_vns_history.csv",
    "qvns_nsga2_best_params.csv",
    "qvns_nsga2_compromise_assignments.csv",
    "qvns_nsga2_station_load.csv",
    "qvns_nsga2_line_load.csv",
)


def _normalize_model_setting(scenario: str) -> str:
    scenario = str(scenario).strip().lower().replace("-", "_")
    aliases = {
        "none": "no_budget_range",
        "no_budget": "no_budget_range",
        "no_budget_range": "no_budget_range",
        "unconstrained": "no_budget_range",
        "no_budget_constraint": "no_budget_range",
        "base": "no_budget_range",
    }
    # 预算区间情景已彻底删除；历史别名不再接受，避免论文口径混淆。
    scenario = aliases.get(scenario, scenario)
    if scenario not in MODEL_SETTINGS:
        raise ValueError(f"Unknown model setting {scenario!r}; choose from {sorted(MODEL_SETTINGS)}")
    return scenario


def _station_type_from_line_count(line_count: np.ndarray) -> np.ndarray:
    return np.where(line_count >= 3, "HB", np.where(line_count == 2, "TR", "SL"))


def _station_values_by_type(labels: np.ndarray, values: Dict[str, float]) -> np.ndarray:
    return np.array([values[str(label)] for label in labels], dtype=float)


@dataclass
class MetroLogisticsInstance:
    scenario: str
    K: int
    J: int
    I: int
    candidate_df: pd.DataFrame
    demand_df: pd.DataFrame
    logistics_df: pd.DataFrame
    access_df: pd.DataFrame
    Q_j: np.ndarray
    station_type_labels: np.ndarray
    station_capacity: np.ndarray
    fixed_cost: np.ndarray
    station_importance: np.ndarray
    d_kj_road: np.ndarray
    d_ij_road: np.ndarray
    d_is_road: np.ndarray
    d_metro: np.ndarray
    d_metro_ordinary: np.ndarray
    d_metro_express: np.ndarray
    metro_time: np.ndarray
    transfer_count: np.ndarray
    d_kk_road: np.ndarray
    D_k_max: np.ndarray
    source_share: np.ndarray
    model_setting: str = "no_budget_range"
    fixed_cost_lower: float = 0.0
    fixed_cost_upper: float = 1.0e12
    fixed_cost_delta: float = MODEL_SETTING_DELTA
    min_open: int = 0
    max_open: int = 17
    min_load_rate: float = 0.15
    max_load_rate: float = 0.85
    max_avg_feeder_distance: float = np.inf
    min_station_distance: float = 2.0
    max_transfer_count: int = 2
    max_open_per_line: int = 3
    direct_road_speed: float = 30.0
    access_road_speed: float = 25.0
    feeder_speed: float = 20.0
    ordinary_metro_speed: float = 35.0
    express_metro_speed: float = 80.0
    transfer_time_hour: float = 10.0 / 60.0
    processing_time_hour: float = 15.0 / 60.0
    road_operating_cost: float = 0.12
    road_emission_factor: float = 3.0e-4
    road_carbon_cost: float = 0.000030
    metro_electric_cost: float = 0.02
    metro_emission_factor: float = 3.0e-5
    metro_carbon_cost: float = 0.000003
    carbon_price: float = 0.10
    feeder_emission_factor: float = 1.0e-4
    feeder_carbon_cost: float = 1.0e-5
    available_train_trips: int = 5
    train_organization_cost_rate: float = 0.06
    train_organization_cost: Dict[str, float] | None = None
    feeder_cost: float = base.C2_FEEDER
    transfer_cost: float = base.C3_TRANSFER
    inout_cost: float = 0.20
    line_names: List[str] | None = None
    line_capacity: Dict[str, float] | None = None
    line_capacity_limit: float = 0.0
    path_line_incidence: np.ndarray | None = None


def build_metro_logistics_instance(scenario="base", top_n=17, road_factor=1.2, model_setting="no_budget_range", candidate_mode="critic_topsis_top") -> MetroLogisticsInstance:
    model_setting = _normalize_model_setting(model_setting)
    fixed_cost_lower, fixed_cost_upper = MODEL_SETTINGS[model_setting]
    source_inst = base.build_instance(scenario, top_n=top_n, candidate_mode=candidate_mode)
    candidate_df = source_inst.candidate_df.copy()
    demand_df = source_inst.demand_df.copy()
    K, J, I = source_inst.K, source_inst.J, source_inst.I
    global_params = _load_global_parameters()

    d_kj_geo = np.zeros((K, J), dtype=float)
    for k, row in candidate_df.iterrows():
        d_kj_geo[k] = base.haversine_km(
            row["经度"], row["纬度"],
            demand_df["经度_GCJ02"].values,
            demand_df["纬度_GCJ02"].values,
        )
    # 确保d_kj_geo是float类型
    d_kj_geo = d_kj_geo.astype(float)
    d_kj_road = _load_real_station_to_demand(candidate_df, float(road_factor) * d_kj_geo, candidate_mode=candidate_mode)

    d_ij_geo = np.zeros((I, J), dtype=float)
    for i, row in base.LOGISTICS.iterrows():
        d_ij_geo[i] = base.haversine_km(
            row["lon"], row["lat"],
            demand_df["经度_GCJ02"].values,
            demand_df["纬度_GCJ02"].values,
        )
    d_ij_road = _load_real_logistics_to_demand(road_factor * d_ij_geo)

    d_is_road = np.zeros(I, dtype=float)
    for i, row in base.LOGISTICS.iterrows():
        access = base.ACCESS_STATIONS.iloc[i]
        d_is_road[i] = road_factor * float(base.haversine_km(row["lon"], row["lat"], access["lon"], access["lat"]))
    d_is_road = _load_real_logistics_to_access(d_is_road)
    source_share, d_is_road = _load_source_info(np.array([0.8, 0.2], dtype=float), d_is_road)

    d_kk_geo = np.zeros((K, K), dtype=float)
    for k, row in candidate_df.iterrows():
        d_kk_geo[k] = base.haversine_km(
            row["经度"], row["纬度"],
            candidate_df["经度"].values,
            candidate_df["纬度"].values,
        )
    d_kk_road = d_kk_geo

    line_count = candidate_df["线路数"].astype(float).to_numpy()
    station_type_labels = _station_type_from_line_count(line_count)
    station_capacity = _station_values_by_type(station_type_labels, STATION_TYPE_CAPACITY)
    fixed_cost = _station_values_by_type(station_type_labels, STATION_TYPE_FIXED_COST)
    final_attrs = _load_candidate_final_attributes(candidate_df, candidate_mode=candidate_mode)
    if final_attrs is not None:
        by_name = {_name_key(row["station_name"]): row for _, row in final_attrs.iterrows()}
        for k, row in candidate_df.iterrows():
            attr = by_name.get(_name_key(row["地铁站名称"]))
            if attr is None:
                continue
            if "station_type" in attr and pd.notna(attr["station_type"]):
                station_type_labels[k] = str(attr["station_type"]).strip()
            if "station_capacity_pieces" in attr and pd.notna(attr["station_capacity_pieces"]):
                station_capacity[k] = float(attr["station_capacity_pieces"])
            if "fixed_cost_yuan" in attr and pd.notna(attr["fixed_cost_yuan"]):
                fixed_cost[k] = float(attr["fixed_cost_yuan"])
    D_k_max = np.full(K, float(global_params.get("D_k_max", 4.0)), dtype=float)

    temp_inst = source_inst
    temp_inst.candidate_df = candidate_df
    importance = _load_station_importance(candidate_df, extract_station_importance(temp_inst))
    d_metro, d_metro_ordinary, d_metro_express, metro_time, transfer_count = _load_real_metro_matrices(
        candidate_df,
        source_inst.d_metro.copy(),
        source_inst.transfer_count.copy(),
        candidate_mode=candidate_mode,
    )
    line_names, line_capacity, line_capacity_limit = _load_line_capacity()
    available_train_trips, train_org_rate, train_org_cost, carbon_price, road_ef, metro_ef, feeder_ef = _load_train_organization_cost(line_capacity)
    path_line_incidence = _load_path_line_incidence(candidate_df, line_names, I, K, candidate_mode=candidate_mode)

    return MetroLogisticsInstance(
        scenario=scenario,
        K=K,
        J=J,
        I=I,
        candidate_df=candidate_df,
        demand_df=demand_df,
        logistics_df=base.LOGISTICS.copy(),
        access_df=base.ACCESS_STATIONS.copy(),
        Q_j=source_inst.Q_j.copy(),
        station_type_labels=station_type_labels,
        station_capacity=station_capacity,
        fixed_cost=fixed_cost,
        station_importance=importance,
        d_kj_road=d_kj_road,
        d_ij_road=d_ij_road,
        d_is_road=d_is_road,
        d_metro=d_metro,
        d_metro_ordinary=d_metro_ordinary,
        d_metro_express=d_metro_express,
        metro_time=metro_time,
        transfer_count=transfer_count,
        d_kk_road=d_kk_road,
        D_k_max=D_k_max,
        source_share=source_share,
        model_setting=model_setting,
        fixed_cost_lower=fixed_cost_lower,
        fixed_cost_upper=float(np.sum(fixed_cost)),
        fixed_cost_delta=0.0,
        min_open=0,
        max_open=K,
        min_load_rate=float(global_params.get("min_load_rate", 0.15)),
        max_load_rate=float(global_params.get("max_load_rate", 0.85)),
        min_station_distance=float(global_params.get("min_station_distance", 2.0)),
        max_transfer_count=int(global_params.get("max_transfer_count", 2)),
        max_open_per_line=int(global_params.get("max_open_per_line", 3)),
        road_emission_factor=road_ef,
        road_carbon_cost=road_ef * carbon_price,
        metro_emission_factor=metro_ef,
        metro_carbon_cost=metro_ef * carbon_price,
        carbon_price=carbon_price,
        feeder_emission_factor=feeder_ef,
        feeder_carbon_cost=feeder_ef * carbon_price,
        available_train_trips=int(global_params.get("available_train_trips", available_train_trips)),
        train_organization_cost_rate=float(global_params.get("train_organization_cost_rate", train_org_rate)),
        train_organization_cost=train_org_cost,
        line_names=line_names,
        line_capacity=line_capacity,
        line_capacity_limit=line_capacity_limit,
        path_line_incidence=path_line_incidence,
    )


def apply_strict_no_repair_settings(inst: MetroLogisticsInstance, p_open: int = 5) -> MetroLogisticsInstance:
    """Backward-compatible wrapper for old fixed-P scripts.

    The formal model now removes both fixed opened-station counts and fixed-cost
    fixed-cost reference bounds. This helper preserves legacy callers but no longer imposes
    ``sum y_k = P`` or any fixed-cost interval.
    """
    return apply_no_budget_settings(inst)


def apply_model_setting(inst: MetroLogisticsInstance, model_setting: str = "no_budget_range") -> MetroLogisticsInstance:
    """Apply the formal no-budget parameterization to an instance."""
    model_setting = _normalize_model_setting(model_setting)
    fixed_cost_lower, fixed_cost_upper = MODEL_SETTINGS[model_setting]
    line_count = inst.candidate_df["线路数"].astype(float).to_numpy()
    inst.station_type_labels = _station_type_from_line_count(line_count)
    inst.station_capacity = _station_values_by_type(inst.station_type_labels, STATION_TYPE_CAPACITY)
    inst.fixed_cost = _station_values_by_type(inst.station_type_labels, STATION_TYPE_FIXED_COST)
    inst.model_setting = model_setting
    inst.fixed_cost_lower = fixed_cost_lower
    inst.fixed_cost_upper = float(np.sum(inst.fixed_cost))
    inst.fixed_cost_delta = 0.0
    inst.min_open = 0
    inst.max_open = inst.K
    global_params = _load_global_parameters()
    inst.min_load_rate = float(global_params.get("min_load_rate", 0.15))
    inst.max_load_rate = float(global_params.get("max_load_rate", 0.85))
    inst.max_avg_feeder_distance = np.inf
    inst.min_station_distance = float(global_params.get("min_station_distance", 2.0))
    inst.max_open_per_line = int(global_params.get("max_open_per_line", 3))
    inst.source_share, inst.d_is_road = _load_source_info(inst.source_share, inst.d_is_road)
    return inst


def apply_no_budget_settings(inst: MetroLogisticsInstance) -> MetroLogisticsInstance:
    """Disable fixed-cost interval constraints for unconstrained benchmark runs."""
    inst.model_setting = "no_budget_range"
    inst.fixed_cost_lower = 0.0
    inst.fixed_cost_upper = float(np.sum(inst.fixed_cost))
    inst.fixed_cost_delta = 0.0
    inst.min_open = 0
    inst.max_open = inst.K
    return inst


def apply_variable_open_settings(inst: MetroLogisticsInstance, min_open: int = 0, max_open: int | None = None,
                                 fixed_cost_per_station: float = 10000.0) -> MetroLogisticsInstance:
    """取消固定开设数量情景，让算法自行搜索开设站点数量。"""
    apply_model_setting(inst, inst.model_setting)
    inst.min_open = int(np.clip(min_open, 0, inst.K))
    inst.max_open = int(np.clip(inst.K if max_open is None else max_open, inst.min_open, inst.K))
    return inst


def decode_solution(dec: np.ndarray, inst: MetroLogisticsInstance):
    """Decode chromosome into station-opening y_k and route-choice genes.

    The route-choice matrix represents thesis variables z_ij and x_ijk:
    value 0 means z_ij=1; value k+1 means x_ijk=1.
    """
    dec = _coerce_chromosome(dec, inst)
    x = np.rint(dec[:inst.K]).astype(int)
    x = np.clip(x, 0, 1)
    assignment = np.rint(dec[_assignment_start(inst):_assignment_end(inst)]).astype(int)
    assignment = np.clip(assignment, 0, inst.K)
    assignment = assignment.reshape((inst.I, inst.J))
    return x, assignment


def repair_solution(dec: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    dec = _coerce_chromosome(dec, inst)
    x, assignment = decode_solution(dec, inst)

    forbidden = _forbidden_station_mask(inst)
    x[forbidden] = 0
    x = _repair_open_count(x, inst)
    x = _repair_station_spacing(x, inst)
    x = _repair_model_setting(x, inst)
    x = _repair_station_spacing(x, inst)
    x = _repair_open_count(x, inst)

    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    for i in range(inst.I):
        for j in range(inst.J):
            if assignment[i, j] == 0:
                continue
            k = assignment[i, j] - 1
            valid = 0 <= k < inst.K and x[k] == 1 and reachable[k, j] and not forbidden[k] and inst.transfer_count[i, k] <= inst.max_transfer_count
            if not valid:
                feasible = np.where((x == 1) & reachable[:, j] & (~forbidden) & (inst.transfer_count[i] <= inst.max_transfer_count))[0]
                if len(feasible) == 0:
                    assignment[i, j] = 0
                else:
                    score = _assignment_score(feasible, j, inst, source_idx=i)
                    assignment[i, j] = int(feasible[np.argmax(score)]) + 1

    dec[:inst.K] = x
    dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
    return dec


def normalize_solution(dec: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    """仅做编码合法化，不修复约束违背。"""
    dec = _coerce_chromosome(dec, inst)
    dec[:inst.K] = np.rint(np.clip(dec[:inst.K], 0, 1))
    dec[_assignment_start(inst):_assignment_end(inst)] = np.rint(np.clip(dec[_assignment_start(inst):_assignment_end(inst)], 0, inst.K))
    return dec


def _forbidden_station_mask(inst: MetroLogisticsInstance) -> np.ndarray:
    return np.all(inst.transfer_count > inst.max_transfer_count, axis=0)


def _fixed_cost_feasible_upper(inst: MetroLogisticsInstance) -> float:
    return float(inst.fixed_cost_upper - inst.fixed_cost_delta)


def _fixed_cost_total(x: np.ndarray, inst: MetroLogisticsInstance) -> float:
    return float(np.sum(inst.fixed_cost * np.rint(np.clip(x, 0, 1))))


def _repair_open_count(x: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    x = x.copy()
    forbidden = _forbidden_station_mask(inst)
    allowed = np.where(~forbidden)[0]
    if len(allowed) == 0:
        return x
    while x.sum() < min(inst.min_open, len(allowed)):
        closed = np.array([k for k in allowed if x[k] == 0])
        if len(closed) == 0:
            break
        open_sites = np.where(x == 1)[0]
        if len(open_sites) > 0:
            compatible = np.array([
                k for k in closed
                if np.all(inst.d_kk_road[k, open_sites] >= inst.min_station_distance)
            ], dtype=int)
            if len(compatible) > 0:
                closed = compatible
        x[closed[np.argmax(inst.station_importance[closed])]] = 1
    while x.sum() > inst.max_open:
        open_sites = np.where(x == 1)[0]
        if len(open_sites) == 0:
            break
        x[open_sites[np.argmin(inst.station_importance[open_sites])]] = 0
    return x


def _repair_model_setting(x: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    """No-op under the formal no-model setting.

    Fixed operation cost remains part of Z1 but is no longer constrained by
    artificial fixed-cost reference bounds.
    """
    x = x.copy()
    if inst.model_setting == "no_budget_range":
        return x
    forbidden = _forbidden_station_mask(inst)
    lower = float(inst.fixed_cost_lower)
    upper = _fixed_cost_feasible_upper(inst)

    while _fixed_cost_total(x, inst) > upper + 1e-9:
        open_sites = np.where(x == 1)[0]
        if len(open_sites) == 0:
            break
        benefit = inst.station_importance[open_sites] / (inst.fixed_cost[open_sites] + 1e-12)
        remove = open_sites[int(np.argmin(benefit))]
        x[remove] = 0

    changed = True
    while _fixed_cost_total(x, inst) < lower - 1e-9 and changed:
        changed = False
        current_cost = _fixed_cost_total(x, inst)
        closed = np.where((x == 0) & (~forbidden))[0]
        if len(closed) == 0:
            break
        open_sites = np.where(x == 1)[0]
        if len(open_sites) > 0:
            closed = np.array([
                k for k in closed
                if np.all(inst.d_kk_road[k, open_sites] >= inst.min_station_distance)
            ], dtype=int)
        if len(closed) == 0:
            break
        affordable = closed[current_cost + inst.fixed_cost[closed] <= upper + 1e-9]
        if len(affordable) == 0:
            break
        remaining = lower - current_cost
        enough = affordable[inst.fixed_cost[affordable] >= remaining - 1e-9]
        pool = enough if len(enough) > 0 else affordable
        score = inst.station_importance[pool] / (inst.fixed_cost[pool] + 1e-12)
        add = pool[int(np.argmax(score))]
        x[add] = 1
        changed = True
    return x


def _repair_station_spacing(x: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    x = x.copy()
    changed = True
    while changed:
        changed = False
        open_sites = np.where(x == 1)[0]
        for i, k in enumerate(open_sites):
            for kk in open_sites[i + 1:]:
                if inst.d_kk_road[k, kk] < inst.min_station_distance:
                    remove = k if inst.station_importance[k] < inst.station_importance[kk] else kk
                    x[remove] = 0
                    changed = True
                    break
            if changed:
                break
    return _repair_line_concentration(x, inst)


def _repair_line_concentration(x: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    x = x.copy()
    while True:
        open_sites = np.where(x == 1)[0]
        violation = None
        for line, count in open_station_counts_by_line(x, inst).items():
            if count > inst.max_open_per_line:
                violation = line
                break
        if violation is None:
            return x
        candidates = [k for k in open_sites if violation in _station_lines(inst, int(k))]
        if not candidates:
            return x
        remove = min(candidates, key=lambda k: inst.station_importance[int(k)])
        x[int(remove)] = 0


def open_station_counts_by_line(x: np.ndarray, inst: MetroLogisticsInstance) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for k, opened in enumerate(np.rint(np.asarray(x)).astype(int)):
        if opened != 1:
            continue
        for line in _station_lines(inst, k):
            counts[line] = counts.get(line, 0) + 1
    return counts


def _assignment_score(feasible: np.ndarray, j: int, inst: MetroLogisticsInstance, source_idx: int | None = None) -> np.ndarray:
    distance_score = 1.0 / (inst.d_kj_road[feasible, j] + 1e-9)
    distance_score = distance_score / (distance_score.sum() + 1e-12)
    importance = inst.station_importance[feasible]
    importance = importance / (importance.sum() + 1e-12)
    score = 0.6 * distance_score + 0.4 * importance
    if source_idx is not None:
        metro = 1.0 / (inst.d_metro[source_idx, feasible] + 1e-9)
        metro = metro / (metro.sum() + 1e-12)
        score = 0.5 * distance_score + 0.25 * metro + 0.25 * importance
    return score


def initialize_population(inst: MetroLogisticsInstance, pop_size: int, use_repair=True) -> np.ndarray:
    pop = np.zeros((pop_size, _chromosome_length(inst)), dtype=float)
    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    coverage = reachable.sum(axis=1).astype(float)
    coverage = coverage / (coverage.sum() + 1e-12)
    prob = 0.5 * inst.station_importance + 0.5 * coverage
    prob = prob / (prob.sum() + 1e-12)
    forbidden = _forbidden_station_mask(inst)
    prob[forbidden] = 0
    prob = prob / (prob.sum() + 1e-12)

    for n in range(pop_size):
        x = np.zeros(inst.K, dtype=float)
        min_station_cost = float(np.min(inst.fixed_cost[~forbidden])) if np.any(~forbidden) else 1.0
        fixed_cost_cap_open = max(1, int(np.floor(_fixed_cost_feasible_upper(inst) / max(min_station_cost, 1.0))))
        init_max_open = min(inst.max_open, fixed_cost_cap_open, int(np.sum(~forbidden)))
        init_min_open = min(inst.min_open, init_max_open)
        open_count = np.random.randint(init_min_open, init_max_open + 1)
        open_count = min(open_count, int(np.sum(~forbidden)))
        if open_count > 0:
            chosen = np.random.choice(inst.K, size=open_count, replace=False, p=prob)
            x[chosen] = 1
        assignment = np.zeros((inst.I, inst.J), dtype=int)
        for i in range(inst.I):
            for j in range(inst.J):
                feasible = np.where((x == 1) & reachable[:, j] & (~forbidden) & (inst.transfer_count[i] <= inst.max_transfer_count))[0]
                if len(feasible) == 0 or np.random.rand() < 0.10:
                    assignment[i, j] = 0
                else:
                    score = _assignment_score(feasible, j, inst, source_idx=i)
                    score = score / (score.sum() + 1e-12)
                    assignment[i, j] = int(np.random.choice(feasible, p=score)) + 1
        dec = np.concatenate([x, assignment.ravel()])
        pop[n] = repair_solution(dec, inst) if use_repair else normalize_solution(dec, inst)
    return pop


def evaluate_solution(dec: np.ndarray, inst: MetroLogisticsInstance, use_repair=True, collect_parts: bool = True) -> Tuple[np.ndarray, float, Dict[str, float]]:
    dec = repair_solution(dec, inst) if use_repair else normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    q = inst.Q_j.astype(float)
    q_total = float(np.sum(q))
    station_flow = np.zeros(inst.K, dtype=float)
    line_items = _empty_line_items(inst)

    fixed_cost = float(np.sum(inst.fixed_cost * x))
    direct_operating_cost = direct_carbon_cost = 0.0
    access_operating_cost = access_carbon_cost = 0.0
    metro_electric_cost = metro_carbon_cost = 0.0
    feeder_operating_cost = feeder_carbon_cost = 0.0
    transfer_cost = inout_cost = 0.0
    road_carbon_kg = metro_carbon_kg = feeder_carbon_kg = 0.0
    direct_volume = metro_volume = weighted_feeder_distance = 0.0
    cv = 0.0
    source_share = inst.source_share / (np.sum(inst.source_share) + 1e-12)

    # 4km 服务圈理论可覆盖需求：覆盖率最终按全局计算，但该指标用于解释空间服务边界。
    space_coverage_4km_volume = 0.0
    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    for i in range(inst.I):
        feasible_by_source = (x == 1) & (inst.transfer_count[i] <= inst.max_transfer_count)
        for j in range(inst.J):
            q_ij_space = source_share[i] * q[j]
            if np.any(feasible_by_source & reachable[:, j]):
                space_coverage_4km_volume += float(q_ij_space)

    for i in range(inst.I):
        for j in range(inst.J):
            q_ij = source_share[i] * q[j]
            if q_ij <= 1e-12:
                continue
            if assignment[i, j] == 0:
                distance = inst.d_ij_road[i, j]
                direct_operating_cost += q_ij * distance * inst.road_operating_cost
                direct_carbon_cost += q_ij * distance * inst.road_carbon_cost
                road_carbon_kg += q_ij * distance * inst.road_emission_factor
                direct_volume += q_ij
                continue

            k = assignment[i, j] - 1
            station_flow[k] += q_ij
            metro_volume += q_ij
            feeder_distance = inst.d_kj_road[k, j]
            weighted_feeder_distance += q_ij * feeder_distance

            access_distance = inst.d_is_road[i]
            access_operating_cost += q_ij * access_distance * inst.road_operating_cost
            access_carbon_cost += q_ij * access_distance * inst.road_carbon_cost
            road_carbon_kg += q_ij * access_distance * inst.road_emission_factor

            metro_distance = inst.d_metro[i, k]
            metro_electric_cost += q_ij * metro_distance * inst.metro_electric_cost
            metro_carbon_cost += q_ij * metro_distance * inst.metro_carbon_cost
            metro_carbon_kg += q_ij * metro_distance * inst.metro_emission_factor

            transfer_cost += q_ij * inst.transfer_count[i, k] * inst.transfer_cost
            inout_cost += q_ij * inst.inout_cost
            feeder_operating_cost += q_ij * inst.feeder_cost * feeder_distance
            feeder_carbon_cost += q_ij * inst.feeder_carbon_cost * feeder_distance
            feeder_carbon_kg += q_ij * inst.feeder_emission_factor * feeder_distance

            for line in _path_lines(inst, i, k):
                line_items.setdefault(line, []).append(float(q_ij))

            if x[k] == 0:
                cv += 1.0
            if feeder_distance > inst.D_k_max[k]:
                cv += feeder_distance - inst.D_k_max[k]
            if inst.transfer_count[i, k] > inst.max_transfer_count:
                cv += 1.0

    packing = _evaluate_train_packing(line_items, inst)
    line_flow = packing["line_flow"]
    train_organization_cost = float(packing["train_organization_cost"])

    road_operating_cost = direct_operating_cost + access_operating_cost
    road_carbon_cost = direct_carbon_cost + access_carbon_cost
    road_transport_cost = direct_operating_cost + direct_carbon_cost
    access_cost = access_operating_cost + access_carbon_cost
    metro_transport_cost = metro_electric_cost + metro_carbon_cost
    feeder_cost = feeder_operating_cost + feeder_carbon_cost
    carbon_cost = road_carbon_cost + metro_carbon_cost + feeder_carbon_cost
    total_carbon_kg = road_carbon_kg + metro_carbon_kg + feeder_carbon_kg
    load_rates = np.divide(station_flow, inst.station_capacity, out=np.zeros_like(station_flow), where=inst.station_capacity > 0)
    total_cost = (
        fixed_cost
        + road_transport_cost
        + access_cost
        + metro_transport_cost
        + transfer_cost
        + inout_cost
        + feeder_cost
        + train_organization_cost
    )

    open_count = int(x.sum())
    if open_count < inst.min_open:
        cv += inst.min_open - open_count
    if open_count > inst.max_open:
        cv += open_count - inst.max_open

    for k in range(inst.K):
        if x[k] == 1:
            upper = inst.max_load_rate * inst.station_capacity[k]
            lower = inst.min_load_rate * inst.station_capacity[k]
            if station_flow[k] > upper:
                cv += (station_flow[k] - upper) / 1000.0
            if station_flow[k] < lower:
                cv += (lower - station_flow[k]) / 1000.0
        elif station_flow[k] > 1e-9:
            cv += station_flow[k] / 1000.0

    avg_feeder_distance = weighted_feeder_distance / (metro_volume + 1e-12)
    if metro_volume > 1e-9 and avg_feeder_distance > inst.max_avg_feeder_distance:
        cv += avg_feeder_distance - inst.max_avg_feeder_distance

    for a in range(inst.K):
        if x[a] != 1:
            continue
        for b in range(a + 1, inst.K):
            if x[b] == 1 and inst.d_kk_road[a, b] < inst.min_station_distance:
                cv += 1.0

    line_open_counts = open_station_counts_by_line(x, inst)
    for count in line_open_counts.values():
        if count > inst.max_open_per_line:
            cv += count - inst.max_open_per_line

    line_capacity_violation = 0.0
    train_capacity_violation_pieces = 0.0
    for line, violation in packing["violation_pieces"].items():
        cap = float((inst.line_capacity or {}).get(line, 0.0))
        train_capacity_violation_pieces += float(violation)
        if violation > 0:
            line_capacity_violation += float(violation) / max(cap, 1.0)
    cv += line_capacity_violation

    metro_service_ratio = metro_volume / (q_total + 1e-12)
    unit_generalized_cost = total_cost / (q_total + 1e-12)
    space_coverage_4km_ratio = space_coverage_4km_volume / (q_total + 1e-12)
    fulfillment_ratio_within_4km = metro_volume / (space_coverage_4km_volume + 1e-12) if space_coverage_4km_volume > 1e-9 else 0.0

    if not collect_parts:
        return np.array([total_cost, -metro_service_ratio], dtype=float), float(cv), {}

    station_labels = np.asarray(inst.station_type_labels)
    num_sl = int(np.sum((x == 1) & (station_labels == "SL")))
    num_tr = int(np.sum((x == 1) & (station_labels == "TR")))
    num_hb = int(np.sum((x == 1) & (station_labels == "HB")))
    ratio_sl = num_sl / open_count if open_count else 0.0
    ratio_tr = num_tr / open_count if open_count else 0.0
    ratio_hb = num_hb / open_count if open_count else 0.0

    parts = {
        "model_setting": inst.model_setting,
        "fixed_cost_reference_lower": inst.fixed_cost_lower,
        "fixed_cost_reference_upper_all_stations": inst.fixed_cost_upper,
        "fixed_cost": fixed_cost,
        "fixed_cost_total": fixed_cost,
        "road_cost": road_transport_cost + access_cost,
        "road_transport_cost": road_transport_cost,
        "road_operating_cost": road_operating_cost,
        "road_fuel_cost": road_operating_cost,
        "road_carbon_cost": road_carbon_cost,
        "direct_cost": road_transport_cost,
        "direct_operating_cost": direct_operating_cost,
        "direct_fuel_cost": direct_operating_cost,
        "direct_carbon_cost": direct_carbon_cost,
        "access_cost": access_cost,
        "access_operating_cost": access_operating_cost,
        "access_fuel_cost": access_operating_cost,
        "access_carbon_cost": access_carbon_cost,
        "metro_cost": metro_transport_cost,
        "metro_electric_cost": metro_electric_cost,
        "metro_carbon_cost": metro_carbon_cost,
        "feeder_cost": feeder_cost,
        "feeder_operating_cost": feeder_operating_cost,
        "feeder_carbon_cost": feeder_carbon_cost,
        "transfer_cost": transfer_cost,
        "inout_cost": inout_cost,
        "train_organization_cost": train_organization_cost,
        "carbon_cost": carbon_cost,
        "road_carbon_kg": road_carbon_kg,
        "metro_carbon_kg": metro_carbon_kg,
        "feeder_carbon_kg": feeder_carbon_kg,
        "total_carbon_kg": total_carbon_kg,
        "total_carbon_tCO2": total_carbon_kg / 1000.0,
        "unit_carbon_kg_per_piece": total_carbon_kg / (q_total + 1e-12),
        "carbon_cost_ratio": carbon_cost / (total_cost + 1e-12),
        "metro_carbon_cost_ratio": metro_carbon_cost / (carbon_cost + 1e-12) if carbon_cost > 0 else 0.0,
        "Z1": total_cost,
        "Z1_unit_cost": unit_generalized_cost,
        "unit_generalized_cost": unit_generalized_cost,
        "total_demand": q_total,
        "Z2": metro_service_ratio,
        "Z2_metro_service_ratio": metro_service_ratio,
        "CV": cv,
        "num_open": open_count,
        "num_sl": num_sl,
        "num_tr": num_tr,
        "num_hb": num_hb,
        "ratio_sl": ratio_sl,
        "ratio_tr": ratio_tr,
        "ratio_hb": ratio_hb,
        "metro_volume": metro_volume,
        "direct_volume": direct_volume,
        "metro_volume_ratio": metro_service_ratio,
        "direct_volume_ratio": direct_volume / (q_total + 1e-12),
        "space_coverage_4km_volume": space_coverage_4km_volume,
        "space_coverage_4km_ratio": space_coverage_4km_ratio,
        "fulfilled_4km_volume": metro_volume,
        "fulfillment_ratio_within_4km": fulfillment_ratio_within_4km,
        "avg_feeder_distance": avg_feeder_distance,
        "max_load_rate": float(np.max(load_rates)) if len(load_rates) else 0.0,
        "max_open_per_line": inst.max_open_per_line,
        "max_line_open_count": max(line_open_counts.values()) if line_open_counts else 0,
        "line_capacity_violation": line_capacity_violation,
        "train_capacity_violation_pieces": train_capacity_violation_pieces,
        "available_train_trips": inst.available_train_trips,
        "used_train_trips_total": packing["total_used_trips"],
        "max_train_load_rate": packing["max_train_load_rate"],
        "max_line_flow": max(line_flow.values()) if line_flow else 0.0,
        "max_line_load_rate": packing["max_train_load_rate"],
    }
    return np.array([total_cost, -metro_service_ratio], dtype=float), float(cv), parts


def _evaluate_objective_cv_fast(dec: np.ndarray, inst: MetroLogisticsInstance, use_repair=True) -> tuple[np.ndarray, float]:
    """Objective/CV evaluation used during iterations.

    The confirmed model includes indivisible train-trip packing, so the fast path
    delegates to the same evaluator used for final reporting to keep CV semantics
    consistent.
    """
    obj, cv, _ = evaluate_solution(dec, inst, use_repair=use_repair, collect_parts=False)
    return obj, cv


def _station_lines_cache(inst: MetroLogisticsInstance) -> List[List[str]]:
    cache = getattr(inst, "_station_lines_cache", None)
    if cache is None:
        cache = [_split_lines(str(text)) for text in inst.candidate_df["所有线路"].astype(str).tolist()]
        setattr(inst, "_station_lines_cache", cache)
    return cache


def _path_lines_cache(inst: MetroLogisticsInstance) -> List[List[List[str]]]:
    cache = getattr(inst, "_path_lines_cache", None)
    if cache is not None:
        return cache
    if inst.path_line_incidence is None or not inst.line_names:
        station_cache = _station_lines_cache(inst)
        cache = [[list(station_cache[k]) for k in range(inst.K)] for _ in range(inst.I)]
    else:
        cache = []
        for i in range(inst.I):
            row = []
            for k in range(inst.K):
                if i >= inst.path_line_incidence.shape[0] or k >= inst.path_line_incidence.shape[1]:
                    row.append([])
                else:
                    flags = inst.path_line_incidence[i, k]
                    row.append([line for line, flag in zip(inst.line_names, flags) if flag > 0.5])
            cache.append(row)
    setattr(inst, "_path_lines_cache", cache)
    return cache


def _empty_line_flow(inst: MetroLogisticsInstance) -> Dict[str, float]:
    lines = set()
    for station_lines in _station_lines_cache(inst):
        lines.update(station_lines)
    return {line: 0.0 for line in lines}


def _empty_line_items(inst: MetroLogisticsInstance) -> Dict[str, List[float]]:
    lines = list(inst.line_names or sorted(_empty_line_flow(inst)))
    return {line: [] for line in lines}


def _pack_volumes_ffd(volumes: List[float], capacity: float, trips: int) -> tuple[List[float], float, int]:
    """First-Fit Decreasing packing for indivisible freight-flow units.

    Returns per-trip loads, violation in pieces, and used trip count.
    """
    trips = max(0, int(trips))
    loads = [0.0 for _ in range(trips)]
    violation = 0.0
    if trips <= 0 or capacity <= 0:
        return loads, float(sum(volumes)), 0
    for vol in sorted([float(v) for v in volumes if float(v) > 1e-9], reverse=True):
        placed = False
        for idx in range(trips):
            if loads[idx] + vol <= capacity + 1e-9:
                loads[idx] += vol
                placed = True
                break
        if not placed:
            violation += vol
    used = sum(1 for load in loads if load > 1e-9)
    return loads, violation, used


def _evaluate_train_packing(line_items: Dict[str, List[float]], inst: MetroLogisticsInstance) -> Dict[str, object]:
    trip_loads: Dict[str, List[float]] = {}
    line_flow: Dict[str, float] = {}
    violation_pieces: Dict[str, float] = {}
    used_trips: Dict[str, int] = {}
    max_load_rate = 0.0
    total_used = 0
    for line in (inst.line_names or sorted(line_items)):
        volumes = line_items.get(line, [])
        capacity = float((inst.line_capacity or {}).get(line, 0.0))
        loads, violation, used = _pack_volumes_ffd(volumes, capacity, inst.available_train_trips)
        trip_loads[line] = loads
        line_flow[line] = float(sum(volumes))
        violation_pieces[line] = float(violation)
        used_trips[line] = int(used)
        total_used += int(used)
        if capacity > 0 and loads:
            max_load_rate = max(max_load_rate, max(loads) / capacity)
    train_org_cost = 0.0
    for line, used in used_trips.items():
        train_org_cost += float((inst.train_organization_cost or {}).get(line, 0.0)) * used
    return {
        "trip_loads": trip_loads,
        "line_flow": line_flow,
        "violation_pieces": violation_pieces,
        "used_trips": used_trips,
        "total_violation_pieces": float(sum(violation_pieces.values())),
        "total_used_trips": int(total_used),
        "max_train_load_rate": float(max_load_rate),
        "train_organization_cost": float(train_org_cost),
    }


def _station_lines(inst: MetroLogisticsInstance, k: int) -> List[str]:
    return _station_lines_cache(inst)[int(k)]


def _path_lines(inst: MetroLogisticsInstance, i: int, k: int) -> List[str]:
    return _path_lines_cache(inst)[int(i)][int(k)]


def _split_lines(text: str) -> List[str]:
    clean = text.replace("|", ",").replace("，", ",").replace("/", ",").replace("、", ",").replace(";", ",")
    return [part.strip() for part in clean.split(",") if part.strip()]


def dominates(obj_a, obj_b, cv_a, cv_b):
    eps = 1e-9
    a_feasible = cv_a <= eps
    b_feasible = cv_b <= eps
    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        return cv_a < cv_b
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def fast_non_dominated_sort(objs: np.ndarray, cvs: np.ndarray):
    """Vectorized constrained non-dominated sorting.

    The previous implementation used nested Python loops over all pairs.  For
    thesis-grade runs (p=200, g=400, many ablation runs) that became the main
    bottleneck.  This version builds the dominance relation with NumPy boolean
    matrices and keeps the same constraint-domination semantics:
      * feasible dominates infeasible;
      * between infeasible solutions, smaller CV dominates;
      * between feasible solutions, Pareto dominance on objective values.
    """
    objs = np.asarray(objs, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    n = len(objs)
    if n == 0:
        return [[]], np.zeros(0, dtype=int)
    eps = 1e-9
    feasible = cvs <= eps

    obj_le = np.all(objs[:, None, :] <= objs[None, :, :] + 1e-12, axis=2)
    obj_lt = np.any(objs[:, None, :] < objs[None, :, :] - 1e-12, axis=2)
    both_feasible = feasible[:, None] & feasible[None, :]
    feasible_dom = both_feasible & obj_le & obj_lt

    feas_over_infeas = feasible[:, None] & (~feasible[None, :])
    infeas_over_infeas = (~feasible[:, None]) & (~feasible[None, :]) & (cvs[:, None] < cvs[None, :] - 1e-12)
    dominates_matrix = feasible_dom | feas_over_infeas | infeas_over_infeas
    np.fill_diagonal(dominates_matrix, False)

    dominated_count = dominates_matrix.sum(axis=0).astype(int)
    dominated_sets = [np.flatnonzero(dominates_matrix[i]).astype(int) for i in range(n)]
    rank = np.zeros(n, dtype=int)
    current = np.flatnonzero(dominated_count == 0).astype(int)
    fronts: List[List[int]] = []
    level = 0
    while len(current) > 0:
        front = current.tolist()
        fronts.append(front)
        next_front = []
        for p in current:
            dominated = dominated_sets[int(p)]
            if len(dominated) == 0:
                continue
            dominated_count[dominated] -= 1
            newly = dominated[dominated_count[dominated] == 0]
            if len(newly) > 0:
                rank[newly] = level + 1
                next_front.extend(newly.tolist())
        if next_front:
            # Preserve uniqueness; ties can be inserted from multiple parents.
            current = np.array(sorted(set(next_front)), dtype=int)
        else:
            current = np.array([], dtype=int)
        level += 1
    if not fronts:
        fronts = [list(range(n))]
    return fronts, rank

def crowding_distance(objs: np.ndarray, front: List[int]):
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    dist = np.zeros(len(front), dtype=float)
    front_objs = objs[front]
    for m in range(front_objs.shape[1]):
        order = np.argsort(front_objs[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        denom = front_objs[order[-1], m] - front_objs[order[0], m]
        if abs(denom) < 1e-12:
            continue
        for i in range(1, len(front) - 1):
            dist[order[i]] += (front_objs[order[i + 1], m] - front_objs[order[i - 1], m]) / denom
    return dist


def rank_and_crowding(objs, cvs):
    fronts, rank = fast_non_dominated_sort(objs, cvs)
    crowd = np.zeros(len(objs), dtype=float)
    for front in fronts:
        cd = crowding_distance(objs, front)
        for local, idx in enumerate(front):
            crowd[idx] = cd[local]
    return fronts, rank, crowd


def tournament_select(pop, objs, cvs, rank, crowd):
    i, j = np.random.randint(0, len(pop), size=2)
    if rank[i] < rank[j] or (rank[i] == rank[j] and crowd[i] >= crowd[j]):
        return pop[i].copy()
    return pop[j].copy()


def crossover(parent1, parent2, inst: MetroLogisticsInstance, pc=0.9):
    if np.random.rand() > pc:
        return parent1.copy(), parent2.copy()
    c1, c2 = parent1.copy(), parent2.copy()
    mask_x = np.random.rand(inst.K) < 0.5
    c1[:inst.K][mask_x] = parent2[:inst.K][mask_x]
    c2[:inst.K][mask_x] = parent1[:inst.K][mask_x]
    start_a, end_a = _assignment_start(inst), _assignment_end(inst)
    mask_a = np.random.rand(inst.I * inst.J) < 0.5
    c1[start_a:end_a][mask_a] = parent2[start_a:end_a][mask_a]
    c2[start_a:end_a][mask_a] = parent1[start_a:end_a][mask_a]
    return repair_solution(c1, inst), repair_solution(c2, inst)


def crossover_no_repair(parent1, parent2, inst: MetroLogisticsInstance, pc=0.9):
    if np.random.rand() > pc:
        return normalize_solution(parent1, inst), normalize_solution(parent2, inst)
    c1, c2 = parent1.copy(), parent2.copy()
    mask_x = np.random.rand(inst.K) < 0.5
    c1[:inst.K][mask_x] = parent2[:inst.K][mask_x]
    c2[:inst.K][mask_x] = parent1[:inst.K][mask_x]
    start_a, end_a = _assignment_start(inst), _assignment_end(inst)
    mask_a = np.random.rand(inst.I * inst.J) < 0.5
    c1[start_a:end_a][mask_a] = parent2[start_a:end_a][mask_a]
    c2[start_a:end_a][mask_a] = parent1[start_a:end_a][mask_a]
    return normalize_solution(c1, inst), normalize_solution(c2, inst)


def mutate(dec, inst: MetroLogisticsInstance, pm_site=None, pm_assign=None, pm_source=None):
    dec = dec.copy()
    pm_site = 1.0 / inst.K if pm_site is None else pm_site
    pm_assign = 1.0 / (inst.I * inst.J) if pm_assign is None else pm_assign
    for k in range(inst.K):
        if np.random.rand() < pm_site:
            dec[k] = 1 - np.rint(dec[k])
    x, _ = decode_solution(dec, inst)
    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    forbidden = _forbidden_station_mask(inst)
    for i in range(inst.I):
        for j in range(inst.J):
            if np.random.rand() < pm_assign:
                feasible = np.where((x == 1) & reachable[:, j] & (~forbidden) & (inst.transfer_count[i] <= inst.max_transfer_count))[0]
                idx = _flat_assignment_index(inst, i, j)
                if len(feasible) > 0 and np.random.rand() > 0.15:
                    score = _assignment_score(feasible, j, inst, source_idx=i)
                    score = score / (score.sum() + 1e-12)
                    dec[idx] = int(np.random.choice(feasible, p=score)) + 1
                else:
                    dec[idx] = 0
    return repair_solution(dec, inst)


def mutate_no_repair(dec, inst: MetroLogisticsInstance, pm_site=None, pm_assign=None, pm_source=None):
    dec = normalize_solution(dec, inst)
    pm_site = 1.0 / inst.K if pm_site is None else pm_site
    pm_assign = 1.0 / (inst.I * inst.J) if pm_assign is None else pm_assign
    for k in range(inst.K):
        if np.random.rand() < pm_site:
            dec[k] = 1 - np.rint(dec[k])
    for i in range(inst.I):
        for j in range(inst.J):
            if np.random.rand() < pm_assign:
                dec[_flat_assignment_index(inst, i, j)] = np.random.randint(0, inst.K + 1)
    return normalize_solution(dec, inst)


def evaluate_population(pop, inst, use_repair=True, collect_parts: bool = False, use_cache: bool = True):
    """Evaluate a population.

    ``collect_parts`` is intentionally False during evolutionary iterations to
    avoid constructing large per-solution dictionaries thousands of times.  Set
    it to True only for final reporting/export.
    """
    objs = np.zeros((len(pop), 2), dtype=float)
    cvs = np.zeros(len(pop), dtype=float)
    parts = [] if collect_parts else None
    cache: Dict[bytes, tuple[np.ndarray, float, Dict[str, float]]] = {} if use_cache else None
    for i, dec in enumerate(pop):
        key = None
        if use_cache:
            normalized = normalize_solution(dec, inst)
            key = normalized.astype(np.int16, copy=False).tobytes()
            cached = cache.get(key) if cache is not None else None
            if cached is not None:
                obj, cv, part = cached
                objs[i] = obj
                cvs[i] = cv
                if collect_parts:
                    parts.append(part)
                continue
        if collect_parts:
            obj, cv, part = evaluate_solution(dec, inst, use_repair=use_repair, collect_parts=True)
        else:
            obj, cv = _evaluate_objective_cv_fast(dec, inst, use_repair=use_repair)
            part = {}
        objs[i], cvs[i] = obj, cv
        if use_cache and key is not None and cache is not None:
            cache[key] = (obj.copy(), float(cv), part)
        if collect_parts:
            parts.append(part)
    return objs, cvs, (parts if collect_parts else [])

def run_nsga2(inst: MetroLogisticsInstance, pop_size=100, max_gen=300, pc=0.9, seed=None, verbose=True,
                       use_repair=True, archive_size=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    start_time = time.time()
    pop = initialize_population(inst, pop_size, use_repair=use_repair)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=use_repair)
    archive_pop = archive_objs = archive_cvs = None
    if archive_size is not None:
        archive_size = max(1, int(archive_size))
        archive_pop, archive_objs, archive_cvs = _update_pareto_archive(pop, objs, cvs, archive_size)
    history = []

    for gen in range(max_gen + 1):
        fronts, rank, crowd = rank_and_crowding(objs, cvs)
        front0 = fronts[0]
        best_cost = float(np.min(objs[front0, 0]))
        best_service_ratio = float(np.max(-objs[front0, 1]))
        feasible_count = int(np.sum(cvs <= 1e-9))
        history.append({
            "gen": gen,
            "best_cost": best_cost,
            "best_service_ratio": best_service_ratio,
            "pareto_size": len(front0),
            "feasible_count": feasible_count,
            "min_cv": float(np.min(cvs)),
        })
        if verbose and gen % 20 == 0:
            print(f"[NSGA-II] Gen {gen:4d}: Pareto={len(front0):3d}, feasible={feasible_count:3d}, minCV={np.min(cvs):.4f}, Z1={best_cost:,.2f}, Z2={best_service_ratio:.4f}")
        if gen == max_gen:
            break

        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, objs, cvs, rank, crowd)
            p2 = tournament_select(pop, objs, cvs, rank, crowd)
            c1, c2 = crossover_no_repair(p1, p2, inst, pc=pc)
            offspring.append(mutate(c1, inst) if use_repair else mutate_no_repair(c1, inst))
            if len(offspring) < pop_size:
                offspring.append(mutate(c2, inst) if use_repair else mutate_no_repair(c2, inst))
        offspring = np.array(offspring[:pop_size])
        off_objs, off_cvs, _ = evaluate_population(offspring, inst, use_repair=use_repair)

        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, off_objs])
        combined_cvs = np.concatenate([cvs, off_cvs])
        if archive_size is not None:
            archive_pop, archive_objs, archive_cvs = _update_pareto_archive(
                np.vstack([archive_pop, combined_pop]),
                np.vstack([archive_objs, combined_objs]),
                np.concatenate([archive_cvs, combined_cvs]),
                archive_size,
            )
        fronts, _, _ = rank_and_crowding(combined_objs, combined_cvs)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                cd = crowding_distance(combined_objs, front)
                order = np.argsort(-cd)
                selected.extend([front[i] for i in order[:pop_size - len(selected)]])
                break
        selected = np.array(selected, dtype=int)
        pop = combined_pop[selected]
        objs = combined_objs[selected]
        cvs = combined_cvs[selected]

    fronts, _, _ = rank_and_crowding(objs, cvs)
    feasible_front = [idx for idx in fronts[0] if cvs[idx] <= 1e-9]
    pareto_idx = np.array(feasible_front if feasible_front else fronts[0], dtype=int)
    result = {
        "algorithm": "NSGA-II",
        "pop": pop,
        "objs": objs,
        "cvs": cvs,
        "pareto_dec": pop[pareto_idx],
        "pareto_objs": objs[pareto_idx],
        "pareto_cvs": cvs[pareto_idx],
        "history": pd.DataFrame(history),
        "runtime": time.time() - start_time,
    }
    if archive_size is not None:
        archive_fronts, _, _ = rank_and_crowding(archive_objs, archive_cvs)
        archive_feasible_front = [idx for idx in archive_fronts[0] if archive_cvs[idx] <= 1e-9]
        archive_pareto_idx = np.array(archive_feasible_front if archive_feasible_front else archive_fronts[0], dtype=int)
        result.update({
            "archive_dec": archive_pop,
            "archive_objs": archive_objs,
            "archive_cvs": archive_cvs,
            "archive_pareto_dec": archive_pop[archive_pareto_idx],
            "archive_pareto_objs": archive_objs[archive_pareto_idx],
            "archive_pareto_cvs": archive_cvs[archive_pareto_idx],
            "archive_size": int(archive_size),
        })
    return result


# The formal algorithm set is intentionally limited to NSGA-II and QVNS-NSGA-II.
# Earlier experimental helper solvers were removed to keep the release code and
# thesis description aligned.


def _select_by_fronts(combined_pop, combined_objs, combined_cvs, fronts, pop_size):
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            cd = crowding_distance(combined_objs, front)
            order = np.argsort(-cd)
            selected.extend([front[i] for i in order[:pop_size - len(selected)]])
            break
    selected = np.array(selected, dtype=int)
    return combined_pop[selected], combined_objs[selected], combined_cvs[selected]


def _nondominated_indices_constrained(objs: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    """Return constrained nondominated indices without full front layering.

    This helper is used by external archives during long p200/g400 runs.  It
    avoids the iterative front construction needed for population selection and
    prevents rare archive-update stalls caused by deep dominated-front layering.
    Constraint semantics are identical to ``fast_non_dominated_sort``.
    """
    objs = np.asarray(objs, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    n = len(objs)
    if n == 0:
        return np.array([], dtype=int)
    eps = 1e-9
    feasible = cvs <= eps
    obj_le = np.all(objs[:, None, :] <= objs[None, :, :] + 1e-12, axis=2)
    obj_lt = np.any(objs[:, None, :] < objs[None, :, :] - 1e-12, axis=2)
    both_feasible = feasible[:, None] & feasible[None, :]
    feasible_dom = both_feasible & obj_le & obj_lt
    feas_over_infeas = feasible[:, None] & (~feasible[None, :])
    infeas_over_infeas = (~feasible[:, None]) & (~feasible[None, :]) & (cvs[:, None] < cvs[None, :] - 1e-12)
    dominates_matrix = feasible_dom | feas_over_infeas | infeas_over_infeas
    np.fill_diagonal(dominates_matrix, False)
    dominated_count = dominates_matrix.sum(axis=0)
    return np.flatnonzero(dominated_count == 0).astype(int)


def _trim_archive_indices(
    objs: np.ndarray,
    selected: np.ndarray,
    archive_size: int,
    boundary_keep_rate: float = 0.0,
) -> np.ndarray:
    """Trim an external archive while preserving boundary and sparse-front points.

    ``boundary_keep_rate`` keeps a small share of extreme cost/service points
    before crowding-distance truncation.  This supports Q8/Q10/Q11, whose
    goal is lower Spacing and more unique Pareto points rather than simply
    higher HV.  With the default value 0.0 the behavior is backward compatible.
    """
    selected = np.asarray(selected, dtype=int)
    archive_size = int(archive_size)
    if len(selected) <= archive_size:
        return selected

    boundary_keep_rate = max(0.0, min(float(boundary_keep_rate), 0.50))
    boundary = []
    if boundary_keep_rate > 0.0 and objs.shape[1] >= 2:
        keep_n = max(2, int(round(archive_size * boundary_keep_rate)))
        selected_objs = objs[selected]
        # Objective 0 is cost to minimize; objective 1 is -service to minimize,
        # so low values in column 1 are high-service boundary points.
        cost_order = selected[np.argsort(selected_objs[:, 0])]
        service_order = selected[np.argsort(selected_objs[:, 1])]
        for idx in list(cost_order[: max(1, keep_n // 2)]) + list(service_order[: max(1, keep_n - keep_n // 2)]):
            if int(idx) not in boundary:
                boundary.append(int(idx))
            if len(boundary) >= keep_n:
                break

    remaining = np.array([idx for idx in selected if int(idx) not in set(boundary)], dtype=int)
    if len(remaining) == 0:
        return np.array(boundary[:archive_size], dtype=int)
    cd = crowding_distance(objs, remaining.tolist())
    order = np.argsort(-cd)
    need = max(0, archive_size - len(boundary))
    trimmed = list(boundary) + remaining[order[:need]].astype(int).tolist()
    return np.array(trimmed[:archive_size], dtype=int)


def _update_archive(
    pop: np.ndarray,
    objs: np.ndarray,
    cvs: np.ndarray,
    archive_size: int,
    boundary_keep_rate: float = 0.0,
):
    selected = _nondominated_indices_constrained(objs, cvs)
    if len(selected) == 0:
        selected = np.argsort(cvs)[:max(1, int(archive_size))]
    selected = _trim_archive_indices(objs, selected, archive_size, boundary_keep_rate=boundary_keep_rate)
    return pop[selected], objs[selected], cvs[selected]


def _update_pareto_archive(
    pop: np.ndarray,
    objs: np.ndarray,
    cvs: np.ndarray,
    archive_size: int,
    boundary_keep_rate: float = 0.0,
):
    selected = _nondominated_indices_constrained(objs, cvs)
    if len(selected) == 0:
        selected = np.argsort(cvs)[:max(1, int(archive_size))]
    feasible_selected = selected[cvs[selected] <= 1e-9]
    if len(feasible_selected) > 0:
        selected = feasible_selected
    selected = _trim_archive_indices(objs, selected, archive_size, boundary_keep_rate=boundary_keep_rate)
    return pop[selected], objs[selected], cvs[selected]


QL_ACTION_NAMES = [
    "standard-conservative",
    "standard-baseline",
    "standard-diversify",
    "service-ratio-oriented",
    "cost-oriented",
]


def _qlearning_nsga2_actions(profile: str = "safe"):
    """Return Q-learning action space.

    The default ``safe`` profile is deliberately conservative: the Q-learning
    module proposes mild mutations and lets the external archive/VNS dominate
    exploitation.  This avoids the earlier failure mode where a single
    aggressive Q action shifted a whole generation away from the good VNS front.
    Set ``profile='legacy'`` to reproduce the previous, stronger operators.
    """
    if str(profile).lower() == "legacy":
        return [
            {"name": "standard-conservative", "operator": "standard", "pm_factor": 0.45, "pc_factor": 1.00, "prior": 0.03},
            {"name": "standard-baseline", "operator": "standard", "pm_factor": 1.00, "pc_factor": 1.00, "prior": 0.06},
            {"name": "standard-diversify", "operator": "standard", "pm_factor": 1.45, "pc_factor": 1.00, "prior": 0.02},
            {"name": "service-ratio-oriented", "operator": "service_ratio", "pm_factor": 0.85, "pc_factor": 1.00, "open_prob": 0.55, "direct_to_metro_prob": 0.65, "improve_assign_prob": 0.45, "prior": 0.08},
            {"name": "cost-oriented", "operator": "cost", "pm_factor": 0.70, "pc_factor": 0.90, "close_prob": 0.45, "direct_prob": 0.30, "prior": 0.01},
        ]
    return [
        {"name": "standard-conservative", "operator": "standard", "pm_factor": 0.40, "pc_factor": 1.00, "prior": 0.06},
        {"name": "standard-baseline", "operator": "standard", "pm_factor": 0.90, "pc_factor": 1.00, "prior": 0.10},
        {"name": "standard-diversify", "operator": "standard", "pm_factor": 1.20, "pc_factor": 1.00, "prior": 0.04},
        {
            "name": "service-ratio-oriented",
            "operator": "service_ratio",
            "pm_factor": 0.65,
            "pc_factor": 1.00,
            "open_prob": 0.22,
            "direct_to_metro_prob": 0.25,
            "improve_assign_prob": 0.16,
            "prior": 0.02,
        },
        {
            "name": "cost-oriented",
            "operator": "cost",
            "pm_factor": 0.55,
            "pc_factor": 0.98,
            "close_prob": 0.12,
            "direct_prob": 0.08,
            "prior": -0.01,
        },
    ]


def _qlearning_state(gen: int, max_gen: int, objs: np.ndarray, cvs: np.ndarray) -> str:
    feasible_rate_value = float(np.mean(cvs <= 1e-9)) if len(cvs) else 0.0
    progress = gen / max(1, max_gen)
    feasible_bucket = int(np.digitize(feasible_rate_value, [0.25, 0.60, 0.90]))
    progress_bucket = int(np.digitize(progress, [0.25, 0.50, 0.75]))
    fronts, _, _ = rank_and_crowding(objs, cvs)
    nd_ratio = len(fronts[0]) / max(1, len(objs)) if fronts else 0.0
    diversity_bucket = int(np.digitize(nd_ratio, [0.10, 0.30, 0.55]))
    return f"p{progress_bucket}_f{feasible_bucket}_d{diversity_bucket}"


def _qlearning_choose_action(values: np.ndarray, epsilon: float):
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, len(values))), "explore"
    max_value = float(np.max(values))
    best = np.flatnonzero(np.abs(values - max_value) <= 1e-12)
    return int(np.random.choice(best)), "exploit"


def _service_potential(inst: MetroLogisticsInstance) -> np.ndarray:
    """Demand that each candidate station can potentially transfer to metro, scaled by fixed cost."""
    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    potential_volume = reachable.astype(float) @ inst.Q_j.astype(float)
    capacity_factor = np.minimum(1.0, inst.station_capacity / (np.max(inst.station_capacity) + 1e-12))
    potential = potential_volume * (0.70 + 0.30 * capacity_factor) / (inst.fixed_cost + 1e-12)
    potential[_forbidden_station_mask(inst)] = 0.0
    return potential


def _line_spacing_model_open_candidates(x: np.ndarray, inst: MetroLogisticsInstance) -> np.ndarray:
    forbidden = _forbidden_station_mask(inst)
    open_sites = np.where(np.rint(x).astype(int) == 1)[0]
    closed = np.where((np.rint(x).astype(int) == 0) & (~forbidden))[0]
    if len(closed) == 0:
        return closed
    current_cost = _fixed_cost_total(x, inst)
    upper = _fixed_cost_feasible_upper(inst)
    candidates = []
    for k in closed:
        if len(open_sites) and np.any(inst.d_kk_road[k, open_sites] < inst.min_station_distance):
            continue
        trial = x.copy()
        trial[k] = 1
        if _fixed_cost_total(trial, inst) > upper + 1e-9:
            continue
        if any(count > inst.max_open_per_line for count in open_station_counts_by_line(trial, inst).values()):
            continue
        candidates.append(int(k))
    return np.array(candidates, dtype=int)


def _best_station_for_pair(open_sites: np.ndarray, i: int, j: int, inst: MetroLogisticsInstance, service_bias: float = 1.0) -> int | None:
    if len(open_sites) == 0:
        return None
    forbidden = _forbidden_station_mask(inst)
    feasible = open_sites[(inst.d_kj_road[open_sites, j] <= inst.D_k_max[open_sites]) & (~forbidden[open_sites]) & (inst.transfer_count[i, open_sites] <= inst.max_transfer_count)]
    if len(feasible) == 0:
        return None
    feeder = inst.d_kj_road[feasible, j]
    metro = inst.d_metro[i, feasible]
    transfer = inst.transfer_count[i, feasible]
    score = service_bias * inst.Q_j[j] / (1.0 + feeder + 0.25 * metro + 0.5 * transfer)
    score += 0.05 * inst.station_importance[feasible]
    return int(feasible[int(np.argmax(score))])


def _assign_direct_to_metro(dec: np.ndarray, inst: MetroLogisticsInstance, probability: float = 0.50, only_direct: bool = True) -> np.ndarray:
    dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    open_sites = np.where(x == 1)[0]
    if len(open_sites) == 0:
        return dec
    source_share = inst.source_share / (np.sum(inst.source_share) + 1e-12)
    pairs = [(source_share[i] * inst.Q_j[j], i, j) for i in range(inst.I) for j in range(inst.J)]
    pairs.sort(reverse=True)
    for _, i, j in pairs:
        if np.random.rand() > probability:
            continue
        if only_direct and assignment[i, j] != 0:
            continue
        k = _best_station_for_pair(open_sites, i, j, inst, service_bias=1.0)
        if k is not None:
            assignment[i, j] = k + 1
    dec[:inst.K] = x
    dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
    return dec


def _service_ratio_mutation(dec: np.ndarray, inst: MetroLogisticsInstance, action: dict, use_repair=True) -> np.ndarray:
    dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    potential = _service_potential(inst)

    # Nudge station opening toward high service-volume / fixed-cost candidates.
    if np.random.rand() < float(action.get("open_prob", 0.50)):
        candidates = _line_spacing_model_open_candidates(x, inst)
        if len(candidates) > 0:
            weights = potential[candidates]
            if np.sum(weights) <= 1e-12:
                add = int(np.random.choice(candidates))
            else:
                weights = weights / (weights.sum() + 1e-12)
                add = int(np.random.choice(candidates, p=weights))
            x[add] = 1
            dec[:inst.K] = x

    # Try turning direct flows into metro flows whenever distance/capacity constraints allow it.
    dec = _assign_direct_to_metro(dec, inst, probability=float(action.get("direct_to_metro_prob", 0.60)), only_direct=True)

    # Also improve a subset of existing metro assignments toward better service-ratio stations.
    dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    open_sites = np.where(x == 1)[0]
    for i in range(inst.I):
        for j in range(inst.J):
            if np.random.rand() < float(action.get("improve_assign_prob", 0.35)):
                k = _best_station_for_pair(open_sites, i, j, inst, service_bias=1.0)
                if k is not None:
                    assignment[i, j] = k + 1
    dec[:inst.K] = x
    dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
    return repair_solution(dec, inst) if use_repair else normalize_solution(dec, inst)


def _cost_oriented_mutation(dec: np.ndarray, inst: MetroLogisticsInstance, action: dict, use_repair=True) -> np.ndarray:
    dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    if np.random.rand() < float(action.get("close_prob", 0.40)):
        open_sites = np.where(x == 1)[0]
        if len(open_sites) > max(1, inst.min_open):
            # Close expensive low-assignment stations first.
            use_count = np.array([np.sum(assignment == k + 1) for k in open_sites], dtype=float)
            close_score = inst.fixed_cost[open_sites] / (1.0 + use_count)
            close = int(open_sites[int(np.argmax(close_score))])
            x[close] = 0
            assignment[assignment == close + 1] = 0
    for i in range(inst.I):
        for j in range(inst.J):
            if np.random.rand() < float(action.get("direct_prob", 0.25)):
                assignment[i, j] = 0
    dec[:inst.K] = x
    dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
    return repair_solution(dec, inst) if use_repair else normalize_solution(dec, inst)


def _qlearning_mutate(dec: np.ndarray, inst: MetroLogisticsInstance, action: dict, use_repair=True) -> np.ndarray:
    dec = dec.copy()
    factor = float(action["pm_factor"])
    pm_site = min(0.75, factor / max(1, inst.K))
    pm_assign = min(0.85, factor / max(1, inst.I * inst.J))
    operator = action.get("operator", "standard")
    if operator == "standard":
        return mutate(dec, inst, pm_site=pm_site, pm_assign=pm_assign) if use_repair else mutate_no_repair(dec, inst, pm_site=pm_site, pm_assign=pm_assign)
    if operator == "service_ratio":
        return _service_ratio_mutation(dec, inst, action, use_repair=use_repair)
    if operator == "cost":
        return _cost_oriented_mutation(dec, inst, action, use_repair=use_repair)
    return mutate(dec, inst, pm_site=pm_site, pm_assign=pm_assign) if use_repair else mutate_no_repair(dec, inst, pm_site=pm_site, pm_assign=pm_assign)


def _normalize_points_for_metrics(objs: np.ndarray, cvs: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    feasible = np.where(cvs <= 1e-9)[0]
    if len(feasible) == 0:
        return np.empty((0, 2), dtype=float)
    front, _, _ = rank_and_crowding(objs[feasible], np.zeros(len(feasible)))
    idx = feasible[np.array(front[0], dtype=int)] if front else feasible
    points = objs[idx].astype(float)
    denom = np.maximum(f_max - f_min, 1e-12)
    norm = (points - f_min) / denom
    return np.clip(norm, 0.0, 1.1)


def _hypervolume_2d_min(points: np.ndarray, ref=(1.1, 1.1)) -> float:
    if len(points) == 0:
        return 0.0
    pts = points[np.argsort(points[:, 0])]
    hv = 0.0
    best_y = float(ref[1])
    for x, y in pts:
        x = float(min(max(x, 0.0), ref[0]))
        y = float(min(max(y, 0.0), ref[1]))
        if y < best_y:
            hv += max(0.0, ref[0] - x) * max(0.0, best_y - y)
            best_y = y
    return float(hv)


def _front_diversity(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    dist = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    dist[dist <= 1e-12] = np.inf
    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.mean(nearest)) if len(nearest) else 0.0


def _population_mo_features(objs: np.ndarray, cvs: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> dict:
    norm_front = _normalize_points_for_metrics(objs, cvs, f_min, f_max)
    fronts, _, _ = rank_and_crowding(objs, cvs)
    feasible = np.where(cvs <= 1e-9)[0]
    if len(feasible):
        feasible_objs = objs[feasible].astype(float)
        best_cost = float(np.min(feasible_objs[:, 0]))
        best_service = float(np.max(-feasible_objs[:, 1]))
        service_span = float(np.max(-feasible_objs[:, 1]) - np.min(-feasible_objs[:, 1])) if len(feasible_objs) > 1 else 0.0
    else:
        best_cost = float("inf")
        best_service = 0.0
        service_span = 0.0
    unique_count = int(len(np.unique(np.round(norm_front, 8), axis=0))) if len(norm_front) else 0
    cost_denom = max(float(f_max[0] - f_min[0]), 1e-12)
    service_loss_min = -float(f_max[1])
    service_loss_max = -float(f_min[1])
    service_denom = max(service_loss_max - service_loss_min, 1e-12)
    best_cost_norm = 1.0 if not np.isfinite(best_cost) else float((best_cost - f_min[0]) / cost_denom)
    best_service_norm = float((best_service - service_loss_min) / service_denom)
    return {
        "hv": _hypervolume_2d_min(norm_front),
        "feasible_rate": float(np.mean(cvs <= 1e-9)) if len(cvs) else 0.0,
        "nd_ratio": float(len(fronts[0]) / max(1, len(objs))) if fronts else 0.0,
        "diversity": _front_diversity(norm_front),
        "cv_mean": float(np.mean(np.maximum(cvs, 0.0))) if len(cvs) else 0.0,
        "unique_ratio": float(unique_count / max(1, len(objs))),
        "best_cost_norm": float(np.clip(best_cost_norm, 0.0, 1.5)),
        "best_service_norm": float(np.clip(best_service_norm, 0.0, 1.5)),
        "service_span": service_span,
    }


def _qlearning_reward(before_objs: np.ndarray, before_cvs: np.ndarray, after_objs: np.ndarray, after_cvs: np.ndarray) -> tuple[float, dict, dict]:
    all_feasible = np.concatenate([before_cvs <= 1e-9, after_cvs <= 1e-9])
    all_objs = np.vstack([before_objs, after_objs]).astype(float)
    if np.any(all_feasible):
        reference_objs = all_objs[all_feasible]
    else:
        reference_objs = all_objs
    f_min = np.min(reference_objs, axis=0)
    f_max = np.max(reference_objs, axis=0)
    before = _population_mo_features(before_objs, before_cvs, f_min, f_max)
    after = _population_mo_features(after_objs, after_cvs, f_min, f_max)
    delta_hv = after["hv"] - before["hv"]
    delta_unique = after["unique_ratio"] - before["unique_ratio"]
    delta_cost = before["best_cost_norm"] - after["best_cost_norm"]
    delta_service = after["best_service_norm"] - before["best_service_norm"]
    reward = (
        8.0 * delta_hv
        + 1.8 * (after["feasible_rate"] - before["feasible_rate"])
        + 0.8 * (after["nd_ratio"] - before["nd_ratio"])
        + 0.8 * (after["diversity"] - before["diversity"])
        + 1.2 * delta_unique
        + 1.1 * delta_cost
        + 1.1 * delta_service
        - 2.5 * (after["cv_mean"] - before["cv_mean"])
    )
    # Strong negative feedback for the two observed failure modes: HV/front collapse
    # and worsening of both cost and service extrema.
    if delta_hv < -1e-9:
        reward -= 3.5 * abs(delta_hv)
    if delta_unique < -1e-9:
        reward -= 1.0 * abs(delta_unique)
    if delta_cost < -1e-9 and delta_service < -1e-9:
        reward -= 1.0 * (abs(delta_cost) + abs(delta_service))
    return float(np.clip(reward, -5.0, 5.0)), before, after


def _solution_accept(old_obj: np.ndarray, old_cv: float, new_obj: np.ndarray, new_cv: float, cost_scale: float) -> bool:
    if dominates(new_obj, old_obj, new_cv, old_cv):
        return True
    if dominates(old_obj, new_obj, old_cv, new_cv):
        return False
    if new_cv <= old_cv + 1e-9:
        old_merit = -old_obj[0] / (cost_scale + 1e-12) - old_obj[1]
        new_merit = -new_obj[0] / (cost_scale + 1e-12) - new_obj[1]
        return bool(new_merit > old_merit + 1e-12)
    return False


def _vns_neighbor(dec: np.ndarray, inst: MetroLogisticsInstance, neighborhood: str) -> np.ndarray:
    dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    potential = _service_potential(inst)
    if neighborhood == "N1_open_service_station":
        candidates = _line_spacing_model_open_candidates(x, inst)
        if len(candidates) > 0:
            add = int(candidates[int(np.argmax(potential[candidates]))])
            x[add] = 1
            dec[:inst.K] = x
            dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
            dec = _assign_direct_to_metro(dec, inst, probability=0.80, only_direct=True)
    elif neighborhood == "N2_close_low_load_station":
        open_sites = np.where(x == 1)[0]
        if len(open_sites) > max(1, inst.min_open):
            use_count = np.array([np.sum(assignment == k + 1) for k in open_sites], dtype=float)
            score = inst.fixed_cost[open_sites] / (1.0 + use_count)
            close = int(open_sites[int(np.argmax(score))])
            x[close] = 0
            assignment[assignment == close + 1] = 0
    elif neighborhood == "N3_swap_station":
        open_sites = np.where(x == 1)[0]
        closed = _line_spacing_model_open_candidates(x, inst)
        if len(open_sites) > 0 and len(closed) > 0:
            remove_score = potential[open_sites] / (inst.fixed_cost[open_sites] + 1e-12)
            remove = int(open_sites[int(np.argmin(remove_score))])
            x[remove] = 0
            closed_after = _line_spacing_model_open_candidates(x, inst)
            if len(closed_after) > 0:
                add = int(closed_after[int(np.argmax(potential[closed_after]))])
                x[add] = 1
                assignment[assignment == remove + 1] = 0
    elif neighborhood == "N4_reassign_flow":
        dec[:inst.K] = x
        dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
        dec = _assign_direct_to_metro(dec, inst, probability=0.90, only_direct=False)
        return repair_solution(dec, inst)
    elif neighborhood == "N5_service_reinforce":
        # High-coverage neighborhood: first open the station with the largest
        # uncovered 4 km service potential, then aggressively convert feasible
        # road flows to metro.  This is intended for Q4/Q5/Q6 tuning groups
        # where the goal is to strengthen the high-service end of the Pareto front.
        candidates = _line_spacing_model_open_candidates(x, inst)
        if len(candidates) > 0:
            open_sites = np.where(x == 1)[0]
            reachable = inst.d_kj_road <= inst.D_k_max[:, None]
            currently_covered = reachable[open_sites].any(axis=0) if len(open_sites) else np.zeros(inst.J, dtype=bool)
            uncovered_volume = reachable[candidates][:, ~currently_covered] @ inst.Q_j[~currently_covered]
            if len(uncovered_volume) and float(np.max(uncovered_volume)) > 0:
                add = int(candidates[int(np.argmax(uncovered_volume))])
            else:
                add = int(candidates[int(np.argmax(potential[candidates]))])
            x[add] = 1
        dec[:inst.K] = x
        dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
        dec = _assign_direct_to_metro(dec, inst, probability=0.95, only_direct=False)
        return repair_solution(dec, inst)
    dec[:inst.K] = x
    dec[_assignment_start(inst):_assignment_end(inst)] = assignment.ravel()
    return repair_solution(dec, inst)


def _vns_local_search(dec: np.ndarray, inst: MetroLogisticsInstance, trials_per_neighborhood: int = 3, use_repair=True, max_neighborhood: int | None = None) -> tuple[np.ndarray, dict]:
    neighborhoods = [
        "N1_open_service_station",
        "N2_close_low_load_station",
        "N3_swap_station",
        "N4_reassign_flow",
        "N5_service_reinforce",
    ]
    if max_neighborhood is not None:
        neighborhoods = neighborhoods[:max(1, min(int(max_neighborhood), len(neighborhoods)))]
    current = repair_solution(dec, inst) if use_repair else normalize_solution(dec, inst)
    current_obj, current_cv = _evaluate_objective_cv_fast(current, inst, use_repair=use_repair)
    cost_scale = max(abs(float(current_obj[0])), 1.0)
    accepted = 0
    tried = 0
    for neighborhood in neighborhoods:
        for _ in range(max(1, int(trials_per_neighborhood))):
            tried += 1
            candidate = _vns_neighbor(current, inst, neighborhood)
            cand_obj, cand_cv = _evaluate_objective_cv_fast(candidate, inst, use_repair=use_repair)
            if _solution_accept(current_obj, current_cv, cand_obj, cand_cv, cost_scale):
                current, current_obj, current_cv = candidate, cand_obj, cand_cv
                accepted += 1
                break
    return current, {"vns_trials": int(tried), "vns_accepted": int(accepted)}


def _select_vns_elites(pop: np.ndarray, objs: np.ndarray, cvs: np.ndarray, elite_count: int) -> np.ndarray:
    fronts, _, _ = rank_and_crowding(objs, cvs)
    if not fronts:
        return pop[:0]
    front0 = np.array(fronts[0], dtype=int)
    feasible = np.array([idx for idx in front0 if cvs[idx] <= 1e-9], dtype=int)
    candidates = feasible if len(feasible) else front0
    if len(candidates) <= elite_count:
        return pop[candidates]
    cd = crowding_distance(objs, candidates.tolist())
    order = np.argsort(-cd)
    return pop[candidates[order[:elite_count]]]


def _select_vns_elites_mixed(
    pop: np.ndarray,
    objs: np.ndarray,
    cvs: np.ndarray,
    elite_count: int,
    boundary_keep_rate: float = 0.20,
) -> np.ndarray:
    """Select VNS candidates from cost boundary, service boundary and sparse points.

    This is used by Q8/Q10/Q11.  The aim is not merely to improve a few best
    solutions, but to keep the Pareto front spread more uniformly and increase
    the number of unique nondominated solutions.
    """
    fronts, _, _ = rank_and_crowding(objs, cvs)
    if not fronts:
        return pop[:0]
    front0 = np.array(fronts[0], dtype=int)
    feasible = np.array([idx for idx in front0 if cvs[idx] <= 1e-9], dtype=int)
    candidates = feasible if len(feasible) else front0
    elite_count = max(1, int(elite_count))
    if len(candidates) <= elite_count:
        return pop[candidates]

    boundary_keep_rate = max(0.0, min(float(boundary_keep_rate), 0.50))
    n_boundary = max(2, int(round(elite_count * boundary_keep_rate)))
    n_cost = max(1, n_boundary // 2)
    n_service = max(1, n_boundary - n_cost)

    cand_objs = objs[candidates]
    cost_part = candidates[np.argsort(cand_objs[:, 0])[:n_cost]]
    service_part = candidates[np.argsort(cand_objs[:, 1])[:n_service]]

    chosen = []
    for idx in list(cost_part) + list(service_part):
        if int(idx) not in chosen:
            chosen.append(int(idx))

    remaining = np.array([idx for idx in candidates if int(idx) not in set(chosen)], dtype=int)
    if len(remaining) > 0:
        cd = crowding_distance(objs, remaining.tolist())
        order = np.argsort(-cd)
        for idx in remaining[order]:
            if len(chosen) >= elite_count:
                break
            chosen.append(int(idx))

    return pop[np.array(chosen[:elite_count], dtype=int)]


def _archive_hv_indicator(objs: np.ndarray, cvs: np.ndarray) -> float:
    """Fast normalized HV indicator for adaptive-VNS stagnation checks."""
    if len(objs) == 0:
        return 0.0
    feasible = cvs <= 1e-9
    ref_objs = objs[feasible] if np.any(feasible) else objs
    if len(ref_objs) == 0:
        return 0.0
    f_min = np.min(ref_objs, axis=0)
    f_max = np.max(ref_objs, axis=0)
    norm_front = _normalize_points_for_metrics(objs, cvs, f_min, f_max)
    return _hypervolume_2d_min(norm_front)


def _should_run_vns(
    gen: int,
    max_gen: int,
    hv_history: list[float],
    last_vns_gen: int,
    vns_mode: str,
    vns_interval: int | None,
    stagnation_window: int,
    hv_improvement_threshold: float,
    min_vns_gap: int,
) -> bool:
    """Fixed or adaptive VNS trigger.

    Adaptive mode runs VNS only when the archive HV has stagnated over a recent
    window, reducing unnecessary local search and runtime.
    """
    mode = str(vns_mode or "fixed").lower()
    if mode == "fixed":
        return bool(vns_interval and int(vns_interval) > 0 and gen > 0 and gen % int(vns_interval) == 0)

    if mode != "adaptive":
        return False
    if gen <= max(1, int(stagnation_window)):
        return False
    if gen - int(last_vns_gen) < max(1, int(min_vns_gap)):
        return False
    if len(hv_history) < max(2, int(stagnation_window)):
        return False
    recent = hv_history[-int(stagnation_window):]
    hv_gain = float(max(recent) - min(recent))
    return hv_gain < float(hv_improvement_threshold)


def _run_standard_nsga2_branch(inst: MetroLogisticsInstance, pop_size: int, max_gen: int, pc: float, seed=None, use_repair=True):
    if seed is not None:
        previous_np_state = np.random.get_state()
        previous_random_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
    else:
        previous_np_state = None
        previous_random_state = None
    pop = initialize_population(inst, pop_size, use_repair=use_repair)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=use_repair)
    for _ in range(max_gen):
        fronts, rank, crowd = rank_and_crowding(objs, cvs)
        offspring = []
        while len(offspring) < pop_size:
            parent1 = tournament_select(pop, objs, cvs, rank, crowd)
            parent2 = tournament_select(pop, objs, cvs, rank, crowd)
            child1, child2 = crossover(parent1, parent2, inst, pc=pc) if use_repair else crossover_no_repair(parent1, parent2, inst, pc=pc)
            offspring.append(mutate(child1, inst) if use_repair else mutate_no_repair(child1, inst))
            if len(offspring) < pop_size:
                offspring.append(mutate(child2, inst) if use_repair else mutate_no_repair(child2, inst))
        offspring = np.array(offspring[:pop_size])
        offspring_objs, offspring_cvs, _ = evaluate_population(offspring, inst, use_repair=use_repair)
        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, offspring_objs])
        combined_cvs = np.concatenate([cvs, offspring_cvs])
        fronts, _, _ = rank_and_crowding(combined_objs, combined_cvs)
        pop, objs, cvs = _select_by_fronts(combined_pop, combined_objs, combined_cvs, fronts, pop_size)
    if previous_np_state is not None:
        np.random.set_state(previous_np_state)
        random.setstate(previous_random_state)
    return pop, objs, cvs


def _q_table_to_frame(q_table: dict, actions: list) -> pd.DataFrame:
    rows = []
    for state, values in q_table.items():
        for idx, value in enumerate(values):
            rows.append({"state": state, "action_id": idx, "action": actions[idx]["name"], "q_value": float(value)})
    return pd.DataFrame(rows).sort_values(["state", "q_value"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame(columns=["state", "action_id", "action", "q_value"])




def _resolve_qvns_phase_value(schedule, gen: int, max_gen: int, key: str, default_value):
    """Return a phase-specific QVNS parameter value.

    ``schedule`` is a list of dictionaries such as::

        [{"until_ratio": 1/3, "vns_interval": 10},
         {"until_ratio": 2/3, "vns_interval": 5},
         {"until_ratio": 1.0, "vns_interval": 3}]

    The helper keeps the formal solver backward-compatible: if no schedule is
    passed, all parameters are constant.
    """
    if not schedule:
        return default_value
    progress = gen / max(1, max_gen)
    for phase in schedule:
        until = float(phase.get("until_ratio", phase.get("until", 1.0)))
        if progress <= until + 1e-12:
            return phase.get(key, default_value)
    return schedule[-1].get(key, default_value)

def _run_qvns_nsga2_core(inst: MetroLogisticsInstance, pop_size=100, max_gen=300, verbose=True, seed=None,
                        pc=0.9, alpha=0.15, gamma=0.88, epsilon=0.30,
                        epsilon_min=0.010, epsilon_decay=None, archive_size=None,
                        preserve_baseline=False, use_repair=True,
                        vns_interval=0, vns_elite_rate=0.12, vns_trials_per_neighborhood=2, vns_max_elites=None,
                        vns_max_neighborhood=3, qvns_phase_schedule=None,
                        archive_feedback_interval=8, archive_feedback_rate=0.15,
                        q_action_apply_rate=0.25, q_learning_enabled=True,
                        q_action_profile="safe", safeguard=True, safeguard_threshold=-0.15,
                        vns_mode="adaptive", stagnation_window=10, hv_improvement_threshold=0.0025,
                        min_vns_gap=5, archive_boundary_keep_rate=0.20,
                        vns_candidate_strategy="mixed", **_ignored):
    """Safeguarded QVNS-NSGA-II.

    The default version treats VNS as the exploitation core and makes
    Q-learning a conservative hyper-heuristic.  A Q action only affects part of
    the offspring, receives archive-level multi-objective feedback, and can be
    rolled back when it clearly degrades the archive.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    epsilon_decay = epsilon_decay if epsilon_decay is not None else _epsilon_decay_for_target(0.30, 0.010, max_gen)
    start_time = time.time()
    epsilon_initial = float(epsilon)
    actions = _qlearning_nsga2_actions(profile=q_action_profile)
    initial_q = np.array([action.get("prior", 0.0) for action in actions], dtype=float)
    q_table = {}
    q_history = []
    vns_history = []
    hv_history: list[float] = []
    actual_vns_calls = 0
    last_vns_gen = -10**9
    archive_size = int(archive_size or max(pop_size, int(round(2.5 * pop_size))))
    archive_boundary_keep_rate = max(0.0, min(float(archive_boundary_keep_rate), 0.50))

    pop = initialize_population(inst, pop_size, use_repair=use_repair)
    objs, cvs, _ = evaluate_population(pop, inst, use_repair=use_repair)
    archive_pop, archive_objs, archive_cvs = _update_archive(
        pop, objs, cvs, archive_size, boundary_keep_rate=archive_boundary_keep_rate
    )
    history = []

    for gen in range(max_gen + 1):
        fronts, rank, crowd = rank_and_crowding(objs, cvs)
        front0 = fronts[0]
        feasible = np.where(cvs <= 1e-9)[0]
        view = feasible if len(feasible) else np.array(front0, dtype=int)
        best_cost_idx = int(view[np.argmin(objs[view, 0])]) if len(view) else int(np.argmin(cvs))
        _, best_cost_cv, best_cost_parts = evaluate_solution(pop[best_cost_idx], inst, use_repair=use_repair, collect_parts=True)
        archive_fronts, _ = fast_non_dominated_sort(archive_objs, archive_cvs)
        archive_front0 = np.array(archive_fronts[0] if archive_fronts else np.arange(len(archive_objs)), dtype=int)
        archive_feasible_front = archive_front0[archive_cvs[archive_front0] <= 1e-9]
        archive_pareto_idx = archive_feasible_front if len(archive_feasible_front) else archive_front0
        archive_pareto_costs = archive_objs[archive_pareto_idx, 0] if len(archive_pareto_idx) else np.array([], dtype=float)
        archive_pareto_services = -archive_objs[archive_pareto_idx, 1] if len(archive_pareto_idx) else np.array([], dtype=float)
        current_front = np.array(front0, dtype=int)
        current_front_feasible = current_front[cvs[current_front] <= 1e-9]
        current_pareto_idx = current_front_feasible if len(current_front_feasible) else current_front
        current_pareto_costs = objs[current_pareto_idx, 0] if len(current_pareto_idx) else np.array([], dtype=float)
        current_pareto_services = -objs[current_pareto_idx, 1] if len(current_pareto_idx) else np.array([], dtype=float)
        history.append({
            "gen": gen,
            "best_cost": float(objs[best_cost_idx, 0]) if len(view) else float("nan"),
            "best_cost_carbon_cost": float(best_cost_parts.get("carbon_cost", float("nan"))),
            "best_cost_cv": float(best_cost_cv),
            "best_service_ratio": float(np.max(-objs[view, 1])) if len(view) else float("nan"),
            "pareto_mean_total_cost": float(np.mean(archive_pareto_costs)) if len(archive_pareto_costs) else float("nan"),
            "pareto_mean_service_ratio": float(np.mean(archive_pareto_services)) if len(archive_pareto_services) else float("nan"),
            "population_pareto_mean_total_cost": float(np.mean(current_pareto_costs)) if len(current_pareto_costs) else float("nan"),
            "population_pareto_mean_service_ratio": float(np.mean(current_pareto_services)) if len(current_pareto_services) else float("nan"),
            "pareto_size": int(len(archive_pareto_idx)),
            "population_pareto_size": len(front0),
            "feasible_count": int(np.sum(cvs <= 1e-9)),
            "min_cv": float(np.min(cvs)),
            "epsilon": float(epsilon),
            "archive_size": int(len(archive_pop)),
        })
        hv_history.append(_archive_hv_indicator(archive_objs, archive_cvs))
        if verbose and gen % 20 == 0:
            best_service = float(np.max(-objs[view, 1])) if len(view) else float("nan")
            print(f"[QVNS-NSGA-II] Gen {gen:4d}: Pareto={len(front0):3d}, feasible={np.sum(cvs <= 1e-9):3d}, minCV={np.min(cvs):.4f}, Z2={best_service:.4f}")
        if gen == max_gen:
            break

        state = _qlearning_state(gen, max_gen, objs, cvs)
        if state not in q_table:
            q_table[state] = initial_q.copy()
        if q_learning_enabled:
            action_idx, policy = _qlearning_choose_action(q_table[state], epsilon)
        else:
            action_idx, policy = 1, "disabled"
        action = actions[action_idx]
        base_action = actions[1] if len(actions) > 1 else action
        before_pop = pop.copy()
        before_objs = objs.copy()
        before_cvs = cvs.copy()
        archive_before_pop = archive_pop.copy()
        archive_before_objs = archive_objs.copy()
        archive_before_cvs = archive_cvs.copy()

        offspring = []
        while len(offspring) < pop_size:
            parent1 = tournament_select(pop, objs, cvs, rank, crowd)
            parent2 = tournament_select(pop, objs, cvs, rank, crowd)
            pc_factor = float(action.get("pc_factor", 1.0)) if q_learning_enabled else 1.0
            child1, child2 = crossover_no_repair(parent1, parent2, inst, pc=pc * pc_factor)
            child1_action = action if (q_learning_enabled and np.random.rand() < float(q_action_apply_rate)) else base_action
            offspring.append(_qlearning_mutate(child1, inst, child1_action, use_repair=use_repair))
            if len(offspring) < pop_size:
                child2_action = action if (q_learning_enabled and np.random.rand() < float(q_action_apply_rate)) else base_action
                offspring.append(_qlearning_mutate(child2, inst, child2_action, use_repair=use_repair))
        offspring = np.array(offspring[:pop_size])
        offspring_objs, offspring_cvs, _ = evaluate_population(offspring, inst, use_repair=use_repair)

        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, offspring_objs])
        combined_cvs = np.concatenate([cvs, offspring_cvs])

        # VNS: every few generations, locally improve archive/Pareto elites.
        # QVNS tuning groups can use piecewise schedules, e.g. Q6 uses weaker
        # early exploration and stronger late exploitation.
        raw_interval = _resolve_qvns_phase_value(qvns_phase_schedule, gen, max_gen, "vns_interval", vns_interval)
        current_vns_interval = None if raw_interval is None else int(raw_interval)
        current_vns_elite_rate = float(_resolve_qvns_phase_value(qvns_phase_schedule, gen, max_gen, "vns_elite_rate", vns_elite_rate))
        current_vns_trials = int(_resolve_qvns_phase_value(qvns_phase_schedule, gen, max_gen, "vns_trials_per_neighborhood", vns_trials_per_neighborhood))
        current_vns_max_neighborhood = int(_resolve_qvns_phase_value(qvns_phase_schedule, gen, max_gen, "vns_max_neighborhood", vns_max_neighborhood))
        run_vns_now = _should_run_vns(
            gen=gen, max_gen=max_gen, hv_history=hv_history, last_vns_gen=last_vns_gen,
            vns_mode=vns_mode, vns_interval=current_vns_interval,
            stagnation_window=stagnation_window, hv_improvement_threshold=hv_improvement_threshold,
            min_vns_gap=min_vns_gap,
        )
        if run_vns_now:
            elite_count = max(1, int(round(pop_size * current_vns_elite_rate)))
            if vns_max_elites is not None:
                elite_count = min(elite_count, max(1, int(vns_max_elites)))
            if str(vns_candidate_strategy).lower() == "mixed" or archive_boundary_keep_rate > 0:
                elites = _select_vns_elites_mixed(archive_pop, archive_objs, archive_cvs, elite_count, boundary_keep_rate=archive_boundary_keep_rate)
            else:
                elites = _select_vns_elites(archive_pop, archive_objs, archive_cvs, elite_count)
            vns_candidates = []
            accepted_total = 0
            tried_total = 0
            for elite in elites:
                improved, stat = _vns_local_search(
                    elite,
                    inst,
                    trials_per_neighborhood=current_vns_trials,
                    use_repair=use_repair,
                    max_neighborhood=current_vns_max_neighborhood,
                )
                vns_candidates.append(improved)
                accepted_total += stat["vns_accepted"]
                tried_total += stat["vns_trials"]
            if vns_candidates:
                vns_candidates = np.array(vns_candidates)
                vns_objs, vns_cvs, _ = evaluate_population(vns_candidates, inst, use_repair=use_repair)
                combined_pop = np.vstack([combined_pop, vns_candidates])
                combined_objs = np.vstack([combined_objs, vns_objs])
                combined_cvs = np.concatenate([combined_cvs, vns_cvs])
                actual_vns_calls += 1
                last_vns_gen = gen
                vns_history.append({
                    "gen": gen,
                    "elite_count": int(len(elites)),
                    "candidate_count": int(len(vns_candidates)),
                    "vns_trials": int(tried_total),
                    "vns_accepted": int(accepted_total),
                    "vns_interval": (-1 if current_vns_interval is None else int(current_vns_interval)),
                    "vns_mode": str(vns_mode),
                    "vns_elite_rate": float(current_vns_elite_rate),
                    "vns_trials_per_neighborhood": int(current_vns_trials),
                    "vns_max_neighborhood": int(current_vns_max_neighborhood),
                    "archive_boundary_keep_rate": float(archive_boundary_keep_rate),
                    "stagnation_window": int(stagnation_window),
                    "hv_improvement_threshold": float(hv_improvement_threshold),
                })

        candidate_archive_pop, candidate_archive_objs, candidate_archive_cvs = _update_archive(
            np.vstack([archive_pop, combined_pop]),
            np.vstack([archive_objs, combined_objs]),
            np.concatenate([archive_cvs, combined_cvs]),
            archive_size,
            boundary_keep_rate=archive_boundary_keep_rate,
        )
        fronts, _, _ = rank_and_crowding(combined_objs, combined_cvs)
        candidate_pop, candidate_objs, candidate_cvs = _select_by_fronts(combined_pop, combined_objs, combined_cvs, fronts, pop_size)

        # Periodically inject archive elites back into the candidate population.
        if archive_feedback_interval and archive_feedback_interval > 0 and gen > 0 and gen % int(archive_feedback_interval) == 0:
            n_feedback = max(1, int(round(pop_size * float(archive_feedback_rate))))
            if str(vns_candidate_strategy).lower() == "mixed" or archive_boundary_keep_rate > 0:
                feedback = _select_vns_elites_mixed(candidate_archive_pop, candidate_archive_objs, candidate_archive_cvs, n_feedback, boundary_keep_rate=archive_boundary_keep_rate)
            else:
                feedback = _select_vns_elites(candidate_archive_pop, candidate_archive_objs, candidate_archive_cvs, n_feedback)
            if len(feedback) > 0:
                feedback_objs, feedback_cvs, _ = evaluate_population(feedback, inst, use_repair=use_repair)
                merged_pop = np.vstack([candidate_pop, feedback])
                merged_objs = np.vstack([candidate_objs, feedback_objs])
                merged_cvs = np.concatenate([candidate_cvs, feedback_cvs])
                fronts, _, _ = rank_and_crowding(merged_objs, merged_cvs)
                candidate_pop, candidate_objs, candidate_cvs = _select_by_fronts(merged_pop, merged_objs, merged_cvs, fronts, pop_size)

        reward, before_feat, after_feat = _qlearning_reward(
            archive_before_objs, archive_before_cvs, candidate_archive_objs, candidate_archive_cvs
        )
        rejected = bool(q_learning_enabled and safeguard and reward < float(safeguard_threshold))
        if rejected:
            pop, objs, cvs = before_pop, before_objs, before_cvs
            archive_pop, archive_objs, archive_cvs = archive_before_pop, archive_before_objs, archive_before_cvs
        else:
            pop, objs, cvs = candidate_pop, candidate_objs, candidate_cvs
            archive_pop, archive_objs, archive_cvs = candidate_archive_pop, candidate_archive_objs, candidate_archive_cvs

        next_state = _qlearning_state(gen + 1, max_gen, objs, cvs)
        if next_state not in q_table:
            q_table[next_state] = initial_q.copy()
        if q_learning_enabled:
            q_table[state][action_idx] += alpha * (reward + gamma * float(np.max(q_table[next_state])) - q_table[state][action_idx])
        q_history.append({
            "gen": gen,
            "state": state,
            "action_id": action_idx,
            "action": action["name"],
            "policy": policy,
            "epsilon": float(epsilon if q_learning_enabled else 0.0),
            "q_action_apply_rate": float(q_action_apply_rate if q_learning_enabled else 0.0),
            "reward": float(reward),
            "rejected_by_safeguard": rejected,
            "hv_before": float(before_feat["hv"]),
            "hv_after": float(after_feat["hv"]),
            "feasible_rate_before": float(before_feat["feasible_rate"]),
            "feasible_rate_after": float(after_feat["feasible_rate"]),
            "nd_ratio_before": float(before_feat["nd_ratio"]),
            "nd_ratio_after": float(after_feat["nd_ratio"]),
            "diversity_before": float(before_feat["diversity"]),
            "diversity_after": float(after_feat["diversity"]),
            "unique_ratio_before": float(before_feat.get("unique_ratio", 0.0)),
            "unique_ratio_after": float(after_feat.get("unique_ratio", 0.0)),
            "best_cost_norm_before": float(before_feat.get("best_cost_norm", 0.0)),
            "best_cost_norm_after": float(after_feat.get("best_cost_norm", 0.0)),
            "best_service_norm_before": float(before_feat.get("best_service_norm", 0.0)),
            "best_service_norm_after": float(after_feat.get("best_service_norm", 0.0)),
            "cv_mean_before": float(before_feat["cv_mean"]),
            "cv_mean_after": float(after_feat["cv_mean"]),
            "q_value": float(q_table[state][action_idx]),
        })
        if q_learning_enabled:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    final_pop, final_objs, final_cvs = archive_pop, archive_objs, archive_cvs
    baseline_pop = baseline_objs = baseline_cvs = None
    if preserve_baseline:
        branch_seed = None if seed is None else int(seed) + 7919
        baseline_pop, baseline_objs, baseline_cvs = _run_standard_nsga2_branch(
            inst, pop_size=pop_size, max_gen=max_gen, pc=pc, seed=branch_seed, use_repair=use_repair
        )
        final_pop = np.vstack([archive_pop, baseline_pop])
        final_objs = np.vstack([archive_objs, baseline_objs])
        final_cvs = np.concatenate([archive_cvs, baseline_cvs])

    fronts, _, _ = rank_and_crowding(final_objs, final_cvs)
    feasible_front = [idx for idx in fronts[0] if final_cvs[idx] <= 1e-9]
    pareto_idx = np.array(feasible_front if feasible_front else fronts[0], dtype=int)
    result = {
        "algorithm": "QVNS-NSGA-II",
        "pop": final_pop,
        "objs": final_objs,
        "cvs": final_cvs,
        "pareto_dec": final_pop[pareto_idx],
        "pareto_objs": final_objs[pareto_idx],
        "pareto_cvs": final_cvs[pareto_idx],
        "history": pd.DataFrame(history),
        "runtime": time.time() - start_time,
        "q_table": _q_table_to_frame(q_table, actions),
        "action_history": pd.DataFrame(q_history),
        "q_history": q_history,
        "vns_history": pd.DataFrame(vns_history),
        "actions": pd.DataFrame(actions),
        "baseline_preservation": bool(preserve_baseline),
        "archive_size": int(archive_size),
        "actual_vns_calls": int(actual_vns_calls),
        "archive_final_size": int(len(archive_pop)),
        "archive_unique_solutions": int(len(np.unique(np.round(archive_pop, 8), axis=0))) if len(archive_pop) else 0,
        "best_params": {
            "alpha_q": float(alpha),
            "gamma_q": float(gamma),
            "epsilon_initial": float(epsilon_initial),
            "epsilon_decay": float(epsilon_decay),
            "epsilon_min": float(epsilon_min),
            "archive_size": int(archive_size),
            "vns_interval": int(vns_interval),
            "vns_elite_rate": float(vns_elite_rate),
            "vns_trials_per_neighborhood": int(vns_trials_per_neighborhood),
            "vns_max_neighborhood": int(vns_max_neighborhood),
            "vns_mode": str(vns_mode),
            "stagnation_window": int(stagnation_window),
            "hv_improvement_threshold": float(hv_improvement_threshold),
            "min_vns_gap": int(min_vns_gap),
            "archive_boundary_keep_rate": float(archive_boundary_keep_rate),
            "vns_candidate_strategy": str(vns_candidate_strategy),
            "actual_vns_calls": int(actual_vns_calls),
            "qvns_phase_schedule": ("" if not qvns_phase_schedule else str(qvns_phase_schedule)),
            "vns_max_elites": (None if vns_max_elites is None else int(vns_max_elites)),
            "archive_feedback_interval": int(archive_feedback_interval),
            "archive_feedback_rate": float(archive_feedback_rate),
            "q_learning_enabled": bool(q_learning_enabled),
            "q_action_profile": str(q_action_profile),
            "q_action_apply_rate": float(q_action_apply_rate),
            "safeguard": bool(safeguard),
            "safeguard_threshold": float(safeguard_threshold),
        },
    }
    if baseline_pop is not None:
        result["baseline_pop"] = baseline_pop
        result["baseline_objs"] = baseline_objs
        result["baseline_cvs"] = baseline_cvs
    return result

def run_qvns_nsga2(*args, **kwargs):
    """Public thesis-facing entry point for the tuned QVNS-NSGA-II solver."""
    return _run_qvns_nsga2_core(*args, **kwargs)



def choose_compromise(pareto_objs: np.ndarray) -> int:
    """TOPSIS compromise using only the two model objectives Z1 and -Z2."""
    f = np.column_stack([pareto_objs[:, 0], pareto_objs[:, 1]]).astype(float)
    f_min = np.min(f, axis=0)
    f_max = np.max(f, axis=0)
    denom = f_max - f_min
    norm = np.divide(f - f_min, denom, out=np.zeros_like(f), where=denom > 1e-12)
    weights = np.array([0.5, 0.5], dtype=float)
    ideal = np.zeros(2, dtype=float)
    nadir = np.ones(2, dtype=float)
    d_pos = np.sqrt(np.sum(weights * (norm - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum(weights * (norm - nadir) ** 2, axis=1))
    cc = d_neg / (d_pos + d_neg + 1e-12)
    return int(np.argmax(cc))


def solution_to_tables(dec: np.ndarray, inst: MetroLogisticsInstance, use_repair: bool = True):
    if use_repair:
        dec = repair_solution(dec, inst)
    else:
        dec = normalize_solution(dec, inst)
    x, assignment = decode_solution(dec, inst)
    rows = []
    station_flow = np.zeros(inst.K, dtype=float)
    line_items = _empty_line_items(inst)
    source_share = inst.source_share / (np.sum(inst.source_share) + 1e-12)
    source_names = [str(v) for v in getattr(inst.logistics_df, "index", [])]
    if len(source_names) < inst.I:
        source_names = [f"L{i + 1}" for i in range(inst.I)]
    for i in range(inst.I):
        source_name = source_names[i] if i < len(source_names) else f"L{i + 1}"
        for j, row in inst.demand_df.iterrows():
            demand = float(source_share[i] * inst.Q_j[j])
            if assignment[i, j] == 0:
                rows.append({
                    "物流园区": source_name,
                    "需求点": row["站点名称"],
                    "经度": row["经度_GCJ02"],
                    "纬度": row["纬度_GCJ02"],
                    "服务方式": "公路运输",
                    "服务路径类型": "D",
                    "服务变量": f"o_{{{i + 1},{j + 1}}}=1",
                    "候选转运点索引": 0,
                    "算法编码值": 0,
                    "中转站": "-",
                    "需求量_件日": demand,
                    "总需求量_件日": inst.Q_j[j],
                    "末端距离_km": 0.0,
                    "路径经过线路": "-",
                })
            else:
                k = assignment[i, j] - 1
                station_flow[k] += demand
                path_lines = _path_lines(inst, i, k)
                for line in path_lines:
                    line_items.setdefault(line, []).append(float(demand))
                rows.append({
                    "物流园区": source_name,
                    "需求点": row["站点名称"],
                    "经度": row["经度_GCJ02"],
                    "纬度": row["纬度_GCJ02"],
                    "服务方式": "地铁联运",
                    "服务路径类型": "M",
                    "服务变量": f"z_{{{i + 1},{j + 1},{k + 1}}}=1",
                    "候选转运点索引": int(k + 1),
                    "算法编码值": int(assignment[i, j]),
                    "中转站": inst.candidate_df.iloc[k]["地铁站名称"],
                    "需求量_件日": demand,
                    "总需求量_件日": inst.Q_j[j],
                    "末端距离_km": inst.d_kj_road[k, j],
                    "路径经过线路": "、".join(path_lines),
                })
    assign_df = pd.DataFrame(rows)
    station_df = inst.candidate_df[["地铁站名称", "经度", "纬度", "线路数"]].copy()
    station_df["站点类型集合"] = inst.station_type_labels
    station_df["固定运营成本_元日"] = inst.fixed_cost
    station_df["是否启用"] = x
    station_df["分配量_件日"] = station_flow
    station_df["容量_件日"] = inst.station_capacity
    station_df["最低负载量_件日"] = inst.min_load_rate * inst.station_capacity
    station_df["最高可用容量_件日"] = inst.max_load_rate * inst.station_capacity
    station_df["负载率"] = station_flow / inst.station_capacity
    station_df["约束口径"] = "no_budget_range"

    packing = _evaluate_train_packing(line_items, inst)
    line_rows = []
    for line in (inst.line_names or sorted(line_items)):
        capacity = float((inst.line_capacity or {}).get(line, 0.0))
        loads = packing["trip_loads"].get(line, [0.0 for _ in range(inst.available_train_trips)])
        for r, load in enumerate(loads, start=1):
            line_rows.append({
                "线路": line,
                "班次": r,
                "单趟列车容量_件": capacity,
                "班次装载量_件": float(load),
                "班次负载率": float(load) / capacity if capacity > 0 else 0.0,
                "是否启用": int(float(load) > 1e-9),
                "班次组织成本_元": float((inst.train_organization_cost or {}).get(line, 0.0)) if load > 1e-9 else 0.0,
            })
        violation = float(packing["violation_pieces"].get(line, 0.0))
        if violation > 1e-9:
            line_rows.append({
                "线路": line,
                "班次": "未装入",
                "单趟列车容量_件": capacity,
                "班次装载量_件": violation,
                "班次负载率": np.nan,
                "是否启用": 0,
                "班次组织成本_元": 0.0,
            })
    line_df = pd.DataFrame(line_rows)
    return assign_df, station_df, line_df


def _save_figure_local(fig, output_dir: str, base_name: str):
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def _load_local_or_desktop_geojson(filename: str):
    """Load package-local GeoJSON first; use old desktop paths only as fallbacks.

    The release package is intended to be reproducible on non-Windows machines,
    so bundled files under data/ must take precedence over a developer desktop.
    """
    local_candidates = [
        DATA_DIR / filename,
        PROJECT_ROOT / filename,
        Path.cwd() / filename,
        Path(base.BASE_DIR) / filename,
    ]
    for path in local_candidates:
        if path.exists():
            found_path = str(path)
            return base.load_geojson(found_path), found_path

    # Legacy developer-machine fallbacks retained only for ad-hoc local runs.
    desktop_roots = [
        Path(r"C:\Users\zhang\Desktop\论文相关\精准数据\武侯区地图"),
        Path(r"C:\Users\zhang\Desktop\论文相关\模型构建\wuhou_nsga2_package_v4"),
        Path(r"C:\Users\zhang\Desktop"),
    ]
    for desktop_root in desktop_roots:
        if not desktop_root.exists():
            continue
        matches = [p for p in desktop_root.rglob(filename) if p.is_file()]
        matches.sort(key=lambda p: ("精准数据" not in str(p), "武侯区地图" not in str(p), len(str(p))))
        if matches:
            found_path = str(matches[0])
            return base.load_geojson(found_path), found_path

    raise FileNotFoundError(f"GeoJSON file not found in package data or legacy desktop locations: {filename}")


def _station_name_key(name) -> str:
    return str(name).strip().replace(" ", "")


def _apply_precise_station_coordinates(station_df: pd.DataFrame, metro_points) -> pd.DataFrame:
    out = station_df.copy()
    if out.empty or not metro_points:
        return out
    point_df = base.extract_geojson_point_df(metro_points)
    if point_df.empty:
        return out
    precise_coord = {}
    for _, row in point_df.iterrows():
        precise_coord[_station_name_key(row["站点名称"])] = (float(row["经度"]), float(row["纬度"]))
    for idx, row in out.iterrows():
        key = _station_name_key(row["地铁站名称"])
        if key in precise_coord:
            out.at[idx, "经度"] = precise_coord[key][0]
            out.at[idx, "纬度"] = precise_coord[key][1]
    return out


def _add_draw_coordinates(df: pd.DataFrame, boundary_geojson, lon_col="经度", lat_col="纬度"):
    if df.empty:
        out = df.copy()
        out["绘图经度"] = []
        out["绘图纬度"] = []
        return out
    if boundary_geojson is None:
        out = df.copy()
        out["绘图经度"] = out[lon_col].astype(float)
        out["绘图纬度"] = out[lat_col].astype(float)
        return out
    out, _ = base.adjust_display_points_to_boundary(df, boundary_geojson, lon_col, lat_col)
    return out


def build_compromise_map_layers(assign_df: pd.DataFrame, load_df: pd.DataFrame, inst: MetroLogisticsInstance, boundary_geojson=None, metro_points=None):
    station_source = _apply_precise_station_coordinates(load_df, metro_points)
    station_source["_name_key"] = station_source["地铁站名称"].map(_station_name_key)
    selected = station_source[station_source["是否启用"] == 1].copy()
    standby = station_source[station_source["是否启用"] != 1].copy()

    selected_plot = _add_draw_coordinates(selected, boundary_geojson, "经度", "纬度")
    standby_plot = _add_draw_coordinates(standby, boundary_geojson, "经度", "纬度")
    demand_plot = _add_draw_coordinates(assign_df.copy(), boundary_geojson, "经度", "纬度")

    candidate_keys = set(station_source["_name_key"])
    other_stations = base.extract_geojson_point_df(metro_points) if metro_points else pd.DataFrame(columns=["站点名称", "经度", "纬度"])
    if not other_stations.empty:
        other_stations["_name_key"] = other_stations["站点名称"].map(_station_name_key)
        other_stations = other_stations[~other_stations["_name_key"].isin(candidate_keys)].copy()
        if not other_stations.empty:
            candidate_lons = station_source["经度"].astype(float).to_numpy()
            candidate_lats = station_source["纬度"].astype(float).to_numpy()
            keep_other = []
            for _, row in other_stations.iterrows():
                distances = base.haversine_km(
                    float(row["经度"]),
                    float(row["纬度"]),
                    candidate_lons,
                    candidate_lats,
                )
                keep_other.append(float(np.min(distances)) > 0.12)
            other_stations = other_stations.loc[keep_other].copy()
    other_plot = _add_draw_coordinates(other_stations, boundary_geojson, "经度", "纬度")

    selected_coord = selected_plot.set_index("地铁站名称")[["绘图经度", "绘图纬度"]].to_dict("index")
    if "服务路径类型" in demand_plot.columns:
        metro_assign = demand_plot[demand_plot["服务路径类型"] == "M"].copy()
    else:
        metro_assign = demand_plot[demand_plot["服务方式"].isin(["地铁联运", "地铁+末端短驳"])].copy()
    metro_assign = metro_assign[metro_assign["中转站"].isin(selected_coord)].copy()
    line_rows = []
    for _, row in metro_assign.iterrows():
        target = selected_coord[row["中转站"]]
        line_rows.append({
            "需求点": row["需求点"],
            "中转站": row["中转站"],
            "需求点经度": row["绘图经度"],
            "需求点纬度": row["绘图纬度"],
            "站点经度": target["绘图经度"],
            "站点纬度": target["绘图纬度"],
        })

    line_columns = ["需求点", "中转站", "需求点经度", "需求点纬度", "站点经度", "站点纬度"]
    line_df = pd.DataFrame(line_rows, columns=line_columns)

    return {
        "demand_points": demand_plot,
        "standby_candidates": standby_plot,
        "selected_candidates": selected_plot,
        "other_metro_stations": other_plot,
        "assignment_lines": line_df,
    }


def plot_compromise_map(assign_df: pd.DataFrame, load_df: pd.DataFrame, inst: MetroLogisticsInstance, output_dir: str):
    plt = _get_pyplot()
    boundary_geojson, boundary_path = _load_local_or_desktop_geojson("510107.json")
    metro_points, metro_points_path = _load_local_or_desktop_geojson("武侯区地铁站点_2024年10月_WGS84.geojson")
    if metro_points is None:
        metro_points, metro_points_path = _load_local_or_desktop_geojson("武侯区地铁站点_2024年10月_WGS84 .geojson")
    metro_lines, metro_lines_path = _load_local_or_desktop_geojson("武侯区地铁线路折线_2024年10月_WGS84.geojson")
    if metro_lines is None:
        metro_lines, metro_lines_path = _load_local_or_desktop_geojson("武侯区地铁线路折线_2024年10月_WGS84 .geojson")

    fig, ax = plt.subplots(figsize=(12.4, 10.9))
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.08, top=0.93)

    boundary_extent = base.draw_wuhou_boundary(ax, boundary_geojson)
    base.draw_metro_lines(ax, metro_lines, linewidth=2.25, alpha=0.68, add_labels=False)

    layers = build_compromise_map_layers(assign_df, load_df, inst, boundary_geojson, metro_points)
    cand_plot_df = layers["standby_candidates"]
    opened_plot_df = layers["selected_candidates"]
    assign_plot_df = layers["demand_points"]
    assignment_lines = layers["assignment_lines"]
    opened_plot_df.to_csv(os.path.join(output_dir, "map_selected_station_coordinates.csv"), index=False, encoding="utf-8-sig")

    ax.scatter(
        assign_plot_df["绘图经度"], assign_plot_df["绘图纬度"],
        s=28, marker="o", facecolors="none", edgecolors="#C93A3A",
        linewidths=1.05, alpha=0.92, label="顺丰需求点", zorder=5
    )

    if not cand_plot_df.empty:
        ax.scatter(
            cand_plot_df["绘图经度"], cand_plot_df["绘图纬度"],
            s=105, marker="^", facecolors="white", edgecolors="black",
            linewidths=1.25, alpha=0.98, label="备选转运点", zorder=7
        )

    sizes = 420 + 360 * opened_plot_df["负载率"].clip(0, 1)
    ax.scatter(
        opened_plot_df["绘图经度"], opened_plot_df["绘图纬度"],
        s=sizes, marker="^", facecolors="black", edgecolors="#FFD54F",
        linewidths=1.85, alpha=0.98, label="选择的转运点", zorder=30
    )

    sample = assignment_lines.sample(min(65, len(assignment_lines)), random_state=base.RANDOM_SEED) if len(assignment_lines) > 0 else assignment_lines
    for _, row in sample.iterrows():
        ax.plot(
            [row["需求点经度"], row["站点经度"]],
            [row["需求点纬度"], row["站点纬度"]],
            linewidth=0.95, alpha=0.42, color="#5DADE2", zorder=4
        )

    offsets = [(6, 6), (8, -4), (-8, 7), (7, 10), (-10, -6), (10, 2), (-6, 10), (8, 8)]
    for i, (_, row) in enumerate(opened_plot_df.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        base.annotate_point(
            ax,
            row["绘图经度"],
            row["绘图纬度"],
            base.short_name(row["地铁站名称"], max_len=8),
            dx=dx,
            dy=dy,
            fontsize=10,
        )

    ax.scatter(
        opened_plot_df["绘图经度"], opened_plot_df["绘图纬度"],
        s=sizes, marker="^", facecolors="black", edgecolors="#FFD54F",
        linewidths=1.85, alpha=0.98, zorder=35
    )

    if boundary_extent is not None:
        xmin, xmax, ymin, ymax = boundary_extent
        pad_x = (xmax - xmin) * 0.08
        pad_y = (ymax - ymin) * 0.10
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("武侯区地铁物流折中解空间布局与分配关系")
    ax.grid(alpha=0.16, linestyle="--")

    note = f"边界：{os.path.basename(boundary_path)}；地铁点：{os.path.basename(metro_points_path)}；地铁线：{os.path.basename(metro_lines_path)}"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8, color="#666666", ha="left", va="bottom")

    legend_handles = [
        base.Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor='none',
                    markeredgecolor='#C93A3A', markeredgewidth=1.1, label='顺丰需求点'),
        base.Line2D([0], [0], marker='^', linestyle='None', markersize=8, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=1.2, label='备选转运点'),
        base.Line2D([0], [0], marker='^', linestyle='None', markersize=8, markerfacecolor='black',
                    markeredgecolor='black', markeredgewidth=1.2, label='选择的转运点'),
        base.Line2D([0], [0], color="#5DADE2", lw=1.2, alpha=0.55, label='分配连接线'),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, ncol=2,
              borderpad=0.8, handlelength=1.5, columnspacing=1.0)

    base.add_full_region_inset(fig, boundary_geojson, inst)
    _save_figure_local(fig, output_dir, "fig_compromise_map")
    plt.close(fig)


def export_nsga2_results(result: dict, inst: MetroLogisticsInstance, output_dir=NSGA2_OUTPUT_DIR, skip_plots: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx, dec in enumerate(result["pareto_dec"]):
        objs, cv, parts = evaluate_solution(dec, inst)
        row = {"solution_id": idx, "Z1_total_cost": objs[0], "Z2_metro_service_ratio": -objs[1], "CV": cv}
        row.update(parts)
        rows.append(row)
    pareto_df = pd.DataFrame(rows).sort_values(["CV", "Z1_total_cost", "Z2_metro_service_ratio"], ascending=[True, True, False])
    pareto_df.to_csv(os.path.join(output_dir, "nsga2_pareto.csv"), index=False, encoding="utf-8-sig")
    result["history"].to_csv(os.path.join(output_dir, "nsga2_history.csv"), index=False, encoding="utf-8-sig")

    compromise_idx = choose_compromise(result["pareto_objs"])
    assign_df, station_df, line_df = solution_to_tables(result["pareto_dec"][compromise_idx], inst)
    assign_df.to_csv(os.path.join(output_dir, "nsga2_compromise_assignments.csv"), index=False, encoding="utf-8-sig")
    station_df.to_csv(os.path.join(output_dir, "nsga2_station_load.csv"), index=False, encoding="utf-8-sig")
    line_df.to_csv(os.path.join(output_dir, "nsga2_line_load.csv"), index=False, encoding="utf-8-sig")

    if not skip_plots:
        plot_pareto(result, output_dir)
        plot_convergence(result, output_dir)
        plot_station_load(station_df, output_dir)
        plot_compromise_map(assign_df, station_df, inst, output_dir)
    return pareto_df, assign_df, station_df


def export_nsga2_results_no_repair(result: dict, inst: MetroLogisticsInstance, output_dir, skip_plots: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx, dec in enumerate(result["pareto_dec"]):
        objs, cv, parts = evaluate_solution(dec, inst, use_repair=False)
        row = {"solution_id": idx, "Z1_total_cost": objs[0], "Z2_metro_service_ratio": -objs[1], "CV": cv}
        row.update(parts)
        rows.append(row)
    pareto_df = pd.DataFrame(rows).sort_values(["CV", "Z1_total_cost", "Z2_metro_service_ratio"], ascending=[True, True, False])
    pareto_df.to_csv(os.path.join(output_dir, "nsga2_no_repair_pareto.csv"), index=False, encoding="utf-8-sig")
    result["history"].to_csv(os.path.join(output_dir, "nsga2_no_repair_history.csv"), index=False, encoding="utf-8-sig")
    compromise_idx = int(pareto_df.index[0]) if len(pareto_df) else 0
    if len(result["pareto_dec"]) > 0:
        assign_df, station_df, line_df = solution_to_tables(result["pareto_dec"][compromise_idx], inst, use_repair=False)
        assign_df.to_csv(os.path.join(output_dir, "nsga2_no_repair_compromise_assignments.csv"), index=False, encoding="utf-8-sig")
        station_df.to_csv(os.path.join(output_dir, "nsga2_no_repair_station_load.csv"), index=False, encoding="utf-8-sig")
        line_df.to_csv(os.path.join(output_dir, "nsga2_no_repair_line_load.csv"), index=False, encoding="utf-8-sig")
        if not skip_plots:
            plot_station_load(station_df, output_dir)
            plot_compromise_map(assign_df, station_df, inst, output_dir)
    if not skip_plots:
        plot_pareto(result, output_dir)
        plot_convergence(result, output_dir)
    return pareto_df



def _export_qvns_nsga2_results_core(result: dict, inst: MetroLogisticsInstance, output_dir, use_repair=True, skip_plots: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx, dec in enumerate(result["pareto_dec"]):
        objs, cv, parts = evaluate_solution(dec, inst, use_repair=use_repair)
        row = {"solution_id": idx, "Z1_total_cost": objs[0], "Z2_metro_service_ratio": -objs[1], "CV": cv}
        row.update(parts)
        rows.append(row)
    pareto_df = pd.DataFrame(rows).sort_values(["CV", "Z1_total_cost", "Z2_metro_service_ratio"], ascending=[True, True, False])
    pareto_df.to_csv(os.path.join(output_dir, "qvns_nsga2_pareto.csv"), index=False, encoding="utf-8-sig")
    result["history"].to_csv(os.path.join(output_dir, "qvns_nsga2_final_history.csv"), index=False, encoding="utf-8-sig")
    result["action_history"].to_csv(os.path.join(output_dir, "qvns_nsga2_training_history.csv"), index=False, encoding="utf-8-sig")
    result["q_table"].to_csv(os.path.join(output_dir, "qvns_nsga2_q_table.csv"), index=False, encoding="utf-8-sig")
    result["actions"].to_csv(os.path.join(output_dir, "qvns_nsga2_action_space.csv"), index=False, encoding="utf-8-sig")
    if "vns_history" in result and isinstance(result["vns_history"], pd.DataFrame):
        result["vns_history"].to_csv(os.path.join(output_dir, "qvns_nsga2_vns_history.csv"), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["gen", "elite_count", "candidate_count", "vns_trials", "vns_accepted"]).to_csv(os.path.join(output_dir, "qvns_nsga2_vns_history.csv"), index=False, encoding="utf-8-sig")
    if "best_params" in result:
        pd.DataFrame([result["best_params"]]).to_csv(os.path.join(output_dir, "qvns_nsga2_best_params.csv"), index=False, encoding="utf-8-sig")
    elif len(result["q_table"]) > 0:
        result["q_table"].sort_values("q_value", ascending=False).head(1).to_csv(os.path.join(output_dir, "qvns_nsga2_best_params.csv"), index=False, encoding="utf-8-sig")

    compromise_idx = choose_compromise(result["pareto_objs"]) if len(result.get("pareto_objs", [])) else 0
    if len(result["pareto_dec"]) > 0:
        assign_df, station_df, line_df = solution_to_tables(result["pareto_dec"][compromise_idx], inst, use_repair=use_repair)
        assign_df.to_csv(os.path.join(output_dir, "qvns_nsga2_compromise_assignments.csv"), index=False, encoding="utf-8-sig")
        station_df.to_csv(os.path.join(output_dir, "qvns_nsga2_station_load.csv"), index=False, encoding="utf-8-sig")
        line_df.to_csv(os.path.join(output_dir, "qvns_nsga2_line_load.csv"), index=False, encoding="utf-8-sig")
        if not skip_plots:
            plot_station_load(station_df, output_dir)
            plot_compromise_map(assign_df, station_df, inst, output_dir)
    if not skip_plots:
        plot_pareto(result, output_dir)
        plot_convergence(result, output_dir)
    return pareto_df


def qvns_nsga2_expected_output_paths(output_dir) -> dict:
    """Return thesis-facing QVNS-NSGA-II CSV paths for one scenario/run folder."""
    output_dir = Path(output_dir)
    return {Path(filename).stem: output_dir / filename for filename in QVNS_NSGA2_OUTPUT_FILES}


def nsga2_expected_output_paths(output_dir) -> dict:
    """Return thesis-facing standard NSGA-II CSV paths for one scenario/run folder."""
    output_dir = Path(output_dir)
    return {Path(filename).stem: output_dir / filename for filename in NSGA2_OUTPUT_FILES}

def _algorithm_file_prefix(result) -> str:
    algorithm = str(result.get("algorithm", "NSGA-II")).upper()
    if "QVNS" in algorithm or "QLEARNING" in algorithm or "Q-LEARNING" in algorithm:
        return "qvns_nsga2"
    return "nsga2"


def _algorithm_display_name(result) -> str:
    return str(result.get("algorithm", "NSGA-II"))


def export_qvns_nsga2_results(*args, **kwargs):
    """Public thesis-facing entry point for QVNS-NSGA-II result export."""
    return _export_qvns_nsga2_results_core(*args, **kwargs)



def plot_pareto(result, output_dir):
    plt = _get_pyplot()
    objs = result["pareto_objs"]
    prefix = _algorithm_file_prefix(result)
    algorithm = _algorithm_display_name(result)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.scatter(objs[:, 0], -objs[:, 1], s=70, c="#1f77b4", alpha=0.85, edgecolors="white")
    ax.set_xlabel("Z1 综合运营成本（元/日）")
    ax.set_ylabel("Z2 综合地铁服务需求占比")
    ax.set_title(f"{algorithm} Pareto 前沿")
    ax.grid(alpha=0.28, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fig_{prefix}_pareto.png"), dpi=420, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"fig_{prefix}_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(result, output_dir):
    plt = _get_pyplot()
    h = result["history"]
    prefix = _algorithm_file_prefix(result)
    algorithm = _algorithm_display_name(result)
    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    ax1.plot(h["gen"], h["best_cost"], color="#1f77b4", label="最优成本")
    ax1.set_xlabel("迭代代数")
    ax1.set_ylabel("最优成本")
    ax1.grid(alpha=0.25, linestyle="--")
    ax2 = ax1.twinx()
    ax2.plot(h["gen"], h["best_service_ratio"], color="#d62728", linestyle="--", label="最优地铁服务需求占比")
    ax2.set_ylabel("最优地铁服务需求占比")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title(f"{algorithm} 收敛过程")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fig_{prefix}_convergence.png"), dpi=420, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"fig_{prefix}_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_station_load(station_df, output_dir):
    plt = _get_pyplot()
    df = station_df[station_df["是否启用"] == 1].sort_values("负载率")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(df["地铁站名称"], df["负载率"] * 100.0, color="#4c78a8", alpha=0.86)
    ax.axvline(15, color="#2ca02c", linestyle="--", label="最小负载率 15%")
    ax.axvline(85, color="#d62728", linestyle="--", label="最大负载率 85%")
    ax.set_xlabel("负载率（%）")
    ax.set_title("折中解启用站点负载率")
    ax.legend()
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_station_load.png"), dpi=420, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig_station_load.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    inst = build_metro_logistics_instance(scenario="base", top_n=17)
    result = run_nsga2(inst, pop_size=100, max_gen=300, pc=0.9, seed=20260426, verbose=True)
    pareto_df, _, station_df = export_nsga2_results(result, inst)
    print("NSGA-II finished")
    print("Pareto size:", len(pareto_df))
    print("Output:", NSGA2_OUTPUT_DIR)
    print(pareto_df[["Z1_total_cost", "Z2_metro_service_ratio", "CV", "num_open", "avg_feeder_distance", "max_load_rate"]].head(10).to_string(index=False))
    print(station_df[station_df["是否启用"] == 1][["地铁站名称", "分配量_件日", "容量_件日", "负载率"]].to_string(index=False))


if __name__ == "__main__":
    main()


