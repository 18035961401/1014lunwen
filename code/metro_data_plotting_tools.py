# -*- coding: utf-8 -*-
"""Shared data loading and plotting utilities for the thesis metro-logistics package.

This module is intentionally **not** an algorithm entry point.  The formal
solver algorithms in this package are only:

- NSGA-II: ``code/run_nsga2_for_thesis.py``
- QVNS-NSGA-II: ``code/run_qvns_nsga2_for_thesis.py``

The functions below provide common paths, distance utilities, the two logistics
parks and access stations, lightweight base-instance construction, and map
plotting helpers used by ``code/main.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    from chinese_font_utils import CJK_FONT_CANDIDATES, configure_matplotlib_cjk, select_cjk_font
except ImportError:  # pragma: no cover - fallback for unusual direct imports
    CJK_FONT_CANDIDATES = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    configure_matplotlib_cjk = None
    select_cjk_font = None

plt = None
MplPath = None
Line2D = None


def _ensure_matplotlib():
    """Lazy-load matplotlib so command-line solvers can run without plotting."""
    global plt, MplPath, Line2D
    if plt is None or MplPath is None or Line2D is None:
        import matplotlib.pyplot as _plt
        from matplotlib.path import Path as _MplPath
        from matplotlib.lines import Line2D as _Line2D
        plt = _plt
        MplPath = _MplPath
        Line2D = _Line2D
    return plt, MplPath, Line2D


class _LazyLine2D:
    def __call__(self, *args, **kwargs):
        return _ensure_matplotlib()[2](*args, **kwargs)


Line2D = _LazyLine2D()


def _select_cjk_font_list():
    if select_cjk_font is not None:
        selected = select_cjk_font()
        return [selected] + [font for font in CJK_FONT_CANDIDATES if font != selected]
    preferred = os.environ.get("MPL_CJK_FONT")
    if preferred:
        return [preferred] + [f for f in CJK_FONT_CANDIDATES if f != preferred]
    try:
        from matplotlib import font_manager as _fm
        available = {font.name for font in _fm.fontManager.ttflist}
        ordered = [font for font in CJK_FONT_CANDIDATES if font in available]
        ordered += [font for font in CJK_FONT_CANDIDATES if font not in ordered]
        return ordered
    except Exception:
        return CJK_FONT_CANDIDATES


def setup_matplotlib_chinese():
    _plt, _, _ = _ensure_matplotlib()
    if configure_matplotlib_cjk is not None:
        font_name = configure_matplotlib_cjk(_plt)
        _plt.rcParams["figure.dpi"] = 140
        _plt.rcParams["savefig.dpi"] = 320
        return font_name
    font_list = _select_cjk_font_list()
    _plt.rcParams["font.family"] = "sans-serif"
    _plt.rcParams["font.sans-serif"] = font_list
    _plt.rcParams["axes.unicode_minus"] = False
    _plt.rcParams["svg.fonttype"] = "none"
    _plt.rcParams["pdf.fonttype"] = 42
    _plt.rcParams["ps.fonttype"] = 42
    _plt.rcParams["figure.dpi"] = 140
    _plt.rcParams["savefig.dpi"] = 320
    return font_list[0]


PLOT_FONT = select_cjk_font() if select_cjk_font is not None else os.environ.get("MPL_CJK_FONT", CJK_FONT_CANDIDATES[0])

# Plot colors used by main.py map export.
COLOR_LOGI = "#d62728"
COLOR_ACCESS = "#9467bd"

# ---------------------------------------------------------------------------
# Package paths and core constants
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA_DIR = PACKAGE_ROOT / "data"
BASE_DIR = str(PACKAGE_DATA_DIR if PACKAGE_DATA_DIR.exists() else Path.cwd())
METRO_STATION_FILE = str(PACKAGE_DATA_DIR / "武侯区地铁站点信息_按线路与官方站序排序_2024年10月.csv")
SF_SITE_FILE = str(PACKAGE_DATA_DIR / "武侯区_顺丰站点_精准清理_20251218_073112.csv")
TOPSIS_FILE = str(PACKAGE_DATA_DIR / "第三章评价结果" / "critic_topsis_top17.csv")
BOUNDARY_REMOTE_CACHE_FILE = str(PACKAGE_DATA_DIR / "510107.json")
METRO_POINT_GEOJSON_FILE = str(PACKAGE_DATA_DIR / "武侯区地铁站点_2024年10月_WGS84.geojson")
METRO_POINT_GEOJSON_FILE_ALT = METRO_POINT_GEOJSON_FILE
METRO_LINE_GEOJSON_FILE = str(PACKAGE_DATA_DIR / "武侯区地铁线路折线_2024年10月_WGS84.geojson")
OUTPUT_DIR = str(PACKAGE_ROOT / "results")
RANDOM_SEED = 42

# Cost parameters retained for plotting/default construction.  Formal costs are
# calibrated and read by main.py from supplementary parameter CSV files.
C2_FEEDER = 0.04
C3_TRANSFER = 0.10
D_MAX = 4.0
ROAD_DISTANCE_FACTOR = 1.5
METRO_DISTANCE_FACTOR = 1.2

# Logistics parks and their metro access stations.
LOGISTICS = pd.DataFrame([
    {"name": "L1顺丰丰泰产业园", "lon": 103.960550, "lat": 30.552564, "access": "S1双流西站"},
    {"name": "L2顺丰丰泰产业园成都天府", "lon": 104.500946, "lat": 30.438863, "access": "S2天府机场北站"},
])
ACCESS_STATIONS = pd.DataFrame([
    {"name": "S1双流西站", "lon": 103.916978, "lat": 30.545125, "lines": "3/10"},
    {"name": "S2天府机场北站", "lon": 104.453300, "lat": 30.345100, "lines": "18/19"},
])

TRANSFER_COUNT = {
    "S1双流西站": {
        "省体育馆": 0, "高升桥": 0, "火车南站": 1, "倪家桥": 1, "高朋大道": 1,
        "衣冠庙": 0, "九兴大道": 1, "孵化园": 2, "太平园": 0, "神仙树": 1,
        "芳草街": 1, "红牌楼": 0, "永丰": 1, "新南门": 0, "科园": 1,
        "桐梓林": 1, "川大望江校区": 1,
    },
    "S2天府机场北站": {
        "省体育馆": 1, "高升桥": 2, "火车南站": 0, "倪家桥": 1, "高朋大道": 1,
        "衣冠庙": 2, "九兴大道": 1, "孵化园": 0, "太平园": 1, "神仙树": 1,
        "芳草街": 2, "红牌楼": 2, "永丰": 2, "新南门": 2, "科园": 2,
        "桐梓林": 1, "川大望江校区": 2,
    },
}

DEMAND_SCENARIO_COLUMNS = {
    "low": "低需求_日均需求量",
    "base": "基准需求_日均需求量",
    "high": "高需求_日均需求量",
}


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Supports scalar or NumPy array inputs."""
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_sf_site_demand(demand_df: pd.DataFrame, scenario: str) -> np.ndarray:
    """Load station-level daily demand from the bundled SF78 scenario columns."""
    scenario_key = str(scenario).strip().lower()
    column = DEMAND_SCENARIO_COLUMNS.get(scenario_key, DEMAND_SCENARIO_COLUMNS["base"])
    if column in demand_df.columns:
        q = pd.to_numeric(demand_df[column], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(q)) and float(np.sum(q)) > 0:
            return q
    if "需求权重" in demand_df.columns:
        weights = pd.to_numeric(demand_df["需求权重"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if weights.sum() > 0:
            return 66970.0 * weights / weights.sum()
    return np.full(len(demand_df), 66970.0 / max(len(demand_df), 1), dtype=float)


def read_topsis_top_stations(topsis_file: str, top_n: int = 17) -> List[str]:
    """Read the top candidate metro stations from the CRITIC-TOPSIS CSV/text file."""
    path = Path(topsis_file)
    names: List[str] = []
    if path.exists() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        name_col = "alternative" if "alternative" in df.columns else ("站点名称" if "站点名称" in df.columns else df.columns[0])
        if "rank" in df.columns:
            df = df.sort_values("rank")
        names = [str(v).strip() for v in df[name_col].tolist() if str(v).strip()]
    elif path.exists():
        pattern = re.compile(r"^\s*\d+\s+([^\d]+?)\s+\d+\.\d+")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pattern.match(line)
            if m:
                names.append(m.group(1).strip())
    fallback = ["高朋大道", "九兴大道", "火车南站", "神仙树", "高升桥", "省体育馆", "孵化园",
                "倪家桥", "太平园", "衣冠庙", "红牌楼", "新南门", "芳草街", "科园", "桐梓林", "永丰", "金融城"]
    if len(names) < top_n:
        names = names + [n for n in fallback if n not in names]
    return names[:top_n]


def select_candidate_stations(metro_df: pd.DataFrame, top_n: int = 17, candidate_mode: str = "critic_topsis_top") -> pd.DataFrame:
    """Select candidate transfer-point stations for the model instance.

    ``critic_topsis_top`` preserves the thesis default. ``transfer_all`` is the
    small-paper scenario that uses every Wuhou transfer station as the input set.
    """
    mode = str(candidate_mode or "critic_topsis_top").strip().lower()
    if mode in {"critic_topsis_top", "critic_topsis", "top", "top17"}:
        top_names = read_topsis_top_stations(TOPSIS_FILE, top_n)
        cand_df = metro_df[metro_df["地铁站名称"].isin(top_names)].copy()
        cand_df["topsis_order"] = cand_df["地铁站名称"].map({n: i + 1 for i, n in enumerate(top_names)})
        cand_df = cand_df.sort_values("topsis_order").reset_index(drop=True)
        if len(cand_df) != top_n:
            missing = [n for n in top_names if n not in set(cand_df["地铁站名称"])]
            raise ValueError(f"候选站点数量不足，缺失：{missing}")
        return cand_df

    if mode in {"transfer_all", "all_transfer", "wuhou_transfer", "transfer"}:
        line_count = pd.to_numeric(metro_df["线路数"], errors="coerce").fillna(0)
        is_transfer = metro_df["是否转换站"].astype(str).str.strip().eq("是") | (line_count >= 2)
        cand_df = metro_df[is_transfer].copy().reset_index(drop=True)
        cand_df["topsis_order"] = np.arange(1, len(cand_df) + 1)
        if cand_df.empty:
            raise ValueError("未从武侯区地铁站点表中筛选到换乘站。")
        return cand_df

    raise ValueError(f"未知候选站点模式：{candidate_mode}")


@dataclass
class Instance:
    scenario: str
    K: int
    J: int
    I: int
    candidate_df: pd.DataFrame
    demand_df: pd.DataFrame
    logistics_df: pd.DataFrame
    access_df: pd.DataFrame
    Q_j: np.ndarray
    d_kj_road: np.ndarray
    d_ij_road: np.ndarray
    d_is_road: np.ndarray
    d_metro: np.ndarray
    transfer_count: np.ndarray


def build_instance(scenario: str = "base", top_n: int = 17, candidate_mode: str = "critic_topsis_top") -> Instance:
    """Build the lightweight base instance consumed by ``main.py``.

    Formal model calibration happens in ``main.build_metro_logistics_instance``;
    this base instance only provides aligned candidate stations, demands and
    fallback matrices.
    """
    metro_df = pd.read_csv(METRO_STATION_FILE, encoding="utf-8-sig")
    sf_df = pd.read_csv(SF_SITE_FILE, encoding="utf-8-sig")

    cand_df = select_candidate_stations(metro_df, top_n=top_n, candidate_mode=candidate_mode)

    demand_df = sf_df.dropna(subset=["经度_GCJ02", "纬度_GCJ02"]).copy().reset_index(drop=True)
    K, J, I = len(cand_df), len(demand_df), len(LOGISTICS)
    Q_j = load_sf_site_demand(demand_df, scenario)

    d_kj_geo = np.zeros((K, J), dtype=float)
    for k, row_k in cand_df.iterrows():
        d_kj_geo[k] = haversine_km(row_k["经度"], row_k["纬度"], demand_df["经度_GCJ02"].values, demand_df["纬度_GCJ02"].values)
    d_kj_road = ROAD_DISTANCE_FACTOR * d_kj_geo

    d_ij_geo = np.zeros((I, J), dtype=float)
    for i, row_i in LOGISTICS.iterrows():
        d_ij_geo[i] = haversine_km(row_i["lon"], row_i["lat"], demand_df["经度_GCJ02"].values, demand_df["纬度_GCJ02"].values)
    d_ij_road = ROAD_DISTANCE_FACTOR * d_ij_geo

    d_is_road = np.zeros(I, dtype=float)
    for i, row_i in LOGISTICS.iterrows():
        row_s = ACCESS_STATIONS.iloc[i]
        d_is_road[i] = ROAD_DISTANCE_FACTOR * float(haversine_km(row_i["lon"], row_i["lat"], row_s["lon"], row_s["lat"]))

    d_metro = np.zeros((I, K), dtype=float)
    transfer_count = np.zeros((I, K), dtype=float)
    for i, row_s in ACCESS_STATIONS.iterrows():
        for k, row_k in cand_df.iterrows():
            d_metro[i, k] = METRO_DISTANCE_FACTOR * float(haversine_km(row_s["lon"], row_s["lat"], row_k["经度"], row_k["纬度"]))
            transfer_count[i, k] = TRANSFER_COUNT.get(row_s["name"], {}).get(row_k["地铁站名称"], 1)

    return Instance(
        scenario=scenario,
        K=K,
        J=J,
        I=I,
        candidate_df=cand_df,
        demand_df=demand_df,
        logistics_df=LOGISTICS.copy(),
        access_df=ACCESS_STATIONS.copy(),
        Q_j=Q_j,
        d_kj_road=d_kj_road,
        d_ij_road=d_ij_road,
        d_is_road=d_is_road,
        d_metro=d_metro,
        transfer_count=transfer_count,
    )


# ---------------------------------------------------------------------------
# GeoJSON and map plotting helpers
# ---------------------------------------------------------------------------
def load_geojson(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_geojson_lines(geom):
    if geom is None:
        return
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    if gtype == "LineString":
        yield coords
    elif gtype == "MultiLineString":
        for line in coords:
            yield line


def iter_geojson_polygons(geom):
    if geom is None:
        return
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon" and coords:
        yield coords[0]
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]


def point_in_boundary(x, y, boundary_geojson):
    _, MplPath_local, _ = _ensure_matplotlib()
    if not boundary_geojson:
        return True
    for feat in boundary_geojson.get("features", []):
        for ring in iter_geojson_polygons(feat.get("geometry")):
            if len(ring) >= 3 and MplPath_local(ring).contains_point((x, y), radius=1e-12):
                return True
    return False


def boundary_centroid(boundary_geojson):
    xs, ys = [], []
    if boundary_geojson:
        for feat in boundary_geojson.get("features", []):
            for ring in iter_geojson_polygons(feat.get("geometry")):
                xs.extend([p[0] for p in ring])
                ys.extend([p[1] for p in ring])
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs and ys else (104.04, 30.60)


def nudge_point_inside_boundary(x, y, boundary_geojson, step_ratio=0.03, max_iter=200):
    x, y = float(x), float(y)
    if point_in_boundary(x, y, boundary_geojson):
        return x, y
    cx, cy = boundary_centroid(boundary_geojson)
    nx, ny = x, y
    for _ in range(max_iter):
        nx = nx + (cx - nx) * step_ratio
        ny = ny + (cy - ny) * step_ratio
        if point_in_boundary(nx, ny, boundary_geojson):
            return nx + (cx - nx) * 0.06, ny + (cy - ny) * 0.06
    return x + (cx - x) * 0.35, y + (cy - y) * 0.35


def adjust_display_points_to_boundary(df, boundary_geojson, lon_col="经度", lat_col="纬度"):
    out = df.copy()
    draw_lons, draw_lats = [], []
    moved = 0
    for x, y in zip(out[lon_col], out[lat_col]):
        nx, ny = nudge_point_inside_boundary(x, y, boundary_geojson)
        moved += int(abs(nx - float(x)) > 1e-12 or abs(ny - float(y)) > 1e-12)
        draw_lons.append(nx)
        draw_lats.append(ny)
    out["绘图经度"] = draw_lons
    out["绘图纬度"] = draw_lats
    return out, moved


def draw_wuhou_boundary(ax, boundary_geojson):
    if not boundary_geojson:
        return None
    all_x, all_y = [], []
    for feat in boundary_geojson.get("features", []):
        for ring in iter_geojson_polygons(feat.get("geometry")):
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            all_x.extend(xs)
            all_y.extend(ys)
            ax.fill(xs, ys, color="#F8E9C8", alpha=0.52, zorder=0)
            ax.plot(xs, ys, color="white", linewidth=5.2, alpha=0.95, zorder=1)
            ax.plot(xs, ys, color="#8C510A", linewidth=2.4, alpha=0.98, zorder=2)
    if not all_x:
        return None
    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    ax.text(cx, cy, "武侯区", fontsize=22, color="#8C510A", ha="center", va="center", alpha=0.18, weight="bold", zorder=0.5)
    return min(all_x), max(all_x), min(all_y), max(all_y)


def draw_metro_lines(ax, line_geojson, linewidth=2.8, alpha=0.92, add_labels=True):
    if not line_geojson:
        return
    used_labels = set()
    for feat in line_geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        line_name = props.get("line_name") or props.get("line") or "地铁线路"
        color = props.get("color") or "#666666"
        label = line_name if (add_labels and line_name not in used_labels) else None
        if label:
            used_labels.add(line_name)
        for coords in iter_geojson_lines(feat.get("geometry")):
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=1)


def extract_geojson_point_df(point_geojson):
    rows = []
    if not point_geojson:
        return pd.DataFrame(columns=["站点名称", "经度", "纬度"])
    for feat in point_geojson.get("features", []):
        geom = feat.get("geometry", {}) or {}
        props = feat.get("properties", {}) or {}
        if geom.get("type") == "Point":
            coord = geom.get("coordinates", [None, None])
            if len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
                name = props.get("name") or props.get("站点名称") or props.get("station_name") or props.get("station") or props.get("NAME") or ""
                rows.append({"站点名称": str(name), "经度": float(coord[0]), "纬度": float(coord[1])})
    return pd.DataFrame(rows)


def annotate_point(ax, x, y, text, dx=4, dy=4, fontsize=10):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        ha="left",
        va="bottom",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#bbbbbb", lw=0.6, alpha=0.9),
    )


def short_name(name: str, max_len: int = 8):
    name = str(name)
    return name if len(name) <= max_len else name[:max_len] + "…"


def add_full_region_inset(fig, boundary_geojson, inst):
    """Add a compact inset showing logistics parks and metro access stations."""
    _plt, _, _Line2D = _ensure_matplotlib()
    inset = fig.add_axes([0.77, 0.11, 0.20, 0.24])
    if boundary_geojson:
        draw_wuhou_boundary(inset, boundary_geojson)
    inset.scatter(inst.logistics_df["lon"], inst.logistics_df["lat"], s=38, marker="D", color=COLOR_LOGI, edgecolors="white", linewidths=0.5, zorder=4)
    inset.scatter(inst.access_df["lon"], inst.access_df["lat"], s=34, marker="X", color=COLOR_ACCESS, edgecolors="white", linewidths=0.5, zorder=4)
    for _, r in inst.logistics_df.iterrows():
        inset.text(r["lon"], r["lat"], str(r["name"])[:2], fontsize=7)
    for _, r in inst.access_df.iterrows():
        inset.text(r["lon"], r["lat"], str(r["name"])[:2], fontsize=7)
    xs = list(inst.logistics_df["lon"]) + list(inst.access_df["lon"])
    ys = list(inst.logistics_df["lat"]) + list(inst.access_df["lat"])
    if boundary_geojson:
        for feat in boundary_geojson.get("features", []):
            for ring in iter_geojson_polygons(feat.get("geometry")):
                xs.extend([p[0] for p in ring])
                ys.extend([p[1] for p in ring])
    if xs and ys:
        inset.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
        inset.set_ylim(min(ys) - 0.03, max(ys) + 0.03)
    inset.set_title("物流中心相对位置", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.18, linestyle="--")


# ``Line2D`` stays as a lazy callable until a plotting path actually needs
# matplotlib.  This keeps data/model imports usable in headless environments.
