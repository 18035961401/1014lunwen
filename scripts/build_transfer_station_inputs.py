# -*- coding: utf-8 -*-
"""Build small-paper transfer-station-only input tables.

The thesis default uses the CRITIC-TOPSIS top-17 candidate table.  The small
paper uses every Wuhou transfer station instead, so these scenario-specific
tables are written under ``data/小论文换乘站输入`` without replacing the thesis
data files.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "小论文换乘站输入"
MATRIX_OUT = OUT / "矩阵数据"

TRANSFER_CAPACITY = {"TR": 7000, "HB": 10000}
TRANSFER_FIXED_COST = {"TR": 12000, "HB": 16000}
ORDINARY_SPEED = 35.0
EXPRESS_SPEED = 80.0
ROAD_FACTOR = 1.5
EXPRESS_LINES = {"18号线"}

ACCESS_NODES = {
    "S1": {"name": "双流西站", "lon": 103.916978, "lat": 30.545125, "lines": ["10号线"]},
    "S2": {"name": "天府机场北", "lon": 104.453300, "lat": 30.345100, "lines": ["18号线"]},
}


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    radius = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def canonical_line(line_name: str) -> str:
    text = str(line_name).strip().replace("地铁", "")
    return text


def load_station_table() -> pd.DataFrame:
    stations = pd.read_csv(DATA / "武侯区地铁站点信息_按线路与官方站序排序_2024年10月.csv", encoding="utf-8-sig")
    line_count = pd.to_numeric(stations["线路数"], errors="coerce").fillna(0).astype(int)
    transfers = stations[(stations["是否转换站"].astype(str).str.strip() == "是") | (line_count >= 2)].copy()
    transfers["线路数"] = pd.to_numeric(transfers["线路数"], errors="coerce").fillna(0).astype(int)
    transfers["station_type"] = transfers["线路数"].map(lambda value: "HB" if int(value) >= 3 else "TR")
    return transfers.reset_index(drop=True)


def write_candidate_attributes(transfers: pd.DataFrame) -> Path:
    rows = []
    for idx, row in transfers.iterrows():
        station_type = row["station_type"]
        capacity = TRANSFER_CAPACITY[station_type]
        rows.append(
            {
                "station_id": idx + 1,
                "candidate_index_0based": idx,
                "station_name": row["地铁站名称"],
                "ict_rank": "",
                "ict_score": "",
                "main_line": row["主线路"],
                "all_lines": row["所有线路"],
                "line_count": int(row["线路数"]),
                "is_transfer_station": "是",
                "station_type": station_type,
                "station_type_basis": "小论文换乘站口径：线路数=2为TR；线路数>=3为HB；普通站不进入候选集合",
                "station_capacity_pieces": capacity,
                "fixed_cost_yuan": TRANSFER_FIXED_COST[station_type],
                "min_load_rate": 0.15,
                "max_load_rate": 0.85,
                "min_load_pieces": int(capacity * 0.15),
                "max_load_pieces": int(capacity * 0.85),
                "lon": float(row["经度"]),
                "lat": float(row["纬度"]),
                "address": row["地址"],
                "data_source": "武侯区地铁站点信息_按线路与官方站序排序_2024年10月.csv；小论文换乘站输入生成",
            }
        )
    out_path = OUT / "候选转运点最终属性表_换乘站.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def write_station_to_demand(transfers: pd.DataFrame) -> Path:
    demand = pd.read_csv(DATA / "武侯区_顺丰站点_精准清理_20251218_073112.csv", encoding="utf-8-sig")
    demand = demand.dropna(subset=["经度_GCJ02", "纬度_GCJ02"]).copy().reset_index(drop=True)
    rows = []
    for _, station in transfers.iterrows():
        row = {"origin_name": station["地铁站名称"]}
        for j, demand_row in demand.iterrows():
            dist = ROAD_FACTOR * haversine_km(
                float(station["经度"]),
                float(station["纬度"]),
                float(demand_row["经度_GCJ02"]),
                float(demand_row["纬度_GCJ02"]),
            )
            row[str(j + 1)] = round(dist, 3)
        rows.append(row)
    out_path = OUT / "地铁站点到顺丰站点_换乘站.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def load_line_features():
    line_geojson = json.loads((DATA / "武侯区地铁线路折线_2024年10月_WGS84.geojson").read_text(encoding="utf-8-sig"))
    station_coords = {}
    graph = defaultdict(list)
    for feature in line_geojson["features"]:
        props = feature["properties"]
        line = canonical_line(props["line_name"])
        stations = props["stations"]
        coords = feature["geometry"]["coordinates"]
        for station, coord in zip(stations, coords):
            station_coords[station] = (float(coord[0]), float(coord[1]))
        for left, right, coord_left, coord_right in zip(stations, stations[1:], coords, coords[1:]):
            weight = haversine_km(float(coord_left[0]), float(coord_left[1]), float(coord_right[0]), float(coord_right[1]))
            graph[left].append((right, weight, line))
            graph[right].append((left, weight, line))
    return graph, station_coords


def add_access_nodes(graph, station_coords):
    graph = defaultdict(list, {node: list(edges) for node, edges in graph.items()})
    station_coords = dict(station_coords)
    for source_id, source in ACCESS_NODES.items():
        source_name = source["name"]
        station_coords[source_name] = (source["lon"], source["lat"])
        for line in source["lines"]:
            line_stations = [node for node, edges in graph.items() if any(edge_line == line for _, _, edge_line in edges)]
            if not line_stations:
                continue
            nearest = min(
                line_stations,
                key=lambda station: haversine_km(source["lon"], source["lat"], station_coords[station][0], station_coords[station][1]),
            )
            dist = haversine_km(source["lon"], source["lat"], station_coords[nearest][0], station_coords[nearest][1])
            graph[source_name].append((nearest, dist, line))
            graph[nearest].append((source_name, dist, line))
    return graph, station_coords


def shortest_path(graph, source: str, target: str):
    heap = [(0.0, source, None, [source], [])]
    best = {}
    while heap:
        dist, node, last_line, path, lines = heappop(heap)
        state = (node, last_line)
        if dist > best.get(state, float("inf")) + 1e-12:
            continue
        if node == target:
            return dist, path, lines
        for nxt, weight, line in graph[node]:
            new_dist = dist + weight
            new_state = (nxt, line)
            if new_dist + 1e-12 < best.get(new_state, float("inf")):
                best[new_state] = new_dist
                heappush(heap, (new_dist, nxt, line, path + [nxt], lines + [line]))
    raise ValueError(f"No metro path found from {source} to {target}")


def compact_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return out


def write_metro_paths(transfers: pd.DataFrame) -> tuple[Path, Path]:
    graph, coords = load_line_features()
    graph, _ = add_access_nodes(graph, coords)
    summary_rows = []
    incidence_rows = []
    line_names = ["1号线", "3号线", "5号线", "7号线", "8号线", "9号线", "10号线", "17号线", "18号线"]
    for source_id, source in ACCESS_NODES.items():
        for rank, (_, station) in enumerate(transfers.iterrows(), start=1):
            distance, path, edge_lines = shortest_path(graph, source["name"], station["地铁站名称"])
            route_lines = compact_lines(edge_lines)
            transfer_count = max(0, len(route_lines) - 1)
            ordinary = sum(weight for left, right in zip(path, path[1:]) for nxt, weight, line in graph[left] if nxt == right and line not in EXPRESS_LINES)
            express = max(0.0, distance - ordinary)
            time_h = ordinary / ORDINARY_SPEED + express / EXPRESS_SPEED
            source_file = f"{source_id}_{source['name']}_线路距离_换乘站.csv"
            summary_rows.append(
                {
                    "排名": rank,
                    "目标站点": station["地铁站名称"],
                    "最短线路组合": " → ".join(route_lines),
                    "换乘站": "、".join([node for node in path[1:-1] if node in set(transfers["地铁站名称"])]) or "无",
                    "主要经过站点": " → ".join(path),
                    "站间段数": len(path) - 1,
                    "换乘次数": transfer_count,
                    "地铁线网距离/km": round(distance, 3),
                    "ordinary_line_distance_km": round(ordinary, 3),
                    "express_line_distance_km": round(express, 3),
                    "ordinary_line_speed_km_h": ORDINARY_SPEED,
                    "express_line_speed_km_h": EXPRESS_SPEED,
                    "segmented_metro_time_h": round(time_h, 6),
                    "segmentation_rule": "Generated from Wuhou line order and station coordinates for transfer-station small-paper scenario",
                    "source_file": source_file,
                }
            )
            used_lines = set(route_lines)
            for line in line_names:
                incidence_rows.append(
                    {
                        "source_id": source_id,
                        "target_station": station["地铁站名称"],
                        "line_id": f"L{line.replace('号线', '')}",
                        "line_name": line,
                        "a_ik_line": 1 if line in used_lines else 0,
                        "path_lines": " → ".join(route_lines),
                        "transfer_stations": "、".join([node for node in path[1:-1] if node in set(transfers["地铁站名称"])]) or "无",
                        "path_stations": " → ".join(path),
                        "metro_distance_km": round(distance, 3),
                        "transfer_count": transfer_count,
                    }
                )
    summary_path = MATRIX_OUT / "S1_S2_metro_distance_segmented_speed_summary_换乘站.csv"
    incidence_path = OUT / "路径线路关联矩阵_a_ik_line_换乘站.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(incidence_rows).to_csv(incidence_path, index=False, encoding="utf-8-sig")
    return summary_path, incidence_path


def write_readme(paths: list[Path]) -> Path:
    readme = OUT / "README_小论文换乘站输入说明.md"
    lines = [
        "# 小论文换乘站输入数据说明",
        "",
        "本目录为小论文独立算例数据，不覆盖大论文默认 CRITIC-TOPSIS 前 17 站输入。",
        "",
        "- 候选集合：武侯区地铁站点表中 `是否转换站=是` 或 `线路数>=2` 的 15 个物理站点。",
        "- 站点类型：线路数=2 记为 TR，线路数>=3 记为 HB；普通站不进入候选集合。",
        "- 站点至顺丰需求点距离：按站点与需求点 WGS/GCJ 坐标的大圆距离乘以 1.5 道路折算系数生成。",
        "- 接入站至候选站地铁路径：基于武侯区地铁线路 GeoJSON 的线路站序和站点坐标构图，按空间距离最短路径生成；用于小论文换乘站情景的可复现近似矩阵。",
        "",
        "生成文件：",
    ]
    lines.extend(f"- `{path.relative_to(OUT).as_posix()}`" for path in paths)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MATRIX_OUT.mkdir(parents=True, exist_ok=True)
    transfers = load_station_table()
    paths = [
        write_candidate_attributes(transfers),
        write_station_to_demand(transfers),
    ]
    paths.extend(write_metro_paths(transfers))
    paths.append(write_readme(paths))
    print("Generated transfer-station small-paper inputs:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
