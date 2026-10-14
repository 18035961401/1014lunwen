# -*- coding: utf-8 -*-
"""Generate Chapter 5 stage-1 candidate identification visual supplements."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from shapely.ops import transform as shapely_transform
from tifffile import TiffFile


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent

BOUNDARY = DATA / "510107.json"
BOUNDARY_WGS84_PATTERN = "*\u884c\u653f\u8fb9\u754c_WGS84.geojson"
STATIONS = DATA / "武侯区地铁站点_2024年10月_WGS84.geojson"
LINES = DATA / "武侯区地铁线路折线_2024年10月_WGS84.geojson"
TOP17 = DATA / "第三章评价结果" / "critic_topsis_top17.csv"
WORLDPOP_NAME = "chn_pop_2025_CN_100m_R2025A_v1.tif"
WUHOU_POP_NAME = "武侯区人口密度.tif"
RANKING_KEYWORD = "CRITIC_TOPSIS"


def _transform_lat(lon: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * np.sqrt(abs(lon))
    ret += (20.0 * np.sin(6.0 * lon * np.pi) + 20.0 * np.sin(2.0 * lon * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lat * np.pi) + 40.0 * np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320.0 * np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * np.sqrt(abs(lon))
    ret += (20.0 * np.sin(6.0 * lon * np.pi) + 20.0 * np.sin(2.0 * lon * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lon * np.pi) + 40.0 * np.sin(lon / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(lon / 12.0 * np.pi) + 300.0 * np.sin(lon / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """Convert mainland-China GCJ-02 lon/lat to WGS84 lon/lat."""
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    mg_lat = lat + dlat
    mg_lon = lon + dlon
    return lon * 2 - mg_lon, lat * 2 - mg_lat


def load_wuhou_boundary_wgs84() -> gpd.GeoDataFrame:
    boundary_wgs84 = sorted(DATA.glob(BOUNDARY_WGS84_PATTERN))
    if boundary_wgs84:
        return gpd.read_file(boundary_wgs84[0]).to_crs("EPSG:4326")

    boundary = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    # AMap administrative-boundary JSON is GCJ-02 lon/lat. Convert it to WGS84
    # before overlaying with WorldPop and the WGS84 metro station/line layers.
    boundary = boundary.copy()
    boundary["geometry"] = boundary.geometry.apply(
        lambda geom: shapely_transform(lambda x, y, z=None: gcj02_to_wgs84(x, y), geom)
    )
    boundary = boundary.set_crs("EPSG:4326", allow_override=True)
    return boundary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 450,
        }
    )


def find_desktop_file(name_or_keyword: str, suffix: str | None = None) -> Path:
    roots = [Path.home() / "Desktop", ROOT]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if suffix and path.suffix.lower() != suffix.lower():
                continue
            if path.name == name_or_keyword or name_or_keyword in path.name:
                return path
    raise FileNotFoundError(name_or_keyword)


def get_tag(page, name: str, default=None):
    tag = page.tags.get(name)
    return tag.value if tag is not None else default


def read_worldpop_window(tif_path: Path, bounds: tuple[float, float, float, float], pad_deg: float = 0.006):
    """Read only the study-area window from the EPSG:4326 WorldPop GeoTIFF."""
    minx, miny, maxx, maxy = bounds
    minx -= pad_deg
    maxx += pad_deg
    miny -= pad_deg
    maxy += pad_deg
    with TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        arr = page.asarray(out="memmap")
        scale = get_tag(page, "ModelPixelScaleTag")
        tie = get_tag(page, "ModelTiepointTag")
        nodata = get_tag(page, "GDAL_NODATA")
        px, py = float(scale[0]), float(scale[1])
        x0, y0 = float(tie[3]), float(tie[4])
        row0 = max(0, int(np.floor((y0 - maxy) / py)))
        row1 = min(arr.shape[0], int(np.ceil((y0 - miny) / py)))
        col0 = max(0, int(np.floor((minx - x0) / px)))
        col1 = min(arr.shape[1], int(np.ceil((maxx - x0) / px)))
        win = np.array(arr[row0:row1, col0:col1], dtype="float32")
        if nodata is not None:
            try:
                win[win == float(nodata)] = np.nan
            except Exception:
                pass
        win[win < 0] = np.nan
        extent = (x0 + col0 * px, x0 + col1 * px, y0 - row1 * py, y0 - row0 * py)
        return win, extent


def read_clipped_population_raster(tif_path: Path):
    """Read the pre-clipped Wuhou population raster and transform its extent to WGS84."""
    with TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        arr = np.array(page.asarray(), dtype="float32")
        scale = get_tag(page, "ModelPixelScaleTag")
        tie = get_tag(page, "ModelTiepointTag")
        nodata = get_tag(page, "GDAL_NODATA")
        if nodata is not None:
            try:
                arr[arr == float(nodata)] = np.nan
            except Exception:
                pass
        arr[arr < 0] = np.nan
        px, py = float(scale[0]), float(scale[1])
        x0, y0 = float(tie[3]), float(tie[4])
        h, w = arr.shape
        xmin, xmax = x0, x0 + w * px
        ymax, ymin = y0, y0 - h * py
        transformer = Transformer.from_crs("EPSG:4524", "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(xmin, ymin)
        lon_max, lat_max = transformer.transform(xmax, ymax)
        extent = (lon_min, lon_max, lat_min, lat_max)
        return arr, extent


def exterior_path(geom) -> MplPath:
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords)
    else:
        poly = max(list(geom.geoms), key=lambda g: g.area)
        coords = np.asarray(poly.exterior.coords)
    return MplPath(coords)


def plot_fig5a(pop_path: Path, ranking: pd.DataFrame, *, is_clipped: bool) -> Path:
    boundary = load_wuhou_boundary_wgs84()
    stations = gpd.read_file(STATIONS).to_crs("EPSG:4326")
    lines = gpd.read_file(LINES).to_crs("EPSG:4326")

    candidate_names = set(ranking.loc[ranking["rank"] <= 17, "alternative"].astype(str))
    stations["is_candidate"] = stations["name"].astype(str).isin(candidate_names)
    if is_clipped:
        pop, extent = read_clipped_population_raster(pop_path)
    else:
        pop, extent = read_worldpop_window(pop_path, tuple(boundary.total_bounds))

    fig, ax = plt.subplots(figsize=(8.5, 8.9), constrained_layout=True)
    positive = pop[np.isfinite(pop) & (pop > 0)]
    vmin = max(1.0, float(np.nanpercentile(positive, 8))) if positive.size else 1.0
    vmax = float(np.nanpercentile(positive, 99)) if positive.size else 100.0
    im = ax.imshow(
        pop,
        extent=extent,
        origin="upper",
        cmap="YlOrRd",
        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1)),
        alpha=0.84,
        zorder=0,
    )
    im.set_clip_path(PathPatch(exterior_path(boundary.geometry.iloc[0]), transform=ax.transData))

    boundary.boundary.plot(ax=ax, color="#334155", linewidth=1.15, zorder=4)
    for _, row in lines.iterrows():
        color = row.get("color") or "#64748B"
        gpd.GeoSeries([row.geometry], crs=lines.crs).plot(
            ax=ax, color=color, linewidth=2.0, alpha=0.88, zorder=5
        )

    stations[~stations["is_candidate"]].plot(
        ax=ax, markersize=22, color="#FFFFFF", edgecolor="#334155", linewidth=0.55, alpha=0.96, zorder=7
    )
    stations[stations["is_candidate"]].plot(
        ax=ax, markersize=58, color="#2563EB", edgecolor="#FFFFFF", linewidth=0.8, alpha=0.98, zorder=8
    )

    label_names = ["高朋大道", "九兴大道", "火车南站", "神仙树", "高升桥", "省体育馆", "太平园", "金融城"]
    for _, row in stations[stations["name"].isin(label_names)].iterrows():
        ax.annotate(
            row["name"],
            (row.geometry.x, row.geometry.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.4,
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#CBD5E1", lw=0.45, alpha=0.88),
            zorder=9,
        )

    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx - 0.008, maxx + 0.008)
    ax.set_ylim(miny - 0.006, maxy + 0.006)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("武侯区地铁线网拓扑与2025年常住人口密度空间叠加", fontsize=14, pad=10)
    ax.grid(color="#CBD5E1", linewidth=0.45, linestyle="--", alpha=0.55)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("100 m人口栅格值（人/像元，对数色阶）", fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor="#F97316", edgecolor="none", alpha=0.75, label="人口密度较高区域"),
            Line2D([0], [0], color="#64748B", lw=2.0, label="Space-L地铁拓扑连线"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB", markeredgecolor="white", markersize=8, label="前17名候选站点"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#334155", markersize=6, label="其他评价站点"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        fontsize=8.8,
    )
    ax.text(
        0.01,
        0.012,
        "数据：WorldPop 2025 100 m人口栅格裁剪结果、武侯区行政边界、2024年10月地铁站点与线路数据；候选点来自CRITIC-TOPSIS前17名。",
        transform=ax.transAxes,
        fontsize=7.1,
        color="#334155",
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.86, pad=2.5),
    )

    png = OUT / "fig5A_wuhou_metro_population_overlay.png"
    pdf = OUT / "fig5A_wuhou_metro_population_overlay.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_fig5b(ranking: pd.DataFrame) -> Path:
    ranking = ranking.sort_values("rank").reset_index(drop=True)
    threshold = float(ranking.loc[ranking["rank"].eq(17), "ict_score"].iloc[0])
    x = np.arange(len(ranking))
    colors = np.where(ranking["rank"].to_numpy() <= 17, "#2563EB", "#CBD5E1")
    edges = np.where(ranking["rank"].to_numpy() <= 17, "#1E3A8A", "#94A3B8")

    fig, ax = plt.subplots(figsize=(13.4, 6.2), constrained_layout=True)
    bars = ax.bar(x, ranking["ict_score"], color=colors, edgecolor=edges, linewidth=0.45, width=0.78)
    for i, bar in enumerate(bars):
        if i >= 17:
            bar.set_hatch("//")
            bar.set_alpha(0.55)

    ax.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1.45, label="截断阈值：金融城（0.298044）")
    ax.axvline(16.5, color="#DC2626", linestyle=":", linewidth=1.25)
    ax.annotate(
        "排名前17位进入第二阶段候选集合",
        xy=(8, threshold + 0.012),
        xytext=(3, threshold + 0.12),
        arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.0),
        fontsize=10,
        color="#1E3A8A",
    )
    ax.annotate(
        "第17名金融城作为初选截断点",
        xy=(16.5, threshold),
        xytext=(20, threshold + 0.055),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.95),
        fontsize=9,
        color="#B91C1C",
    )

    for idx in [0, 1, 2, 4, 5, 8, 16, 24, 34, 50]:
        score = float(ranking.loc[idx, "ict_score"])
        ax.text(idx, score + 0.008, f"{score:.3f}", ha="center", va="bottom", fontsize=7.2, color="#0F172A")

    ax.set_title("基于CRITIC-TOPSIS的地铁站点综合重要度得分排行与初选阈值", fontsize=14, pad=10)
    ax.set_ylabel("综合贴近度得分")
    ax.set_xlabel("地铁站点（按综合贴近度降序排列）")
    ax.set_ylim(0, max(float(ranking["ict_score"].max()) * 1.22, threshold + 0.16))
    ax.set_xlim(-0.8, len(ranking) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ranking["alternative"].astype(str), rotation=72, ha="right", fontsize=7.1)
    for i, tick in enumerate(ax.get_xticklabels()):
        if i >= 17:
            tick.set_color("#64748B")
    ax.grid(axis="y", color="#CBD5E1", linestyle="--", linewidth=0.55, alpha=0.7)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.92)
    ax.text(
        0.0,
        -0.31,
        "注：蓝色柱为最终进入第二阶段布局优化模型的17个候选地铁物流配送转运点；灰色柱为未进入候选集合的评价站点。红色虚线表示第17名金融城的综合贴近度得分。",
        transform=ax.transAxes,
        fontsize=8,
        color="#334155",
    )

    png = OUT / "fig5B_critic_topsis_candidate_threshold.png"
    pdf = OUT / "fig5B_critic_topsis_candidate_threshold.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def write_notes(worldpop_path: Path, pop_path: Path, ranking_path: Path, fig_a: Path, fig_b: Path) -> None:
    note = f"""# 第五章任务包一图件数据说明

## 已生成图件

- 图 5-A：`{fig_a.name}` / `{fig_a.with_suffix('.pdf').name}`
- 图 5-B：`{fig_b.name}` / `{fig_b.with_suffix('.pdf').name}`

## 数据来源

- WorldPop 2025 100 m人口栅格原始文件：`{worldpop_path}`
- 武侯区人口密度裁剪栅格：`{pop_path}`
- 武侯区行政边界：`{BOUNDARY}`
- 武侯区地铁站点：`{STATIONS}`
- 武侯区地铁线路折线：`{LINES}`
- CRITIC-TOPSIS全量排序：`{ranking_path}`，sheet：`ranking`
- CRITIC-TOPSIS前17名校验表：`{TOP17}`

## 处理逻辑

1. 图 5-A 使用已裁剪到研究区的武侯区人口密度栅格，采用对数色阶显示人口集聚差异，并叠加 Space-L 地铁拓扑连线、51个地铁站点及CRITIC-TOPSIS前17名候选站点。该裁剪栅格与 WorldPop 2025 100 m人口栅格作为同一人口密度底图来源体系使用。
2. 图 5-B 使用 `地铁物流转运点_CRITIC_TOPSIS排序结果.xlsx` 中 `ranking` sheet 的51个站点全量综合贴近度得分绘制降序柱状图，并以第17名金融城 `0.298044` 作为截断阈值。
3. 输出图件同时保存为 PNG 与 PDF，PNG 适合直接插入 Word，PDF 适合后续矢量排版。

## 可插入正文文本

为了直观呈现候选站点与现实物流需求之间的空间匹配关系，本文进一步将 2025 年 WorldPop 100 m 常住人口栅格、武侯区地铁线网 Space-L 拓扑结构和 51 个地铁站点位置进行叠加，如图 5-A 所示。可以看出，武侯区中东部核心片区人口密度较高，形成了较强的末端物流需求引力；省体育馆、火车南站、高升桥等候选站点位于人口集聚区与多条地铁线路交汇或邻近区域，具备较好的“网络可达性—需求承载性”协同基础。

进一步地，图 5-B 给出了 51 个地铁站点基于 CRITIC-TOPSIS 方法得到的综合贴近度降序排行。本文以第 17 名金融城的综合贴近度得分 0.298044 作为初选截断阈值，将排名前 17 位的站点纳入第二阶段布局优化模型的候选集合。该处理一方面压缩了后续优化模型的决策规模，另一方面保留了网络结构优势较强和站域需求潜力较高的站点，从而降低了由候选集合设置不透明引起的“黑箱感”。
"""
    (OUT / "data_sources_and_notes.md").write_text(note, encoding="utf-8")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    worldpop_path = find_desktop_file(WORLDPOP_NAME, ".tif")
    try:
        pop_path = find_desktop_file(WUHOU_POP_NAME, ".tif")
        is_clipped = True
    except FileNotFoundError:
        pop_path = worldpop_path
        is_clipped = False
    ranking_path = find_desktop_file(RANKING_KEYWORD, ".xlsx")
    ranking = pd.read_excel(ranking_path, sheet_name="ranking")
    required = {"rank", "alternative", "ict_score"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Ranking sheet missing columns: {required - set(ranking.columns)}")
    fig_a = plot_fig5a(pop_path, ranking, is_clipped=is_clipped)
    fig_b = plot_fig5b(ranking)
    write_notes(worldpop_path, pop_path, ranking_path, fig_a, fig_b)
    print(fig_a)
    print(fig_b)
    print(OUT / "data_sources_and_notes.md")


if __name__ == "__main__":
    main()
