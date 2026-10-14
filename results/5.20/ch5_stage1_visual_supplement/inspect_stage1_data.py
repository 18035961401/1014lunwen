from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[3]

patterns = {
    "critic_top17": "data/**/critic_topsis_top17.csv",
    "candidate_attr": "data/**/候选转运点最终属性表.csv",
    "station_csv": "data/*地铁站点信息*.csv",
    "station_geojson": "data/*地铁站点*_WGS84.geojson",
    "line_geojson": "data/*地铁线路折线*_WGS84.geojson",
    "boundary": "data/510107.json",
    "sf_geojson": "data/**/顺丰站点*.geojson",
}

for name, pattern in patterns.items():
    files = list(ROOT.glob(pattern))
    print(f"\n[{name}] {len(files)}")
    for f in files[:5]:
        print(f"  {f.relative_to(ROOT)}")
        try:
            if f.suffix.lower() == ".csv":
                df = pd.read_csv(f, encoding="utf-8-sig")
                print("  shape:", df.shape)
                print("  columns:", list(df.columns))
                print(df.head(5).to_string())
            elif f.suffix.lower() in {".geojson", ".json"}:
                gdf = gpd.read_file(f)
                print("  shape:", gdf.shape, "crs:", gdf.crs)
                print("  columns:", list(gdf.columns))
                print(gdf.head(3).to_string())
        except Exception as exc:
            print("  ERROR:", repr(exc))

