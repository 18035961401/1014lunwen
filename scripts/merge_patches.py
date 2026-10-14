# -*- coding: utf-8 -*-
"""Merge patch_*.csv into main ablation_run_metrics.csv."""
import pandas as pd
import shutil
from pathlib import Path
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "results" / "qvns_ablation" / "ablation_run_metrics.csv"

print("=" * 60)
print("合并 patch 结果到主文件")
print("=" * 60)

df_main = pd.read_csv(MAIN_FILE, encoding='utf-8-sig')
print(f"主文件: {len(df_main)} 条")

patch_files = list(PROJECT_ROOT.glob("results/patch_*.csv"))
print(f"找到 {len(patch_files)} 个 patch 文件")

all_new = []
for pf in sorted(patch_files):
    df = pd.read_csv(pf, encoding='utf-8-sig')
    print(f"  {pf.name}: {len(df)} 条")
    for _, row in df.iterrows():
        algo = row['algorithm']
        rid = row['run_id']
        ms = row['model_setting']
        exists = df_main[(df_main['algorithm']==algo)&(df_main['run_id']==rid)&(df_main['model_setting']==ms)]
        if len(exists) == 0:
            all_new.append(row)

print(f"\n新记录: {len(all_new)} 条")

if all_new:
    df_new = pd.DataFrame(all_new)
    df_merged = pd.concat([df_main, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(['algorithm','model_setting','run_id']).reset_index(drop=True)
    backup = str(MAIN_FILE) + '.backup'
    shutil.copy2(MAIN_FILE, backup)
    df_merged.to_csv(MAIN_FILE, index=False, encoding='utf-8-sig')
    print(f"合并完成: {len(df_main)} -> {len(df_merged)}")
    df_main = df_merged

print("\n各算法:")
for algo in sorted(df_main['algorithm'].unique()):
    cnt = len(df_main[df_main['algorithm']==algo])
    done = sorted(df_main[df_main['algorithm']==algo]['run_id'].unique())
    mis = [i for i in range(1,21) if i not in done]
    s = '完成' if not mis else f'缺{len(mis)}次: {mis}'
    print(f"  {algo:20s}: {cnt:2d}/20 {s}")
