# -*- coding: utf-8 -*-
"""Merge all parallel results into main file."""
import pandas as pd
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "results" / "qvns_ablation" / "ablation_run_metrics.csv"

print("=" * 60)
print("合并所有并行结果到主文件")
print("=" * 60)

df_main = pd.read_csv(MAIN_FILE, encoding='utf-8-sig')
print(f"主文件记录数: {len(df_main)}")

remaining_dirs = [
    'remaining_q9a', 'remaining_q9b', 'remaining_q9c', 'remaining_q9d',
    'remaining_q11', 'remaining_q11a', 'remaining_q11b', 'remaining_q11c',
    'qvns_ablation_remaining'
]

all_new = []
for d in remaining_dirs:
    csv_file = PROJECT_ROOT / "results" / d / "ablation_run_metrics.csv"
    if not csv_file.exists():
        continue
    
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"\n{d}: {len(df)} 条记录")
    
    for _, row in df.iterrows():
        algo = row['algorithm']
        run_id = row['run_id']
        model_setting = row['model_setting']
        
        exists = df_main[
            (df_main['algorithm'] == algo) & 
            (df_main['run_id'] == run_id) &
            (df_main['model_setting'] == model_setting)
        ]
        if len(exists) == 0:
            all_new.append(row)

print(f"\n找到 {len(all_new)} 条新记录")

if len(all_new) > 0:
    df_new = pd.DataFrame(all_new)
    df_merged = pd.concat([df_main, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(['algorithm', 'model_setting', 'run_id']).reset_index(drop=True)
    
    backup = str(MAIN_FILE) + '.backup_final'
    shutil.copy2(MAIN_FILE, backup)
    df_merged.to_csv(MAIN_FILE, index=False, encoding='utf-8-sig')
    
    print(f"合并完成: {len(df_main)} -> {len(df_merged)}")
    df_main = df_merged
else:
    print("没有新记录需要合并")

print("\n各算法完成情况:")
for algo in sorted(df_main['algorithm'].unique()):
    count = len(df_main[df_main['algorithm'] == algo])
    status = '完成' if count >= 20 else f'还需 {20-count} 轮'
    print(f"  {algo:20s}: {count:2d}/20 {status}")

remaining = sum(max(0, 20 - len(df_main[df_main['algorithm'] == algo])) 
               for algo in df_main['algorithm'].unique())
print(f"\n总剩余轮数: {remaining}")
