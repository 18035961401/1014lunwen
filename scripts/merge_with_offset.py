# -*- coding: utf-8 -*-
"""Merge remaining results with run_id offset to avoid conflicts."""
import pandas as pd
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "results" / "qvns_ablation" / "ablation_run_metrics.csv"

REMAINING_DIRS = [
    "remaining_q9a", "remaining_q9b", "remaining_q9c", "remaining_q9d",
    "remaining_q11", "remaining_q11a", "remaining_q11b", "remaining_q11c"
]

print("=" * 70)
print("合并 remaining_* 结果（run_id +10 偏移）到主文件")
print("=" * 70)

df_main = pd.read_csv(MAIN_FILE, encoding='utf-8-sig')
print(f"主文件当前记录数: {len(df_main)}")

all_new_records = []

for dir_name in REMAINING_DIRS:
    csv_path = PROJECT_ROOT / "results" / dir_name / "ablation_run_metrics.csv"
    if not csv_path.exists():
        print(f"\n{dir_name}: 文件不存在，跳过")
        continue
    
    df_remaining = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"\n{dir_name}: {len(df_remaining)} 条记录")
    
    # Offset run_id by +10
    df_remaining['run_id'] = df_remaining['run_id'] + 10
    
    # Check which records are new (not in main file)
    new_count = 0
    for _, row in df_remaining.iterrows():
        algo = row['algorithm']
        run_id = row['run_id']
        model_setting = row['model_setting']
        
        # Check if exists in main
        exists = df_main[
            (df_main['algorithm'] == algo) & 
            (df_main['run_id'] == run_id) &
            (df_main['model_setting'] == model_setting)
        ]
        
        if len(exists) == 0:
            all_new_records.append(row)
            new_count += 1
    
    print(f"  -> 新记录（偏移后）: {new_count} 条")

print(f"\n总共发现新记录: {len(all_new_records)} 条")

if len(all_new_records) > 0:
    df_new = pd.DataFrame(all_new_records)
    df_merged = pd.concat([df_main, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(['algorithm', 'model_setting', 'run_id']).reset_index(drop=True)
    
    # Backup
    backup = str(MAIN_FILE) + f'.backup_offset_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(MAIN_FILE, backup)
    print(f"已备份原文件到: {backup}")
    
    # Save
    df_merged.to_csv(MAIN_FILE, index=False, encoding='utf-8-sig')
    print(f"已保存合并结果到: {MAIN_FILE}")
    print(f"合并前: {len(df_main)} -> 合并后: {len(df_merged)}")
    df_main = df_merged
else:
    print("没有新记录需要合并")

# Print final status
print("\n" + "=" * 70)
print("各算法完成情况:")
print("=" * 70)

total_remaining = 0
for algo in sorted(df_main['algorithm'].unique()):
    subset = df_main[df_main['algorithm'] == algo]
    count = len(subset)
    run_ids = sorted(subset['run_id'].unique().tolist())
    missing = [i for i in range(1, 21) if i not in run_ids]
    remaining = len(missing)
    total_remaining += remaining
    
    status = "完成" if count >= 20 else f"还需 {remaining} 轮"
    print(f"  {algo:20s}: {count:2d}/20 {status}")
    if missing:
        print(f"    缺失run_id: {missing}")

print(f"\n总剩余轮数: {total_remaining}")
print("=" * 70)
