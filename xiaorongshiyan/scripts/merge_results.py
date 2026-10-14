# -*- coding: utf-8 -*-
"""Merge parallel experiment results into the main CSV file."""
import pandas as pd
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MAIN_FILE = RESULTS_DIR / "qvns_ablation" / "ablation_run_metrics.csv"

def merge_results():
    """Merge all parallel results into the main file."""
    print("=" * 60)
    print("合并并行实验结果到主文件")
    print("=" * 60)
    
    # Read main file
    if not MAIN_FILE.exists():
        print(f"错误: 主文件不存在: {MAIN_FILE}")
        return 1
    
    df_main = pd.read_csv(MAIN_FILE, encoding="utf-8-sig")
    print(f"主文件记录数: {len(df_main)}")
    
    # Find all parallel result directories
    parallel_dirs = [
        "remaining_q9a",
        "remaining_q9b",
        "remaining_q9c",
        "remaining_q9d",
        "remaining_q11",
        "remaining_q11a",
        "remaining_q11b",
        "remaining_q11c",
        # Also check old directories for compatibility
        "qvns_ablation_q9a",
        "qvns_ablation_q9b",
        "qvns_ablation_q9c",
        "qvns_ablation_q9d",
        "qvns_ablation_q11",
        "qvns_ablation_q11a",
        "qvns_ablation_q11b",
        "qvns_ablation_q11c",
        "qvns_ablation_remaining"
    ]
    
    all_new_rows = []
    
    for dir_name in parallel_dirs:
        dir_path = RESULTS_DIR / dir_name
        csv_file = dir_path / "ablation_run_metrics.csv"
        
        if not csv_file.exists():
            continue
        
        df_parallel = pd.read_csv(csv_file, encoding="utf-8-sig")
        print(f"\n读取 {dir_name}: {len(df_parallel)} 条记录")
        
        # Filter only records not in main file
        new_rows_count = 0
        for _, row in df_parallel.iterrows():
            algo = row['algorithm']
            run_id = row['run_id']
            model_setting = row['model_setting']
            
            # Check if this record already exists in main file
            exists = df_main[
                (df_main['algorithm'] == algo) & 
                (df_main['run_id'] == run_id) & 
                (df_main['model_setting'] == model_setting)
            ]
            
            if len(exists) == 0:
                all_new_rows.append(row)
                new_rows_count += 1
        
        print(f"  -> 其中新记录: {new_rows_count} 条")
    
    print(f"\n发现新记录总数: {len(all_new_rows)} 条")
    
    if len(all_new_rows) == 0:
        print("没有新记录需要合并")
        return 0
    
    # Add new rows to main dataframe
    df_new = pd.DataFrame(all_new_rows)
    df_merged = pd.concat([df_main, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(['algorithm', 'model_setting', 'run_id']).reset_index(drop=True)
    
    # Backup main file
    backup_file = MAIN_FILE.with_suffix(f".csv.backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(MAIN_FILE, backup_file)
    print(f"\n已备份原文件到: {backup_file}")
    
    # Save merged file
    df_merged.to_csv(MAIN_FILE, index=False, encoding="utf-8-sig")
    print(f"已保存合并结果到: {MAIN_FILE}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("合并完成!")
    print(f"合并前: {len(df_main)} 条")
    print(f"新增:   {len(all_new_rows)} 条")
    print(f"合并后: {len(df_merged)} 条")
    print("=" * 60)
    
    # Print completion status
    print("\n各算法完成情况:")
    for algo in sorted(df_merged['algorithm'].unique()):
        count = len(df_merged[df_merged['algorithm'] == algo])
        status = "完成" if count >= 20 else f"还需 {20-count} 轮"
        print(f"  {algo:20s}: {count:2d}/20 {status}")
    
    remaining = sum(max(0, 20 - len(df_merged[df_merged['algorithm'] == algo])) 
                   for algo in df_merged['algorithm'].unique())
    print(f"\n总剩余轮数: {remaining}")
    
    return 0

if __name__ == "__main__":
    sys.exit(merge_results())
