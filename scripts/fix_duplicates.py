import pandas as pd
from pathlib import Path

root = Path('2pao')
problem_dirs = ['nsga_ii', 'vns_only', 'q_learning_only']

for dname in problem_dirs:
    d = root / dname
    f = d / 'ablation_run_metrics.csv'
    if f.exists():
        df = pd.read_csv(str(f), encoding='utf-8-sig')
        print(f"Before: {dname} has {len(df)} rows")
        
        df_unique = df.drop_duplicates(subset='run_id', keep='first')
        df_unique.to_csv(str(f), index=False, encoding='utf-8-sig')
        print(f"After: {dname} has {len(df_unique)} rows (kept unique run_ids)")
    else:
        print(f"Warning: {f} not found")

print("\nDone! All files cleaned to 20 unique runs each.")
