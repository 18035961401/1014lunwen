import pandas as pd
from pathlib import Path

root = Path('2pao')
df_main = pd.read_csv('results/qvns_ablation/ablation_run_metrics.csv', encoding='utf-8-sig')

algo_map = {
    'nsga_ii': 'NSGA-II',
    'vns_only': 'VNS-only',
    'q_learning_only': 'Q-learning-only'
}

for dname, algo_name in algo_map.items():
    dest = root / dname
    dest.mkdir(parents=True, exist_ok=True)
    df_algo = df_main[df_main['algorithm'] == algo_name].copy()
    df_algo.to_csv(dest / 'ablation_run_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"Restored {algo_name}: {len(df_algo)} rows -> {dest}")

print("\nDone!")
