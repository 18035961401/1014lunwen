from pathlib import Path
import pandas as pd
import time

root = Path('2pao')
algo_map = {
    'nsga_ii': 'NSGA-II', 'vns_only': 'VNS-only', 'q_learning_only': 'Q-learning-only',
    'qvns_q3': 'QVNS-Q3', 'qvns_q9': 'QVNS-Q9',
    'qvns_q9a': 'QVNS-Q9A', 'qvns_q9b': 'QVNS-Q9B', 'qvns_q9c': 'QVNS-Q9C',
    'qvns_q9d': 'QVNS-Q9D', 'qvns_q11': 'QVNS-Q11',
    'qvns_q11a': 'QVNS-Q11A', 'qvns_q11b': 'QVNS-Q11B', 'qvns_q11c': 'QVNS-Q11C',
}

total = 0
for d in sorted(root.iterdir()):
    if d.is_dir():
        f = d / 'ablation_run_metrics.csv'
        if f.exists():
            df = pd.read_csv(str(f), encoding='utf-8-sig')
            cnt = len(df)
            total += cnt
            name = algo_map.get(d.name, d.name)
            status = 'done' if cnt >= 20 else f'{cnt}/20'
            print(f'{name:20s} [{d.name:20s}]: {status}')
        else:
            name = algo_map.get(d.name, d.name)
            print(f'{name:20s} [{d.name:20s}]: waiting')

print(f'\nTotal: {total} / 260 ({total/260*100:.1f}%)')
print(f'Time: {time.strftime("%H:%M:%S")}')
