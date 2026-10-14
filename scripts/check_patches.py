from pathlib import Path
for algo in ['patch_qvns_q9a.csv', 'patch_qvns_q9b.csv']:
    p = Path('results') / algo
    if p.exists():
        import pandas as pd
        df = pd.read_csv(str(p), encoding='utf-8-sig')
        print(f'{algo}: {len(df)} rows')
    else:
        print(f'{algo}: not yet')
