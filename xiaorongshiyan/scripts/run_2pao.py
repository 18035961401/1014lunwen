# -*- coding: utf-8 -*-
"""Complete re-run of 13 algorithms × 20 runs, parallel in 2 batches, output to 2pao/."""
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "2pao"

ALL_ALGORITHMS = [
    ("NSGA-II",           "ablation",  None),
    ("VNS-only",          "ablation",  None),
    ("Q-learning-only",   "ablation",  None),
    ("QVNS-Q3",           "tuning",    "Q3"),
    ("QVNS-Q9",           "tuning",    "Q9"),
    ("QVNS-Q9A",          "tuning",    "Q9A"),
    ("QVNS-Q9B",          "tuning",    "Q9B"),
    ("QVNS-Q9C",          "tuning",    "Q9C"),
    ("QVNS-Q9D",          "tuning",    "Q9D"),
    ("QVNS-Q11",          "tuning",    "Q11"),
    ("QVNS-Q11A",         "tuning",    "Q11A"),
    ("QVNS-Q11B",         "tuning",    "Q11B"),
    ("QVNS-Q11C",         "tuning",    "Q11C"),
]

POP_SIZE = 100
MAX_GEN = 300
RUNS = 20
BATCH_SIZE = 8  # match CPU cores


def run_batch(batch_algorithms, batch_num):
    print(f"\n{'=' * 60}")
    print(f"Batch {batch_num}: {len(batch_algorithms)} algorithms")
    print(f"{'=' * 60}")

    processes = []
    for algo_name, mode, tuning_key in batch_algorithms:
        safe_name = algo_name.replace("-", "_").lower()
        out_dir = OUTPUT_ROOT / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == "ablation":
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_qvns_ablation_experiment.py"),
                "--comparison-mode", "ablation",
                "--pop-size", str(POP_SIZE),
                "--max-gen", str(MAX_GEN),
                "--runs", str(RUNS),
                "--output-dir", str(out_dir),
                "--no-resume",
                "--quiet",
            ]
        else:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_qvns_ablation_experiment.py"),
                "--comparison-mode", "tuning",
                "--qvns-tuning-groups", tuning_key,
                "--pop-size", str(POP_SIZE),
                "--max-gen", str(MAX_GEN),
                "--runs", str(RUNS),
                "--output-dir", str(out_dir),
                "--no-resume",
                "--quiet",
            ]

        print(f"  [{algo_name}] -> {out_dir}")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        processes.append((algo_name, proc))
        time.sleep(2)

    print(f"\nWaiting for Batch {batch_num} to complete...")
    print("(This may take 40-60 minutes per batch)")
    for name, proc in processes:
        proc.wait()
        print(f"  [{name}] done!")

    print(f"\nBatch {batch_num} complete!")
    return True


def main():
    print("=" * 60)
    print("2pao: Full Re-run — 13 algorithms × 20 runs")
    print(f"CPU Cores: 8, Batch Size: {BATCH_SIZE}")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    batch1 = ALL_ALGORITHMS[:8]
    batch2 = ALL_ALGORITHMS[8:]

    t0 = time.time()
    run_batch(batch1, 1)
    run_batch(batch2, 2)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"All experiments complete! Total time: {elapsed/60:.1f} min")
    print(f"Results in: {OUTPUT_ROOT}")
    print("=" * 60)
    print("\nNext: run scripts/merge_2pao.py to combine all results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
