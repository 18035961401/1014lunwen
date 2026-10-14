# -*- coding: utf-8 -*-
"""Run remaining experiments in parallel, output to qvns_ablation_remaining."""
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Algorithm groups and their remaining runs
ALGORITHMS = [
    ("QVNS-Q9A", "Q9A"),
    ("QVNS-Q9B", "Q9B"),
    ("QVNS-Q9C", "Q9C"),
    ("QVNS-Q9D", "Q9D"),
    ("QVNS-Q11", "Q11"),
    ("QVNS-Q11A", "Q11A"),
    ("QVNS-Q11B", "Q11B"),
    ("QVNS-Q11C", "Q11C"),
]


def main():
    print("=" * 60)
    print("并行运行剩余实验")
    print("输出到: results/qvns_ablation_remaining")
    print("=" * 60)
    
    processes = []
    
    for name, group in ALGORITHMS:
        print(f"启动 {name} ...")
        
        cmd = [
            sys.executable,
            "scripts/run_qvns_ablation_experiment.py",
            "--comparison-mode", "tuning",
            "--qvns-tuning-groups", group,
            "--pop-size", "100",
            "--max-gen", "300",
            "--runs", "20",
            "--output-dir", "results/qvns_ablation_remaining",
            "--no-resume",  # Force rerun all 20 runs
            "--quiet"
        ]
        
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        processes.append((name, proc))
        time.sleep(2)
    
    print("\n已启动8个并行进程！")
    print("等待所有进程完成...")
    
    for name, proc in processes:
        proc.wait()
        print(f"  {name} 完成!")
    
    print("\n所有进程已完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
