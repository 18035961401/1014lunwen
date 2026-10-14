# -*- coding: utf-8 -*-
"""Launch parallel experiments with separate output files for each algorithm."""
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Algorithm groups to run with their output directories
ALGORITHMS = [
    ("QVNS-Q9A", "Q9A", "results/qvns_ablation_q9a"),
    ("QVNS-Q9B", "Q9B", "results/qvns_ablation_q9b"),
    ("QVNS-Q9C", "Q9C", "results/qvns_ablation_q9c"),
    ("QVNS-Q9D", "Q9D", "results/qvns_ablation_q9d"),
    ("QVNS-Q11", "Q11", "results/qvns_ablation_q11"),
    ("QVNS-Q11A", "Q11A", "results/qvns_ablation_q11a"),
    ("QVNS-Q11B", "Q11B", "results/qvns_ablation_q11b"),
    ("QVNS-Q11C", "Q11C", "results/qvns_ablation_q11c"),
]


def main():
    print("=" * 60)
    print("启动8个并行进程运行剩余实验")
    print("每个进程独立输出到不同文件")
    print("=" * 60)
    
    processes = []
    
    for name, group, output_dir in ALGORITHMS:
        print(f"启动 {name} -> {output_dir} ...")
        
        cmd = [
            sys.executable,
            "scripts/run_qvns_ablation_experiment.py",
            "--comparison-mode", "tuning",
            "--qvns-tuning-groups", group,
            "--pop-size", "100",
            "--max-gen", "300",
            "--runs", "20",
            "--output-dir", output_dir,
            "--quiet"
        ]
        
        # Start process
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        processes.append((name, proc))
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("已启动8个并行进程！")
    print("每个进程输出到独立目录:")
    for name, _, output_dir in ALGORITHMS:
        print(f"  {name:15s} -> {output_dir}")
    print("=" * 60)
    
    # Wait for all processes to complete
    print("\n等待所有进程完成...")
    for name, proc in processes:
        proc.wait()
        print(f"  {name} 完成!")
    
    print("\n所有进程已完成！")
    print("请运行 merge_results.py 合并结果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
