# -*- coding: utf-8 -*-
"""Launch parallel experiments for remaining runs using resume feature.

Each algorithm group runs in its own process with a separate output directory.
The resume feature automatically skips already completed runs.
After all processes finish, run merge_results.py to combine into the main file.
"""
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Algorithm groups that still need runs (based on current progress)
# Format: (display_name, tuning_group, output_subdirectory)
ALGORITHMS = [
    ("QVNS-Q9A", "Q9A", "remaining_q9a"),
    ("QVNS-Q9B", "Q9B", "remaining_q9b"),
    ("QVNS-Q9C", "Q9C", "remaining_q9c"),
    ("QVNS-Q9D", "Q9D", "remaining_q9d"),
    ("QVNS-Q11", "Q11", "remaining_q11"),
    ("QVNS-Q11A", "Q11A", "remaining_q11a"),
    ("QVNS-Q11B", "Q11B", "remaining_q11b"),
    ("QVNS-Q11C", "Q11C", "remaining_q11c"),
]


def main():
    print("=" * 60)
    print("启动8个并行进程运行剩余实验")
    print("利用resume功能自动跳过已完成的轮次")
    print("每个进程输出到独立目录，避免写入冲突")
    print("=" * 60)
    
    processes = []
    
    for name, group, subdir in ALGORITHMS:
        output_dir = f"results/{subdir}"
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
            # Note: NOT using --no-resume, so it will skip existing runs
        ]
        
        # Start process in new console window
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        processes.append((name, proc, output_dir))
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("已启动8个并行进程！")
    print("每个进程利用resume功能，自动跳过已完成的轮次")
    print("独立输出目录:")
    for name, _, output_dir in ALGORITHMS:
        print(f"  {name:15s} -> {output_dir}")
    print("=" * 60)
    
    # Wait for all processes to complete
    print("\n等待所有进程完成...")
    print("(这可能需要30-60分钟)")
    
    for name, proc, _ in processes:
        proc.wait()
        print(f"  {name} 完成!")
    
    print("\n" + "=" * 60)
    print("所有进程已完成！")
    print("\n现在运行合并脚本将结果汇总到主文件:")
    print("  python scripts/merge_results.py")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
