import subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]

# 单独运行每个消融算法
commands = [
    ("nsga_ii", ["--comparison-mode", "ablation", "--ablation-algo", "NSGA-II"]),
    ("vns_only", ["--comparison-mode", "ablation", "--ablation-algo", "VNS-only"]),
    ("q_learning_only", ["--comparison-mode", "ablation", "--ablation-algo", "Q-learning-only"]),
]

procs = []
for name, extra_args in commands:
    d = f"2pao/{name}"
    cmd = [
        sys.executable,
        "scripts/run_qvns_ablation_experiment.py",
        *extra_args,
        "--pop-size", "100",
        "--max-gen", "300",
        "--runs", "20",
        "--no-resume",
        "--output-dir", d,
        "--quiet",
    ]
    print(f"Launching {name} -> {d}")
    p = subprocess.Popen(cmd, cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE)
    procs.append((name, p))
    time.sleep(2)

print("\nAll 3 ablation algorithms launched. Waiting...")
for name, p in procs:
    p.wait()
    print(f"  {name} done!")

print("\nAblation algorithms complete!")
