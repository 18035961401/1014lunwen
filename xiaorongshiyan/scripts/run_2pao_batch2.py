import subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
algorithms = [
    ("QVNS-Q9D",  "Q9D"),
    ("QVNS-Q11",  "Q11"),
    ("QVNS-Q11A", "Q11A"),
    ("QVNS-Q11B", "Q11B"),
    ("QVNS-Q11C", "Q11C"),
]

procs = []
for name, group in algorithms:
    d = f"2pao/{name.replace('-','_').lower()}"
    print(f"Launching {name} -> {d}")
    p = subprocess.Popen([
        sys.executable,
        "scripts/run_qvns_ablation_experiment.py",
        "--comparison-mode", "tuning",
        "--qvns-tuning-groups", group,
        "--pop-size", "100",
        "--max-gen", "300",
        "--runs", "20",
        "--no-resume",
        "--output-dir", d,
        "--quiet",
    ], cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE)
    procs.append((name, p))
    time.sleep(2)

print("\nAll 5 launched. Waiting...")
for name, p in procs:
    p.wait()
    print(f"  {name} done!")

print("\nBatch 2 complete!")
