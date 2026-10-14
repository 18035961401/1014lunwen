# -*- coding: utf-8 -*-
"""Merge all 2pao subdirectories into a single comprehensive result."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "2pao"
OUTPUT_FILE = INPUT_ROOT / "merged_all_results.csv"
REPORT_DIR = INPUT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_ORDER = [
    "NSGA-II", "VNS-only", "Q-learning-only",
    "QVNS-Q3", "QVNS-Q9",
    "QVNS-Q9A", "QVNS-Q9B", "QVNS-Q9C", "QVNS-Q9D",
    "QVNS-Q11", "QVNS-Q11A", "QVNS-Q11B", "QVNS-Q11C",
]

METRICS = ["HV", "IGD", "Spread", "unique_points", "min_cost", "max_service_ratio", "runtime_seconds"]
METRIC_DIR = {"HV": 1, "IGD": -1, "Spread": -1, "unique_points": 1, "min_cost": -1, "max_service_ratio": 1, "runtime_seconds": -1}
METRIC_LABEL = {"HV": "HV", "IGD": "IGD", "Spread": "Spread", "unique_points": "UniquePts", "min_cost": "MinCost", "max_service_ratio": "MaxSvc", "runtime_seconds": "Runtime(s)"}


def find_all_csvs(root: Path):
    files = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            f = d / "ablation_run_metrics.csv"
            if f.exists():
                files.append((d.name, f))
    return files


def merge_results():
    all_parts = {}
    for dir_name, csv_path in find_all_csvs(INPUT_ROOT):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        all_parts[dir_name] = df
        print(f"  {dir_name}: {len(df)} rows")

    if not all_parts:
        print("No results found!")
        return None

    combined = pd.concat(all_parts.values(), ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nMerged: {len(combined)} rows -> {OUTPUT_FILE}")
    return combined


def comprehensive_stats(df):
    rows = []
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        if len(sub) == 0:
            continue
        row = {"Algorithm": algo, "Runs": len(sub)}
        for m in METRICS:
            v = sub[m].dropna()
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std()
            row[f"{m}_min"] = v.min()
            row[f"{m}_max"] = v.max()
            row[f"{m}_median"] = v.median()
        rows.append(row)

    stats = pd.DataFrame(rows)
    stats.to_csv(REPORT_DIR / "comprehensive_statistics.csv", index=False, encoding="utf-8-sig")

    # Also generate a summary markdown
    lines = []
    lines.append("# 2pao — 13 Algorithms × 20 Runs — Comprehensive Report")
    lines.append(f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Total runs**: {len(df)}")
    lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    header = "| Algorithm | Runs |" + "|".join(f" {METRIC_LABEL[m]} Mean±Std |" for m in METRICS) + "|"
    sep = "|" + "|".join("---" for _ in range(len(METRICS)+2))
    lines.append(header)
    lines.append(sep)
    for _, r in stats.iterrows():
        parts = [r["Algorithm"], str(r["Runs"])]
        for m in METRICS:
            parts.append(f"{r[f'{m}_mean']:.4f}±{r[f'{m}_std']:.4f}")
        lines.append("| " + " | ".join(parts) + " |")

    lines.append("")
    lines.append("## Best & Worst per Metric")
    lines.append("")
    for m in METRICS:
        direction = "higher" if METRIC_DIR[m] > 0 else "lower"
        best = stats.iloc[stats[f"{m}_mean"].argmax()] if METRIC_DIR[m] > 0 else stats.iloc[stats[f"{m}_mean"].argmin()]
        worst = stats.iloc[stats[f"{m}_mean"].argmin()] if METRIC_DIR[m] > 0 else stats.iloc[stats[f"{m}_mean"].argmax()]
        lines.append(f"**{METRIC_LABEL[m]}** ({direction} is better):")
        lines.append(f"- Best: {best['Algorithm']} = {best[f'{m}_mean']:.4f}")
        lines.append(f"- Worst: {worst['Algorithm']} = {worst[f'{m}_mean']:.4f}")
        lines.append("")

    lines.append("## Standard Deviation (Stability)")
    lines.append("")
    lines.append("| Algorithm |" + "|".join(f" {METRIC_LABEL[m]} CV |" for m in METRICS))
    lines.append("|" + "|".join("---" for _ in range(len(METRICS)+1)))
    for _, r in stats.iterrows():
        parts = [r["Algorithm"]]
        for m in METRICS:
            cv = r[f"{m}_std"] / abs(r[f"{m}_mean"]) if abs(r[f"{m}_mean"]) > 1e-9 else float("nan")
            parts.append(f"{cv:.4f}")
        lines.append("| " + " | ".join(parts) + " |")

    report_path = REPORT_DIR / "comprehensive_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")

    return stats


def main():
    print("=" * 60)
    print("2pao Merge & Report")
    print("=" * 60)

    df = merge_results()
    if df is None:
        return 1

    stats = comprehensive_stats(df)
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Merged CSV: {OUTPUT_FILE}")
    print(f"Report:     {REPORT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
