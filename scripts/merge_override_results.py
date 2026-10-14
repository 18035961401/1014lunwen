# -*- coding: utf-8 -*-
"""Merge rerun outputs into the main ablation CSV with override semantics.

Later sources win for the same (algorithm, model_setting, run_id) key.
The script writes a backup of the main file plus an audit CSV that records
which source ultimately supplied each key and which keys were overridden.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MAIN_FILE = RESULTS_DIR / "qvns_ablation" / "ablation_run_metrics.csv"
KEY_COLUMNS = ("algorithm", "model_setting", "run_id")
SOURCE_GLOBS = (
    "remaining_*",
    "qvns_ablation_remaining",
    "qvns_ablation_q*",
    "patch_*.csv",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge rerun results into the main CSV with override semantics.")
    parser.add_argument("--main-file", type=Path, default=DEFAULT_MAIN_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be overridden without writing files.",
    )
    return parser.parse_args(argv)


def make_key(row) -> tuple[str, str, int]:
    return (
        str(row["algorithm"]),
        str(row["model_setting"]),
        int(row["run_id"]),
    )


def collect_columns(dataframes: list[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    for df in dataframes:
        for column in df.columns:
            if column not in columns:
                columns.append(column)
    return columns


def merge_with_override(base_df: pd.DataFrame, overlay_sources: list[tuple[str, pd.DataFrame]]):
    records: dict[tuple[str, str, int], dict] = {}
    chosen_source: dict[tuple[str, str, int], str] = {}
    audit_rows: list[dict] = []
    all_frames = [base_df] + [df for _, df in overlay_sources]
    final_columns = collect_columns(all_frames)

    for row in base_df.to_dict("records"):
        key = make_key(row)
        records[key] = row
        chosen_source[key] = "main"

    for source_name, df in overlay_sources:
        for row in df.to_dict("records"):
            key = make_key(row)
            action = "override" if key in records else "insert"
            previous_source = chosen_source.get(key)
            records[key] = row
            chosen_source[key] = source_name
            audit_rows.append(
                {
                    "algorithm": key[0],
                    "model_setting": key[1],
                    "run_id": key[2],
                    "action": action,
                    "previous_source": previous_source,
                    "chosen_source": source_name,
                }
            )

    merged_df = pd.DataFrame(list(records.values()))
    for column in final_columns:
        if column not in merged_df.columns:
            merged_df[column] = pd.NA
    merged_df = merged_df.reindex(columns=final_columns)
    merged_df = merged_df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

    audit_df = pd.DataFrame(audit_rows)
    return merged_df, audit_df


def build_final_sources_df(merged_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame(columns=["algorithm", "model_setting", "run_id", "chosen_source"])

    latest_by_key: dict[tuple[str, str, int], str] = {}
    if not audit_df.empty:
        for row in audit_df.to_dict("records"):
            latest_by_key[(str(row["algorithm"]), str(row["model_setting"]), int(row["run_id"]))] = str(row["chosen_source"])

    rows = []
    for row in merged_df.to_dict("records"):
        key = make_key(row)
        rows.append(
            {
                "algorithm": key[0],
                "model_setting": key[1],
                "run_id": key[2],
                "chosen_source": latest_by_key.get(key, "main"),
            }
        )
    return pd.DataFrame(rows).sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def discover_overlay_sources(results_dir: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for pattern in SOURCE_GLOBS:
        for path in results_dir.glob(pattern):
            if path == results_dir / "qvns_ablation":
                continue
            if path.is_dir():
                csv_path = path / "ablation_run_metrics.csv"
                if csv_path.exists():
                    found[path.name] = csv_path
            elif path.is_file() and path.suffix.lower() == ".csv":
                found[path.stem] = path

    return sorted(found.items(), key=lambda item: (item[1].stat().st_mtime, item[0]))


def load_overlay_sources(results_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    sources = []
    for source_name, csv_path in discover_overlay_sources(results_dir):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        sources.append((source_name, df))
    return sources


def write_outputs(
    merged_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    final_sources_df: pd.DataFrame,
    main_file: Path,
):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_file = main_file.with_suffix(f".csv.backup_override_{timestamp}")
    shutil.copy2(main_file, backup_file)
    merged_df.to_csv(main_file, index=False, encoding="utf-8-sig")

    report_dir = main_file.parent
    audit_file = report_dir / f"override_audit_{timestamp}.csv"
    final_sources_file = report_dir / f"override_final_sources_{timestamp}.csv"
    audit_df.to_csv(audit_file, index=False, encoding="utf-8-sig")
    final_sources_df.to_csv(final_sources_file, index=False, encoding="utf-8-sig")
    return backup_file, audit_file, final_sources_file


def print_summary(base_df: pd.DataFrame, merged_df: pd.DataFrame, audit_df: pd.DataFrame, overlay_sources: list[tuple[str, pd.DataFrame]]):
    override_count = int((audit_df["action"] == "override").sum()) if not audit_df.empty else 0
    insert_count = int((audit_df["action"] == "insert").sum()) if not audit_df.empty else 0

    print("=" * 70)
    print("Override Merge Results")
    print("=" * 70)
    print(f"Main rows before merge: {len(base_df)}")
    print(f"Main rows after merge:  {len(merged_df)}")
    print(f"Inserted keys:          {insert_count}")
    print(f"Overridden keys:        {override_count}")
    print("")
    print("Source order (later wins on conflict):")
    for source_name, df in overlay_sources:
        print(f"  {source_name:24s} {len(df):4d} rows")
    print("")

    for algo in sorted(merged_df["algorithm"].astype(str).unique()):
        subset = merged_df[merged_df["algorithm"].astype(str) == algo]
        run_ids = {int(v) for v in subset["run_id"].tolist()}
        missing = [idx for idx in range(1, 21) if idx not in run_ids]
        status = "complete" if not missing else f"missing {missing}"
        print(f"  {algo:20s}: {len(subset):2d} rows, {status}")


def main(argv=None):
    args = parse_args(argv)
    main_file = args.main_file
    results_dir = args.results_dir

    if not main_file.exists():
        print(f"Main file not found: {main_file}")
        return 1

    base_df = pd.read_csv(main_file, encoding="utf-8-sig")
    overlay_sources = load_overlay_sources(results_dir)

    if not overlay_sources:
        print("No overlay sources found.")
        return 0

    merged_df, audit_df = merge_with_override(base_df, overlay_sources)
    final_sources_df = build_final_sources_df(merged_df, audit_df)
    print_summary(base_df, merged_df, audit_df, overlay_sources)

    if args.dry_run:
        print("")
        print("Dry run only. No files were written.")
        return 0

    backup_file, audit_file, final_sources_file = write_outputs(
        merged_df,
        audit_df,
        final_sources_df,
        main_file,
    )
    print("")
    print(f"Backup written:       {backup_file}")
    print(f"Audit written:        {audit_file}")
    print(f"Final source map:     {final_sources_file}")
    print(f"Merged main updated:  {main_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
