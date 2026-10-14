# -*- coding: utf-8 -*-
"""Run exactly the missing experiments and merge into main CSV."""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import (
    _normalize_model_setting, build_metro_logistics_instance,
    run_qvns_nsga2,
)
from run_qvns_ablation_experiment import (
    summarize_result, add_quality_indicators, _qvns_config,
)

# Missing runs: (algorithm_name, tuning_group, missing_run_ids)
MISSING_RUNS = {
    "QVNS-Q9A":  ([20],      "Q9A"),
    "QVNS-Q9B":  ([18,19,20],"Q9B"),
    "QVNS-Q9C":  ([17,18,19,20],"Q9C"),
    "QVNS-Q9D":  ([17,18,19,20],"Q9D"),
    "QVNS-Q11":  ([15,16,17,18,19,20],"Q11"),
    "QVNS-Q11A": ([18,19,20],"Q11A"),
    "QVNS-Q11B": ([16,17,18,19,20],"Q11B"),
    "QVNS-Q11C": ([18,19,20],"Q11C"),
}

BASE_SEED = 20260506
POP_SIZE = 100
MAX_GEN = 300
MODEL_SETTING = "no_budget_range"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, help="Algorithm name, e.g. QVNS-Q9A")
    args_parsed = parser.parse_args()

    algo_name = args_parsed.algorithm
    if algo_name not in MISSING_RUNS:
        print(f"Unknown algorithm: {algo_name}")
        return 1

    run_ids, tuning_key = MISSING_RUNS[algo_name]

    # Build instance once
    inst = build_metro_logistics_instance(
        "base", top_n=17,
        model_setting=_normalize_model_setting(MODEL_SETTING)
    )

    # Build config
    class FakeArgs:
        vns_max_elites = 20
        vns_trials = 3
        max_gen = MAX_GEN
        pop_size = POP_SIZE

    kwargs = _qvns_config(tuning_key, FakeArgs())

    output_file = PROJECT_ROOT / "results" / f"patch_{algo_name.lower().replace('-','_')}.csv"
    
    rows = []
    for run_id in run_ids:
        seed = BASE_SEED + 0 + run_id - 1  # scenario_idx=0
        print(f"[{algo_name}] run_id={run_id}, seed={seed}")

        started = time.time()
        result = run_qvns_nsga2(
            inst, pop_size=POP_SIZE, max_gen=MAX_GEN, seed=seed,
            verbose=False, archive_size=kwargs.pop("archive_size", int(2.0 * POP_SIZE)),
            **kwargs
        )
        row, _ = summarize_result(
            algo_name, MODEL_SETTING, run_id, seed, result,
            time.time() - started, group_type="qvns_tuning", config=kwargs
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} runs to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
