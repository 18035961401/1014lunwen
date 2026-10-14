# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "merge_override_results.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("merge_override_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MergeOverrideResultsTests(unittest.TestCase):
    def test_later_source_overrides_same_algorithm_model_and_run(self):
        module = load_script_module()

        base = pd.DataFrame(
            [
                {"algorithm": "A", "model_setting": "m1", "run_id": 1, "HV": 0.10, "IGD": 0.90},
                {"algorithm": "A", "model_setting": "m1", "run_id": 2, "HV": 0.20, "IGD": 0.80},
            ]
        )
        overlay_old = pd.DataFrame(
            [
                {"algorithm": "A", "model_setting": "m1", "run_id": 1, "HV": 0.30, "IGD": 0.70},
                {"algorithm": "A", "model_setting": "m1", "run_id": 3, "HV": 0.40, "IGD": 0.60},
            ]
        )
        overlay_new = pd.DataFrame(
            [
                {"algorithm": "A", "model_setting": "m1", "run_id": 1, "HV": 0.50, "IGD": 0.50},
            ]
        )

        merged, audit = module.merge_with_override(
            base,
            [
                ("overlay_old", overlay_old),
                ("overlay_new", overlay_new),
            ],
        )

        row_run1 = merged[(merged["algorithm"] == "A") & (merged["model_setting"] == "m1") & (merged["run_id"] == 1)].iloc[0]
        row_run2 = merged[(merged["algorithm"] == "A") & (merged["model_setting"] == "m1") & (merged["run_id"] == 2)].iloc[0]
        row_run3 = merged[(merged["algorithm"] == "A") & (merged["model_setting"] == "m1") & (merged["run_id"] == 3)].iloc[0]

        self.assertAlmostEqual(row_run1["HV"], 0.50)
        self.assertAlmostEqual(row_run1["IGD"], 0.50)
        self.assertAlmostEqual(row_run2["HV"], 0.20)
        self.assertAlmostEqual(row_run3["HV"], 0.40)

        final_run1_audit = audit[(audit["algorithm"] == "A") & (audit["model_setting"] == "m1") & (audit["run_id"] == 1)].iloc[-1]
        final_run3_audit = audit[(audit["algorithm"] == "A") & (audit["model_setting"] == "m1") & (audit["run_id"] == 3)].iloc[-1]

        self.assertEqual(final_run1_audit["chosen_source"], "overlay_new")
        self.assertEqual(final_run1_audit["action"], "override")
        self.assertEqual(final_run3_audit["chosen_source"], "overlay_old")
        self.assertEqual(final_run3_audit["action"], "insert")


if __name__ == "__main__":
    unittest.main()
