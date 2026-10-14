# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "run_moo_comparison_experiment.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_moo_comparison_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MooComparisonAlgorithmTests(unittest.TestCase):
    def test_script_exports_required_algorithm_interfaces(self):
        module = load_script_module()
        self.assertTrue(callable(module.run_mopso))
        self.assertTrue(callable(module.run_moead))
        self.assertIn("MOPSO", module.COMPARISON_ALGORITHMS)
        self.assertIn("MOEA/D", module.COMPARISON_ALGORITHMS)

    def test_empty_metric_row_has_required_fields(self):
        module = load_script_module()
        row = module.empty_run_metric_row("MOPSO", 1, 20260508, runtime_seconds=0.5)
        for column in module.RUN_METRIC_COLUMNS:
            self.assertIn(column, row)
        self.assertEqual(row["algorithm"], "MOPSO")
        self.assertEqual(row["run_id"], 1)


if __name__ == "__main__":
    unittest.main()
