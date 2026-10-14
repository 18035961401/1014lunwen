# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "plot_algorithm_performance_metrics.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("plot_algorithm_performance_metrics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AlgorithmPerformanceMetricTests(unittest.TestCase):
    def test_metric_functions_return_finite_values(self):
        module = load_script_module()
        points = np.array([[0.2, 0.8], [0.4, 0.5], [0.7, 0.2]], dtype=float)
        reference = np.array([[0.1, 0.9], [0.3, 0.6], [0.6, 0.3]], dtype=float)

        self.assertGreaterEqual(module.compute_hypervolume_2d(points, (1.1, 1.1)), 0.0)
        self.assertTrue(np.isfinite(module.compute_igd(points, reference)))
        self.assertTrue(np.isfinite(module.compute_spacing(points)))
        self.assertTrue(np.isfinite(module.compute_spread(points)))

    def test_build_run_metrics_and_plot_outputs(self):
        module = load_script_module()
        pareto = pd.DataFrame(
            {
                "algorithm": ["A", "A", "B", "B"],
                "algorithm_label": ["A", "A", "B", "B"],
                "run_id": [1, 1, 1, 1],
                "Z1_total_cost": [100.0, 120.0, 110.0, 130.0],
                "Z2_metro_service_ratio": [0.5, 0.7, 0.4, 0.6],
                "CV": [0.0, 0.0, 0.0, 0.0],
            }
        )
        summary = pd.DataFrame(
            {
                "algorithm": ["A", "B"],
                "run_id": [1, 1],
                "runtime_seconds": [10.0, 12.0],
            }
        )
        run_metrics, summary_metrics = module.build_performance_tables(pareto, summary)
        self.assertIn("HV", run_metrics.columns)
        self.assertIn("IGD", run_metrics.columns)
        self.assertIn("runtime_seconds", run_metrics.columns)
        self.assertIn("HV_mean", summary_metrics.columns)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            module.plot_performance_metrics(summary_metrics, output_dir)
            self.assertTrue((output_dir / "fig_algorithm_performance_metrics.png").exists())

    def test_plot_stability_distribution_outputs_publication_files(self):
        module = load_script_module()
        run_metrics = pd.DataFrame(
            {
                "algorithm": ["QVNS-NSGA2", "QVNS-NSGA2", "NSGA2", "NSGA2"],
                "algorithm_label": ["QVNS-NSGA-II", "QVNS-NSGA-II", "NSGA-II", "NSGA-II"],
                "HV": [0.9, 0.85, 0.82, 0.78],
                "IGD": [0.12, 0.14, 0.15, 0.17],
                "Spacing": [0.01, 0.02, 0.02, 0.03],
                "Spread": [1.2, 1.3, 1.4, 1.5],
                "min_total_cost_10k": [19.5, 19.7, 19.9, 20.1],
                "max_service_percent": [52.0, 48.0, 45.0, 42.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            module.plot_stability_distribution(run_metrics, output_dir, prefix="test")
            self.assertTrue((output_dir / "test_stability_distribution.png").exists())
            self.assertTrue((output_dir / "test_stability_distribution.pdf").exists())


if __name__ == "__main__":
    unittest.main()
