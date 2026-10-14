# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "run_four_algorithm_convergence.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_four_algorithm_convergence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FourAlgorithmConvergenceTests(unittest.TestCase):
    def test_extract_generation_metrics_uses_real_carbon_cost(self):
        module = load_script_module()
        population = np.array([[1, 0, 0], [0, 1, 0]])
        objectives = np.array([[100.0, -0.4], [80.0, -0.3]])
        cvs = np.array([0.0, 0.0])

        def fake_evaluate_solution(decision, inst, use_repair=True, collect_parts=True):
            if np.array_equal(decision, population[0]):
                return np.array([100.0, -0.4]), 0.0, {"carbon_cost": 12.5}
            return np.array([80.0, -0.3]), 0.0, {"carbon_cost": 9.75}

        with patch.object(module, "evaluate_solution", side_effect=fake_evaluate_solution):
            row = module.extract_generation_metrics(
                algorithm="NSGA2",
                run_id=1,
                seed=20260512,
                gen=0,
                population=population,
                objectives=objectives,
                cvs=cvs,
                inst=object(),
            )

        self.assertEqual(row["total_cost"], 80.0)
        self.assertEqual(row["carbon_cost"], 9.75)
        self.assertEqual(row["service_ratio"], 0.3)
        self.assertEqual(row["feasible_count"], 2)

    def test_extract_generation_metrics_tracks_min_carbon_cost_separately(self):
        module = load_script_module()
        population = np.array([[1, 0, 0], [0, 1, 0]])
        objectives = np.array([[100.0, -0.4], [80.0, -0.3]])
        cvs = np.array([0.0, 0.0])

        def fake_evaluate_solution(decision, inst, use_repair=True, collect_parts=True):
            if np.array_equal(decision, population[0]):
                return np.array([100.0, -0.4]), 0.0, {"carbon_cost": 5.0}
            return np.array([80.0, -0.3]), 0.0, {"carbon_cost": 9.75}

        with patch.object(module, "evaluate_solution", side_effect=fake_evaluate_solution):
            row = module.extract_generation_metrics(
                algorithm="NSGA2",
                run_id=1,
                seed=20260512,
                gen=0,
                population=population,
                objectives=objectives,
                cvs=cvs,
                inst=object(),
            )

        self.assertEqual(row["total_cost"], 80.0)
        self.assertEqual(row["carbon_cost"], 5.0)
        self.assertEqual(row["carbon_cost_at_min_total_cost"], 9.75)

    def test_extract_generation_metrics_records_pareto_front_mean_values(self):
        module = load_script_module()
        population = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        objectives = np.array([[100.0, -0.5], [90.0, -0.4], [120.0, -0.3]])
        cvs = np.array([0.0, 0.0, 0.0])

        def fake_evaluate_solution(decision, inst, use_repair=True, collect_parts=True):
            if np.array_equal(decision, population[0]):
                return np.array([100.0, -0.5]), 0.0, {"carbon_cost": 10.0}
            if np.array_equal(decision, population[1]):
                return np.array([90.0, -0.4]), 0.0, {"carbon_cost": 8.0}
            return np.array([120.0, -0.3]), 0.0, {"carbon_cost": 12.0}

        with patch.object(module, "evaluate_solution", side_effect=fake_evaluate_solution):
            row = module.extract_generation_metrics(
                algorithm="NSGA2",
                run_id=1,
                seed=20260512,
                gen=0,
                population=population,
                objectives=objectives,
                cvs=cvs,
                inst=object(),
            )

        self.assertAlmostEqual(row["pareto_mean_total_cost"], 95.0)
        self.assertAlmostEqual(row["pareto_mean_service_ratio"], 0.45)

    def test_cost_improvement_rate_uses_initial_generation(self):
        module = load_script_module()
        self.assertAlmostEqual(module.compute_cost_improvement_rate(100.0, 75.0), 25.0)
        self.assertEqual(module.compute_cost_improvement_rate(0.0, 75.0), 0.0)

    def test_plot_thesis_combined_convergence_writes_expected_files(self):
        module = load_script_module()
        aggregated = pd.DataFrame(
            {
                "algorithm": ["QVNS-NSGA2", "QVNS-NSGA2", "NSGA2", "NSGA2"],
                "algorithm_label": ["QVNS-NSGA-II", "QVNS-NSGA-II", "NSGA-II", "NSGA-II"],
                "gen": [0, 1, 0, 1],
                "total_cost_mean": [280000.0, 220000.0, 285000.0, 230000.0],
                "total_cost_std": [0.0, 0.0, 0.0, 0.0],
                "carbon_cost_mean": [15.0, 10.6, 15.2, 11.3],
                "carbon_cost_std": [0.0, 0.0, 0.0, 0.0],
                "hypervolume_mean": [0.64, 0.90, 0.62, 0.86],
                "hypervolume_std": [0.0, 0.0, 0.0, 0.0],
                "cost_improvement_rate_mean": [0.0, 21.0, 0.0, 19.0],
                "cost_improvement_rate_std": [0.0, 0.0, 0.0, 0.0],
                "feasible_count_mean": [1, 1, 1, 1],
                "feasible_count_std": [0, 0, 0, 0],
                "pareto_size_mean": [1, 1, 1, 1],
                "pareto_size_std": [0, 0, 0, 0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            module.plot_thesis_combined_convergence(aggregated, output_dir)
            self.assertTrue((output_dir / "fig_algorithm_convergence_comparison.png").exists())
            self.assertTrue((output_dir / "fig_algorithm_convergence_comparison.pdf").exists())

    def test_plot_dual_axis_iteration_effects_writes_one_file_per_algorithm(self):
        module = load_script_module()
        aggregated = pd.DataFrame(
            {
                "algorithm": ["QVNS-NSGA2", "QVNS-NSGA2", "NSGA2", "NSGA2"],
                "algorithm_label": ["QVNS-NSGA-II", "QVNS-NSGA-II", "NSGA-II", "NSGA-II"],
                "gen": [0, 1, 0, 1],
                "total_cost_mean": [280000.0, 220000.0, 285000.0, 230000.0],
                "total_cost_std": [0.0, 0.0, 0.0, 0.0],
                "service_ratio_mean": [0.55, 0.72, 0.50, 0.68],
                "service_ratio_std": [0.0, 0.0, 0.0, 0.0],
                "pareto_mean_total_cost_mean": [282000.0, 225000.0, 287000.0, 233000.0],
                "pareto_mean_total_cost_std": [0.0, 0.0, 0.0, 0.0],
                "pareto_mean_service_ratio_mean": [0.54, 0.70, 0.49, 0.66],
                "pareto_mean_service_ratio_std": [0.0, 0.0, 0.0, 0.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            module.plot_dual_axis_iteration_effects(aggregated, output_dir)
            self.assertTrue((output_dir / "fig_iteration_effect_qvns_nsga2.png").exists())
            self.assertTrue((output_dir / "fig_iteration_effect_nsga2.png").exists())

    def test_plot_pareto_front_comparison_writes_expected_files(self):
        module = load_script_module()
        points = pd.DataFrame(
            {
                "algorithm": ["QVNS-NSGA2", "QVNS-NSGA2", "NSGA2", "NSGA2"],
                "run_id": [1, 1, 1, 1],
                "solution_id": [0, 1, 0, 1],
                "Z1_total_cost": [210000.0, 220000.0, 215000.0, 230000.0],
                "Z2_metro_service_ratio": [0.35, 0.50, 0.30, 0.48],
                "CV": [0.0, 0.0, 0.0, 0.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            module.plot_pareto_front_comparison(points, output_dir)
            self.assertTrue((output_dir / "fig_pareto_front_cost_service.png").exists())
            self.assertTrue((output_dir / "fig_pareto_front_cost_service.pdf").exists())

    def test_spea2_is_available_as_comparison_algorithm(self):
        module = load_script_module()
        self.assertIn("SPEA2", module.ALGORITHMS)
        self.assertIn("SPEA2", module.ALGORITHM_LABELS)
        self.assertTrue(callable(module.run_spea2_tracked))


if __name__ == "__main__":
    unittest.main()
