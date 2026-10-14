# -*- coding: utf-8 -*-
"""Lightweight checks that code stays aligned with the QVNS-NSGA-II thesis model."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import main  # noqa: E402



def test_calibrated_cost_parameters_are_loaded():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert main.STATION_TYPE_FIXED_COST == {"SL": 8000.0, "TR": 12000.0, "HB": 16000.0}
    assert set(inst.fixed_cost.tolist()) == {8000.0, 12000.0, 16000.0}
    assert inst.road_operating_cost == 0.12
    assert inst.inout_cost == 0.20

def test_site_level_demand_scenarios_are_loaded_from_sf78_file():
    import pandas as pd

    demand_file = PACKAGE_ROOT / "data" / "武侯区_顺丰站点_精准清理_20251218_073112.csv"
    demand_df = pd.read_csv(demand_file, encoding="utf-8-sig")
    expected = float(demand_df["基准需求_日均需求量"].sum())
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert inst.J == 78
    assert abs(float(inst.Q_j.sum()) - expected) < 1e-6
    assert len(set(round(float(v), 6) for v in inst.Q_j)) > 1

def test_default_model_setting_replaces_fixed_open_count():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert inst.model_setting == "no_budget_range"
    assert inst.fixed_cost_lower == 0.0
    assert inst.fixed_cost_upper == float(inst.fixed_cost.sum())
    assert inst.fixed_cost_delta == 0.0
    assert inst.min_station_distance == 2.0
    assert inst.min_open == 0
    assert inst.max_open == inst.K


def test_station_type_sets_and_cost_levels():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert set(inst.station_type_labels) == {"SL", "TR", "HB"}
    assert int((inst.station_type_labels == "SL").sum()) == 8
    assert int((inst.station_type_labels == "TR").sum()) == 6
    assert int((inst.station_type_labels == "HB").sum()) == 3
    assert set(inst.station_capacity.tolist()) == {5000.0, 7000.0, 10000.0}
    assert set(inst.fixed_cost.tolist()) == {8000.0, 12000.0, 16000.0}


def test_transfer_all_candidate_mode_uses_only_wuhou_transfer_stations():
    inst = main.build_metro_logistics_instance(scenario="base", candidate_mode="transfer_all")

    expected_names = {
        "省体育馆", "倪家桥", "火车南站", "孵化园", "世纪城",
        "高升桥", "太平园", "武青南路", "九兴大道", "神仙树",
        "锦城大道", "高朋大道", "三元", "华兴", "机投桥",
    }

    assert inst.K == 15
    assert set(inst.candidate_df["地铁站名称"]) == expected_names
    assert set(inst.station_type_labels) == {"TR", "HB"}
    assert int((inst.station_type_labels == "TR").sum()) == 12
    assert int((inst.station_type_labels == "HB").sum()) == 3
    assert set(inst.station_capacity.tolist()) == {7000.0, 10000.0}
    assert set(inst.fixed_cost.tolist()) == {12000.0, 16000.0}


def test_line_concentration_limit_is_three_transfer_points():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert inst.max_open_per_line == 3
    dec = main.normalize_solution([1] * inst.K + [0] * (inst.I * inst.J), inst)
    repaired, _ = main.decode_solution(main.repair_solution(dec, inst), inst)
    counts = main.open_station_counts_by_line(repaired, inst)

    assert counts
    assert max(counts.values()) <= 3


def test_supplementary_constraint_data_files_exist():
    data_dir = PACKAGE_ROOT / "data" / "补充约束数据"
    expected_files = {
        "平峰期地铁货运体积与装箱能力参数.csv",
        "路径线路关联矩阵_a_ik_line.csv",
        "模型全局参数_无预算.csv",
        "候选转运点最终属性表.csv",
        "平峰期列车班次与碳成本参数.csv",
        "物流园区货源比例与接入信息.csv",
        "README_补充约束数据说明.md",
    }

    assert expected_files.issubset({path.name for path in data_dir.iterdir() if path.is_file()})


def test_core_volume_table_uses_integer_box_packing():
    import pandas as pd

    data_dir = PACKAGE_ROOT / "data" / "补充约束数据"
    table_path = data_dir / "平峰期地铁货运体积与装箱能力参数.csv"
    table = pd.read_csv(table_path, encoding="utf-8-sig")

    required_columns = {
        "平峰期单节车厢货运区域长度_m",
        "最优箱体摆放方向_长宽高_m",
        "长度方向可排箱数",
        "宽度方向可排箱数",
        "高度方向可排箱数",
        "装箱数计算方法",
    }

    assert required_columns.issubset(set(table.columns))
    first = table.iloc[0]
    assert int(first["单节车厢可装箱数_件"]) == (
        int(first["长度方向可排箱数"])
        * int(first["宽度方向可排箱数"])
        * int(first["高度方向可排箱数"])
    )
    assert "三维整数排布" in str(first["装箱数计算方法"])


def test_line_capacity_data_is_loaded_into_instance():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)

    assert inst.line_capacity_limit > 0
    assert inst.path_line_incidence.shape[0] == inst.I
    assert inst.path_line_incidence.shape[1] == inst.K
    assert inst.path_line_incidence.shape[2] == len(inst.line_names)
    assert {re.search(r"\d+", name).group(0) for name in inst.line_names} >= {"1", "3", "5", "7", "8", "10", "18"}
    assert all(value > 0 for value in inst.line_capacity.values())



def test_line_capacity_violation_is_counted_in_cv():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)
    inst.line_capacity = {line: 1.0 for line in inst.line_names}
    dec = main.normalize_solution([1] * inst.K + [1] * (inst.I * inst.J), inst)

    _, cv, parts = main.evaluate_solution(dec, inst, use_repair=False)

    assert parts["line_capacity_violation"] > 0
    assert parts["train_capacity_violation_pieces"] > 0
    assert cv >= parts["line_capacity_violation"]


def test_solution_tables_include_path_lines_and_line_loads():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)
    dec = main.normalize_solution([1] * inst.K + [1] * (inst.I * inst.J), inst)

    assign_df, station_df, line_df = main.solution_to_tables(dec, inst, use_repair=False)

    assert any("\u7ebf\u8def" in col for col in assign_df.columns)
    assert any(col.endswith("_\u4ef6\u65e5") and "\u5bb9\u91cf" in col for col in station_df.columns)
    assert not any("\u5047\u8bbe" in col for col in station_df.columns)
    assert len(line_df.columns) >= 5
    assert len(line_df) >= len(inst.line_names) * inst.available_train_trips
    assert line_df["单趟列车容量_件"].min() > 0

def test_result_parts_include_model_setting_and_station_type_metrics():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17)
    dec = main.normalize_solution([1] * inst.K + [0] * (inst.I * inst.J), inst)
    _, _, parts = main.evaluate_solution(dec, inst)

    for key in [
        "model_setting",
        "fixed_cost_total",
        "fixed_cost_reference_upper_all_stations",
        "num_sl",
        "num_tr",
        "num_hb",
        "ratio_sl",
        "ratio_tr",
        "ratio_hb",
        "unit_carbon_kg_per_piece",
    ]:
        assert key in parts


def test_parameter_documents_do_not_contain_mojibake_placeholders():
    import pandas as pd

    checked_files = [
        PACKAGE_ROOT / "docs" / "5.1第五章数学模型.md",
        PACKAGE_ROOT / "docs" / "第五章模型参数与约束确认表.md",
    ]
    for path in checked_files:
        text = path.read_text(encoding="utf-8-sig")
        assert "????" not in text
        assert "?/?" not in text

    parameter_table = pd.read_csv(PACKAGE_ROOT / "data" / "模型参数总表_确认版.csv", encoding="utf-8-sig")
    table_text = parameter_table.to_csv(index=False)
    assert "????" not in table_text
    assert "?/?" not in table_text


def test_model_symbols_use_current_non_overlapping_notation():
    import pandas as pd

    model_text = (PACKAGE_ROOT / "docs" / "5.1第五章数学模型.md").read_text(encoding="utf-8-sig")
    parameter_table = pd.read_csv(PACKAGE_ROOT / "data" / "模型参数总表_确认版.csv", encoding="utf-8-sig")
    table_text = parameter_table.to_csv(index=False)

    assert "Cap_k" not in model_text
    assert "Cap_k / station_capacity" not in table_text
    assert "\\theta^F" not in model_text
    assert "theta^F" not in table_text
    assert "\\theta=" not in model_text
    assert "\\theta\\eta" not in model_text

    assert "U_k" in model_text
    assert "U_k / station_capacity" in table_text
    assert "\\theta^{fre}" in model_text
    assert "theta^{fre}" in table_text
    assert "5000, & k\\in K^{SL}" in model_text
    assert "7000, & k\\in K^{TR}" in model_text
    assert "10000, & k\\in K^{HB}" in model_text
    assert "有效体积的 20%" in model_text


def test_parameter_table_uses_no_budget_model_setting():
    model_text = (PACKAGE_ROOT / "docs" / "5.1第五章数学模型.md").read_text(encoding="utf-8-sig")
    import pandas as pd

    parameter_table = pd.read_csv(PACKAGE_ROOT / "data" / "模型参数总表_确认版.csv", encoding="utf-8-sig")
    table_text = parameter_table.to_csv(index=False)

    assert "no_budget_range" in table_text
    assert "Cap_k" not in table_text
    assert "普通单线站7000;一般换乘站7000;复合换乘枢纽站10000" not in table_text
    legacy_tokens = [f"B_{{s_{i}}}^L" for i in (1, 2, 3)] + [f"{n}0000<=C^{{FIX}}" for n in (5, 6, 7)]
    for old_text in legacy_tokens:
        assert old_text not in model_text
        assert old_text not in table_text
    assert "U_k / station_capacity" in table_text

def test_parameter_table_has_required_qvns_columns_and_no_duplicate_codes():
    import pandas as pd

    parameter_table = pd.read_csv(PACKAGE_ROOT / "data" / "模型参数总表_确认版.csv", encoding="utf-8-sig")
    assert {"类别", "参数名称", "代码变量", "确认值", "说明"}.issubset(parameter_table.columns)
    codes = parameter_table["代码变量"].dropna().astype(str)
    assert int((codes == "QVNS-NSGA-II").sum()) == 1
    assert int((codes == "no_budget_range").sum()) == 1
    assert "q_action_apply_rate" in codes.values
    assert "vns_interval" in codes.values


def test_thesis_runner_defaults_match_formal_solver_settings():
    nsga_text = (CODE_DIR / "run_nsga2_for_thesis.py").read_text(encoding="utf-8")
    qvns_text = (CODE_DIR / "run_qvns_nsga2_for_thesis.py").read_text(encoding="utf-8")
    main_text = (CODE_DIR / "main.py").read_text(encoding="utf-8")

    assert '"--pop-size", type=int, default=100' in nsga_text
    assert '"--max-gen", type=int, default=300' in nsga_text
    assert '"--pop-size", type=int, default=100' in qvns_text
    assert '"--max-gen", type=int, default=300' in qvns_text
    assert "def run_nsga2(inst: MetroLogisticsInstance, pop_size=100, max_gen=300" in main_text
    assert "def _run_qvns_nsga2_core(inst: MetroLogisticsInstance, pop_size=100, max_gen=300" in main_text


def test_qvns_nsga2_core_defaults_use_q11c_final_configuration():
    main_text = (CODE_DIR / "main.py").read_text(encoding="utf-8")

    expected_tokens = [
        "pc=0.9, alpha=0.15, gamma=0.88, epsilon=0.30",
        "epsilon_min=0.010, epsilon_decay=None, archive_size=None",
        "vns_interval=0, vns_elite_rate=0.12, vns_trials_per_neighborhood=2",
        "vns_max_neighborhood=3, qvns_phase_schedule=None",
        "archive_feedback_interval=8, archive_feedback_rate=0.15",
        "q_action_apply_rate=0.25, q_learning_enabled=True",
        "q_action_profile=\"safe\", safeguard=True, safeguard_threshold=-0.15",
        "vns_mode=\"adaptive\", stagnation_window=10, hv_improvement_threshold=0.0025",
        "min_vns_gap=5, archive_boundary_keep_rate=0.20",
        "vns_candidate_strategy=\"mixed\"",
        "epsilon_decay = epsilon_decay if epsilon_decay is not None else _epsilon_decay_for_target(0.30, 0.010, max_gen)",
        "archive_size = int(archive_size or max(pop_size, int(round(2.5 * pop_size))))",
    ]
    for token in expected_tokens:
        assert token in main_text


def test_qvns_nsga2_expected_output_paths_are_thesis_named():
    paths = main.qvns_nsga2_expected_output_paths(PACKAGE_ROOT / "results" / "qvns_nsga2_thesis" / "no_budget_range")

    expected_files = {
        "qvns_nsga2_pareto.csv",
        "qvns_nsga2_final_history.csv",
        "qvns_nsga2_training_history.csv",
        "qvns_nsga2_q_table.csv",
        "qvns_nsga2_action_space.csv",
        "qvns_nsga2_best_params.csv",
        "qvns_nsga2_compromise_assignments.csv",
        "qvns_nsga2_station_load.csv",
        "qvns_nsga2_line_load.csv",
        "qvns_nsga2_vns_history.csv",
    }
    actual_files = {path.name for path in paths.values()}

    assert actual_files == expected_files
    assert all("ql_nsga2" not in path.name for path in paths.values())
    assert all(path.parent.name == "no_budget_range" for path in paths.values())


def test_nsga2_export_includes_same_compromise_map_figure_as_qvns():
    main_text = (CODE_DIR / "main.py").read_text(encoding="utf-8")
    nsga_export = main_text.split("def export_nsga2_results", 1)[1].split("def export_nsga2_results_no_repair", 1)[0]
    qvns_export = main_text.split("def _export_qvns_nsga2_results_core", 1)[1].split("def qvns_nsga2_expected_output_paths", 1)[0]

    assert "plot_compromise_map(assign_df, station_df, inst, output_dir)" in nsga_export
    assert "plot_compromise_map(assign_df, station_df, inst, output_dir)" in qvns_export
    assert "fig_compromise_map" in main_text


def test_nsga2_expected_output_paths_are_thesis_named():
    paths = main.nsga2_expected_output_paths(PACKAGE_ROOT / "results" / "nsga2_no_budget_baseline" / "no_budget_range")

    expected_files = {
        "nsga2_pareto.csv",
        "nsga2_history.csv",
        "nsga2_compromise_assignments.csv",
        "nsga2_station_load.csv",
        "nsga2_line_load.csv",
    }
    actual_files = {path.name for path in paths.values()}

    assert actual_files == expected_files
    assert all(path.parent.name == "no_budget_range" for path in paths.values())


if __name__ == "__main__":
    test_calibrated_cost_parameters_are_loaded()
    test_site_level_demand_scenarios_are_loaded_from_sf78_file()
    test_default_model_setting_replaces_fixed_open_count()
    test_station_type_sets_and_cost_levels()
    test_line_concentration_limit_is_three_transfer_points()
    test_supplementary_constraint_data_files_exist()
    test_core_volume_table_uses_integer_box_packing()
    test_line_capacity_data_is_loaded_into_instance()
    test_line_capacity_violation_is_counted_in_cv()
    test_solution_tables_include_path_lines_and_line_loads()
    test_result_parts_include_model_setting_and_station_type_metrics()
    test_parameter_documents_do_not_contain_mojibake_placeholders()
    test_model_symbols_use_current_non_overlapping_notation()
    test_parameter_table_uses_no_budget_model_setting()
    test_parameter_table_has_required_qvns_columns_and_no_duplicate_codes()
    test_thesis_runner_defaults_match_formal_p200_g400_setting()
    test_qvns_nsga2_expected_output_paths_are_thesis_named()
    test_nsga2_export_includes_same_compromise_map_figure_as_qvns()
    test_nsga2_expected_output_paths_are_thesis_named()
    print("model alignment checks passed")



def test_ablation_script_supports_igd_spread_and_q9_branches():
    script_text = (PACKAGE_ROOT / "scripts" / "run_qvns_ablation_experiment.py").read_text(encoding="utf-8-sig")
    for token in ["def igd", "def spread_delta", "IGD_mean", "Spread_mean"]:
        assert token in script_text
    for group in ["Q3", "Q9", "Q9A", "Q9B", "Q9C", "Q9D", "Q11", "Q11A", "Q11B", "Q11C"]:
        assert group in script_text


def test_q9_branch_parameter_table_exists_and_has_expected_groups():
    import pandas as pd

    path = PACKAGE_ROOT / "data" / "补充约束数据" / "QVNS调优参数组_Q3_Q9分支_Q11.csv"
    table = pd.read_csv(path, encoding="utf-8-sig")
    assert {"Q3", "Q9", "Q9A", "Q9B", "Q9C", "Q9D", "Q11", "Q11A", "Q11B", "Q11C"}.issubset(set(table["tuning_group"]))
    assert "adaptive" in set(table["vns_mode"].astype(str))
    assert table.loc[table["tuning_group"] == "Q9C", "archive_multiplier"].iloc[0] >= 4.0


def test_ablation_default_runs_are_ten_for_stability():
    import importlib.util

    script_path = PACKAGE_ROOT / "scripts" / "run_qvns_ablation_experiment.py"
    spec = importlib.util.spec_from_file_location("run_qvns_ablation_experiment_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module.parse_args([])
    assert args.runs == 10
    assert args.qvns_tuning_groups == ["Q3", "Q9", "Q9A", "Q9B", "Q9C", "Q9D", "Q11", "Q11A", "Q11B", "Q11C"]
