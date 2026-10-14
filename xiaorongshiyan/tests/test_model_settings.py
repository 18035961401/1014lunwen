# -*- coding: utf-8 -*-
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import main


def test_only_no_budget_model_setting_is_supported():
    assert set(main.MODEL_SETTINGS) == {"no_budget_range"}
    assert main._normalize_model_setting("no_budget") == "no_budget_range"
    assert main._normalize_model_setting("base") == "no_budget_range"


def test_removed_fixed_cost_intervals_are_rejected():
    for legacy in ["legacy_budget_range", "fixed_interval", "restricted_budget"]:
        try:
            main._normalize_model_setting(legacy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported model setting should be rejected: {legacy}")


def test_no_budget_instance_uses_all_station_fixed_cost_reference_upper():
    inst = main.build_metro_logistics_instance(scenario="base", top_n=17, model_setting="no_budget_range")
    assert inst.model_setting == "no_budget_range"
    assert inst.fixed_cost_lower == 0.0
    assert inst.fixed_cost_delta == 0.0
    assert inst.fixed_cost_upper == float(inst.fixed_cost.sum())
