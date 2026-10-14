from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from chinese_font_utils import configure_matplotlib_cjk, select_cjk_font, write_csv_excel_compatible


def test_cjk_font_helper_selects_installed_or_safe_fallback():
    font_name = select_cjk_font()
    assert isinstance(font_name, str)
    assert font_name
    assert font_name != "DejaVu Sans"


def test_matplotlib_cjk_configuration_uses_selected_font():
    import matplotlib.pyplot as plt

    font_name = configure_matplotlib_cjk(plt)
    family = plt.rcParams["font.family"]
    assert font_name in family
    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["pdf.fonttype"] == 42


def test_excel_compatible_csv_writes_utf8_bom(tmp_path):
    path = tmp_path / "中文.csv"
    write_csv_excel_compatible(pd.DataFrame({"地铁站名称": ["火车南站"]}), path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "火车南站" in path.read_text(encoding="utf-8-sig")
