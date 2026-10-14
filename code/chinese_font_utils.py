"""Matplotlib Chinese font helpers for Windows-friendly figure exports."""

from __future__ import annotations

import os
from pathlib import Path


CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DengXian",
    "Noto Sans SC",
    "Noto Serif SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def select_cjk_font() -> str:
    """Return the first installed Chinese-capable font name."""
    preferred = os.environ.get("MPL_CJK_FONT")
    if preferred:
        return preferred
    try:
        from matplotlib import font_manager as fm

        available = {font.name for font in fm.fontManager.ttflist}
        for candidate in CJK_FONT_CANDIDATES:
            if candidate in available:
                return candidate
    except Exception:
        pass
    return "DejaVu Sans"


def configure_matplotlib_cjk(plt_module=None) -> str:
    """Configure Matplotlib to render Chinese text in PNG/SVG/PDF outputs."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    font_name = select_cjk_font()
    plt_module.rcParams["font.family"] = [font_name]
    plt_module.rcParams["font.sans-serif"] = [font_name] + [
        font for font in CJK_FONT_CANDIDATES if font != font_name
    ]
    plt_module.rcParams["axes.unicode_minus"] = False
    plt_module.rcParams["svg.fonttype"] = "none"
    plt_module.rcParams["pdf.fonttype"] = 42
    plt_module.rcParams["ps.fonttype"] = 42
    return font_name


def get_cjk_font_properties():
    """Return a Matplotlib FontProperties object for the selected CJK font."""
    from matplotlib import font_manager as fm

    font_name = select_cjk_font()
    for font in fm.fontManager.ttflist:
        if font.name == font_name:
            return fm.FontProperties(fname=font.fname)
    return fm.FontProperties(family=font_name)


def write_csv_excel_compatible(df, path: str | Path, **kwargs) -> None:
    """Write CSV with UTF-8 BOM so Chinese opens correctly in Excel."""
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8-sig")
    df.to_csv(path, **kwargs)
