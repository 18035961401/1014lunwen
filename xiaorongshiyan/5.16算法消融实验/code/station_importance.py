# -*- coding: utf-8 -*-
"""Station importance utilities for the formal thesis solver."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9

def extract_station_importance(inst: Any) -> np.ndarray:
    """提取并归一化候选站点 CRITIC-TOPSIS 综合重要度。"""
    df = inst.candidate_df
    score_columns = [
        "综合得分", "综合评价值", "贴近度", "相对贴近度", "TOPSIS得分",
        "score", "Score", "topsis_score", "critic_topsis_score",
    ]
    for col in score_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(values) > np.min(values):
                values = values - np.min(values)
                return _normalize_positive(values)

    if "topsis_order" in df.columns:
        order = pd.to_numeric(df["topsis_order"], errors="coerce").to_numpy(dtype=float)
        fallback = np.arange(1, inst.K + 1, dtype=float)
        order = np.where(np.isfinite(order), order, fallback)
        values = inst.K - order + 1.0
        return _normalize_positive(values)

    reachable = inst.d_kj_road <= inst.D_k_max[:, None]
    coverage_count = reachable.sum(axis=1).astype(float)
    return _normalize_positive(coverage_count)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.maximum(values, 0.0)
    if values.size == 0:
        return values
    if float(values.sum()) <= EPS:
        return np.full(values.size, 1.0 / values.size)
    return values / float(values.sum())
