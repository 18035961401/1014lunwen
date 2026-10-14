# -*- coding: utf-8 -*-
"""使用配置文件运行实验

直接运行此脚本，它会读取 experiment_config.py 中的配置
"""
from __future__ import annotations

import sys
from pathlib import Path

# 导入配置
try:
    from experiment_config import (
        ALGORITHMS,
        RUNS,
        POP_SIZE,
        MAX_GEN,
        SEED,
        TOP_N,
        SCENARIO,
        MODEL_SETTING,
        OUTPUT_DIR,
        QUIET,
        SKIP_PLOTS,
    )
except ImportError:
    print("错误: 无法导入 experiment_config.py")
    print("请确保 experiment_config.py 文件存在于同一目录")
    sys.exit(1)

from run_comprehensive_algorithm_comparison import main


def run_with_config():
    """使用配置文件运行实验"""
    print("=" * 60)
    print("使用配置文件运行实验")
    print("=" * 60)
    print(f"算法: {', '.join(ALGORITHMS)}")
    print(f"运行次数: {RUNS}")
    print(f"种群大小: {POP_SIZE}")
    print(f"最大迭代代数: {MAX_GEN}")
    print(f"随机种子: {SEED}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # 构建命令行参数
    argv = [
        "--algorithms", *ALGORITHMS,
        "--runs", str(RUNS),
        "--pop-size", str(POP_SIZE),
        "--max-gen", str(MAX_GEN),
        "--seed", str(SEED),
        "--top-n", str(TOP_N),
        "--scenario", SCENARIO,
        "--model-setting", MODEL_SETTING,
        "--output-dir", OUTPUT_DIR,
    ]

    if QUIET:
        argv.append("--quiet")

    if SKIP_PLOTS:
        argv.append("--skip-plots")

    # 运行实验
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run_with_config())
