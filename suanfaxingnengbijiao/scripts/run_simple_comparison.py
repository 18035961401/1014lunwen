# -*- coding: utf-8 -*-
"""简化版综合算法对比实验

直接运行四种算法并生成收敛曲线，无需复杂依赖
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
PARENT_CODE_DIR = PROJECT_ROOT.parent / "code"

# 添加路径并切换工作目录到code目录
for import_dir in (CODE_DIR, PARENT_CODE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

# 切换工作目录到code目录，确保数据文件路径正确
import os
if PARENT_CODE_DIR.exists():
    os.chdir(str(PARENT_CODE_DIR))
    print(f"工作目录已切换到: {os.getcwd()}")
elif CODE_DIR.exists():
    os.chdir(str(CODE_DIR))
    print(f"工作目录已切换到: {os.getcwd()}")

from main import (  # noqa: E402
    build_metro_logistics_instance,
    evaluate_population,
    fast_non_dominated_sort,
    initialize_population,
    run_nsga2,
)


def run_qvns_nsga2_simple(inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    """简化的QVNS-NSGA2调用"""
    from main import run_qvns_nsga2

    return run_qvns_nsga2(
        inst,
        pop_size=pop_size,
        max_gen=max_gen,
        seed=seed,
        verbose=verbose,
        use_repair=True,
        alpha=0.15,
        gamma=0.88,
        epsilon=0.30,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        pc=0.90,
    )


def run_mopso_simple(inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    """简化的MOPSO实现"""
    from run_moo_comparison_experiment import run_mopso
    return run_mopso(inst, pop_size, max_gen, seed, verbose=verbose)


def run_moead_simple(inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    """简化的MOEA/D实现"""
    from run_moo_comparison_experiment import run_moead
    return run_moead(inst, pop_size, max_gen, seed, verbose=verbose)


def extract_history(result: dict, algorithm: str) -> pd.DataFrame:
    """提取收敛历史"""
    history = result.get("history", pd.DataFrame())

    if isinstance(history, pd.DataFrame) and not history.empty:
        df = history.copy()
        df["algorithm"] = algorithm

        if "gen" not in df.columns:
            df["gen"] = range(1, len(df) + 1)

        # 统一列名
        if "min_cost" in df.columns:
            df["total_cost"] = df["min_cost"]
        elif "best_cost" in df.columns:
            df["total_cost"] = df["best_cost"]

        return df

    return pd.DataFrame()


def run_algorithm(algorithm: str, inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    """运行单个算法"""
    print(f"\n{'='*60}")
    print(f"运行: {algorithm}, 种子: {seed}")
    print(f"{'='*60}")

    start = time.time()

    try:
        if algorithm == "QVNS-NSGA2":
            result = run_qvns_nsga2_simple(inst, pop_size, max_gen, seed, verbose)
        elif algorithm == "NSGA2":
            result = run_nsga2(inst, pop_size, max_gen, seed=seed, verbose=verbose, use_repair=True)
        elif algorithm == "MOPSO":
            result = run_mopso_simple(inst, pop_size, max_gen, seed, verbose)
        elif algorithm == "MOEAD":
            result = run_moead_simple(inst, pop_size, max_gen, seed, verbose)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        runtime = time.time() - start
        result["runtime"] = runtime
        result["algorithm"] = algorithm

        print(f"\n{algorithm} 完成，耗时: {runtime:.2f}秒")
        return result

    except Exception as e:
        print(f"\n错误: {algorithm} 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="简化版算法对比实验")
    parser.add_argument("--algorithms", nargs="+", default=["QVNS-NSGA2", "NSGA2", "MOPSO", "MOEAD"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--max-gen", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--top-n", type=int, default=17)
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--model-setting", default="no_budget_range")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "comprehensive_comparison")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("简化版综合算法对比实验")
    print(f"{'='*60}")
    print(f"算法: {', '.join(args.algorithms)}")
    print(f"运行次数: {args.runs}")
    print(f"种群大小: {args.pop_size}")
    print(f"迭代代数: {args.max_gen}")
    print(f"{'='*60}\n")

    # 构建问题实例
    print("构建问题实例...")
    inst = build_metro_logistics_instance(
        scenario=args.scenario,
        top_n=args.top_n,
        road_factor=1.2,
        model_setting=args.model_setting
    )
    print(f"问题实例构建完成\n")

    # 运行实验
    all_histories = []

    for algorithm in args.algorithms:
        for run_id in range(1, args.runs + 1):
            seed = args.seed + (run_id - 1) * 100

            result = run_algorithm(algorithm, inst, args.pop_size, args.max_gen, seed, not args.quiet)

            if result:
                history = extract_history(result, algorithm)
                if not history.empty:
                    history["run_id"] = run_id
                    all_histories.append(history)

                # 实时保存
                if all_histories:
                    combined = pd.concat(all_histories, ignore_index=True)
                    combined.to_csv(args.output_dir / "convergence_raw.csv", index=False, encoding="utf-8-sig")

    # 聚合数据
    if all_histories:
        print("\n处理收敛数据...")
        combined = pd.concat(all_histories, ignore_index=True)
        combined.to_csv(args.output_dir / "convergence_raw.csv", index=False, encoding="utf-8-sig")

        # 计算均值和标准差
        if "total_cost" in combined.columns:
            agg = combined.groupby(["algorithm", "gen"]).agg(
                total_cost_mean=("total_cost", "mean"),
                total_cost_std=("total_cost", "std"),
            ).reset_index()
            agg.to_csv(args.output_dir / "convergence_aggregated.csv", index=False, encoding="utf-8-sig")

            print(f"数据已保存到: {args.output_dir}")

            # 绘图
            if not args.skip_plots:
                try:
                    print("\n生成收敛曲线...")
                    from plot_convergence_curves import plot_all_convergence_curves
                    plot_all_convergence_curves(combined, agg, inst, args.output_dir)
                except Exception as e:
                    print(f"绘图失败: {e}")
                    print("数据已保存，可以稍后手动绘图")

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"结果: {args.output_dir}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
