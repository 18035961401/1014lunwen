# -*- coding: utf-8 -*-
"""综合算法对比实验：QVNS-NSGA2, NSGA2, MOPSO, MOEA/D

本脚本运行四种算法的对比实验，生成：
1. 总成本收敛曲线
2. 碳排放成本收敛曲线
3. 超体积指标收敛曲线
4. 收敛速度对比（成本改进率）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (CODE_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from main import (  # noqa: E402
    build_metro_logistics_instance,
    run_nsga2,
)
from run_moo_comparison_experiment import (  # noqa: E402
    run_moead,
    run_mopso,
)


def run_qvns_nsga2_wrapper(inst, pop_size: int, max_gen: int, seed: int, verbose: bool):
    """包装QVNS-NSGA2调用"""
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
        archive_size=None,
        preserve_baseline=False,
        vns_interval=0,
        vns_elite_rate=0.12,
        vns_trials_per_neighborhood=2,
        vns_max_neighborhood=3,
        archive_feedback_interval=8,
        archive_feedback_rate=0.15,
        q_action_apply_rate=0.25,
        q_action_profile="safe",
        q_learning_enabled=True,
        safeguard=True,
        safeguard_threshold=-0.15,
        pc=0.90,
    )


def extract_convergence_history(result: dict, algorithm: str) -> pd.DataFrame:
    """从结果中提取收敛历史数据"""
    history = result.get("history", pd.DataFrame())

    if isinstance(history, pd.DataFrame) and not history.empty:
        df = history.copy()
    else:
        # 如果没有历史记录，创建空的DataFrame
        df = pd.DataFrame()

    if not df.empty:
        df["algorithm"] = algorithm

        # 确保必要的列存在
        if "gen" not in df.columns:
            df["gen"] = range(1, len(df) + 1)

        # 重命名列以统一格式
        column_mapping = {
            "min_cost": "total_cost",
            "best_cost": "total_cost",
            "Z1_total_cost": "total_cost",
        }
        df = df.rename(columns=column_mapping)

    return df


def calculate_carbon_cost(total_cost: float, inst) -> float:
    """估算碳排放成本（假设占总成本的一定比例）"""
    # 这里使用简化估算，实际应根据模型计算
    # 假设碳排放成本约占运输成本的15-20%
    return total_cost * 0.18


def compute_hypervolume_2d(points: np.ndarray, ref_point: tuple[float, float]) -> float:
    """计算2D超体积指标"""
    if len(points) == 0:
        return 0.0

    # 按第一个目标排序
    sorted_points = points[np.argsort(points[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]

    for x, y in sorted_points:
        if x <= ref_point[0] and y < prev_y:
            width = ref_point[0] - x
            height = prev_y - y
            hv += max(0.0, width * height)
            prev_y = y

    return float(hv)


def calculate_improvement_rate(costs: np.ndarray) -> np.ndarray:
    """计算成本改进率"""
    if len(costs) <= 1:
        return np.zeros_like(costs)

    initial_cost = costs[0]
    improvement_rate = (initial_cost - costs) / (initial_cost + 1e-9) * 100
    return improvement_rate


def run_single_algorithm(
    algorithm: str,
    inst,
    pop_size: int,
    max_gen: int,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    """运行单个算法"""
    print(f"\n{'='*60}")
    print(f"运行算法: {algorithm}, 种子: {seed}")
    print(f"{'='*60}")

    start_time = time.time()

    if algorithm == "QVNS-NSGA2":
        result = run_qvns_nsga2_wrapper(inst, pop_size, max_gen, seed, verbose)
    elif algorithm == "NSGA2":
        result = run_nsga2(inst, pop_size, max_gen, seed=seed, verbose=verbose, use_repair=True)
    elif algorithm == "MOPSO":
        result = run_mopso(inst, pop_size, max_gen, seed=seed, verbose=verbose)
    elif algorithm == "MOEAD":
        result = run_moead(inst, pop_size, max_gen, seed=seed, verbose=verbose)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    runtime = time.time() - start_time
    result["runtime"] = runtime
    result["algorithm"] = algorithm

    print(f"\n{algorithm} 完成，耗时: {runtime:.2f}秒")

    return result


def process_convergence_data(results: dict[str, list[dict]]) -> pd.DataFrame:
    """处理所有算法的收敛数据"""
    all_data = []

    for algorithm, runs in results.items():
        for run_idx, result in enumerate(runs):
            history = extract_convergence_history(result, algorithm)

            if not history.empty and "total_cost" in history.columns:
                history["run_id"] = run_idx + 1
                all_data.append(history)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def aggregate_convergence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """聚合多次运行的收敛指标（计算均值和标准差）"""
    if df.empty:
        return pd.DataFrame()

    # 按算法和代数分组
    grouped = df.groupby(["algorithm", "gen"])

    agg_dict = {}

    # 对每个数值列计算均值和标准差
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = {"run_id", "gen"}

    for col in numeric_cols:
        if col not in exclude_cols and col in df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    result = grouped.agg(**agg_dict).reset_index()

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行QVNS-NSGA2, NSGA2, MOPSO, MOEA/D综合对比实验"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["QVNS-NSGA2", "NSGA2", "MOPSO", "MOEAD"],
        choices=["QVNS-NSGA2", "NSGA2", "MOPSO", "MOEAD"],
        help="要运行的算法列表",
    )
    parser.add_argument("--runs", type=int, default=10, help="每个算法的独立运行次数")
    parser.add_argument("--pop-size", type=int, default=100, help="种群大小")
    parser.add_argument("--max-gen", type=int, default=300, help="最大迭代代数")
    parser.add_argument("--seed", type=int, default=20260512, help="基础随机种子")
    parser.add_argument("--top-n", type=int, default=17, help="候选地铁站数量")
    parser.add_argument("--scenario", default="base", help="需求场景")
    parser.add_argument("--model-setting", default="no_budget_range", help="模型设置")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "comprehensive_comparison",
        help="输出目录",
    )
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--skip-plots", action="store_true", help="跳过绘图")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("综合算法对比实验")
    print(f"{'='*60}")
    print(f"算法: {', '.join(args.algorithms)}")
    print(f"运行次数: {args.runs}")
    print(f"种群大小: {args.pop_size}")
    print(f"最大迭代代数: {args.max_gen}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 构建问题实例
    print("构建问题实例...")
    inst = build_metro_logistics_instance(
        scenario=args.scenario,
        top_n=args.top_n,
        model_setting=args.model_setting,
    )
    print(f"问题实例构建完成: {args.top_n}个候选站点\n")

    # 运行所有算法
    results = {alg: [] for alg in args.algorithms}

    for algorithm in args.algorithms:
        for run_id in range(1, args.runs + 1):
            seed = args.seed + (run_id - 1) * 100

            result = run_single_algorithm(
                algorithm=algorithm,
                inst=inst,
                pop_size=args.pop_size,
                max_gen=args.max_gen,
                seed=seed,
                verbose=not args.quiet,
            )

            results[algorithm].append(result)

            # 保存中间结果
            convergence_df = process_convergence_data(results)
            if not convergence_df.empty:
                convergence_df.to_csv(
                    args.output_dir / "convergence_history_raw.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    # 处理和保存收敛数据
    print("\n处理收敛数据...")
    convergence_df = process_convergence_data(results)

    if not convergence_df.empty:
        # 保存原始数据
        convergence_df.to_csv(
            args.output_dir / "convergence_history_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 聚合数据
        aggregated_df = aggregate_convergence_metrics(convergence_df)
        aggregated_df.to_csv(
            args.output_dir / "convergence_history_aggregated.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"收敛数据已保存到: {args.output_dir}")

    # 生成绘图（如果需要）
    if not args.skip_plots:
        print("\n生成收敛曲线...")
        from plot_convergence_curves import plot_all_convergence_curves

        plot_all_convergence_curves(
            convergence_df=convergence_df,
            aggregated_df=aggregated_df,
            inst=inst,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"结果保存在: {args.output_dir}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
