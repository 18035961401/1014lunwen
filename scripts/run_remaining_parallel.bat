@echo off
chcp 65001 >nul
echo ==========================================
echo 并行运行剩余实验 - 命令行方式
echo 利用resume功能跳过已完成的轮次
echo ==========================================

:: 创建统一的输出目录
mkdir results\qvns_ablation_remaining 2>nul

:: 复制已有的结果到统一目录
copy results\qvns_ablation\ablation_run_metrics.csv results\qvns_ablation_remaining\ 2>nul

:: 使用start命令并行运行8个窗口（CPU核心数）
:: 每个窗口运行一个算法组，利用resume跳过已完成轮次

:: QVNS-Q9A 剩余9轮 (已有11轮，会跳过1-11，跑12-20)
start "QVNS-Q9A" cmd /c "echo Running QVNS-Q9A... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q9A --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q9A Done && pause"

:: QVNS-Q9B 剩余10轮 (已有10轮，会跳过1-10，跑11-20)
start "QVNS-Q9B" cmd /c "echo Running QVNS-Q9B... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q9B --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q9B Done && pause"

:: QVNS-Q9C 剩余10轮
start "QVNS-Q9C" cmd /c "echo Running QVNS-Q9C... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q9C --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q9C Done && pause"

:: QVNS-Q9D 剩余10轮
start "QVNS-Q9D" cmd /c "echo Running QVNS-Q9D... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q9D --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q9D Done && pause"

:: QVNS-Q11 剩余10轮
start "QVNS-Q11" cmd /c "echo Running QVNS-Q11... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q11 --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q11 Done && pause"

:: QVNS-Q11A 剩余10轮
start "QVNS-Q11A" cmd /c "echo Running QVNS-Q11A... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q11A --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q11A Done && pause"

:: QVNS-Q11B 剩余10轮
start "QVNS-Q11B" cmd /c "echo Running QVNS-Q11B... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q11B --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q11B Done && pause"

:: QVNS-Q11C 剩余10轮
start "QVNS-Q11C" cmd /c "echo Running QVNS-Q11C... && python scripts/run_qvns_ablation_experiment.py --comparison-mode tuning --qvns-tuning-groups Q11C --pop-size 100 --max-gen 300 --runs 20 --output-dir results/qvns_ablation_remaining --quiet && echo QVNS-Q11C Done && pause"

echo.
echo 已启动8个并行窗口运行剩余实验！
echo 每个窗口独立运行一个算法组
echo 利用resume功能自动跳过已完成的轮次
echo ==========================================
pause
