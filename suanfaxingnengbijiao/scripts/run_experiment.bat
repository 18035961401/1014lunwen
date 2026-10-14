@echo off
REM 综合算法对比实验运行脚本
REM 运行 QVNS-NSGA2, NSGA2, MOPSO, MOEA/D 四种算法的对比实验

echo ========================================
echo 综合算法对比实验
echo ========================================
echo.

REM 设置Python路径（如果需要）
REM set PYTHON=python

REM 切换到脚本目录
cd /d "%~dp0"

REM 运行实验（使用配置文件）
echo 开始运行实验...
echo 提示: 可以在 experiment_config.py 中修改实验参数
echo.

python run_experiment_with_config.py

echo.
echo ========================================
echo 实验完成！
echo 结果保存在: ../results/comprehensive_comparison
echo ========================================
echo.

pause
