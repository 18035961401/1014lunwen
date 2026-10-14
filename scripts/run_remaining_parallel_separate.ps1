# 并行运行剩余实验 - 每个算法组独立文件
# 避免写入冲突

Write-Host "=========================================="
Write-Host "启动8个并行进程运行剩余实验"
Write-Host "每个进程独立输出到不同文件"
Write-Host "=========================================="

# 定义需要运行的算法组
$algorithms = @(
    @{Name="QVNS-Q9A"; Group="Q9A"; Runs=20; OutputDir="results/qvns_ablation_q9a"},
    @{Name="QVNS-Q9B"; Group="Q9B"; Runs=20; OutputDir="results/qvns_ablation_q9b"},
    @{Name="QVNS-Q9C"; Group="Q9C"; Runs=20; OutputDir="results/qvns_ablation_q9c"},
    @{Name="QVNS-Q9D"; Group="Q9D"; Runs=20; OutputDir="results/qvns_ablation_q9d"},
    @{Name="QVNS-Q11"; Group="Q11"; Runs=20; OutputDir="results/qvns_ablation_q11"},
    @{Name="QVNS-Q11A"; Group="Q11A"; Runs=20; OutputDir="results/qvns_ablation_q11a"},
    @{Name="QVNS-Q11B"; Group="Q11B"; Runs=20; OutputDir="results/qvns_ablation_q11b"},
    @{Name="QVNS-Q11C"; Group="Q11C"; Runs=20; OutputDir="results/qvns_ablation_q11c"}
)

foreach ($algo in $algorithms) {
    $name = $algo.Name
    $group = $algo.Group
    $runs = $algo.Runs
    $outputDir = $algo.OutputDir
    
    Write-Host "启动 $name -> $outputDir ..."
    
    # 使用Start-Process启动独立进程
    Start-Process -FilePath "python" -ArgumentList @(
        "scripts/run_qvns_ablation_experiment.py",
        "--comparison-mode", "tuning",
        "--qvns-tuning-groups", $group,
        "--pop-size", "100",
        "--max-gen", "300",
        "--runs", $runs,
        "--output-dir", $outputDir,
        "--quiet"
    ) -WindowStyle Normal -WorkingDirectory (Get-Location)
    
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "已启动8个并行进程！"
Write-Host "每个进程输出到独立目录:"
foreach ($algo in $algorithms) {
    Write-Host "  $($algo.Name) -> $($algo.OutputDir)"
}
Write-Host "=========================================="
