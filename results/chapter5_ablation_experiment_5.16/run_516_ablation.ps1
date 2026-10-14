$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Script = Join-Path $Here 'scripts\run_qvns_ablation_experiment.py'
$Output = Join-Path $Here 'results'

python $Script `
  --comparison-mode all `
  --qvns-tuning-groups Q11C `
  --model-settings no_budget_range `
  --demand-scenario base `
  --top-n 17 `
  --pop-size 100 `
  --max-gen 300 `
  --runs 10 `
  --seed 20260512 `
  --vns-trials 3 `
  --output-dir $Output `
  --quiet
