$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$python = "C:\Users\leems\AppData\Local\Programs\Python\Python311\python.exe"
$statusPath = Join-Path $repo "experiment_status.json"
$trainLog = Join-Path $repo "train_balanced_seat.log"
$evalLog = Join-Path $repo "eval_balanced_seat.log"
$evalProgress = Join-Path $repo "eval_balanced_seat_progress.json"
$evalResults = Join-Path $repo "eval_balanced_seat_results.jsonl"
$modelPath = "rummikub_ppo_balanced_seat_model.pt"

function Write-ExperimentStatus {
    param(
        [string]$Stage,
        [string]$Detail
    )

    @{
        stage = $Stage
        detail = $Detail
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

Set-Location -LiteralPath $repo

try {
    Write-ExperimentStatus -Stage "training" -Detail "0/1000 alternating-seat episodes"
    & $python train_ppo.py `
        --n-steps 1000 `
        --total-updates 1000 `
        --target-episodes 1000 `
        --model-path $modelPath `
        --status-path $statusPath 2>&1 |
        Tee-Object -FilePath $trainLog

    if ($LASTEXITCODE -ne 0) {
        throw "training exited with code $LASTEXITCODE"
    }

    Write-ExperimentStatus -Stage "evaluating" -Detail "1000 games: 500 PPO + 500 greedy baseline"
    & $python eval_ppo.py `
        --model-path $modelPath `
        --episodes 500 `
        --workers 4 `
        --results-path $evalResults `
        --progress-path $evalProgress `
        --status-path $statusPath `
        --progress-every 50 2>&1 |
        Tee-Object -FilePath $evalLog

    if ($LASTEXITCODE -ne 0) {
        throw "evaluation exited with code $LASTEXITCODE"
    }

    Write-ExperimentStatus -Stage "complete" -Detail "training and evaluation completed"
}
catch {
    Write-ExperimentStatus -Stage "failed" -Detail $_.Exception.Message
    throw
}
