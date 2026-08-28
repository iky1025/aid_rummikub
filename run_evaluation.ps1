$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$python = "C:\Users\leems\AppData\Local\Programs\Python\Python311\python.exe"
$statusPath = Join-Path $repo "experiment_status.json"
$progressPath = Join-Path $repo "eval_diverse_progress.json"
$resultsPath = Join-Path $repo "eval_diverse_results.jsonl"
$evalLog = Join-Path $repo "eval_diverse.log"

Set-Location -LiteralPath $repo

try {
    & $python eval_ppo.py `
        --model-path rummikub_ppo_diverse_model.pt `
        --episodes 100 `
        --workers 4 `
        --results-path $resultsPath `
        --progress-path $progressPath `
        --status-path $statusPath `
        --progress-every 50 2>&1 |
        Tee-Object -FilePath $evalLog -Append

    if ($LASTEXITCODE -ne 0) {
        throw "evaluation exited with code $LASTEXITCODE"
    }
}
catch {
    @{
        stage = "failed"
        detail = $_.Exception.Message
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
    throw
}
